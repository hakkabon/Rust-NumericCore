//! Bounded-variable revised simplex — the first real `Solver`
//! implementation (`StubSolver` was the only one before this).
//!
//! ## Formulation
//! `Problem`'s general form (`row_bounds.lower <= Ax <= row_bounds.upper`,
//! `var_bounds.lower <= x <= var_bounds.upper`) is converted to bounded
//! equality form by introducing one slack per row:
//!
//! ```text
//! [A | -I] [x; s] = 0,   var_bounds.lower <= x <= var_bounds.upper,
//!                        row_bounds.lower <= s <= row_bounds.upper
//! ```
//!
//! so `s = A x` and the row bounds apply directly to `s`. This gives a
//! trivially available, always-invertible starting basis — the slack
//! columns, which form `-I` — regardless of whether that starting point
//! is primal-feasible.
//!
//! ## Two phases, same machinery
//! Phase 1 minimizes the sum of basic-variable bound infeasibilities
//! (a piecewise-linear, convex function of `x`) using a *temporary* cost
//! vector recomputed every iteration from the current infeasibilities'
//! subgradient (`-1` for a basic variable below its lower bound, `+1`
//! for one above its upper bound, `0` otherwise). This needs no
//! artificial variables and no Big-M constant — the slacks already
//! introduced play that role. Phase 2 runs the identical iteration loop
//! against the real objective, starting from the feasible basis Phase 1
//! found.
//!
//! ## Scope, deliberately
//! - **Dense**, not sparse-with-factorization-updates. `Binv` (the
//!   explicit basis inverse) is a full `m x m` matrix, updated via
//!   standard Gauss-Jordan pivoting each iteration, and reduced costs
//!   are recomputed from scratch every iteration rather than maintained
//!   incrementally with an eta-file/product-form-of-inverse scheme. This
//!   matches the "simplex does well for smallish problems" framing this
//!   solver was scoped under — the sparse/incremental-factorization
//!   version is real future work if a large-problem need shows up, not
//!   a design mistake to fix reflexively.
//! - **Dantzig's rule** (largest-reduced-cost-magnitude) for entering
//!   variable selection, first-blocking-row for leaving. No Bland's-rule
//!   anti-cycling safeguard beyond an iteration cap — degenerate cycling
//!   is rare for well-posed smallish problems, and the cap
//!   (`max_iterations`, reported as `SolveStatus::IterationLimit` if
//!   hit) is the safety net. Revisit if cycling shows up in practice.
//! - Three-state nonbasic status (`AtLower`/`AtUpper`/`Free`) so a
//!   variable with one or both bounds absent (reachable today — e.g.
//!   `NumericCoreAMPL`'s `var y <= 10;` leaves the lower bound
//!   unspecified) is handled correctly, not treated as a limitation.

use crate::{Bound, OptimizeError, Problem, SolveStatus, Solution, Solver};

/// Bounded-variable revised simplex. See this module's docs for the
/// formulation and scope.
pub struct RevisedSimplexSolver {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for RevisedSimplexSolver {
    fn default() -> Self {
        Self { max_iterations: 10_000, tolerance: 1e-9 }
    }
}

impl Solver for RevisedSimplexSolver {
    fn name(&self) -> &'static str {
        "revised-simplex"
    }

    fn solve(&self, problem: &Problem) -> Result<Solution, OptimizeError> {
        let mut workspace = Workspace::new(problem)?;
        let n = problem.objective.len();
        let tol = self.tolerance;

        // Phase 1: minimize sum of bound infeasibilities using a
        // temporary cost vector recomputed each iteration (see module
        // docs). The real objective plays no role yet.
        let zero_cost = vec![0.0; workspace.n_total];
        match workspace.run_phase(&zero_cost, true, self.max_iterations, tol)? {
            PhaseOutcome::IterationLimit => {
                return Ok(workspace.solution(n, SolveStatus::IterationLimit, &problem.objective))
            }
            PhaseOutcome::Unbounded => {
                // Phase 1's objective is bounded below by 0 by
                // construction; reaching this would indicate a bug in
                // the ratio test, not a legitimate outcome. Surfaced as
                // an error rather than silently mis-reporting a status.
                return Err(OptimizeError::NotImplemented(
                    "revised simplex: phase 1 reported unbounded, which should be impossible",
                ));
            }
            PhaseOutcome::Optimal => {}
        }

        if workspace.total_infeasibility(tol) > tol.sqrt() {
            return Ok(workspace.solution(n, SolveStatus::Infeasible, &problem.objective));
        }

        // Phase 2: same machinery, real cost vector, starting from the
        // feasible basis Phase 1 found.
        match workspace.run_phase(&workspace.real_cost.clone(), false, self.max_iterations, tol)? {
            PhaseOutcome::Optimal => Ok(workspace.solution(n, SolveStatus::Optimal, &problem.objective)),
            PhaseOutcome::Unbounded => Ok(workspace.solution(n, SolveStatus::Unbounded, &problem.objective)),
            PhaseOutcome::IterationLimit => {
                Ok(workspace.solution(n, SolveStatus::IterationLimit, &problem.objective))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonbasicBound {
    AtLower,
    AtUpper,
    Free,
}

enum PhaseOutcome {
    Optimal,
    Unbounded,
    IterationLimit,
}

/// Which of a basic variable's bounds a ratio-test candidate targets —
/// needed so a completed pivot knows which bound the leaving variable
/// becomes nonbasic at.
#[derive(Clone, Copy)]
enum TargetBound {
    Lower,
    Upper,
}

struct Workspace {
    m: usize,
    n_total: usize,
    /// Dense `m x n_total` constraint matrix `[A | -I]`. Dense, not
    /// sparse — see module docs.
    matrix: Vec<Vec<f64>>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    real_cost: Vec<f64>,
    basis: Vec<usize>,
    is_basic: Vec<bool>,
    nb_status: Vec<NonbasicBound>,
    /// Explicit basis inverse, `m x m`, updated in place each pivot.
    binv: Vec<Vec<f64>>,
    x: Vec<f64>,
}

impl Workspace {
    fn new(problem: &Problem) -> Result<Self, OptimizeError> {
        let m = problem.constraints.rows();
        let n = problem.objective.len();
        let n_total = n + m;

        if problem.var_bounds.len() != n {
            return Err(OptimizeError::NotImplemented(
                "revised simplex: var_bounds length must match objective length",
            ));
        }
        if problem.row_bounds.len() != m {
            return Err(OptimizeError::NotImplemented(
                "revised simplex: row_bounds length must match constraint row count",
            ));
        }

        let mut matrix = vec![vec![0.0; n_total]; m];
        for (row, col, value) in problem.constraints.iter_entries() {
            matrix[row][col] = value;
        }
        for i in 0..m {
            matrix[i][n + i] = -1.0;
        }

        let mut lo = vec![0.0; n_total];
        let mut hi = vec![0.0; n_total];
        for j in 0..n {
            lo[j] = bound_lower(&problem.var_bounds[j]);
            hi[j] = bound_upper(&problem.var_bounds[j]);
        }
        for i in 0..m {
            lo[n + i] = bound_lower(&problem.row_bounds[i]);
            hi[n + i] = bound_upper(&problem.row_bounds[i]);
        }

        let mut real_cost = vec![0.0; n_total];
        real_cost[..n].copy_from_slice(&problem.objective);

        let mut nb_status = vec![NonbasicBound::AtLower; n_total];
        let mut x = vec![0.0; n_total];
        for j in 0..n {
            let (status, value) = initial_nonbasic(lo[j], hi[j]);
            nb_status[j] = status;
            x[j] = value;
        }

        // Basic variables (the slacks) start at s = A * x_N — direct
        // substitution, not via Binv, for clarity at this one-time
        // initialization (Binv is trivial anyway: the slack columns
        // form -I, so Binv starts as -I too).
        let mut binv = vec![vec![0.0; m]; m];
        for i in 0..m {
            binv[i][i] = -1.0;
            let mut s = 0.0;
            for j in 0..n {
                s += matrix[i][j] * x[j];
            }
            x[n + i] = s;
        }

        let basis: Vec<usize> = (0..m).map(|i| n + i).collect();
        let mut is_basic = vec![false; n_total];
        for &b in &basis {
            is_basic[b] = true;
        }

        Ok(Self { m, n_total, matrix, lo, hi, real_cost, basis, is_basic, nb_status, binv, x })
    }

    fn column(&self, j: usize) -> Vec<f64> {
        (0..self.m).map(|i| self.matrix[i][j]).collect()
    }

    /// `Binv * A_j` — the pivot column for candidate entering variable `j`.
    fn compute_alpha(&self, j: usize) -> Vec<f64> {
        let col = self.column(j);
        (0..self.m).map(|i| (0..self.m).map(|k| self.binv[i][k] * col[k]).sum()).collect()
    }

    /// `cost_B^T * Binv` — the simplex multiplier row vector.
    fn compute_y(&self, cost: &[f64]) -> Vec<f64> {
        let cost_b: Vec<f64> = self.basis.iter().map(|&b| cost[b]).collect();
        (0..self.m).map(|k| (0..self.m).map(|i| cost_b[i] * self.binv[i][k]).sum()).collect()
    }

    fn reduced_cost(&self, cost: &[f64], y: &[f64], j: usize) -> f64 {
        let col = self.column(j);
        cost[j] - (0..self.m).map(|k| y[k] * col[k]).sum::<f64>()
    }

    /// Sum of basic-variable bound violations at the current point.
    fn total_infeasibility(&self, tol: f64) -> f64 {
        self.basis
            .iter()
            .map(|&b| {
                let v = self.x[b];
                if v < self.lo[b] - tol {
                    self.lo[b] - v
                } else if v > self.hi[b] + tol {
                    v - self.hi[b]
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn pivot(&mut self, alpha: &[f64], r: usize) {
        let pivot_value = alpha[r];
        for k in 0..self.m {
            self.binv[r][k] /= pivot_value;
        }
        for i in 0..self.m {
            if i == r {
                continue;
            }
            let factor = alpha[i];
            if factor == 0.0 {
                continue;
            }
            for k in 0..self.m {
                self.binv[i][k] -= factor * self.binv[r][k];
            }
        }
    }

    /// Runs simplex iterations until optimal, unbounded, or the
    /// iteration cap is hit. `phase1` selects both the cost vector
    /// (Phase 1 ignores `real_cost_if_phase2` and instead recomputes
    /// its own temporary cost — see `phase1_cost` — fresh every
    /// iteration, since it depends on which basic variables are
    /// currently infeasible) and the ratio-test rule (Phase 1's allows
    /// an already-infeasible basic variable to move toward the bound
    /// it's violating without blocking early — see module docs).
    fn run_phase(
        &mut self,
        real_cost_if_phase2: &[f64],
        phase1: bool,
        max_iterations: usize,
        tol: f64,
    ) -> Result<PhaseOutcome, OptimizeError> {
        for _ in 0..max_iterations {
            let cost: Vec<f64> =
                if phase1 { self.phase1_cost(tol) } else { real_cost_if_phase2.to_vec() };
            let y = self.compute_y(&cost);

            // Entering variable: Dantzig's rule (largest eligible
            // reduced-cost magnitude).
            let mut best: Option<(usize, f64, f64)> = None; // (j, delta, score)
            for j in 0..self.n_total {
                if self.is_basic[j] {
                    continue;
                }
                let rc = self.reduced_cost(&cost, &y, j);
                let (eligible, delta, score) = match self.nb_status[j] {
                    NonbasicBound::AtLower => (rc < -tol, 1.0, -rc),
                    NonbasicBound::AtUpper => (rc > tol, -1.0, rc),
                    NonbasicBound::Free => (rc.abs() > tol, if rc < 0.0 { 1.0 } else { -1.0 }, rc.abs()),
                };
                if eligible && best.map_or(true, |(_, _, best_score)| score > best_score) {
                    best = Some((j, delta, score));
                }
            }

            let Some((q, delta, _)) = best else {
                return Ok(PhaseOutcome::Optimal);
            };

            let alpha = self.compute_alpha(q);

            let theta_self = if delta > 0.0 {
                if self.hi[q].is_finite() { self.hi[q] - self.x[q] } else { f64::INFINITY }
            } else if self.lo[q].is_finite() {
                self.x[q] - self.lo[q]
            } else {
                f64::INFINITY
            };

            let mut best_row: Option<(f64, usize, TargetBound)> = None;
            for i in 0..self.m {
                let rate = -alpha[i] * delta;
                if rate.abs() < tol {
                    continue;
                }
                let bi = self.basis[i];
                let v = self.x[bi];
                let (lo_i, hi_i) = (self.lo[bi], self.hi[bi]);

                let target: Option<TargetBound> = if phase1 {
                    if v < lo_i - tol {
                        if rate > 0.0 { Some(TargetBound::Lower) } else { None }
                    } else if v > hi_i + tol {
                        if rate < 0.0 { Some(TargetBound::Upper) } else { None }
                    } else if rate > 0.0 {
                        if hi_i.is_finite() { Some(TargetBound::Upper) } else { None }
                    } else if lo_i.is_finite() {
                        Some(TargetBound::Lower)
                    } else {
                        None
                    }
                } else if rate > 0.0 {
                    if hi_i.is_finite() { Some(TargetBound::Upper) } else { None }
                } else if lo_i.is_finite() {
                    Some(TargetBound::Lower)
                } else {
                    None
                };

                let Some(target) = target else { continue };
                let bound = match target {
                    TargetBound::Lower => lo_i,
                    TargetBound::Upper => hi_i,
                };
                let theta_i = ((bound - v) / rate).max(0.0);

                if best_row.map_or(true, |(best_theta, _, _)| theta_i < best_theta) {
                    best_row = Some((theta_i, i, target));
                }
            }

            let theta_max = match best_row {
                Some((row_theta, _, _)) => theta_self.min(row_theta),
                None => theta_self,
            };

            if !theta_max.is_finite() {
                return Ok(PhaseOutcome::Unbounded);
            }

            // Apply the step to every basic variable and the entering
            // variable itself.
            for i in 0..self.m {
                let bi = self.basis[i];
                self.x[bi] += -alpha[i] * delta * theta_max;
            }
            self.x[q] += delta * theta_max;

            let is_bound_flip = match best_row {
                Some((row_theta, _, _)) => theta_self <= row_theta,
                None => true,
            };

            if is_bound_flip {
                self.nb_status[q] = if delta > 0.0 { NonbasicBound::AtUpper } else { NonbasicBound::AtLower };
            } else {
                let (_, r, target) = best_row.expect("row pivot implies best_row is Some");
                let leaving = self.basis[r];
                self.x[leaving] = match target {
                    TargetBound::Lower => self.lo[leaving],
                    TargetBound::Upper => self.hi[leaving],
                };
                self.nb_status[leaving] = match target {
                    TargetBound::Lower => NonbasicBound::AtLower,
                    TargetBound::Upper => NonbasicBound::AtUpper,
                };
                self.is_basic[leaving] = false;
                self.is_basic[q] = true;
                self.basis[r] = q;
                self.pivot(&alpha, r);
            }
        }
        Ok(PhaseOutcome::IterationLimit)
    }

    /// Phase 1's temporary cost vector — the subgradient of the total
    /// infeasibility function at the current point (see module docs).
    fn phase1_cost(&self, tol: f64) -> Vec<f64> {
        let mut cost = vec![0.0; self.n_total];
        for &b in &self.basis {
            let v = self.x[b];
            if v < self.lo[b] - tol {
                cost[b] = -1.0;
            } else if v > self.hi[b] + tol {
                cost[b] = 1.0;
            }
        }
        cost
    }

    fn solution(&self, n: usize, status: SolveStatus, objective: &[f64]) -> Solution {
        let variable_values: Vec<f64> = self.x[..n].to_vec();
        let objective_value: f64 = objective.iter().zip(&variable_values).map(|(c, x)| c * x).sum();
        Solution { variable_values, objective_value, status }
    }
}

fn bound_lower(bound: &Bound) -> f64 {
    bound.lower.unwrap_or(f64::NEG_INFINITY)
}

fn bound_upper(bound: &Bound) -> f64 {
    bound.upper.unwrap_or(f64::INFINITY)
}

fn initial_nonbasic(lo: f64, hi: f64) -> (NonbasicBound, f64) {
    if lo.is_finite() {
        (NonbasicBound::AtLower, lo)
    } else if hi.is_finite() {
        (NonbasicBound::AtUpper, hi)
    } else {
        (NonbasicBound::Free, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_sparse::CsrMatrix;

    fn dense_problem(
        objective: Vec<f64>,
        rows: Vec<Vec<f64>>,
        row_bounds: Vec<Bound>,
        var_bounds: Vec<Bound>,
    ) -> Problem {
        let n = objective.len();
        let m = rows.len();
        let mut row_ptr = vec![0];
        let mut col_indices = vec![];
        let mut values = vec![];
        for row in &rows {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_indices.len());
        }
        let constraints = CsrMatrix::new(m, n, row_ptr, col_indices, values).unwrap();
        Problem { objective, constraints, row_bounds, var_bounds }
    }

    #[test]
    fn solves_the_ampl_grammar_doc_example() {
        // maximize 3x + 2y  ->  minimize -3x - 2y (Presolve's job, done
        // by hand here since this is a Rust-level test of the solver,
        // not the Swift presolve step).
        // subject to: x + y <= 4, x <= 3; x >= 0, y in [0, 10].
        // Hand-computed optimum (checking every vertex of the feasible
        // region): (x, y) = (3, 1), objective (maximize sense) = 11.
        let problem = dense_problem(
            vec![-3.0, -2.0],
            vec![vec![1.0, 1.0], vec![1.0, 0.0]],
            vec![
                Bound { lower: None, upper: Some(4.0) },
                Bound { lower: None, upper: Some(3.0) },
            ],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: Some(10.0) }],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 3.0).abs() < 1e-6, "x = {}", solution.variable_values[0]);
        assert!((solution.variable_values[1] - 1.0).abs() < 1e-6, "y = {}", solution.variable_values[1]);
        assert!((solution.objective_value - (-11.0)).abs() < 1e-6);
    }

    #[test]
    fn requires_phase_1_for_a_greater_than_constraint() {
        // minimize x + y  s.t.  x + y >= 1, x >= 0, y >= 0.
        // Starting point (0, 0) has slack s = 0, violating the lower
        // bound of 1 -> Phase 1 is genuinely exercised here, not
        // skipped. Optimum: x + y = 1 exactly (multiple optima; only
        // the objective value and feasibility are checked).
        let problem = dense_problem(
            vec![1.0, 1.0],
            vec![vec![1.0, 1.0]],
            vec![Bound { lower: Some(1.0), upper: None }],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: None }],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.objective_value - 1.0).abs() < 1e-6);
        let sum = solution.variable_values[0] + solution.variable_values[1];
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(solution.variable_values[0] >= -1e-9);
        assert!(solution.variable_values[1] >= -1e-9);
    }

    #[test]
    fn equality_constraint_pins_the_solution() {
        // minimize x  s.t.  x = 5, x in [0, 10].
        let problem = dense_problem(
            vec![1.0],
            vec![vec![1.0]],
            vec![Bound::fixed(5.0)],
            vec![Bound { lower: Some(0.0), upper: Some(10.0) }],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 5.0).abs() < 1e-6);
        assert!((solution.objective_value - 5.0).abs() < 1e-6);
    }

    #[test]
    fn detects_infeasibility() {
        // x <= 1 (row 0) and x >= 2 (row 1) can't both hold.
        let problem = dense_problem(
            vec![1.0],
            vec![vec![1.0], vec![1.0]],
            vec![
                Bound { lower: None, upper: Some(1.0) },
                Bound { lower: Some(2.0), upper: None },
            ],
            vec![Bound { lower: Some(0.0), upper: None }],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Infeasible);
    }

    #[test]
    fn detects_unboundedness_with_no_constraints() {
        // minimize -x, x >= 0, no upper bound and no constraint rows.
        let problem = dense_problem(
            vec![-1.0],
            vec![],
            vec![],
            vec![Bound { lower: Some(0.0), upper: None }],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Unbounded);
    }

    #[test]
    fn bound_flip_with_no_constraint_rows() {
        // minimize -x, x in [0, 5], no constraint rows at all -> the
        // optimum is reached via a pure bound flip, no pivot.
        let problem = dense_problem(
            vec![-1.0],
            vec![],
            vec![],
            vec![Bound { lower: Some(0.0), upper: Some(5.0) }],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 5.0).abs() < 1e-6);
        assert!((solution.objective_value - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn free_variable_is_handled() {
        // minimize (x - 3)... not linear, so instead: minimize x s.t.
        // x + y = 0, y free (no bounds at all), x >= 0.
        // Forces y = -x; minimizing x with x >= 0 gives x = 0, y = 0.
        let problem = dense_problem(
            vec![1.0, 0.0],
            vec![vec![1.0, 1.0]],
            vec![Bound::fixed(0.0)],
            vec![Bound { lower: Some(0.0), upper: None }, Bound::free()],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 0.0).abs() < 1e-6);
        assert!((solution.variable_values[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(RevisedSimplexSolver::default().name(), "revised-simplex");
    }

    #[test]
    fn greedy_knapsack_style_lp_with_three_variables() {
        // minimize 2x + 3y + z  s.t.  x + y + z >= 10,
        // x, y, z in [0, 4]. Distinct costs + a shared capacity-style
        // constraint means the optimum is the "cheapest first" greedy
        // allocation: z (cost 1) to its cap of 4, then x (cost 2) to
        // its cap of 4, then y (cost 3) covers the remaining 2.
        // Expected: x=4, y=2, z=4, objective = 2*4 + 3*2 + 1*4 = 18.
        let problem = dense_problem(
            vec![2.0, 3.0, 1.0],
            vec![vec![1.0, 1.0, 1.0]],
            vec![Bound { lower: Some(10.0), upper: None }],
            vec![
                Bound { lower: Some(0.0), upper: Some(4.0) },
                Bound { lower: Some(0.0), upper: Some(4.0) },
                Bound { lower: Some(0.0), upper: Some(4.0) },
            ],
        );

        let solution = RevisedSimplexSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 4.0).abs() < 1e-6, "x = {}", solution.variable_values[0]);
        assert!((solution.variable_values[1] - 2.0).abs() < 1e-6, "y = {}", solution.variable_values[1]);
        assert!((solution.variable_values[2] - 4.0).abs() < 1e-6, "z = {}", solution.variable_values[2]);
        assert!((solution.objective_value - 18.0).abs() < 1e-6);
    }
}
