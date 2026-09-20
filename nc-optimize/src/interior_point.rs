//! Primal-dual interior-point method for bounded-variable LP — the
//! second real `Solver`, alongside `simplex::RevisedSimplexSolver`.
//!
//! ## Formulation
//! Same starting point as `simplex`: convert to `A x = b`,
//! `l <= x <= u` by giving each constraint row a slack variable
//! (`Ax - s = 0`, row bounds applied to `s` directly — see
//! `simplex`'s module docs for the shared derivation). The two solvers
//! diverge from there because a barrier method needs a strictly
//! interior starting point, which the box `[l, u]` must actually have.
//!
//! ## Two scope boundaries this needs that simplex didn't
//! **1. No zero-width boxes.** A barrier method has no interior to work
//! in if `l_j == u_j` for some component — which is exactly what an
//! equality constraint (`row_bounds.lower == row_bounds.upper`) or a
//! fixed variable (`var_bounds.lower == var_bounds.upper`) produces
//! under the slack formulation above. This is not a corner being cut:
//! essentially every real barrier-method implementation handles
//! equalities/fixed variables via a **separate presolve/elimination
//! step** before the barrier method ever runs, not inside the barrier
//! iteration itself. This solver doesn't yet have that presolve step —
//! `solve` returns `OptimizeError::NotImplemented` up front if it finds
//! a zero-width box, rather than silently producing garbage (dividing
//! by a zero-width gap) or looping forever. Adding the elimination step
//! is real, separate, well-scoped future work, not a fix to this file.
//!
//! **2. One-sided/free bounds are handled via a finite substitute,
//! with a heuristic (not certificate-based) unboundedness check.**
//! A missing bound is replaced with `±big_bound` (a large finite
//! number) so every component has a genuine interior. This means an
//! actually-unbounded LP will drive that component toward
//! `±big_bound` rather than diverging outright; `solve` checks for a
//! final solution suspiciously close to a substituted bound and
//! reports `SolveStatus::Unbounded` on that basis. This is a heuristic,
//! not a proof — rigorous infeasibility/unboundedness detection in
//! barrier methods normally needs a self-dual embedding (HSD), which
//! is real future work, not attempted here. `RevisedSimplexSolver`'s
//! detection is the rigorous one; prefer it when a certificate matters.
//!
//! ## The Newton system, and why it's an SPD solve
//! Eliminating `dz`/`dw` from the linearized KKT conditions (see the
//! derivation this module was built from) reduces the system to
//! `(A D⁻¹ Aᵀ) dy = rhs`, where `D` is a positive diagonal matrix — so
//! `A D⁻¹ Aᵀ` is symmetric positive definite whenever `A` has full row
//! rank. `nc_decomp::cholesky_solve` solves it each iteration. This is
//! the SPD connection flagged during the simplex-vs-interior-point
//! scoping discussion — `solveSPD` on the Swift/Accelerate side and
//! this Rust-native Cholesky are the same mathematical idea serving two
//! different solvers in two different languages.
//!
//! ## Scope, deliberately (mirroring `simplex`'s stance)
//! Fixed centering parameter (`sigma`), not an adaptive/Mehrotra
//! predictor-corrector scheme — fewer iterations is a real future
//! improvement, not a correctness concern; a single common step length
//! for primal and dual variables, not separate ones — simpler, more
//! conservative, still convergent.

use crate::{Bound, OptimizeError, Problem, SolveStatus, Solution, Solver};
use nc_decomp::cholesky_solve;

pub struct InteriorPointSolver {
    pub max_iterations: usize,
    /// Convergence tolerance on primal/dual residuals and the
    /// complementarity gap.
    pub tolerance: f64,
    /// Centering parameter — target complementarity gap each iteration
    /// is `sigma` times the current gap. Fixed, not adaptive; see
    /// module docs.
    pub sigma: f64,
    /// Finite substitute for a missing (`None`) bound. Large enough to
    /// not distort a genuinely bounded problem's optimum, small enough
    /// that `f64` arithmetic on values near it stays well-conditioned.
    pub big_bound: f64,
    /// Fraction-to-the-boundary safety margin (standard IPM practice:
    /// never step exactly onto a bound).
    pub step_fraction: f64,
}

impl Default for InteriorPointSolver {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            sigma: 0.1,
            big_bound: 1e7,
            step_fraction: 0.995,
        }
    }
}

impl Solver for InteriorPointSolver {
    fn name(&self) -> &'static str {
        "interior-point"
    }

    fn solve(&self, problem: &Problem) -> Result<Solution, OptimizeError> {
        let workspace = Workspace::new(problem, self.big_bound)?;
        workspace.run(problem, self)
    }
}

struct Workspace {
    m: usize,
    n_total: usize,
    /// Dense `m x n_total` constraint matrix `[A | -I]` — same
    /// construction as `simplex::Workspace`, duplicated locally rather
    /// than shared, since the two solvers' internal representations
    /// are otherwise unrelated (this one never needs a basis or an
    /// explicit inverse).
    matrix: Vec<Vec<f64>>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Whether `lo[j]`/`hi[j]` is a real problem bound (`false`) or a
    /// `big_bound` substitute for a missing one (`true`) — used only
    /// by the final heuristic unboundedness check.
    lo_is_substitute: Vec<bool>,
    hi_is_substitute: Vec<bool>,
    real_cost: Vec<f64>,
}

impl Workspace {
    fn new(problem: &Problem, big_bound: f64) -> Result<Self, OptimizeError> {
        let m = problem.constraints.rows();
        let n = problem.objective.len();
        let n_total = n + m;

        if problem.var_bounds.len() != n || problem.row_bounds.len() != m {
            return Err(OptimizeError::NotImplemented(
                "interior point: var_bounds/row_bounds length must match problem dimensions",
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
        let mut lo_is_substitute = vec![false; n_total];
        let mut hi_is_substitute = vec![false; n_total];

        for j in 0..n {
            fill_bound(
                &problem.var_bounds[j],
                big_bound,
                &mut lo,
                &mut hi,
                &mut lo_is_substitute,
                &mut hi_is_substitute,
                j,
            )?;
        }
        for i in 0..m {
            fill_bound(
                &problem.row_bounds[i],
                big_bound,
                &mut lo,
                &mut hi,
                &mut lo_is_substitute,
                &mut hi_is_substitute,
                n + i,
            )?;
        }

        let mut real_cost = vec![0.0; n_total];
        real_cost[..n].copy_from_slice(&problem.objective);

        Ok(Self { m, n_total, matrix, lo, hi, lo_is_substitute, hi_is_substitute, real_cost })
    }

    fn run(&self, problem: &Problem, config: &InteriorPointSolver) -> Result<Solution, OptimizeError> {
        let n = problem.objective.len();
        let tol = config.tolerance;
        let m = self.m;
        let n_total = self.n_total;

        // Naive but robust starting point: midpoint of each box, unit
        // dual multipliers. Not the most efficient start available,
        // but a simple, always-valid interior point.
        let mut x: Vec<f64> = (0..n_total).map(|j| 0.5 * (self.lo[j] + self.hi[j])).collect();
        let mut y = vec![0.0; m];
        let mut z = vec![1.0; n_total];
        let mut w = vec![1.0; n_total];

        for _ in 0..config.max_iterations {
            // Residuals: r_p = A x (target 0, since b = 0 by
            // construction); r_d = A^T y + z - w - c.
            let r_p: Vec<f64> = (0..m).map(|i| dot(&self.matrix[i], &x)).collect();
            let r_d: Vec<f64> = (0..n_total)
                .map(|j| {
                    let at_y_j: f64 = (0..m).map(|i| self.matrix[i][j] * y[i]).sum();
                    at_y_j + z[j] - w[j] - self.real_cost[j]
                })
                .collect();

            let gap: f64 = (0..n_total)
                .map(|j| (x[j] - self.lo[j]) * z[j] + (self.hi[j] - x[j]) * w[j])
                .sum::<f64>()
                / (2.0 * n_total as f64);

            let r_p_norm = norm(&r_p);
            let r_d_norm = norm(&r_d);
            if r_p_norm < tol && r_d_norm < tol && gap < tol {
                return Ok(self.finish(&x, n, SolveStatus::Optimal, config));
            }

            let mu_target = config.sigma * gap;

            // D_j = z_j/(x_j - lo_j) + w_j/(hi_j - x_j); rc_j as
            // derived in this module's docs (the eliminated-dz/dw
            // stationarity residual).
            let mut d = vec![0.0; n_total];
            let mut rc = vec![0.0; n_total];
            for j in 0..n_total {
                let xl = x[j] - self.lo[j];
                let ux = self.hi[j] - x[j];
                d[j] = z[j] / xl + w[j] / ux;
                rc[j] = -r_d[j] - mu_target / xl + z[j] + mu_target / ux - w[j];
            }

            // Normal equations: (A D^-1 A^T) dy = A D^-1 rc - r_p.
            let mut n_matrix = vec![vec![0.0; m]; m];
            for i in 0..m {
                for k in 0..m {
                    let mut s = 0.0;
                    for j in 0..n_total {
                        s += self.matrix[i][j] * self.matrix[k][j] / d[j];
                    }
                    n_matrix[i][k] = s;
                }
            }
            let rhs: Vec<f64> = (0..m)
                .map(|i| {
                    let a_dinv_rc: f64 = (0..n_total).map(|j| self.matrix[i][j] * rc[j] / d[j]).sum();
                    a_dinv_rc - r_p[i]
                })
                .collect();

            let Some(dy) = cholesky_solve(&n_matrix, &rhs) else {
                // A D^-1 A^T should always be SPD for full-row-rank A;
                // failure here means a genuinely degenerate/rank-
                // deficient constraint matrix, which this solver
                // doesn't attempt to recover from.
                return Err(OptimizeError::NotImplemented(
                    "interior point: normal equations matrix was not positive definite (degenerate or rank-deficient constraints?)",
                ));
            };

            let mut dx = vec![0.0; n_total];
            for j in 0..n_total {
                let at_dy_j: f64 = (0..m).map(|i| self.matrix[i][j] * dy[i]).sum();
                dx[j] = (at_dy_j - rc[j]) / d[j];
            }
            let dz: Vec<f64> = (0..n_total)
                .map(|j| {
                    let xl = x[j] - self.lo[j];
                    mu_target / xl - z[j] - (z[j] / xl) * dx[j]
                })
                .collect();
            let dw: Vec<f64> = (0..n_total)
                .map(|j| {
                    let ux = self.hi[j] - x[j];
                    mu_target / ux - w[j] + (w[j] / ux) * dx[j]
                })
                .collect();

            // Fraction-to-the-boundary step length, single common alpha
            // for x, z, and w (see module docs).
            let mut alpha = 1.0_f64;
            for j in 0..n_total {
                if dx[j] < 0.0 {
                    alpha = alpha.min(config.step_fraction * (self.lo[j] - x[j]) / dx[j]);
                } else if dx[j] > 0.0 {
                    alpha = alpha.min(config.step_fraction * (self.hi[j] - x[j]) / dx[j]);
                }
                if dz[j] < 0.0 {
                    alpha = alpha.min(config.step_fraction * (-z[j] / dz[j]));
                }
                if dw[j] < 0.0 {
                    alpha = alpha.min(config.step_fraction * (-w[j] / dw[j]));
                }
            }
            alpha = alpha.max(0.0);

            for j in 0..n_total {
                x[j] += alpha * dx[j];
                z[j] += alpha * dz[j];
                w[j] += alpha * dw[j];
            }
            for i in 0..m {
                y[i] += alpha * dy[i];
            }
        }

        Ok(self.finish(&x, n, SolveStatus::IterationLimit, config))
    }

    /// Heuristic unboundedness check (see module docs) plus objective
    /// extraction, shared by every exit path from `run`.
    fn finish(&self, x: &[f64], n: usize, status: SolveStatus, config: &InteriorPointSolver) -> Solution {
        let detection_margin = config.big_bound * 1e-3;
        let looks_unbounded = (0..self.n_total).any(|j| {
            (self.hi_is_substitute[j] && (self.hi[j] - x[j]).abs() < detection_margin)
                || (self.lo_is_substitute[j] && (x[j] - self.lo[j]).abs() < detection_margin)
        });

        let final_status = if status == SolveStatus::Optimal && looks_unbounded {
            SolveStatus::Unbounded
        } else {
            status
        };

        let variable_values: Vec<f64> = x[..n].to_vec();
        let objective_value: f64 =
            self.real_cost[..n].iter().zip(&variable_values).map(|(c, v)| c * v).sum();
        Solution { variable_values, objective_value, status: final_status }
    }
}

fn fill_bound(
    bound: &Bound,
    big_bound: f64,
    lo: &mut [f64],
    hi: &mut [f64],
    lo_is_substitute: &mut [bool],
    hi_is_substitute: &mut [bool],
    j: usize,
) -> Result<(), OptimizeError> {
    if let (Some(l), Some(u)) = (bound.lower, bound.upper) {
        if (u - l).abs() < 1e-12 {
            return Err(OptimizeError::NotImplemented(
                "interior point: equality-constrained rows and fixed variables (zero-width bounds) require a presolve elimination step this solver doesn't implement yet — see interior_point module docs",
            ));
        }
    }
    match bound.lower {
        Some(l) => lo[j] = l,
        None => {
            lo[j] = -big_bound;
            lo_is_substitute[j] = true;
        }
    }
    match bound.upper {
        Some(u) => hi[j] = u,
        None => {
            hi[j] = big_bound;
            hi_is_substitute[j] = true;
        }
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
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
        Problem { objective, constraints, row_bounds, var_bounds, is_integer: vec![false; n] }
    }

    #[test]
    fn agrees_with_simplex_on_the_ampl_grammar_doc_example() {
        // Same problem, and same hand-verified expected answer, as
        // simplex.rs's equivalent test: maximize 3x+2y (given here
        // pre-negated to minimize -3x-2y, matching what Presolve
        // would produce) s.t. x+y<=4, x<=3, x in [0,inf), y in [0,10].
        // Optimum: (3, 1), objective -11.
        let problem = dense_problem(
            vec![-3.0, -2.0],
            vec![vec![1.0, 1.0], vec![1.0, 0.0]],
            vec![
                Bound { lower: None, upper: Some(4.0) },
                Bound { lower: None, upper: Some(3.0) },
            ],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: Some(10.0) }],
        );

        let solution = InteriorPointSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 3.0).abs() < 1e-3, "x = {}", solution.variable_values[0]);
        assert!((solution.variable_values[1] - 1.0).abs() < 1e-3, "y = {}", solution.variable_values[1]);
        assert!((solution.objective_value - (-11.0)).abs() < 1e-3);
    }

    #[test]
    fn agrees_with_simplex_on_the_greedy_knapsack_lp() {
        // Same problem as simplex.rs's three-variable test: minimize
        // 2x+3y+z s.t. x+y+z>=10, x,y,z in [0,4]. Known unique optimum
        // (distinct costs): x=4, y=2, z=4, objective 18.
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

        let solution = InteriorPointSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 4.0).abs() < 1e-3, "x = {}", solution.variable_values[0]);
        assert!((solution.variable_values[1] - 2.0).abs() < 1e-3, "y = {}", solution.variable_values[1]);
        assert!((solution.variable_values[2] - 4.0).abs() < 1e-3, "z = {}", solution.variable_values[2]);
        assert!((solution.objective_value - 18.0).abs() < 1e-3);
    }

    #[test]
    fn rejects_equality_row_bound() {
        let problem = dense_problem(
            vec![1.0],
            vec![vec![1.0]],
            vec![Bound::fixed(5.0)],
            vec![Bound { lower: Some(0.0), upper: Some(10.0) }],
        );
        let result = InteriorPointSolver::default().solve(&problem);
        assert!(matches!(result, Err(OptimizeError::NotImplemented(_))));
    }

    #[test]
    fn rejects_fixed_variable() {
        let problem = dense_problem(
            vec![1.0, 1.0],
            vec![vec![1.0, 1.0]],
            vec![Bound { lower: None, upper: Some(10.0) }],
            vec![Bound::fixed(3.0), Bound { lower: Some(0.0), upper: None }],
        );
        let result = InteriorPointSolver::default().solve(&problem);
        assert!(matches!(result, Err(OptimizeError::NotImplemented(_))));
    }

    #[test]
    fn handles_no_constraint_rows_via_bound_flip_equivalent() {
        // minimize -x, x in [0, 5], no constraint rows at all (m = 0).
        let problem = dense_problem(vec![-1.0], vec![], vec![], vec![Bound { lower: Some(0.0), upper: Some(5.0) }]);

        let solution = InteriorPointSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 5.0).abs() < 1e-3);
        assert!((solution.objective_value - (-5.0)).abs() < 1e-3);
    }

    #[test]
    fn detects_unboundedness_heuristically() {
        // minimize -x, x >= 0, no upper bound, no constraints -> x
        // should be driven toward the big_bound substitute.
        let problem = dense_problem(vec![-1.0], vec![], vec![], vec![Bound { lower: Some(0.0), upper: None }]);

        let solution = InteriorPointSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Unbounded);
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(InteriorPointSolver::default().name(), "interior-point");
    }
}
