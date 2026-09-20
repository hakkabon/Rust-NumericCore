//! Branch-and-bound for mixed-integer linear programs — the third real
//! `Solver`, and the first to consume `Problem::is_integer`.
//!
//! ## The algorithm, at the level this implements it
//! Standard LP-relaxation-based branch-and-bound: solve the LP
//! relaxation (ignoring integrality) at the root. If every
//! integer-restricted variable already has an integer value (within
//! `integer_tolerance`), that relaxation solution *is* the MILP
//! optimum — done. Otherwise pick one fractional integer variable and
//! branch: one child adds `x_j <= floor(value)`, the other
//! `x_j >= ceil(value)`, each explored as its own node (a fresh LP
//! relaxation with that one tightened bound). Depth-first, with a
//! stack, not a queue.
//!
//! **Why this is correct (the bounding argument):** for a minimize
//! problem, relaxing integrality only enlarges the feasible region, so
//! a node's LP-relaxation objective is always a lower bound on the
//! true integer optimum anywhere in that node's subtree. Once an
//! integer-feasible solution (the "incumbent") is found, any node whose
//! own relaxation bound is already no better than the incumbent can be
//! pruned without exploring it further — it cannot possibly contain a
//! better integer solution. This is the only pruning rule implemented;
//! it's sufficient for correctness on its own (more sophisticated
//! bounding — e.g. cutting planes — only improves speed).
//!
//! ## Scope, deliberately (matching `simplex`/`interior_point`'s stance)
//! - **Most-fractional branching** (the integer variable whose value is
//!   closest to `x.floor() + 0.5`), not pseudocost or strong branching.
//!   Simpler, correct, slower to converge on hard instances — a real
//!   future improvement, not a correctness gap.
//! - **Depth-first search**, not best-first/best-bound. Uses less
//!   memory (a stack of node bound-overrides, not a priority queue of
//!   fully-expanded nodes) and finds *a* feasible incumbent quickly,
//!   at the cost of not necessarily closing the optimality gap as fast
//!   as best-first would on a large tree. Matches this project's
//!   "smallish problems" framing for the whole LP/MILP path.
//! - **A node limit (`max_nodes`), not a rigorous gap-based stopping
//!   rule.** If the limit is hit, the best incumbent found so far
//!   (if any) is returned with `SolveStatus::IterationLimit` — a valid
//!   feasible solution, just not proven optimal. If no incumbent was
//!   ever found before the limit, `IterationLimit` is returned with an
//!   empty solution rather than misreporting `Infeasible` (the problem
//!   might still be feasible; the search just didn't find out in time).
//! - **The relaxation solver is pluggable but must itself be exact for
//!   this to be correct** — `BranchAndBoundSolver` trusts its
//!   relaxation solver's `Optimal`/`Infeasible`/`Unbounded` reports at
//!   face value. `RevisedSimplexSolver` (the default) has rigorous,
//!   finite-arithmetic termination for both; `InteriorPointSolver`'s
//!   unboundedness detection is documented as heuristic — passing it as
//!   the relaxation solver here would inherit that same heuristic-ness
//!   at the MILP level, and it also rejects equality-constrained rows
//!   and fixed variables outright (see its own module docs), which a
//!   branch's added bound can easily produce (a floor/ceil branch can
//!   pin a variable to a single value). `RevisedSimplexSolver` doesn't
//!   have either limitation, which is why it's the default rather than
//!   an arbitrary choice.

use crate::{Bound, OptimizeError, Problem, RevisedSimplexSolver, SolveStatus, Solution, Solver};

pub struct BranchAndBoundSolver {
    pub max_nodes: usize,
    pub integer_tolerance: f64,
    pub relaxation_solver: Box<dyn Solver>,
}

impl Default for BranchAndBoundSolver {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            integer_tolerance: 1e-6,
            relaxation_solver: Box::new(RevisedSimplexSolver::default()),
        }
    }
}

/// One node's bound overrides, layered on top of the original
/// problem's `var_bounds` — a node never *relaxes* a bound, only
/// tightens one relative to its parent, so overrides compose by
/// intersecting with whatever was already in force.
#[derive(Clone)]
struct NodeBounds {
    var_bounds: Vec<Bound>,
}

impl NodeBounds {
    fn tightened(&self, var_index: usize, lower: Option<f64>, upper: Option<f64>) -> Self {
        let mut var_bounds = self.var_bounds.clone();
        let current = var_bounds[var_index];
        var_bounds[var_index] = Bound {
            lower: tighter_lower(current.lower, lower),
            upper: tighter_upper(current.upper, upper),
        };
        Self { var_bounds }
    }
}

fn tighter_lower(current: Option<f64>, candidate: Option<f64>) -> Option<f64> {
    match (current, candidate) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

fn tighter_upper(current: Option<f64>, candidate: Option<f64>) -> Option<f64> {
    match (current, candidate) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

impl Solver for BranchAndBoundSolver {
    fn name(&self) -> &'static str {
        "branch-and-bound"
    }

    fn solve(&self, problem: &Problem) -> Result<Solution, OptimizeError> {
        if problem.is_integer.len() != problem.objective.len() {
            return Err(OptimizeError::NotImplemented(
                "branch-and-bound: is_integer length must match objective length",
            ));
        }
        if !problem.is_integer.iter().any(|&b| b) {
            // No integer variables at all - this is just an LP. Solve
            // it directly rather than paying for a branch-and-bound
            // tree with nothing to branch on.
            return self.relaxation_solver.solve(problem);
        }

        let mut stack = vec![NodeBounds { var_bounds: problem.var_bounds.clone() }];
        let mut incumbent: Option<Solution> = None;
        let mut nodes_explored = 0usize;

        while let Some(node) = stack.pop() {
            nodes_explored += 1;
            if nodes_explored > self.max_nodes {
                return Ok(match incumbent {
                    Some(sol) => Solution { status: SolveStatus::IterationLimit, ..sol },
                    None => Solution {
                        variable_values: vec![],
                        objective_value: 0.0,
                        status: SolveStatus::IterationLimit,
                    },
                });
            }

            let node_problem = Problem {
                objective: problem.objective.clone(),
                constraints: problem.constraints.clone(),
                row_bounds: problem.row_bounds.clone(),
                var_bounds: node.var_bounds.clone(),
                is_integer: problem.is_integer.clone(),
            };

            let relaxed = self.relaxation_solver.solve(&node_problem)?;

            match relaxed.status {
                SolveStatus::Infeasible => {
                    continue; // prune: this branch has no feasible region at all
                }
                SolveStatus::Unbounded => {
                    // An unbounded relaxation with no integer variables
                    // fixed yet means the MILP itself is unbounded (the
                    // objective can be driven arbitrarily far while
                    // still eventually crossing integer points, in
                    // every case this project's scope cares about).
                    // Deeper in the tree, added bounds normally rule
                    // this out; if it still happens, treat it the same
                    // way.
                    return Ok(relaxed);
                }
                SolveStatus::IterationLimit => {
                    // The relaxation itself didn't converge - can't
                    // trust its bound for pruning. Skip this node
                    // rather than either wrongly pruning or wrongly
                    // accepting it as a bound.
                    continue;
                }
                SolveStatus::Optimal => {}
            }

            // Bounding: prune if this node cannot possibly beat the
            // incumbent (minimize sense - Problem/Solution are always
            // in minimize form, per Presolve's convention).
            if let Some(ref best) = incumbent {
                if relaxed.objective_value >= best.objective_value - self.integer_tolerance {
                    continue;
                }
            }

            match most_fractional_integer_variable(&relaxed, &problem.is_integer, self.integer_tolerance) {
                None => {
                    // Every integer-restricted variable already has an
                    // integer value - this relaxation solution is
                    // integer-feasible, and (by the bounding check just
                    // above) strictly better than any prior incumbent.
                    incumbent = Some(relaxed);
                }
                Some((var_index, value)) => {
                    let floor_value = value.floor();
                    let ceil_value = value.ceil();
                    stack.push(node.tightened(var_index, None, Some(floor_value)));
                    stack.push(node.tightened(var_index, Some(ceil_value), None));
                }
            }
        }

        match incumbent {
            Some(sol) => Ok(sol),
            None => Ok(Solution { variable_values: vec![], objective_value: 0.0, status: SolveStatus::Infeasible }),
        }
    }
}

/// Finds the integer-restricted variable furthest from an integer
/// value (closest to `.floor() + 0.5`), or `None` if every
/// integer-restricted variable is already within `tolerance` of an
/// integer.
fn most_fractional_integer_variable(
    solution: &Solution,
    is_integer: &[bool],
    tolerance: f64,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64, f64)> = None; // (index, value, fractionality)
    for (j, &integer) in is_integer.iter().enumerate() {
        if !integer {
            continue;
        }
        let value = solution.variable_values[j];
        let fractional_part = value - value.floor();
        let distance_from_integer = fractional_part.min(1.0 - fractional_part);
        if distance_from_integer <= tolerance {
            continue;
        }
        if best.map_or(true, |(_, _, best_dist)| distance_from_integer > best_dist) {
            best = Some((j, value, distance_from_integer));
        }
    }
    best.map(|(j, value, _)| (j, value))
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
        is_integer: Vec<bool>,
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
        Problem { objective, constraints, row_bounds, var_bounds, is_integer }
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(BranchAndBoundSolver::default().name(), "branch-and-bound");
    }

    #[test]
    fn falls_through_to_relaxation_solver_when_nothing_is_integer() {
        // No integer variables at all - should match RevisedSimplexSolver
        // exactly (this is literally the same code path).
        let problem = dense_problem(
            vec![-3.0, -2.0],
            vec![vec![1.0, 1.0], vec![1.0, 0.0]],
            vec![
                Bound { lower: None, upper: Some(4.0) },
                Bound { lower: None, upper: Some(3.0) },
            ],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: Some(10.0) }],
            vec![false, false],
        );
        let solution = BranchAndBoundSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 3.0).abs() < 1e-6);
        assert!((solution.variable_values[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn solves_a_classic_two_variable_milp_requiring_real_branching() {
        // maximize 5x + 4y (given here as minimize -5x - 4y)
        // s.t. 6x + 4y <= 24, x + 2y <= 6, x, y >= 0 integer.
        //
        // LP relaxation optimum: (x, y) = (3, 1.5), objective 21 - a
        // genuinely fractional vertex, so this exercises real branching,
        // not a lucky already-integer relaxation.
        //
        // Integer optimum, hand-verified by enumerating every integer
        // point reachable near the relaxation's optimal face: (4, 0),
        // objective 20. (3,1) gives 19; (2,2) gives 18; (4,1) is
        // infeasible (28 > 24); (5,0) is infeasible (30 > 24).
        let problem = dense_problem(
            vec![-5.0, -4.0],
            vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            vec![
                Bound { lower: None, upper: Some(24.0) },
                Bound { lower: None, upper: Some(6.0) },
            ],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: None }],
            vec![true, true],
        );

        let solution = BranchAndBoundSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 4.0).abs() < 1e-6, "x = {}", solution.variable_values[0]);
        assert!((solution.variable_values[1] - 0.0).abs() < 1e-6, "y = {}", solution.variable_values[1]);
        assert!((solution.objective_value - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn solves_a_01_knapsack() {
        // maximize 6x0 + 10x1 + 12x2 (minimize the negation)
        // s.t. x0 + 2x1 + 3x2 <= 5, each xi in {0, 1}.
        //
        // Hand-verified optimum: items {1, 2} (weight 2+3=5, value
        // 10+12=22) beats every other combination that fits (all three
        // items together weigh 6 > 5).
        let problem = dense_problem(
            vec![-6.0, -10.0, -12.0],
            vec![vec![1.0, 2.0, 3.0]],
            vec![Bound { lower: None, upper: Some(5.0) }],
            vec![
                Bound { lower: Some(0.0), upper: Some(1.0) },
                Bound { lower: Some(0.0), upper: Some(1.0) },
                Bound { lower: Some(0.0), upper: Some(1.0) },
            ],
            vec![true, true, true],
        );

        let solution = BranchAndBoundSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 0.0).abs() < 1e-6);
        assert!((solution.variable_values[1] - 1.0).abs() < 1e-6);
        assert!((solution.variable_values[2] - 1.0).abs() < 1e-6);
        assert!((solution.objective_value - (-22.0)).abs() < 1e-6);
    }

    #[test]
    fn detects_infeasibility_caused_by_integrality_itself() {
        // minimize x s.t. 2x = 1, x integer, x in [0, 10]. The LP
        // relaxation (x = 0.5) is feasible, but no integer x can ever
        // satisfy 2x = 1 - both branches (x <= 0 and x >= 1) make the
        // equality infeasible.
        let problem = dense_problem(
            vec![1.0],
            vec![vec![2.0]],
            vec![Bound::fixed(1.0)],
            vec![Bound { lower: Some(0.0), upper: Some(10.0) }],
            vec![true],
        );

        let solution = BranchAndBoundSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Infeasible);
    }

    #[test]
    fn already_integer_relaxation_needs_no_branching() {
        // minimize x + y s.t. x + y >= 4, x, y >= 0 integer, where the
        // LP relaxation's own optimum already happens to land on
        // integers (e.g. x=4, y=0) - confirms the "no branching needed"
        // path returns the right answer, not just that it terminates.
        let problem = dense_problem(
            vec![1.0, 1.0],
            vec![vec![1.0, 1.0]],
            vec![Bound { lower: Some(4.0), upper: None }],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: None }],
            vec![true, true],
        );

        let solution = BranchAndBoundSolver::default().solve(&problem).unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.objective_value - 4.0).abs() < 1e-6);
        let sum = solution.variable_values[0] + solution.variable_values[1];
        assert!((sum - 4.0).abs() < 1e-6);
        for &v in &solution.variable_values {
            assert!((v - v.round()).abs() < 1e-6, "expected integer, got {v}");
        }
    }

    #[test]
    fn rejects_mismatched_is_integer_length() {
        let mut problem = dense_problem(
            vec![1.0],
            vec![vec![1.0]],
            vec![Bound { lower: None, upper: Some(1.0) }],
            vec![Bound { lower: Some(0.0), upper: None }],
            vec![true],
        );
        problem.is_integer = vec![true, true]; // wrong length on purpose
        let result = BranchAndBoundSolver::default().solve(&problem);
        assert!(matches!(result, Err(OptimizeError::NotImplemented(_))));
    }
}
