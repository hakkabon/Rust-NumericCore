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
//! - **Selectable depth-first or best-bound search.** Best-bound is the
//!   default because it focuses work on closing the global certificate;
//!   depth-first remains available for memory-sensitive workloads.
//! - **Node and rigorous gap limits.** If the node limit is hit, the best incumbent found so far
//!   (if any) is returned with `SolveStatus::IterationLimit` — a valid
//!   feasible solution, just not proven optimal. If no incumbent was
//!   ever found before the limit, `IterationLimit` is returned with an
//!   empty solution rather than misreporting `Infeasible` (the problem
//!   might still be feasible; the search just didn't find out in time).
//!   Absolute and relative MIP-gap tolerances can also terminate search;
//!   `BranchAndBoundReport::termination` distinguishes that accepted
//!   certificate from exact tree exhaustion.
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

/// Policy used to choose the next open node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSelection {
    /// Low memory and often finds an incumbent quickly.
    DepthFirst,
    /// Expands the node with the smallest valid lower bound first and
    /// therefore concentrates work on closing the global optimality gap.
    BestBound,
}

/// Why a branch-and-bound search returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchAndBoundTermination {
    Exhausted,
    GapSatisfied,
    NodeLimit,
    RelaxationLimit,
    Unbounded,
    ContinuousRelaxation,
}

pub struct BranchAndBoundSolver {
    pub max_nodes: usize,
    pub integer_tolerance: f64,
    /// Stop with a feasible incumbent once `incumbent - best_bound`
    /// is at most this value. Zero requires an exact closed gap.
    pub absolute_gap_tolerance: f64,
    /// Scale-independent counterpart to `absolute_gap_tolerance`.
    pub relative_gap_tolerance: f64,
    pub node_selection: NodeSelection,
    pub relaxation_solver: Box<dyn Solver>,
}

/// Search information needed to interpret an early MILP termination.
#[derive(Debug, Clone)]
pub struct BranchAndBoundReport {
    pub solution: Solution,
    pub nodes_explored: usize,
    /// Global lower bound for this minimization problem, when known.
    pub best_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub nodes_pruned_infeasible: usize,
    pub nodes_pruned_by_bound: usize,
    pub maximum_depth: usize,
    pub termination: BranchAndBoundTermination,
}

impl Default for BranchAndBoundSolver {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            integer_tolerance: 1e-6,
            absolute_gap_tolerance: 0.0,
            relative_gap_tolerance: 0.0,
            node_selection: NodeSelection::BestBound,
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
    /// Valid lower bound inherited from this node's parent relaxation.
    lower_bound: Option<f64>,
    depth: usize,
}

impl NodeBounds {
    fn tightened(&self, var_index: usize, lower: Option<f64>, upper: Option<f64>) -> Self {
        let mut var_bounds = self.var_bounds.clone();
        let current = var_bounds[var_index];
        var_bounds[var_index] = Bound {
            lower: tighter_lower(current.lower, lower),
            upper: tighter_upper(current.upper, upper),
        };
        Self { var_bounds, lower_bound: self.lower_bound, depth: self.depth + 1 }
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
        Ok(self.solve_with_report(problem)?.solution)
    }
}

impl BranchAndBoundSolver {
    pub fn solve_with_report(
        &self,
        problem: &Problem,
    ) -> Result<BranchAndBoundReport, OptimizeError> {
        problem.validate()?;
        if self.max_nodes == 0
            || !self.integer_tolerance.is_finite()
            || self.integer_tolerance <= 0.0
            || !self.absolute_gap_tolerance.is_finite()
            || self.absolute_gap_tolerance < 0.0
            || !self.relative_gap_tolerance.is_finite()
            || self.relative_gap_tolerance < 0.0
        {
            return Err(OptimizeError::InvalidConfiguration(
                "branch-and-bound requires max_nodes > 0, finite integer_tolerance > 0, and finite gap tolerances >= 0",
            ));
        }
        if !problem.is_integer.iter().any(|&b| b) {
            // No integer variables at all - this is just an LP. Solve
            // it directly rather than paying for a branch-and-bound
            // tree with nothing to branch on.
            let solution = self.relaxation_solver.solve(problem)?;
            let bound = (solution.status == SolveStatus::Optimal).then_some(solution.objective_value);
            return Ok(report(solution, 1, bound, SearchStatistics::default(),
                BranchAndBoundTermination::ContinuousRelaxation));
        }

        let mut frontier = vec![NodeBounds {
            var_bounds: problem.var_bounds.clone(), lower_bound: None, depth: 0,
        }];
        let mut incumbent: Option<Solution> = None;
        let mut nodes_explored = 0usize;
        let mut statistics = SearchStatistics::default();
        let mut search_incomplete = false;
        let mut incomplete_bound: Option<f64> = None;

        while !frontier.is_empty() {
            let node = pop_node(&mut frontier, self.node_selection);
            if nodes_explored == self.max_nodes {
                let best_bound = open_best_bound(&frontier, node.lower_bound);
                let solution = match incumbent {
                    Some(sol) => Solution { status: SolveStatus::IterationLimit, ..sol },
                    None => Solution {
                        variable_values: vec![],
                        objective_value: 0.0,
                        status: SolveStatus::IterationLimit,
                    },
                };
                return Ok(report(solution, nodes_explored, best_bound, statistics,
                    BranchAndBoundTermination::NodeLimit));
            }
            nodes_explored += 1;
            statistics.maximum_depth = statistics.maximum_depth.max(node.depth);

            if node.var_bounds.iter().any(|bound| {
                matches!((bound.lower, bound.upper), (Some(lower), Some(upper)) if lower > upper)
            }) {
                statistics.nodes_pruned_infeasible += 1;
                continue;
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
                    statistics.nodes_pruned_infeasible += 1;
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
                    return Ok(report(relaxed, nodes_explored, None, statistics,
                        BranchAndBoundTermination::Unbounded));
                }
                SolveStatus::IterationLimit => {
                    // The relaxation itself didn't converge - can't
                    // trust its bound for pruning. Skip this node
                    // rather than either wrongly pruning or wrongly
                    // accepting it as a bound.
                    search_incomplete = true;
                    incomplete_bound = match (incomplete_bound, node.lower_bound) {
                        (Some(current), Some(candidate)) => Some(current.min(candidate)),
                        (None, candidate) => candidate,
                        (current, None) => current,
                    };
                    continue;
                }
                SolveStatus::Optimal => {}
            }

            // Bounding: prune if this node cannot possibly beat the
            // incumbent (minimize sense - Problem/Solution are always
            // in minimize form, per Presolve's convention).
            if let Some(ref best) = incumbent {
                if relaxed.objective_value >= best.objective_value - self.integer_tolerance {
                    statistics.nodes_pruned_by_bound += 1;
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
                    let mut upper = node.tightened(var_index, None, Some(floor_value));
                    let mut lower = node.tightened(var_index, Some(ceil_value), None);
                    upper.lower_bound = Some(relaxed.objective_value);
                    lower.lower_bound = Some(relaxed.objective_value);
                    frontier.push(upper);
                    frontier.push(lower);
                }
            }

            if let Some(ref best) = incumbent {
                let best_bound = open_best_bound(&frontier, incomplete_bound);
                if gap_satisfied(best.objective_value, best_bound,
                    self.absolute_gap_tolerance, self.relative_gap_tolerance)
                {
                    return Ok(report(best.clone(), nodes_explored, best_bound, statistics,
                        BranchAndBoundTermination::GapSatisfied));
                }
            }
        }

        match incumbent {
            Some(sol) if search_incomplete => {
                let solution = Solution { status: SolveStatus::IterationLimit, ..sol };
                Ok(report(solution, nodes_explored, incomplete_bound, statistics,
                    BranchAndBoundTermination::RelaxationLimit))
            }
            Some(sol) => {
                let bound = Some(sol.objective_value);
                Ok(report(sol, nodes_explored, bound, statistics,
                    BranchAndBoundTermination::Exhausted))
            }
            None if search_incomplete => Ok(report(Solution {
                variable_values: vec![], objective_value: 0.0,
                status: SolveStatus::IterationLimit,
            }, nodes_explored, None, statistics, BranchAndBoundTermination::RelaxationLimit)),
            None => Ok(report(Solution {
                variable_values: vec![], objective_value: 0.0,
                status: SolveStatus::Infeasible,
            }, nodes_explored, None, statistics, BranchAndBoundTermination::Exhausted)),
        }
    }
}

#[derive(Default, Clone, Copy)]
struct SearchStatistics {
    nodes_pruned_infeasible: usize,
    nodes_pruned_by_bound: usize,
    maximum_depth: usize,
}

fn pop_node(frontier: &mut Vec<NodeBounds>, selection: NodeSelection) -> NodeBounds {
    let index = match selection {
        NodeSelection::DepthFirst => frontier.len() - 1,
        NodeSelection::BestBound => frontier.iter().enumerate().min_by(|(_, a), (_, b)| {
            a.lower_bound.unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&b.lower_bound.unwrap_or(f64::NEG_INFINITY))
                .then_with(|| b.depth.cmp(&a.depth))
        }).map(|(index, _)| index).expect("non-empty frontier"),
    };
    frontier.swap_remove(index)
}

fn gap_satisfied(
    incumbent: f64,
    best_bound: Option<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    let Some(bound) = best_bound else { return false };
    let gap = (incumbent - bound).max(0.0);
    gap <= absolute_tolerance || gap / incumbent.abs().max(1.0) <= relative_tolerance
}

fn open_best_bound(stack: &[NodeBounds], current: Option<f64>) -> Option<f64> {
    stack.iter().filter_map(|node| node.lower_bound).chain(current).reduce(f64::min)
}

fn report(
    solution: Solution,
    nodes_explored: usize,
    best_bound: Option<f64>,
    statistics: SearchStatistics,
    termination: BranchAndBoundTermination,
) -> BranchAndBoundReport {
    let has_incumbent = !solution.variable_values.is_empty();
    let absolute_gap = if has_incumbent {
        best_bound.map(|bound| (solution.objective_value - bound).max(0.0))
    } else { None };
    let relative_gap = absolute_gap.map(|gap| gap / solution.objective_value.abs().max(1.0));
    BranchAndBoundReport {
        solution, nodes_explored, best_bound, absolute_gap, relative_gap,
        nodes_pruned_infeasible: statistics.nodes_pruned_infeasible,
        nodes_pruned_by_bound: statistics.nodes_pruned_by_bound,
        maximum_depth: statistics.maximum_depth,
        termination,
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
        assert!(matches!(result, Err(OptimizeError::InvalidProblem(_))));
    }

    #[test]
    fn node_limit_reports_incumbent_bound_and_gap() {
        let problem = dense_problem(
            vec![-5.0, -4.0],
            vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            vec![
                Bound { lower: None, upper: Some(24.0) },
                Bound { lower: None, upper: Some(6.0) },
            ],
            vec![
                Bound { lower: Some(0.0), upper: None },
                Bound { lower: Some(0.0), upper: None },
            ],
            vec![true, true],
        );
        let solver = BranchAndBoundSolver {
            max_nodes: 2,
            integer_tolerance: 1e-6,
            absolute_gap_tolerance: 0.0,
            relative_gap_tolerance: 0.0,
            node_selection: NodeSelection::DepthFirst,
            relaxation_solver: Box::new(RevisedSimplexSolver::default()),
        };
        let report = solver.solve_with_report(&problem).unwrap();
        assert_eq!(report.solution.status, SolveStatus::IterationLimit);
        assert_eq!(report.nodes_explored, 2);
        assert!(!report.solution.variable_values.is_empty());
        assert_eq!(report.best_bound, Some(-21.0));
        assert_eq!(report.absolute_gap, Some(3.0));
        assert!((report.relative_gap.unwrap() - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn best_bound_search_can_stop_at_a_configured_gap() {
        let problem = dense_problem(
            vec![-5.0, -4.0],
            vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            vec![
                Bound { lower: None, upper: Some(24.0) },
                Bound { lower: None, upper: Some(6.0) },
            ],
            vec![
                Bound { lower: Some(0.0), upper: None },
                Bound { lower: Some(0.0), upper: None },
            ],
            vec![true, true],
        );
        let solver = BranchAndBoundSolver {
            absolute_gap_tolerance: 3.0,
            ..BranchAndBoundSolver::default()
        };
        let report = solver.solve_with_report(&problem).unwrap();
        assert_eq!(report.solution.status, SolveStatus::Optimal);
        assert_eq!(report.termination, BranchAndBoundTermination::GapSatisfied);
        assert!(report.absolute_gap.unwrap() <= 3.0);
        assert!(report.nodes_explored < 5);
    }

    #[test]
    fn report_accounts_for_pruning_and_depth() {
        let problem = dense_problem(
            vec![1.0], vec![vec![2.0]], vec![Bound::fixed(1.0)],
            vec![Bound { lower: Some(0.0), upper: Some(10.0) }], vec![true],
        );
        let report = BranchAndBoundSolver::default().solve_with_report(&problem).unwrap();
        assert_eq!(report.termination, BranchAndBoundTermination::Exhausted);
        assert!(report.nodes_pruned_infeasible >= 2);
        assert_eq!(report.maximum_depth, 1);
    }
}
