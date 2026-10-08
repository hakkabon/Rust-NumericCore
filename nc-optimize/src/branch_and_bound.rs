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
//! - **Learned pseudo-cost branching by default.** Objective degradation from
//!   solved children is accumulated per variable and direction. Variables
//!   without reliable observations use deterministic most-fractional fallback;
//!   most-fractional remains directly selectable for comparison.
//! - **Singleton-row bound propagation** tightens variable domains before each
//!   relaxation and rounds implied integer bounds inward. Contradictions prune
//!   a node without spending an LP solve.
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

/// Policy used to select the integer variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategy {
    MostFractional,
    /// Learn objective degradation per unit branch displacement from solved
    /// children. Variables without observations fall back deterministically to
    /// most-fractional selection.
    PseudoCost,
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
    pub branching_strategy: BranchingStrategy,
    /// Tighten variable bounds implied by singleton constraint rows before
    /// solving each node relaxation.
    pub bound_propagation: bool,
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
    pub relaxations_solved: usize,
    pub bounds_tightened: usize,
    pub incumbents_found: usize,
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
            branching_strategy: BranchingStrategy::PseudoCost,
            bound_propagation: true,
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
    branch_origin: Option<BranchOrigin>,
}

#[derive(Clone, Copy)]
struct BranchOrigin {
    variable: usize,
    direction: BranchDirection,
    parent_objective: f64,
    distance: f64,
}

#[derive(Clone, Copy)]
enum BranchDirection { Down, Up }

impl NodeBounds {
    fn tightened(&self, var_index: usize, lower: Option<f64>, upper: Option<f64>) -> Self {
        let mut var_bounds = self.var_bounds.clone();
        let current = var_bounds[var_index];
        var_bounds[var_index] = Bound {
            lower: tighter_lower(current.lower, lower),
            upper: tighter_upper(current.upper, upper),
        };
        Self {
            var_bounds, lower_bound: self.lower_bound, depth: self.depth + 1,
            branch_origin: None,
        }
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
        self.solve_with_report_warm(problem, None)
    }

    /// Solve with an optional integer-feasible incumbent. The incumbent is
    /// independently validated before it is allowed to prune any node.
    pub fn solve_with_report_warm(
        &self,
        problem: &Problem,
        initial_incumbent: Option<&[f64]>,
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
            return Ok(report(solution, 1, bound, SearchStatistics {
                relaxations_solved: 1, ..Default::default()
            },
                BranchAndBoundTermination::ContinuousRelaxation));
        }

        let mut frontier = vec![NodeBounds {
            var_bounds: problem.var_bounds.clone(), lower_bound: None, depth: 0,
            branch_origin: None,
        }];
        let mut incumbent = match initial_incumbent {
            Some(values) => Some(validated_incumbent(problem, values, self.integer_tolerance)?),
            None => None,
        };
        let mut nodes_explored = 0usize;
        let mut statistics = SearchStatistics {
            incumbents_found: usize::from(incumbent.is_some()), ..Default::default()
        };
        let mut pseudo_costs = PseudoCosts::new(problem.objective.len());
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

            let mut node_bounds = node.var_bounds.clone();
            if self.bound_propagation {
                match propagate_singleton_bounds(problem, &mut node_bounds, self.integer_tolerance) {
                    Ok(tightened) => statistics.bounds_tightened += tightened,
                    Err(()) => {
                        statistics.nodes_pruned_infeasible += 1;
                        continue;
                    }
                }
            }
            if node_bounds.iter().any(|bound| {
                matches!((bound.lower, bound.upper), (Some(lower), Some(upper)) if lower > upper)
            }) {
                statistics.nodes_pruned_infeasible += 1;
                continue;
            }

            let node_problem = Problem {
                objective: problem.objective.clone(),
                constraints: problem.constraints.clone(),
                row_bounds: problem.row_bounds.clone(),
                var_bounds: node_bounds,
                is_integer: problem.is_integer.clone(),
            };

            let relaxed = self.relaxation_solver.solve(&node_problem)?;
            statistics.relaxations_solved += 1;

            if let Some(origin) = node.branch_origin {
                if relaxed.status == SolveStatus::Optimal {
                    pseudo_costs.observe(origin, relaxed.objective_value);
                }
            }

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

            match branching_variable(
                &relaxed, &problem.is_integer, self.integer_tolerance,
                self.branching_strategy, &pseudo_costs,
            ) {
                None => {
                    // Every integer-restricted variable already has an
                    // integer value - this relaxation solution is
                    // integer-feasible, and (by the bounding check just
                    // above) strictly better than any prior incumbent.
                    incumbent = Some(relaxed);
                    statistics.incumbents_found += 1;
                }
                Some((var_index, value)) => {
                    let floor_value = value.floor();
                    let ceil_value = value.ceil();
                    let mut upper = node.tightened(var_index, None, Some(floor_value));
                    let mut lower = node.tightened(var_index, Some(ceil_value), None);
                    upper.lower_bound = Some(relaxed.objective_value);
                    lower.lower_bound = Some(relaxed.objective_value);
                    upper.branch_origin = Some(BranchOrigin {
                        variable: var_index, direction: BranchDirection::Down,
                        parent_objective: relaxed.objective_value,
                        distance: value - floor_value,
                    });
                    lower.branch_origin = Some(BranchOrigin {
                        variable: var_index, direction: BranchDirection::Up,
                        parent_objective: relaxed.objective_value,
                        distance: ceil_value - value,
                    });
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

fn validated_incumbent(
    problem: &Problem,
    values: &[f64],
    tolerance: f64,
) -> Result<Solution, OptimizeError> {
    if values.len() != problem.objective.len() || !values.iter().all(|v| v.is_finite()) {
        return Err(OptimizeError::InvalidConfiguration(
            "MILP warm start must contain one finite value per variable",
        ));
    }
    let objective_value = problem.objective.iter().zip(values).map(|(c, x)| c * x).sum();
    let solution = Solution {
        variable_values: values.to_vec(), objective_value, status: SolveStatus::Optimal,
    };
    if !solution.diagnostics(problem).is_verified(tolerance) {
        return Err(OptimizeError::InvalidConfiguration(
            "MILP warm start must be feasible and integer within integer_tolerance",
        ));
    }
    Ok(solution)
}

#[derive(Default, Clone, Copy)]
struct SearchStatistics {
    nodes_pruned_infeasible: usize,
    nodes_pruned_by_bound: usize,
    maximum_depth: usize,
    relaxations_solved: usize,
    bounds_tightened: usize,
    incumbents_found: usize,
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
        relaxations_solved: statistics.relaxations_solved,
        bounds_tightened: statistics.bounds_tightened,
        incumbents_found: statistics.incumbents_found,
        termination,
    }
}

struct PseudoCosts {
    down_sum: Vec<f64>, down_count: Vec<usize>,
    up_sum: Vec<f64>, up_count: Vec<usize>,
}

impl PseudoCosts {
    fn new(variables: usize) -> Self {
        Self {
            down_sum: vec![0.0; variables], down_count: vec![0; variables],
            up_sum: vec![0.0; variables], up_count: vec![0; variables],
        }
    }

    fn observe(&mut self, origin: BranchOrigin, objective: f64) {
        if origin.distance <= 0.0 { return; }
        let cost = ((objective - origin.parent_objective) / origin.distance).max(0.0);
        match origin.direction {
            BranchDirection::Down => {
                self.down_sum[origin.variable] += cost;
                self.down_count[origin.variable] += 1;
            }
            BranchDirection::Up => {
                self.up_sum[origin.variable] += cost;
                self.up_count[origin.variable] += 1;
            }
        }
    }

    fn score(&self, variable: usize, fraction: f64) -> Option<f64> {
        let down_count = self.down_count[variable];
        let up_count = self.up_count[variable];
        if down_count == 0 || up_count == 0 { return None; }
        let down = self.down_sum[variable] / down_count as f64 * fraction;
        let up = self.up_sum[variable] / up_count as f64 * (1.0 - fraction);
        Some(down.min(up) + 0.1 * down.max(up))
    }
}

fn branching_variable(
    solution: &Solution,
    is_integer: &[bool],
    tolerance: f64,
    strategy: BranchingStrategy,
    pseudo_costs: &PseudoCosts,
) -> Option<(usize, f64)> {
    if strategy == BranchingStrategy::MostFractional {
        return most_fractional_integer_variable(solution, is_integer, tolerance);
    }
    let mut best: Option<(usize, f64, f64)> = None;
    for (variable, &integer) in is_integer.iter().enumerate() {
        if !integer { continue; }
        let value = solution.variable_values[variable];
        let fraction = value - value.floor();
        let distance = fraction.min(1.0 - fraction);
        if distance <= tolerance { continue; }
        let Some(score) = pseudo_costs.score(variable, fraction) else { continue };
        if best.map_or(true, |(_, _, current)| score > current) {
            best = Some((variable, value, score));
        }
    }
    best.map(|(variable, value, _)| (variable, value))
        .or_else(|| most_fractional_integer_variable(solution, is_integer, tolerance))
}

/// Tighten bounds implied by rows containing exactly one nonzero coefficient.
/// Repeats are unnecessary because singleton rows do not depend on other
/// variables. Integer variables are rounded inward to their lattice.
fn propagate_singleton_bounds(
    problem: &Problem,
    bounds: &mut [Bound],
    tolerance: f64,
) -> Result<usize, ()> {
    let mut entries_by_row = vec![Vec::<(usize, f64)>::new(); problem.constraints.rows()];
    for (row, column, value) in problem.constraints.iter_entries() {
        entries_by_row[row].push((column, value));
    }
    let mut tightened = 0;
    for (row, entries) in entries_by_row.iter().enumerate() {
        if entries.len() != 1 { continue; }
        let (column, coefficient) = entries[0];
        let row_bound = problem.row_bounds[row];
        let mut implied = if coefficient > 0.0 {
            Bound {
                lower: row_bound.lower.map(|value| value / coefficient),
                upper: row_bound.upper.map(|value| value / coefficient),
            }
        } else {
            Bound {
                lower: row_bound.upper.map(|value| value / coefficient),
                upper: row_bound.lower.map(|value| value / coefficient),
            }
        };
        if problem.is_integer[column] {
            implied.lower = implied.lower.map(|value| (value - tolerance).ceil());
            implied.upper = implied.upper.map(|value| (value + tolerance).floor());
        }
        let old = bounds[column];
        let new = Bound {
            lower: tighter_lower(old.lower, implied.lower),
            upper: tighter_upper(old.upper, implied.upper),
        };
        if matches!((new.lower, new.upper), (Some(lower), Some(upper)) if lower > upper) {
            return Err(());
        }
        if new != old { tightened += 1; bounds[column] = new; }
    }
    Ok(tightened)
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
            branching_strategy: BranchingStrategy::MostFractional,
            bound_propagation: true,
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
        assert_eq!(report.nodes_pruned_infeasible, 1);
        assert_eq!(report.maximum_depth, 0);
        assert_eq!(report.relaxations_solved, 0);
    }

    #[test]
    fn feasible_warm_incumbent_is_used_at_a_node_limit() {
        let problem = dense_problem(
            vec![-5.0, -4.0], vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            vec![Bound { lower: None, upper: Some(24.0) }, Bound { lower: None, upper: Some(6.0) }],
            vec![Bound { lower: Some(0.0), upper: None }, Bound { lower: Some(0.0), upper: None }],
            vec![true, true],
        );
        let solver = BranchAndBoundSolver { max_nodes: 1, ..Default::default() };
        let report = solver.solve_with_report_warm(&problem, Some(&[4.0, 0.0])).unwrap();
        assert_eq!(report.solution.status, SolveStatus::IterationLimit);
        assert_eq!(report.solution.variable_values, vec![4.0, 0.0]);
        assert_eq!(report.solution.objective_value, -20.0);
    }

    #[test]
    fn infeasible_warm_incumbent_is_rejected() {
        let problem = dense_problem(
            vec![-1.0], vec![vec![1.0]], vec![Bound { lower: None, upper: Some(1.0) }],
            vec![Bound { lower: Some(0.0), upper: None }], vec![true],
        );
        let result = BranchAndBoundSolver::default().solve_with_report_warm(&problem, Some(&[2.0]));
        assert!(matches!(result, Err(OptimizeError::InvalidConfiguration(_))));
    }

    #[test]
    fn singleton_propagation_tightens_an_integer_domain_before_relaxation() {
        let problem = dense_problem(
            vec![1.0], vec![vec![2.0]], vec![Bound { lower: Some(4.0), upper: None }],
            vec![Bound { lower: Some(0.0), upper: None }], vec![true],
        );
        let report = BranchAndBoundSolver::default().solve_with_report(&problem).unwrap();
        assert_eq!(report.solution.status, SolveStatus::Optimal);
        assert_eq!(report.solution.variable_values, vec![2.0]);
        assert_eq!(report.bounds_tightened, 1);
        assert_eq!(report.relaxations_solved, 1);
    }

    #[test]
    fn pseudo_costs_learn_directional_balanced_gain() {
        let mut costs = PseudoCosts::new(1);
        costs.observe(BranchOrigin {
            variable: 0, direction: BranchDirection::Down,
            parent_objective: 10.0, distance: 0.25,
        }, 11.0);
        costs.observe(BranchOrigin {
            variable: 0, direction: BranchDirection::Up,
            parent_objective: 10.0, distance: 0.75,
        }, 13.0);
        let score = costs.score(0, 0.25).unwrap();
        assert!((score - 1.3).abs() < 1e-12);
    }
}
