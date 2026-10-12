//! Certified spatial branch-and-bound for bounded factorable NLPs.

use crate::{
    minimize_model_lbfgsb, minimize_sqp, propagate_nonlinear_bounds, relax_expression, Bound,
    ConstrainedNonlinearProblem, LbfgsOptions, NonlinearConstraint, NonlinearModel,
    NonlinearPropagationOptions, NonlinearTermination, OptimizeError, Problem,
    RevisedSimplexSolver, SolveStatus, Solver, SqpOptions, SqpTermination,
};
use nc_sparse::CsrMatrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialTermination {
    GlobalGap,
    SearchExhausted,
    SpatialResolution,
    NodeLimit,
    Infeasible,
    RelaxationFailure,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SpatialBranchAndBoundOptions {
    pub max_nodes: usize,
    pub minimum_width: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub feasibility_tolerance: f64,
    pub simplex_iterations: usize,
    pub propagation_passes: usize,
    pub propagation_tolerance: f64,
    pub local_options: LbfgsOptions,
    pub sqp_options: SqpOptions,
}

impl Default for SpatialBranchAndBoundOptions {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            minimum_width: 1e-4,
            absolute_gap_tolerance: 1e-6,
            relative_gap_tolerance: 1e-6,
            feasibility_tolerance: 1e-7,
            simplex_iterations: 10_000,
            propagation_passes: 8,
            propagation_tolerance: 1e-10,
            local_options: LbfgsOptions::default(),
            sqp_options: SqpOptions::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialBranchAndBoundProblem {
    pub model: NonlinearModel,
    pub constraints: Vec<NonlinearConstraint>,
}

impl SpatialBranchAndBoundProblem {
    pub fn validate(&self) -> Result<(), OptimizeError> {
        self.model.validate()?;
        if self.model.objective.is_none() {
            return Err(invalid(
                "spatial branch-and-bound requires a scalar objective",
            ));
        }
        if !self.model.bounds.iter().all(|bound| {
            matches!((bound.lower, bound.upper), (Some(lower), Some(upper)) if lower.is_finite() && upper.is_finite() && lower <= upper)
        }) {
            return Err(invalid("spatial branch-and-bound requires a finite parameter box"));
        }
        for constraint in &self.constraints {
            constraint.expression.validate(self.model.parameter_count)?;
            if constraint.bound.lower.is_none() && constraint.bound.upper.is_none() {
                return Err(invalid(
                    "each nonlinear constraint requires at least one bound",
                ));
            }
        }
        let _ = compile_relaxation(self, &self.model.bounds)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SpatialBranchAndBoundResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub global_lower_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub nodes_explored: usize,
    pub relaxations_solved: usize,
    pub nodes_pruned_by_bound: usize,
    pub nodes_pruned_infeasible: usize,
    pub nodes_pruned_by_propagation: usize,
    pub propagation_passes: usize,
    pub bounds_tightened: usize,
    pub local_solves: usize,
    pub incumbents_found: usize,
    pub maximum_depth: usize,
    pub global_optimality_certified: bool,
    pub termination: SpatialTermination,
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialIteration {
    pub nodes_explored: usize,
    pub depth: usize,
    pub relaxation_bound: f64,
    pub incumbent_objective: Option<f64>,
    pub open_global_bound: f64,
}

#[derive(Clone)]
struct Node {
    bounds: Vec<Bound>,
    inherited_bound: f64,
    depth: usize,
}

#[derive(Clone)]
struct Incumbent {
    point: Vec<f64>,
    objective: f64,
}

pub fn minimize_spatial(
    problem: &SpatialBranchAndBoundProblem,
    options: SpatialBranchAndBoundOptions,
) -> Result<SpatialBranchAndBoundResult, OptimizeError> {
    minimize_spatial_with_observer(problem, options, |_| true)
}

pub fn minimize_spatial_with_observer<O>(
    problem: &SpatialBranchAndBoundProblem,
    options: SpatialBranchAndBoundOptions,
    mut observer: O,
) -> Result<SpatialBranchAndBoundResult, OptimizeError>
where
    O: FnMut(SpatialIteration) -> bool,
{
    problem.validate()?;
    validate_options(&options)?;
    let mut nodes = vec![Node {
        bounds: problem.model.bounds.clone(),
        inherited_bound: f64::NEG_INFINITY,
        depth: 0,
    }];
    let mut incumbent = None::<Incumbent>;
    let mut unresolved_bound = f64::INFINITY;
    let mut explored = 0;
    let mut solved = 0;
    let mut pruned_bound = 0;
    let mut pruned_infeasible = 0;
    let mut pruned_propagation = 0;
    let mut propagation_passes = 0;
    let mut bounds_tightened = 0;
    let mut local_solves = 0;
    let mut incumbents = 0;
    let mut maximum_depth = 0;
    let mut relaxation_failures = 0;
    let mut cancelled = false;
    let mut gap_reached = false;

    while !nodes.is_empty() && explored < options.max_nodes {
        let node_index = nodes
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.inherited_bound.total_cmp(&b.1.inherited_bound))
            .map(|item| item.0)
            .unwrap();
        let mut node = nodes.swap_remove(node_index);
        explored += 1;
        maximum_depth = maximum_depth.max(node.depth);
        let propagation = propagate_nonlinear_bounds(
            &problem.model,
            &problem.constraints,
            &node.bounds,
            incumbent
                .as_ref()
                .map(|value| value.objective - options.absolute_gap_tolerance),
            NonlinearPropagationOptions {
                max_passes: options.propagation_passes,
                tolerance: options.propagation_tolerance,
            },
        )?;
        propagation_passes += propagation.passes;
        bounds_tightened += propagation.bounds_tightened;
        if propagation.infeasible {
            pruned_infeasible += 1;
            pruned_propagation += 1;
            continue;
        }
        node.bounds = propagation.bounds;
        let relaxation_problem = compile_relaxation(problem, &node.bounds)?;
        let solution = RevisedSimplexSolver {
            max_iterations: options.simplex_iterations,
            tolerance: options.feasibility_tolerance.min(1e-9).max(1e-12),
        }
        .solve(&relaxation_problem)?;
        match solution.status {
            SolveStatus::Infeasible => {
                pruned_infeasible += 1;
                continue;
            }
            SolveStatus::Optimal => {}
            _ => {
                relaxation_failures += 1;
                unresolved_bound = unresolved_bound.min(node.inherited_bound);
                continue;
            }
        }
        solved += 1;
        let lower_bound = solution.objective_value;
        if incumbent
            .as_ref()
            .is_some_and(|value| lower_bound >= value.objective - options.absolute_gap_tolerance)
        {
            pruned_bound += 1;
            continue;
        }
        let relaxation_point = solution.variable_values[..problem.model.parameter_count].to_vec();
        local_solves += 1;
        if let Some(candidate) =
            local_candidate(problem, &node.bounds, &relaxation_point, &options)?
        {
            if incumbent
                .as_ref()
                .is_none_or(|old| candidate.objective < old.objective)
            {
                incumbent = Some(candidate);
                incumbents += 1;
            }
        }
        let Some(branch_index) = widest_parameter(&node.bounds, options.minimum_width) else {
            unresolved_bound = unresolved_bound.min(lower_bound);
            continue;
        };
        let bound = node.bounds[branch_index];
        let lower = bound.lower.unwrap();
        let upper = bound.upper.unwrap();
        let midpoint = lower + 0.5 * (upper - lower);
        let mut left = node.bounds.clone();
        let mut right = node.bounds.clone();
        left[branch_index].upper = Some(midpoint);
        right[branch_index].lower = Some(midpoint);
        nodes.push(Node {
            bounds: right,
            inherited_bound: lower_bound,
            depth: node.depth + 1,
        });
        nodes.push(Node {
            bounds: left,
            inherited_bound: lower_bound,
            depth: node.depth + 1,
        });

        let open_bound = global_bound(&nodes, unresolved_bound).min(lower_bound);
        if !observer(SpatialIteration {
            nodes_explored: explored,
            depth: node.depth,
            relaxation_bound: lower_bound,
            incumbent_objective: incumbent.as_ref().map(|value| value.objective),
            open_global_bound: open_bound,
        }) {
            cancelled = true;
            break;
        }
        if let Some(value) = &incumbent {
            let gap = (value.objective - open_bound).max(0.0);
            let relative = gap / value.objective.abs().max(1.0);
            if gap <= options.absolute_gap_tolerance || relative <= options.relative_gap_tolerance {
                gap_reached = true;
                break;
            }
        }
    }

    let lower = global_bound(&nodes, unresolved_bound);
    let lower = lower
        .is_finite()
        .then_some(lower)
        .or_else(|| incumbent.as_ref().map(|x| x.objective));
    let gap = incumbent
        .as_ref()
        .and_then(|value| lower.map(|bound| (value.objective - bound).max(0.0)));
    let relative = incumbent
        .as_ref()
        .and_then(|value| gap.map(|g| g / value.objective.abs().max(1.0)));
    let termination = if cancelled {
        SpatialTermination::Cancelled
    } else if incumbent.is_none() && relaxation_failures > 0 {
        SpatialTermination::RelaxationFailure
    } else if incumbent.is_none() && nodes.is_empty() && !unresolved_bound.is_finite() {
        SpatialTermination::Infeasible
    } else if gap_reached {
        SpatialTermination::GlobalGap
    } else if !nodes.is_empty() {
        SpatialTermination::NodeLimit
    } else if unresolved_bound.is_finite() {
        SpatialTermination::SpatialResolution
    } else {
        SpatialTermination::SearchExhausted
    };
    let certified = incumbent.is_some()
        && matches!(
            termination,
            SpatialTermination::GlobalGap | SpatialTermination::SearchExhausted
        )
        && gap.is_some_and(|value| {
            value <= options.absolute_gap_tolerance
                || relative.is_some_and(|r| r <= options.relative_gap_tolerance)
        });
    let value = incumbent.unwrap_or(Incumbent {
        point: vec![],
        objective: f64::NAN,
    });
    Ok(SpatialBranchAndBoundResult {
        point: value.point,
        objective: value.objective,
        global_lower_bound: lower,
        absolute_gap: gap,
        relative_gap: relative,
        nodes_explored: explored,
        relaxations_solved: solved,
        nodes_pruned_by_bound: pruned_bound,
        nodes_pruned_infeasible: pruned_infeasible,
        nodes_pruned_by_propagation: pruned_propagation,
        propagation_passes,
        bounds_tightened,
        local_solves,
        incumbents_found: incumbents,
        maximum_depth,
        global_optimality_certified: certified,
        termination,
    })
}

fn compile_relaxation(
    problem: &SpatialBranchAndBoundProblem,
    bounds: &[Bound],
) -> Result<Problem, OptimizeError> {
    let objective_expression = problem.model.objective.as_ref().unwrap();
    let mut expressions = Vec::with_capacity(1 + problem.constraints.len());
    expressions.push(objective_expression);
    expressions.extend(
        problem
            .constraints
            .iter()
            .map(|constraint| &constraint.expression),
    );
    let relaxations = expressions
        .iter()
        .map(|expression| relax_expression(expression, bounds))
        .collect::<Result<Vec<_>, _>>()?;
    if relaxations.iter().any(|relaxation| {
        !relaxation.complete
            || relaxation
                .analysis
                .nodes
                .iter()
                .any(|node| !node.interval.is_finite())
    }) {
        return Err(OptimizeError::NotImplemented(
            "spatial branch-and-bound requires complete finite graph relaxations",
        ));
    }
    let parameter_count = problem.model.parameter_count;
    let total_nodes: usize = expressions
        .iter()
        .map(|expression| expression.nodes.len())
        .sum();
    let variables = parameter_count + total_nodes;
    let mut objective = vec![0.0; variables];
    let mut var_bounds = bounds.to_vec();
    let mut dense_rows: Vec<Vec<f64>> = Vec::new();
    let mut row_bounds: Vec<Bound> = Vec::new();
    let mut offset = parameter_count;
    for (expression_index, (expression, relaxation)) in
        expressions.iter().zip(&relaxations).enumerate()
    {
        for node in &relaxation.analysis.nodes {
            var_bounds.push(Bound {
                lower: Some(node.interval.lower),
                upper: Some(node.interval.upper),
            });
        }
        for (node_index, node) in expression.nodes.iter().enumerate() {
            if let crate::NonlinearNode::Parameter(parameter) = node {
                let mut row = vec![0.0; variables];
                row[offset + node_index] = 1.0;
                row[*parameter] = -1.0;
                dense_rows.push(row);
                row_bounds.push(Bound::fixed(0.0));
            }
        }
        for source in &relaxation.rows {
            let mut row = vec![0.0; variables];
            row[offset..offset + source.coefficients.len()].copy_from_slice(&source.coefficients);
            dense_rows.push(row);
            row_bounds.push(Bound {
                lower: source.lower,
                upper: source.upper,
            });
        }
        let output = offset + expression.output;
        if expression_index == 0 {
            objective[output] = 1.0;
        } else {
            let constraint = &problem.constraints[expression_index - 1];
            let mut row = vec![0.0; variables];
            row[output] = 1.0;
            dense_rows.push(row);
            row_bounds.push(constraint.bound);
        }
        offset += expression.nodes.len();
    }
    let mut row_ptr = Vec::with_capacity(dense_rows.len() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in dense_rows {
        for (column, value) in row.into_iter().enumerate() {
            if value != 0.0 {
                col_idx.push(column);
                values.push(value);
            }
        }
        row_ptr.push(values.len());
    }
    Ok(Problem {
        objective,
        constraints: CsrMatrix::new(row_bounds.len(), variables, row_ptr, col_idx, values)?,
        row_bounds,
        var_bounds,
        is_integer: vec![false; variables],
    })
}

fn local_candidate(
    problem: &SpatialBranchAndBoundProblem,
    bounds: &[Bound],
    start: &[f64],
    options: &SpatialBranchAndBoundOptions,
) -> Result<Option<Incumbent>, OptimizeError> {
    let model = NonlinearModel {
        bounds: bounds.to_vec(),
        ..problem.model.clone()
    };
    let point = if problem.constraints.is_empty() {
        let result = minimize_model_lbfgsb(&model, start, options.local_options)?;
        if !matches!(
            result.termination,
            NonlinearTermination::ConvergedGradient
                | NonlinearTermination::ConvergedObjective
                | NonlinearTermination::ConvergedStep
        ) {
            return Ok(None);
        }
        result.point
    } else {
        let constrained = ConstrainedNonlinearProblem {
            model: model.clone(),
            constraints: problem.constraints.clone(),
        };
        let result = minimize_sqp(&constrained, start, options.sqp_options)?;
        if result.termination != SqpTermination::Converged
            || result.maximum_violation > options.feasibility_tolerance
        {
            return Ok(None);
        }
        result.point
    };
    let objective = model.evaluate_objective(&point)?.value;
    let feasible = problem.constraints.iter().all(|constraint| {
        constraint.expression.evaluate(&point).is_ok_and(|value| {
            constraint
                .bound
                .lower
                .is_none_or(|lower| value.value >= lower - options.feasibility_tolerance)
                && constraint
                    .bound
                    .upper
                    .is_none_or(|upper| value.value <= upper + options.feasibility_tolerance)
        })
    });
    Ok(feasible.then_some(Incumbent { point, objective }))
}

fn widest_parameter(bounds: &[Bound], minimum_width: f64) -> Option<usize> {
    bounds
        .iter()
        .enumerate()
        .filter_map(|(index, bound)| {
            let width = bound.upper? - bound.lower?;
            (width > minimum_width).then_some((index, width))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|item| item.0)
}

fn global_bound(nodes: &[Node], unresolved: f64) -> f64 {
    nodes
        .iter()
        .map(|node| node.inherited_bound)
        .fold(unresolved, f64::min)
}

fn validate_options(options: &SpatialBranchAndBoundOptions) -> Result<(), OptimizeError> {
    if options.max_nodes == 0
        || options.simplex_iterations == 0
        || options.propagation_passes == 0
        || !options.propagation_tolerance.is_finite()
        || options.propagation_tolerance < 0.0
        || !options.minimum_width.is_finite()
        || options.minimum_width <= 0.0
        || !options.absolute_gap_tolerance.is_finite()
        || options.absolute_gap_tolerance < 0.0
        || !options.relative_gap_tolerance.is_finite()
        || options.relative_gap_tolerance < 0.0
        || !options.feasibility_tolerance.is_finite()
        || options.feasibility_tolerance <= 0.0
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid spatial branch-and-bound options",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearExpression, NonlinearNode};

    #[test]
    fn propagation_prunes_before_relaxation() {
        let square = NonlinearExpression::new(
            vec![NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0)], 1);
        let model = NonlinearModel::objective(1,
            vec![Bound { lower: Some(-10.0), upper: Some(10.0) }], square.clone());
        let problem = SpatialBranchAndBoundProblem {
            model,
            constraints: vec![NonlinearConstraint {
                expression: square, bound: Bound { lower: None, upper: Some(-1.0) },
            }],
        };
        let result = minimize_spatial(&problem, SpatialBranchAndBoundOptions::default()).unwrap();
        assert_eq!(result.termination, SpatialTermination::Infeasible);
        assert_eq!(result.relaxations_solved, 0);
        assert_eq!(result.nodes_pruned_by_propagation, 1);
    }

    #[test]
    fn globally_solves_concave_quadratic_on_a_box() {
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Powf(0, 2.0),
                NonlinearNode::Negate(1),
            ],
            2,
        );
        let problem = SpatialBranchAndBoundProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(-2.0),
                    upper: Some(1.0),
                }],
                objective: Some(expression),
                residuals: vec![],
            },
            constraints: vec![],
        };
        let result = minimize_spatial(&problem, SpatialBranchAndBoundOptions::default()).unwrap();
        assert!((result.point[0] + 2.0).abs() < 1e-6, "{result:?}");
        assert!((result.objective + 4.0).abs() < 1e-6, "{result:?}");
        assert!(result.global_optimality_certified, "{result:?}");
    }

    #[test]
    fn rejects_incomplete_relaxation() {
        let expression =
            NonlinearExpression::new(vec![NonlinearNode::Parameter(0), NonlinearNode::Sin(0)], 1);
        let problem = SpatialBranchAndBoundProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(-1.0),
                    upper: Some(1.0),
                }],
                objective: Some(expression),
                residuals: vec![],
            },
            constraints: vec![],
        };
        assert!(problem.validate().is_err());
    }

    #[test]
    fn honors_constraint_graphs_in_the_lp_relaxation() {
        let objective = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Powf(0, 2.0),
                NonlinearNode::Negate(1),
            ],
            2,
        );
        let constraint = NonlinearExpression::new(vec![NonlinearNode::Parameter(0)], 0);
        let problem = SpatialBranchAndBoundProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(-2.0),
                    upper: Some(2.0),
                }],
                objective: Some(objective),
                residuals: vec![],
            },
            constraints: vec![NonlinearConstraint {
                expression: constraint,
                bound: Bound {
                    lower: Some(-1.0),
                    upper: None,
                },
            }],
        };
        let result = minimize_spatial(&problem, SpatialBranchAndBoundOptions::default()).unwrap();
        assert!((result.point[0] - 2.0).abs() < 1e-6, "{result:?}");
        assert!(result.global_optimality_certified, "{result:?}");
    }
}
