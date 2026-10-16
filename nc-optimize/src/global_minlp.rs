//! Certified joint integer/spatial branch-and-bound for bounded factorable MINLPs.

use crate::spatial_branch_and_bound::compile_relaxation;
use crate::{
    minimize_model_lbfgsb, minimize_sqp, propagate_nonlinear_bounds, Bound,
    ConstrainedNonlinearProblem, LbfgsOptions, NonlinearConstraint, NonlinearModel,
    NonlinearPropagationOptions, NonlinearTermination, OptimizeError, RevisedSimplexSolver,
    SolveStatus, Solver, SpatialBranchAndBoundProblem, SqpOptions, SqpTermination,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalMinlpTermination {
    GlobalGap,
    SearchExhausted,
    SpatialResolution,
    NodeLimit,
    Infeasible,
    RelaxationFailure,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct GlobalMinlpOptions {
    pub max_nodes: usize,
    pub minimum_width: f64,
    pub integer_tolerance: f64,
    pub feasibility_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub simplex_iterations: usize,
    pub propagation_passes: usize,
    pub propagation_tolerance: f64,
    pub enable_rounding_heuristic: bool,
    pub enable_integer_envelope_cuts: bool,
    pub maximum_integer_envelope_cuts: usize,
    pub local_options: LbfgsOptions,
    pub sqp_options: SqpOptions,
}
impl Default for GlobalMinlpOptions {
    fn default() -> Self {
        Self {
            max_nodes: 20_000,
            minimum_width: 1e-4,
            integer_tolerance: 1e-7,
            feasibility_tolerance: 1e-7,
            absolute_gap_tolerance: 1e-6,
            relative_gap_tolerance: 1e-6,
            simplex_iterations: 10_000,
            propagation_passes: 8,
            propagation_tolerance: 1e-10,
            enable_rounding_heuristic: true,
            enable_integer_envelope_cuts: true,
            maximum_integer_envelope_cuts: 256,
            local_options: LbfgsOptions::default(),
            sqp_options: SqpOptions::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalMinlpProblem {
    pub model: NonlinearModel,
    pub constraints: Vec<NonlinearConstraint>,
    pub is_integer: Vec<bool>,
}
impl GlobalMinlpProblem {
    pub fn validate(&self) -> Result<(), OptimizeError> {
        if self.is_integer.len() != self.model.parameter_count || !self.is_integer.contains(&true) {
            return Err(invalid(
                "global MINLP requires one integrality flag per parameter",
            ));
        }
        SpatialBranchAndBoundProblem {
            model: self.model.clone(),
            constraints: self.constraints.clone(),
        }
        .validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GlobalMinlpResult {
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
    pub integer_branches: usize,
    pub spatial_branches: usize,
    pub local_solves: usize,
    pub heuristic_attempts: usize,
    pub heuristic_successes: usize,
    pub incumbents_found: usize,
    pub maximum_depth: usize,
    pub integer_envelope_cuts_generated: usize,
    pub global_optimality_certified: bool,
    pub termination: GlobalMinlpTermination,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalMinlpIteration {
    pub nodes_explored: usize,
    pub depth: usize,
    pub relaxation_bound: f64,
    pub incumbent_objective: Option<f64>,
    pub open_global_bound: f64,
}

#[derive(Clone)]
struct Node {
    bounds: Vec<Bound>,
    inherited: f64,
    depth: usize,
}
#[derive(Clone)]
struct Incumbent {
    point: Vec<f64>,
    objective: f64,
}

pub fn minimize_global_minlp(
    problem: &GlobalMinlpProblem,
    options: GlobalMinlpOptions,
) -> Result<GlobalMinlpResult, OptimizeError> {
    minimize_global_minlp_with_observer(problem, options, |_| true)
}

pub fn minimize_global_minlp_with_observer<O>(
    problem: &GlobalMinlpProblem,
    options: GlobalMinlpOptions,
    mut observer: O,
) -> Result<GlobalMinlpResult, OptimizeError>
where
    O: FnMut(GlobalMinlpIteration) -> bool,
{
    problem.validate()?;
    validate_options(&options)?;
    let spatial = SpatialBranchAndBoundProblem {
        model: problem.model.clone(),
        constraints: problem.constraints.clone(),
    };
    let mut nodes = vec![Node {
        bounds: problem.model.bounds.clone(),
        inherited: f64::NEG_INFINITY,
        depth: 0,
    }];
    let mut incumbent: Option<Incumbent> = None;
    let mut unresolved = f64::INFINITY;
    let (mut explored, mut solved, mut prune_bound, mut prune_infeasible) = (0, 0, 0, 0);
    let (mut prune_propagation, mut passes, mut tightened) = (0, 0, 0);
    let (mut integer_branches, mut spatial_branches, mut local_solves) = (0, 0, 0);
    let (mut heuristic_attempts, mut heuristic_successes, mut incumbents, mut depth) = (0, 0, 0, 0);
    let mut integer_envelope_cuts = 0;
    let (mut failures, mut cancelled, mut gap_reached) = (0, false, false);
    while !nodes.is_empty() && explored < options.max_nodes {
        let index = nodes
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.inherited.total_cmp(&b.1.inherited))
            .unwrap()
            .0;
        let mut node = nodes.swap_remove(index);
        explored += 1;
        depth = depth.max(node.depth);
        if !normalize_integer_bounds(
            &mut node.bounds,
            &problem.is_integer,
            options.integer_tolerance,
        ) {
            prune_infeasible += 1;
            continue;
        }
        let propagated = propagate_nonlinear_bounds(
            &problem.model,
            &problem.constraints,
            &node.bounds,
            incumbent
                .as_ref()
                .map(|x| x.objective - options.absolute_gap_tolerance),
            NonlinearPropagationOptions {
                max_passes: options.propagation_passes,
                tolerance: options.propagation_tolerance,
            },
        )?;
        passes += propagated.passes;
        tightened += propagated.bounds_tightened;
        if propagated.infeasible {
            prune_infeasible += 1;
            prune_propagation += 1;
            continue;
        }
        node.bounds = propagated.bounds;
        if !normalize_integer_bounds(
            &mut node.bounds,
            &problem.is_integer,
            options.integer_tolerance,
        ) {
            prune_infeasible += 1;
            prune_propagation += 1;
            continue;
        }
        let relaxation = compile_relaxation(&spatial, &node.bounds)?;
        let (relaxation, generated) = if options.enable_integer_envelope_cuts {
            strengthen_integer_square_envelopes(
                problem,
                &node.bounds,
                relaxation,
                options.maximum_integer_envelope_cuts,
            )?
        } else {
            (relaxation, 0)
        };
        integer_envelope_cuts += generated;
        let solution = RevisedSimplexSolver {
            max_iterations: options.simplex_iterations,
            tolerance: options.feasibility_tolerance.min(1e-9).max(1e-12),
        }
        .solve(&relaxation)?;
        match solution.status {
            SolveStatus::Infeasible => {
                prune_infeasible += 1;
                continue;
            }
            SolveStatus::Optimal => {}
            _ => {
                failures += 1;
                unresolved = unresolved.min(node.inherited);
                continue;
            }
        }
        solved += 1;
        let lower = solution.objective_value;
        if incumbent
            .as_ref()
            .is_some_and(|x| lower >= x.objective - options.absolute_gap_tolerance)
        {
            prune_bound += 1;
            continue;
        }
        let point = &solution.variable_values[..problem.model.parameter_count];
        let fractional = fractional_integer(problem, point, options.integer_tolerance);
        if options.enable_rounding_heuristic || fractional.is_none() {
            heuristic_attempts += 1;
            local_solves += 1;
            if let Some(candidate) = rounded_candidate(problem, &node.bounds, point, &options)? {
                if incumbent
                    .as_ref()
                    .is_none_or(|x| candidate.objective < x.objective)
                {
                    incumbent = Some(candidate);
                    incumbents += 1;
                    heuristic_successes += 1;
                }
            }
        }
        if incumbent
            .as_ref()
            .is_some_and(|x| lower >= x.objective - options.absolute_gap_tolerance)
        {
            prune_bound += 1;
            continue;
        }
        if let Some(i) = fractional {
            integer_branches += 1;
            let value = point[i];
            let mut left = node.bounds.clone();
            let mut right = node.bounds.clone();
            left[i].upper = Some(left[i].upper.unwrap().min(value.floor()));
            right[i].lower = Some(right[i].lower.unwrap().max(value.ceil()));
            push_if_valid(&mut nodes, right, lower, node.depth + 1);
            push_if_valid(&mut nodes, left, lower, node.depth + 1);
        } else if let Some(i) = widest(&node.bounds, &problem.is_integer, options.minimum_width) {
            spatial_branches += 1;
            let lo = node.bounds[i].lower.unwrap();
            let hi = node.bounds[i].upper.unwrap();
            let mut left = node.bounds.clone();
            let mut right = node.bounds.clone();
            if problem.is_integer[i] {
                let cut = (lo + 0.5 * (hi - lo)).floor();
                left[i].upper = Some(cut);
                right[i].lower = Some(cut + 1.0);
            } else {
                let cut = lo + 0.5 * (hi - lo);
                left[i].upper = Some(cut);
                right[i].lower = Some(cut);
            }
            push_if_valid(&mut nodes, right, lower, node.depth + 1);
            push_if_valid(&mut nodes, left, lower, node.depth + 1);
        } else {
            unresolved = unresolved.min(lower);
        }
        let open = global_bound(&nodes, unresolved).min(lower);
        if !observer(GlobalMinlpIteration {
            nodes_explored: explored,
            depth: node.depth,
            relaxation_bound: lower,
            incumbent_objective: incumbent.as_ref().map(|x| x.objective),
            open_global_bound: open,
        }) {
            cancelled = true;
            break;
        }
        if let Some(value) = &incumbent {
            let gap = (value.objective - open).max(0.0);
            if gap <= options.absolute_gap_tolerance
                || gap / value.objective.abs().max(1.0) <= options.relative_gap_tolerance
            {
                gap_reached = true;
                break;
            }
        }
    }
    let lower = global_bound(&nodes, unresolved);
    let lower = lower
        .is_finite()
        .then_some(lower)
        .or_else(|| incumbent.as_ref().map(|x| x.objective));
    let gap = incumbent
        .as_ref()
        .and_then(|x| lower.map(|b| (x.objective - b).max(0.0)));
    let relative = incumbent
        .as_ref()
        .and_then(|x| gap.map(|g| g / x.objective.abs().max(1.0)));
    let termination = if cancelled {
        GlobalMinlpTermination::Cancelled
    } else if failures > 0 && nodes.is_empty() {
        GlobalMinlpTermination::RelaxationFailure
    } else if incumbent.is_none() && nodes.is_empty() && !unresolved.is_finite() {
        GlobalMinlpTermination::Infeasible
    } else if gap_reached {
        GlobalMinlpTermination::GlobalGap
    } else if !nodes.is_empty() {
        GlobalMinlpTermination::NodeLimit
    } else if unresolved.is_finite() {
        GlobalMinlpTermination::SpatialResolution
    } else {
        GlobalMinlpTermination::SearchExhausted
    };
    let certified = failures == 0
        && incumbent.is_some()
        && matches!(
            termination,
            GlobalMinlpTermination::GlobalGap | GlobalMinlpTermination::SearchExhausted
        )
        && gap.is_some_and(|g| {
            g <= options.absolute_gap_tolerance
                || relative.is_some_and(|r| r <= options.relative_gap_tolerance)
        });
    let value = incumbent.unwrap_or(Incumbent {
        point: vec![],
        objective: f64::NAN,
    });
    Ok(GlobalMinlpResult {
        point: value.point,
        objective: value.objective,
        global_lower_bound: lower,
        absolute_gap: gap,
        relative_gap: relative,
        nodes_explored: explored,
        relaxations_solved: solved,
        nodes_pruned_by_bound: prune_bound,
        nodes_pruned_infeasible: prune_infeasible,
        nodes_pruned_by_propagation: prune_propagation,
        propagation_passes: passes,
        bounds_tightened: tightened,
        integer_branches,
        spatial_branches,
        local_solves,
        heuristic_attempts,
        heuristic_successes,
        incumbents_found: incumbents,
        maximum_depth: depth,
        integer_envelope_cuts_generated: integer_envelope_cuts,
        global_optimality_certified: certified,
        termination,
    })
}

fn strengthen_integer_square_envelopes(
    problem: &GlobalMinlpProblem,
    bounds: &[Bound],
    relaxation: crate::Problem,
    maximum_cuts: usize,
) -> Result<(crate::Problem, usize), OptimizeError> {
    if maximum_cuts == 0 {
        return Ok((relaxation, 0));
    }
    let expressions = std::iter::once(problem.model.objective.as_ref().unwrap()).chain(
        problem
            .constraints
            .iter()
            .map(|constraint| &constraint.expression),
    );
    let mut cuts: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut offset = problem.model.parameter_count;
    'expressions: for expression in expressions {
        for (node_index, node) in expression.nodes.iter().enumerate() {
            let crate::NonlinearNode::Powf(operand, exponent) = node else {
                continue;
            };
            if *exponent != 2.0 {
                continue;
            }
            let crate::NonlinearNode::Parameter(parameter) = expression.nodes[*operand] else {
                continue;
            };
            if !problem.is_integer[parameter] {
                continue;
            }
            let mut k = bounds[parameter].lower.unwrap().ceil();
            let last = bounds[parameter].upper.unwrap().floor();
            while k < last {
                if cuts.len() == maximum_cuts {
                    break 'expressions;
                }
                let slope = 2.0 * k + 1.0;
                let intercept = -k * (k + 1.0);
                if !slope.is_finite() || !intercept.is_finite() {
                    break;
                }
                cuts.push((parameter, offset + node_index, slope, intercept));
                k += 1.0;
            }
        }
        offset += expression.nodes.len();
    }
    if cuts.is_empty() {
        return Ok((relaxation, 0));
    }
    let original_rows = relaxation.constraints.rows();
    let mut coo =
        nc_sparse::CooMatrix::new(original_rows + cuts.len(), relaxation.constraints.cols());
    for (row, column, value) in relaxation.constraints.iter_entries() {
        coo.push(row, column, value)?;
    }
    let mut row_bounds = relaxation.row_bounds;
    for (index, &(parameter, square, slope, intercept)) in cuts.iter().enumerate() {
        coo.push(original_rows + index, parameter, -slope)?;
        coo.push(original_rows + index, square, 1.0)?;
        row_bounds.push(Bound {
            lower: Some(intercept),
            upper: None,
        });
    }
    let cut_count = cuts.len();
    Ok((
        crate::Problem {
            objective: relaxation.objective,
            constraints: coo.to_csr()?,
            row_bounds,
            var_bounds: relaxation.var_bounds,
            is_integer: relaxation.is_integer,
        },
        cut_count,
    ))
}

fn rounded_candidate(
    problem: &GlobalMinlpProblem,
    bounds: &[Bound],
    point: &[f64],
    options: &GlobalMinlpOptions,
) -> Result<Option<Incumbent>, OptimizeError> {
    let mut fixed = bounds.to_vec();
    let mut start = point.to_vec();
    for i in 0..start.len() {
        if problem.is_integer[i] {
            let v = start[i].round();
            if v < fixed[i].lower.unwrap() || v > fixed[i].upper.unwrap() {
                return Ok(None);
            }
            fixed[i] = Bound::fixed(v);
            start[i] = v;
        }
    }
    let model = NonlinearModel::objective(
        problem.model.parameter_count,
        fixed,
        problem.model.objective.clone().unwrap(),
    );
    let candidate = if problem.constraints.is_empty() {
        let x = minimize_model_lbfgsb(&model, &start, options.local_options.clone())?;
        matches!(
            x.termination,
            NonlinearTermination::ConvergedGradient
                | NonlinearTermination::ConvergedObjective
                | NonlinearTermination::ConvergedStep
        )
        .then_some((x.point, x.objective))
    } else {
        let p = ConstrainedNonlinearProblem {
            model,
            constraints: problem.constraints.clone(),
        };
        let x = minimize_sqp(&p, &start, options.sqp_options.clone())?;
        (x.termination == SqpTermination::Converged
            && x.maximum_violation <= options.feasibility_tolerance)
            .then_some((x.point, x.objective))
    };
    Ok(candidate.map(|(point, objective)| Incumbent { point, objective }))
}
fn normalize_integer_bounds(bounds: &mut [Bound], flags: &[bool], tol: f64) -> bool {
    for (i, b) in bounds.iter_mut().enumerate() {
        if flags[i] {
            b.lower = Some((b.lower.unwrap() - tol).ceil());
            b.upper = Some((b.upper.unwrap() + tol).floor());
        }
        if b.lower.unwrap() > b.upper.unwrap() {
            return false;
        }
    }
    true
}
fn fractional_integer(p: &GlobalMinlpProblem, x: &[f64], tol: f64) -> Option<usize> {
    p.is_integer
        .iter()
        .zip(x)
        .enumerate()
        .filter_map(|(i, (f, x))| {
            (*f && (*x - x.round()).abs() > tol).then_some((i, (*x - x.round()).abs()))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|x| x.0)
}
fn widest(bounds: &[Bound], flags: &[bool], minimum: f64) -> Option<usize> {
    bounds
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            let w = b.upper.unwrap() - b.lower.unwrap();
            (w > if flags[i] { 0.5 } else { minimum }).then_some((i, w))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|x| x.0)
}
fn push_if_valid(nodes: &mut Vec<Node>, bounds: Vec<Bound>, inherited: f64, depth: usize) {
    if bounds.iter().all(|b| b.lower.unwrap() <= b.upper.unwrap()) {
        nodes.push(Node {
            bounds,
            inherited,
            depth,
        })
    }
}
fn global_bound(nodes: &[Node], unresolved: f64) -> f64 {
    nodes.iter().fold(unresolved, |x, n| x.min(n.inherited))
}
fn validate_options(o: &GlobalMinlpOptions) -> Result<(), OptimizeError> {
    let values = [
        o.minimum_width,
        o.integer_tolerance,
        o.feasibility_tolerance,
        o.absolute_gap_tolerance,
        o.relative_gap_tolerance,
        o.propagation_tolerance,
    ];
    if o.max_nodes == 0
        || o.simplex_iterations == 0
        || o.propagation_passes == 0
        || !values.iter().all(|x| x.is_finite())
        || o.minimum_width <= 0.0
        || o.integer_tolerance < 0.0
        || o.feasibility_tolerance <= 0.0
        || o.absolute_gap_tolerance < 0.0
        || o.relative_gap_tolerance < 0.0
        || o.propagation_tolerance < 0.0
    {
        return Err(invalid("invalid global MINLP options"));
    }
    Ok(())
}
fn invalid(s: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(s.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearExpression, NonlinearNode};
    #[test]
    fn globally_solves_integer_concave_quadratic() {
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Powf(0, 2.0),
                NonlinearNode::Negate(1),
            ],
            2,
        );
        let problem = GlobalMinlpProblem {
            model: NonlinearModel::objective(
                1,
                vec![Bound {
                    lower: Some(-2.0),
                    upper: Some(1.0),
                }],
                expression,
            ),
            constraints: vec![],
            is_integer: vec![true],
        };
        let result = minimize_global_minlp(&problem, GlobalMinlpOptions::default()).unwrap();
        assert_eq!(result.point[0], -2.0);
        assert!((result.objective + 4.0).abs() < 1e-8);
        assert!(result.global_optimality_certified);
    }
    #[test]
    fn branches_on_fractional_integer_relaxation() {
        let sum = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Parameter(1),
                NonlinearNode::Add(0, 1),
            ],
            2,
        );
        let objective = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Parameter(1),
                NonlinearNode::Add(0, 1),
                NonlinearNode::Negate(2),
            ],
            3,
        );
        let problem = GlobalMinlpProblem {
            model: NonlinearModel::objective(
                2,
                vec![
                    Bound {
                        lower: Some(0.0),
                        upper: Some(2.0)
                    };
                    2
                ],
                objective,
            ),
            constraints: vec![NonlinearConstraint {
                expression: sum,
                bound: Bound {
                    lower: None,
                    upper: Some(1.5),
                },
            }],
            is_integer: vec![true, true],
        };
        let result = minimize_global_minlp(&problem, GlobalMinlpOptions::default()).unwrap();
        assert!((result.objective + 1.0).abs() < 1e-8);
        assert!(result.integer_branches > 0);
        assert!(result.global_optimality_certified);
    }

    #[test]
    fn adds_integer_square_envelope_cuts() {
        // x^2-x has continuous optimum -0.25 but integer optimum zero.
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Powf(0, 2.0),
                NonlinearNode::Subtract(1, 0),
            ],
            2,
        );
        let problem = GlobalMinlpProblem {
            model: NonlinearModel::objective(
                1,
                vec![Bound {
                    lower: Some(0.0),
                    upper: Some(3.0),
                }],
                expression,
            ),
            constraints: vec![],
            is_integer: vec![true],
        };
        let result = minimize_global_minlp(&problem, GlobalMinlpOptions::default()).unwrap();
        assert!(result.objective.abs() < 1e-10);
        assert!(result.integer_envelope_cuts_generated >= 3);
        assert!(result.global_optimality_certified);
        let unstrengthened = minimize_global_minlp(
            &problem,
            GlobalMinlpOptions {
                enable_integer_envelope_cuts: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(unstrengthened.integer_envelope_cuts_generated, 0);
        assert!(unstrengthened.objective.abs() < 1e-10);
        assert!(result.nodes_explored < unstrengthened.nodes_explored);
    }
}
