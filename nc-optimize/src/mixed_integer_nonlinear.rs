//! Mixed-integer nonlinear search using the shared graph and local NLP relaxations.

use crate::{
    minimize_constrained, minimize_model_lbfgsb, minimize_sqp, Bound, ConstrainedNonlinearProblem,
    ConstrainedOptions, ConstrainedTermination, ConstraintMultiplier, LbfgsOptions,
    NonlinearConstraint, NonlinearModel, NonlinearTermination, OptimizeError, SqpOptions,
    SqpTermination,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearRelaxationStrategy {
    Sqp,
    AugmentedLagrangian,
}

#[derive(Debug, Clone, Copy)]
pub struct MixedIntegerNonlinearOptions {
    pub max_nodes: usize,
    pub integer_tolerance: f64,
    pub feasibility_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub relaxation_strategy: NonlinearRelaxationStrategy,
    pub sqp_options: SqpOptions,
    pub constrained_options: ConstrainedOptions,
    pub unconstrained_options: LbfgsOptions,
}

impl Default for MixedIntegerNonlinearOptions {
    fn default() -> Self {
        Self {
            max_nodes: 1_000,
            integer_tolerance: 1e-7,
            feasibility_tolerance: 1e-7,
            absolute_gap_tolerance: 0.0,
            relative_gap_tolerance: 0.0,
            relaxation_strategy: NonlinearRelaxationStrategy::Sqp,
            sqp_options: SqpOptions::default(),
            constrained_options: ConstrainedOptions::default(),
            unconstrained_options: LbfgsOptions::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MixedIntegerNonlinearProblem {
    pub model: NonlinearModel,
    pub constraints: Vec<NonlinearConstraint>,
    pub is_integer: Vec<bool>,
}

impl MixedIntegerNonlinearProblem {
    pub fn validate(&self) -> Result<(), OptimizeError> {
        self.model.validate()?;
        if self.model.objective.is_none() {
            return Err(invalid("MINLP requires an objective"));
        }
        if self.is_integer.len() != self.model.parameter_count {
            return Err(invalid("is_integer must contain one flag per parameter"));
        }
        if !self.is_integer.iter().any(|value| *value) {
            return Err(invalid("MINLP requires at least one integer parameter"));
        }
        if self.constraints.is_empty() {
            return Ok(());
        }
        ConstrainedNonlinearProblem {
            model: self.model.clone(),
            constraints: self.constraints.clone(),
        }
        .validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedIntegerNonlinearTermination {
    SearchExhausted,
    LocalGapLimit,
    NodeLimit,
    Infeasible,
    RelaxationFailure,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct MixedIntegerNonlinearResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<ConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub nodes_explored: usize,
    pub relaxations_solved: usize,
    pub nodes_pruned_infeasible: usize,
    pub maximum_depth: usize,
    pub incumbents_found: usize,
    pub best_relaxation_objective: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    /// Always false until the expression graph carries a convexity certificate.
    pub global_optimality_certified: bool,
    pub termination: MixedIntegerNonlinearTermination,
}

#[derive(Debug, Clone, Copy)]
pub struct MixedIntegerNonlinearIteration {
    pub nodes_explored: usize,
    pub depth: usize,
    pub relaxation_objective: f64,
    pub incumbent_objective: Option<f64>,
}

#[derive(Clone)]
struct Node {
    bounds: Vec<Bound>,
    start: Vec<f64>,
    depth: usize,
}

struct Relaxation {
    point: Vec<f64>,
    objective: f64,
    values: Vec<f64>,
    multipliers: Vec<ConstraintMultiplier>,
    violation: f64,
    stationarity: f64,
}

pub fn minimize_mixed_integer_nonlinear(
    problem: &MixedIntegerNonlinearProblem,
    initial: &[f64],
    options: MixedIntegerNonlinearOptions,
) -> Result<MixedIntegerNonlinearResult, OptimizeError> {
    minimize_mixed_integer_nonlinear_with_observer(problem, initial, options, |_| true)
}

pub fn minimize_mixed_integer_nonlinear_with_observer<O>(
    problem: &MixedIntegerNonlinearProblem,
    initial: &[f64],
    options: MixedIntegerNonlinearOptions,
    mut observer: O,
) -> Result<MixedIntegerNonlinearResult, OptimizeError>
where
    O: FnMut(MixedIntegerNonlinearIteration) -> bool,
{
    problem.validate()?;
    validate_options(options)?;
    if initial.len() != problem.model.parameter_count || !initial.iter().all(|x| x.is_finite()) {
        return Err(invalid(
            "initial point must contain one finite value per parameter",
        ));
    }
    let root_start = clamp_start(initial, &problem.model.bounds);
    let mut nodes = vec![Node {
        bounds: problem.model.bounds.clone(),
        start: root_start,
        depth: 0,
    }];
    let mut incumbent: Option<Relaxation> = None;
    let mut nodes_explored = 0;
    let mut relaxations_solved = 0;
    let mut pruned = 0;
    let mut maximum_depth = 0;
    let mut incumbents = 0;
    let mut best_relaxation = None::<f64>;
    let mut relaxation_failures = 0;
    let mut local_gap_reached = false;

    while !nodes.is_empty() {
        if nodes_explored >= options.max_nodes {
            break;
        }
        let node = nodes.pop().unwrap();
        nodes_explored += 1;
        maximum_depth = maximum_depth.max(node.depth);
        let Some(relaxation) = solve_relaxation(problem, &node, options)? else {
            relaxation_failures += 1;
            continue;
        };
        relaxations_solved += 1;
        best_relaxation =
            Some(best_relaxation.map_or(relaxation.objective, |x| x.min(relaxation.objective)));
        if !observer(MixedIntegerNonlinearIteration {
            nodes_explored,
            depth: node.depth,
            relaxation_objective: relaxation.objective,
            incumbent_objective: incumbent.as_ref().map(|value| value.objective),
        }) {
            return Ok(finish(
                incumbent,
                nodes_explored,
                relaxations_solved,
                pruned,
                maximum_depth,
                incumbents,
                best_relaxation,
                MixedIntegerNonlinearTermination::Cancelled,
            ));
        }
        if relaxation.violation > options.feasibility_tolerance {
            pruned += 1;
            continue;
        }
        if let Some(index) =
            branching_parameter(problem, &relaxation.point, options.integer_tolerance)
        {
            let value = relaxation.point[index];
            let lower_upper = value.floor();
            let upper_lower = value.ceil();
            let mut lower_bounds = node.bounds.clone();
            let mut upper_bounds = node.bounds.clone();
            let lower_valid = tighten_upper(&mut lower_bounds[index], lower_upper);
            let upper_valid = tighten_lower(&mut upper_bounds[index], upper_lower);
            if upper_valid {
                nodes.push(Node {
                    bounds: upper_bounds,
                    start: clamp_start(
                        &relaxation.point,
                        &node_with_lower(&node.bounds, index, upper_lower),
                    ),
                    depth: node.depth + 1,
                });
            }
            if lower_valid {
                nodes.push(Node {
                    bounds: lower_bounds,
                    start: clamp_start(
                        &relaxation.point,
                        &node_with_upper(&node.bounds, index, lower_upper),
                    ),
                    depth: node.depth + 1,
                });
            }
        } else {
            let Some(candidate) =
                snapped_candidate(problem, relaxation, options.feasibility_tolerance)?
            else {
                pruned += 1;
                continue;
            };
            if incumbent
                .as_ref()
                .is_none_or(|value| candidate.objective < value.objective)
            {
                incumbent = Some(candidate);
                incumbents += 1;
            }
            if let (Some(value), Some(bound)) = (&incumbent, best_relaxation) {
                let gap = (value.objective - bound).max(0.0);
                let relative = gap / value.objective.abs().max(1.0);
                if (options.absolute_gap_tolerance > 0.0 && gap <= options.absolute_gap_tolerance)
                    || (options.relative_gap_tolerance > 0.0
                        && relative <= options.relative_gap_tolerance)
                {
                    local_gap_reached = true;
                    break;
                }
            }
        }
    }
    let termination = if incumbent.is_none() {
        if relaxations_solved == 0 && relaxation_failures > 0 {
            MixedIntegerNonlinearTermination::RelaxationFailure
        } else {
            MixedIntegerNonlinearTermination::Infeasible
        }
    } else if local_gap_reached {
        MixedIntegerNonlinearTermination::LocalGapLimit
    } else if nodes.is_empty() {
        MixedIntegerNonlinearTermination::SearchExhausted
    } else {
        MixedIntegerNonlinearTermination::NodeLimit
    };
    Ok(finish(
        incumbent,
        nodes_explored,
        relaxations_solved,
        pruned,
        maximum_depth,
        incumbents,
        best_relaxation,
        termination,
    ))
}

fn solve_relaxation(
    problem: &MixedIntegerNonlinearProblem,
    node: &Node,
    options: MixedIntegerNonlinearOptions,
) -> Result<Option<Relaxation>, OptimizeError> {
    let model = NonlinearModel {
        bounds: node.bounds.clone(),
        ..problem.model.clone()
    };
    if problem.constraints.is_empty() {
        let value = minimize_model_lbfgsb(&model, &node.start, options.unconstrained_options)?;
        if !matches!(
            value.termination,
            NonlinearTermination::ConvergedGradient
                | NonlinearTermination::ConvergedStep
                | NonlinearTermination::ConvergedObjective
        ) {
            return Ok(None);
        }
        return Ok(Some(Relaxation {
            point: value.point,
            objective: value.objective,
            values: vec![],
            multipliers: vec![],
            violation: 0.0,
            stationarity: value.gradient_norm,
        }));
    }
    let constrained = ConstrainedNonlinearProblem {
        model,
        constraints: problem.constraints.clone(),
    };
    match options.relaxation_strategy {
        NonlinearRelaxationStrategy::Sqp => {
            let value = minimize_sqp(&constrained, &node.start, options.sqp_options)?;
            if !matches!(value.termination, SqpTermination::Converged) {
                return Ok(None);
            }
            Ok(Some(Relaxation {
                point: value.point,
                objective: value.objective,
                values: value.constraint_values,
                multipliers: value.multipliers,
                violation: value.maximum_violation,
                stationarity: value.stationarity_norm,
            }))
        }
        NonlinearRelaxationStrategy::AugmentedLagrangian => {
            let value =
                minimize_constrained(&constrained, &node.start, options.constrained_options)?;
            if !matches!(value.termination, ConstrainedTermination::Converged) {
                return Ok(None);
            }
            Ok(Some(Relaxation {
                point: value.point,
                objective: value.objective,
                values: value.constraint_values,
                multipliers: value.multipliers,
                violation: value.maximum_violation,
                stationarity: value.stationarity_norm,
            }))
        }
    }
}

fn branching_parameter(
    problem: &MixedIntegerNonlinearProblem,
    point: &[f64],
    tolerance: f64,
) -> Option<usize> {
    problem
        .is_integer
        .iter()
        .zip(point)
        .enumerate()
        .filter_map(|(index, (integer, value))| {
            if !integer {
                return None;
            }
            let fractionality = (value - value.round()).abs();
            (fractionality > tolerance).then_some((index, fractionality))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|value| value.0)
}

fn snapped_candidate(
    problem: &MixedIntegerNonlinearProblem,
    mut value: Relaxation,
    tolerance: f64,
) -> Result<Option<Relaxation>, OptimizeError> {
    for (index, integer) in problem.is_integer.iter().enumerate() {
        if *integer {
            value.point[index] = value.point[index].round();
        }
    }
    for (point, bound) in value.point.iter().zip(&problem.model.bounds) {
        if bound.lower.is_some_and(|lower| *point < lower - tolerance)
            || bound.upper.is_some_and(|upper| *point > upper + tolerance)
        {
            return Ok(None);
        }
    }
    let objective = problem.model.evaluate_objective(&value.point)?;
    let evaluated: Vec<_> = problem
        .constraints
        .iter()
        .map(|constraint| constraint.expression.evaluate(&value.point))
        .collect::<Result<_, _>>()?;
    let violation =
        problem
            .constraints
            .iter()
            .zip(&evaluated)
            .fold(0.0_f64, |maximum, (constraint, item)| {
                maximum
                    .max(
                        constraint
                            .bound
                            .lower
                            .map_or(0.0, |lower| (lower - item.value).max(0.0)),
                    )
                    .max(
                        constraint
                            .bound
                            .upper
                            .map_or(0.0, |upper| (item.value - upper).max(0.0)),
                    )
            });
    if violation > tolerance {
        return Ok(None);
    }
    value.objective = objective.value;
    value.values = evaluated.iter().map(|item| item.value).collect();
    value.violation = violation;
    Ok(Some(value))
}

fn finish(
    incumbent: Option<Relaxation>,
    nodes: usize,
    relaxations: usize,
    pruned: usize,
    depth: usize,
    incumbents: usize,
    best_relaxation: Option<f64>,
    termination: MixedIntegerNonlinearTermination,
) -> MixedIntegerNonlinearResult {
    let gap = incumbent
        .as_ref()
        .and_then(|value| best_relaxation.map(|bound| (value.objective - bound).max(0.0)));
    let relative = incumbent
        .as_ref()
        .and_then(|value| gap.map(|g| g / value.objective.abs().max(1.0)));
    let value = incumbent.unwrap_or(Relaxation {
        point: vec![],
        objective: f64::NAN,
        values: vec![],
        multipliers: vec![],
        violation: f64::INFINITY,
        stationarity: f64::INFINITY,
    });
    MixedIntegerNonlinearResult {
        point: value.point,
        objective: value.objective,
        constraint_values: value.values,
        multipliers: value.multipliers,
        maximum_violation: value.violation,
        stationarity_norm: value.stationarity,
        nodes_explored: nodes,
        relaxations_solved: relaxations,
        nodes_pruned_infeasible: pruned,
        maximum_depth: depth,
        incumbents_found: incumbents,
        best_relaxation_objective: best_relaxation,
        absolute_gap: gap,
        relative_gap: relative,
        global_optimality_certified: false,
        termination,
    }
}

fn clamp_start(initial: &[f64], bounds: &[Bound]) -> Vec<f64> {
    initial
        .iter()
        .zip(bounds)
        .map(|(value, bound)| {
            let mut x = *value;
            if let Some(lower) = bound.lower {
                x = x.max(lower);
            }
            if let Some(upper) = bound.upper {
                x = x.min(upper);
            }
            x
        })
        .collect()
}
fn node_with_lower(bounds: &[Bound], index: usize, lower: f64) -> Vec<Bound> {
    let mut result = bounds.to_vec();
    result[index].lower = Some(lower);
    result
}
fn node_with_upper(bounds: &[Bound], index: usize, upper: f64) -> Vec<Bound> {
    let mut result = bounds.to_vec();
    result[index].upper = Some(upper);
    result
}
fn tighten_lower(bound: &mut Bound, value: f64) -> bool {
    bound.lower = Some(bound.lower.map_or(value, |old| old.max(value)));
    bound
        .upper
        .is_none_or(|upper| bound.lower.unwrap() <= upper)
}
fn tighten_upper(bound: &mut Bound, value: f64) -> bool {
    bound.upper = Some(bound.upper.map_or(value, |old| old.min(value)));
    bound
        .lower
        .is_none_or(|lower| lower <= bound.upper.unwrap())
}
fn validate_options(options: MixedIntegerNonlinearOptions) -> Result<(), OptimizeError> {
    if options.max_nodes == 0
        || !options.integer_tolerance.is_finite()
        || options.integer_tolerance <= 0.0
        || !options.feasibility_tolerance.is_finite()
        || options.feasibility_tolerance <= 0.0
        || !options.absolute_gap_tolerance.is_finite()
        || options.absolute_gap_tolerance < 0.0
        || !options.relative_gap_tolerance.is_finite()
        || options.relative_gap_tolerance < 0.0
    {
        return Err(OptimizeError::InvalidConfiguration("invalid MINLP options"));
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

    fn shifted_square(shift: f64) -> NonlinearExpression {
        NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Constant(shift),
                NonlinearNode::Subtract(0, 1),
                NonlinearNode::Powf(2, 2.0),
            ],
            3,
        )
    }

    #[test]
    fn solves_bounded_integer_nonlinear_objective() {
        let problem = MixedIntegerNonlinearProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(0.0),
                    upper: Some(4.0),
                }],
                objective: Some(shifted_square(2.4)),
                residuals: vec![],
            },
            constraints: vec![],
            is_integer: vec![true],
        };
        let result = minimize_mixed_integer_nonlinear(
            &problem,
            &[1.0],
            MixedIntegerNonlinearOptions::default(),
        )
        .unwrap();
        assert_eq!(
            result.termination,
            MixedIntegerNonlinearTermination::SearchExhausted
        );
        assert_eq!(result.point, vec![2.0]);
        assert!((result.objective - 0.16).abs() < 1e-8, "{:?}", result);
        assert!(result.nodes_explored >= 3);
        assert!(!result.global_optimality_certified);
    }

    #[test]
    fn node_limit_returns_incumbent_without_global_claim() {
        let problem = MixedIntegerNonlinearProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(0.0),
                    upper: Some(4.0),
                }],
                objective: Some(shifted_square(2.4)),
                residuals: vec![],
            },
            constraints: vec![],
            is_integer: vec![true],
        };
        let mut options = MixedIntegerNonlinearOptions::default();
        options.max_nodes = 2;
        let result = minimize_mixed_integer_nonlinear(&problem, &[1.0], options).unwrap();
        assert_eq!(
            result.termination,
            MixedIntegerNonlinearTermination::NodeLimit
        );
        assert!(!result.point.is_empty());
        assert!(!result.global_optimality_certified);
    }

    #[test]
    fn constrained_search_supports_specialized_relaxations() {
        let constraint = NonlinearExpression::new(
            vec![NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0)],
            1,
        );
        let problem = MixedIntegerNonlinearProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(0.0),
                    upper: Some(4.0),
                }],
                objective: Some(shifted_square(2.4)),
                residuals: vec![],
            },
            constraints: vec![NonlinearConstraint {
                expression: constraint,
                bound: Bound {
                    lower: Some(1.0),
                    upper: None,
                },
            }],
            is_integer: vec![true],
        };
        for strategy in [
            NonlinearRelaxationStrategy::Sqp,
            NonlinearRelaxationStrategy::AugmentedLagrangian,
        ] {
            let mut options = MixedIntegerNonlinearOptions::default();
            options.relaxation_strategy = strategy;
            let result = minimize_mixed_integer_nonlinear(&problem, &[1.5], options).unwrap();
            assert_eq!(
                result.termination,
                MixedIntegerNonlinearTermination::SearchExhausted
            );
            assert_eq!(result.point, vec![2.0], "{:?}", result);
            assert!(result.maximum_violation <= 1e-7);
        }
    }

    #[test]
    fn local_gap_limit_is_not_a_global_certificate() {
        let problem = MixedIntegerNonlinearProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound {
                    lower: Some(0.0),
                    upper: Some(4.0),
                }],
                objective: Some(shifted_square(2.4)),
                residuals: vec![],
            },
            constraints: vec![],
            is_integer: vec![true],
        };
        let mut options = MixedIntegerNonlinearOptions::default();
        options.absolute_gap_tolerance = 0.2;
        let result = minimize_mixed_integer_nonlinear(&problem, &[1.0], options).unwrap();
        assert_eq!(
            result.termination,
            MixedIntegerNonlinearTermination::LocalGapLimit
        );
        assert!(result.absolute_gap.unwrap() <= 0.2);
        assert!(!result.global_optimality_certified);
    }
}
