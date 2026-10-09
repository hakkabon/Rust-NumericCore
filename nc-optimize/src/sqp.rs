//! Sequential quadratic programming for graph-represented constrained models.

use crate::{
    restore_feasibility, solve_convex_qp, Bound, ConstrainedNonlinearProblem,
    ConstraintMultiplier, DifferentiableValue, FeasibilityRestorationOptions,
    FeasibilityRestorationTermination, OptimizeError, QuadraticOptions, QuadraticProblem,
    QuadraticTermination,
};
use nc_sparse::CsrMatrix;

#[derive(Debug, Clone, Copy)]
pub struct SqpOptions {
    pub max_iterations: usize,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub step_tolerance: f64,
    pub merit_penalty: f64,
    pub penalty_increase: f64,
    pub armijo: f64,
    pub backtracking: f64,
    pub max_line_search_iterations: usize,
    pub hessian_regularization: f64,
    pub qp_options: QuadraticOptions,
    pub restoration: bool,
    pub restoration_options: FeasibilityRestorationOptions,
    pub globalization: SqpGlobalization,
    pub filter_constraint_margin: f64,
    pub filter_objective_margin: f64,
    pub curvature: SqpCurvature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqpGlobalization {
    Merit,
    Filter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqpCurvature { Bfgs, ExactLagrangian }

impl Default for SqpOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            feasibility_tolerance: 1e-7,
            stationarity_tolerance: 1e-6,
            step_tolerance: 1e-10,
            merit_penalty: 10.0,
            penalty_increase: 10.0,
            armijo: 1e-4,
            backtracking: 0.5,
            max_line_search_iterations: 30,
            hessian_regularization: 1e-8,
            qp_options: QuadraticOptions::default(),
            restoration: false,
            restoration_options: FeasibilityRestorationOptions::default(),
            globalization: SqpGlobalization::Merit,
            filter_constraint_margin: 1e-4,
            filter_objective_margin: 1e-4,
            curvature: SqpCurvature::Bfgs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqpTermination {
    Converged,
    IterationLimit,
    StepLimit,
    LineSearchFailed,
    QpFailure,
    RestorationFailed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SqpResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<ConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub final_merit_penalty: f64,
    pub last_step_norm: f64,
    pub termination: SqpTermination,
}

#[derive(Debug, Clone, Copy)]
pub struct SqpIteration {
    pub iteration: usize,
    pub objective: f64,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub step_norm: f64,
    pub step_length: f64,
    pub merit_penalty: f64,
}

pub fn minimize_sqp(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: SqpOptions,
) -> Result<SqpResult, OptimizeError> {
    minimize_sqp_with_observer(problem, initial, options, |_| true)
}

pub fn minimize_sqp_with_observer<O>(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: SqpOptions,
    mut observer: O,
) -> Result<SqpResult, OptimizeError>
where
    O: FnMut(SqpIteration) -> bool,
{
    problem.validate()?;
    validate_options(options)?;
    let n = problem.model.parameter_count;
    if initial.len() != n || !initial.iter().all(|v| v.is_finite()) {
        return Err(invalid("initial point must match the SQP problem"));
    }
    let mut point = project(initial, &problem.model.bounds);
    let mut restoration_evaluations = 0;
    let initial_constraints = problem.evaluate_constraints(&point)?;
    if options.restoration
        && maximum_violation(problem, &initial_constraints) > options.feasibility_tolerance
    {
        let restored = restore_feasibility(problem, &point, options.restoration_options)?;
        restoration_evaluations = restored.evaluations;
        point = restored.point;
        if restored.maximum_violation > options.feasibility_tolerance
            || matches!(restored.termination,
                FeasibilityRestorationTermination::IterationLimit
                | FeasibilityRestorationTermination::Stalled)
        {
            let objective = problem.model.evaluate_objective(&point)?;
            let constraints = problem.evaluate_constraints(&point)?;
            return Ok(result(
                point, objective, constraints,
                vec![ConstraintMultiplier::default(); problem.constraints.len()],
                0, restored.evaluations, 0, 0, options.merit_penalty, 0.0,
                SqpTermination::RestorationFailed, problem,
            ));
        }
    }
    let mut objective = problem.model.evaluate_objective(&point)?;
    let mut constraints = problem.evaluate_constraints(&point)?;
    let mut evaluations = 1 + restoration_evaluations;
    let mut hessian = identity(n);
    let mut multipliers = vec![ConstraintMultiplier::default(); problem.constraints.len()];
    let mut penalty = options.merit_penalty;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut last_step_norm = 0.0;
    let mut filter = vec![(violation_sum(problem, &constraints), objective.value)];

    for iteration in 1..=options.max_iterations {
        let violation = maximum_violation(problem, &constraints);
        let stationarity = stationarity_norm(
            problem,
            &point,
            &objective.gradient,
            &constraints,
            &multipliers,
        );
        if violation <= options.feasibility_tolerance
            && stationarity <= options.stationarity_tolerance
        {
            return Ok(result(
                point,
                objective,
                constraints,
                multipliers,
                iteration - 1,
                evaluations,
                accepted_steps,
                rejected_steps,
                penalty,
                last_step_norm,
                SqpTermination::Converged,
                problem,
            ));
        }

        if options.curvature == SqpCurvature::ExactLagrangian {
            hessian = problem.lagrangian_hessian(&point, &lagrangian_weights(&multipliers))?;
        }
        regularize(&mut hessian, options.hessian_regularization);
        let qp =
            quadratic_subproblem(problem, &point, &objective.gradient, &constraints, &hessian)?;
        let qp_result = solve_convex_qp(&qp, options.qp_options)?;
        if qp_result.termination != QuadraticTermination::Converged {
            return Ok(result(
                point,
                objective,
                constraints,
                multipliers,
                iteration - 1,
                evaluations,
                accepted_steps,
                rejected_steps,
                penalty,
                last_step_norm,
                SqpTermination::QpFailure,
                problem,
            ));
        }
        let step = qp_result.point;
        last_step_norm = inf_norm(&step);
        let candidate_multipliers = multipliers_from_duals(problem, &qp_result.row_dual);
        let multiplier_scale = candidate_multipliers
            .iter()
            .fold(0.0_f64, |maximum, value| {
                maximum
                    .max(value.lower.abs())
                    .max(value.upper.abs())
                    .max(value.equality.abs())
            });
        if penalty <= multiplier_scale {
            penalty = (penalty * options.penalty_increase).max(1.1 * multiplier_scale);
        }
        if last_step_norm <= options.step_tolerance {
            multipliers = candidate_multipliers;
            let termination = if violation <= options.feasibility_tolerance
                && stationarity_norm(
                    problem,
                    &point,
                    &objective.gradient,
                    &constraints,
                    &multipliers,
                ) <= options.stationarity_tolerance
            {
                SqpTermination::Converged
            } else {
                SqpTermination::StepLimit
            };
            return Ok(result(
                point,
                objective,
                constraints,
                multipliers,
                iteration,
                evaluations,
                accepted_steps,
                rejected_steps,
                penalty,
                last_step_norm,
                termination,
                problem,
            ));
        }

        let current_merit = objective.value + penalty * violation_sum(problem, &constraints);
        let predicted = predicted_reduction(
            problem,
            &objective.gradient,
            &constraints,
            &hessian,
            &step,
            penalty,
        )
        .max(1e-12);
        let old_lagrangian = lagrangian_gradient(
            problem,
            &objective.gradient,
            &constraints,
            &candidate_multipliers,
        );
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..options.max_line_search_iterations {
            let trial = project(
                &point
                    .iter()
                    .zip(&step)
                    .map(|(x, d)| x + alpha * d)
                    .collect::<Vec<_>>(),
                &problem.model.bounds,
            );
            let trial_objective = problem.model.evaluate_objective(&trial)?;
            let trial_constraints = problem.evaluate_constraints(&trial)?;
            evaluations += 1;
            let trial_merit =
                trial_objective.value + penalty * violation_sum(problem, &trial_constraints);
            let trial_violation = violation_sum(problem, &trial_constraints);
            let accepted_by_merit = trial_merit
                <= current_merit - options.armijo * alpha * predicted;
            let accepted_by_filter = filter.iter().all(|&(violation, value)| {
                trial_violation <= (1.0 - options.filter_constraint_margin) * violation
                    || trial_objective.value
                        <= value - options.filter_objective_margin * violation
            });
            if match options.globalization {
                SqpGlobalization::Merit => accepted_by_merit,
                SqpGlobalization::Filter
                    if violation_sum(problem, &constraints) <= options.feasibility_tolerance =>
                {
                    accepted_by_merit
                }
                SqpGlobalization::Filter => {
                    accepted_by_filter
                        && (trial_violation < violation_sum(problem, &constraints)
                            || accepted_by_merit)
                }
            } {
                accepted = Some((trial, trial_objective, trial_constraints));
                break;
            }
            rejected_steps += 1;
            alpha *= options.backtracking;
        }
        let Some((new_point, new_objective, new_constraints)) = accepted else {
            return Ok(result(
                point,
                objective,
                constraints,
                multipliers,
                iteration - 1,
                evaluations,
                accepted_steps,
                rejected_steps,
                penalty,
                last_step_norm,
                SqpTermination::LineSearchFailed,
                problem,
            ));
        };
        accepted_steps += 1;
        if options.globalization == SqpGlobalization::Filter {
            let new_pair = (violation_sum(problem, &new_constraints), new_objective.value);
            filter.retain(|&(violation, value)| {
                violation < new_pair.0 || value < new_pair.1
            });
            filter.push(new_pair);
        }
        let actual_step: Vec<f64> = new_point.iter().zip(&point).map(|(a, b)| a - b).collect();
        let new_lagrangian = lagrangian_gradient(
            problem,
            &new_objective.gradient,
            &new_constraints,
            &candidate_multipliers,
        );
        let y: Vec<f64> = new_lagrangian
            .iter()
            .zip(&old_lagrangian)
            .map(|(a, b)| a - b)
            .collect();
        if options.curvature == SqpCurvature::Bfgs {
            bfgs_update(
                &mut hessian,
                &actual_step,
                &y,
                options.hessian_regularization,
            );
        }
        point = new_point;
        objective = new_objective;
        constraints = new_constraints;
        multipliers = candidate_multipliers;
        let new_violation = maximum_violation(problem, &constraints);
        let new_stationarity = stationarity_norm(
            problem,
            &point,
            &objective.gradient,
            &constraints,
            &multipliers,
        );
        if !observer(SqpIteration {
            iteration,
            objective: objective.value,
            maximum_violation: new_violation,
            stationarity_norm: new_stationarity,
            step_norm: inf_norm(&actual_step),
            step_length: alpha,
            merit_penalty: penalty,
        }) {
            return Ok(result(
                point,
                objective,
                constraints,
                multipliers,
                iteration,
                evaluations,
                accepted_steps,
                rejected_steps,
                penalty,
                inf_norm(&actual_step),
                SqpTermination::Cancelled,
                problem,
            ));
        }
    }
    Ok(result(
        point,
        objective,
        constraints,
        multipliers,
        options.max_iterations,
        evaluations,
        accepted_steps,
        rejected_steps,
        penalty,
        last_step_norm,
        SqpTermination::IterationLimit,
        problem,
    ))
}

fn quadratic_subproblem(
    problem: &ConstrainedNonlinearProblem,
    point: &[f64],
    gradient: &[f64],
    values: &[DifferentiableValue],
    hessian: &[Vec<f64>],
) -> Result<QuadraticProblem, OptimizeError> {
    let mut row_pointers = Vec::with_capacity(values.len() + 1);
    let mut columns = Vec::new();
    let mut entries = Vec::new();
    row_pointers.push(0);
    for value in values {
        for (column, &entry) in value.gradient.iter().enumerate() {
            if entry != 0.0 {
                columns.push(column);
                entries.push(entry);
            }
        }
        row_pointers.push(entries.len());
    }
    let constraints = CsrMatrix::new(values.len(), point.len(), row_pointers, columns, entries)
        .map_err(|error| invalid(&format!("invalid SQP Jacobian: {error}")))?;
    let row_bounds = problem
        .constraints
        .iter()
        .zip(values)
        .map(|(constraint, value)| Bound {
            lower: constraint.bound.lower.map(|bound| bound - value.value),
            upper: constraint.bound.upper.map(|bound| bound - value.value),
        })
        .collect();
    let variable_bounds = problem
        .model
        .bounds
        .iter()
        .zip(point)
        .map(|(bound, value)| Bound {
            lower: bound.lower.map(|limit| limit - value),
            upper: bound.upper.map(|limit| limit - value),
        })
        .collect();
    Ok(QuadraticProblem {
        quadratic: hessian.to_vec(),
        linear: gradient.to_vec(),
        constant: 0.0,
        constraints,
        row_bounds,
        variable_bounds,
    })
}

fn multipliers_from_duals(
    problem: &ConstrainedNonlinearProblem,
    duals: &[f64],
) -> Vec<ConstraintMultiplier> {
    problem
        .constraints
        .iter()
        .zip(duals)
        .map(|(constraint, &dual)| {
            if matches!((constraint.bound.lower, constraint.bound.upper), (Some(l), Some(u)) if l == u)
            {
                ConstraintMultiplier {
                    equality: dual,
                    ..Default::default()
                }
            } else {
                ConstraintMultiplier {
                    lower: (-dual).max(0.0),
                    upper: dual.max(0.0),
                    equality: 0.0,
                }
            }
        })
        .collect()
}

fn predicted_reduction(
    problem: &ConstrainedNonlinearProblem,
    gradient: &[f64],
    values: &[DifferentiableValue],
    hessian: &[Vec<f64>],
    step: &[f64],
    penalty: f64,
) -> f64 {
    let linearized: Vec<DifferentiableValue> = values
        .iter()
        .map(|value| DifferentiableValue {
            value: value.value + dot(&value.gradient, step),
            gradient: Vec::new(),
        })
        .collect();
    let model_change = dot(gradient, step) + 0.5 * dot(step, &matvec(hessian, step));
    let feasibility_change = violation_sum(problem, &linearized) - violation_sum(problem, values);
    -(model_change + penalty * feasibility_change)
}

fn violation_sum(problem: &ConstrainedNonlinearProblem, values: &[DifferentiableValue]) -> f64 {
    problem.constraints.iter().zip(values).fold(0.0, |sum, (constraint, value)| {
        if matches!((constraint.bound.lower, constraint.bound.upper), (Some(l), Some(u)) if l == u)
        {
            sum + (value.value - constraint.bound.lower.unwrap()).abs()
        } else {
            sum + constraint.bound.lower.map_or(0.0, |l| (l - value.value).max(0.0))
                + constraint.bound.upper.map_or(0.0, |u| (value.value - u).max(0.0))
        }
    })
}

fn maximum_violation(problem: &ConstrainedNonlinearProblem, values: &[DifferentiableValue]) -> f64 {
    problem
        .constraints
        .iter()
        .zip(values)
        .fold(0.0, |maximum, (constraint, value)| {
            maximum
                .max(
                    constraint
                        .bound
                        .lower
                        .map_or(0.0, |l| (l - value.value).max(0.0)),
                )
                .max(
                    constraint
                        .bound
                        .upper
                        .map_or(0.0, |u| (value.value - u).max(0.0)),
                )
        })
}

fn lagrangian_gradient(
    problem: &ConstrainedNonlinearProblem,
    objective: &[f64],
    values: &[DifferentiableValue],
    multipliers: &[ConstraintMultiplier],
) -> Vec<f64> {
    let mut gradient = objective.to_vec();
    for ((constraint, value), multiplier) in problem.constraints.iter().zip(values).zip(multipliers)
    {
        let scale = if matches!((constraint.bound.lower, constraint.bound.upper), (Some(l), Some(u)) if l == u)
        {
            multiplier.equality
        } else {
            multiplier.upper - multiplier.lower
        };
        add_scaled(&mut gradient, &value.gradient, scale);
    }
    gradient
}

fn lagrangian_weights(multipliers: &[ConstraintMultiplier]) -> Vec<f64> {
    multipliers.iter().map(|value| {
        if value.equality != 0.0 { value.equality } else { value.upper - value.lower }
    }).collect()
}

fn stationarity_norm(
    problem: &ConstrainedNonlinearProblem,
    point: &[f64],
    objective: &[f64],
    values: &[DifferentiableValue],
    multipliers: &[ConstraintMultiplier],
) -> f64 {
    let mut gradient = lagrangian_gradient(problem, objective, values, multipliers);
    for ((value, bound), entry) in point.iter().zip(&problem.model.bounds).zip(&mut gradient) {
        if (bound.lower.is_some_and(|lower| *value <= lower + 1e-12) && *entry > 0.0)
            || (bound.upper.is_some_and(|upper| *value >= upper - 1e-12) && *entry < 0.0)
        {
            *entry = 0.0;
        }
    }
    inf_norm(&gradient)
}

fn bfgs_update(hessian: &mut Vec<Vec<f64>>, step: &[f64], y: &[f64], regularization: f64) {
    let bs = matvec(hessian, step);
    let sbs = dot(step, &bs);
    let sy = dot(step, y);
    let scale = inf_norm(step).max(1.0) * inf_norm(y).max(1.0);
    if !sbs.is_finite() || !sy.is_finite() || sbs <= regularization || sy <= 1e-10 * scale {
        *hessian = identity(step.len());
        return;
    }
    for i in 0..step.len() {
        for j in 0..step.len() {
            hessian[i][j] += y[i] * y[j] / sy - bs[i] * bs[j] / sbs;
        }
    }
    regularize(hessian, regularization);
}

fn regularize(hessian: &mut [Vec<f64>], regularization: f64) {
    for (index, row) in hessian.iter_mut().enumerate() {
        row[index] = row[index].max(regularization);
    }
}

#[allow(clippy::too_many_arguments)]
fn result(
    point: Vec<f64>,
    objective: DifferentiableValue,
    constraints: Vec<DifferentiableValue>,
    multipliers: Vec<ConstraintMultiplier>,
    iterations: usize,
    evaluations: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    penalty: f64,
    last_step_norm: f64,
    termination: SqpTermination,
    problem: &ConstrainedNonlinearProblem,
) -> SqpResult {
    SqpResult {
        maximum_violation: maximum_violation(problem, &constraints),
        stationarity_norm: stationarity_norm(
            problem,
            &point,
            &objective.gradient,
            &constraints,
            &multipliers,
        ),
        point,
        objective: objective.value,
        constraint_values: constraints.into_iter().map(|value| value.value).collect(),
        multipliers,
        iterations,
        evaluations,
        accepted_steps,
        rejected_steps,
        final_merit_penalty: penalty,
        last_step_norm,
        termination,
    }
}

fn validate_options(options: SqpOptions) -> Result<(), OptimizeError> {
    if options.max_iterations == 0
        || options.max_line_search_iterations == 0
        || !options.feasibility_tolerance.is_finite()
        || options.feasibility_tolerance <= 0.0
        || !options.stationarity_tolerance.is_finite()
        || options.stationarity_tolerance <= 0.0
        || !options.step_tolerance.is_finite()
        || options.step_tolerance <= 0.0
        || !options.merit_penalty.is_finite()
        || options.merit_penalty <= 0.0
        || !options.penalty_increase.is_finite()
        || options.penalty_increase <= 1.0
        || !options.armijo.is_finite()
        || !(0.0..1.0).contains(&options.armijo)
        || !options.backtracking.is_finite()
        || !(0.0..1.0).contains(&options.backtracking)
        || !options.hessian_regularization.is_finite()
        || options.hessian_regularization <= 0.0
        || !options.filter_constraint_margin.is_finite()
        || !(0.0..1.0).contains(&options.filter_constraint_margin)
        || !options.filter_objective_margin.is_finite()
        || options.filter_objective_margin <= 0.0
    {
        return Err(OptimizeError::InvalidConfiguration("invalid SQP options"));
    }
    Ok(())
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n]; n];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    result
}
fn project(point: &[f64], bounds: &[Bound]) -> Vec<f64> {
    point
        .iter()
        .zip(bounds)
        .map(|(&value, bound)| {
            value
                .max(bound.lower.unwrap_or(f64::NEG_INFINITY))
                .min(bound.upper.unwrap_or(f64::INFINITY))
        })
        .collect()
}
fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn inf_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0, |maximum, value| maximum.max(value.abs()))
}
fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += scale * source;
    }
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearConstraint, NonlinearExpression, NonlinearModel, NonlinearNode::*};

    fn expression(nodes: Vec<crate::NonlinearNode>) -> NonlinearExpression {
        let output = nodes.len() - 1;
        NonlinearExpression::new(nodes, output)
    }

    #[test]
    fn solves_equality_constrained_quadratic() {
        let objective = expression(vec![
            Parameter(0),
            Powf(0, 2.0),
            Parameter(1),
            Powf(2, 2.0),
            Add(1, 3),
        ]);
        let equality = expression(vec![Parameter(0), Parameter(1), Add(0, 1)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(2, vec![Bound::free(), Bound::free()], objective),
            constraints: vec![NonlinearConstraint {
                expression: equality,
                bound: Bound::fixed(1.0),
            }],
        };
        let result = minimize_sqp(&problem, &[0.0, 0.0], Default::default()).unwrap();
        assert_eq!(result.termination, SqpTermination::Converged);
        assert!((result.point[0] - 0.5).abs() < 1e-5);
        assert!((result.point[1] - 0.5).abs() < 1e-5);
        assert!(result.maximum_violation < 1e-7);
        assert!(result.stationarity_norm < 1e-5);
    }

    #[test]
    fn solves_active_nonlinear_inequality() {
        let objective = expression(vec![
            Parameter(0),
            Constant(2.0),
            Subtract(0, 1),
            Powf(2, 2.0),
        ]);
        let constraint = expression(vec![Parameter(0), Powf(0, 2.0)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(1, vec![Bound::free()], objective),
            constraints: vec![NonlinearConstraint {
                expression: constraint,
                bound: Bound {
                    lower: None,
                    upper: Some(1.0),
                },
            }],
        };
        let result = minimize_sqp(&problem, &[0.5], Default::default()).unwrap();
        assert_eq!(result.termination, SqpTermination::Converged);
        assert!((result.point[0] - 1.0).abs() < 1e-5);
        assert!(result.multipliers[0].upper > 0.0);
    }

    #[test]
    fn filter_globalization_with_restoration_solves_infeasible_start() {
        let objective = expression(vec![Parameter(0), Constant(2.0), Subtract(0, 1), Powf(2, 2.0)]);
        let constraint = expression(vec![Parameter(0)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(1, vec![Bound::free()], objective),
            constraints: vec![NonlinearConstraint { expression: constraint,
                bound: Bound { lower: Some(1.0), upper: None } }],
        };
        let result = minimize_sqp(&problem, &[0.0], SqpOptions {
            restoration: true, globalization: SqpGlobalization::Filter, ..Default::default()
        }).unwrap();
        assert_eq!(result.termination, SqpTermination::Converged, "{result:?}");
        assert!(result.point[0] >= 1.0 - 1e-7);
    }

    #[test]
    fn exact_lagrangian_curvature_solves_nonlinear_inequality() {
        let objective = expression(vec![Parameter(0), Constant(2.0), Subtract(0, 1), Powf(2, 2.0)]);
        let constraint = expression(vec![Parameter(0), Powf(0, 2.0)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(1, vec![Bound::free()], objective),
            constraints: vec![NonlinearConstraint { expression: constraint,
                bound: Bound { lower: None, upper: Some(1.0) } }],
        };
        let result = minimize_sqp(&problem, &[0.5], SqpOptions {
            curvature: SqpCurvature::ExactLagrangian, ..Default::default()
        }).unwrap();
        assert_eq!(result.termination, SqpTermination::Converged, "{result:?}");
        assert!((result.point[0] - 1.0).abs() < 1e-5);
    }
}
