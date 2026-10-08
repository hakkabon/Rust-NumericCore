//! Feasible-start interior-point method for graph-represented nonlinear programs.

use crate::{
    Bound, ConstrainedNonlinearProblem, ConstraintMultiplier, DifferentiableValue, OptimizeError,
};

#[derive(Debug, Clone, Copy)]
pub struct NonlinearInteriorPointOptions {
    pub max_outer_iterations: usize,
    pub max_inner_iterations: usize,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub complementarity_tolerance: f64,
    pub initial_barrier: f64,
    pub barrier_reduction: f64,
    pub minimum_barrier: f64,
    pub equality_penalty: f64,
    pub armijo: f64,
    pub backtracking: f64,
    pub fraction_to_boundary: f64,
    pub max_line_search_iterations: usize,
}

impl Default for NonlinearInteriorPointOptions {
    fn default() -> Self {
        Self {
            max_outer_iterations: 50, max_inner_iterations: 100,
            feasibility_tolerance: 1e-7, stationarity_tolerance: 1e-4,
            complementarity_tolerance: 1e-7, initial_barrier: 0.1,
            barrier_reduction: 0.2, minimum_barrier: 1e-7,
            equality_penalty: 10.0, armijo: 1e-4, backtracking: 0.5,
            fraction_to_boundary: 0.995, max_line_search_iterations: 40,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearInteriorPointTermination {
    Converged, IterationLimit, InfeasibleStart, LineSearchFailed, NumericalFailure, Cancelled,
}

#[derive(Debug, Clone)]
pub struct NonlinearInteriorPointResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<ConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub complementarity: f64,
    pub outer_iterations: usize,
    pub inner_iterations: usize,
    pub evaluations: usize,
    pub final_barrier: f64,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub termination: NonlinearInteriorPointTermination,
}

#[derive(Debug, Clone, Copy)]
pub struct NonlinearInteriorPointIteration {
    pub outer_iteration: usize,
    pub inner_iterations: usize,
    pub objective: f64,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub complementarity: f64,
    pub barrier: f64,
}

pub fn minimize_nonlinear_interior_point(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: NonlinearInteriorPointOptions,
) -> Result<NonlinearInteriorPointResult, OptimizeError> {
    minimize_nonlinear_interior_point_with_observer(problem, initial, options, |_| true)
}

pub fn minimize_nonlinear_interior_point_with_observer<O>(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: NonlinearInteriorPointOptions,
    mut observer: O,
) -> Result<NonlinearInteriorPointResult, OptimizeError>
where O: FnMut(NonlinearInteriorPointIteration) -> bool {
    problem.validate()?;
    validate_options(options)?;
    let n = problem.model.parameter_count;
    if initial.len() != n || !initial.iter().all(|value| value.is_finite()) {
        return Err(invalid("initial point must match the nonlinear interior-point problem"));
    }
    let mut point = initial.to_vec();
    let initial_values = problem.evaluate_constraints(&point)?;
    if !strictly_interior(problem, &point, &initial_values) {
        return Ok(finish(problem, point, vec![0.0; equality_count(problem)], 0, 1,
            options.initial_barrier, 0, 0, NonlinearInteriorPointTermination::InfeasibleStart)?);
    }

    let mut equality_multipliers = vec![0.0; equality_count(problem)];
    let mut barrier = options.initial_barrier;
    let mut total_inner = 0;
    let mut evaluations = 1;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;

    for outer in 1..=options.max_outer_iterations {
        let mut inverse_hessian = identity(n);
        for _ in 0..options.max_inner_iterations {
            let current = barrier_value_gradient(
                problem, &point, barrier, &equality_multipliers, options.equality_penalty)?;
            let gradient_norm = inf_norm(&current.1);
            let inner_tolerance = options.stationarity_tolerance
                .min(options.feasibility_tolerance * options.equality_penalty)
                .max(0.1 * barrier);
            if gradient_norm <= inner_tolerance { break; }
            let mut direction = matvec(&inverse_hessian, &current.1);
            for value in &mut direction { *value = -*value; }
            if dot(&current.1, &direction) >= -1e-14 {
                direction = current.1.iter().map(|value| -value).collect();
                inverse_hessian = identity(n);
            }
            let slope = dot(&current.1, &direction);
            let mut alpha = options.fraction_to_boundary.min(1.0);
            let mut accepted = None;
            for _ in 0..options.max_line_search_iterations {
                let trial: Vec<f64> = point.iter().zip(&direction)
                    .map(|(value, step)| value + alpha * step).collect();
                let trial_values = problem.evaluate_constraints(&trial)?;
                evaluations += 1;
                if strictly_interior(problem, &trial, &trial_values) {
                    let trial_evaluation = barrier_value_gradient(
                        problem, &trial, barrier, &equality_multipliers,
                        options.equality_penalty)?;
                    if trial_evaluation.0 <= current.0 + options.armijo * alpha * slope {
                        accepted = Some((trial, trial_evaluation));
                        break;
                    }
                }
                rejected_steps += 1;
                alpha *= options.backtracking;
            }
            let Some((trial, trial_evaluation)) = accepted else {
                // A barrier subproblem can become numerically flat near its
                // central-path solution. Assess the outer KKT conditions and
                // reduce the barrier before declaring the solve exhausted.
                break;
            };
            let step: Vec<f64> = trial.iter().zip(&point).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = trial_evaluation.1.iter().zip(&current.1).map(|(a, b)| a - b).collect();
            bfgs_inverse_update(&mut inverse_hessian, &step, &y);
            point = trial;
            total_inner += 1;
            accepted_steps += 1;
        }

        let objective = problem.model.evaluate_objective(&point)?;
        let values = problem.evaluate_constraints(&point)?;
        evaluations += 1;
        update_equalities(problem, &values, &mut equality_multipliers, options.equality_penalty);
        let multipliers = recover_multipliers(problem, &values, barrier, &equality_multipliers);
        let violation = maximum_violation(problem, &point, &values);
        let stationarity = stationarity_norm(problem, &point, &objective.gradient, &values, &multipliers, barrier);
        let complementarity = complementarity(problem, &values, &multipliers, barrier);
        let snapshot = NonlinearInteriorPointIteration {
            outer_iteration: outer, inner_iterations: total_inner, objective: objective.value,
            maximum_violation: violation, stationarity_norm: stationarity,
            complementarity, barrier,
        };
        if !observer(snapshot) {
            return finish(problem, point, equality_multipliers, outer, evaluations, barrier,
                accepted_steps, rejected_steps, NonlinearInteriorPointTermination::Cancelled);
        }
        if violation <= options.feasibility_tolerance
            && stationarity <= options.stationarity_tolerance
            && complementarity <= options.complementarity_tolerance
        {
            return finish(problem, point, equality_multipliers, outer, evaluations, barrier,
                accepted_steps, rejected_steps, NonlinearInteriorPointTermination::Converged);
        }
        barrier = (barrier * options.barrier_reduction).max(options.minimum_barrier);
    }
    finish(problem, point, equality_multipliers, options.max_outer_iterations, evaluations,
        barrier, accepted_steps, rejected_steps, NonlinearInteriorPointTermination::IterationLimit)
}

fn barrier_value_gradient(
    problem: &ConstrainedNonlinearProblem, point: &[f64], barrier: f64,
    equality_multipliers: &[f64], equality_penalty: f64,
) -> Result<(f64, Vec<f64>), OptimizeError> {
    let objective = problem.model.evaluate_objective(point)?;
    let values = problem.evaluate_constraints(point)?;
    let mut value = objective.value;
    let mut gradient = objective.gradient;
    let mut equality = 0;
    for (constraint, evaluated) in problem.constraints.iter().zip(&values) {
        if is_equality(constraint.bound) {
            let residual = evaluated.value - constraint.bound.lower.unwrap();
            let coefficient = equality_multipliers[equality] + equality_penalty * residual;
            value += equality_multipliers[equality] * residual + 0.5 * equality_penalty * residual * residual;
            add_scaled(&mut gradient, &evaluated.gradient, coefficient);
            equality += 1;
        } else {
            if let Some(lower) = constraint.bound.lower {
                let slack = evaluated.value - lower;
                if slack <= 0.0 { return Err(invalid("barrier evaluated outside its interior")); }
                value -= barrier * slack.ln();
                add_scaled(&mut gradient, &evaluated.gradient, -barrier / slack);
            }
            if let Some(upper) = constraint.bound.upper {
                let slack = upper - evaluated.value;
                if slack <= 0.0 { return Err(invalid("barrier evaluated outside its interior")); }
                value -= barrier * slack.ln();
                add_scaled(&mut gradient, &evaluated.gradient, barrier / slack);
            }
        }
    }
    for ((&x, bound), index) in point.iter().zip(&problem.model.bounds).zip(0..) {
        if let Some(lower) = bound.lower {
            let slack = x - lower; if slack <= 0.0 { return Err(invalid("point is outside variable interior")); }
            value -= barrier * slack.ln(); gradient[index] -= barrier / slack;
        }
        if let Some(upper) = bound.upper {
            let slack = upper - x; if slack <= 0.0 { return Err(invalid("point is outside variable interior")); }
            value -= barrier * slack.ln(); gradient[index] += barrier / slack;
        }
    }
    Ok((value, gradient))
}

fn strictly_interior(problem: &ConstrainedNonlinearProblem, point: &[f64], values: &[DifferentiableValue]) -> bool {
    if point.iter().zip(&problem.model.bounds).any(|(&x, bound)|
        bound.lower.is_some_and(|lower| x <= lower) || bound.upper.is_some_and(|upper| x >= upper)) { return false; }
    problem.constraints.iter().zip(values).all(|(constraint, value)| {
        is_equality(constraint.bound)
            || (constraint.bound.lower.is_none_or(|lower| value.value > lower)
                && constraint.bound.upper.is_none_or(|upper| value.value < upper))
    })
}

fn recover_multipliers(problem: &ConstrainedNonlinearProblem, values: &[DifferentiableValue], barrier: f64, equalities: &[f64]) -> Vec<ConstraintMultiplier> {
    let mut equality = 0;
    problem.constraints.iter().zip(values).map(|(constraint, value)| {
        if is_equality(constraint.bound) {
            let result = ConstraintMultiplier { equality: equalities[equality], ..Default::default() };
            equality += 1; result
        } else {
            ConstraintMultiplier {
                lower: constraint.bound.lower.map_or(0.0, |lower| barrier / (value.value - lower)),
                upper: constraint.bound.upper.map_or(0.0, |upper| barrier / (upper - value.value)),
                equality: 0.0,
            }
        }
    }).collect()
}

fn update_equalities(problem: &ConstrainedNonlinearProblem, values: &[DifferentiableValue], multipliers: &mut [f64], penalty: f64) {
    let mut index = 0;
    for (constraint, value) in problem.constraints.iter().zip(values) {
        if is_equality(constraint.bound) {
            multipliers[index] += penalty * (value.value - constraint.bound.lower.unwrap()); index += 1;
        }
    }
}

fn finish(problem: &ConstrainedNonlinearProblem, point: Vec<f64>, equalities: Vec<f64>, outer: usize,
    evaluations: usize, barrier: f64, accepted_steps: usize, rejected_steps: usize,
    termination: NonlinearInteriorPointTermination) -> Result<NonlinearInteriorPointResult, OptimizeError> {
    let objective = problem.model.evaluate_objective(&point)?;
    let values = problem.evaluate_constraints(&point)?;
    let multipliers = if strictly_interior(problem, &point, &values) {
        recover_multipliers(problem, &values, barrier, &equalities)
    } else { vec![ConstraintMultiplier::default(); problem.constraints.len()] };
    Ok(NonlinearInteriorPointResult {
        point: point.clone(), objective: objective.value,
        constraint_values: values.iter().map(|value| value.value).collect(),
        maximum_violation: maximum_violation(problem, &point, &values),
        stationarity_norm: stationarity_norm(problem, &point, &objective.gradient, &values, &multipliers, barrier),
        complementarity: complementarity(problem, &values, &multipliers, barrier), multipliers,
        outer_iterations: outer, inner_iterations: accepted_steps, evaluations, final_barrier: barrier,
        accepted_steps, rejected_steps, termination,
    })
}

fn is_equality(bound: Bound) -> bool { matches!((bound.lower, bound.upper), (Some(lower), Some(upper)) if lower == upper) }
fn equality_count(problem: &ConstrainedNonlinearProblem) -> usize { problem.constraints.iter().filter(|constraint| is_equality(constraint.bound)).count() }
fn maximum_violation(problem: &ConstrainedNonlinearProblem, point: &[f64], values: &[DifferentiableValue]) -> f64 {
    let constraint_violation = problem.constraints.iter().zip(values).fold(0.0_f64, |maximum, (constraint, value)| maximum
        .max(constraint.bound.lower.map_or(0.0, |lower| (lower - value.value).max(0.0)))
        .max(constraint.bound.upper.map_or(0.0, |upper| (value.value - upper).max(0.0))));
    problem.model.bounds.iter().zip(point).fold(constraint_violation, |maximum, (bound, value)| maximum
        .max(bound.lower.map_or(0.0, |lower| (lower - value).max(0.0)))
        .max(bound.upper.map_or(0.0, |upper| (value - upper).max(0.0))))
}
fn complementarity(problem: &ConstrainedNonlinearProblem, values: &[DifferentiableValue], multipliers: &[ConstraintMultiplier], barrier: f64) -> f64 {
    let constraint_complementarity = problem.constraints.iter().zip(values).zip(multipliers).fold(0.0_f64, |maximum, ((constraint, value), multiplier)| {
        if is_equality(constraint.bound) { maximum } else { maximum
            .max(constraint.bound.lower.map_or(0.0, |lower| multiplier.lower * (value.value - lower)))
            .max(constraint.bound.upper.map_or(0.0, |upper| multiplier.upper * (upper - value.value))) }
    });
    if problem.model.bounds.iter().any(|bound| bound.lower.is_some() || bound.upper.is_some()) {
        constraint_complementarity.max(barrier)
    } else { constraint_complementarity }
}
fn stationarity_norm(problem: &ConstrainedNonlinearProblem, point: &[f64], objective: &[f64], values: &[DifferentiableValue], multipliers: &[ConstraintMultiplier], barrier: f64) -> f64 {
    let mut gradient = objective.to_vec();
    for ((constraint, value), multiplier) in problem.constraints.iter().zip(values).zip(multipliers) {
        let scale = if is_equality(constraint.bound) { multiplier.equality } else { multiplier.upper - multiplier.lower };
        add_scaled(&mut gradient, &value.gradient, scale);
    }
    for ((&x, bound), value) in point.iter().zip(&problem.model.bounds).zip(&mut gradient) {
        if let Some(lower) = bound.lower { *value -= barrier / (x - lower); }
        if let Some(upper) = bound.upper { *value += barrier / (upper - x); }
    }
    inf_norm(&gradient)
}
fn bfgs_inverse_update(h: &mut [Vec<f64>], s: &[f64], y: &[f64]) {
    let sy = dot(s, y); if !sy.is_finite() || sy <= 1e-12 * inf_norm(s).max(1.0) * inf_norm(y).max(1.0) { return; }
    let hy = matvec(h, y); let yhy = dot(y, &hy); let coefficient = (sy + yhy) / (sy * sy);
    for i in 0..h.len() { for j in 0..h.len() { h[i][j] += coefficient * s[i] * s[j] - (hy[i] * s[j] + s[i] * hy[j]) / sy; } }
}
fn identity(n: usize) -> Vec<Vec<f64>> { (0..n).map(|i| (0..n).map(|j| f64::from(i == j)).collect()).collect() }
fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> { matrix.iter().map(|row| dot(row, vector)).collect() }
fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(a, b)| a * b).sum() }
fn inf_norm(values: &[f64]) -> f64 { values.iter().fold(0.0_f64, |maximum, value| maximum.max(value.abs())) }
fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) { for (target, source) in target.iter_mut().zip(source) { *target += scale * source; } }
fn validate_options(options: NonlinearInteriorPointOptions) -> Result<(), OptimizeError> {
    if options.max_outer_iterations == 0 || options.max_inner_iterations == 0
        || options.max_line_search_iterations == 0
        || !options.feasibility_tolerance.is_finite() || options.feasibility_tolerance <= 0.0
        || !options.stationarity_tolerance.is_finite() || options.stationarity_tolerance <= 0.0
        || !options.complementarity_tolerance.is_finite() || options.complementarity_tolerance <= 0.0
        || !options.initial_barrier.is_finite() || options.initial_barrier <= 0.0
        || !options.minimum_barrier.is_finite() || options.minimum_barrier <= 0.0 || options.minimum_barrier > options.initial_barrier
        || !options.barrier_reduction.is_finite() || !(0.0..1.0).contains(&options.barrier_reduction)
        || !options.equality_penalty.is_finite() || options.equality_penalty <= 0.0
        || !options.armijo.is_finite() || !(0.0..1.0).contains(&options.armijo)
        || !options.backtracking.is_finite() || !(0.0..1.0).contains(&options.backtracking)
        || !options.fraction_to_boundary.is_finite() || !(0.0..1.0).contains(&options.fraction_to_boundary) {
        return Err(OptimizeError::InvalidConfiguration("invalid nonlinear interior-point options"));
    }
    Ok(())
}
fn invalid(message: &str) -> OptimizeError { OptimizeError::InvalidProblem(message.to_owned()) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearConstraint, NonlinearExpression, NonlinearModel, NonlinearNode};

    fn expression(nodes: Vec<NonlinearNode>) -> NonlinearExpression {
        let output = nodes.len() - 1;
        NonlinearExpression::new(nodes, output)
    }

    #[test]
    fn solves_active_nonlinear_inequality() {
        let objective = expression(vec![
            NonlinearNode::Parameter(0), NonlinearNode::Constant(2.0),
            NonlinearNode::Subtract(0, 1), NonlinearNode::Powf(2, 2.0),
        ]);
        let constraint = expression(vec![NonlinearNode::Parameter(0)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel { parameter_count: 1, bounds: vec![Bound::free()],
                objective: Some(objective), residuals: vec![] },
            constraints: vec![NonlinearConstraint { expression: constraint,
                bound: Bound { lower: None, upper: Some(1.0) } }],
        };
        let result = minimize_nonlinear_interior_point(
            &problem, &[0.0], NonlinearInteriorPointOptions::default()).unwrap();
        assert_eq!(result.termination, NonlinearInteriorPointTermination::Converged, "{:?}", result);
        assert!((result.point[0] - 1.0).abs() < 1e-4, "{:?}", result);
        assert!(result.multipliers[0].upper > 1.9);
    }

    #[test]
    fn solves_equality_from_infeasible_initial_point() {
        let objective = expression(vec![
            NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0),
            NonlinearNode::Parameter(1), NonlinearNode::Powf(2, 2.0),
            NonlinearNode::Add(1, 3),
        ]);
        let constraint = expression(vec![
            NonlinearNode::Parameter(0), NonlinearNode::Parameter(1),
            NonlinearNode::Add(0, 1),
        ]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel { parameter_count: 2,
                bounds: vec![Bound::free(), Bound::free()], objective: Some(objective), residuals: vec![] },
            constraints: vec![NonlinearConstraint { expression: constraint,
                bound: Bound { lower: Some(1.0), upper: Some(1.0) } }],
        };
        let result = minimize_nonlinear_interior_point(
            &problem, &[0.0, 0.0], NonlinearInteriorPointOptions::default()).unwrap();
        assert_eq!(result.termination, NonlinearInteriorPointTermination::Converged, "{:?}", result);
        assert!((result.point[0] - 0.5).abs() < 1e-4, "{:?}", result);
        assert!((result.point[1] - 0.5).abs() < 1e-4, "{:?}", result);
    }

    #[test]
    fn reports_non_strict_inequality_start() {
        let objective = expression(vec![NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0)]);
        let constraint = expression(vec![NonlinearNode::Parameter(0)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel { parameter_count: 1, bounds: vec![Bound::free()],
                objective: Some(objective), residuals: vec![] },
            constraints: vec![NonlinearConstraint { expression: constraint,
                bound: Bound { lower: None, upper: Some(1.0) } }],
        };
        let result = minimize_nonlinear_interior_point(
            &problem, &[1.0], NonlinearInteriorPointOptions::default()).unwrap();
        assert_eq!(result.termination, NonlinearInteriorPointTermination::InfeasibleStart);
    }
}
