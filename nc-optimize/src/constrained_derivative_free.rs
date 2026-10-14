//! Constrained value-only optimization with local affine interpolation models.

use crate::{Bound, OptimizeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeFreeConstraintKind {
    LessEqual,
    Equality,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstrainedDerivativeFreeEvaluation {
    pub objective: f64,
    pub constraints: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ConstrainedDerivativeFreeOptions {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub initial_radius: f64,
    pub final_radius: f64,
    pub maximum_radius: f64,
    pub feasibility_tolerance: f64,
    pub objective_tolerance: f64,
    pub initial_penalty: f64,
    pub penalty_growth: f64,
    pub maximum_penalty: f64,
    pub acceptance_threshold: f64,
    pub contraction: f64,
    pub expansion: f64,
    pub subproblem_iterations: usize,
}
impl Default for ConstrainedDerivativeFreeOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 20_000,
            initial_radius: 1.,
            final_radius: 1e-7,
            maximum_radius: 100.,
            feasibility_tolerance: 1e-7,
            objective_tolerance: 1e-12,
            initial_penalty: 10.,
            penalty_growth: 10.,
            maximum_penalty: 1e12,
            acceptance_threshold: 0.1,
            contraction: 0.5,
            expansion: 1.5,
            subproblem_iterations: 80,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrainedDerivativeFreeTermination {
    Converged,
    InfeasibleStationary,
    IterationLimit,
    EvaluationLimit,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct ConstrainedDerivativeFreeIteration {
    pub iteration: usize,
    pub point: Vec<f64>,
    pub objective: f64,
    pub maximum_violation: f64,
    pub evaluations: usize,
    pub radius: f64,
    pub penalty: f64,
}

#[derive(Debug, Clone)]
pub struct ConstrainedDerivativeFreeResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub maximum_violation: f64,
    pub violation_sum: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub model_builds: usize,
    pub accepted_steps: usize,
    pub feasibility_improving_steps: usize,
    pub penalty_updates: usize,
    pub final_penalty: f64,
    pub final_radius: f64,
    pub last_reduction_ratio: f64,
    pub termination: ConstrainedDerivativeFreeTermination,
}

pub fn minimize_constrained_derivative_free<F>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    kinds: &[DerivativeFreeConstraintKind],
    options: ConstrainedDerivativeFreeOptions,
    evaluate: F,
) -> Result<ConstrainedDerivativeFreeResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
{
    minimize_constrained_derivative_free_with_observer(
        initial,
        bounds,
        kinds,
        options,
        evaluate,
        |_| true,
    )
}

pub fn minimize_constrained_derivative_free_with_observer<F, O>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    kinds: &[DerivativeFreeConstraintKind],
    options: ConstrainedDerivativeFreeOptions,
    mut evaluate: F,
    mut observer: O,
) -> Result<ConstrainedDerivativeFreeResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
    O: FnMut(&ConstrainedDerivativeFreeIteration) -> bool,
{
    validate(initial, bounds, kinds, options)?;
    let box_bounds = bounds
        .map(<[Bound]>::to_vec)
        .unwrap_or_else(|| (0..initial.len()).map(|_| Bound::free()).collect());
    let mut point = project(initial, &box_bounds);
    let mut current = checked(&mut evaluate, &point, kinds.len())?;
    let mut evaluations = 1;
    let mut radius = options.initial_radius;
    let mut penalty = options.initial_penalty;
    let mut model_builds = 0;
    let mut accepted_steps = 0;
    let mut feasibility_steps = 0;
    let mut penalty_updates = 0;
    let mut last_ratio = f64::NAN;
    for iteration in 0..options.max_iterations {
        let (maximum, _) = violations(&current.constraints, kinds);
        if radius <= options.final_radius {
            let termination = if maximum <= options.feasibility_tolerance {
                ConstrainedDerivativeFreeTermination::Converged
            } else {
                ConstrainedDerivativeFreeTermination::InfeasibleStationary
            };
            return Ok(result(
                point,
                current,
                iteration,
                evaluations,
                model_builds,
                accepted_steps,
                feasibility_steps,
                penalty_updates,
                penalty,
                radius,
                last_ratio,
                termination,
                kinds,
            ));
        }
        if evaluations + initial.len() + 1 > options.max_evaluations {
            return Ok(result(
                point,
                current,
                iteration,
                evaluations,
                model_builds,
                accepted_steps,
                feasibility_steps,
                penalty_updates,
                penalty,
                radius,
                last_ratio,
                ConstrainedDerivativeFreeTermination::EvaluationLimit,
                kinds,
            ));
        }
        if !observer(&ConstrainedDerivativeFreeIteration {
            iteration,
            point: point.clone(),
            objective: current.objective,
            maximum_violation: maximum,
            evaluations,
            radius,
            penalty,
        }) {
            return Ok(result(
                point,
                current,
                iteration,
                evaluations,
                model_builds,
                accepted_steps,
                feasibility_steps,
                penalty_updates,
                penalty,
                radius,
                last_ratio,
                ConstrainedDerivativeFreeTermination::Cancelled,
                kinds,
            ));
        }
        let (objective_gradient, constraint_gradients) = linear_models(
            &point,
            &current,
            radius,
            &box_bounds,
            kinds.len(),
            &mut evaluate,
            &mut evaluations,
        )?;
        model_builds += 1;
        let step = linearized_merit_step(
            &point,
            &current,
            &objective_gradient,
            &constraint_gradients,
            kinds,
            penalty,
            radius,
            &box_bounds,
            options.subproblem_iterations,
        );
        let step_norm = norm(&step);
        if step_norm <= options.final_radius * 0.1 {
            radius *= options.contraction;
            continue;
        }
        let predicted = merit(&current, kinds, penalty)
            - linearized_merit(
                &current,
                &objective_gradient,
                &constraint_gradients,
                &step,
                kinds,
                penalty,
            );
        if !predicted.is_finite() || predicted <= 0. {
            radius *= options.contraction;
            continue;
        }
        let candidate_point = project(
            &point
                .iter()
                .zip(&step)
                .map(|(x, s)| x + s)
                .collect::<Vec<_>>(),
            &box_bounds,
        );
        let candidate = checked(&mut evaluate, &candidate_point, kinds.len())?;
        evaluations += 1;
        let actual = merit(&current, kinds, penalty) - merit(&candidate, kinds, penalty);
        last_ratio = actual / predicted;
        let (candidate_max, _) = violations(&candidate.constraints, kinds);
        let feasibility_improved = candidate_max + options.feasibility_tolerance * 0.1 < maximum;
        let accepted = (last_ratio >= options.acceptance_threshold
            && actual > options.objective_tolerance)
            || feasibility_improved;
        if accepted {
            if feasibility_improved {
                feasibility_steps += 1;
            }
            point = candidate_point;
            current = candidate;
            accepted_steps += 1;
        }
        if maximum > options.feasibility_tolerance
            && candidate_max >= 0.9 * maximum
            && penalty < options.maximum_penalty
        {
            penalty = (penalty * options.penalty_growth).min(options.maximum_penalty);
            penalty_updates += 1;
        }
        if !accepted || last_ratio < 0.1 {
            radius *= options.contraction;
        } else if last_ratio > 0.75 && step_norm >= 0.8 * radius {
            radius = (radius * options.expansion).min(options.maximum_radius);
        }
    }
    Ok(result(
        point,
        current,
        options.max_iterations,
        evaluations,
        model_builds,
        accepted_steps,
        feasibility_steps,
        penalty_updates,
        penalty,
        radius,
        last_ratio,
        ConstrainedDerivativeFreeTermination::IterationLimit,
        kinds,
    ))
}

fn linear_models<F>(
    point: &[f64],
    center: &ConstrainedDerivativeFreeEvaluation,
    radius: f64,
    bounds: &[Bound],
    count: usize,
    evaluate: &mut F,
    evaluations: &mut usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
{
    let mut objective = vec![0.; point.len()];
    let mut constraints = vec![vec![0.; point.len()]; count];
    for i in 0..point.len() {
        let plus = radius.min(bounds[i].upper.unwrap_or(f64::INFINITY) - point[i]);
        let minus = radius.min(point[i] - bounds[i].lower.unwrap_or(f64::NEG_INFINITY));
        let signed_step = if plus >= minus && plus > 1e-14 {
            plus
        } else {
            -minus
        };
        let mut sample = point.to_vec();
        sample[i] += signed_step;
        let value = checked(evaluate, &sample, count)?;
        *evaluations += 1;
        objective[i] = (value.objective - center.objective) / signed_step;
        for j in 0..count {
            constraints[j][i] = (value.constraints[j] - center.constraints[j]) / signed_step;
        }
    }
    Ok((objective, constraints))
}

fn linearized_merit_step(
    point: &[f64],
    center: &ConstrainedDerivativeFreeEvaluation,
    objective_gradient: &[f64],
    constraint_gradients: &[Vec<f64>],
    kinds: &[DerivativeFreeConstraintKind],
    penalty: f64,
    radius: f64,
    bounds: &[Bound],
    iterations: usize,
) -> Vec<f64> {
    let mut step = vec![0.; point.len()];
    let mut best = step.clone();
    let mut best_value = linearized_merit(
        center,
        objective_gradient,
        constraint_gradients,
        &step,
        kinds,
        penalty,
    );
    for iteration in 0..iterations {
        let mut gradient = objective_gradient.to_vec();
        for (j, kind) in kinds.iter().enumerate() {
            let value = center.constraints[j] + dot(&constraint_gradients[j], &step);
            let weight = match kind {
                DerivativeFreeConstraintKind::LessEqual if value > 0. => penalty,
                DerivativeFreeConstraintKind::Equality if value > 0. => penalty,
                DerivativeFreeConstraintKind::Equality if value < 0. => -penalty,
                _ => 0.,
            };
            for i in 0..gradient.len() {
                gradient[i] += weight * constraint_gradients[j][i];
            }
        }
        let gradient_norm = norm(&gradient);
        if gradient_norm <= 1e-14 {
            break;
        }
        let length = radius / ((iteration + 1) as f64).sqrt();
        for i in 0..step.len() {
            step[i] -= length * gradient[i] / gradient_norm;
            step[i] = step[i]
                .max(bounds[i].lower.unwrap_or(f64::NEG_INFINITY) - point[i])
                .min(bounds[i].upper.unwrap_or(f64::INFINITY) - point[i]);
        }
        let step_norm = norm(&step);
        if step_norm > radius {
            for value in &mut step {
                *value *= radius / step_norm;
            }
        }
        let value = linearized_merit(
            center,
            objective_gradient,
            constraint_gradients,
            &step,
            kinds,
            penalty,
        );
        if value < best_value {
            best_value = value;
            best = step.clone();
        }
    }
    if violations(&center.constraints, kinds).0 <= 1e-4 {
        let mut tangent = objective_gradient.to_vec();
        for (index, kind) in kinds.iter().enumerate() {
            if *kind == DerivativeFreeConstraintKind::Equality || center.constraints[index] >= -1e-4
            {
                let denominator = dot(&constraint_gradients[index], &constraint_gradients[index]);
                if denominator > 1e-20 {
                    let factor = dot(&tangent, &constraint_gradients[index]) / denominator;
                    for (value, normal) in tangent.iter_mut().zip(&constraint_gradients[index]) {
                        *value -= factor * normal;
                    }
                }
            }
        }
        let tangent_norm = norm(&tangent);
        if tangent_norm > 1e-14 {
            let mut candidate = tangent
                .iter()
                .map(|value| -radius * value / tangent_norm)
                .collect::<Vec<_>>();
            for i in 0..candidate.len() {
                candidate[i] = candidate[i]
                    .max(bounds[i].lower.unwrap_or(f64::NEG_INFINITY) - point[i])
                    .min(bounds[i].upper.unwrap_or(f64::INFINITY) - point[i]);
            }
            let value = linearized_merit(
                center,
                objective_gradient,
                constraint_gradients,
                &candidate,
                kinds,
                penalty,
            );
            if value < best_value {
                best = candidate;
            }
        }
    }
    best
}

fn linearized_merit(
    center: &ConstrainedDerivativeFreeEvaluation,
    objective_gradient: &[f64],
    constraint_gradients: &[Vec<f64>],
    step: &[f64],
    kinds: &[DerivativeFreeConstraintKind],
    penalty: f64,
) -> f64 {
    let objective = center.objective + dot(objective_gradient, step);
    objective
        + penalty
            * kinds
                .iter()
                .enumerate()
                .map(|(i, kind)| {
                    violation(
                        center.constraints[i] + dot(&constraint_gradients[i], step),
                        *kind,
                    )
                })
                .sum::<f64>()
}
fn merit(
    value: &ConstrainedDerivativeFreeEvaluation,
    kinds: &[DerivativeFreeConstraintKind],
    penalty: f64,
) -> f64 {
    value.objective + penalty * violations(&value.constraints, kinds).1
}
fn violation(value: f64, kind: DerivativeFreeConstraintKind) -> f64 {
    match kind {
        DerivativeFreeConstraintKind::LessEqual => value.max(0.),
        DerivativeFreeConstraintKind::Equality => value.abs(),
    }
}
fn violations(values: &[f64], kinds: &[DerivativeFreeConstraintKind]) -> (f64, f64) {
    values
        .iter()
        .zip(kinds)
        .map(|(&v, &k)| violation(v, k))
        .fold((0., 0.), |(m, s), v| (m.max(v), s + v))
}
fn checked<F>(
    evaluate: &mut F,
    point: &[f64],
    count: usize,
) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
{
    let value = evaluate(point)?;
    if value.objective.is_finite()
        && value.constraints.len() == count
        && value.constraints.iter().all(|v| v.is_finite())
    {
        Ok(value)
    } else {
        Err(invalid(
            "non-finite or dimensionally invalid constrained derivative-free evaluation",
        ))
    }
}
fn validate(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    kinds: &[DerivativeFreeConstraintKind],
    o: ConstrainedDerivativeFreeOptions,
) -> Result<(), OptimizeError> {
    let invalid_bounds = bounds.is_some_and(|b| {
        b.len() != initial.len()
            || b.iter().any(|v| {
                v.lower.is_some_and(|x| !x.is_finite())
                    || v.upper.is_some_and(|x| !x.is_finite())
                    || v.lower.unwrap_or(f64::NEG_INFINITY) >= v.upper.unwrap_or(f64::INFINITY)
            })
    });
    if initial.is_empty()
        || kinds.is_empty()
        || initial.iter().any(|v| !v.is_finite())
        || invalid_bounds
        || o.max_iterations == 0
        || o.max_evaluations < initial.len() + 2
        || !positive(o.initial_radius)
        || !positive(o.final_radius)
        || o.final_radius >= o.initial_radius
        || !positive(o.maximum_radius)
        || o.maximum_radius < o.initial_radius
        || !positive(o.feasibility_tolerance)
        || o.objective_tolerance < 0.
        || !o.objective_tolerance.is_finite()
        || !positive(o.initial_penalty)
        || !o.penalty_growth.is_finite()
        || o.penalty_growth <= 1.
        || !positive(o.maximum_penalty)
        || o.maximum_penalty < o.initial_penalty
        || !(0. ..1.).contains(&o.acceptance_threshold)
        || !(0. ..1.).contains(&o.contraction)
        || !o.expansion.is_finite()
        || o.expansion <= 1.
        || o.subproblem_iterations == 0
    {
        Err(invalid(
            "invalid constrained derivative-free problem or options",
        ))
    } else {
        Ok(())
    }
}
fn project(point: &[f64], bounds: &[Bound]) -> Vec<f64> {
    point
        .iter()
        .zip(bounds)
        .map(|(x, b)| {
            x.max(b.lower.unwrap_or(f64::NEG_INFINITY))
                .min(b.upper.unwrap_or(f64::INFINITY))
        })
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
#[allow(clippy::too_many_arguments)]
fn result(
    point: Vec<f64>,
    value: ConstrainedDerivativeFreeEvaluation,
    iterations: usize,
    evaluations: usize,
    model_builds: usize,
    accepted_steps: usize,
    feasibility_improving_steps: usize,
    penalty_updates: usize,
    final_penalty: f64,
    final_radius: f64,
    last_reduction_ratio: f64,
    termination: ConstrainedDerivativeFreeTermination,
    kinds: &[DerivativeFreeConstraintKind],
) -> ConstrainedDerivativeFreeResult {
    let (max, sum) = violations(&value.constraints, kinds);
    ConstrainedDerivativeFreeResult {
        point,
        objective: value.objective,
        constraint_values: value.constraints,
        maximum_violation: max,
        violation_sum: sum,
        iterations,
        evaluations,
        model_builds,
        accepted_steps,
        feasibility_improving_steps,
        penalty_updates,
        final_penalty,
        final_radius,
        last_reduction_ratio,
        termination,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves_equality_constrained_value_only_problem() {
        let kinds = [DerivativeFreeConstraintKind::Equality];
        let result = minimize_constrained_derivative_free(
            &[-1., 2.],
            None,
            &kinds,
            Default::default(),
            |x| {
                Ok(ConstrainedDerivativeFreeEvaluation {
                    objective: x[0] * x[0] + x[1] * x[1],
                    constraints: vec![x[0] + x[1] - 1.],
                })
            },
        )
        .unwrap();
        assert_eq!(
            result.termination,
            ConstrainedDerivativeFreeTermination::Converged
        );
        assert!(
            (result.point[0] - 0.5).abs() < 2e-3 && (result.point[1] - 0.5).abs() < 2e-3,
            "{result:?}"
        );
        assert!(result.maximum_violation < 1e-6);
    }
    #[test]
    fn solves_active_inequality_from_infeasible_start() {
        let kinds = [DerivativeFreeConstraintKind::LessEqual];
        let result =
            minimize_constrained_derivative_free(&[0.], None, &kinds, Default::default(), |x| {
                Ok(ConstrainedDerivativeFreeEvaluation {
                    objective: x[0] * x[0],
                    constraints: vec![1. - x[0]],
                })
            })
            .unwrap();
        assert!((result.point[0] - 1.).abs() < 2e-3);
        assert!(result.maximum_violation < 1e-6 && result.feasibility_improving_steps > 0);
    }
    #[test]
    fn budgets_cancellation_and_malformed_evaluations_are_explicit() {
        let kinds = [DerivativeFreeConstraintKind::LessEqual];
        let options = ConstrainedDerivativeFreeOptions {
            max_evaluations: 3,
            ..Default::default()
        };
        let limited = minimize_constrained_derivative_free(&[0.], None, &kinds, options, |x| {
            Ok(ConstrainedDerivativeFreeEvaluation {
                objective: x[0] * x[0],
                constraints: vec![-1.],
            })
        })
        .unwrap();
        assert_eq!(
            limited.termination,
            ConstrainedDerivativeFreeTermination::EvaluationLimit
        );
        let cancelled = minimize_constrained_derivative_free_with_observer(
            &[0.],
            None,
            &kinds,
            Default::default(),
            |_| {
                Ok(ConstrainedDerivativeFreeEvaluation {
                    objective: 0.,
                    constraints: vec![-1.],
                })
            },
            |_| false,
        )
        .unwrap();
        assert_eq!(
            cancelled.termination,
            ConstrainedDerivativeFreeTermination::Cancelled
        );
        assert!(minimize_constrained_derivative_free(
            &[0.],
            None,
            &kinds,
            Default::default(),
            |_| Ok(ConstrainedDerivativeFreeEvaluation {
                objective: 0.,
                constraints: vec![]
            })
        )
        .is_err());
    }
}
