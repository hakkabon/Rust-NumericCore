//! Levenberg-Marquardt nonlinear least squares with an analytic Jacobian.

use crate::{Bound, NonlinearTermination, OptimizeError};
use nc_decomp::least_squares_qr;

#[derive(Debug, Clone, Copy)]
pub struct NonlinearLeastSquaresOptions {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
    pub initial_damping: f64,
    pub damping_increase: f64,
    pub damping_decrease: f64,
    pub max_damping_iterations: usize,
}

impl Default for NonlinearLeastSquaresOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-10,
            cost_tolerance: 1e-12,
            initial_damping: 1e-3,
            damping_increase: 10.0,
            damping_decrease: 0.3,
            max_damping_iterations: 20,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearLeastSquaresResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub cost: f64,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: NonlinearTermination,
    pub final_damping: f64,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct NonlinearLeastSquaresIteration {
    pub iteration: usize,
    pub cost: f64,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub damping: f64,
    pub evaluations: usize,
}

/// Loss applied independently to each weighted residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustLoss {
    Squared,
    Huber { scale: f64 },
    Cauchy { scale: f64 },
}

/// `evaluate` returns residuals and a row-major Jacobian (`m` rows by `n`
/// parameters). The objective is `0.5 * ||residuals||²`.
pub fn nonlinear_least_squares<F>(
    initial: &[f64],
    options: NonlinearLeastSquaresOptions,
    evaluate: F,
) -> Result<NonlinearLeastSquaresResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>,
{
    let bounds = vec![Bound::free(); initial.len()];
    nonlinear_least_squares_configured_with_observer(
        initial,
        &bounds,
        &[],
        RobustLoss::Squared,
        options,
        evaluate,
        |_| true,
    )
}

pub fn nonlinear_least_squares_with_observer<F, O>(
    initial: &[f64],
    options: NonlinearLeastSquaresOptions,
    evaluate: F,
    observer: O,
) -> Result<NonlinearLeastSquaresResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>,
    O: FnMut(NonlinearLeastSquaresIteration) -> bool,
{
    let bounds = vec![Bound::free(); initial.len()];
    nonlinear_least_squares_configured_with_observer(
        initial,
        &bounds,
        &[],
        RobustLoss::Squared,
        options,
        evaluate,
        observer,
    )
}

/// Weighted robust nonlinear least squares with optional box constraints.
/// Empty `weights` means unit weights; otherwise there must be one finite,
/// non-negative weight per residual.
pub fn nonlinear_least_squares_configured<F>(
    initial: &[f64],
    bounds: &[Bound],
    weights: &[f64],
    loss: RobustLoss,
    options: NonlinearLeastSquaresOptions,
    evaluate: F,
) -> Result<NonlinearLeastSquaresResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>,
{
    nonlinear_least_squares_configured_with_observer(
        initial,
        bounds,
        weights,
        loss,
        options,
        evaluate,
        |_| true,
    )
}

pub fn nonlinear_least_squares_configured_with_observer<F, O>(
    initial: &[f64],
    bounds: &[Bound],
    weights: &[f64],
    loss: RobustLoss,
    options: NonlinearLeastSquaresOptions,
    mut evaluate: F,
    mut observer: O,
) -> Result<NonlinearLeastSquaresResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>,
    O: FnMut(NonlinearLeastSquaresIteration) -> bool,
{
    validate_options(options)?;
    if initial.is_empty() || !initial.iter().all(|v| v.is_finite()) {
        return Err(OptimizeError::InvalidProblem(
            "nonlinear least squares requires a non-empty finite initial point".to_owned(),
        ));
    }
    validate_configuration(initial.len(), bounds, loss)?;
    let mut point: Vec<f64> = initial
        .iter()
        .zip(bounds)
        .map(|(&x, b)| project(x, b))
        .collect();
    let (mut residuals, mut jacobian) = evaluate_checked(&mut evaluate, &point)?;
    validate_weights(weights, residuals.len())?;
    let mut evaluations = 1;
    let mut cost = robust_cost(&residuals, weights, loss);
    let mut damping = options.initial_damping;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;

    for iteration in 0..options.max_iterations {
        let (scaled_residuals, scaled_jacobian) =
            scaled_problem(&residuals, &jacobian, weights, loss);
        let (normal, gradient) = normal_equations(&scaled_jacobian, &scaled_residuals, point.len());
        let gradient_norm = infinity_norm(&projected_gradient(&point, &gradient, bounds));
        if gradient_norm <= options.gradient_tolerance {
            return Ok(result(
                point,
                residuals,
                cost,
                gradient_norm,
                iteration,
                evaluations,
                NonlinearTermination::ConvergedGradient,
                damping,
                accepted_steps,
                rejected_steps,
            ));
        }

        let mut accepted = None;
        for _ in 0..options.max_damping_iterations {
            let (augmented, rhs) =
                damped_least_squares_system(&scaled_jacobian, &scaled_residuals, &normal, damping);
            if let Some(step) = least_squares_qr(&augmented, &rhs, 1e-12) {
                let candidate: Vec<f64> = point
                    .iter()
                    .zip(&step)
                    .zip(bounds)
                    .map(|((x, delta), bound)| project(x + delta, bound))
                    .collect();
                let actual_step: Vec<f64> =
                    candidate.iter().zip(&point).map(|(a, b)| a - b).collect();
                if candidate.iter().all(|v| v.is_finite()) {
                    let evaluated = evaluate_checked(&mut evaluate, &candidate);
                    evaluations += 1;
                    if let Ok((next_residuals, next_jacobian)) = evaluated {
                        if next_residuals.len() != residuals.len() {
                            return Err(OptimizeError::InvalidProblem(
                                "residual count must remain constant".into(),
                            ));
                        }
                        let next_cost = robust_cost(&next_residuals, weights, loss);
                        let predicted = predicted_reduction(&gradient, &normal, &actual_step);
                        let gain_ratio = if predicted > 0.0 {
                            (cost - next_cost) / predicted
                        } else {
                            f64::NEG_INFINITY
                        };
                        if gain_ratio > 0.0 && next_cost < cost {
                            accepted = Some((
                                candidate,
                                next_residuals,
                                next_jacobian,
                                next_cost,
                                euclidean_norm(&actual_step),
                            ));
                            if gain_ratio > 0.75 {
                                damping =
                                    (damping * options.damping_decrease).max(f64::MIN_POSITIVE);
                            } else if gain_ratio < 0.25 {
                                damping *= options.damping_increase;
                            }
                            accepted_steps += 1;
                            break;
                        }
                    }
                }
            }
            damping *= options.damping_increase;
            rejected_steps += 1;
            if !damping.is_finite() {
                break;
            }
        }

        let Some((next_point, next_residuals, next_jacobian, next_cost, step_norm)) = accepted
        else {
            return Ok(result(
                point,
                residuals,
                cost,
                gradient_norm,
                iteration,
                evaluations,
                NonlinearTermination::DampingLimit,
                damping,
                accepted_steps,
                rejected_steps,
            ));
        };
        let cost_change = cost - next_cost;
        point = next_point;
        residuals = next_residuals;
        jacobian = next_jacobian;
        cost = next_cost;
        let (sr, sj) = scaled_problem(&residuals, &jacobian, weights, loss);
        let (_, observed_gradient) = normal_equations(&sj, &sr, point.len());
        if !observer(NonlinearLeastSquaresIteration {
            iteration: iteration + 1,
            cost,
            gradient_norm: infinity_norm(&projected_gradient(&point, &observed_gradient, bounds)),
            step_norm,
            damping,
            evaluations,
        }) {
            let projected_norm =
                infinity_norm(&projected_gradient(&point, &observed_gradient, bounds));
            return Ok(result(
                point,
                residuals,
                cost,
                projected_norm,
                iteration + 1,
                evaluations,
                NonlinearTermination::Cancelled,
                damping,
                accepted_steps,
                rejected_steps,
            ));
        }
        if step_norm <= options.step_tolerance * (1.0 + euclidean_norm(&point)) {
            let (sr, sj) = scaled_problem(&residuals, &jacobian, weights, loss);
            let (_, gradient) = normal_equations(&sj, &sr, point.len());
            let projected_norm = infinity_norm(&projected_gradient(&point, &gradient, bounds));
            return Ok(result(
                point,
                residuals,
                cost,
                projected_norm,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedStep,
                damping,
                accepted_steps,
                rejected_steps,
            ));
        }
        if cost_change <= options.cost_tolerance * (1.0 + cost) {
            let (sr, sj) = scaled_problem(&residuals, &jacobian, weights, loss);
            let (_, gradient) = normal_equations(&sj, &sr, point.len());
            let projected_norm = infinity_norm(&projected_gradient(&point, &gradient, bounds));
            return Ok(result(
                point,
                residuals,
                cost,
                projected_norm,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedObjective,
                damping,
                accepted_steps,
                rejected_steps,
            ));
        }
    }
    let (sr, sj) = scaled_problem(&residuals, &jacobian, weights, loss);
    let (_, gradient) = normal_equations(&sj, &sr, point.len());
    let projected_norm = infinity_norm(&projected_gradient(&point, &gradient, bounds));
    Ok(result(
        point,
        residuals,
        cost,
        projected_norm,
        options.max_iterations,
        evaluations,
        NonlinearTermination::IterationLimit,
        damping,
        accepted_steps,
        rejected_steps,
    ))
}

fn validate_configuration(
    n: usize,
    bounds: &[Bound],
    loss: RobustLoss,
) -> Result<(), OptimizeError> {
    if bounds.len() != n
        || bounds.iter().any(|b| {
            b.lower.is_some_and(|v| !v.is_finite())
                || b.upper.is_some_and(|v| !v.is_finite())
                || matches!((b.lower,b.upper),(Some(l),Some(u)) if l>u)
        })
    {
        return Err(OptimizeError::InvalidProblem(
            "parameter bounds must match the point and be valid".into(),
        ));
    }
    let scale = match loss {
        RobustLoss::Squared => return Ok(()),
        RobustLoss::Huber { scale } | RobustLoss::Cauchy { scale } => scale,
    };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(OptimizeError::InvalidConfiguration(
            "robust loss scale must be finite and positive",
        ));
    }
    Ok(())
}
fn validate_weights(weights: &[f64], m: usize) -> Result<(), OptimizeError> {
    if !weights.is_empty()
        && (weights.len() != m || weights.iter().any(|w| !w.is_finite() || *w < 0.0))
    {
        return Err(OptimizeError::InvalidProblem(
            "weights must be empty or one finite non-negative value per residual".into(),
        ));
    }
    Ok(())
}
fn project(x: f64, b: &Bound) -> f64 {
    b.lower
        .map_or(x, |l| x.max(l))
        .min(b.upper.unwrap_or(f64::INFINITY))
}
fn projected_gradient(x: &[f64], g: &[f64], bounds: &[Bound]) -> Vec<f64> {
    x.iter()
        .zip(g)
        .zip(bounds)
        .map(|((&x, &g), b)| {
            if (b.lower.is_some_and(|l| x <= l) && g > 0.0)
                || (b.upper.is_some_and(|u| x >= u) && g < 0.0)
            {
                0.0
            } else {
                g
            }
        })
        .collect()
}
fn observation_weight(weights: &[f64], i: usize) -> f64 {
    if weights.is_empty() {
        1.0
    } else {
        weights[i]
    }
}
fn robust_cost(r: &[f64], w: &[f64], loss: RobustLoss) -> f64 {
    r.iter()
        .enumerate()
        .map(|(i, &v)| {
            let rho = match loss {
                RobustLoss::Squared => 0.5 * v * v,
                RobustLoss::Huber { scale } => {
                    if v.abs() <= scale {
                        0.5 * v * v
                    } else {
                        scale * (v.abs() - 0.5 * scale)
                    }
                }
                RobustLoss::Cauchy { scale } => {
                    0.5 * scale * scale * (1.0 + (v / scale).powi(2)).ln()
                }
            };
            observation_weight(w, i) * rho
        })
        .sum()
}
fn scaled_problem(
    r: &[f64],
    j: &[Vec<f64>],
    w: &[f64],
    loss: RobustLoss,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut sr = Vec::with_capacity(r.len());
    let mut sj = Vec::with_capacity(r.len());
    for (i, &v) in r.iter().enumerate() {
        let rw = match loss {
            RobustLoss::Squared => 1.0,
            RobustLoss::Huber { scale } => {
                if v.abs() <= scale || v == 0.0 {
                    1.0
                } else {
                    scale / v.abs()
                }
            }
            RobustLoss::Cauchy { scale } => 1.0 / (1.0 + (v / scale).powi(2)),
        };
        let s = (observation_weight(w, i) * rw).sqrt();
        sr.push(s * v);
        sj.push(j[i].iter().map(|x| s * x).collect());
    }
    (sr, sj)
}

fn validate_options(options: NonlinearLeastSquaresOptions) -> Result<(), OptimizeError> {
    if options.max_iterations == 0
        || options.max_damping_iterations == 0
        || !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.0
        || !options.step_tolerance.is_finite()
        || options.step_tolerance <= 0.0
        || !options.cost_tolerance.is_finite()
        || options.cost_tolerance <= 0.0
        || !options.initial_damping.is_finite()
        || options.initial_damping <= 0.0
        || !options.damping_increase.is_finite()
        || options.damping_increase <= 1.0
        || !options.damping_decrease.is_finite()
        || !(0.0..1.0).contains(&options.damping_decrease)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid nonlinear least-squares options",
        ));
    }
    Ok(())
}

fn evaluate_checked<F>(
    evaluate: &mut F,
    point: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError>,
{
    let (residuals, jacobian) = evaluate(point)?;
    if residuals.is_empty()
        || jacobian.len() != residuals.len()
        || jacobian.iter().any(|row| row.len() != point.len())
        || !residuals.iter().all(|v| v.is_finite())
        || !jacobian.iter().flatten().all(|v| v.is_finite())
    {
        return Err(OptimizeError::InvalidProblem(
            "residuals must be non-empty and the finite Jacobian must be m by n".to_owned(),
        ));
    }
    Ok((residuals, jacobian))
}

fn normal_equations(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut normal = vec![vec![0.0; n]; n];
    let mut gradient = vec![0.0; n];
    for (row, residual) in jacobian.iter().zip(residuals) {
        for j in 0..n {
            gradient[j] += row[j] * residual;
            for k in 0..=j {
                normal[j][k] += row[j] * row[k];
            }
        }
    }
    for j in 0..n {
        for k in 0..j {
            normal[k][j] = normal[j][k];
        }
    }
    (normal, gradient)
}

fn result(
    point: Vec<f64>,
    residuals: Vec<f64>,
    cost: f64,
    gradient_norm: f64,
    iterations: usize,
    evaluations: usize,
    termination: NonlinearTermination,
    final_damping: f64,
    accepted_steps: usize,
    rejected_steps: usize,
) -> NonlinearLeastSquaresResult {
    NonlinearLeastSquaresResult {
        point,
        residuals,
        cost,
        gradient_norm,
        iterations,
        evaluations,
        termination,
        final_damping,
        accepted_steps,
        rejected_steps,
    }
}

fn predicted_reduction(gradient: &[f64], normal: &[Vec<f64>], step: &[f64]) -> f64 {
    let linear: f64 = gradient.iter().zip(step).map(|(g, s)| g * s).sum();
    let quadratic: f64 = normal
        .iter()
        .enumerate()
        .map(|(i, row)| 0.5 * step[i] * row.iter().zip(step).map(|(a, s)| a * s).sum::<f64>())
        .sum();
    -linear - quadratic
}

fn damped_least_squares_system(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    normal: &[Vec<f64>],
    damping: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = normal.len();
    let mut matrix = jacobian.to_vec();
    let mut rhs: Vec<f64> = residuals.iter().map(|value| -value).collect();
    for j in 0..n {
        let mut row = vec![0.0; n];
        row[j] = (damping * normal[j][j].abs().max(1.0)).sqrt();
        matrix.push(row);
        rhs.push(0.0);
    }
    (matrix, rhs)
}

fn euclidean_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_an_exponential_curve() {
        let xs = [0.0_f64, 0.5, 1.0, 1.5, 2.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.5 * (-0.7 * x).exp()).collect();
        let result = nonlinear_least_squares(&[1.0, -0.1], Default::default(), |p| {
            let mut residuals = Vec::new();
            let mut jacobian = Vec::new();
            for (&x, &y) in xs.iter().zip(&ys) {
                let exponential = (p[1] * x).exp();
                residuals.push(p[0] * exponential - y);
                jacobian.push(vec![exponential, p[0] * x * exponential]);
            }
            Ok((residuals, jacobian))
        })
        .unwrap();
        assert!((result.point[0] - 2.5).abs() < 1e-6);
        assert!((result.point[1] + 0.7).abs() < 1e-6);
        assert!(result.cost < 1e-18);
        assert!(result.accepted_steps > 0);
        assert!(result.final_damping.is_finite());
    }

    #[test]
    fn rejects_bad_jacobian_shape() {
        let value = nonlinear_least_squares(&[0.0], Default::default(), |_| {
            Ok((vec![1.0], vec![vec![]]))
        });
        assert!(matches!(value, Err(OptimizeError::InvalidProblem(_))));
    }

    #[test]
    fn huber_fit_resists_an_outlier_and_honors_bounds() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.0, 3.0, 5.0, 7.0, 100.0];
        let bounds = [
            Bound {
                lower: Some(0.0),
                upper: Some(1.5),
            },
            Bound::free(),
        ];
        let fit = nonlinear_least_squares_configured(
            &[0.0, 0.0],
            &bounds,
            &[],
            RobustLoss::Huber { scale: 1.0 },
            Default::default(),
            |p| {
                Ok((
                    xs.iter()
                        .zip(ys)
                        .map(|(&x, y)| p[0] + p[1] * x - y)
                        .collect(),
                    xs.iter().map(|&x| vec![1.0, x]).collect(),
                ))
            },
        )
        .unwrap();
        assert!(fit.point[0] >= 0.0 && fit.point[0] <= 1.5);
        assert!((fit.point[0] - 1.0).abs() < 0.51);
        assert!((fit.point[1] - 2.0).abs() < 0.6);
    }

    #[test]
    fn zero_weight_excludes_an_outlier() {
        let bounds = [Bound::free(), Bound::free()];
        let weights = [1.0, 1.0, 1.0, 0.0];
        let fit = nonlinear_least_squares_configured(
            &[0.0, 0.0],
            &bounds,
            &weights,
            RobustLoss::Squared,
            Default::default(),
            |p| {
                let xs = [0.0, 1.0, 2.0, 3.0];
                let ys = [1.0, 3.0, 5.0, 99.0];
                Ok((
                    xs.iter()
                        .zip(ys)
                        .map(|(&x, y)| p[0] + p[1] * x - y)
                        .collect(),
                    xs.iter().map(|&x| vec![1.0, x]).collect(),
                ))
            },
        )
        .unwrap();
        assert!((fit.point[0] - 1.0).abs() < 1e-6 && (fit.point[1] - 2.0).abs() < 1e-6);
    }
}
