//! Limited-memory BFGS for smooth unconstrained objectives.

use crate::OptimizeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearTermination {
    ConvergedGradient,
    ConvergedStep,
    ConvergedObjective,
    IterationLimit,
    LineSearchFailed,
    DampingLimit,
}

#[derive(Debug, Clone, Copy)]
pub struct LbfgsOptions {
    pub max_iterations: usize,
    pub history_size: usize,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub objective_tolerance: f64,
    pub max_line_search_iterations: usize,
    pub armijo: f64,
    pub backtracking: f64,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            history_size: 10,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-12,
            objective_tolerance: 1e-12,
            max_line_search_iterations: 30,
            armijo: 1e-4,
            backtracking: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LbfgsResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: NonlinearTermination,
}

pub fn minimize_lbfgs<F>(
    initial: &[f64],
    options: LbfgsOptions,
    mut evaluate: F,
) -> Result<LbfgsResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
{
    validate_options(options)?;
    if initial.is_empty() || !initial.iter().all(|v| v.is_finite()) {
        return Err(OptimizeError::InvalidProblem(
            "L-BFGS requires a non-empty finite initial point".to_owned(),
        ));
    }
    let mut x = initial.to_vec();
    let (mut value, mut gradient) = evaluate_checked(&mut evaluate, &x)?;
    let mut evaluations = 1;
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();

    for iteration in 0..options.max_iterations {
        if infinity_norm(&gradient) <= options.gradient_tolerance {
            return Ok(result(
                x,
                value,
                gradient,
                iteration,
                evaluations,
                NonlinearTermination::ConvergedGradient,
            ));
        }

        let mut direction = two_loop_direction(&gradient, &s_history, &y_history, &rho_history);
        let mut directional_derivative = dot(&gradient, &direction);
        if !directional_derivative.is_finite() || directional_derivative >= 0.0 {
            s_history.clear();
            y_history.clear();
            rho_history.clear();
            direction = gradient.iter().map(|v| -v).collect();
            directional_derivative = -dot(&gradient, &gradient);
        }

        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..options.max_line_search_iterations {
            let candidate: Vec<f64> = x
                .iter()
                .zip(&direction)
                .map(|(xj, dj)| xj + step * dj)
                .collect();
            if candidate.iter().all(|v| v.is_finite()) {
                let evaluated = evaluate_checked(&mut evaluate, &candidate);
                evaluations += 1;
                if let Ok((candidate_value, candidate_gradient)) = evaluated {
                    if candidate_value <= value + options.armijo * step * directional_derivative {
                        accepted = Some((candidate, candidate_value, candidate_gradient, step));
                        break;
                    }
                }
            }
            step *= options.backtracking;
        }
        let Some((next_x, next_value, next_gradient, accepted_step)) = accepted else {
            return Ok(result(
                x,
                value,
                gradient,
                iteration,
                evaluations,
                NonlinearTermination::LineSearchFailed,
            ));
        };

        let s: Vec<f64> = next_x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = next_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        let curvature = dot(&s, &y);
        if curvature > 1e-12 * euclidean_norm(&s) * euclidean_norm(&y) {
            if s_history.len() == options.history_size {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
            rho_history.push(1.0 / curvature);
        }

        let step_norm = accepted_step * euclidean_norm(&direction);
        let objective_change = (value - next_value).abs();
        x = next_x;
        value = next_value;
        gradient = next_gradient;
        if step_norm <= options.step_tolerance * (1.0 + euclidean_norm(&x)) {
            return Ok(result(
                x,
                value,
                gradient,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedStep,
            ));
        }
        if objective_change <= options.objective_tolerance * (1.0 + value.abs()) {
            return Ok(result(
                x,
                value,
                gradient,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedObjective,
            ));
        }
    }
    Ok(result(
        x,
        value,
        gradient,
        options.max_iterations,
        evaluations,
        NonlinearTermination::IterationLimit,
    ))
}

fn validate_options(options: LbfgsOptions) -> Result<(), OptimizeError> {
    if options.max_iterations == 0
        || options.history_size == 0
        || options.max_line_search_iterations == 0
        || !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.0
        || !options.step_tolerance.is_finite()
        || options.step_tolerance <= 0.0
        || !options.objective_tolerance.is_finite()
        || options.objective_tolerance <= 0.0
        || !options.armijo.is_finite()
        || !(0.0..1.0).contains(&options.armijo)
        || !options.backtracking.is_finite()
        || !(0.0..1.0).contains(&options.backtracking)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid L-BFGS options",
        ));
    }
    Ok(())
}

fn evaluate_checked<F>(evaluate: &mut F, x: &[f64]) -> Result<(f64, Vec<f64>), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
{
    let (value, gradient) = evaluate(x)?;
    if !value.is_finite() || gradient.len() != x.len() || !gradient.iter().all(|v| v.is_finite()) {
        return Err(OptimizeError::InvalidProblem(
            "objective must return a finite value and finite gradient matching the point"
                .to_owned(),
        ));
    }
    Ok((value, gradient))
}

fn two_loop_direction(g: &[f64], s: &[Vec<f64>], y: &[Vec<f64>], rho: &[f64]) -> Vec<f64> {
    let mut q = g.to_vec();
    let mut alpha = vec![0.0; s.len()];
    for i in (0..s.len()).rev() {
        alpha[i] = rho[i] * dot(&s[i], &q);
        add_scaled(&mut q, &y[i], -alpha[i]);
    }
    if let (Some(last_s), Some(last_y)) = (s.last(), y.last()) {
        let scale = dot(last_s, last_y) / dot(last_y, last_y);
        for value in &mut q {
            *value *= scale;
        }
    }
    for i in 0..s.len() {
        let beta = rho[i] * dot(&y[i], &q);
        add_scaled(&mut q, &s[i], alpha[i] - beta);
    }
    q.into_iter().map(|v| -v).collect()
}

fn result(
    point: Vec<f64>,
    objective: f64,
    gradient: Vec<f64>,
    iterations: usize,
    evaluations: usize,
    termination: NonlinearTermination,
) -> LbfgsResult {
    LbfgsResult {
        point,
        objective,
        gradient,
        iterations,
        evaluations,
        termination,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn euclidean_norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0, f64::max)
}
fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += scale * source;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_rosenbrock() {
        let result = minimize_lbfgs(&[-1.2, 1.0], LbfgsOptions::default(), |x| {
            let a = x[1] - x[0] * x[0];
            let b = 1.0 - x[0];
            Ok((
                100.0 * a * a + b * b,
                vec![-400.0 * x[0] * a - 2.0 * b, 200.0 * a],
            ))
        })
        .unwrap();
        assert!(matches!(
            result.termination,
            NonlinearTermination::ConvergedGradient
                | NonlinearTermination::ConvergedStep
                | NonlinearTermination::ConvergedObjective
        ));
        assert!((result.point[0] - 1.0).abs() < 1e-5);
        assert!((result.point[1] - 1.0).abs() < 1e-5);
        assert!(result.objective < 1e-12);
    }

    #[test]
    fn rejects_gradient_dimension_mismatch() {
        let error = minimize_lbfgs(&[0.0], LbfgsOptions::default(), |_| Ok((0.0, vec![])));
        assert!(matches!(error, Err(OptimizeError::InvalidProblem(_))));
    }
}
