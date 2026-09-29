//! Projected limited-memory BFGS for box-constrained smooth objectives.

use crate::{
    Bound, LbfgsIteration, LbfgsOptions, LbfgsResult, NonlinearTermination, OptimizeError,
};

/// Minimize a smooth objective subject to independent variable bounds.
///
/// The stopping test uses the infinity norm of the projected gradient.  The
/// line search follows the projected path and uses the actual feasible
/// displacement in its Armijo test.
pub fn minimize_lbfgsb<F>(
    initial: &[f64],
    bounds: &[Bound],
    options: LbfgsOptions,
    evaluate: F,
) -> Result<LbfgsResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
{
    minimize_lbfgsb_with_observer(initial, bounds, options, evaluate, |_| true)
}

pub fn minimize_lbfgsb_with_observer<F, O>(
    initial: &[f64],
    bounds: &[Bound],
    options: LbfgsOptions,
    mut evaluate: F,
    mut observer: O,
) -> Result<LbfgsResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
    O: FnMut(LbfgsIteration) -> bool,
{
    validate(initial, bounds, options)?;
    let mut x: Vec<f64> = initial
        .iter()
        .zip(bounds)
        .map(|(&v, b)| project(v, b))
        .collect();
    let (mut value, mut gradient) = checked(&mut evaluate, &x)?;
    let mut evaluations = 1;
    let (mut ss, mut ys, mut rhos) = (
        Vec::<Vec<f64>>::new(),
        Vec::<Vec<f64>>::new(),
        Vec::<f64>::new(),
    );
    let mut last_step = None;
    for iteration in 0..options.max_iterations {
        let pg = projected_gradient(&x, &gradient, bounds);
        if inf_norm(&pg) <= options.gradient_tolerance {
            return Ok(result(
                x,
                value,
                gradient,
                iteration,
                evaluations,
                NonlinearTermination::ConvergedGradient,
                last_step,
                ss.len(),
                inf_norm(&pg),
            ));
        }
        let mut direction = two_loop(&pg, &ss, &ys, &rhos);
        make_feasible_direction(&x, bounds, &mut direction);
        if dot(&pg, &direction) >= 0.0 || !direction.iter().all(|v| v.is_finite()) {
            ss.clear();
            ys.clear();
            rhos.clear();
            direction = pg.iter().map(|v| -v).collect();
            make_feasible_direction(&x, bounds, &mut direction);
        }
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..options.max_line_search_iterations {
            let candidate: Vec<f64> = x
                .iter()
                .zip(&direction)
                .zip(bounds)
                .map(|((&v, &d), b)| project(v + alpha * d, b))
                .collect();
            let displacement: Vec<f64> = candidate.iter().zip(&x).map(|(a, b)| a - b).collect();
            let slope = dot(&gradient, &displacement);
            if norm(&displacement) > 0.0 && slope < 0.0 {
                evaluations += 1;
                if let Ok((next_value, next_gradient)) = checked(&mut evaluate, &candidate) {
                    if next_value <= value + options.armijo * slope {
                        accepted =
                            Some((candidate, next_value, next_gradient, displacement, alpha));
                        break;
                    }
                }
            }
            alpha *= options.backtracking;
        }
        let Some((next_x, next_value, next_gradient, s, accepted_alpha)) = accepted else {
            let pg_norm = inf_norm(&projected_gradient(&x, &gradient, bounds));
            return Ok(result(
                x,
                value,
                gradient,
                iteration,
                evaluations,
                NonlinearTermination::LineSearchFailed,
                last_step,
                ss.len(),
                pg_norm,
            ));
        };
        let y: Vec<f64> = next_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        let curvature = dot(&s, &y);
        if curvature > 1e-12 * norm(&s) * norm(&y) {
            if ss.len() == options.history_size {
                ss.remove(0);
                ys.remove(0);
                rhos.remove(0);
            }
            ss.push(s.clone());
            ys.push(y);
            rhos.push(1.0 / curvature);
        }
        let step_norm = norm(&s);
        let objective_change = (value - next_value).abs();
        x = next_x;
        value = next_value;
        gradient = next_gradient;
        last_step = Some(accepted_alpha);
        let pg_norm = inf_norm(&projected_gradient(&x, &gradient, bounds));
        if !observer(LbfgsIteration {
            iteration: iteration + 1,
            objective: value,
            gradient_norm: pg_norm,
            step: accepted_alpha,
            evaluations,
        }) {
            return Ok(result(
                x,
                value,
                gradient,
                iteration + 1,
                evaluations,
                NonlinearTermination::Cancelled,
                last_step,
                ss.len(),
                pg_norm,
            ));
        }
        if step_norm <= options.step_tolerance * (1.0 + norm(&x)) {
            return Ok(result(
                x,
                value,
                gradient,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedStep,
                last_step,
                ss.len(),
                pg_norm,
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
                last_step,
                ss.len(),
                pg_norm,
            ));
        }
    }
    let pg_norm = inf_norm(&projected_gradient(&x, &gradient, bounds));
    Ok(result(
        x,
        value,
        gradient,
        options.max_iterations,
        evaluations,
        NonlinearTermination::IterationLimit,
        last_step,
        ss.len(),
        pg_norm,
    ))
}

fn validate(initial: &[f64], bounds: &[Bound], o: LbfgsOptions) -> Result<(), OptimizeError> {
    if initial.is_empty() || initial.len() != bounds.len() || !initial.iter().all(|v| v.is_finite())
    {
        return Err(OptimizeError::InvalidProblem(
            "bounded L-BFGS requires matching non-empty initial point and bounds".into(),
        ));
    }
    if bounds.iter().any(|b| {
        b.lower.is_some_and(|v| !v.is_finite())
            || b.upper.is_some_and(|v| !v.is_finite())
            || matches!((b.lower,b.upper),(Some(l),Some(u)) if l>u)
    }) {
        return Err(OptimizeError::InvalidProblem(
            "bounds must be finite when present and lower must not exceed upper".into(),
        ));
    }
    if o.max_iterations == 0
        || o.history_size == 0
        || o.max_line_search_iterations == 0
        || !(o.gradient_tolerance > 0.0 && o.gradient_tolerance.is_finite())
        || !(o.step_tolerance > 0.0 && o.step_tolerance.is_finite())
        || !(o.objective_tolerance > 0.0 && o.objective_tolerance.is_finite())
        || !(0.0 < o.armijo && o.armijo < o.wolfe && o.wolfe < 1.0)
        || !(0.0 < o.backtracking && o.backtracking < 1.0)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid L-BFGS options",
        ));
    }
    Ok(())
}
fn checked<F>(f: &mut F, x: &[f64]) -> Result<(f64, Vec<f64>), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
{
    let (v, g) = f(x)?;
    if !v.is_finite() || g.len() != x.len() || !g.iter().all(|v| v.is_finite()) {
        return Err(OptimizeError::InvalidProblem(
            "objective must return a finite value and matching finite gradient".into(),
        ));
    }
    Ok((v, g))
}
fn project(v: f64, b: &Bound) -> f64 {
    b.lower
        .map_or(v, |l| v.max(l))
        .min(b.upper.unwrap_or(f64::INFINITY))
}
fn projected_gradient(x: &[f64], g: &[f64], b: &[Bound]) -> Vec<f64> {
    x.iter()
        .zip(g)
        .zip(b)
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
fn make_feasible_direction(x: &[f64], b: &[Bound], d: &mut [f64]) {
    for ((&x, b), d) in x.iter().zip(b).zip(d) {
        if (b.lower.is_some_and(|l| x <= l) && *d < 0.0)
            || (b.upper.is_some_and(|u| x >= u) && *d > 0.0)
        {
            *d = 0.0
        }
    }
}
fn two_loop(g: &[f64], ss: &[Vec<f64>], ys: &[Vec<f64>], rhos: &[f64]) -> Vec<f64> {
    let mut q = g.to_vec();
    let mut a = vec![0.0; ss.len()];
    for i in (0..ss.len()).rev() {
        a[i] = rhos[i] * dot(&ss[i], &q);
        for j in 0..q.len() {
            q[j] -= a[i] * ys[i][j]
        }
    }
    if let (Some(s), Some(y)) = (ss.last(), ys.last()) {
        let scale = dot(s, y) / dot(y, y);
        for v in &mut q {
            *v *= scale
        }
    }
    for i in 0..ss.len() {
        let beta = rhos[i] * dot(&ys[i], &q);
        for j in 0..q.len() {
            q[j] += (a[i] - beta) * ss[i][j]
        }
    }
    q.iter().map(|v| -v).collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}
fn result(
    point: Vec<f64>,
    objective: f64,
    gradient: Vec<f64>,
    iterations: usize,
    evaluations: usize,
    termination: NonlinearTermination,
    accepted_step: Option<f64>,
    stored_curvature_pairs: usize,
    projected_norm: f64,
) -> LbfgsResult {
    LbfgsResult {
        point,
        objective,
        gradient,
        iterations,
        evaluations,
        termination,
        gradient_norm: projected_norm,
        accepted_step,
        stored_curvature_pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finds_solution_on_two_active_bounds() {
        let bounds = [
            Bound {
                lower: None,
                upper: Some(1.0),
            },
            Bound {
                lower: Some(-1.0),
                upper: None,
            },
        ];
        let r = minimize_lbfgsb(&[0.0, 0.0], &bounds, Default::default(), |x| {
            Ok((
                (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2),
                vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)],
            ))
        })
        .unwrap();
        assert!((r.point[0] - 1.0).abs() < 1e-10);
        assert!((r.point[1] + 1.0).abs() < 1e-10);
        assert!(r.gradient_norm < 1e-10);
    }
}
