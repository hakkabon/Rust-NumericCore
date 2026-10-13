//! Advanced unconstrained methods: matrix-free trust-region Newton-CG and
//! Powell's derivative-free direction-set method.

use crate::{NonlinearTermination, OptimizeError};

#[derive(Debug, Clone, Copy)]
pub struct NewtonCgOptions {
    pub max_iterations: usize,
    pub max_cg_iterations: usize,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub initial_radius: f64,
    pub maximum_radius: f64,
    pub acceptance_ratio: f64,
}
impl Default for NewtonCgOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            max_cg_iterations: 100,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-12,
            initial_radius: 1.0,
            maximum_radius: 1e4,
            acceptance_ratio: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewtonCgResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub hessian_products: usize,
    pub accepted_steps: usize,
    pub final_radius: f64,
    pub termination: NonlinearTermination,
}

pub fn minimize_newton_cg<F, H>(
    initial: &[f64],
    options: NewtonCgOptions,
    mut evaluate: F,
    mut hessian_product: H,
) -> Result<NewtonCgResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
    H: FnMut(&[f64], &[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    validate_newton(initial, options)?;
    let mut point = initial.to_vec();
    let (mut value, mut gradient) = checked_evaluation(&mut evaluate, &point)?;
    let mut evaluations = 1;
    let mut products = 0;
    let mut accepted = 0;
    let mut radius = options.initial_radius;
    for iteration in 0..options.max_iterations {
        if norm_inf(&gradient) <= options.gradient_tolerance {
            return Ok(newton_result(
                point,
                value,
                gradient,
                iteration,
                evaluations,
                products,
                accepted,
                radius,
                NonlinearTermination::ConvergedGradient,
            ));
        }
        let (step, hp, used) = steihaug(
            &point,
            &gradient,
            radius,
            options.max_cg_iterations,
            &mut hessian_product,
        )?;
        products += used;
        let step_norm = norm(&step);
        if step_norm <= options.step_tolerance {
            return Ok(newton_result(
                point,
                value,
                gradient,
                iteration,
                evaluations,
                products,
                accepted,
                radius,
                NonlinearTermination::ConvergedStep,
            ));
        }
        let predicted = -(dot(&gradient, &step) + 0.5 * dot(&step, &hp));
        if !predicted.is_finite() || predicted <= 0.0 {
            radius *= 0.25;
            continue;
        }
        let candidate: Vec<_> = point.iter().zip(&step).map(|(x, s)| x + s).collect();
        let next = checked_evaluation(&mut evaluate, &candidate)?;
        evaluations += 1;
        let ratio = (value - next.0) / predicted;
        if ratio < 0.25 {
            radius *= 0.25;
        } else if ratio > 0.75 && step_norm >= 0.9 * radius {
            radius = (2.0 * radius).min(options.maximum_radius);
        }
        if ratio > options.acceptance_ratio {
            point = candidate;
            value = next.0;
            gradient = next.1;
            accepted += 1;
        }
    }
    Ok(newton_result(
        point,
        value,
        gradient,
        options.max_iterations,
        evaluations,
        products,
        accepted,
        radius,
        NonlinearTermination::IterationLimit,
    ))
}

fn steihaug<H>(
    point: &[f64],
    gradient: &[f64],
    radius: f64,
    limit: usize,
    hessian_product: &mut H,
) -> Result<(Vec<f64>, Vec<f64>, usize), OptimizeError>
where
    H: FnMut(&[f64], &[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    let n = gradient.len();
    let mut step = vec![0.0; n];
    let mut residual: Vec<_> = gradient.iter().map(|x| -x).collect();
    let mut direction = residual.clone();
    let mut rr = dot(&residual, &residual);
    let target = (rr.sqrt() * 0.1).min(0.5 * rr.sqrt());
    let mut used = 0;
    for _ in 0..limit {
        if rr.sqrt() <= target {
            break;
        }
        let hd = hessian_product(point, &direction)?;
        used += 1;
        check_vector(&hd, n, "Hessian-vector product")?;
        let curvature = dot(&direction, &hd);
        if curvature <= 0.0 {
            let tau = boundary_tau(&step, &direction, radius);
            axpy(&mut step, tau, &direction);
            break;
        }
        let alpha = rr / curvature;
        let mut trial = step.clone();
        axpy(&mut trial, alpha, &direction);
        if norm(&trial) >= radius {
            let tau = boundary_tau(&step, &direction, radius);
            axpy(&mut step, tau, &direction);
            break;
        }
        step = trial;
        axpy(&mut residual, -alpha, &hd);
        let next_rr = dot(&residual, &residual);
        if next_rr.sqrt() <= target {
            break;
        }
        let beta = next_rr / rr;
        for i in 0..n {
            direction[i] = residual[i] + beta * direction[i];
        }
        rr = next_rr;
    }
    let hp = hessian_product(point, &step)?;
    used += 1;
    check_vector(&hp, n, "Hessian-vector product")?;
    Ok((step, hp, used))
}

#[derive(Debug, Clone, Copy)]
pub struct PowellOptions {
    pub max_iterations: usize,
    pub max_line_iterations: usize,
    pub step_tolerance: f64,
    pub objective_tolerance: f64,
    pub initial_step: f64,
}
impl Default for PowellOptions {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            max_line_iterations: 80,
            step_tolerance: 1e-8,
            objective_tolerance: 1e-10,
            initial_step: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PowellResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: NonlinearTermination,
}

pub fn minimize_powell<F>(
    initial: &[f64],
    options: PowellOptions,
    mut evaluate: F,
) -> Result<PowellResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    if initial.is_empty()
        || !initial.iter().all(|x| x.is_finite())
        || options.max_iterations == 0
        || options.max_line_iterations == 0
        || !options.step_tolerance.is_finite()
        || options.step_tolerance <= 0.0
        || !options.objective_tolerance.is_finite()
        || options.objective_tolerance <= 0.0
        || !options.initial_step.is_finite()
        || options.initial_step <= 0.0
    {
        return Err(invalid("invalid Powell direction-set problem or options"));
    }
    let n = initial.len();
    let mut point = initial.to_vec();
    let mut value = checked_scalar(&mut evaluate, &point)?;
    let mut evaluations = 1;
    let mut directions: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut d = vec![0.0; n];
            d[i] = options.initial_step;
            d
        })
        .collect();
    for iteration in 0..options.max_iterations {
        let start = point.clone();
        let start_value = value;
        let mut largest = 0.0;
        let mut largest_index = 0;
        for (index, direction) in directions.iter().enumerate() {
            let before = value;
            let line = line_minimize(&point, direction, value, options, &mut evaluate)?;
            point = line.0;
            value = line.1;
            evaluations += line.2;
            if before - value > largest {
                largest = before - value;
                largest_index = index;
            }
        }
        let displacement: Vec<_> = point.iter().zip(&start).map(|(a, b)| a - b).collect();
        let sweep_step = norm(&displacement);
        let line = line_minimize(&point, &displacement, value, options, &mut evaluate)?;
        point = line.0;
        value = line.1;
        evaluations += line.2;
        directions[largest_index] = displacement;
        if sweep_step <= options.step_tolerance {
            return Ok(PowellResult {
                point,
                objective: value,
                iterations: iteration + 1,
                evaluations,
                termination: NonlinearTermination::ConvergedStep,
            });
        }
        if 2.0 * (start_value - value).abs()
            <= options.objective_tolerance * (start_value.abs() + value.abs() + 1.0)
        {
            return Ok(PowellResult {
                point,
                objective: value,
                iterations: iteration + 1,
                evaluations,
                termination: NonlinearTermination::ConvergedObjective,
            });
        }
    }
    Ok(PowellResult {
        point,
        objective: value,
        iterations: options.max_iterations,
        evaluations,
        termination: NonlinearTermination::IterationLimit,
    })
}

fn line_minimize<F>(
    point: &[f64],
    direction: &[f64],
    center_value: f64,
    options: PowellOptions,
    evaluate: &mut F,
) -> Result<(Vec<f64>, f64, usize), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    if norm(direction) == 0.0 {
        return Ok((point.to_vec(), center_value, 0));
    }
    let calls = std::cell::Cell::new(0usize);
    let mut eval = |alpha: f64| -> Result<f64, OptimizeError> {
        let x: Vec<_> = point
            .iter()
            .zip(direction)
            .map(|(p, d)| p + alpha * d)
            .collect();
        calls.set(calls.get() + 1);
        checked_scalar(evaluate, &x)
    };
    let plus = eval(1.0)?;
    let minus = eval(-1.0)?;
    let (mut lo, mut hi) = if plus >= center_value && minus >= center_value {
        (-1.0, 1.0)
    } else {
        let sign = if plus < minus { 1.0 } else { -1.0 };
        let mut a = 0.0;
        let mut b = sign;
        let mut fb = if sign > 0.0 { plus } else { minus };
        let mut c = b + sign * 1.61803398875;
        let mut fc = eval(c)?;
        while fc < fb && calls.get() < options.max_line_iterations / 2 {
            a = b;
            b = c;
            fb = fc;
            c = b + sign * (b - a).abs() * 1.61803398875;
            fc = eval(c)?;
        }
        if a < c {
            (a, c)
        } else {
            (c, a)
        }
    };
    let phi = 0.6180339887498949;
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let mut f1 = eval(x1)?;
    let mut f2 = eval(x2)?;
    while calls.get() < options.max_line_iterations && (hi - lo).abs() > options.step_tolerance {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            f1 = eval(x1)?;
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            f2 = eval(x2)?;
        }
    }
    let (alpha, best) = if f1 < f2 { (x1, f1) } else { (x2, f2) };
    let result = point
        .iter()
        .zip(direction)
        .map(|(p, d)| p + alpha * d)
        .collect();
    Ok((result, best, calls.get()))
}

fn checked_evaluation<F>(f: &mut F, x: &[f64]) -> Result<(f64, Vec<f64>), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptimizeError>,
{
    let result = f(x)?;
    if !result.0.is_finite() {
        return Err(invalid("non-finite objective"));
    }
    check_vector(&result.1, x.len(), "gradient")?;
    Ok(result)
}
fn checked_scalar<F>(f: &mut F, x: &[f64]) -> Result<f64, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    let value = f(x)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("non-finite objective"))
    }
}
fn check_vector(x: &[f64], n: usize, name: &str) -> Result<(), OptimizeError> {
    if x.len() == n && x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(invalid(&format!("invalid {name}")))
    }
}
fn validate_newton(x: &[f64], o: NewtonCgOptions) -> Result<(), OptimizeError> {
    if x.is_empty()
        || !x.iter().all(|v| v.is_finite())
        || o.max_iterations == 0
        || o.max_cg_iterations == 0
        || !o.gradient_tolerance.is_finite()
        || o.gradient_tolerance <= 0.0
        || !o.step_tolerance.is_finite()
        || o.step_tolerance <= 0.0
        || !o.initial_radius.is_finite()
        || o.initial_radius <= 0.0
        || !o.maximum_radius.is_finite()
        || o.maximum_radius < o.initial_radius
        || !o.acceptance_ratio.is_finite()
        || !(0.0..0.25).contains(&o.acceptance_ratio)
    {
        Err(invalid("invalid trust-region Newton-CG problem or options"))
    } else {
        Ok(())
    }
}
fn boundary_tau(p: &[f64], d: &[f64], radius: f64) -> f64 {
    let a = dot(d, d);
    let b = 2.0 * dot(p, d);
    let c = dot(p, p) - radius * radius;
    (-b + (b * b - 4.0 * a * c).max(0.0).sqrt()) / (2.0 * a)
}
fn newton_result(
    point: Vec<f64>,
    objective: f64,
    gradient: Vec<f64>,
    iterations: usize,
    evaluations: usize,
    hessian_products: usize,
    accepted_steps: usize,
    final_radius: f64,
    termination: NonlinearTermination,
) -> NewtonCgResult {
    NewtonCgResult {
        point,
        objective,
        gradient,
        iterations,
        evaluations,
        hessian_products,
        accepted_steps,
        final_radius,
        termination,
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn norm_inf(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0, f64::max)
}
fn axpy(x: &mut [f64], alpha: f64, y: &[f64]) {
    for (x, y) in x.iter_mut().zip(y) {
        *x += alpha * y;
    }
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn newton_cg_solves_matrix_free_quadratic() {
        let result = minimize_newton_cg(
            &[8.0, -5.0],
            NewtonCgOptions::default(),
            |x| {
                Ok((
                    0.5 * (4.0 * x[0] * x[0] + x[1] * x[1]) - 8.0 * x[0] + 3.0 * x[1],
                    vec![4.0 * x[0] - 8.0, x[1] + 3.0],
                ))
            },
            |_, v| Ok(vec![4.0 * v[0], v[1]]),
        )
        .unwrap();
        assert_eq!(result.termination, NonlinearTermination::ConvergedGradient);
        assert!((result.point[0] - 2.0).abs() < 1e-7 && (result.point[1] + 3.0).abs() < 1e-7);
        assert!(result.hessian_products > 0);
    }
    #[test]
    fn powell_solves_rotated_derivative_free_quadratic() {
        let result = minimize_powell(&[6.0, -5.0], PowellOptions::default(), |x| {
            let a = x[0] + x[1] - 1.0;
            let b = 2.0 * x[0] - x[1] - 5.0;
            Ok(a * a + b * b)
        })
        .unwrap();
        assert!(
            (result.point[0] - 2.0).abs() < 1e-4 && (result.point[1] + 1.0).abs() < 1e-4,
            "{:?}",
            result.point
        );
        assert!(result.objective < 1e-8);
    }
}
