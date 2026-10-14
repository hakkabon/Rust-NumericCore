//! Square nonlinear equation systems, Newton-Krylov, and continuation.

use crate::{NonlinearExpression, OptimizeError};

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearSystemEvaluation {
    pub residuals: Vec<f64>,
    pub jacobian: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearSystemTermination {
    ConvergedResidual,
    ConvergedStep,
    IterationLimit,
    LineSearchFailed,
    SingularJacobian,
    KrylovFailed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct NonlinearSystemOptions {
    pub max_iterations: usize,
    pub residual_tolerance: f64,
    pub step_tolerance: f64,
    pub max_line_search_iterations: usize,
    pub armijo: f64,
    pub backtracking: f64,
    pub jacobian_reuse_iterations: usize,
    pub max_krylov_iterations: usize,
    pub krylov_tolerance: f64,
    pub variable_scale: Vec<f64>,
    pub residual_scale: Vec<f64>,
}
impl Default for NonlinearSystemOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            residual_tolerance: 1e-9,
            step_tolerance: 1e-11,
            max_line_search_iterations: 24,
            armijo: 1e-4,
            backtracking: 0.5,
            jacobian_reuse_iterations: 0,
            max_krylov_iterations: 200,
            krylov_tolerance: 1e-8,
            variable_scale: vec![],
            residual_scale: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearSystemIteration {
    pub iteration: usize,
    pub residual_norm: f64,
    pub step_norm: f64,
    pub step_length: f64,
    pub evaluations: usize,
    pub jacobian_evaluations: usize,
}

#[derive(Debug, Clone)]
pub struct NonlinearSystemResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub jacobian_evaluations: usize,
    pub linear_iterations: usize,
    pub accepted_steps: usize,
    pub reused_jacobians: usize,
    pub termination: NonlinearSystemTermination,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearEquationModel {
    pub parameter_count: usize,
    pub equations: Vec<NonlinearExpression>,
}
impl NonlinearEquationModel {
    pub fn new(
        parameter_count: usize,
        equations: Vec<NonlinearExpression>,
    ) -> Result<Self, OptimizeError> {
        if parameter_count == 0 || equations.len() != parameter_count {
            return Err(invalid(
                "a nonlinear equation model must be square and non-empty",
            ));
        }
        for equation in &equations {
            equation.validate(parameter_count)?;
        }
        Ok(Self {
            parameter_count,
            equations,
        })
    }
    pub fn evaluate(&self, parameters: &[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError> {
        if parameters.len() != self.parameter_count {
            return Err(invalid("invalid equation point"));
        }
        let values: Result<Vec<_>, _> = self
            .equations
            .iter()
            .map(|e| e.evaluate(parameters))
            .collect();
        let values = values?;
        Ok(NonlinearSystemEvaluation {
            residuals: values.iter().map(|v| v.value).collect(),
            jacobian: values.into_iter().map(|v| v.gradient).collect(),
        })
    }
}

pub fn solve_equation_model(
    model: &NonlinearEquationModel,
    initial: &[f64],
    options: NonlinearSystemOptions,
) -> Result<NonlinearSystemResult, OptimizeError> {
    solve_nonlinear_system(initial, options, |x| model.evaluate(x))
}

pub fn solve_nonlinear_system<F>(
    initial: &[f64],
    options: NonlinearSystemOptions,
    system: F,
) -> Result<NonlinearSystemResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
{
    solve_nonlinear_system_with_observer(initial, options, system, |_| true)
}

pub fn solve_nonlinear_system_with_observer<F, O>(
    initial: &[f64],
    options: NonlinearSystemOptions,
    mut system: F,
    mut observer: O,
) -> Result<NonlinearSystemResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
    O: FnMut(&NonlinearSystemIteration) -> bool,
{
    let (variable_scale, residual_scale) = validate(initial, &options)?;
    let n = initial.len();
    let mut point = initial.to_vec();
    let mut current = checked_evaluation(&mut system, &point, n)?;
    let mut evaluations = 1;
    let mut jacobians = 1;
    let mut accepted = 0;
    let mut reused = 0;
    let mut cached = current.jacobian.clone();
    let mut age = 0;
    for iteration in 0..options.max_iterations {
        let residual_norm = norm(&scaled(&current.residuals, &residual_scale));
        if residual_norm <= options.residual_tolerance {
            return Ok(result(
                point,
                current.residuals,
                residual_norm,
                iteration,
                evaluations,
                jacobians,
                0,
                accepted,
                reused,
                NonlinearSystemTermination::ConvergedResidual,
            ));
        }
        let use_cached = iteration > 0 && age <= options.jacobian_reuse_iterations;
        if use_cached {
            reused += 1;
        } else {
            cached = current.jacobian.clone();
            age = 0;
        }
        let matrix: Vec<Vec<_>> = cached
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, value)| residual_scale[i] * value * variable_scale[j])
                    .collect()
            })
            .collect();
        let rhs: Vec<_> = current
            .residuals
            .iter()
            .zip(&residual_scale)
            .map(|(r, s)| -r * s)
            .collect();
        let Some(z) = dense_solve(matrix, rhs) else {
            return Ok(result(
                point,
                current.residuals,
                residual_norm,
                iteration,
                evaluations,
                jacobians,
                0,
                accepted,
                reused,
                NonlinearSystemTermination::SingularJacobian,
            ));
        };
        let step = scaled(&z, &variable_scale);
        let step_norm = norm(&step);
        if step_norm <= options.step_tolerance * (1.0 + norm(&point)) {
            return Ok(result(
                point,
                current.residuals,
                residual_norm,
                iteration,
                evaluations,
                jacobians,
                0,
                accepted,
                reused,
                NonlinearSystemTermination::ConvergedStep,
            ));
        }
        let Some((next_point, next, next_norm, alpha)) = line_search(
            &point,
            &step,
            residual_norm,
            &residual_scale,
            &options,
            &mut evaluations,
            &mut system,
        )?
        else {
            return Ok(result(
                point,
                current.residuals,
                residual_norm,
                iteration,
                evaluations,
                jacobians,
                0,
                accepted,
                reused,
                NonlinearSystemTermination::LineSearchFailed,
            ));
        };
        point = next_point;
        current = next;
        accepted += 1;
        age += 1;
        if !use_cached || age > options.jacobian_reuse_iterations {
            jacobians += 1;
        }
        if !observer(&NonlinearSystemIteration {
            iteration: iteration + 1,
            residual_norm: next_norm,
            step_norm: alpha * step_norm,
            step_length: alpha,
            evaluations,
            jacobian_evaluations: jacobians,
        }) {
            return Ok(result(
                point,
                current.residuals,
                next_norm,
                iteration + 1,
                evaluations,
                jacobians,
                0,
                accepted,
                reused,
                NonlinearSystemTermination::Cancelled,
            ));
        }
    }
    let residual_norm = norm(&scaled(&current.residuals, &residual_scale));
    Ok(result(
        point,
        current.residuals,
        residual_norm,
        options.max_iterations,
        evaluations,
        jacobians,
        0,
        accepted,
        reused,
        NonlinearSystemTermination::IterationLimit,
    ))
}

pub fn solve_newton_krylov<R, J>(
    initial: &[f64],
    options: NonlinearSystemOptions,
    residual: R,
    jacobian_product: J,
) -> Result<NonlinearSystemResult, OptimizeError>
where
    R: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
    J: FnMut(&[f64], &[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    solve_newton_krylov_with_observer(initial, options, residual, jacobian_product, |_| true)
}

pub fn solve_newton_krylov_with_observer<R, J, O>(
    initial: &[f64],
    options: NonlinearSystemOptions,
    mut residual: R,
    mut jacobian_product: J,
    mut observer: O,
) -> Result<NonlinearSystemResult, OptimizeError>
where
    R: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
    J: FnMut(&[f64], &[f64]) -> Result<Vec<f64>, OptimizeError>,
    O: FnMut(&NonlinearSystemIteration) -> bool,
{
    let (variable_scale, residual_scale) = validate(initial, &options)?;
    let n = initial.len();
    let mut point = initial.to_vec();
    let mut values = checked_residual(&mut residual, &point, n)?;
    let mut evaluations = 1;
    let mut products = 0;
    let mut accepted = 0;
    for iteration in 0..options.max_iterations {
        let residual_norm = norm(&scaled(&values, &residual_scale));
        if residual_norm <= options.residual_tolerance {
            return Ok(result(
                point,
                values,
                residual_norm,
                iteration,
                evaluations,
                iteration,
                products,
                accepted,
                0,
                NonlinearSystemTermination::ConvergedResidual,
            ));
        }
        let rhs: Vec<_> = values
            .iter()
            .zip(&residual_scale)
            .map(|(r, s)| -r * s)
            .collect();
        let krylov = gmres(
            &rhs,
            options.max_krylov_iterations,
            options.krylov_tolerance,
            |z| {
                let direction = scaled(z, &variable_scale);
                let value = jacobian_product(&point, &direction)?;
                products += 1;
                if value.len() != n || !value.iter().all(|v| v.is_finite()) {
                    return Err(invalid("invalid Jacobian product"));
                }
                Ok(scaled(&value, &residual_scale))
            },
        )?;
        if !krylov.2 {
            return Ok(result(
                point,
                values,
                residual_norm,
                iteration,
                evaluations,
                iteration + 1,
                products,
                accepted,
                0,
                NonlinearSystemTermination::KrylovFailed,
            ));
        }
        let step = scaled(&krylov.0, &variable_scale);
        let step_norm = norm(&step);
        if step_norm <= options.step_tolerance * (1.0 + norm(&point)) {
            return Ok(result(
                point,
                values,
                residual_norm,
                iteration,
                evaluations,
                iteration + 1,
                products,
                accepted,
                0,
                NonlinearSystemTermination::ConvergedStep,
            ));
        }
        let Some((next_point, next, next_norm, alpha)) = residual_line_search(
            &point,
            &step,
            residual_norm,
            &residual_scale,
            &options,
            &mut evaluations,
            &mut residual,
        )?
        else {
            return Ok(result(
                point,
                values,
                residual_norm,
                iteration,
                evaluations,
                iteration + 1,
                products,
                accepted,
                0,
                NonlinearSystemTermination::LineSearchFailed,
            ));
        };
        point = next_point;
        values = next;
        accepted += 1;
        if !observer(&NonlinearSystemIteration {
            iteration: iteration + 1,
            residual_norm: next_norm,
            step_norm: alpha * step_norm,
            step_length: alpha,
            evaluations,
            jacobian_evaluations: iteration + 1,
        }) {
            return Ok(result(
                point,
                values,
                next_norm,
                iteration + 1,
                evaluations,
                iteration + 1,
                products,
                accepted,
                0,
                NonlinearSystemTermination::Cancelled,
            ));
        }
    }
    let residual_norm = norm(&scaled(&values, &residual_scale));
    Ok(result(
        point,
        values,
        residual_norm,
        options.max_iterations,
        evaluations,
        options.max_iterations,
        products,
        accepted,
        0,
        NonlinearSystemTermination::IterationLimit,
    ))
}

#[derive(Debug, Clone)]
pub struct ContinuationOptions {
    pub initial_step: f64,
    pub minimum_step: f64,
    pub maximum_step: f64,
    pub growth: f64,
    pub shrink: f64,
    pub max_stages: usize,
    pub corrector: NonlinearSystemOptions,
}
impl Default for ContinuationOptions {
    fn default() -> Self {
        Self {
            initial_step: 0.1,
            minimum_step: 1e-5,
            maximum_step: 0.25,
            growth: 1.5,
            shrink: 0.5,
            max_stages: 200,
            corrector: Default::default(),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationStage {
    pub parameter: f64,
    pub step: f64,
    pub iterations: usize,
    pub residual_norm: f64,
}
#[derive(Debug, Clone)]
pub struct ContinuationResult {
    pub point: Vec<f64>,
    pub parameter: f64,
    pub stages: Vec<ContinuationStage>,
    pub rejected_stages: usize,
    pub evaluations: usize,
    pub reached_target: bool,
}

pub fn continue_solution<F>(
    initial: &[f64],
    start: f64,
    target: f64,
    options: ContinuationOptions,
    mut system: F,
) -> Result<ContinuationResult, OptimizeError>
where
    F: FnMut(f64, &[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
{
    if !start.is_finite()
        || !target.is_finite()
        || !positive(options.initial_step)
        || !positive(options.minimum_step)
        || options.maximum_step < options.initial_step
        || !options.maximum_step.is_finite()
        || options.growth <= 1.0
        || !options.growth.is_finite()
        || !(0.0..1.0).contains(&options.shrink)
        || options.max_stages == 0
    {
        return Err(invalid("invalid continuation options"));
    }
    if start == target {
        return Ok(ContinuationResult {
            point: initial.to_vec(),
            parameter: start,
            stages: vec![],
            rejected_stages: 0,
            evaluations: 0,
            reached_target: true,
        });
    }
    let direction = if target > start { 1.0 } else { -1.0 };
    let mut point = initial.to_vec();
    let mut parameter = start;
    let mut step = options.initial_step;
    let mut stages = vec![];
    let mut rejected = 0;
    let mut evaluations = 0;
    while stages.len() + rejected < options.max_stages && parameter != target {
        let trial = if direction > 0.0 {
            (parameter + step).min(target)
        } else {
            (parameter - step).max(target)
        };
        let solved =
            solve_nonlinear_system(&point, options.corrector.clone(), |x| system(trial, x))?;
        evaluations += solved.evaluations;
        if matches!(
            solved.termination,
            NonlinearSystemTermination::ConvergedResidual
                | NonlinearSystemTermination::ConvergedStep
        ) {
            stages.push(ContinuationStage {
                parameter: trial,
                step: (trial - parameter).abs(),
                iterations: solved.iterations,
                residual_norm: solved.residual_norm,
            });
            point = solved.point;
            parameter = trial;
            step = (step * options.growth).min(options.maximum_step);
        } else {
            rejected += 1;
            step *= options.shrink;
            if step < options.minimum_step {
                break;
            }
        }
    }
    Ok(ContinuationResult {
        point,
        parameter,
        stages,
        rejected_stages: rejected,
        evaluations,
        reached_target: parameter == target,
    })
}

#[derive(Debug, Clone)]
pub struct PseudoTransientOptions {
    pub initial_time_step: f64,
    pub maximum_time_step: f64,
    pub growth: f64,
    pub shrink: f64,
    pub corrector: NonlinearSystemOptions,
}
impl Default for PseudoTransientOptions {
    fn default() -> Self {
        Self {
            initial_time_step: 0.1,
            maximum_time_step: 1e6,
            growth: 2.0,
            shrink: 0.5,
            corrector: Default::default(),
        }
    }
}

/// Pseudo-transient continuation using `(J + I/dt) step = -F`.
pub fn solve_pseudo_transient<F>(
    initial: &[f64],
    options: PseudoTransientOptions,
    system: F,
) -> Result<NonlinearSystemResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
{
    solve_pseudo_transient_with_observer(initial, options, system, |_| true)
}

pub fn solve_pseudo_transient_with_observer<F, O>(
    initial: &[f64],
    options: PseudoTransientOptions,
    mut system: F,
    mut observer: O,
) -> Result<NonlinearSystemResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
    O: FnMut(&NonlinearSystemIteration) -> bool,
{
    let (variable_scale, residual_scale) = validate(initial, &options.corrector)?;
    if !positive(options.initial_time_step)
        || !options.maximum_time_step.is_finite()
        || options.maximum_time_step < options.initial_time_step
        || !options.growth.is_finite()
        || options.growth <= 1.0
        || !(0.0..1.0).contains(&options.shrink)
    {
        return Err(invalid("invalid pseudo-transient options"));
    }
    let n = initial.len();
    let mut point = initial.to_vec();
    let mut current = checked_evaluation(&mut system, &point, n)?;
    let mut evaluations = 1;
    let mut accepted = 0;
    let mut time_step = options.initial_time_step;
    for iteration in 0..options.corrector.max_iterations {
        let residual_norm = norm(&scaled(&current.residuals, &residual_scale));
        if residual_norm <= options.corrector.residual_tolerance {
            return Ok(result(
                point,
                current.residuals,
                residual_norm,
                iteration,
                evaluations,
                iteration + 1,
                0,
                accepted,
                0,
                NonlinearSystemTermination::ConvergedResidual,
            ));
        }
        let mut matrix: Vec<Vec<_>> = current
            .jacobian
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, value)| residual_scale[i] * value * variable_scale[j])
                    .collect()
            })
            .collect();
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] += 1.0 / time_step;
        }
        let rhs: Vec<_> = current
            .residuals
            .iter()
            .zip(&residual_scale)
            .map(|(r, s)| -r * s)
            .collect();
        let Some(z) = dense_solve(matrix, rhs) else {
            return Ok(result(
                point,
                current.residuals,
                residual_norm,
                iteration,
                evaluations,
                iteration + 1,
                0,
                accepted,
                0,
                NonlinearSystemTermination::SingularJacobian,
            ));
        };
        let step = scaled(&z, &variable_scale);
        let candidate: Vec<_> = point.iter().zip(&step).map(|(x, s)| x + s).collect();
        let next = checked_evaluation(&mut system, &candidate, n)?;
        evaluations += 1;
        let next_norm = norm(&scaled(&next.residuals, &residual_scale));
        if next_norm < residual_norm {
            point = candidate;
            current = next;
            accepted += 1;
            time_step = (time_step * options.growth).min(options.maximum_time_step);
            if !observer(&NonlinearSystemIteration {
                iteration: iteration + 1,
                residual_norm: next_norm,
                step_norm: norm(&step),
                step_length: 1.0,
                evaluations,
                jacobian_evaluations: iteration + 1,
            }) {
                return Ok(result(
                    point,
                    current.residuals,
                    next_norm,
                    iteration + 1,
                    evaluations,
                    iteration + 1,
                    0,
                    accepted,
                    0,
                    NonlinearSystemTermination::Cancelled,
                ));
            }
        } else {
            time_step *= options.shrink;
            if time_step <= options.corrector.step_tolerance {
                return Ok(result(
                    point,
                    current.residuals,
                    residual_norm,
                    iteration + 1,
                    evaluations,
                    iteration + 1,
                    0,
                    accepted,
                    0,
                    NonlinearSystemTermination::LineSearchFailed,
                ));
            }
        }
    }
    let residual_norm = norm(&scaled(&current.residuals, &residual_scale));
    Ok(result(
        point,
        current.residuals,
        residual_norm,
        options.corrector.max_iterations,
        evaluations,
        options.corrector.max_iterations,
        0,
        accepted,
        0,
        NonlinearSystemTermination::IterationLimit,
    ))
}

fn validate(x: &[f64], o: &NonlinearSystemOptions) -> Result<(Vec<f64>, Vec<f64>), OptimizeError> {
    if x.is_empty()
        || !x.iter().all(|v| v.is_finite())
        || o.max_iterations == 0
        || !positive(o.residual_tolerance)
        || !positive(o.step_tolerance)
        || o.max_line_search_iterations == 0
        || !(0.0..1.0).contains(&o.armijo)
        || !(0.0..1.0).contains(&o.backtracking)
        || o.max_krylov_iterations == 0
        || !positive(o.krylov_tolerance)
    {
        return Err(invalid("invalid nonlinear-system options"));
    }
    fn scale(values: &[f64], n: usize) -> Result<Vec<f64>, OptimizeError> {
        let result = if values.is_empty() {
            vec![1.0; n]
        } else {
            values.to_vec()
        };
        if result.len() != n || !result.iter().all(|v| positive(*v)) {
            Err(invalid("scales must be positive and dimension-matched"))
        } else {
            Ok(result)
        }
    }
    Ok((
        scale(&o.variable_scale, x.len())?,
        scale(&o.residual_scale, x.len())?,
    ))
}
fn checked_evaluation<F>(
    f: &mut F,
    x: &[f64],
    n: usize,
) -> Result<NonlinearSystemEvaluation, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
{
    let value = f(x)?;
    if value.residuals.len() != n
        || value.jacobian.len() != n
        || value.jacobian.iter().any(|row| row.len() != n)
        || !value.residuals.iter().all(|v| v.is_finite())
        || !value.jacobian.iter().flatten().all(|v| v.is_finite())
    {
        Err(invalid(
            "nonlinear system must return finite square residuals and Jacobian",
        ))
    } else {
        Ok(value)
    }
}
fn checked_residual<F>(f: &mut F, x: &[f64], n: usize) -> Result<Vec<f64>, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    let value = f(x)?;
    if value.len() == n && value.iter().all(|v| v.is_finite()) {
        Ok(value)
    } else {
        Err(invalid("nonlinear residual must be finite and square"))
    }
}
fn line_search<F>(
    x: &[f64],
    step: &[f64],
    old_norm: f64,
    scale: &[f64],
    o: &NonlinearSystemOptions,
    evaluations: &mut usize,
    f: &mut F,
) -> Result<Option<(Vec<f64>, NonlinearSystemEvaluation, f64, f64)>, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError>,
{
    let mut alpha = 1.0;
    for _ in 0..o.max_line_search_iterations {
        let candidate: Vec<_> = x.iter().zip(step).map(|(x, s)| x + alpha * s).collect();
        if candidate.iter().all(|v| v.is_finite()) {
            if let Ok(value) = checked_evaluation(f, &candidate, x.len()) {
                *evaluations += 1;
                let next = norm(&scaled(&value.residuals, scale));
                if next <= (1.0 - o.armijo * alpha) * old_norm {
                    return Ok(Some((candidate, value, next, alpha)));
                }
            }
        }
        alpha *= o.backtracking;
    }
    Ok(None)
}
fn residual_line_search<F>(
    x: &[f64],
    step: &[f64],
    old_norm: f64,
    scale: &[f64],
    o: &NonlinearSystemOptions,
    evaluations: &mut usize,
    f: &mut F,
) -> Result<Option<(Vec<f64>, Vec<f64>, f64, f64)>, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    let mut alpha = 1.0;
    for _ in 0..o.max_line_search_iterations {
        let candidate: Vec<_> = x.iter().zip(step).map(|(x, s)| x + alpha * s).collect();
        if candidate.iter().all(|v| v.is_finite()) {
            if let Ok(value) = checked_residual(f, &candidate, x.len()) {
                *evaluations += 1;
                let next = norm(&scaled(&value, scale));
                if next <= (1.0 - o.armijo * alpha) * old_norm {
                    return Ok(Some((candidate, value, next, alpha)));
                }
            }
        }
        alpha *= o.backtracking;
    }
    Ok(None)
}
fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for column in 0..n {
        let pivot =
            (column..n).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column].abs() <= 1e-14 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..n {
            let factor = a[row][column] / a[column][column];
            for entry in column..n {
                a[row][entry] -= factor * a[column][entry];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|j| a[row][j] * x[j]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}
fn gmres<F>(
    rhs: &[f64],
    maximum: usize,
    tolerance: f64,
    mut product: F,
) -> Result<(Vec<f64>, usize, bool), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    let beta = norm(rhs);
    let n = rhs.len();
    let limit = maximum.min(n);
    if beta == 0.0 {
        return Ok((vec![0.0; n], 0, true));
    }
    let mut basis = vec![rhs.iter().map(|v| v / beta).collect::<Vec<_>>()];
    let mut h = vec![vec![0.0; limit]; limit + 1];
    let mut c = vec![0.0; limit];
    let mut s = vec![0.0; limit];
    let mut g = vec![0.0; limit + 1];
    g[0] = beta;
    for column in 0..limit {
        let mut w = product(&basis[column])?;
        for row in 0..=column {
            h[row][column] = dot(&w, &basis[row]);
            axpy(&mut w, -h[row][column], &basis[row]);
        }
        h[column + 1][column] = norm(&w);
        if h[column + 1][column] > 0.0 {
            basis.push(w.iter().map(|v| v / h[column + 1][column]).collect());
        }
        for row in 0..column {
            let first = c[row] * h[row][column] + s[row] * h[row + 1][column];
            h[row + 1][column] = -s[row] * h[row][column] + c[row] * h[row + 1][column];
            h[row][column] = first;
        }
        let radius = h[column][column].hypot(h[column + 1][column]);
        if radius == 0.0 {
            return Ok((vec![0.0; n], column + 1, false));
        }
        c[column] = h[column][column] / radius;
        s[column] = h[column + 1][column] / radius;
        h[column][column] = radius;
        h[column + 1][column] = 0.0;
        g[column + 1] = -s[column] * g[column];
        g[column] *= c[column];
        if g[column + 1].abs() <= tolerance * beta.max(1.0) || column + 1 == limit {
            let mut y = vec![0.0; column + 1];
            for row in (0..=column).rev() {
                let tail: f64 = (row + 1..=column).map(|k| h[row][k] * y[k]).sum();
                y[row] = (g[row] - tail) / h[row][row];
            }
            let mut solution = vec![0.0; n];
            for k in 0..=column {
                axpy(&mut solution, y[k], &basis[k]);
            }
            return Ok((
                solution,
                column + 1,
                g[column + 1].abs() <= tolerance * beta.max(1.0),
            ));
        }
    }
    Ok((vec![0.0; n], limit, false))
}
fn result(
    point: Vec<f64>,
    residuals: Vec<f64>,
    residual_norm: f64,
    iterations: usize,
    evaluations: usize,
    jacobians: usize,
    linear_iterations: usize,
    accepted: usize,
    reused: usize,
    termination: NonlinearSystemTermination,
) -> NonlinearSystemResult {
    NonlinearSystemResult {
        point,
        residuals,
        residual_norm,
        iterations,
        evaluations,
        jacobian_evaluations: jacobians,
        linear_iterations,
        accepted_steps: accepted,
        reused_jacobians: reused,
        termination,
    }
}
fn scaled(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| a * b).collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn axpy(x: &mut [f64], alpha: f64, y: &[f64]) {
    for (x, y) in x.iter_mut().zip(y) {
        *x += alpha * y;
    }
}
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn system(x: &[f64]) -> Result<NonlinearSystemEvaluation, OptimizeError> {
        Ok(NonlinearSystemEvaluation {
            residuals: vec![x[0] * x[0] + x[1] * x[1] - 5.0, x[0] - x[1] - 1.0],
            jacobian: vec![vec![2.0 * x[0], 2.0 * x[1]], vec![1.0, -1.0]],
        })
    }
    #[test]
    fn damped_newton_solves_system() {
        let r = solve_nonlinear_system(&[1.5, 0.5], Default::default(), system).unwrap();
        assert_eq!(r.termination, NonlinearSystemTermination::ConvergedResidual);
        assert!((r.point[0] - 2.0).abs() < 1e-8);
    }
    #[test]
    fn newton_krylov_solves_system() {
        let r = solve_newton_krylov(
            &[1.5, 0.5],
            Default::default(),
            |x| Ok(system(x)?.residuals),
            |x, v| Ok(vec![2.0 * x[0] * v[0] + 2.0 * x[1] * v[1], v[0] - v[1]]),
        )
        .unwrap();
        assert_eq!(r.termination, NonlinearSystemTermination::ConvergedResidual);
        assert!(r.linear_iterations > 0);
    }
    #[test]
    fn graph_and_continuation_work() {
        let e = NonlinearExpression::new(
            vec![
                crate::NonlinearNode::Parameter(0),
                crate::NonlinearNode::Powf(0, 2.0),
                crate::NonlinearNode::Constant(2.0),
                crate::NonlinearNode::Subtract(1, 2),
            ],
            3,
        );
        let m = NonlinearEquationModel::new(1, vec![e]).unwrap();
        let r = solve_equation_model(&m, &[1.0], Default::default()).unwrap();
        assert!((r.point[0] - 2.0_f64.sqrt()).abs() < 1e-8);
        let c = continue_solution(
            &[1.0],
            1.0,
            4.0,
            ContinuationOptions {
                initial_step: 0.2,
                maximum_step: 0.75,
                ..Default::default()
            },
            |p, x| {
                Ok(NonlinearSystemEvaluation {
                    residuals: vec![x[0] * x[0] - p],
                    jacobian: vec![vec![2.0 * x[0]]],
                })
            },
        )
        .unwrap();
        assert!(c.reached_target && (c.point[0] - 2.0).abs() < 1e-8);
    }
    #[test]
    fn singularity_reuse_and_cancellation_are_reported() {
        let singular = solve_nonlinear_system(&[0.0], Default::default(), |x| {
            Ok(NonlinearSystemEvaluation {
                residuals: vec![x[0] * x[0] + 1.0],
                jacobian: vec![vec![2.0 * x[0]]],
            })
        })
        .unwrap();
        assert_eq!(
            singular.termination,
            NonlinearSystemTermination::SingularJacobian
        );
        let o = NonlinearSystemOptions {
            jacobian_reuse_iterations: 2,
            ..Default::default()
        };
        let reused = solve_nonlinear_system(&[1.0], o, |x| {
            Ok(NonlinearSystemEvaluation {
                residuals: vec![x[0] * x[0] - 2.0],
                jacobian: vec![vec![2.0 * x[0]]],
            })
        })
        .unwrap();
        assert!(reused.reused_jacobians > 0);
        let cancelled = solve_nonlinear_system_with_observer(
            &[1.0],
            Default::default(),
            |x| {
                Ok(NonlinearSystemEvaluation {
                    residuals: vec![x[0] * x[0] - 2.0],
                    jacobian: vec![vec![2.0 * x[0]]],
                })
            },
            |_| false,
        )
        .unwrap();
        assert_eq!(cancelled.termination, NonlinearSystemTermination::Cancelled);
    }

    #[test]
    fn pseudo_transient_escapes_singular_initial_jacobian() {
        let r = solve_pseudo_transient(&[0.0], Default::default(), |x| {
            Ok(NonlinearSystemEvaluation {
                residuals: vec![x[0] * x[0] * x[0] - 1.0],
                jacobian: vec![vec![3.0 * x[0] * x[0]]],
            })
        })
        .unwrap();
        assert_eq!(r.termination, NonlinearSystemTermination::ConvergedResidual);
        assert!((r.point[0] - 1.0).abs() < 1e-7);
    }
}
