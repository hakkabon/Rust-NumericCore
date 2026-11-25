//! Dense Powell hybrid method with analytic Jacobian refreshes.

use super::{checked_residual, dot, invalid, positive, scale, NonlinearEquationModel};
use crate::OptimizeError;

#[derive(Debug, Clone, PartialEq)]
pub struct PowellHybridOptions {
    pub max_iterations: u64,
    pub max_residual_evaluations: u64,
    pub residual_tolerance: f64,
    /// Absolute tolerance for steps and the trust radius in scaled coordinates.
    pub step_tolerance: f64,
    pub initial_radius: f64,
    pub maximum_radius: f64,
    /// A scaled step is multiplied componentwise by this vector before use.
    pub variable_scale: Vec<f64>,
    /// Multiplier for each residual row. Empty scales mean all ones.
    pub residual_scale: Vec<f64>,
}

impl Default for PowellHybridOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            max_residual_evaluations: 500,
            residual_tolerance: 1e-9,
            step_tolerance: 1e-12,
            initial_radius: 1.0,
            maximum_radius: 100.0,
            variable_scale: vec![],
            residual_scale: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowellHybridTermination {
    ConvergedResidual,
    /// A small step, small radius, or stationary residual merit above tolerance.
    Stalled,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowellHybridResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    /// Euclidean norm after applying the configured residual scales.
    pub residual_norm: f64,
    pub iterations: u64,
    pub residual_evaluations: u64,
    pub jacobian_evaluations: u64,
    pub rank_one_updates: u64,
    pub accepted_steps: u64,
    pub final_radius: f64,
    pub termination: PowellHybridTermination,
}

/// Solve a square equation graph using value-only trials and analytic refreshes.
pub fn solve_powell_hybrid_model(
    model: &NonlinearEquationModel,
    initial: &[f64],
    options: PowellHybridOptions,
) -> Result<PowellHybridResult, OptimizeError> {
    if initial.len() != model.parameter_count || model.equations.len() != model.parameter_count {
        return Err(invalid(
            "Powell hybrid equation model dimensions do not agree",
        ));
    }
    solve_powell_hybrid(
        initial,
        options,
        |point| {
            model
                .equations
                .iter()
                .map(|eq| eq.evaluate_value(point))
                .collect()
        },
        |point| Ok(model.evaluate(point)?.jacobian),
    )
}

/// Solve a square system with scaled dogleg steps and good Broyden updates.
/// The residual callback runs at trial points. The Jacobian callback runs
/// at the current accepted point initially and on model refresh.
pub fn solve_powell_hybrid<R, J>(
    initial: &[f64],
    options: PowellHybridOptions,
    mut residual: R,
    mut jacobian: J,
) -> Result<PowellHybridResult, OptimizeError>
where
    R: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
    J: FnMut(&[f64]) -> Result<Vec<Vec<f64>>, OptimizeError>,
{
    if initial.is_empty()
        || initial.iter().any(|x| !x.is_finite())
        || options.max_iterations == 0
        || options.max_residual_evaluations == 0
        || !positive(options.residual_tolerance)
        || !positive(options.step_tolerance)
        || !positive(options.initial_radius)
        || !positive(options.maximum_radius)
        || options.initial_radius > options.maximum_radius
        || options.step_tolerance >= options.initial_radius
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid Powell hybrid options or initial point",
        ));
    }
    let n = initial.len();
    let variable_scale = scale(&options.variable_scale, n)?;
    let residual_scale = scale(&options.residual_scale, n)?;

    let residuals = checked_residual(&mut residual, initial, n)?;
    let (mut values, residual_norm) = scaled_residual(&residuals, &residual_scale)?;
    let mut report = PowellHybridResult {
        point: initial.to_vec(),
        residuals,
        residual_norm,
        iterations: 0,
        residual_evaluations: 1,
        jacobian_evaluations: 0,
        rank_one_updates: 0,
        accepted_steps: 0,
        final_radius: options.initial_radius,
        termination: PowellHybridTermination::IterationLimit,
    };
    let mut matrix = Vec::new();
    let mut updated = false;
    let mut poor_trials = 0;
    report.termination = loop {
        use PowellHybridTermination::*;
        if report.residual_norm <= options.residual_tolerance {
            break ConvergedResidual;
        }
        if report.residual_evaluations == options.max_residual_evaluations {
            break EvaluationLimit;
        }
        if report.iterations == options.max_iterations {
            break IterationLimit;
        }
        let radius = report.final_radius;
        if radius <= options.step_tolerance {
            break Stalled;
        }
        if matrix.is_empty() {
            matrix = scaled_jacobian(jacobian(&report.point)?, &residual_scale, &variable_scale)?;
            report.jacobian_evaluations += 1;
            updated = false;
            poor_trials = 0;
        }

        let Some(step) = dogleg(&matrix, &values, report.residual_norm, radius) else {
            break NumericalFailure;
        };
        let trial: Vec<_> = report
            .point
            .iter()
            .zip(&variable_scale)
            .zip(step)
            .map(|((&x, &s), p)| s.mul_add(p, x))
            .collect();
        if trial.iter().any(|x| !x.is_finite()) {
            break NumericalFailure;
        }
        // The secant equation uses the displacement represented in floating point.
        let step: Vec<_> = trial
            .iter()
            .zip(&report.point)
            .zip(&variable_scale)
            .map(|((&y, &x), &s)| (y - x) / s)
            .collect();
        let length = norm(&step);
        if !length.is_finite() {
            break NumericalFailure;
        }
        let image: Vec<_> = matrix.iter().map(|row| dot(row, &step)).collect();
        let model_norm = values
            .iter()
            .zip(&image)
            .fold(0.0_f64, |length, (r, jp)| length.hypot(r + jp));
        if !model_norm.is_finite() {
            break NumericalFailure;
        }
        let predicted = reduction(report.residual_norm, model_norm);
        if length <= options.step_tolerance || predicted <= 0.0 {
            if updated {
                matrix.clear();
                continue;
            }
            break Stalled;
        }

        let trial_residuals = checked_residual(&mut residual, &trial, n)?;
        let (trial_values, trial_norm) = scaled_residual(&trial_residuals, &residual_scale)?;
        report.iterations += 1;
        report.residual_evaluations += 1;
        let ratio = reduction(report.residual_norm, trial_norm) / predicted;
        if ratio < 0.1 {
            report.final_radius = 0.5 * radius;
            poor_trials += 1;
        } else {
            poor_trials = 0;
            if ratio >= 0.75 && length >= 0.8 * radius {
                report.final_radius = (2.0 * radius).min(options.maximum_radius);
            }
        }

        // Update even when a finite trial is rejected: the same secant
        // equation applies between the two sampled points. Refresh after
        // two poor trials or when an update cannot be represented.
        if poor_trials >= 2
            || !broyden_update(&mut matrix, &values, &trial_values, &image, &step, length)
        {
            matrix.clear();
        } else {
            report.rank_one_updates += 1;
            updated = true;
        }
        if trial_norm <= options.residual_tolerance
            || (trial_norm < report.residual_norm && ratio >= 1e-4)
        {
            report.point = trial;
            report.residuals = trial_residuals;
            report.residual_norm = trial_norm;
            report.accepted_steps += 1;
            values = trial_values;
        }
    };
    Ok(report)
}

fn dogleg(
    jacobian: &[Vec<f64>],
    residual: &[f64],
    residual_norm: f64,
    radius: f64,
) -> Option<Vec<f64>> {
    let n = residual.len();
    let scale = jacobian
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Some(vec![0.0; n]);
    }
    let matrix: Vec<Vec<_>> = jacobian
        .iter()
        .map(|row| row.iter().map(|v| v / scale).collect())
        .collect();
    let rhs: Vec<_> = residual.iter().map(|r| -r / residual_norm).collect();
    let step_scale = residual_norm / scale;
    let newton = nc_decomp::least_squares_qr(&matrix, &rhs, n as f64 * f64::EPSILON)
        .map(|mut p| {
            for x in &mut p {
                *x *= step_scale;
            }
            p
        })
        .filter(|p| p.iter().all(|x| x.is_finite()));
    if let Some(p) = &newton {
        if norm(p) <= radius {
            return newton;
        }
    }

    let mut direction = vec![0.0; n];
    for (row, &minus_r) in matrix.iter().zip(&rhs) {
        for (g, &entry) in direction.iter_mut().zip(row) {
            *g += minus_r * entry;
        }
    }
    let gradient_norm = norm(&direction);
    if gradient_norm == 0.0 {
        return Some(direction);
    }
    for d in &mut direction {
        *d /= gradient_norm;
    }
    let image_norm = matrix
        .iter()
        .fold(0.0_f64, |length, row| length.hypot(dot(row, &direction)));
    let cauchy_length = (gradient_norm / image_norm / image_norm) * step_scale;
    if cauchy_length.is_nan() {
        return None;
    }
    let mut step: Vec<_> = direction
        .iter()
        .map(|d| cauchy_length.min(radius) * d)
        .collect();
    if let Some(p) = newton.filter(|_| cauchy_length < radius) {
        // Work in units of the boundary radius and parameterize the segment
        // by distance, avoiding squared terms in the Newton step's magnitude.
        let segment: Vec<_> = p.iter().zip(&step).map(|(p, c)| p - c).collect();
        let segment_length = norm(&segment);
        if !positive(segment_length) {
            return Some(step);
        }
        let unit: Vec<_> = segment.iter().map(|v| v / segment_length).collect();
        let fraction = cauchy_length / radius;
        let projection = dot(&direction, &unit) * fraction;
        let gap = ((radius - cauchy_length) / radius) * (1.0 + fraction);
        let root = projection.hypot(gap.sqrt());
        let distance = if projection >= 0.0 {
            gap / (root + projection)
        } else {
            root - projection
        };
        for (s, d) in step.iter_mut().zip(unit) {
            *s = (radius * distance).mul_add(d, *s);
        }
    }
    Some(step)
}

fn broyden_update(
    jacobian: &mut [Vec<f64>],
    old: &[f64],
    new: &[f64],
    image: &[f64],
    step: &[f64],
    length: f64,
) -> bool {
    for ((row, (&old, &new)), &jp) in jacobian.iter_mut().zip(old.iter().zip(new)).zip(image) {
        let correction = ((new - old) - jp) / length;
        for (entry, &p) in row.iter_mut().zip(step) {
            *entry = correction.mul_add(p / length, *entry);
            if !entry.is_finite() {
                return false;
            }
        }
    }
    true
}

fn scaled_residual(values: &[f64], scale: &[f64]) -> Result<(Vec<f64>, f64), OptimizeError> {
    let scaled: Vec<_> = values
        .iter()
        .zip(scale)
        .map(|(value, scale)| value * scale)
        .collect();
    let length = norm(&scaled);
    if length.is_finite() {
        Ok((scaled, length))
    } else {
        Err(invalid("Powell hybrid scaled residual overflow"))
    }
}

fn scaled_jacobian(
    mut jacobian: Vec<Vec<f64>>,
    residual_scale: &[f64],
    variable_scale: &[f64],
) -> Result<Vec<Vec<f64>>, OptimizeError> {
    if jacobian.len() != residual_scale.len()
        || jacobian.iter().any(|r| r.len() != variable_scale.len())
    {
        return Err(invalid(
            "Powell hybrid Jacobian must be square and dimension-matched",
        ));
    }
    for (row, &r) in jacobian.iter_mut().zip(residual_scale) {
        for (entry, &s) in row.iter_mut().zip(variable_scale) {
            *entry = (*entry * s) * r;
            if !entry.is_finite() {
                return Err(invalid(
                    "Powell hybrid Jacobian must be finite after scaling",
                ));
            }
        }
    }
    Ok(jacobian)
}

// Relative reduction in squared norm without squaring the original norm.
fn reduction(current: f64, new: f64) -> f64 {
    ((current - new) / current) * (1.0 + new / current)
}

fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |norm, &x| norm.hypot(x))
}
