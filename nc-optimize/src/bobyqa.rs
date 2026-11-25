//! Powell's bound-constrained quadratic interpolation solver.

use crate::{Bound, OptimizeError};
use ::bobyqa::{Bobyqa, Config, Status, FUNCMAX};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BobyqaOptions {
    pub max_evaluations: usize,
    pub initial_radius: f64,
    pub final_radius: f64,
    /// Defaults to twice the number of free variables plus one.
    pub interpolation_points: Option<usize>,
}

impl Default for BobyqaOptions {
    fn default() -> Self {
        Self {
            max_evaluations: 1_000,
            initial_radius: 1.0,
            final_radius: 1e-6,
            interpolation_points: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BobyqaTermination {
    /// The solver reached its final trust-region resolution.
    RadiusReached,
    EvaluationLimit,
    ModelDegenerate,
    AllVariablesFixed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BobyqaResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub evaluations: usize,
    pub termination: BobyqaTermination,
}

// PRIMA treats larger absolute bounds as unbounded. Reject such explicit
// bounds so NumericCore never loses one during conversion to the backend.
const BOUND_LIMIT: f64 = f64::MAX / 4.0;
// Reserved objective target used to stop the infallible engine callback
// immediately after a user error or when a small caller budget runs out.
const STOP: f64 = -FUNCMAX;

/// Minimize using PRIMA's BOBYQA algorithm. Fixed coordinates are removed
/// from interpolation and restored before every objective call. The
/// objective must be finite with magnitude below PRIMA's `FUNCMAX`.
pub fn minimize_bobyqa<F>(
    initial: &[f64],
    bounds: &[Bound],
    options: BobyqaOptions,
    mut objective: F,
) -> Result<BobyqaResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    if initial.is_empty()
        || bounds.len() != initial.len()
        || initial
            .iter()
            .any(|x| !x.is_finite() || x.abs() >= BOUND_LIMIT)
        || options.max_evaluations == 0
        || options.interpolation_points == Some(0)
        || !options.initial_radius.is_finite()
        || !options.final_radius.is_finite()
        || options.final_radius <= 0.0
        || options.final_radius > options.initial_radius
    {
        return Err(invalid(
            "invalid BOBYQA dimensions, initial point, radii, or budget",
        ));
    }

    let mut point = initial.to_vec();
    let mut free = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut initial_radius = options.initial_radius;
    for (index, bound) in bounds.iter().enumerate() {
        if bound
            .lower
            .into_iter()
            .chain(bound.upper)
            .any(|b| !b.is_finite() || b.abs() >= BOUND_LIMIT)
            || matches!((bound.lower, bound.upper), (Some(lo), Some(hi)) if lo > hi)
        {
            return Err(invalid(
                "BOBYQA bounds must be ordered and within PRIMA's finite range",
            ));
        }
        let lo = bound.lower.unwrap_or(f64::NEG_INFINITY);
        let hi = bound.upper.unwrap_or(f64::INFINITY);
        point[index] = point[index].clamp(lo, hi);
        if lo != hi {
            free.push(index);
            lower.push(lo);
            upper.push(hi);
            initial_radius = initial_radius.min((hi - lo) * 0.5);
        }
    }

    if free.is_empty() {
        return Ok(BobyqaResult {
            objective: checked_value(objective(&point)?)?,
            point,
            evaluations: 1,
            termination: BobyqaTermination::AllVariablesFixed,
        });
    }

    let mut config = Config::new(free.len());
    config.npt = options.interpolation_points.unwrap_or(config.npt);
    config.rho_begin = initial_radius;
    config.rho_end = options.final_radius.min(initial_radius);
    config.max_fun = options.max_evaluations.max(config.npt.saturating_add(1));
    config.f_target = STOP;
    let mut solver = Bobyqa::new(free.len(), config).map_err(backend_error)?;
    let mut iterate: Vec<_> = free.iter().map(|&index| point[index]).collect();
    let mut best_point = point.clone();
    let mut best_value = f64::INFINITY;
    let mut evaluations = 0;
    let mut failure = None;
    let run = solver.minimize(
        |sample| {
            if failure.is_some() || evaluations == options.max_evaluations {
                return STOP;
            }
            for (&index, &value) in free.iter().zip(sample) {
                let bound = bounds[index];
                if !value.is_finite()
                    || bound.lower.is_some_and(|lo| value < lo)
                    || bound.upper.is_some_and(|hi| value > hi)
                {
                    failure = Some(invalid("BOBYQA produced an infeasible sample"));
                    return STOP;
                }
                point[index] = value;
            }
            evaluations += 1;
            match objective(&point).and_then(checked_value) {
                Ok(value) => {
                    if value < best_value {
                        best_value = value;
                        best_point.copy_from_slice(&point);
                    }
                    value
                }
                Err(error) => {
                    failure = Some(error);
                    STOP
                }
            }
        },
        &mut iterate,
        &lower,
        &upper,
    );
    if let Some(error) = failure {
        return Err(error);
    }
    let termination = match run.status {
        Status::Converged => BobyqaTermination::RadiusReached,
        Status::MaxFunReached => BobyqaTermination::EvaluationLimit,
        Status::TargetReached if evaluations == options.max_evaluations => {
            BobyqaTermination::EvaluationLimit
        }
        Status::ModelDegenerate => BobyqaTermination::ModelDegenerate,
        status => return Err(backend_error(status)),
    };
    if !best_value.is_finite() {
        return Err(invalid(
            "BOBYQA returned without a valid objective evaluation",
        ));
    }
    Ok(BobyqaResult {
        point: best_point,
        objective: best_value,
        evaluations,
        termination,
    })
}

fn checked_value(value: f64) -> Result<f64, OptimizeError> {
    if value.is_finite() && value.abs() < FUNCMAX {
        Ok(value)
    } else {
        Err(invalid(
            "BOBYQA objective must be finite with magnitude below 1e30",
        ))
    }
}

fn backend_error(status: Status) -> OptimizeError {
    OptimizeError::InvalidProblem(format!("BOBYQA: {status}"))
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
