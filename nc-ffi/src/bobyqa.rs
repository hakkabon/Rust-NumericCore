use crate::{FfiBound, FfiError};
use nc_optimize::{minimize_bobyqa, BobyqaOptions, BobyqaTermination, Bound, OptimizeError};

/// Synchronous objective callback. No value signals caller failure.
#[uniffi::export(callback_interface)]
pub trait FfiBobyqaObjective: Send + Sync {
    fn evaluate(&self, point: Vec<f64>) -> Option<f64>;
}

#[derive(uniffi::Record)]
pub struct FfiBobyqaOptions {
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
    pub interpolation_points: Option<u64>,
}

#[derive(uniffi::Enum)]
pub enum FfiBobyqaTermination {
    RadiusReached,
    EvaluationLimit,
    ModelDegenerate,
    AllVariablesFixed,
}

#[derive(uniffi::Record)]
pub struct FfiBobyqaResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub evaluations: u64,
    pub termination: FfiBobyqaTermination,
}

#[uniffi::export]
pub fn solve_bobyqa(
    initial: Vec<f64>,
    bounds: Vec<FfiBound>,
    options: FfiBobyqaOptions,
    objective: Box<dyn FfiBobyqaObjective>,
) -> Result<FfiBobyqaResult, FfiError> {
    let index = |value| {
        usize::try_from(value).map_err(|_| {
            OptimizeError::InvalidConfiguration("BOBYQA option outside the platform range")
        })
    };
    let bounds: Vec<_> = bounds
        .into_iter()
        .map(|bound| Bound {
            lower: bound.lower,
            upper: bound.upper,
        })
        .collect();
    let result = minimize_bobyqa(
        &initial,
        &bounds,
        BobyqaOptions {
            max_evaluations: index(options.max_evaluations)?,
            initial_radius: options.initial_radius,
            final_radius: options.final_radius,
            interpolation_points: options.interpolation_points.map(index).transpose()?,
        },
        |point| {
            objective.evaluate(point.to_vec()).ok_or_else(|| {
                OptimizeError::InvalidProblem("BOBYQA objective callback failed".into())
            })
        },
    )?;
    Ok(FfiBobyqaResult {
        point: result.point,
        objective: result.objective,
        evaluations: result.evaluations as u64,
        termination: match result.termination {
            BobyqaTermination::RadiusReached => FfiBobyqaTermination::RadiusReached,
            BobyqaTermination::EvaluationLimit => FfiBobyqaTermination::EvaluationLimit,
            BobyqaTermination::ModelDegenerate => FfiBobyqaTermination::ModelDegenerate,
            BobyqaTermination::AllVariablesFixed => FfiBobyqaTermination::AllVariablesFixed,
        },
    })
}
