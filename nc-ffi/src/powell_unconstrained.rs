use crate::FfiError;
use nc_optimize::{
    minimize_newuoa, minimize_uobyqa, OptimizeError, PowellUnconstrainedOptions,
    PowellUnconstrainedResult, PowellUnconstrainedTermination,
};

/// Synchronous objective callback. No value signals caller failure.
#[uniffi::export(callback_interface)]
pub trait FfiPowellObjective: Send + Sync {
    fn evaluate(&self, point: Vec<f64>) -> Option<f64>;
}

#[derive(uniffi::Record)]
pub struct FfiPowellUnconstrainedOptions {
    pub max_iterations: u64,
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiPowellUnconstrainedTermination {
    RadiusReached,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
}

#[derive(uniffi::Record)]
pub struct FfiPowellUnconstrainedResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub evaluations: u64,
    pub termination: FfiPowellUnconstrainedTermination,
}

#[uniffi::export]
pub fn solve_newuoa(
    initial: Vec<f64>,
    options: FfiPowellUnconstrainedOptions,
    objective: Box<dyn FfiPowellObjective>,
) -> Result<FfiPowellUnconstrainedResult, FfiError> {
    Ok(minimize_newuoa(&initial, options.into(), |point| {
        evaluate(&*objective, point)
    })?
    .into())
}

#[uniffi::export]
pub fn solve_uobyqa(
    initial: Vec<f64>,
    options: FfiPowellUnconstrainedOptions,
    objective: Box<dyn FfiPowellObjective>,
) -> Result<FfiPowellUnconstrainedResult, FfiError> {
    Ok(minimize_uobyqa(&initial, options.into(), |point| {
        evaluate(&*objective, point)
    })?
    .into())
}

fn evaluate(callback: &dyn FfiPowellObjective, point: &[f64]) -> Result<f64, OptimizeError> {
    callback.evaluate(point.to_vec()).ok_or_else(|| {
        OptimizeError::InvalidProblem("unconstrained Powell objective callback failed".into())
    })
}

impl From<FfiPowellUnconstrainedOptions> for PowellUnconstrainedOptions {
    fn from(options: FfiPowellUnconstrainedOptions) -> Self {
        Self {
            max_iterations: options.max_iterations,
            max_evaluations: options.max_evaluations,
            initial_radius: options.initial_radius,
            final_radius: options.final_radius,
        }
    }
}

impl From<PowellUnconstrainedResult> for FfiPowellUnconstrainedResult {
    fn from(result: PowellUnconstrainedResult) -> Self {
        use PowellUnconstrainedTermination::*;
        Self {
            point: result.point,
            objective: result.objective,
            evaluations: result.evaluations,
            termination: match result.termination {
                RadiusReached => FfiPowellUnconstrainedTermination::RadiusReached,
                EvaluationLimit => FfiPowellUnconstrainedTermination::EvaluationLimit,
                IterationLimit => FfiPowellUnconstrainedTermination::IterationLimit,
                NumericalFailure => FfiPowellUnconstrainedTermination::NumericalFailure,
            },
        }
    }
}
