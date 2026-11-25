use crate::{to_domain_problem, FfiError, FfiProblem};
use nc_optimize::{minimize_lincoa, LincoaOptions, LincoaTermination, OptimizeError};

/// Synchronous objective callback. No value signals caller failure.
#[uniffi::export(callback_interface)]
pub trait FfiLincoaObjective: Send + Sync {
    fn evaluate(&self, point: Vec<f64>) -> Option<f64>;
}

#[derive(uniffi::Record)]
pub struct FfiLincoaOptions {
    pub max_iterations: u64,
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
    pub feasibility_tolerance: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiLincoaTermination {
    RadiusReached,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
    AllVariablesFixed,
}

#[derive(uniffi::Record)]
pub struct FfiLincoaResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub maximum_violation: f64,
    pub evaluations: u64,
    pub termination: FfiLincoaTermination,
}

#[uniffi::export]
pub fn solve_lincoa(
    initial: Vec<f64>,
    problem: FfiProblem,
    options: FfiLincoaOptions,
    objective: Box<dyn FfiLincoaObjective>,
) -> Result<FfiLincoaResult, FfiError> {
    let result = minimize_lincoa(
        &initial,
        &to_domain_problem(problem)?,
        LincoaOptions {
            max_iterations: options.max_iterations,
            max_evaluations: options.max_evaluations,
            initial_radius: options.initial_radius,
            final_radius: options.final_radius,
            feasibility_tolerance: options.feasibility_tolerance,
        },
        |point| {
            objective.evaluate(point.to_vec()).ok_or_else(|| {
                OptimizeError::InvalidProblem("LINCOA objective callback failed".into())
            })
        },
    )?;
    Ok(FfiLincoaResult {
        point: result.point,
        objective: result.objective,
        maximum_violation: result.maximum_violation,
        evaluations: result.evaluations,
        termination: match result.termination {
            LincoaTermination::RadiusReached => FfiLincoaTermination::RadiusReached,
            LincoaTermination::EvaluationLimit => FfiLincoaTermination::EvaluationLimit,
            LincoaTermination::IterationLimit => FfiLincoaTermination::IterationLimit,
            LincoaTermination::NumericalFailure => FfiLincoaTermination::NumericalFailure,
            LincoaTermination::AllVariablesFixed => FfiLincoaTermination::AllVariablesFixed,
        },
    })
}
