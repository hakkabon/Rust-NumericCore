use crate::{to_domain_bound, FfiBound, FfiError};
use nc_optimize::{
    minimize_cobyla, CobylaOptions, CobylaTermination, ConstrainedDerivativeFreeEvaluation,
    DerivativeFreeConstraintKind, OptimizeError,
};

#[derive(uniffi::Enum)]
pub enum FfiCobylaConstraintKind {
    LessEqual,
    Equality,
}

#[derive(uniffi::Record)]
pub struct FfiCobylaEvaluation {
    pub objective: f64,
    pub constraints: Vec<f64>,
}

#[uniffi::export(callback_interface)]
pub trait FfiCobylaEvaluator: Send + Sync {
    fn evaluate(&self, point: Vec<f64>) -> Option<FfiCobylaEvaluation>;
}

#[derive(uniffi::Record)]
pub struct FfiCobylaOptions {
    pub max_iterations: u64,
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
    pub feasibility_tolerance: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiCobylaTermination {
    RadiusReached,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
    AllVariablesFixed,
}

#[derive(uniffi::Record)]
pub struct FfiCobylaResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub maximum_violation: f64,
    pub feasible: bool,
    pub evaluations: u64,
    pub termination: FfiCobylaTermination,
}

#[uniffi::export]
pub fn solve_cobyla(
    initial: Vec<f64>,
    bounds: Option<Vec<FfiBound>>,
    constraint_kinds: Vec<FfiCobylaConstraintKind>,
    options: FfiCobylaOptions,
    evaluator: Box<dyn FfiCobylaEvaluator>,
) -> Result<FfiCobylaResult, FfiError> {
    let bounds: Option<Vec<_>> =
        bounds.map(|values| values.into_iter().map(to_domain_bound).collect());
    let kinds: Vec<_> = constraint_kinds
        .into_iter()
        .map(|kind| match kind {
            FfiCobylaConstraintKind::LessEqual => DerivativeFreeConstraintKind::LessEqual,
            FfiCobylaConstraintKind::Equality => DerivativeFreeConstraintKind::Equality,
        })
        .collect();
    let result = minimize_cobyla(
        &initial,
        bounds.as_deref(),
        &kinds,
        CobylaOptions {
            max_iterations: options.max_iterations,
            max_evaluations: options.max_evaluations,
            initial_radius: options.initial_radius,
            final_radius: options.final_radius,
            feasibility_tolerance: options.feasibility_tolerance,
        },
        |point| {
            let sample = evaluator
                .evaluate(point.to_vec())
                .ok_or_else(|| OptimizeError::InvalidProblem("COBYLA callback failed".into()))?;
            Ok(ConstrainedDerivativeFreeEvaluation {
                objective: sample.objective,
                constraints: sample.constraints,
            })
        },
    )?;
    Ok(FfiCobylaResult {
        point: result.point,
        objective: result.objective,
        constraint_values: result.constraint_values,
        maximum_violation: result.maximum_violation,
        feasible: result.feasible,
        evaluations: result.evaluations,
        termination: match result.termination {
            CobylaTermination::RadiusReached => FfiCobylaTermination::RadiusReached,
            CobylaTermination::EvaluationLimit => FfiCobylaTermination::EvaluationLimit,
            CobylaTermination::IterationLimit => FfiCobylaTermination::IterationLimit,
            CobylaTermination::NumericalFailure => FfiCobylaTermination::NumericalFailure,
            CobylaTermination::AllVariablesFixed => FfiCobylaTermination::AllVariablesFixed,
        },
    })
}
