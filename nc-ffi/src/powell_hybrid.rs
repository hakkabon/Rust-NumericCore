use crate::{nonlinear::expression, FfiError, FfiNonlinearExpression};
use nc_optimize::{
    solve_powell_hybrid, solve_powell_hybrid_model, NonlinearEquationModel, OptimizeError,
    PowellHybridOptions, PowellHybridResult, PowellHybridTermination,
};

/// Synchronous callbacks. No value signals caller failure.
#[uniffi::export(callback_interface)]
pub trait FfiPowellHybridSystem: Send + Sync {
    fn residual(&self, point: Vec<f64>) -> Option<Vec<f64>>;
    fn jacobian(&self, point: Vec<f64>) -> Option<Vec<Vec<f64>>>;
}

#[derive(uniffi::Record)]
pub struct FfiPowellHybridOptions {
    pub max_iterations: u64,
    pub max_residual_evaluations: u64,
    pub residual_tolerance: f64,
    pub step_tolerance: f64,
    pub initial_radius: f64,
    pub maximum_radius: f64,
    pub variable_scale: Vec<f64>,
    pub residual_scale: Vec<f64>,
}

#[derive(uniffi::Enum)]
pub enum FfiPowellHybridTermination {
    ConvergedResidual,
    Stalled,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
}

#[derive(uniffi::Record)]
pub struct FfiPowellHybridResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: u64,
    pub residual_evaluations: u64,
    pub jacobian_evaluations: u64,
    pub rank_one_updates: u64,
    pub accepted_steps: u64,
    pub final_radius: f64,
    pub termination: FfiPowellHybridTermination,
}

#[uniffi::export]
pub fn run_powell_hybrid(
    initial: Vec<f64>,
    options: FfiPowellHybridOptions,
    system: Box<dyn FfiPowellHybridSystem>,
) -> Result<FfiPowellHybridResult, FfiError> {
    Ok(solve_powell_hybrid(
        &initial,
        options.into(),
        |point| {
            system
                .residual(point.to_vec())
                .ok_or_else(|| callback_failure("residual"))
        },
        |point| {
            system
                .jacobian(point.to_vec())
                .ok_or_else(|| callback_failure("Jacobian"))
        },
    )?
    .into())
}

#[uniffi::export]
pub fn run_powell_hybrid_model(
    initial: Vec<f64>,
    equations: Vec<FfiNonlinearExpression>,
    options: FfiPowellHybridOptions,
) -> Result<FfiPowellHybridResult, FfiError> {
    let model = NonlinearEquationModel::new(
        initial.len(),
        equations.into_iter().map(expression).collect(),
    )?;
    Ok(solve_powell_hybrid_model(&model, &initial, options.into())?.into())
}

fn callback_failure(kind: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(format!("Powell hybrid {kind} callback failed"))
}

impl From<FfiPowellHybridOptions> for PowellHybridOptions {
    fn from(o: FfiPowellHybridOptions) -> Self {
        Self {
            max_iterations: o.max_iterations,
            max_residual_evaluations: o.max_residual_evaluations,
            residual_tolerance: o.residual_tolerance,
            step_tolerance: o.step_tolerance,
            initial_radius: o.initial_radius,
            maximum_radius: o.maximum_radius,
            variable_scale: o.variable_scale,
            residual_scale: o.residual_scale,
        }
    }
}

impl From<PowellHybridResult> for FfiPowellHybridResult {
    fn from(r: PowellHybridResult) -> Self {
        use FfiPowellHybridTermination as FfiTermination;
        use PowellHybridTermination::*;
        Self {
            point: r.point,
            residuals: r.residuals,
            residual_norm: r.residual_norm,
            iterations: r.iterations,
            residual_evaluations: r.residual_evaluations,
            jacobian_evaluations: r.jacobian_evaluations,
            rank_one_updates: r.rank_one_updates,
            accepted_steps: r.accepted_steps,
            final_radius: r.final_radius,
            termination: match r.termination {
                ConvergedResidual => FfiTermination::ConvergedResidual,
                Stalled => FfiTermination::Stalled,
                EvaluationLimit => FfiTermination::EvaluationLimit,
                IterationLimit => FfiTermination::IterationLimit,
                NumericalFailure => FfiTermination::NumericalFailure,
            },
        }
    }
}
