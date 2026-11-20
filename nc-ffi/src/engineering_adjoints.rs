use std::sync::Arc;

use crate::{nonlinear::expression, FfiError, FfiNonlinearExpression};
use nc_optimize as opt;

#[derive(uniffi::Enum)]
pub enum FfiEngineeringUnknownRole {
    Differential,
    Algebraic,
    Multiplier,
    Reference,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringUnknown {
    pub name: String,
    pub unit: String,
    pub role: FfiEngineeringUnknownRole,
    pub scale: f64,
    pub initial_value: f64,
    pub initial_derivative: f64,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringParameter {
    pub name: String,
    pub unit: String,
    pub value: f64,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringPort {
    pub name: String,
    pub unknown: u32,
    pub orientation: f64,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringEquation {
    pub row: u32,
    pub expression: FfiNonlinearExpression,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringElement {
    pub name: String,
    pub kind: String,
    pub ports: Vec<FfiEngineeringPort>,
    pub equations: Vec<FfiEngineeringEquation>,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringModel {
    pub unknowns: Vec<FfiEngineeringUnknown>,
    pub parameters: Vec<FfiEngineeringParameter>,
    pub elements: Vec<FfiEngineeringElement>,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringResponseGradient {
    pub name: String,
    pub state_gradient: Vec<f64>,
    pub direct_parameter_gradient: Vec<f64>,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringAdjointResponse {
    pub name: String,
    pub adjoint: Vec<f64>,
    pub parameter_gradient: Vec<f64>,
    pub adjoint_residual_norm: f64,
    pub adjoint_backward_error: f64,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringAdjointResult {
    pub responses: Vec<FfiEngineeringAdjointResponse>,
    pub operating_residual_norm: f64,
}

#[uniffi::export]
pub fn solve_engineering_point_adjoints(
    model: FfiEngineeringModel,
    time: f64,
    state: Vec<f64>,
    derivative: Vec<f64>,
    alpha: f64,
    responses: Vec<FfiEngineeringResponseGradient>,
) -> Result<FfiEngineeringAdjointResult, FfiError> {
    let requests: Vec<_> = responses
        .into_iter()
        .map(|r| opt::EngineeringResponseGradient {
            name: r.name,
            state_gradient: r.state_gradient,
            direct_parameter_gradient: r.direct_parameter_gradient,
        })
        .collect();
    let result = engineering_model(model)?.adjoint_sensitivities(
        time,
        &state,
        &derivative,
        alpha,
        &requests,
    )?;
    Ok(FfiEngineeringAdjointResult {
        responses: result
            .responses
            .into_iter()
            .map(|r| FfiEngineeringAdjointResponse {
                name: r.name,
                adjoint: r.adjoint,
                parameter_gradient: r.parameter_gradient,
                adjoint_residual_norm: r.adjoint_residual_norm,
                adjoint_backward_error: r.adjoint_backward_error,
            })
            .collect(),
        operating_residual_norm: result.operating_residual_norm,
    })
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringTrajectoryOptions {
    pub checkpoint_stride: u64,
    pub max_newton_iterations: u64,
    pub residual_tolerance: f64,
    pub linear_backward_error_tolerance: f64,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringTrajectoryCheckpoint {
    pub step: u64,
    pub time: f64,
    pub state: Vec<f64>,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringTrajectoryResponse {
    pub name: String,
    pub running: Option<FfiNonlinearExpression>,
    pub terminal: Option<FfiNonlinearExpression>,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringTrajectoryGradient {
    pub name: String,
    pub value: f64,
    pub parameter_gradient: Vec<f64>,
    pub initial_state_gradient: Vec<f64>,
    pub maximum_adjoint_backward_error: f64,
}

#[derive(uniffi::Record)]
pub struct FfiEngineeringTrajectoryAdjoints {
    pub responses: Vec<FfiEngineeringTrajectoryGradient>,
    pub maximum_primal_residual: f64,
    pub replayed_steps: u64,
    pub peak_segment_states: u64,
    pub transpose_solves: u64,
    pub symbolic_analyses: u64,
}

#[derive(uniffi::Object)]
pub struct FfiEngineeringTrajectory {
    trajectory: opt::EngineeringTrajectory,
}

#[uniffi::export]
pub fn integrate_engineering_trajectory(
    model: FfiEngineeringModel,
    times: Vec<f64>,
    initial_state: Vec<f64>,
    options: FfiEngineeringTrajectoryOptions,
) -> Result<Arc<FfiEngineeringTrajectory>, FfiError> {
    let size = |value| {
        usize::try_from(value).map_err(|_| FfiError::SolverError {
            message: "engineering trajectory option outside platform range".into(),
        })
    };
    let trajectory = opt::EngineeringTrajectory::integrate(
        engineering_model(model)?,
        &times,
        &initial_state,
        opt::EngineeringTrajectoryOptions {
            checkpoint_stride: size(options.checkpoint_stride)?,
            max_newton_iterations: size(options.max_newton_iterations)?,
            residual_tolerance: options.residual_tolerance,
            linear_backward_error_tolerance: options.linear_backward_error_tolerance,
        },
    )?;
    Ok(Arc::new(FfiEngineeringTrajectory { trajectory }))
}

#[uniffi::export]
impl FfiEngineeringTrajectory {
    pub fn checkpoints(&self) -> Vec<FfiEngineeringTrajectoryCheckpoint> {
        self.trajectory
            .checkpoints()
            .iter()
            .map(|point| FfiEngineeringTrajectoryCheckpoint {
                step: point.step as u64,
                time: point.time,
                state: point.state.clone(),
            })
            .collect()
    }

    pub fn adjoints(
        &self,
        responses: Vec<FfiEngineeringTrajectoryResponse>,
    ) -> Result<FfiEngineeringTrajectoryAdjoints, FfiError> {
        let responses: Vec<_> = responses
            .into_iter()
            .map(|r| opt::EngineeringTrajectoryResponse {
                name: r.name,
                running: r.running.map(expression),
                terminal: r.terminal.map(expression),
            })
            .collect();
        let result = self.trajectory.adjoints(&responses)?;
        Ok(FfiEngineeringTrajectoryAdjoints {
            responses: result
                .responses
                .into_iter()
                .map(|r| FfiEngineeringTrajectoryGradient {
                    name: r.name,
                    value: r.value,
                    parameter_gradient: r.parameter_gradient,
                    initial_state_gradient: r.initial_state_gradient,
                    maximum_adjoint_backward_error: r.maximum_adjoint_backward_error,
                })
                .collect(),
            maximum_primal_residual: result.maximum_primal_residual,
            replayed_steps: result.replayed_steps as u64,
            peak_segment_states: result.peak_segment_states as u64,
            transpose_solves: result.transpose_solves as u64,
            symbolic_analyses: result.symbolic_analyses as u64,
        })
    }
}

fn engineering_model(model: FfiEngineeringModel) -> Result<opt::EngineeringModel, FfiError> {
    let unknowns = model
        .unknowns
        .into_iter()
        .map(|u| opt::EngineeringUnknown {
            name: u.name,
            unit: u.unit,
            scale: u.scale,
            initial_value: u.initial_value,
            initial_derivative: u.initial_derivative,
            role: match u.role {
                FfiEngineeringUnknownRole::Differential => {
                    opt::EngineeringUnknownRole::Differential
                }
                FfiEngineeringUnknownRole::Algebraic => opt::EngineeringUnknownRole::Algebraic,
                FfiEngineeringUnknownRole::Multiplier => opt::EngineeringUnknownRole::Multiplier,
                FfiEngineeringUnknownRole::Reference => opt::EngineeringUnknownRole::Reference,
            },
        })
        .collect();
    let parameters = model
        .parameters
        .into_iter()
        .map(|p| opt::EngineeringParameter {
            name: p.name,
            unit: p.unit,
            value: p.value,
        })
        .collect();
    let elements = model
        .elements
        .into_iter()
        .map(|e| opt::EngineeringElement {
            name: e.name,
            kind: e.kind,
            ports: e
                .ports
                .into_iter()
                .map(|p| opt::EngineeringPort {
                    name: p.name,
                    unknown: p.unknown as usize,
                    orientation: p.orientation,
                })
                .collect(),
            equations: e
                .equations
                .into_iter()
                .map(|eq| opt::EngineeringEquation {
                    row: eq.row as usize,
                    expression: expression(eq.expression),
                })
                .collect(),
        })
        .collect();
    Ok(opt::EngineeringModel::new(unknowns, parameters, elements)?)
}
