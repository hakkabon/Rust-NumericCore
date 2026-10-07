//! Stable value-only FFI transport for the shared nonlinear graph.
//!
//! Callbacks do not cross UniFFI. Instead Swift sends the Phase 9 expression
//! graph once and Rust performs every value/derivative evaluation locally.

use crate::FfiError;
use nc_optimize::{
    minimize_constrained, minimize_model_lbfgsb, solve_model_least_squares, Bound,
    ConstrainedNonlinearProblem, ConstrainedOptions, ConstrainedResult, ConstrainedTermination,
    ConstraintMultiplier, LbfgsOptions, LbfgsResult, NonlinearConstraint, NonlinearExpression,
    NonlinearLeastSquaresOptions, NonlinearLeastSquaresResult, NonlinearModel, NonlinearNode,
    NonlinearTermination, RobustLoss,
};

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiNonlinearNodeKind {
    Constant,
    Parameter,
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Pow,
}

/// Flat graph node. `first`/`second` are operand indices; `value` carries a
/// constant or power exponent. Fields unused by a given `kind` are ignored.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNonlinearNode {
    pub kind: FfiNonlinearNodeKind,
    pub first: u32,
    pub second: u32,
    pub value: f64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNonlinearExpression {
    pub nodes: Vec<FfiNonlinearNode>,
    pub output: u32,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNonlinearModel {
    pub parameter_count: u32,
    pub bounds: Vec<crate::FfiBound>,
    pub objective: Option<FfiNonlinearExpression>,
    pub residuals: Vec<FfiNonlinearExpression>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNonlinearConstraint {
    pub expression: FfiNonlinearExpression,
    pub bound: crate::FfiBound,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiLbfgsOptions {
    pub max_iterations: u64,
    pub history_size: u64,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub objective_tolerance: f64,
    pub max_line_search_iterations: u64,
    pub armijo: f64,
    pub wolfe: f64,
    pub backtracking: f64,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiNonlinearTermination {
    ConvergedGradient,
    ConvergedStep,
    ConvergedObjective,
    IterationLimit,
    LineSearchFailed,
    DampingLimit,
    Cancelled,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLbfgsResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub gradient: Vec<f64>,
    pub iterations: u64,
    pub evaluations: u64,
    pub termination: FfiNonlinearTermination,
    pub gradient_norm: f64,
    pub accepted_step: Option<f64>,
    pub stored_curvature_pairs: u64,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiRobustLossKind {
    Squared,
    Huber,
    Cauchy,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiRobustLoss {
    pub kind: FfiRobustLossKind,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiNonlinearLeastSquaresOptions {
    pub max_iterations: u64,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
    pub initial_damping: f64,
    pub damping_increase: f64,
    pub damping_decrease: f64,
    pub max_damping_iterations: u64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNonlinearLeastSquaresResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub cost: f64,
    pub gradient_norm: f64,
    pub iterations: u64,
    pub evaluations: u64,
    pub termination: FfiNonlinearTermination,
    pub final_damping: f64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiConstrainedOptions {
    pub max_outer_iterations: u64,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub initial_penalty: f64,
    pub penalty_increase: f64,
    pub maximum_penalty: f64,
    pub inner_options: FfiLbfgsOptions,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiConstrainedTermination {
    Converged,
    IterationLimit,
    PenaltyLimit,
    Cancelled,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiConstraintMultiplier {
    pub lower: f64,
    pub upper: f64,
    pub equality: f64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiConstrainedResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<FfiConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub outer_iterations: u64,
    pub inner_iterations: u64,
    pub evaluations: u64,
    pub final_penalty: f64,
    pub termination: FfiConstrainedTermination,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseDerivative {
    pub dimension: u64,
    pub indices: Vec<u64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseJacobian {
    pub rows: u64,
    pub columns: u64,
    pub row_pointers: Vec<u64>,
    pub column_indices: Vec<u64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseObjectiveEvaluation {
    pub value: f64,
    pub derivative: FfiSparseDerivative,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseResidualEvaluation {
    pub residuals: Vec<f64>,
    pub jacobian: FfiSparseJacobian,
}

fn count(value: u64, name: &str) -> Result<usize, FfiError> {
    usize::try_from(value).map_err(|_| FfiError::SolverError {
        message: format!("{name} does not fit this platform"),
    })
}

fn bound(value: crate::FfiBound) -> Bound {
    Bound {
        lower: value.lower,
        upper: value.upper,
    }
}

fn expression(value: FfiNonlinearExpression) -> NonlinearExpression {
    let nodes = value
        .nodes
        .into_iter()
        .map(|node| match node.kind {
            FfiNonlinearNodeKind::Constant => NonlinearNode::Constant(node.value),
            FfiNonlinearNodeKind::Parameter => NonlinearNode::Parameter(node.first as usize),
            FfiNonlinearNodeKind::Add => {
                NonlinearNode::Add(node.first as usize, node.second as usize)
            }
            FfiNonlinearNodeKind::Subtract => {
                NonlinearNode::Subtract(node.first as usize, node.second as usize)
            }
            FfiNonlinearNodeKind::Multiply => {
                NonlinearNode::Multiply(node.first as usize, node.second as usize)
            }
            FfiNonlinearNodeKind::Divide => {
                NonlinearNode::Divide(node.first as usize, node.second as usize)
            }
            FfiNonlinearNodeKind::Negate => NonlinearNode::Negate(node.first as usize),
            FfiNonlinearNodeKind::Exp => NonlinearNode::Exp(node.first as usize),
            FfiNonlinearNodeKind::Log => NonlinearNode::Log(node.first as usize),
            FfiNonlinearNodeKind::Sqrt => NonlinearNode::Sqrt(node.first as usize),
            FfiNonlinearNodeKind::Sin => NonlinearNode::Sin(node.first as usize),
            FfiNonlinearNodeKind::Cos => NonlinearNode::Cos(node.first as usize),
            FfiNonlinearNodeKind::Pow => NonlinearNode::Powf(node.first as usize, node.value),
        })
        .collect();
    NonlinearExpression::new(nodes, value.output as usize)
}

fn model(value: FfiNonlinearModel) -> NonlinearModel {
    NonlinearModel {
        parameter_count: value.parameter_count as usize,
        bounds: value.bounds.into_iter().map(bound).collect(),
        objective: value.objective.map(expression),
        residuals: value.residuals.into_iter().map(expression).collect(),
    }
}

fn lbfgs_options(value: FfiLbfgsOptions) -> Result<LbfgsOptions, FfiError> {
    Ok(LbfgsOptions {
        max_iterations: count(value.max_iterations, "max_iterations")?,
        history_size: count(value.history_size, "history_size")?,
        gradient_tolerance: value.gradient_tolerance,
        step_tolerance: value.step_tolerance,
        objective_tolerance: value.objective_tolerance,
        max_line_search_iterations: count(
            value.max_line_search_iterations,
            "max_line_search_iterations",
        )?,
        armijo: value.armijo,
        wolfe: value.wolfe,
        backtracking: value.backtracking,
    })
}

fn termination(value: NonlinearTermination) -> FfiNonlinearTermination {
    match value {
        NonlinearTermination::ConvergedGradient => FfiNonlinearTermination::ConvergedGradient,
        NonlinearTermination::ConvergedStep => FfiNonlinearTermination::ConvergedStep,
        NonlinearTermination::ConvergedObjective => FfiNonlinearTermination::ConvergedObjective,
        NonlinearTermination::IterationLimit => FfiNonlinearTermination::IterationLimit,
        NonlinearTermination::LineSearchFailed => FfiNonlinearTermination::LineSearchFailed,
        NonlinearTermination::DampingLimit => FfiNonlinearTermination::DampingLimit,
        NonlinearTermination::Cancelled => FfiNonlinearTermination::Cancelled,
    }
}

fn lbfgs_result(value: LbfgsResult) -> FfiLbfgsResult {
    FfiLbfgsResult {
        point: value.point,
        objective: value.objective,
        gradient: value.gradient,
        iterations: value.iterations as u64,
        evaluations: value.evaluations as u64,
        termination: termination(value.termination),
        gradient_norm: value.gradient_norm,
        accepted_step: value.accepted_step,
        stored_curvature_pairs: value.stored_curvature_pairs as u64,
    }
}

#[uniffi::export]
pub fn solve_nonlinear_objective(
    model_value: FfiNonlinearModel,
    initial: Vec<f64>,
    options: FfiLbfgsOptions,
) -> Result<FfiLbfgsResult, FfiError> {
    Ok(lbfgs_result(minimize_model_lbfgsb(
        &model(model_value),
        &initial,
        lbfgs_options(options)?,
    )?))
}

#[uniffi::export]
pub fn solve_nonlinear_least_squares(
    model_value: FfiNonlinearModel,
    initial: Vec<f64>,
    weights: Vec<f64>,
    loss: FfiRobustLoss,
    options: FfiNonlinearLeastSquaresOptions,
) -> Result<FfiNonlinearLeastSquaresResult, FfiError> {
    let loss = match loss.kind {
        FfiRobustLossKind::Squared => RobustLoss::Squared,
        FfiRobustLossKind::Huber => RobustLoss::Huber { scale: loss.scale },
        FfiRobustLossKind::Cauchy => RobustLoss::Cauchy { scale: loss.scale },
    };
    let options = NonlinearLeastSquaresOptions {
        max_iterations: count(options.max_iterations, "max_iterations")?,
        gradient_tolerance: options.gradient_tolerance,
        step_tolerance: options.step_tolerance,
        cost_tolerance: options.cost_tolerance,
        initial_damping: options.initial_damping,
        damping_increase: options.damping_increase,
        damping_decrease: options.damping_decrease,
        max_damping_iterations: count(options.max_damping_iterations, "max_damping_iterations")?,
    };
    let value = solve_model_least_squares(&model(model_value), &initial, &weights, loss, options)?;
    Ok(nls_result(value))
}

fn nls_result(value: NonlinearLeastSquaresResult) -> FfiNonlinearLeastSquaresResult {
    FfiNonlinearLeastSquaresResult {
        point: value.point,
        residuals: value.residuals,
        cost: value.cost,
        gradient_norm: value.gradient_norm,
        iterations: value.iterations as u64,
        evaluations: value.evaluations as u64,
        termination: termination(value.termination),
        final_damping: value.final_damping,
        accepted_steps: value.accepted_steps as u64,
        rejected_steps: value.rejected_steps as u64,
    }
}

#[uniffi::export]
pub fn solve_constrained_nonlinear(
    model_value: FfiNonlinearModel,
    constraints: Vec<FfiNonlinearConstraint>,
    initial: Vec<f64>,
    options: FfiConstrainedOptions,
) -> Result<FfiConstrainedResult, FfiError> {
    let problem = ConstrainedNonlinearProblem {
        model: model(model_value),
        constraints: constraints
            .into_iter()
            .map(|value| NonlinearConstraint {
                expression: expression(value.expression),
                bound: bound(value.bound),
            })
            .collect(),
    };
    let options = ConstrainedOptions {
        max_outer_iterations: count(options.max_outer_iterations, "max_outer_iterations")?,
        feasibility_tolerance: options.feasibility_tolerance,
        stationarity_tolerance: options.stationarity_tolerance,
        initial_penalty: options.initial_penalty,
        penalty_increase: options.penalty_increase,
        maximum_penalty: options.maximum_penalty,
        inner_options: lbfgs_options(options.inner_options)?,
    };
    Ok(constrained_result(minimize_constrained(
        &problem, &initial, options,
    )?))
}

fn constrained_result(value: ConstrainedResult) -> FfiConstrainedResult {
    FfiConstrainedResult {
        point: value.point,
        objective: value.objective,
        constraint_values: value.constraint_values,
        multipliers: value
            .multipliers
            .into_iter()
            .map(|m: ConstraintMultiplier| FfiConstraintMultiplier {
                lower: m.lower,
                upper: m.upper,
                equality: m.equality,
            })
            .collect(),
        maximum_violation: value.maximum_violation,
        stationarity_norm: value.stationarity_norm,
        outer_iterations: value.outer_iterations as u64,
        inner_iterations: value.inner_iterations as u64,
        evaluations: value.evaluations as u64,
        final_penalty: value.final_penalty,
        termination: match value.termination {
            ConstrainedTermination::Converged => FfiConstrainedTermination::Converged,
            ConstrainedTermination::IterationLimit => FfiConstrainedTermination::IterationLimit,
            ConstrainedTermination::PenaltyLimit => FfiConstrainedTermination::PenaltyLimit,
            ConstrainedTermination::Cancelled => FfiConstrainedTermination::Cancelled,
        },
    }
}

fn sparse_derivative(value: nc_optimize::SparseDerivative) -> FfiSparseDerivative {
    FfiSparseDerivative {
        dimension: value.dimension as u64,
        indices: value
            .indices
            .into_iter()
            .map(|index| index as u64)
            .collect(),
        values: value.values,
    }
}

fn sparse_jacobian(value: nc_optimize::SparseJacobian) -> FfiSparseJacobian {
    FfiSparseJacobian {
        rows: value.rows as u64,
        columns: value.columns as u64,
        row_pointers: value
            .row_pointers
            .into_iter()
            .map(|index| index as u64)
            .collect(),
        column_indices: value
            .column_indices
            .into_iter()
            .map(|index| index as u64)
            .collect(),
        values: value.values,
    }
}

#[uniffi::export]
pub fn evaluate_nonlinear_objective_sparse(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
) -> Result<FfiSparseObjectiveEvaluation, FfiError> {
    let model = model(model_value);
    model.validate()?;
    let expression = model
        .objective
        .as_ref()
        .ok_or_else(|| FfiError::SolverError {
            message: "model does not contain an objective".to_owned(),
        })?;
    let (value, derivative) = expression.evaluate_sparse(&parameters)?;
    Ok(FfiSparseObjectiveEvaluation {
        value,
        derivative: sparse_derivative(derivative),
    })
}

#[uniffi::export]
pub fn evaluate_nonlinear_residuals_sparse(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
) -> Result<FfiSparseResidualEvaluation, FfiError> {
    let (residuals, jacobian) = model(model_value).evaluate_sparse_residuals(&parameters)?;
    Ok(FfiSparseResidualEvaluation {
        residuals,
        jacobian: sparse_jacobian(jacobian),
    })
}

#[uniffi::export]
pub fn nonlinear_jacobian_vector_product(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
    direction: Vec<f64>,
) -> Result<Vec<f64>, FfiError> {
    Ok(model(model_value).jacobian_vector_product(&parameters, &direction)?)
}

#[uniffi::export]
pub fn nonlinear_jacobian_transpose_vector_product(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
    weights: Vec<f64>,
) -> Result<Vec<f64>, FfiError> {
    Ok(model(model_value).jacobian_transpose_vector_product(&parameters, &weights)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(kind: FfiNonlinearNodeKind, first: u32, second: u32, value: f64) -> FfiNonlinearNode {
        FfiNonlinearNode {
            kind,
            first,
            second,
            value,
        }
    }
    fn options() -> FfiLbfgsOptions {
        FfiLbfgsOptions {
            max_iterations: 1000,
            history_size: 10,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-12,
            objective_tolerance: 1e-12,
            max_line_search_iterations: 30,
            armijo: 1e-4,
            wolfe: 0.9,
            backtracking: 0.5,
        }
    }

    #[test]
    fn objective_graph_solves_across_transport() {
        let expression = FfiNonlinearExpression {
            nodes: vec![
                node(FfiNonlinearNodeKind::Parameter, 0, 0, 0.0),
                node(FfiNonlinearNodeKind::Constant, 0, 0, 3.0),
                node(FfiNonlinearNodeKind::Subtract, 0, 1, 0.0),
                node(FfiNonlinearNodeKind::Pow, 2, 0, 2.0),
            ],
            output: 3,
        };
        let result = solve_nonlinear_objective(
            FfiNonlinearModel {
                parameter_count: 1,
                bounds: vec![crate::FfiBound {
                    lower: None,
                    upper: Some(1.0),
                }],
                objective: Some(expression),
                residuals: vec![],
            },
            vec![0.0],
            options(),
        )
        .unwrap();
        assert!((result.point[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn malformed_graph_fails_as_solver_error() {
        let result = solve_nonlinear_objective(
            FfiNonlinearModel {
                parameter_count: 1,
                bounds: vec![crate::FfiBound {
                    lower: None,
                    upper: None,
                }],
                objective: Some(FfiNonlinearExpression {
                    nodes: vec![node(FfiNonlinearNodeKind::Add, 0, 0, 0.0)],
                    output: 0,
                }),
                residuals: vec![],
            },
            vec![0.0],
            options(),
        );
        assert!(matches!(result, Err(FfiError::SolverError { .. })));
    }

    #[test]
    fn sparse_and_matrix_free_derivatives_cross_transport() {
        let residual = FfiNonlinearExpression {
            nodes: vec![
                node(FfiNonlinearNodeKind::Parameter, 2, 0, 0.0),
                node(FfiNonlinearNodeKind::Constant, 0, 0, 4.0),
                node(FfiNonlinearNodeKind::Multiply, 0, 1, 0.0),
            ],
            output: 2,
        };
        let model = FfiNonlinearModel {
            parameter_count: 4,
            bounds: vec![
                crate::FfiBound {
                    lower: None,
                    upper: None,
                };
                4
            ],
            objective: None,
            residuals: vec![residual],
        };
        let point = vec![1.0, 2.0, 3.0, 4.0];
        let sparse = evaluate_nonlinear_residuals_sparse(model.clone(), point.clone()).unwrap();
        assert_eq!(sparse.residuals, vec![12.0]);
        assert_eq!(sparse.jacobian.row_pointers, vec![0, 1]);
        assert_eq!(sparse.jacobian.column_indices, vec![2]);
        assert_eq!(sparse.jacobian.values, vec![4.0]);
        assert_eq!(
            nonlinear_jacobian_vector_product(
                model.clone(),
                point.clone(),
                vec![5.0, 6.0, 7.0, 8.0]
            )
            .unwrap(),
            vec![28.0]
        );
        assert_eq!(
            nonlinear_jacobian_transpose_vector_product(model, point, vec![3.0]).unwrap(),
            vec![0.0, 0.0, 12.0, 0.0]
        );
    }
}
