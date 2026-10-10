//! Stable value-only FFI transport for the shared nonlinear graph.
//!
//! Callbacks do not cross UniFFI. Instead Swift sends the Phase 9 expression
//! graph once and Rust performs every value/derivative evaluation locally.

use crate::FfiError;
use nc_optimize::{
    minimize_constrained, minimize_mixed_integer_nonlinear, minimize_model_lbfgsb,
    minimize_nonlinear_interior_point, minimize_sqp, solve_model_least_squares,
    solve_model_least_squares_matrix_free, Bound,
    ConstrainedNonlinearProblem, ConstrainedOptions, ConstrainedResult, ConstrainedTermination,
    ConstraintMultiplier, LbfgsOptions, LbfgsResult, MatrixFreeLeastSquaresOptions,
    MixedIntegerNonlinearOptions,
    MixedIntegerNonlinearProblem, MixedIntegerNonlinearResult, MixedIntegerNonlinearTermination,
    NonlinearConstraint, NonlinearExpression, NonlinearInteriorPointOptions,
    NonlinearInteriorPointResult, NonlinearInteriorPointTermination, NonlinearLeastSquaresOptions,
    NonlinearLeastSquaresResult, NonlinearModel, NonlinearNode, NonlinearRelaxationStrategy,
    NonlinearTermination, QuadraticOptions, RobustLoss, SqpCurvature, SqpGlobalization,
    SqpOptions, SqpResult, SqpTermination, FeasibilityRestorationOptions,
    FeasibilityRestorationTermination,
    restore_feasibility, solve_sparse_kkt, SparseHessian, SparseJacobian, SparseKktProblem,
};

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiFeasibilityRestorationOptions {
    pub max_iterations: u64,
    pub feasibility_tolerance: f64,
    pub interior_margin: f64,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiFeasibilityRestorationTermination {
    AlreadyFeasible,
    Converged,
    IterationLimit,
    Stalled,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiFeasibilityRestorationResult {
    pub point: Vec<f64>,
    pub maximum_violation: f64,
    pub squared_violation: f64,
    pub iterations: u64,
    pub evaluations: u64,
    pub termination: FfiFeasibilityRestorationTermination,
}

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
pub struct FfiMatrixFreeLeastSquaresOptions {
    pub outer: FfiNonlinearLeastSquaresOptions,
    pub max_krylov_iterations: u64,
    pub krylov_tolerance: f64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMatrixFreeLeastSquaresResult {
    pub solution: FfiNonlinearLeastSquaresResult,
    pub krylov_iterations: u64,
    pub jacobian_products: u64,
    pub transpose_jacobian_products: u64,
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

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiNonlinearInteriorPointOptions {
    pub max_outer_iterations: u64,
    pub max_inner_iterations: u64,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub complementarity_tolerance: f64,
    pub initial_barrier: f64,
    pub barrier_reduction: f64,
    pub minimum_barrier: f64,
    pub equality_penalty: f64,
    pub armijo: f64,
    pub backtracking: f64,
    pub fraction_to_boundary: f64,
    pub max_line_search_iterations: u64,
    pub restoration: bool,
    pub restoration_options: FfiFeasibilityRestorationOptions,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiNonlinearInteriorPointTermination {
    Converged,
    IterationLimit,
    InfeasibleStart,
    RestorationFailed,
    LineSearchFailed,
    NumericalFailure,
    Cancelled,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNonlinearInteriorPointResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<FfiConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub complementarity: f64,
    pub outer_iterations: u64,
    pub inner_iterations: u64,
    pub evaluations: u64,
    pub final_barrier: f64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
    pub termination: FfiNonlinearInteriorPointTermination,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiNonlinearRelaxationStrategy {
    Sqp,
    AugmentedLagrangian,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiMixedIntegerNonlinearOptions {
    pub max_nodes: u64,
    pub integer_tolerance: f64,
    pub feasibility_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub relaxation_strategy: FfiNonlinearRelaxationStrategy,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiMixedIntegerNonlinearTermination {
    SearchExhausted,
    LocalGapLimit,
    NodeLimit,
    Infeasible,
    RelaxationFailure,
    Cancelled,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMixedIntegerNonlinearResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<FfiConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub nodes_explored: u64,
    pub relaxations_solved: u64,
    pub nodes_pruned_infeasible: u64,
    pub maximum_depth: u64,
    pub incumbents_found: u64,
    pub best_relaxation_objective: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub global_optimality_certified: bool,
    pub termination: FfiMixedIntegerNonlinearTermination,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiSqpOptions {
    pub max_iterations: u64,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub step_tolerance: f64,
    pub merit_penalty: f64,
    pub penalty_increase: f64,
    pub armijo: f64,
    pub backtracking: f64,
    pub max_line_search_iterations: u64,
    pub hessian_regularization: f64,
    pub qp_max_iterations: u64,
    pub qp_rho: f64,
    pub qp_absolute_tolerance: f64,
    pub qp_relative_tolerance: f64,
    pub qp_convexity_tolerance: f64,
    pub restoration: bool,
    pub restoration_options: FfiFeasibilityRestorationOptions,
    pub globalization: FfiSqpGlobalization,
    pub filter_constraint_margin: f64,
    pub filter_objective_margin: f64,
    pub curvature: FfiSqpCurvature,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiSqpGlobalization { Merit, Filter }

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiSqpCurvature { Bfgs, ExactLagrangian }

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiSqpTermination {
    Converged,
    IterationLimit,
    StepLimit,
    LineSearchFailed,
    QpFailure,
    RestorationFailed,
    Cancelled,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSqpResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<FfiConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub iterations: u64,
    pub evaluations: u64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
    pub final_merit_penalty: f64,
    pub last_step_norm: f64,
    pub termination: FfiSqpTermination,
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
pub struct FfiSecondOrderValue {
    pub value: f64,
    pub gradient: Vec<f64>,
    /// Row-major square Hessian.
    pub hessian: Vec<f64>,
    pub dimension: u64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseHessian {
    pub dimension: u64,
    pub row_pointers: Vec<u64>,
    pub column_indices: Vec<u64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseKktResult {
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub factor_nonzeros: u64,
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

#[uniffi::export]
pub fn solve_nonlinear_least_squares_matrix_free(
    model_value: FfiNonlinearModel,
    initial: Vec<f64>,
    weights: Vec<f64>,
    loss: FfiRobustLoss,
    options: FfiMatrixFreeLeastSquaresOptions,
) -> Result<FfiMatrixFreeLeastSquaresResult, FfiError> {
    let loss = match loss.kind {
        FfiRobustLossKind::Squared => RobustLoss::Squared,
        FfiRobustLossKind::Huber => RobustLoss::Huber { scale: loss.scale },
        FfiRobustLossKind::Cauchy => RobustLoss::Cauchy { scale: loss.scale },
    };
    let o = options.outer;
    let outer = NonlinearLeastSquaresOptions {
        max_iterations: count(o.max_iterations, "max_iterations")?,
        gradient_tolerance: o.gradient_tolerance,
        step_tolerance: o.step_tolerance,
        cost_tolerance: o.cost_tolerance,
        initial_damping: o.initial_damping,
        damping_increase: o.damping_increase,
        damping_decrease: o.damping_decrease,
        max_damping_iterations: count(o.max_damping_iterations, "max_damping_iterations")?,
    };
    let value = solve_model_least_squares_matrix_free(
        &model(model_value), &initial, &weights, loss,
        MatrixFreeLeastSquaresOptions {
            outer,
            max_krylov_iterations: count(options.max_krylov_iterations, "max_krylov_iterations")?,
            krylov_tolerance: options.krylov_tolerance,
        },
    )?;
    Ok(FfiMatrixFreeLeastSquaresResult {
        solution: nls_result(value.solution),
        krylov_iterations: value.krylov_iterations as u64,
        jacobian_products: value.jacobian_products as u64,
        transpose_jacobian_products: value.transpose_jacobian_products as u64,
    })
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

fn restoration_options(value: FfiFeasibilityRestorationOptions)
    -> Result<FeasibilityRestorationOptions, FfiError> {
    Ok(FeasibilityRestorationOptions {
        max_iterations: count(value.max_iterations, "restoration max_iterations")?,
        feasibility_tolerance: value.feasibility_tolerance,
        interior_margin: value.interior_margin,
    })
}

#[uniffi::export]
pub fn restore_nonlinear_feasibility(
    model_value: FfiNonlinearModel,
    constraints: Vec<FfiNonlinearConstraint>,
    initial: Vec<f64>,
    options: FfiFeasibilityRestorationOptions,
) -> Result<FfiFeasibilityRestorationResult, FfiError> {
    let problem = ConstrainedNonlinearProblem {
        model: model(model_value),
        constraints: constraints.into_iter().map(|value| NonlinearConstraint {
            expression: expression(value.expression), bound: bound(value.bound),
        }).collect(),
    };
    let value = restore_feasibility(&problem, &initial, restoration_options(options)?)?;
    Ok(FfiFeasibilityRestorationResult {
        point: value.point,
        maximum_violation: value.maximum_violation,
        squared_violation: value.squared_violation,
        iterations: value.iterations as u64,
        evaluations: value.evaluations as u64,
        termination: match value.termination {
            FeasibilityRestorationTermination::AlreadyFeasible =>
                FfiFeasibilityRestorationTermination::AlreadyFeasible,
            FeasibilityRestorationTermination::Converged =>
                FfiFeasibilityRestorationTermination::Converged,
            FeasibilityRestorationTermination::IterationLimit =>
                FfiFeasibilityRestorationTermination::IterationLimit,
            FeasibilityRestorationTermination::Stalled =>
                FfiFeasibilityRestorationTermination::Stalled,
        },
    })
}

#[uniffi::export]
pub fn solve_nonlinear_interior_point(
    model_value: FfiNonlinearModel,
    constraints: Vec<FfiNonlinearConstraint>,
    initial: Vec<f64>,
    options: FfiNonlinearInteriorPointOptions,
) -> Result<FfiNonlinearInteriorPointResult, FfiError> {
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
    let options = NonlinearInteriorPointOptions {
        max_outer_iterations: count(options.max_outer_iterations, "max_outer_iterations")?,
        max_inner_iterations: count(options.max_inner_iterations, "max_inner_iterations")?,
        feasibility_tolerance: options.feasibility_tolerance,
        stationarity_tolerance: options.stationarity_tolerance,
        complementarity_tolerance: options.complementarity_tolerance,
        initial_barrier: options.initial_barrier,
        barrier_reduction: options.barrier_reduction,
        minimum_barrier: options.minimum_barrier,
        equality_penalty: options.equality_penalty,
        armijo: options.armijo,
        backtracking: options.backtracking,
        fraction_to_boundary: options.fraction_to_boundary,
        max_line_search_iterations: count(
            options.max_line_search_iterations,
            "max_line_search_iterations",
        )?,
        restoration: options.restoration,
        restoration_options: restoration_options(options.restoration_options)?,
    };
    Ok(interior_point_result(minimize_nonlinear_interior_point(
        &problem, &initial, options,
    )?))
}

fn interior_point_result(value: NonlinearInteriorPointResult) -> FfiNonlinearInteriorPointResult {
    FfiNonlinearInteriorPointResult {
        point: value.point,
        objective: value.objective,
        constraint_values: value.constraint_values,
        multipliers: value
            .multipliers
            .into_iter()
            .map(|multiplier| FfiConstraintMultiplier {
                lower: multiplier.lower,
                upper: multiplier.upper,
                equality: multiplier.equality,
            })
            .collect(),
        maximum_violation: value.maximum_violation,
        stationarity_norm: value.stationarity_norm,
        complementarity: value.complementarity,
        outer_iterations: value.outer_iterations as u64,
        inner_iterations: value.inner_iterations as u64,
        evaluations: value.evaluations as u64,
        final_barrier: value.final_barrier,
        accepted_steps: value.accepted_steps as u64,
        rejected_steps: value.rejected_steps as u64,
        termination: match value.termination {
            NonlinearInteriorPointTermination::Converged => {
                FfiNonlinearInteriorPointTermination::Converged
            }
            NonlinearInteriorPointTermination::IterationLimit => {
                FfiNonlinearInteriorPointTermination::IterationLimit
            }
            NonlinearInteriorPointTermination::InfeasibleStart => {
                FfiNonlinearInteriorPointTermination::InfeasibleStart
            }
            NonlinearInteriorPointTermination::RestorationFailed => {
                FfiNonlinearInteriorPointTermination::RestorationFailed
            }
            NonlinearInteriorPointTermination::LineSearchFailed => {
                FfiNonlinearInteriorPointTermination::LineSearchFailed
            }
            NonlinearInteriorPointTermination::NumericalFailure => {
                FfiNonlinearInteriorPointTermination::NumericalFailure
            }
            NonlinearInteriorPointTermination::Cancelled => {
                FfiNonlinearInteriorPointTermination::Cancelled
            }
        },
    }
}

#[uniffi::export]
pub fn solve_mixed_integer_nonlinear(
    model_value: FfiNonlinearModel,
    constraints: Vec<FfiNonlinearConstraint>,
    is_integer: Vec<bool>,
    initial: Vec<f64>,
    options: FfiMixedIntegerNonlinearOptions,
) -> Result<FfiMixedIntegerNonlinearResult, FfiError> {
    let problem = MixedIntegerNonlinearProblem {
        model: model(model_value),
        constraints: constraints
            .into_iter()
            .map(|value| NonlinearConstraint {
                expression: expression(value.expression),
                bound: bound(value.bound),
            })
            .collect(),
        is_integer,
    };
    let strategy = match options.relaxation_strategy {
        FfiNonlinearRelaxationStrategy::Sqp => NonlinearRelaxationStrategy::Sqp,
        FfiNonlinearRelaxationStrategy::AugmentedLagrangian => {
            NonlinearRelaxationStrategy::AugmentedLagrangian
        }
    };
    let value = minimize_mixed_integer_nonlinear(
        &problem,
        &initial,
        MixedIntegerNonlinearOptions {
            max_nodes: count(options.max_nodes, "max_nodes")?,
            integer_tolerance: options.integer_tolerance,
            feasibility_tolerance: options.feasibility_tolerance,
            absolute_gap_tolerance: options.absolute_gap_tolerance,
            relative_gap_tolerance: options.relative_gap_tolerance,
            relaxation_strategy: strategy,
            ..MixedIntegerNonlinearOptions::default()
        },
    )?;
    Ok(mixed_integer_nonlinear_result(value))
}

fn mixed_integer_nonlinear_result(
    value: MixedIntegerNonlinearResult,
) -> FfiMixedIntegerNonlinearResult {
    FfiMixedIntegerNonlinearResult {
        point: value.point,
        objective: value.objective,
        constraint_values: value.constraint_values,
        multipliers: value
            .multipliers
            .into_iter()
            .map(|multiplier| FfiConstraintMultiplier {
                lower: multiplier.lower,
                upper: multiplier.upper,
                equality: multiplier.equality,
            })
            .collect(),
        maximum_violation: value.maximum_violation,
        stationarity_norm: value.stationarity_norm,
        nodes_explored: value.nodes_explored as u64,
        relaxations_solved: value.relaxations_solved as u64,
        nodes_pruned_infeasible: value.nodes_pruned_infeasible as u64,
        maximum_depth: value.maximum_depth as u64,
        incumbents_found: value.incumbents_found as u64,
        best_relaxation_objective: value.best_relaxation_objective,
        absolute_gap: value.absolute_gap,
        relative_gap: value.relative_gap,
        global_optimality_certified: value.global_optimality_certified,
        termination: match value.termination {
            MixedIntegerNonlinearTermination::SearchExhausted => {
                FfiMixedIntegerNonlinearTermination::SearchExhausted
            }
            MixedIntegerNonlinearTermination::LocalGapLimit => {
                FfiMixedIntegerNonlinearTermination::LocalGapLimit
            }
            MixedIntegerNonlinearTermination::NodeLimit => {
                FfiMixedIntegerNonlinearTermination::NodeLimit
            }
            MixedIntegerNonlinearTermination::Infeasible => {
                FfiMixedIntegerNonlinearTermination::Infeasible
            }
            MixedIntegerNonlinearTermination::RelaxationFailure => {
                FfiMixedIntegerNonlinearTermination::RelaxationFailure
            }
            MixedIntegerNonlinearTermination::Cancelled => {
                FfiMixedIntegerNonlinearTermination::Cancelled
            }
        },
    }
}

#[uniffi::export]
pub fn solve_sqp(
    model_value: FfiNonlinearModel,
    constraints: Vec<FfiNonlinearConstraint>,
    initial: Vec<f64>,
    options: FfiSqpOptions,
) -> Result<FfiSqpResult, FfiError> {
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
    let options = SqpOptions {
        max_iterations: count(options.max_iterations, "max_iterations")?,
        feasibility_tolerance: options.feasibility_tolerance,
        stationarity_tolerance: options.stationarity_tolerance,
        step_tolerance: options.step_tolerance,
        merit_penalty: options.merit_penalty,
        penalty_increase: options.penalty_increase,
        armijo: options.armijo,
        backtracking: options.backtracking,
        max_line_search_iterations: count(
            options.max_line_search_iterations,
            "max_line_search_iterations",
        )?,
        hessian_regularization: options.hessian_regularization,
        qp_options: QuadraticOptions {
            max_iterations: count(options.qp_max_iterations, "qp_max_iterations")?,
            rho: options.qp_rho,
            absolute_tolerance: options.qp_absolute_tolerance,
            relative_tolerance: options.qp_relative_tolerance,
            convexity_tolerance: options.qp_convexity_tolerance,
        },
        restoration: options.restoration,
        restoration_options: restoration_options(options.restoration_options)?,
        globalization: match options.globalization {
            FfiSqpGlobalization::Merit => SqpGlobalization::Merit,
            FfiSqpGlobalization::Filter => SqpGlobalization::Filter,
        },
        filter_constraint_margin: options.filter_constraint_margin,
        filter_objective_margin: options.filter_objective_margin,
        curvature: match options.curvature {
            FfiSqpCurvature::Bfgs => SqpCurvature::Bfgs,
            FfiSqpCurvature::ExactLagrangian => SqpCurvature::ExactLagrangian,
        },
    };
    Ok(sqp_result(minimize_sqp(&problem, &initial, options)?))
}

fn sqp_result(value: SqpResult) -> FfiSqpResult {
    FfiSqpResult {
        point: value.point,
        objective: value.objective,
        constraint_values: value.constraint_values,
        multipliers: value
            .multipliers
            .into_iter()
            .map(|multiplier| FfiConstraintMultiplier {
                lower: multiplier.lower,
                upper: multiplier.upper,
                equality: multiplier.equality,
            })
            .collect(),
        maximum_violation: value.maximum_violation,
        stationarity_norm: value.stationarity_norm,
        iterations: value.iterations as u64,
        evaluations: value.evaluations as u64,
        accepted_steps: value.accepted_steps as u64,
        rejected_steps: value.rejected_steps as u64,
        final_merit_penalty: value.final_merit_penalty,
        last_step_norm: value.last_step_norm,
        termination: match value.termination {
            SqpTermination::Converged => FfiSqpTermination::Converged,
            SqpTermination::IterationLimit => FfiSqpTermination::IterationLimit,
            SqpTermination::StepLimit => FfiSqpTermination::StepLimit,
            SqpTermination::LineSearchFailed => FfiSqpTermination::LineSearchFailed,
            SqpTermination::QpFailure => FfiSqpTermination::QpFailure,
            SqpTermination::RestorationFailed => FfiSqpTermination::RestorationFailed,
            SqpTermination::Cancelled => FfiSqpTermination::Cancelled,
        },
    }
}

#[uniffi::export]
pub fn evaluate_nonlinear_objective_second_order(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
) -> Result<FfiSecondOrderValue, FfiError> {
    let model = model(model_value);
    model.validate()?;
    let value = model.objective.as_ref()
        .ok_or_else(|| nc_optimize::OptimizeError::InvalidProblem(
            "model does not contain an objective".into()))?
        .evaluate_second_order(&parameters)?;
    let dimension = value.gradient.len();
    Ok(FfiSecondOrderValue { value: value.value, gradient: value.gradient,
        hessian: value.hessian.into_iter().flatten().collect(), dimension: dimension as u64 })
}

#[uniffi::export]
pub fn evaluate_nonlinear_objective_sparse_hessian(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
    zero_tolerance: f64,
) -> Result<FfiSparseHessian, FfiError> {
    let model = model(model_value);
    model.validate()?;
    let (_, _, value) = model.objective.as_ref()
        .ok_or_else(|| nc_optimize::OptimizeError::InvalidProblem(
            "model does not contain an objective".into()))?
        .evaluate_sparse_hessian(&parameters, zero_tolerance)?;
    Ok(ffi_sparse_hessian(value))
}

#[uniffi::export]
pub fn evaluate_nonlinear_objective_hessian_vector_product(
    model_value: FfiNonlinearModel,
    parameters: Vec<f64>,
    direction: Vec<f64>,
) -> Result<Vec<f64>, FfiError> {
    let model = model(model_value);
    model.validate()?;
    Ok(model.objective.as_ref()
        .ok_or_else(|| nc_optimize::OptimizeError::InvalidProblem(
            "model does not contain an objective".into()))?
        .hessian_vector_product(&parameters, &direction)?)
}

#[uniffi::export]
pub fn evaluate_nonlinear_lagrangian_hessian_vector_product(
    model_value: FfiNonlinearModel,
    constraints: Vec<FfiNonlinearConstraint>,
    parameters: Vec<f64>,
    constraint_weights: Vec<f64>,
    direction: Vec<f64>,
) -> Result<Vec<f64>, FfiError> {
    let problem = ConstrainedNonlinearProblem {
        model: model(model_value),
        constraints: constraints.into_iter().map(|value| NonlinearConstraint {
            expression: expression(value.expression), bound: bound(value.bound),
        }).collect(),
    };
    Ok(problem.lagrangian_hessian_vector_product(
        &parameters, &constraint_weights, &direction)?)
}

#[uniffi::export]
pub fn solve_nonlinear_sparse_kkt(
    hessian: FfiSparseHessian,
    jacobian: FfiSparseJacobian,
    primal_rhs: Vec<f64>,
    constraint_rhs: Vec<f64>,
    primal_regularization: f64,
    dual_regularization: f64,
    drop_tolerance: f64,
) -> Result<FfiSparseKktResult, FfiError> {
    let value = solve_sparse_kkt(&SparseKktProblem {
        hessian: SparseHessian {
            dimension: count(hessian.dimension, "Hessian dimension")?,
            row_pointers: indices(hessian.row_pointers, "Hessian row pointers")?,
            column_indices: indices(hessian.column_indices, "Hessian column indices")?,
            values: hessian.values,
        },
        jacobian: SparseJacobian {
            rows: count(jacobian.rows, "Jacobian rows")?,
            columns: count(jacobian.columns, "Jacobian columns")?,
            row_pointers: indices(jacobian.row_pointers, "Jacobian row pointers")?,
            column_indices: indices(jacobian.column_indices, "Jacobian column indices")?,
            values: jacobian.values,
        },
        primal_rhs, constraint_rhs, primal_regularization, dual_regularization, drop_tolerance,
    })?;
    Ok(FfiSparseKktResult { primal: value.primal, dual: value.dual,
        residual_norm: value.residual_norm, relative_residual: value.relative_residual,
        factor_nonzeros: value.factor_nonzeros as u64 })
}

fn ffi_sparse_hessian(value: SparseHessian) -> FfiSparseHessian {
    FfiSparseHessian { dimension: value.dimension as u64,
        row_pointers: value.row_pointers.into_iter().map(|value| value as u64).collect(),
        column_indices: value.column_indices.into_iter().map(|value| value as u64).collect(),
        values: value.values }
}

fn indices(values: Vec<u64>, label: &'static str) -> Result<Vec<usize>, FfiError> {
    values.into_iter().map(|value| count(value, label)).collect()
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
