//! Swift-facing FFI surface, via UniFFI (ADR 0005).
//!
//! This intentionally exposes a **small, proven numerical slice**:
//! `nc-kernels-generic`'s matmul/dot/axpy/norm2, `nc-sparse`'s CSR SpMV,
//! selected LP solvers, and `nc-iterative`'s CSR weighted/penalized
//! least-squares solves. The latter are the stable statistical bridge for
//! sparse GAM and IRLS workloads: their sparse design and penalty operators
//! remain CSR across the Swift/Rust boundary and the result makes convergence
//! explicit.
//!
//! ## Why `Vec<f64>` copies, not shared buffers
//! UniFFI's proc-macro `#[export]` surface passes plain values
//! (`Vec<f64>`, records) across the boundary by copying, not by sharing
//! memory. That means every call here pays a copy in each direction —
//! acceptable for proving the pipeline, but exactly the cost ADR 0006
//! flags as the thing to revisit once this pattern is trusted. When that
//! happens, the fix is an opaque handle type (`Arc<Buffer>` exposed via
//! `#[derive(uniffi::Object)]`) that Swift holds by reference instead of
//! a `Vec` that gets copied on every call — not a change to this file's
//! function signatures, which can stay as the "small value" convenience
//! API even after a zero-copy path exists alongside it.
//!
//! ## `f32` exports
//! `matmul_f32`/`dot_f32`/`axpy_f32`/`norm2_f32`/`spmv_f32` mirror their
//! `f64` counterparts exactly (`nc-kernels-generic`'s kernels and
//! `nc-sparse::CsrMatrix` are already generic over the element type, so
//! this added no new Rust-side numerics — only the FFI-boundary
//! wrappers). Added once `NumericCore`'s `RustFallbackBackend`/
//! `SparseMatrix` were actually rewired to call through for `Double`
//! (ADR 0006's update) and it became clear `Float` was the one
//! remaining gap in that retirement.

//! ## Zero-copy vector buffers (`FfiVectorF64`/`FfiVectorF32`)
//! The functions above (`axpy_f64`, etc.) copy their `Vec` arguments
//! across the FFI boundary on every single call — fine for one-shot
//! operations, wasteful for a chain of operations on the same data
//! (e.g. several `axpy`/`dot` calls against the same working vector
//! inside an iterative solver's hot loop). `FfiVectorF64`/`FfiVectorF32`
//! are `uniffi::Object`s — reference-counted handles Swift holds
//! opaquely — whose data lives in Rust for the handle's whole lifetime.
//! Construction and `.toVec()`/`.toArray()` (whatever the generated
//! Swift method ends up named — see the same naming caveat as
//! elsewhere in this file) still copy once at each end, but any number
//! of `axpyInPlace`/`dot`/`norm2` calls in between touch no Swift
//! memory and cross no per-call copy.
//!
//! This is **additive, opt-in infrastructure** — `NumericCore.Matrix`/
//! `Vector`'s default storage and `RustFallbackBackend`'s dispatch are
//! unchanged (still one `FFIKernels` call, one copy each way, per ADR
//! 0006). Nothing currently requires a caller to use this; it exists
//! for a future hot-loop consumer (an iterative solver, or
//! `Swift-DataLens` if a LOESS inner loop turns out to need it) to opt
//! into once a concrete need is measured, per this project's
//! established "don't build ahead of need" principle — see ADR 0006's
//! update.

use nc_iterative::{IterativeSolveError, StatisticalSolveError};
use nc_kernels_generic as kernels;
use nc_sparse::CsrMatrix;
use std::sync::{Arc, Mutex};

mod nonlinear;
pub use nonlinear::*;

uniffi::setup_scaffolding!();

/// Errors that can cross the FFI boundary. Deliberately flat and
/// string-carrying rather than mirroring each source crate's error type
/// exactly — UniFFI generates a Swift `enum FfiError: Error` from this,
/// and `NCBindings` is expected to translate it into `NumericCore`'s
/// `NCError` at the call site, not re-expose this type to application code.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    // NOTE: struct variant (named field), not `DimensionMismatch(String)`.
    // UniFFI 0.27's Swift codegen emits uncompilable bindings for a
    // single-field tuple variant (`: try ...`, `write(, into:)`), which
    // breaks Swift-NumericCore's NCBindings target. The struct variant
    // generates valid `case DimensionMismatch(message:)` + correct
    // read/write, with an identical wire encoding.
    #[error("dimension mismatch: {message}")]
    DimensionMismatch { message: String },

    /// Carries any `nc_optimize::OptimizeError` (a solver reporting
    /// "not implemented" for a problem shape it doesn't handle, or a
    /// numerical failure like a non-positive-definite normal-equations
    /// matrix in `InteriorPointSolver`). Kept as a separate variant
    /// from `DimensionMismatch` rather than folding solver errors into
    /// that one — they're a different failure category, not a
    /// dimension problem, and a caller may reasonably want to tell them
    /// apart.
    #[error("solver error: {message}")]
    SolverError { message: String },
}

impl From<nc_sparse::SparseError> for FfiError {
    fn from(err: nc_sparse::SparseError) -> Self {
        FfiError::DimensionMismatch { message: err.to_string() }
    }
}

impl From<nc_sparse::SparseDirectError> for FfiError {
    fn from(err: nc_sparse::SparseDirectError) -> Self {
        match err {
            nc_sparse::SparseDirectError::NonSquare { .. }
            | nc_sparse::SparseDirectError::RightHandSideLength { .. } => {
                FfiError::DimensionMismatch { message: err.to_string() }
            }
            _ => FfiError::SolverError { message: err.to_string() },
        }
    }
}

impl From<nc_optimize::OptimizeError> for FfiError {
    fn from(err: nc_optimize::OptimizeError) -> Self {
        FfiError::SolverError { message: err.to_string() }
    }
}

impl From<StatisticalSolveError> for FfiError {
    fn from(err: StatisticalSolveError) -> Self {
        match err {
            // Malformed CSR is an input-shape error at the public boundary,
            // consistent with the existing SpMV and LP adapters.
            StatisticalSolveError::Sparse(err) => err.into(),
            other @ (StatisticalSolveError::ObservationLength { .. }
            | StatisticalSolveError::PenaltyWidth { .. }
            | StatisticalSolveError::InitialSolutionLength { .. }) => FfiError::DimensionMismatch {
                message: other.to_string(),
            },
            other => FfiError::SolverError { message: other.to_string() },
        }
    }
}

impl From<IterativeSolveError> for FfiError {
    fn from(err: IterativeSolveError) -> Self {
        match err {
            IterativeSolveError::Sparse(err) => err.into(),
            other @ (IterativeSolveError::NonSquare { .. }
                | IterativeSolveError::RightHandSideLength { .. }
                | IterativeSolveError::InitialSolutionLength { .. }) => {
                FfiError::DimensionMismatch { message: other.to_string() }
            }
            other => FfiError::SolverError { message: other.to_string() },
        }
    }
}

/// A dense `f64` matrix crossing the FFI boundary, column-major
/// (matching `Matrix<T>`'s Swift-side layout — ADR 0001 — so no
/// reordering happens in either direction).
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMatrixF64 {
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<f64>,
}

/// The `f32` counterpart of `FfiMatrixF64`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMatrixF32 {
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<f32>,
}

/// A CSR sparse `f64` matrix crossing the FFI boundary — field names and
/// shape mirror `nc_sparse::CsrMatrix` directly (ADR 0002).
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCsrMatrixF64 {
    pub rows: u32,
    pub cols: u32,
    pub row_ptr: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
}

/// The `f32` counterpart of `FfiCsrMatrixF64`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCsrMatrixF32 {
    pub rows: u32,
    pub cols: u32,
    pub row_ptr: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Coordinate-form sparse matrix for incremental construction.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCooMatrixF64 {
    pub rows: u32,
    pub cols: u32,
    pub row_indices: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
}

fn from_csr_f64(matrix: CsrMatrix<f64>) -> FfiCsrMatrixF64 {
    let rows = matrix.rows() as u32;
    let cols = matrix.cols() as u32;
    let entries: Vec<_> = matrix.iter_entries().collect();
    let mut row_ptr = vec![0u32; rows as usize + 1];
    let mut col_indices = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (row, column, value) in entries {
        row_ptr[row + 1] += 1;
        col_indices.push(column as u32);
        values.push(value);
    }
    for row in 0..rows as usize { row_ptr[row + 1] += row_ptr[row]; }
    FfiCsrMatrixF64 { rows, cols, row_ptr, col_indices, values }
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiLinearPreconditioner {
    None,
    Jacobi,
    Ilu0,
    IncompleteCholesky,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLinearSolveOptions {
    pub max_iterations: u64,
    pub tolerance: f64,
    pub initial_solution: Option<Vec<f64>>,
    pub preconditioner: FfiLinearPreconditioner,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiIterativeTermination {
    Converged,
    IterationLimit,
    Breakdown,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLinearSolveResult {
    pub solution: Vec<f64>,
    pub iterations: u64,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub termination: FfiIterativeTermination,
}

fn linear_options(options: FfiLinearSolveOptions) -> Result<nc_iterative::LinearSolveOptions, FfiError> {
    Ok(nc_iterative::LinearSolveOptions {
        max_iterations: statistical_iteration_limit(options.max_iterations)?,
        tolerance: options.tolerance,
        initial_solution: options.initial_solution,
        preconditioner: match options.preconditioner {
            FfiLinearPreconditioner::None => nc_iterative::LinearPreconditioner::None,
            FfiLinearPreconditioner::Jacobi => nc_iterative::LinearPreconditioner::Jacobi,
            FfiLinearPreconditioner::Ilu0 => nc_iterative::LinearPreconditioner::Ilu0,
            FfiLinearPreconditioner::IncompleteCholesky => {
                nc_iterative::LinearPreconditioner::IncompleteCholesky
            }
        },
    })
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseDirectResult {
    pub solution: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub factor_nonzeros: u64,
}

fn from_sparse_direct(result: nc_sparse::SparseDirectReport) -> FfiSparseDirectResult {
    FfiSparseDirectResult {
        solution: result.solution,
        residual_norm: result.residual_norm,
        relative_residual: result.relative_residual,
        factor_nonzeros: result.factor_nonzeros as u64,
    }
}

fn from_linear_solve(result: nc_iterative::LinearSolveReport) -> FfiLinearSolveResult {
    FfiLinearSolveResult {
        solution: result.solution,
        iterations: result.iterations as u64,
        residual_norm: result.residual_norm,
        relative_residual: result.relative_residual,
        termination: match result.termination {
            nc_iterative::IterativeTermination::Converged => FfiIterativeTermination::Converged,
            nc_iterative::IterativeTermination::IterationLimit => FfiIterativeTermination::IterationLimit,
            nc_iterative::IterativeTermination::Breakdown => FfiIterativeTermination::Breakdown,
        },
    }
}

/// Observable outcome of a sparse weighted or penalized least-squares solve.
///
/// `converged` refers to the relative normal residual of the augmented CGLS
/// system. An unconverged result is diagnostic information, not a valid fitted
/// model; callers must require `converged` before using `solution` for
/// inference or prediction.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseStatisticalSolveResult {
    pub solution: Vec<f64>,
    pub iterations: u64,
    pub residual_norm: f64,
    pub converged: bool,
    pub weighted_residual_sum_of_squares: f64,
    pub penalty_contribution: f64,
    pub objective: f64,
}

/// Scaling strategy for configurable sparse statistical solves.
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiSparseStatisticalPreconditioner {
    None,
    Jacobi,
}

/// Solver state and convergence settings. `initial_solution` is the previous
/// coefficient vector in an IRLS loop or regularization path.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSparseStatisticalSolveOptions {
    pub max_iterations: u64,
    pub tolerance: f64,
    pub initial_solution: Option<Vec<f64>>,
    pub preconditioner: FfiSparseStatisticalPreconditioner,
}

fn statistical_options(
    options: FfiSparseStatisticalSolveOptions,
) -> Result<nc_iterative::StatisticalSolveOptions, FfiError> {
    Ok(nc_iterative::StatisticalSolveOptions {
        max_iterations: statistical_iteration_limit(options.max_iterations)?,
        tolerance: options.tolerance,
        initial_solution: options.initial_solution,
        preconditioner: match options.preconditioner {
            FfiSparseStatisticalPreconditioner::None => nc_iterative::StatisticalPreconditioner::None,
            FfiSparseStatisticalPreconditioner::Jacobi => nc_iterative::StatisticalPreconditioner::Jacobi,
        },
    })
}

fn to_csr_f64(matrix: FfiCsrMatrixF64) -> Result<CsrMatrix<f64>, FfiError> {
    Ok(CsrMatrix::new(
        matrix.rows as usize,
        matrix.cols as usize,
        matrix.row_ptr.into_iter().map(|value| value as usize).collect(),
        matrix.col_indices.into_iter().map(|value| value as usize).collect(),
        matrix.values,
    )?)
}

fn from_statistical_solve(result: nc_iterative::StatisticalSolveResult) -> FfiSparseStatisticalSolveResult {
    FfiSparseStatisticalSolveResult {
        objective: result.objective(),
        solution: result.solution,
        iterations: result.iterations as u64,
        residual_norm: result.residual_norm,
        converged: result.converged,
        weighted_residual_sum_of_squares: result.weighted_residual_sum_of_squares,
        penalty_contribution: result.penalty_contribution,
    }
}

fn statistical_iteration_limit(max_iterations: u64) -> Result<usize, FfiError> {
    usize::try_from(max_iterations).map_err(|_| FfiError::SolverError {
        message: "max_iterations does not fit this platform".to_owned(),
    })
}

#[uniffi::export]
pub fn matmul_f64(a: FfiMatrixF64, b: FfiMatrixF64) -> Result<FfiMatrixF64, FfiError> {
    if a.cols != b.rows {
        return Err(FfiError::DimensionMismatch {
            message: format!("matmul: {}x{} * {}x{}", a.rows, a.cols, b.rows, b.cols),
        });
    }
    let (m, k, n) = (a.rows as usize, a.cols as usize, b.cols as usize);
    let mut c = vec![0.0f64; m * n];
    kernels::matmul(&a.data, &b.data, &mut c, m, k, n);
    Ok(FfiMatrixF64 { rows: a.rows, cols: b.cols, data: c })
}

/// The `f32` counterpart of `matmul_f64`.
#[uniffi::export]
pub fn matmul_f32(a: FfiMatrixF32, b: FfiMatrixF32) -> Result<FfiMatrixF32, FfiError> {
    if a.cols != b.rows {
        return Err(FfiError::DimensionMismatch {
            message: format!("matmul: {}x{} * {}x{}", a.rows, a.cols, b.rows, b.cols),
        });
    }
    let (m, k, n) = (a.rows as usize, a.cols as usize, b.cols as usize);
    let mut c = vec![0.0f32; m * n];
    kernels::matmul(&a.data, &b.data, &mut c, m, k, n);
    Ok(FfiMatrixF32 { rows: a.rows, cols: b.cols, data: c })
}

#[uniffi::export]
pub fn dot_f64(x: Vec<f64>, y: Vec<f64>) -> Result<f64, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch {
            message: format!("dot: lengths {} and {}", x.len(), y.len()),
        });
    }
    Ok(kernels::dot(&x, &y))
}

/// The `f32` counterpart of `dot_f64`.
#[uniffi::export]
pub fn dot_f32(x: Vec<f32>, y: Vec<f32>) -> Result<f32, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch {
            message: format!("dot: lengths {} and {}", x.len(), y.len()),
        });
    }
    Ok(kernels::dot(&x, &y))
}

/// `result = alpha * x + y`. Returns a new vector rather than mutating,
/// since UniFFI's proc-macro surface passes by value/copy anyway (see
/// module docs) — an `inout`-style Swift API is layered on top of this
/// in `NCBindings`, not expressed here.
#[uniffi::export]
pub fn axpy_f64(alpha: f64, x: Vec<f64>, y: Vec<f64>) -> Result<Vec<f64>, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch {
            message: format!("axpy: lengths {} and {}", x.len(), y.len()),
        });
    }
    let mut result = y;
    kernels::axpy(alpha, &x, &mut result);
    Ok(result)
}

/// The `f32` counterpart of `axpy_f64`.
#[uniffi::export]
pub fn axpy_f32(alpha: f32, x: Vec<f32>, y: Vec<f32>) -> Result<Vec<f32>, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch {
            message: format!("axpy: lengths {} and {}", x.len(), y.len()),
        });
    }
    let mut result = y;
    kernels::axpy(alpha, &x, &mut result);
    Ok(result)
}

#[uniffi::export]
pub fn norm2_f64(x: Vec<f64>) -> f64 {
    kernels::norm2(&x)
}

/// The `f32` counterpart of `norm2_f64`.
#[uniffi::export]
pub fn norm2_f32(x: Vec<f32>) -> f32 {
    kernels::norm2(&x)
}

#[uniffi::export]
pub fn spmv_f64(matrix: FfiCsrMatrixF64, x: Vec<f64>) -> Result<Vec<f64>, FfiError> {
    let csr = to_csr_f64(matrix)?;
    Ok(csr.spmv(&x)?)
}

/// Canonicalize coordinate entries into sorted, duplicate-coalesced CSR.
#[uniffi::export]
pub fn coo_to_csr_f64(matrix: FfiCooMatrixF64) -> Result<FfiCsrMatrixF64, FfiError> {
    if matrix.row_indices.len() != matrix.col_indices.len()
        || matrix.row_indices.len() != matrix.values.len()
    {
        return Err(FfiError::DimensionMismatch {
            message: "COO row, column, and value arrays must have equal length".to_owned(),
        });
    }
    let mut coo = nc_sparse::CooMatrix::with_capacity(
        matrix.rows as usize, matrix.cols as usize, matrix.values.len(),
    );
    for ((row, column), value) in matrix.row_indices.into_iter()
        .zip(matrix.col_indices).zip(matrix.values)
    {
        coo.push(row as usize, column as usize, value)?;
    }
    Ok(from_csr_f64(coo.to_csr()?))
}

#[uniffi::export]
pub fn transpose_csr_f64(matrix: FfiCsrMatrixF64) -> Result<FfiCsrMatrixF64, FfiError> {
    Ok(from_csr_f64(to_csr_f64(matrix)?.transpose()?))
}

#[uniffi::export]
pub fn solve_sparse_conjugate_gradient(
    matrix: FfiCsrMatrixF64,
    rhs: Vec<f64>,
    options: FfiLinearSolveOptions,
) -> Result<FfiLinearSolveResult, FfiError> {
    let result = nc_iterative::conjugate_gradient_with_options(
        &to_csr_f64(matrix)?, &rhs, linear_options(options)?,
    )?;
    Ok(from_linear_solve(result))
}

#[uniffi::export]
pub fn solve_sparse_bicgstab(
    matrix: FfiCsrMatrixF64,
    rhs: Vec<f64>,
    options: FfiLinearSolveOptions,
) -> Result<FfiLinearSolveResult, FfiError> {
    let result = nc_iterative::bicgstab(
        &to_csr_f64(matrix)?, &rhs, linear_options(options)?,
    )?;
    Ok(from_linear_solve(result))
}

#[uniffi::export]
pub fn solve_sparse_gmres(
    matrix: FfiCsrMatrixF64,
    rhs: Vec<f64>,
    options: FfiLinearSolveOptions,
    restart: u64,
) -> Result<FfiLinearSolveResult, FfiError> {
    let result = nc_iterative::gmres(
        &to_csr_f64(matrix)?, &rhs, linear_options(options)?,
        statistical_iteration_limit(restart)?,
    )?;
    Ok(from_linear_solve(result))
}

/// Factor and solve a general square CSR system with sparse pivoted LU.
#[uniffi::export]
pub fn solve_sparse_lu(
    matrix: FfiCsrMatrixF64, rhs: Vec<f64>, drop_tolerance: f64,
) -> Result<FfiSparseDirectResult, FfiError> {
    let matrix = to_csr_f64(matrix)?;
    let factor = nc_sparse::SparseLu::factor(&matrix, drop_tolerance)?;
    Ok(from_sparse_direct(factor.solve_report(&matrix, &rhs)?))
}

/// Factor and solve a symmetric positive-definite CSR system with sparse
/// Cholesky. Symmetry and positive pivots are checked explicitly.
#[uniffi::export]
pub fn solve_sparse_cholesky(
    matrix: FfiCsrMatrixF64, rhs: Vec<f64>, drop_tolerance: f64,
) -> Result<FfiSparseDirectResult, FfiError> {
    let matrix = to_csr_f64(matrix)?;
    let factor = nc_sparse::SparseCholesky::factor(&matrix, drop_tolerance)?;
    Ok(from_sparse_direct(factor.solve_report(&matrix, &rhs)?))
}

/// The `f32` counterpart of `spmv_f64`.
#[uniffi::export]
pub fn spmv_f32(matrix: FfiCsrMatrixF32, x: Vec<f32>) -> Result<Vec<f32>, FfiError> {
    let csr = CsrMatrix::new(
        matrix.rows as usize,
        matrix.cols as usize,
        matrix.row_ptr.iter().map(|&v| v as usize).collect(),
        matrix.col_indices.iter().map(|&v| v as usize).collect(),
        matrix.values,
    )?;
    Ok(csr.spmv(&x)?)
}

/// Solve `min Σᵢ wᵢ(yᵢ − xᵢᵀβ)²` using CGLS over a CSR design matrix.
///
/// Zero weights exclude observations. The solve is intentionally iterative;
/// inspect `converged` before accepting the returned coefficients.
#[uniffi::export]
pub fn solve_sparse_weighted_least_squares(
    design: FfiCsrMatrixF64,
    response: Vec<f64>,
    weights: Vec<f64>,
    max_iterations: u64,
    tolerance: f64,
) -> Result<FfiSparseStatisticalSolveResult, FfiError> {
    let design = to_csr_f64(design)?;
    let max_iterations = statistical_iteration_limit(max_iterations)?;
    let result = nc_iterative::weighted_least_squares(
        &design, &response, &weights, max_iterations, tolerance,
    )?;
    Ok(from_statistical_solve(result))
}

/// Configurable sparse weighted solve for IRLS and ill-scaled models.
#[uniffi::export]
pub fn solve_sparse_weighted_least_squares_with_options(
    design: FfiCsrMatrixF64,
    response: Vec<f64>,
    weights: Vec<f64>,
    options: FfiSparseStatisticalSolveOptions,
) -> Result<FfiSparseStatisticalSolveResult, FfiError> {
    let design = to_csr_f64(design)?;
    let result = nc_iterative::weighted_least_squares_with_options(
        &design, &response, &weights, statistical_options(options)?,
    )?;
    Ok(from_statistical_solve(result))
}

/// Solve `min Σᵢ wᵢ(yᵢ − xᵢᵀβ)² + λ‖Pβ‖²` using matrix-free CGLS over
/// CSR design and penalty operators. `penalty` must have one column per
/// design coefficient and `penalty_weight` must be finite and positive.
#[uniffi::export]
pub fn solve_sparse_penalized_weighted_least_squares(
    design: FfiCsrMatrixF64,
    response: Vec<f64>,
    weights: Vec<f64>,
    penalty: FfiCsrMatrixF64,
    penalty_weight: f64,
    max_iterations: u64,
    tolerance: f64,
) -> Result<FfiSparseStatisticalSolveResult, FfiError> {
    let design = to_csr_f64(design)?;
    let penalty = to_csr_f64(penalty)?;
    let max_iterations = statistical_iteration_limit(max_iterations)?;
    let result = nc_iterative::penalized_weighted_least_squares(
        &design, &response, &weights, &penalty, penalty_weight, max_iterations, tolerance,
    )?;
    Ok(from_statistical_solve(result))
}

/// Configurable penalized sparse solve with warm-start and preconditioning.
#[uniffi::export]
pub fn solve_sparse_penalized_weighted_least_squares_with_options(
    design: FfiCsrMatrixF64,
    response: Vec<f64>,
    weights: Vec<f64>,
    penalty: FfiCsrMatrixF64,
    penalty_weight: f64,
    options: FfiSparseStatisticalSolveOptions,
) -> Result<FfiSparseStatisticalSolveResult, FfiError> {
    let design = to_csr_f64(design)?;
    let penalty = to_csr_f64(penalty)?;
    let result = nc_iterative::penalized_weighted_least_squares_with_options(
        &design, &response, &weights, &penalty, penalty_weight,
        statistical_options(options)?,
    )?;
    Ok(from_statistical_solve(result))
}

// ---------------------------------------------------------------------
// LP/MILP solving — closes the loop from NumericCoreAMPL's
// CompiledProblem through to
// nc-optimize::{RevisedSimplexSolver, InteriorPointSolver, BranchAndBoundSolver}.
//
// `FfiBound`/`FfiProblem`/`FfiSolution`/`FfiSolveStatus` mirror
// `nc_optimize`'s `Bound`/`Problem`/`Solution`/`SolveStatus` field for
// field — this file's job is only the boundary crossing, not any new
// logic. `solve_lp_simplex`/`solve_lp_interior_point`/
// `solve_milp_branch_and_bound` are three separate exported functions
// rather than one function taking a solver-choice enum, matching the
// explicit (not policy-based) solver selection decision from
// `nc-optimize`'s ADR 0004 update — the choice is made in Swift by
// which function it calls, not by a parameter this file has to
// validate.
// ---------------------------------------------------------------------

/// Mirrors `nc_optimize::Bound`. `None` means unbounded in that
/// direction, same convention as the Rust type.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiBound {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

/// Mirrors `nc_optimize::Problem`, with the constraint matrix crossing
/// as `FfiCsrMatrixF64` (already established above) rather than a
/// separate ad hoc shape.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiProblem {
    pub objective: Vec<f64>,
    pub constraints: FfiCsrMatrixF64,
    pub row_bounds: Vec<FfiBound>,
    pub var_bounds: Vec<FfiBound>,
    /// Mirrors `nc_optimize::Problem::is_integer`. Length must match
    /// `objective`/`var_bounds`. Ignored entirely by
    /// `solve_lp_simplex`/`solve_lp_interior_point` (an LP relaxation
    /// is well-defined regardless of its contents); only
    /// `solve_milp_branch_and_bound` reads it.
    pub is_integer: Vec<bool>,
}

/// Mirrors `nc_optimize::SolveStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiSolveStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
}

/// Mirrors `nc_optimize::Solution`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSolution {
    pub variable_values: Vec<f64>,
    pub objective_value: f64,
    pub status: FfiSolveStatus,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSimplexOptions {
    pub max_iterations: u64,
    pub tolerance: f64,
    pub scaling: bool,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiInteriorPointOptions {
    pub max_iterations: u64,
    pub tolerance: f64,
    pub sigma: f64,
    pub big_bound: f64,
    pub step_fraction: f64,
    pub scaling: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiNodeSelection { DepthFirst, BestBound }

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiBranchingStrategy { MostFractional, PseudoCost }

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiBranchAndBoundTermination {
    Exhausted, GapSatisfied, NodeLimit, RelaxationLimit, Unbounded, ContinuousRelaxation,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiBranchAndBoundOptions {
    pub max_nodes: u64,
    pub integer_tolerance: f64,
    pub scaling: bool,
    /// Empty means no incumbent. Otherwise one value per variable.
    pub initial_incumbent: Vec<f64>,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub node_selection: FfiNodeSelection,
    pub branching_strategy: FfiBranchingStrategy,
    pub bound_propagation: bool,
}

/// MILP result plus the search certificate available at termination.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMilpSolveResult {
    pub solution: FfiSolution,
    pub nodes_explored: u64,
    pub best_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub nodes_pruned_infeasible: u64,
    pub nodes_pruned_by_bound: u64,
    pub maximum_depth: u64,
    pub relaxations_solved: u64,
    pub bounds_tightened: u64,
    pub incumbents_found: u64,
    pub termination: FfiBranchAndBoundTermination,
}

fn optimization_limit(value: u64, name: &'static str) -> Result<usize, FfiError> {
    usize::try_from(value).map_err(|_| FfiError::SolverError {
        message: format!("{name} does not fit this platform"),
    })
}

fn to_domain_problem(problem: FfiProblem) -> Result<nc_optimize::Problem, FfiError> {
    let variable_count = problem.objective.len();
    let constraints = CsrMatrix::new(
        problem.constraints.rows as usize,
        problem.constraints.cols as usize,
        problem.constraints.row_ptr.iter().map(|&v| v as usize).collect(),
        problem.constraints.col_indices.iter().map(|&v| v as usize).collect(),
        problem.constraints.values,
    )?;
    let is_integer = if problem.is_integer.is_empty() {
        // Older callers (before this field existed) or a caller that
        // genuinely has no integer variables both send an empty list —
        // treat as "all continuous" rather than a length-mismatch error;
        // a real length mismatch (non-empty but wrong length) is still
        // caught by BranchAndBoundSolver itself.
        vec![false; variable_count]
    } else {
        problem.is_integer
    };
    Ok(nc_optimize::Problem {
        objective: problem.objective,
        constraints,
        row_bounds: problem.row_bounds.into_iter().map(to_domain_bound).collect(),
        var_bounds: problem.var_bounds.into_iter().map(to_domain_bound).collect(),
        is_integer,
    })
}

fn to_domain_bound(bound: FfiBound) -> nc_optimize::Bound {
    nc_optimize::Bound { lower: bound.lower, upper: bound.upper }
}

fn from_domain_solution(solution: nc_optimize::Solution) -> FfiSolution {
    FfiSolution {
        variable_values: solution.variable_values,
        objective_value: solution.objective_value,
        status: match solution.status {
            nc_optimize::SolveStatus::Optimal => FfiSolveStatus::Optimal,
            nc_optimize::SolveStatus::Infeasible => FfiSolveStatus::Infeasible,
            nc_optimize::SolveStatus::Unbounded => FfiSolveStatus::Unbounded,
            nc_optimize::SolveStatus::IterationLimit => FfiSolveStatus::IterationLimit,
        },
    }
}

/// Solves `problem` via `nc_optimize::RevisedSimplexSolver` (default
/// configuration). Prefer this over `solve_lp_interior_point` when
/// exact/rigorous infeasibility or unboundedness detection matters, or
/// the problem has equality constraints or fixed variables — see
/// `nc_optimize::interior_point`'s module docs for why the
/// interior-point path rejects those.
#[uniffi::export]
pub fn solve_lp_simplex(problem: FfiProblem) -> Result<FfiSolution, FfiError> {
    use nc_optimize::Solver;
    let domain_problem = to_domain_problem(problem)?;
    let solution = nc_optimize::ScaledSolver {
        inner: nc_optimize::RevisedSimplexSolver::default(), options: Default::default(),
    }.solve(&domain_problem)?;
    Ok(from_domain_solution(solution))
}

#[uniffi::export]
pub fn solve_lp_simplex_with_options(
    problem: FfiProblem,
    options: FfiSimplexOptions,
) -> Result<FfiSolution, FfiError> {
    use nc_optimize::Solver;
    let domain_problem = to_domain_problem(problem)?;
    let solver = nc_optimize::RevisedSimplexSolver {
        max_iterations: optimization_limit(options.max_iterations, "max_iterations")?,
        tolerance: options.tolerance,
    };
    let solution = if options.scaling {
        nc_optimize::ScaledSolver { inner: solver, options: Default::default() }.solve(&domain_problem)?
    } else {
        solver.solve(&domain_problem)?
    };
    Ok(from_domain_solution(solution))
}

/// Solves `problem` via `nc_optimize::InteriorPointSolver` (default
/// configuration). See that solver's module docs for its two scope
/// boundaries: it rejects equality constraints/fixed variables outright
/// (surfaced here as `FfiError::SolverError`, not a crash or silent
/// wrong answer), and its unboundedness detection is heuristic.
#[uniffi::export]
pub fn solve_lp_interior_point(problem: FfiProblem) -> Result<FfiSolution, FfiError> {
    use nc_optimize::Solver;
    let domain_problem = to_domain_problem(problem)?;
    let solution = nc_optimize::ScaledSolver {
        inner: nc_optimize::InteriorPointSolver::default(), options: Default::default(),
    }.solve(&domain_problem)?;
    Ok(from_domain_solution(solution))
}

#[uniffi::export]
pub fn solve_lp_interior_point_with_options(
    problem: FfiProblem,
    options: FfiInteriorPointOptions,
) -> Result<FfiSolution, FfiError> {
    use nc_optimize::Solver;
    let domain_problem = to_domain_problem(problem)?;
    let solver = nc_optimize::InteriorPointSolver {
        max_iterations: optimization_limit(options.max_iterations, "max_iterations")?,
        tolerance: options.tolerance,
        sigma: options.sigma,
        big_bound: options.big_bound,
        step_fraction: options.step_fraction,
    };
    let solution = if options.scaling {
        nc_optimize::ScaledSolver { inner: solver, options: Default::default() }.solve(&domain_problem)?
    } else {
        solver.solve(&domain_problem)?
    };
    Ok(from_domain_solution(solution))
}

/// Solves `problem` via `nc_optimize::BranchAndBoundSolver` (default
/// configuration — best-bound, `RevisedSimplexSolver` as the LP
/// relaxation solver). `problem.is_integer` selects which variables are
/// integer-restricted; an empty list is treated as "all continuous"
/// (see `to_domain_problem`), which for this function specifically
/// means it will solve a plain LP with no branching at all — not
/// useful on its own, but harmless, and avoids a separate "did you mean
/// to call solve_lp_simplex instead?" error for what is otherwise a
/// valid (if pointless) call.
#[uniffi::export]
pub fn solve_milp_branch_and_bound(problem: FfiProblem) -> Result<FfiSolution, FfiError> {
    let domain_problem = to_domain_problem(problem)?;
    let scaled = nc_optimize::ScaledProblem::new(&domain_problem, Default::default())?;
    let report = nc_optimize::BranchAndBoundSolver::default().solve_with_report(&scaled.problem)?;
    let solution = scaled.restore(report.solution, &domain_problem);
    Ok(from_domain_solution(solution))
}

#[uniffi::export]
pub fn solve_milp_branch_and_bound_with_options(
    problem: FfiProblem,
    options: FfiBranchAndBoundOptions,
) -> Result<FfiMilpSolveResult, FfiError> {
    let domain_problem = to_domain_problem(problem)?;
    let solver = nc_optimize::BranchAndBoundSolver {
        max_nodes: optimization_limit(options.max_nodes, "max_nodes")?,
        integer_tolerance: options.integer_tolerance,
        absolute_gap_tolerance: options.absolute_gap_tolerance,
        relative_gap_tolerance: options.relative_gap_tolerance,
        node_selection: match options.node_selection {
            FfiNodeSelection::DepthFirst => nc_optimize::NodeSelection::DepthFirst,
            FfiNodeSelection::BestBound => nc_optimize::NodeSelection::BestBound,
        },
        branching_strategy: match options.branching_strategy {
            FfiBranchingStrategy::MostFractional => nc_optimize::BranchingStrategy::MostFractional,
            FfiBranchingStrategy::PseudoCost => nc_optimize::BranchingStrategy::PseudoCost,
        },
        bound_propagation: options.bound_propagation,
        relaxation_solver: Box::new(nc_optimize::RevisedSimplexSolver::default()),
    };
    let scaled = options.scaling.then(|| nc_optimize::ScaledProblem::new(
        &domain_problem, Default::default()
    )).transpose()?;
    let solve_problem = scaled.as_ref().map_or(&domain_problem, |value| &value.problem);
    let incumbent = if options.initial_incumbent.is_empty() {
        None
    } else if let Some(ref scaled_problem) = scaled {
        Some(options.initial_incumbent.iter().zip(&scaled_problem.report.variable_factors)
            .map(|(value, factor)| value / factor).collect::<Vec<_>>())
    } else {
        Some(options.initial_incumbent)
    };
    let mut report = solver.solve_with_report_warm(solve_problem, incumbent.as_deref())?;
    if let Some(ref scaled_problem) = scaled {
        report.solution = scaled_problem.restore(report.solution, &domain_problem);
    }
    Ok(FfiMilpSolveResult {
        solution: from_domain_solution(report.solution),
        nodes_explored: report.nodes_explored as u64,
        best_bound: report.best_bound,
        absolute_gap: report.absolute_gap,
        relative_gap: report.relative_gap,
        nodes_pruned_infeasible: report.nodes_pruned_infeasible as u64,
        nodes_pruned_by_bound: report.nodes_pruned_by_bound as u64,
        maximum_depth: report.maximum_depth as u64,
        relaxations_solved: report.relaxations_solved as u64,
        bounds_tightened: report.bounds_tightened as u64,
        incumbents_found: report.incumbents_found as u64,
        termination: match report.termination {
            nc_optimize::BranchAndBoundTermination::Exhausted => FfiBranchAndBoundTermination::Exhausted,
            nc_optimize::BranchAndBoundTermination::GapSatisfied => FfiBranchAndBoundTermination::GapSatisfied,
            nc_optimize::BranchAndBoundTermination::NodeLimit => FfiBranchAndBoundTermination::NodeLimit,
            nc_optimize::BranchAndBoundTermination::RelaxationLimit => FfiBranchAndBoundTermination::RelaxationLimit,
            nc_optimize::BranchAndBoundTermination::Unbounded => FfiBranchAndBoundTermination::Unbounded,
            nc_optimize::BranchAndBoundTermination::ContinuousRelaxation => FfiBranchAndBoundTermination::ContinuousRelaxation,
        },
    })
}

/// A reference-counted `f64` vector buffer living entirely in Rust —
/// see this file's module docs for why this exists alongside the
/// copy-per-call functions above. `Mutex`-guarded interior mutability
/// is required here (not just `RefCell`) because UniFFI objects cross
/// the FFI boundary as `Arc<Self>`, which needs `Send + Sync`.
#[derive(uniffi::Object)]
pub struct FfiVectorF64 {
    data: Mutex<Vec<f64>>,
}

#[uniffi::export]
impl FfiVectorF64 {
    #[uniffi::constructor]
    pub fn new(data: Vec<f64>) -> Arc<Self> {
        Arc::new(Self { data: Mutex::new(data) })
    }

    /// The one copy back out to Swift — call once, at the end of a
    /// chain of in-place operations, not after every step.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.lock().expect("FfiVectorF64 mutex poisoned").clone()
    }

    pub fn len(&self) -> u64 {
        self.data.lock().expect("FfiVectorF64 mutex poisoned").len() as u64
    }

    /// `self <- self + alpha * other`, entirely in Rust — no data
    /// crosses the FFI boundary for this call beyond the two handles
    /// and the scalar `alpha`.
    pub fn axpy_in_place(&self, alpha: f64, other: &FfiVectorF64) -> Result<(), FfiError> {
        let mut mine = self.data.lock().expect("FfiVectorF64 mutex poisoned");
        let theirs = other.data.lock().expect("FfiVectorF64 mutex poisoned");
        if mine.len() != theirs.len() {
            return Err(FfiError::DimensionMismatch {
                message: format!("axpyInPlace: lengths {} and {}", mine.len(), theirs.len()),
            });
        }
        kernels::axpy(alpha, &theirs, &mut mine);
        Ok(())
    }

    pub fn dot(&self, other: &FfiVectorF64) -> Result<f64, FfiError> {
        let mine = self.data.lock().expect("FfiVectorF64 mutex poisoned");
        let theirs = other.data.lock().expect("FfiVectorF64 mutex poisoned");
        if mine.len() != theirs.len() {
            return Err(FfiError::DimensionMismatch {
                message: format!("dot: lengths {} and {}", mine.len(), theirs.len()),
            });
        }
        Ok(kernels::dot(&mine, &theirs))
    }

    pub fn norm2(&self) -> f64 {
        let mine = self.data.lock().expect("FfiVectorF64 mutex poisoned");
        kernels::norm2(&mine)
    }
}

/// The `f32` counterpart of `FfiVectorF64`.
#[derive(uniffi::Object)]
pub struct FfiVectorF32 {
    data: Mutex<Vec<f32>>,
}

#[uniffi::export]
impl FfiVectorF32 {
    #[uniffi::constructor]
    pub fn new(data: Vec<f32>) -> Arc<Self> {
        Arc::new(Self { data: Mutex::new(data) })
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.lock().expect("FfiVectorF32 mutex poisoned").clone()
    }

    pub fn len(&self) -> u64 {
        self.data.lock().expect("FfiVectorF32 mutex poisoned").len() as u64
    }

    pub fn axpy_in_place(&self, alpha: f32, other: &FfiVectorF32) -> Result<(), FfiError> {
        let mut mine = self.data.lock().expect("FfiVectorF32 mutex poisoned");
        let theirs = other.data.lock().expect("FfiVectorF32 mutex poisoned");
        if mine.len() != theirs.len() {
            return Err(FfiError::DimensionMismatch {
                message: format!("axpyInPlace: lengths {} and {}", mine.len(), theirs.len()),
            });
        }
        kernels::axpy(alpha, &theirs, &mut mine);
        Ok(())
    }

    pub fn dot(&self, other: &FfiVectorF32) -> Result<f32, FfiError> {
        let mine = self.data.lock().expect("FfiVectorF32 mutex poisoned");
        let theirs = other.data.lock().expect("FfiVectorF32 mutex poisoned");
        if mine.len() != theirs.len() {
            return Err(FfiError::DimensionMismatch {
                message: format!("dot: lengths {} and {}", mine.len(), theirs.len()),
            });
        }
        Ok(kernels::dot(&mine, &theirs))
    }

    pub fn norm2(&self) -> f32 {
        let mine = self.data.lock().expect("FfiVectorF32 mutex poisoned");
        kernels::norm2(&mine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_f64_identity() {
        let identity = FfiMatrixF64 { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] };
        let a = FfiMatrixF64 { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let result = matmul_f64(identity, a.clone()).unwrap();
        assert_eq!(result.data, a.data);
    }

    #[test]
    fn matmul_f64_rejects_dimension_mismatch() {
        let a = FfiMatrixF64 { rows: 2, cols: 3, data: vec![0.0; 6] };
        let b = FfiMatrixF64 { rows: 2, cols: 2, data: vec![0.0; 4] };
        assert!(matmul_f64(a, b).is_err());
    }

    #[test]
    fn dot_f64_matches_hand_computation() {
        assert_eq!(dot_f64(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap(), 32.0);
    }

    #[test]
    fn axpy_f64_updates_correctly() {
        let result = axpy_f64(2.0, vec![1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(result, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn norm2_f64_matches_hand_computation() {
        assert_eq!(norm2_f64(vec![3.0, 4.0]), 5.0);
    }

    #[test]
    fn spmv_f64_matches_hand_computation() {
        // [[1, 0, 2], [0, 3, 0]] * [1, 1, 1] = [3, 3]
        let matrix = FfiCsrMatrixF64 {
            rows: 2,
            cols: 3,
            row_ptr: vec![0, 2, 3],
            col_indices: vec![0, 2, 1],
            values: vec![1.0, 2.0, 3.0],
        };
        let result = spmv_f64(matrix, vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(result, vec![3.0, 3.0]);
    }

    #[test]
    fn sparse_assembly_transpose_and_bicgstab_cross_ffi() {
        let csr = coo_to_csr_f64(FfiCooMatrixF64 {
            rows: 2, cols: 2,
            row_indices: vec![1, 0, 1, 0],
            col_indices: vec![0, 1, 1, 0],
            values: vec![2.0, 1.0, 3.0, 4.0],
        }).unwrap();
        let transposed = transpose_csr_f64(csr.clone()).unwrap();
        assert_eq!(spmv_f64(transposed, vec![1.0, 1.0]).unwrap(), vec![6.0, 4.0]);
        let solved = solve_sparse_bicgstab(
            csr.clone(), vec![6.0, 8.0], FfiLinearSolveOptions {
                max_iterations: 20, tolerance: 1e-12, initial_solution: None,
                preconditioner: FfiLinearPreconditioner::Jacobi,
            },
        ).unwrap();
        assert_eq!(solved.termination, FfiIterativeTermination::Converged);
        assert!((solved.solution[0] - 1.0).abs() < 1e-10);
        assert!((solved.solution[1] - 2.0).abs() < 1e-10);
        let gmres = solve_sparse_gmres(
            csr, vec![6.0, 8.0], FfiLinearSolveOptions {
                max_iterations: 20, tolerance: 1e-12, initial_solution: None,
                preconditioner: FfiLinearPreconditioner::Jacobi,
            }, 2,
        ).unwrap();
        assert_eq!(gmres.termination, FfiIterativeTermination::Converged);
    }

    #[test]
    fn sparse_direct_and_factor_preconditioners_cross_ffi() {
        let matrix = FfiCsrMatrixF64 {
            rows: 3, cols: 3, row_ptr: vec![0, 2, 5, 7],
            col_indices: vec![0, 1, 0, 1, 2, 1, 2],
            values: vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
        };
        let rhs = vec![6.0, 10.0, 8.0];
        let lu = solve_sparse_lu(matrix.clone(), rhs.clone(), 0.0).unwrap();
        let cholesky = solve_sparse_cholesky(matrix.clone(), rhs.clone(), 0.0).unwrap();
        assert!(lu.relative_residual < 1e-12);
        assert!(cholesky.relative_residual < 1e-12);
        let cg = solve_sparse_conjugate_gradient(matrix, rhs, FfiLinearSolveOptions {
            max_iterations: 20, tolerance: 1e-12, initial_solution: None,
            preconditioner: FfiLinearPreconditioner::IncompleteCholesky,
        }).unwrap();
        assert_eq!(cg.termination, FfiIterativeTermination::Converged);
        assert_eq!(cg.iterations, 1);
    }

    #[test]
    fn matmul_f32_identity() {
        let identity = FfiMatrixF32 { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] };
        let a = FfiMatrixF32 { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let result = matmul_f32(identity, a.clone()).unwrap();
        assert_eq!(result.data, a.data);
    }

    #[test]
    fn matmul_f32_rejects_dimension_mismatch() {
        let a = FfiMatrixF32 { rows: 2, cols: 3, data: vec![0.0; 6] };
        let b = FfiMatrixF32 { rows: 2, cols: 2, data: vec![0.0; 4] };
        assert!(matmul_f32(a, b).is_err());
    }

    #[test]
    fn dot_f32_matches_hand_computation() {
        assert_eq!(dot_f32(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap(), 32.0);
    }

    #[test]
    fn axpy_f32_updates_correctly() {
        let result = axpy_f32(2.0, vec![1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(result, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn norm2_f32_matches_hand_computation() {
        assert_eq!(norm2_f32(vec![3.0, 4.0]), 5.0);
    }

    #[test]
    fn spmv_f32_matches_hand_computation() {
        let matrix = FfiCsrMatrixF32 {
            rows: 2,
            cols: 3,
            row_ptr: vec![0, 2, 3],
            col_indices: vec![0, 2, 1],
            values: vec![1.0, 2.0, 3.0],
        };
        let result = spmv_f32(matrix, vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(result, vec![3.0, 3.0]);
    }

    #[test]
    fn sparse_weighted_least_squares_crosses_the_ffi_boundary() {
        let design = FfiCsrMatrixF64 {
            rows: 4,
            cols: 2,
            row_ptr: vec![0, 1, 3, 5, 7],
            col_indices: vec![0, 0, 1, 0, 1, 0, 1],
            values: vec![1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
        };
        let result = solve_sparse_weighted_least_squares(
            design,
            vec![1.0, 3.0, 5.0, 100.0],
            vec![1.0, 1.0, 1.0, 0.0],
            20,
            1e-12,
        ).unwrap();

        assert!(result.converged);
        assert!((result.solution[0] - 1.0).abs() < 1e-10);
        assert!((result.solution[1] - 2.0).abs() < 1e-10);
        assert!(result.weighted_residual_sum_of_squares < 1e-18);
        assert_eq!(result.penalty_contribution, 0.0);
        assert!(result.objective < 1e-18);
    }

    #[test]
    fn sparse_penalized_least_squares_preserves_objective_terms_over_ffi() {
        let design = FfiCsrMatrixF64 {
            rows: 3,
            cols: 2,
            row_ptr: vec![0, 2, 4, 6],
            col_indices: vec![0, 1, 0, 1, 0, 1],
            values: vec![1.0; 6],
        };
        let penalty = FfiCsrMatrixF64 {
            rows: 2,
            cols: 2,
            row_ptr: vec![0, 1, 2],
            col_indices: vec![0, 1],
            values: vec![1.0, 1.0],
        };
        let result = solve_sparse_penalized_weighted_least_squares(
            design, vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 1.0],
            penalty, 1.0, 20, 1e-12,
        ).unwrap();

        assert!(result.converged);
        assert!((result.solution[0] - 6.0 / 7.0).abs() < 1e-10);
        assert!((result.solution[1] - 6.0 / 7.0).abs() < 1e-10);
        assert!((result.weighted_residual_sum_of_squares - 12.0 / 49.0).abs() < 1e-10);
        assert!((result.penalty_contribution - 72.0 / 49.0).abs() < 1e-10);
        assert!((result.objective - 12.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn configurable_sparse_solve_crosses_ffi_with_warm_start() {
        let design = FfiCsrMatrixF64 {
            rows: 2, cols: 2, row_ptr: vec![0, 1, 2],
            col_indices: vec![0, 1], values: vec![1.0, 1.0],
        };
        let result = solve_sparse_weighted_least_squares_with_options(
            design, vec![2.0, 3.0], vec![1.0, 1.0],
            FfiSparseStatisticalSolveOptions {
                max_iterations: 10, tolerance: 1e-12,
                initial_solution: Some(vec![2.0, 3.0]),
                preconditioner: FfiSparseStatisticalPreconditioner::Jacobi,
            },
        ).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.solution, vec![2.0, 3.0]);
    }

    #[test]
    fn sparse_statistics_distinguish_malformed_csr_from_solver_input_errors() {
        let malformed = FfiCsrMatrixF64 {
            rows: 1, cols: 1, row_ptr: vec![0, 1], col_indices: vec![2], values: vec![1.0],
        };
        assert!(matches!(
            solve_sparse_weighted_least_squares(malformed, vec![1.0], vec![1.0], 10, 1e-8),
            Err(FfiError::DimensionMismatch { .. })
        ));

        let valid = FfiCsrMatrixF64 {
            rows: 1, cols: 1, row_ptr: vec![0, 1], col_indices: vec![0], values: vec![1.0],
        };
        assert!(matches!(
            solve_sparse_weighted_least_squares(valid, vec![1.0], vec![-1.0], 10, 1e-8),
            Err(FfiError::SolverError { .. })
        ));
    }

    fn ampl_example_ffi_problem() -> FfiProblem {
        // Same problem as nc-optimize's own solver tests: maximize
        // 3x+2y (given here pre-negated to minimize -3x-2y) s.t.
        // x+y<=4, x<=3, x in [0,inf), y in [0,10]. Known optimum:
        // (3, 1), objective -11.
        FfiProblem {
            objective: vec![-3.0, -2.0],
            constraints: FfiCsrMatrixF64 {
                rows: 2,
                cols: 2,
                row_ptr: vec![0, 2, 3],
                col_indices: vec![0, 1, 0],
                values: vec![1.0, 1.0, 1.0],
            },
            row_bounds: vec![
                FfiBound { lower: None, upper: Some(4.0) },
                FfiBound { lower: None, upper: Some(3.0) },
            ],
            var_bounds: vec![
                FfiBound { lower: Some(0.0), upper: None },
                FfiBound { lower: Some(0.0), upper: Some(10.0) },
            ],
            is_integer: vec![],
        }
    }

    #[test]
    fn solve_lp_simplex_matches_the_known_optimum() {
        let solution = solve_lp_simplex(ampl_example_ffi_problem()).unwrap();
        assert_eq!(solution.status, FfiSolveStatus::Optimal);
        assert!((solution.variable_values[0] - 3.0).abs() < 1e-6);
        assert!((solution.variable_values[1] - 1.0).abs() < 1e-6);
        assert!((solution.objective_value - (-11.0)).abs() < 1e-6);
    }

    #[test]
    fn solve_lp_interior_point_matches_the_known_optimum() {
        let solution = solve_lp_interior_point(ampl_example_ffi_problem()).unwrap();
        assert_eq!(solution.status, FfiSolveStatus::Optimal);
        assert!((solution.variable_values[0] - 3.0).abs() < 1e-3);
        assert!((solution.variable_values[1] - 1.0).abs() < 1e-3);
        assert!((solution.objective_value - (-11.0)).abs() < 1e-3);
    }

    #[test]
    fn solve_lp_simplex_and_interior_point_agree() {
        let problem = ampl_example_ffi_problem();
        let simplex = solve_lp_simplex(problem.clone()).unwrap();
        let interior = solve_lp_interior_point(problem).unwrap();
        assert!((simplex.objective_value - interior.objective_value).abs() < 1e-3);
    }

    #[test]
    fn solve_lp_detects_infeasibility() {
        // x <= 1 and x >= 2 can't both hold.
        let problem = FfiProblem {
            objective: vec![1.0],
            constraints: FfiCsrMatrixF64 {
                rows: 2,
                cols: 1,
                row_ptr: vec![0, 1, 2],
                col_indices: vec![0, 0],
                values: vec![1.0, 1.0],
            },
            row_bounds: vec![
                FfiBound { lower: None, upper: Some(1.0) },
                FfiBound { lower: Some(2.0), upper: None },
            ],
            var_bounds: vec![FfiBound { lower: Some(0.0), upper: None }],
            is_integer: vec![],
        };
        let solution = solve_lp_simplex(problem).unwrap();
        assert_eq!(solution.status, FfiSolveStatus::Infeasible);
    }

    #[test]
    fn solve_lp_interior_point_surfaces_equality_constraint_rejection_as_solver_error() {
        // Interior point rejects equality rows (zero-width bound) —
        // confirms this crosses the FFI boundary as FfiError::SolverError,
        // not a panic or a silently wrong answer.
        let problem = FfiProblem {
            objective: vec![1.0],
            constraints: FfiCsrMatrixF64 {
                rows: 1,
                cols: 1,
                row_ptr: vec![0, 1],
                col_indices: vec![0],
                values: vec![1.0],
            },
            row_bounds: vec![FfiBound { lower: Some(5.0), upper: Some(5.0) }],
            var_bounds: vec![FfiBound { lower: Some(0.0), upper: Some(10.0) }],
            is_integer: vec![],
        };
        let result = solve_lp_interior_point(problem);
        assert!(matches!(result, Err(FfiError::SolverError { .. })));
    }

    #[test]
    fn solve_lp_rejects_malformed_csr() {
        let problem = FfiProblem {
            objective: vec![1.0],
            constraints: FfiCsrMatrixF64 {
                rows: 1,
                cols: 1,
                row_ptr: vec![0, 1],
                col_indices: vec![5], // out of bounds for 1 column
                values: vec![1.0],
            },
            row_bounds: vec![FfiBound { lower: None, upper: Some(1.0) }],
            var_bounds: vec![FfiBound { lower: Some(0.0), upper: None }],
            is_integer: vec![],
        };
        let result = solve_lp_simplex(problem);
        assert!(matches!(result, Err(FfiError::DimensionMismatch { .. })));
    }

    #[test]
    fn solve_milp_matches_the_known_integer_optimum() {
        // Same classic 2-variable MILP as nc-optimize's own
        // branch_and_bound tests: maximize 5x+4y (given here pre-negated)
        // s.t. 6x+4y<=24, x+2y<=6, x,y>=0 integer. LP relaxation lands
        // on a genuinely fractional vertex (3, 1.5); hand-verified
        // integer optimum is (4, 0), objective -20.
        let problem = FfiProblem {
            objective: vec![-5.0, -4.0],
            constraints: FfiCsrMatrixF64 {
                rows: 2,
                cols: 2,
                row_ptr: vec![0, 2, 4],
                col_indices: vec![0, 1, 0, 1],
                values: vec![6.0, 4.0, 1.0, 2.0],
            },
            row_bounds: vec![
                FfiBound { lower: None, upper: Some(24.0) },
                FfiBound { lower: None, upper: Some(6.0) },
            ],
            var_bounds: vec![
                FfiBound { lower: Some(0.0), upper: None },
                FfiBound { lower: Some(0.0), upper: None },
            ],
            is_integer: vec![true, true],
        };

        let solution = solve_milp_branch_and_bound(problem).unwrap();
        assert_eq!(solution.status, FfiSolveStatus::Optimal);
        assert!((solution.variable_values[0] - 4.0).abs() < 1e-6);
        assert!((solution.variable_values[1] - 0.0).abs() < 1e-6);
        assert!((solution.objective_value - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn solve_milp_with_empty_is_integer_behaves_as_pure_lp() {
        // Empty is_integer -> "all continuous" -> should match
        // solve_lp_simplex exactly on the same problem.
        let problem = ampl_example_ffi_problem();
        let milp_solution = solve_milp_branch_and_bound(problem.clone()).unwrap();
        let lp_solution = solve_lp_simplex(problem).unwrap();
        assert_eq!(milp_solution.status, lp_solution.status);
        assert!((milp_solution.objective_value - lp_solution.objective_value).abs() < 1e-9);
    }

    #[test]
    fn solve_milp_detects_infeasibility_caused_by_integrality() {
        // minimize x s.t. 2x = 1, x integer, x in [0, 10] - the LP
        // relaxation (x = 0.5) is feasible but no integer x satisfies it.
        let problem = FfiProblem {
            objective: vec![1.0],
            constraints: FfiCsrMatrixF64 {
                rows: 1,
                cols: 1,
                row_ptr: vec![0, 1],
                col_indices: vec![0],
                values: vec![2.0],
            },
            row_bounds: vec![FfiBound { lower: Some(1.0), upper: Some(1.0) }],
            var_bounds: vec![FfiBound { lower: Some(0.0), upper: Some(10.0) }],
            is_integer: vec![true],
        };
        let solution = solve_milp_branch_and_bound(problem).unwrap();
        assert_eq!(solution.status, FfiSolveStatus::Infeasible);
    }

    #[test]
    fn configurable_milp_report_crosses_the_ffi_boundary() {
        let problem = FfiProblem {
            objective: vec![-2.0],
            constraints: FfiCsrMatrixF64 {
                rows: 0, cols: 1, row_ptr: vec![0],
                col_indices: vec![], values: vec![],
            },
            row_bounds: vec![],
            var_bounds: vec![FfiBound { lower: Some(0.0), upper: Some(3.0) }],
            is_integer: vec![true],
        };
        let report = solve_milp_branch_and_bound_with_options(
            problem,
            FfiBranchAndBoundOptions {
                max_nodes: 10, integer_tolerance: 1e-8,
                scaling: true, initial_incumbent: vec![],
                absolute_gap_tolerance: 0.0, relative_gap_tolerance: 0.0,
                node_selection: FfiNodeSelection::BestBound,
                branching_strategy: FfiBranchingStrategy::PseudoCost,
                bound_propagation: true,
            },
        ).unwrap();
        assert_eq!(report.solution.status, FfiSolveStatus::Optimal);
        assert_eq!(report.solution.variable_values, vec![3.0]);
        assert_eq!(report.nodes_explored, 1);
        assert_eq!(report.best_bound, Some(-6.0));
        assert_eq!(report.absolute_gap, Some(0.0));
        assert_eq!(report.relative_gap, Some(0.0));
        assert_eq!(report.termination, FfiBranchAndBoundTermination::Exhausted);
        assert_eq!(report.relaxations_solved, 1);
    }

    #[test]
    fn ffi_vector_f64_axpy_in_place_updates_correctly() {
        let x = FfiVectorF64::new(vec![1.0, 1.0, 1.0]);
        let y = FfiVectorF64::new(vec![1.0, 2.0, 3.0]);
        y.axpy_in_place(2.0, &x).unwrap();
        assert_eq!(y.to_vec(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn ffi_vector_f64_dot_matches_hand_computation() {
        let x = FfiVectorF64::new(vec![1.0, 2.0, 3.0]);
        let y = FfiVectorF64::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(x.dot(&y).unwrap(), 32.0);
    }

    #[test]
    fn ffi_vector_f64_norm2_matches_hand_computation() {
        let x = FfiVectorF64::new(vec![3.0, 4.0]);
        assert_eq!(x.norm2(), 5.0);
    }

    #[test]
    fn ffi_vector_f64_len_matches_construction() {
        let x = FfiVectorF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x.len(), 4);
    }

    #[test]
    fn ffi_vector_f64_axpy_in_place_rejects_dimension_mismatch() {
        let x = FfiVectorF64::new(vec![1.0, 1.0]);
        let y = FfiVectorF64::new(vec![1.0, 2.0, 3.0]);
        assert!(y.axpy_in_place(2.0, &x).is_err());
    }

    #[test]
    fn ffi_vector_f64_dot_rejects_dimension_mismatch() {
        let x = FfiVectorF64::new(vec![1.0, 1.0]);
        let y = FfiVectorF64::new(vec![1.0, 2.0, 3.0]);
        assert!(x.dot(&y).is_err());
    }

    #[test]
    fn ffi_vector_f64_chained_axpy_matches_repeated_kernel_axpy() {
        // Cross-check: several in-place axpy calls against the same
        // handle should match doing the equivalent with the plain
        // Vec-based kernel directly.
        let handle = FfiVectorF64::new(vec![0.0, 0.0, 0.0]);
        let step = FfiVectorF64::new(vec![1.0, 1.0, 1.0]);
        for _ in 0..3 {
            handle.axpy_in_place(1.0, &step).unwrap();
        }

        let mut plain = vec![0.0, 0.0, 0.0];
        for _ in 0..3 {
            kernels::axpy(1.0, &[1.0, 1.0, 1.0], &mut plain);
        }

        assert_eq!(handle.to_vec(), plain);
    }

    #[test]
    fn ffi_vector_f32_axpy_in_place_updates_correctly() {
        let x = FfiVectorF32::new(vec![1.0, 1.0, 1.0]);
        let y = FfiVectorF32::new(vec![1.0, 2.0, 3.0]);
        y.axpy_in_place(2.0, &x).unwrap();
        assert_eq!(y.to_vec(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn ffi_vector_f32_dot_matches_hand_computation() {
        let x = FfiVectorF32::new(vec![1.0, 2.0, 3.0]);
        let y = FfiVectorF32::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(x.dot(&y).unwrap(), 32.0);
    }

    #[test]
    fn ffi_vector_f32_norm2_matches_hand_computation() {
        let x = FfiVectorF32::new(vec![3.0, 4.0]);
        assert_eq!(x.norm2(), 5.0);
    }
}
