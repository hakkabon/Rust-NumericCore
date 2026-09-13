//! Swift-facing FFI surface, via UniFFI (ADR 0005).
//!
//! This is intentionally a **small, proven slice**, not the full API:
//! `nc-kernels-generic`'s matmul/dot/axpy/norm2, and `nc-sparse`'s CSR
//! SpMV. The goal right now is to prove the whole round trip — Rust
//! function, UniFFI-generated Swift bindings, a Swift call site, a
//! result back — works end to end, before more of the surface (buffers
//! shared by reference rather than copied, `nc-optimize::Solver`, etc.)
//! is built on top of a pattern that hasn't been proven yet.
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

use nc_kernels_generic as kernels;
use nc_sparse::CsrMatrix;

uniffi::setup_scaffolding!();

/// Errors that can cross the FFI boundary. Deliberately flat and
/// string-carrying rather than mirroring each source crate's error type
/// exactly — UniFFI generates a Swift `enum FfiError: Error` from this,
/// and `NCBindings` is expected to translate it into `NumericCore`'s
/// `NCError` at the call site, not re-expose this type to application code.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}

impl From<nc_sparse::SparseError> for FfiError {
    fn from(err: nc_sparse::SparseError) -> Self {
        FfiError::DimensionMismatch(err.to_string())
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

#[uniffi::export]
pub fn matmul_f64(a: FfiMatrixF64, b: FfiMatrixF64) -> Result<FfiMatrixF64, FfiError> {
    if a.cols != b.rows {
        return Err(FfiError::DimensionMismatch(format!(
            "matmul: {}x{} * {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
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
        return Err(FfiError::DimensionMismatch(format!(
            "matmul: {}x{} * {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let (m, k, n) = (a.rows as usize, a.cols as usize, b.cols as usize);
    let mut c = vec![0.0f32; m * n];
    kernels::matmul(&a.data, &b.data, &mut c, m, k, n);
    Ok(FfiMatrixF32 { rows: a.rows, cols: b.cols, data: c })
}

#[uniffi::export]
pub fn dot_f64(x: Vec<f64>, y: Vec<f64>) -> Result<f64, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch(format!(
            "dot: lengths {} and {}",
            x.len(),
            y.len()
        )));
    }
    Ok(kernels::dot(&x, &y))
}

/// The `f32` counterpart of `dot_f64`.
#[uniffi::export]
pub fn dot_f32(x: Vec<f32>, y: Vec<f32>) -> Result<f32, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch(format!(
            "dot: lengths {} and {}",
            x.len(),
            y.len()
        )));
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
        return Err(FfiError::DimensionMismatch(format!(
            "axpy: lengths {} and {}",
            x.len(),
            y.len()
        )));
    }
    let mut result = y;
    kernels::axpy(alpha, &x, &mut result);
    Ok(result)
}

/// The `f32` counterpart of `axpy_f64`.
#[uniffi::export]
pub fn axpy_f32(alpha: f32, x: Vec<f32>, y: Vec<f32>) -> Result<Vec<f32>, FfiError> {
    if x.len() != y.len() {
        return Err(FfiError::DimensionMismatch(format!(
            "axpy: lengths {} and {}",
            x.len(),
            y.len()
        )));
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
    let csr = CsrMatrix::new(
        matrix.rows as usize,
        matrix.cols as usize,
        matrix.row_ptr.iter().map(|&v| v as usize).collect(),
        matrix.col_indices.iter().map(|&v| v as usize).collect(),
        matrix.values,
    )?;
    Ok(csr.spmv(&x)?)
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
}
