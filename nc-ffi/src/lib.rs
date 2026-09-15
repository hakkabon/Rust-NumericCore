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

use nc_kernels_generic as kernels;
use nc_sparse::CsrMatrix;
use std::sync::{Arc, Mutex};

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
}

impl From<nc_sparse::SparseError> for FfiError {
    fn from(err: nc_sparse::SparseError) -> Self {
        FfiError::DimensionMismatch { message: err.to_string() }
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
