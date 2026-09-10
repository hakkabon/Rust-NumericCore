//! `nc-core` — storage, strides, shape, and dtype bookkeeping.
//!
//! This crate deliberately contains **no numeric kernels** (no matmul, no
//! dot product, nothing that touches Accelerate or does floating-point
//! arithmetic beyond bookkeeping). It exists purely to answer:
//!
//! - Where does an element at logical index `(i, j, ...)` live in memory?
//! - Is this buffer owned or borrowed?
//! - What runtime dtype tag crosses the FFI boundary?
//!
//! Kernels (in `nc-kernels-generic` / `nc-kernels-simd`), decompositions
//! (`nc-decomp`), and backends (Swift side) all build on top of the types
//! defined here. See `docs/decisions/0001-column-major-layout.md` for why
//! `Layout` defaults to column-major.

mod buffer;
mod dtype;
mod layout;
mod shape;

pub use buffer::Buffer;
pub use dtype::DType;
pub use layout::{Layout, Order};
pub use shape::Shape;

/// Crate-wide error type. Kept small and FFI-friendly — see
/// `nc-ffi` for how this maps to C-compatible error codes.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Shape, actual: Shape },

    #[error("index out of bounds: {index:?} for shape {shape:?}")]
    IndexOutOfBounds { index: Vec<usize>, shape: Shape },

    #[error("buffer length {actual} does not match shape {shape:?} (needs {expected})")]
    BufferLengthMismatch {
        shape: Shape,
        expected: usize,
        actual: usize,
    },
}

pub type Result<T> = std::result::Result<T, CoreError>;
