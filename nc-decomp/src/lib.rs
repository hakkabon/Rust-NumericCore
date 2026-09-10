//! Matrix decompositions — **scaffold**.
//!
//! Deliberately empty of real numerics for v1. Per the lean-v1 scoping
//! decision (`docs/decisions/0003-decomposition-scope.md`), LU/QR/SVD are
//! wrapped from Accelerate/LAPACKE on the Swift side
//! (`NumericCoreAccelerate`) for v1. This crate only grows real content
//! when a specific decomposition is needed that Accelerate doesn't cover
//! well (e.g. a portable fallback for non-Mac targets, or an
//! extended-precision variant).
//!
//! Do not start here — get `NumericCoreAccelerate`'s LAPACKE wrapping
//! working first and come back only when a concrete gap shows up.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionKind {
    Lu,
    Qr,
    Cholesky,
    Svd,
}
