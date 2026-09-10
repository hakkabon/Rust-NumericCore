//! SIMD-accelerated kernels — **scaffold, not yet implemented**.
//!
//! Intent: `std::simd` (portable_simd) explicit vectorization of the
//! `nc-kernels-generic` operations, for platforms/dtypes where autovec
//! from the generic crate isn't hitting good codegen. `portable_simd` is
//! nightly-only as of this writing, so this crate currently just
//! re-exports the generic implementations so the dependency graph and
//! `Backend` dispatch wiring can be built and tested end-to-end before
//! the SIMD work itself is scheduled.
//!
//! When picking this up: gate the real implementation behind a
//! `#[cfg(feature = "nightly-simd")]` feature so `cargo test` on stable
//! keeps working via the re-exported fallback.

pub use nc_kernels_generic::{axpy, dot, matmul, norm2, Scalar};
