# 0003 — Matrix decompositions: wrap Accelerate/LAPACKE first, don't reimplement

## Status
Accepted.

## Context
LU, QR, Cholesky, and SVD decompositions are available, well-tested, and
fast in Apple's Accelerate framework (via LAPACK/LAPACKE linkage).
Reimplementing them in `nc-decomp` (Rust) would mean redoing genuinely
hard numerical work — pivoting strategies, rank-deficient and
near-singular edge cases, numerical stability — that LAPACK has decades
of engineering behind.

This was flagged during initial scoping conversation as the
highest-variance item in the whole project: "naive LU/QR is a week; LU
with proper pivoting, SVD with edge-case handling, numerically stable
eigenvalue algorithms — that's where 'I thought this was done'
surprises live."

## Decision
v1 does **not** implement decompositions in `nc-decomp`. Instead,
`NumericCoreAccelerate` wraps Accelerate's LAPACKE-linked routines
directly (`sgetrf`/`dgetrf` for LU, `sgeqrf`/`dgeqrf` for QR, etc.).

`nc-decomp` (Rust) stays a near-empty scaffold — `DecompositionKind`
enum only — until a *specific, concrete* gap shows up that Accelerate
doesn't cover: a portable (non-Accelerate) fallback for a future non-Mac
target, an extended-precision variant, or similar.

## Consequences
- Decomposition support ships far sooner and far more reliably than a
  from-scratch implementation would allow.
- `NumericCore` (pure Swift/Rust fallback, no Accelerate dependency) has
  **no decomposition support at all** until `nc-decomp` grows real
  content. This is an accepted gap for v1 — anyone using `NumericCore`
  without `NumericCoreAccelerate` cannot factorize matrices yet.
- Column-major storage (ADR 0001) is what makes this wrapping
  straightforward — LAPACKE also expects column-major/Fortran-ordered
  input, so no transpose is needed at this boundary either.

## Alternatives considered
- **Implement LU/QR/SVD in Rust from the start**, for portability beyond
  Apple platforms. Rejected for v1: this project's primary target is
  explicitly the Mac platform (per the original scoping conversation);
  portability is a real but secondary goal that shouldn't gate getting a
  correct, fast decomposition story shipped first.
