# 0006 — Portable sparse weighted and penalized least squares

## Status

Accepted.

## Context

Swift-DataLens now has additive-model workflows, and its Apple numerical
layer exposes dense QR-based weighted/penalized least squares. Large smooth
bases, high-cardinality effects, and future likelihood-GAM IRLS iterations
need the same statistical objective away from Accelerate and without
densifying an `n × p` design or penalty matrix.

## Decision

`nc-iterative` provides matrix-free CGLS entry points over `CsrMatrix<f64>`:

```text
min Σᵢ wᵢ(yᵢ − xᵢᵀβ)²
min Σᵢ wᵢ(yᵢ − xᵢᵀβ)² + λ‖Pβ‖²
```

The implementation applies the augmented operator `[W½X; √λP]` and its
adjoint using CSR `spmv`/`transpose_spmv`. It never builds `XᵀWX + λPᵀP` and
never creates a dense augmented matrix. Zero weights exclude observations;
negative/non-finite values and mismatched shapes are errors. The result
reports coefficients, the weighted residual contribution, the penalty
contribution, iteration count, and augmented-residual norm.

CGLS stops on the relative normal residual `‖Bᵀr‖`, not `‖r‖`: a regularized
least-squares optimum intentionally has nonzero data/penalty residual. A
non-converged iterate is returned with `converged == false`, and must not be
presented as a fitted model.

## Consequences

- The statistical objective is portable across Rust-supported targets and
  scales with nonzeros rather than dense design size.
- CSR remains the only sparse representation; transpose products are
  accumulated directly, avoiding an unneeded CSC maintenance burden.
- This is an iterative solver, not a replacement for Swift's rank-revealing
  QR on small dense problems. It intentionally does not claim numerical rank,
  effective degrees of freedom, covariance, or penalty selection.
- The UniFFI bridge exports the two solves and the full convergence/objective
  result as `FfiSparseStatisticalSolveResult`. Rust release automation verifies
  that generated Swift bindings contain both exports, while Swift-NumericCore's
  sync workflow refreshes bindings and the checksummed XCFramework together.
  That lockstep update is required: source bindings and framework ABI are one
  release unit.
