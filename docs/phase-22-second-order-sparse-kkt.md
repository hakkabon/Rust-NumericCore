# Phase 22: second-order derivatives and sparse KKT systems

Phase 22 extends the shared nonlinear graph with exact second derivatives for
every existing node. Expressions expose value, gradient, dense Hessian,
Hessian-vector products, and tolerance-filtered CSR Hessians. Constrained
models additionally evaluate weighted Hessians of the Lagrangian and
Lagrangian Hessian-vector products.

`SparseKktProblem` represents the regularized saddle-point system

```text
[ H + δI   Jᵀ ] [p] = [rₚ]
[ J       -γI ] [y]   [r𝚌]
```

without converting its Hessian or Jacobian inputs to dense storage. The Rust
solver assembles canonical CSR, applies pivoted sparse LU, and reports absolute
and relative residuals plus factor nonzeros. The full contract crosses UniFFI.

SQP can select exact Lagrangian curvature instead of BFGS. BFGS remains the
default for backward compatibility and for models where dense exact Hessian
formation is not attractive.

Exact graph Hessians currently require O(nodes × parameters²) work and
storage before sparsification. The KKT solver lacks fill-reducing ordering,
symbolic reuse, inertia control, and iterative block preconditioning. These
are the next requirements for genuinely large constrained models.
