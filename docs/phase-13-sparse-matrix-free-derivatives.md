# Phase 13: sparse and matrix-free derivatives

Phase 13 adds scalable first-derivative evaluation to the shared nonlinear
expression graph. Scalar expressions now support a reverse-mode value and
sparse-gradient sweep, while residual models can assemble a canonical CSR
Jacobian. Repeated parameter nodes are accumulated, zero derivatives are
omitted, and column indices are emitted in ascending order.

Residual models also expose matrix-free `Jv` and `Jᵀv` products. `Jv` uses a
forward directional sweep through each expression and never constructs a
gradient or Jacobian. `Jᵀv` uses reverse accumulation one residual at a time,
so it does not materialize the full Jacobian. These operations are the core
derivative primitives needed by truncated-Newton, Gauss-Newton/Krylov, and
large constrained nonlinear methods.

The UniFFI surface transports sparse gradients and CSR Jacobians with explicit
dimensions and exports both matrix-free products. Swift and Rust therefore use
the same model graph and expose matching derivative semantics without foreign
callbacks in numerical hot loops.

This phase supplies the first-order representation and products. Phase 22 adds
exact graph Hessians, Hessian-vector products, and sparse KKT systems. Future
work can add sparsity-pattern caching, coloring, and matrix-free nonlinear
solver integration.
