# Phase 37: nonlinear equation systems and continuation

`nc-optimize` now represents and solves square systems `F(x) = 0` directly,
without a least-squares reformulation.

The dense path uses damped Newton, pivoted solves, residual-norm line search,
scaling, and configurable modified-Newton Jacobian reuse. The matrix-free path
uses inexact Newton with GMRES and Jacobian-vector products. Both report
residual and work diagnostics, validate every callback, and distinguish
singularity, Krylov failure, globalization failure, limits, and cancellation.

`NonlinearEquationModel` reuses the portable expression graph for exact dense
Jacobians. Adaptive natural-parameter continuation tracks nearby roots, while
pseudo-transient continuation regularizes difficult or initially singular
operating points through `(J + I/dt) s = -F`.

The APIs mirror Swift's native implementation. Callback entry points remain
outside UniFFI; a later value-only equation-model transport can run the shared
graph wholly inside Rust after the schema is stable.
