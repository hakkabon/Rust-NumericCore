# Phase 23: matrix-free nonlinear least squares

Phase 23 adds a Gauss–Newton/Levenberg–Marquardt path that never constructs a
Jacobian or normal matrix. Each Krylov iteration applies `(Jᵀ W J + λI)v`
using directional (`Jv`) and reverse (`Jᵀv`) graph products. Storage is linear
in the parameter and residual dimensions.

Options control the outer damping loop and inner conjugate-gradient limit and
tolerance. Observation weights, Huber/Cauchy IRLS weights, box projection, and
gain-ratio acceptance are supported. Results report Krylov iterations and both
derivative-product counts.

The dense QR solver remains the default compatibility path. The matrix-free
solver is explicit and available through Rust, UniFFI, and both Swift backends.
Current damping is scalar (`λI`); preconditioning, adaptive forcing sequences,
and trust-region CG remain follow-up work.
