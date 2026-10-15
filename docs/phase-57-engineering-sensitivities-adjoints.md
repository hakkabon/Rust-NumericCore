# Phase 57 — Engineering sensitivities and adjoints

Phase 57 converts the shared engineering representation's exact sparse
`F_y`, `F_ydot`, and `F_p` blocks into implicit solution and response
sensitivities. The effective tangent is `A = F_y + alpha F_ydot`: steady
analysis uses zero, while an implicit DAE step supplies its local derivative
coefficient.

Forward mode reuses a sparse LU factorization to solve `A dy/dp = -F_p` for
each model parameter. Scalar responses use a transposed factorization for
`A^T lambda = dg/dy`, followed by
`dg/dp = partial(g)/partial(p) - F_p^T lambda`. Direct parameter dependence is
therefore retained. Residual norms and an independent forward contraction are
reported, making forward/adjoint agreement a numerical correctness check.

The capability is domain neutral and applies to circuit, mechanical, thermal,
and implicit-step models expressed as `EngineeringModel`. Full trajectory
adjoints, event-time derivatives, eigenvalue clusters, contact active sets,
and mesh-changing sensitivities remain outside this phase.
