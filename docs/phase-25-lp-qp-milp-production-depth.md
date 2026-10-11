# Phase 25: LP/QP/MILP production depth

Phase 25 hardens the established linear and convex optimization paths rather
than adding another solver family.

The revised simplex solver still uses a dense explicit basis inverse, but now
reconstructs that inverse from the current basis every 64 pivots. This limits
roundoff accumulated by repeated rank-one updates while retaining the existing
repeated-basis Bland fallback.

The convex QP ADMM solver now balances its penalty parameter from primal and
dual residuals every 25 iterations. A penalty change preserves the unscaled
dual estimate and refactors the penalized Hessian. This makes convergence less
sensitive to the caller's initial `rho`, including warm-started solves.

MILP branch-and-bound can attempt a nearest-integer incumbent at each
fractional LP relaxation. Candidates pass the same independent feasibility,
integrality, and objective diagnostics as other incumbents before acceptance.
The option is enabled by default and reports attempt and success counts through
the native and FFI result contracts.

These changes improve numerical resilience and early incumbent discovery, but
do not replace the dense simplex representation, add cut generation, or turn
the current solver into a large-scale commercial MILP implementation.
