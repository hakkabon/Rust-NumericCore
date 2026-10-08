# Phase 19: nonlinear interior point

Phase 19 adds a feasible-start interior-point solver for smooth constrained
models represented by the shared nonlinear expression graph. Nonlinear
inequalities and finite variable bounds use logarithmic barriers; equalities
use multiplier updates with a quadratic penalty. A dense inverse-BFGS model
provides curvature while preserving the graph's first-derivative contract.

The solver reports primal violation, Lagrangian stationarity,
complementarity, constraint multipliers, barrier progress, evaluations, and
accepted/rejected steps. Its observer receives one snapshot per outer barrier
iteration and can cancel cleanly. A non-strict inequality or variable-bound
start returns `InfeasibleStart` explicitly; it is never projected silently.

`nc-ffi` exports the complete options, termination, multiplier, and result
contract. This lets Swift select either implementation against the same model
and compare numerical certificates rather than merely final points.

This phase deliberately remains a first-derivative, feasible-start method.
Restoration phases, filter globalization, sparse KKT factorization, exact or
limited-memory Hessian operators, and warm starts for primal/dual/barrier state
are natural follow-on work.
