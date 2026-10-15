# Phase 49 — Nonlinear scaling and sensitivity

Phase 49 builds an equilibrated implicit-sensitivity layer on the common
nonlinear graph. It accepts a constrained solution or an explicit point and
multiplier vector, detects active nonlinear and variable bounds, evaluates the
exact Lagrangian Hessian, and solves the active-set KKT derivative system.

Automatic variable factors use point magnitudes and finite bound widths;
callers may override them when engineering units provide better characteristic
scales. Active Jacobian rows are equilibrated after variable scaling. Results
are restored to original units and include primal and multiplier derivatives,
the active set, Jacobian norm ranges, pivot ratio, and residual norm.

Sensitivities are local: they assume a stable active set and nonsingular
regularized KKT matrix. Degenerate systems report an error instead of silently
returning an unstable derivative.
