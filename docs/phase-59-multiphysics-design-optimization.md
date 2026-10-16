# Phase 59 — Multiphysics and design optimization

Phase 59 groups the shared engineering residual system into named disciplines
without splitting its state, sparse Jacobian, or parameter space. Discipline
rows must form an exact partition, so a monolithic sparse Newton solve retains
all cross-physics derivatives while results expose per-discipline residual
norms.

The steady solver uses sparse LU Newton steps and residual-decreasing
backtracking. It reports global and discipline residuals, iterations, linear
solves, and explicit termination. Parameter replacement preserves engineering
metadata and graph topology for repeated design analyses.

Bounded design optimization evaluates a graph response after each converged
coupled solve. Exact response derivatives feed the Phase 57 adjoint, and the
total parameter gradient drives L-BFGS-B. Nearby designs reuse the last state;
a cold retry prevents a poor warm state from causing a false failure.

Validation couples field and thermal balance equations and recovers an
analytic regularized optimum. Trajectory adjoints, partitioned co-simulation,
mesh morphing, topology-changing contact, and Pareto optimization remain
future extensions.
