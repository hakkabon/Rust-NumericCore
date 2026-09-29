# Phase 10: constrained nonlinear foundation

Phase 10 extends the shared nonlinear graph into general smooth constrained
optimization. `NonlinearConstraint` uses the same bound form as the linear
layer:

```text
lower <= expression(parameters) <= upper
```

This represents equalities, one-sided inequalities, and ranges without separate
constraint kinds. `ConstrainedNonlinearProblem` combines a Phase 9 objective
model with one or more validated constraint expressions.

`minimize_constrained` implements a Powell–Hestenes–Rockafellar augmented
Lagrangian outer iteration. Each smooth bound-constrained subproblem is solved
by the Phase 8 projected limited-memory solver. Equality multipliers are signed;
lower and upper inequality multipliers are non-negative and updated separately.

The result reports the original objective, raw constraint values, multipliers,
maximum primal violation, projected Lagrangian stationarity, outer and inner
iteration counts, augmented-objective evaluations, final penalty, and an
explicit termination reason. An outer-iteration observer supports cooperative
cancellation.

This is a constrained nonlinear foundation, not yet a full production SQP or
interior-point implementation. In particular, it does not provide second-order
constraint information, a filter globalization strategy, feasibility
restoration, or constraint qualifications/rank diagnostics. The shared problem
and diagnostic contract is designed so those solvers can be added without
changing model representation.
