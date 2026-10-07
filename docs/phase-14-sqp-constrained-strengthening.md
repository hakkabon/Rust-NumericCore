# Phase 14: SQP and constrained nonlinear strengthening

Phase 14 adds sequential quadratic programming alongside the existing
augmented-Lagrangian constrained solver. Each SQP iteration evaluates the
shared nonlinear objective and constraint graph, linearizes the constraints,
and solves a convex step-space QP using the Phase 11 solver. Original variable
bounds become bounds on the step, so every accepted iterate remains inside the
model domain.

The Hessian of the Lagrangian is approximated with a positive-definite BFGS
update. Curvature failures reset the approximation rather than passing an
indefinite matrix to the convex QP layer. An exact L1 constraint-violation
merit function globalizes the step with Armijo backtracking, and the merit
penalty is raised above the current multiplier scale when required.

Results distinguish convergence, iteration and step limits, failed line
searches, failed QP subproblems, and observer cancellation. They also expose
constraint multipliers, maximum primal violation, projected Lagrangian
stationarity, evaluation and line-search counts, final merit penalty, and the
last step norm.

The complete options/result contract crosses UniFFI, allowing the same
graph-represented problem to run in Swift or Rust without callbacks crossing
the numerical boundary. The augmented-Lagrangian solver remains available as
an independent method and potential restoration fallback.

This is a dense-Hessian SQP foundation. Sparse Hessian approximations,
elastic-mode feasibility restoration, second-order corrections, and
filter/trust-region globalization remain future extensions.
