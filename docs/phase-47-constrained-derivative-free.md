# Phase 47: constrained derivative-free optimization

The constrained derivative-free API evaluates the objective and all black-box
constraints together. Inequalities follow `c(x) <= 0`; equalities follow
`h(x) = 0`. Shared coordinate samples construct affine objective and constraint
models without derivative callbacks or repeated simulations per constraint.

An adaptive exact-penalty merit function drives the bound-constrained
trust-region step. Infeasible starts receive feasibility-improving steps;
near-feasible iterates also consider objective descent projected tangentially
to equality and active-inequality models. Actual/predicted reduction adapts the
radius, while stalled feasibility increases the penalty.

Reports contain the final constraint vector, maximum and summed violation,
evaluation/model counts, accepted and restoration steps, penalty updates,
radius, and reduction ratio. Radius exhaustion distinguishes feasible
`Converged` from `InfeasibleStationary`.

This is COBYLA-inspired infrastructure, not a compatibility claim. General
poised simplices, exact linear trust-region subproblem solves, and filter-based
globalization remain possible later extensions.
