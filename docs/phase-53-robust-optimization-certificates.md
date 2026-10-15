# Phase 53 — Robust optimization certificates

Phase 53 separates a solver's termination report from independently recomputed
evidence about the returned point. `OptimizationCertificate` records a typed
claim (`Optimality`, `Feasibility`, `Candidate`, or `Rejected`), named residual
metrics with their tolerances, and non-fatal structural warnings.

`certify_linear_solution` recomputes the original-model objective, row and
variable feasibility, and integrality. A successful solve is certified as
feasible, not optimal, unless its caller supplies a finite independently
established global gap. Solver status alone is intentionally insufficient
while the production LP interface does not expose a complete dual or ray
certificate.

`certify_convex_qp` independently recomputes primal feasibility, objective,
stationarity, dual sign feasibility, and complementarity from the primal point
and row and variable multipliers. It also checks the numerical rank of active
constraint gradients. Dependent or overdetermined active sets are warnings,
because degeneracy alone does not invalidate valid KKT evidence.

Dimensions, tolerances, and numerical values are validated, and non-finite or
inconsistent evidence fails closed as `Rejected`. The standard validation
corpus contains an independently certified convex-QP case, while focused tests
corrupt a valid solution to prove that the verifier detects bad evidence.

This phase does not yet claim rigorous interval enclosures, LP/MILP Farkas
infeasibility certificates, unbounded rays, or exact branch-and-bound proof
logs. Those require solver interfaces that retain the corresponding dual, ray,
and search-tree evidence and are natural extensions of this format.
