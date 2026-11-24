# Phase 53 — Robust optimization certificates

Phase 53 separates a solver's termination report from independently recomputed
evidence about the returned point. `OptimizationCertificate` records a typed
claim (`Optimality`, `Feasibility`, `Candidate`, or `Rejected`), named residual
metrics with their tolerances, and non-fatal structural warnings.

| Evidence | Scope of the claim |
| --- | --- |
| Linear primal audit | `certify_linear_solution` rebuilds the objective, row and variable feasibility, and integrality from the supplied model and point. A successful status and passing evidence establish `Feasibility` within the configured tolerances. |
| Caller-supplied gap | The named `caller-supplied gap` metric checks that the scalar is finite and belongs to `[0, tolerance.gap]`. Its provenance is the caller, and a passing check preserves the `Feasibility` claim. A failing check returns `Rejected`. |
| Local nonlinear stationarity | The reported gradient or KKT residual measures first-order stationarity at the returned point. Check primal feasibility and termination alongside it. Its scope is local and first order. |
| Convex QP KKT evidence | `certify_convex_qp` checks convexity and recomputes the objective and primal-dual residuals. Passing evidence establishes numerical `Optimality` for the validated convex model at the configured tolerances. |
| Bounds from a search | Local Benders, branch-and-bound, and spatial search report the bounds and gap produced by the solve. Limits can leave partial evidence. Read the termination reason and the bounds together. |
| Independently replayed proof | `audit_benders_proof` checks exported witnesses against the original continuous two-stage LP, rebuilds cuts and bounds, and computes a claim. `Optimality` requires a verified incumbent and a closed gap under the audit tolerances. `Bounds` can describe partial evidence, and `Infeasibility` requires a verified infeasibility witness. |

`verified()` applies to the displayed `OptimizationCertificate.claim`. In
particular, a verified feasibility claim describes the point and its
reconstructed objective.

`certify_convex_qp` independently recomputes primal feasibility, objective,
stationarity, dual sign feasibility, and complementarity from the primal point
and row and variable multipliers. It also checks the numerical rank of active
constraint gradients. Dependent or overdetermined active sets are warnings,
because degeneracy alone does not invalidate valid KKT evidence.

The certificate checks dimensions, tolerances, and numerical values. Invalid
tolerance settings return an error. Non-finite or inconsistent solution
evidence fails closed as `Rejected`. The standard validation corpus covers
convex QP KKT evidence, a zero gap supplied for a feasible suboptimal LP point,
local and distributed Benders limits, and replay of valid and altered proofs.

This phase does not yet claim rigorous interval enclosures, LP/MILP Farkas
infeasibility certificates, unbounded rays, or exact branch-and-bound proof
logs. Those require solver interfaces that retain the corresponding dual, ray,
and search-tree evidence and are natural extensions of this format.
