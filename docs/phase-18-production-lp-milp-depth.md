# Phase 18: production LP/MILP depth

Phase 18 strengthens the native branch-and-bound engine without changing its
role as an embedded solver for small and medium structured models.

## Bound propagation

Before each node relaxation, singleton constraint rows are converted into
variable bounds and intersected with the current node domain. Integer bounds
are rounded inward to the integer lattice. A contradictory domain is pruned
without invoking simplex. Propagation can be disabled for controlled solver
comparisons and its tightening count is reported.

## Branching

Pseudo-cost branching is now the default. Each solved child records objective
degradation per unit down/up branch displacement. Once both directional costs
exist, candidate variables are ranked using their estimated balanced branch
gain. Unobserved candidates fall back to deterministic most-fractional
branching, which remains available as an explicit strategy.

## Production telemetry

The complete search certificate now crosses UniFFI: termination reason, node
selection, branching policy, gap tolerances, infeasibility and bound pruning,
maximum depth, LP relaxations solved, propagated bounds, and incumbents found.
This makes early termination and comparative tuning observable from Swift.

This phase does not claim parity with large commercial MILP engines. Sparse
basis updates, general multi-row propagation, cutting planes, and strong or
reliability branching remain later work.
