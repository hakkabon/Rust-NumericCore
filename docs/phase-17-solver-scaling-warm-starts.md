# Phase 17: solver scaling and warm starts

Phase 17 adds a shared, reversible scaling layer for linear optimization and
turns warm starts into an explicit cross-language workflow.

## Linear scaling

`ScaledProblem` performs alternating max-norm row and column equilibration.
It records every diagonal factor, transforms bounds and objective coefficients,
and restores the solution and objective in source units. `ScaledSolver<S>`
decorates any LP solver, so revised simplex and interior point share one
implementation. Swift FFI entry points enable scaling by default and configured
calls can disable it for controlled comparisons.

Integer columns always retain a unit variable factor. Row scaling is safe for
MILP, but scaling an integer variable would alter its lattice; the implementation
therefore refuses to make that mathematically invalid shortcut.

## Warm starts

Branch-and-bound accepts an optional incumbent. It is independently checked for
dimension, finiteness, row and variable feasibility, and integrality before it
can prune a node. A valid incumbent remains available when a node limit ends the
search early and therefore yields a useful feasible result and gap certificate.

QP results now produce a complete warm-start snapshot containing primal, row
dual, and variable-dual values. Existing sparse iterative/statistical warm starts
remain unchanged.

## Boundary and tests

The UniFFI option records carry scaling controls and the MILP incumbent through
to Swift. Conformance tests cover ill-scaled LP restoration, preservation of the
integer lattice, accepted and rejected MILP incumbents, early termination, and
QP result reuse.
