# Phase 36: derivative-free optimization

`nc-optimize` now complements Powell direction-set with bound-aware
`minimize_nelder_mead` and `minimize_pattern_search`. Both accept value-only
callbacks, optional box bounds, explicit iteration and evaluation budgets, and
observer-enabled variants for cancellation.

Nelder-Mead reports its final simplex diameter and objective spread. Coordinate
pattern search reports its final poll step and successful-poll count. Trial
points are projected into the box, and malformed bounds or non-finite objective
values fail closed.

The APIs mirror Swift's native implementation. They intentionally remain
outside UniFFI because arbitrary callbacks cannot cross that boundary; a future
shared-model adapter may evaluate the existing nonlinear graph wholly in Rust.
