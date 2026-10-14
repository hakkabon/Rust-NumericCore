# Phase 39: time integration and DAE foundation

Phase 39 establishes transient numerical simulation without coupling the core
to a particular engineering domain.

The explicit path is an adaptive Dormand–Prince 5(4) integrator with weighted
RMS error control, forward or reverse integration, observer cancellation,
piecewise cubic Hermite dense output, and directional terminal or nonterminal
zero-crossing events.

The implicit path accepts an index-1 residual model
`F(t, y, yDot) = 0` with analytic `F_y` and `F_yDot`. Backward-Euler steps solve
`(F_y + F_yDot/h) delta = -F` by Newton iteration. Step doubling supplies the
local error estimate and adaptive controller. `DaeEvaluation::from_mass_matrix`
is the convenience seam for `M*yDot = f(t,y)` and deliberately permits a
singular mass matrix for semi-explicit DAEs.

Both paths validate dimensions and finite arithmetic, preserve accepted-step
history, and distinguish end-time completion, events, cancellation, step
limits, underflow, and Newton failure. The implementation remains in the Rust
domain layer for this phase; callback-driven integration is not forced through
UniFFI.
