# Phase 55 — Production DAE integration

Phase 55 strengthens residual-form index-1 DAE integration around engineering
failure modes: inconsistent starts, events, and known discontinuities. It
extends the shared adaptive implicit integrator rather than adding a separate
circuit time solver.

`consistent_dae_initial_conditions` Newton-corrects `F(t, y, y_dot) = 0`.
For each component the caller selects state or derivative as the free unknown;
the square Newton matrix takes the corresponding column from `F_y` or
`F_yDot`. This supports singular mass matrices while preserving selected
physical initial data. Results report the corrected pair, residual, work, and
correction partition. Bad shapes, singular selections, non-finite iterates,
and nonconvergence fail explicitly.

`integrate_dae_with_events` gives DAEs the ODE event contract: multiple events,
direction filters, terminal records, Hermite root location and derivative, and
observer cancellation. Validated hard stop times prevent steps from crossing
known discontinuities. `CircuitModel::transient` automatically passes waveform
breakpoints, including Phase 54 SPICE `PULSE` edges.

The standard validation corpus corrects a singular-mass index-1 initial pair,
observes a hard stop, and locates a terminal event against an analytic result.
Rust and Swift expose matching behavior.

The stepper remains adaptive Newton-corrected backward Euler with step
doubling. Variable-order BDF, sparse Newton factor reuse, algebraic event
projection, and index reduction remain future numerical depth.
