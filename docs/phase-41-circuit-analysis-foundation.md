# Phase 41: circuit-analysis foundation

Phase 41 specializes the shared engineering and DAE infrastructure as a small,
native circuit-analysis kernel. `CircuitModel` compiles named nodes and devices
into modified nodal analysis (MNA), removes the reference-node voltage from the
unknown vector, and adds branch-current unknowns only for ideal voltage sources
and inductors.

The initial device set is deliberately foundational:

- resistors, capacitors, and inductors;
- independent current and voltage sources with DC, sinusoidal, and pulse data;
- exponential diodes with analytic conductance stamps.

Every assembly produces sparse state and derivative Jacobians through
`EngineeringAssembly`. DC operating points reuse the nonlinear-system solver;
transients reuse the residual-form DAE integrator. Operating-point results
recover every node voltage and voltage-defined branch current and separate KCL
from branch-equation residual diagnostics.

This is a numerical circuit layer, not yet a SPICE implementation. Netlist
syntax, model cards, controlled sources, semiconductor compact models,
temperature/noise/AC analysis, topology diagnostics, discontinuity scheduling,
and SPICE-compatible convergence aids remain later phases. Keeping parsing out
of this phase preserves a stable typed model that a future netlist or AMPL-like
front end can target without coupling syntax to numerical assembly.
