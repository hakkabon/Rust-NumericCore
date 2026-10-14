# Phase 42: nonlinear and transient circuits

Phase 42 extends the modified-nodal-analysis kernel beyond independent linear
networks. Voltage-controlled current and voltage sources provide the dependent
source primitives needed by macromodels. A three-terminal level-1 NMOS uses the
piecewise cutoff, triode, and saturation equations, including channel-length
modulation, with analytic transconductance and output-conductance stamps.

DC analysis now has an explicit continuation path. It begins with independent
sources scaled toward zero and a conductance from every non-reference node to
ground, then advances the source scale while removing that `gmin`. Every stage
warm-starts the next, and the final reported diagnostics are evaluated against
the original, unmodified circuit. The ordinary direct Newton entry point remains
available for easy problems.

Pulse sources expose sorted, deduplicated transition times in either integration
direction. A transient driver can use these breakpoints to end steps exactly at
source corners rather than interpolate across a discontinuity. Circuit residuals
continue to use the shared index-1 DAE integrator and analytic state/derivative
Jacobians, so nonlinear diode and MOS currents participate in Newton correction
at every implicit step.

The level-1 MOS model is intentionally a foundation, not a claim of modern SPICE
compact-model compatibility. Body effect, body diode and charge storage, device
temperature, PN-junction limiting, automatic segmented restart, controlled-source
netlist syntax, BSIM models, AC small-signal analysis, and noise remain future
work.
