# Phase 54 — SPICE netlist integration

Phase 54 adds a SPICE-compatible textual front end to the existing typed
`CircuitModel`. The parser is a compiler boundary: circuit syntax, engineering
suffixes, model cards, and analysis directives are resolved before the native
MNA, DAE, and AC solvers run. SPICE syntax is not embedded in numerical kernels,
and netlists are not routed through AMPL because topology and dynamic device
stamps carry semantics a generic optimization grammar would obscure.

The supported vertical subset includes case-insensitive named nodes, `0`/`gnd`,
comments and continuation lines; engineering values from tera through femto;
`.PARAM`, `.IC`, `.MODEL`, `.OP`, `.TRAN`, `.AC`, and `.END`; R/C/L, independent
voltage and current sources, VCCS/VCVS, diodes, and source-body-tied level-1
NMOS devices; `DC`, `AC`, `SIN`, and `PULSE` source specifications; and linear,
decade, and octave AC grids.

`ParsedSpiceNetlist` retains the compiled circuit, analysis requests, AC source
phasors, title, and source-located warnings. Syntax and semantic errors retain
line numbers. Unknown element types and lossy device semantics fail explicitly;
unsupported dot directives become warnings for caller policy.

The standard validation corpus parses an RC low-pass netlist and executes its
DC and AC analyses, checking its solution, grid, and residuals. Focused tests
also cover parameters, diode models, initial conditions, pulse breakpoints, and
source-located rejection.

This is an engineering subset, not a full-SPICE claim. Subcircuits,
includes/libraries, behavioral sources, mutual inductors, transmission lines,
temperature models, BSIM devices, noise, sensitivity syntax, and vendor
dialects remain future work. They should be added only when the typed circuit
representation and numerical kernel can preserve and validate their semantics.
