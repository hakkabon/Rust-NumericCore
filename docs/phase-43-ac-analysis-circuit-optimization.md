# Phase 43: AC analysis and circuit optimization

Phase 43 adds small-signal frequency-domain analysis without creating a second
circuit assembly system. At a supplied DC operating state, `CircuitModel`
evaluates the existing analytic state and derivative Jacobians as the
conductance and storage matrices, `G` and `C`. For each frequency it solves

`(G + j omega C) x = b`

through the equivalent real block system. This works for RLC networks and also
linearizes diode and level-1 MOS conductances at their nonlinear bias point.
Independent current and voltage sources can be assigned magnitude/phase
phasors, multiple excitations superpose, and every MNA unknown is returned as a
real/imaginary response with magnitude and phase helpers.

Circuit optimization introduces named bounded design variables and weighted AC
magnitude targets. A caller-provided builder maps each parameter vector to a
typed `CircuitModel`; the optimization driver obtains a continued DC operating
point, evaluates the requested AC sweep, constructs the weighted least-squares
objective, estimates design gradients with bound-aware central differences, and
uses the existing L-BFGS-B solver. The result retains solver termination data and
the final response sweep.

The real block solve is currently dense. This is intentional while NumericCore's
complex dtypes remain reserved rather than kernel-backed. Sparse complex
factorization, analytic/adjoint design sensitivities, phase/group-delay targets,
multi-corner and tolerance optimization, AC source declarations in a netlist,
and noise analysis remain future work.
