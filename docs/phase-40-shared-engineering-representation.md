# Phase 40: shared engineering representation

Phase 40 defines a domain-neutral handoff between engineering front ends and
the numerical solvers. It is deliberately neither a SPICE netlist nor a finite
element mesh. Those layers can compile into the same component model.

An `EngineeringModel` contains typed/scaled unknowns, unit-bearing parameters,
and named elements. Each element declares oriented ports and one or more
contributions to global residual rows. Port orientation produces a signed
element-by-unknown incidence matrix for topology consumers.

Constitutive laws use the existing portable `NonlinearExpression` graph. The
input layout for an element with `n` ports and `p` model parameters is:

1. time;
2. `n` local state values in port order;
3. `n` local derivatives in port order;
4. `p` global parameter values.

Reverse-mode gradients are scattered into CSR `F_y`, `F_yDot`, and `F_p`
matrices. Contributions from several elements, or repeated ports, are summed
during canonical COO-to-CSR assembly. The residual and first two Jacobian
blocks convert directly to Phase 39's `DaeEvaluation`.

This design supports modified nodal analysis, branch-current unknowns,
Lagrange multipliers, and element-level FEM residuals while leaving reference
node elimination, mesh generation, material libraries, and domain-specific
parsing to later layers.
