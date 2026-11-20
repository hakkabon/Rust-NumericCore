# Phase 61 — Decomposition-ready solver contracts

Use `LpWorkspace` for repeated solves of a continuous LP with the existing
revised simplex algorithm. Use a separate workspace for each independent
scenario. `LpSubproblemID` carries the caller's name and optional scenario
label through every report.

- `LpUpdate` applies indexed objective, variable-bound and row-bound edits
  as a batch. Change a right-hand side by editing that row's bound. Appended
  rows reference existing columns. Appended columns can also reference rows
  appended in the same batch.
- Numerical edits retain the basis inverse. Structural appends extend the
  basis with a slack for each new row and refactor. Reports include basis
  statuses, warm-start use, pricing iterations, pivots, and the maximum
  basic reduced cost for optimal results. A saved basis can seed another
  compatible workspace.
- The default workspace uses the original, unscaled model. Its backend
  retains the revised simplex solver's dense basis representation.

## Evidence in minimization coordinates

Given row multipliers `y`, the variable reduced costs are `c - Aᵀy`.
Positive coefficients select lower bounds and negative coefficients select
upper bounds. The dual objective is the sum of those bound contributions.
The verifier checks original-model primal feasibility and gap at the
requested tolerance.

A Farkas certificate uses `-Aᵀy` for the variable coefficients instead.
A positive minimum of `yᵀs - yᵀAx` over variable and row-slack bounds
contradicts the equality `s = Ax`. An unboundedness ray requires a feasible
anchor, a descending objective, and non-outward movement at every finite
bound. The computed anchor and recession bounds are checked without an
additive feasibility tolerance.

Evidence is optional and numerically verified. The solver status and the
presence of evidence are separate. A decomposition driver should require
the corresponding evidence before using a bound or deriving a cut.
Presolve's `lift_lp_*` methods verify the lifted candidate on the original
model. Zero extension may fail after bound tightening, in which case solve
the original LP to obtain evidence. These checks use floating-point
arithmetic.

## Coordination and related QPs

`DecompositionCutPool` tracks caller-derived inequalities, origin
(subproblem, iteration, and kind), scope, and inactivity. It keeps stable
IDs and insertion order, and only deduplicates exact inequalities within
the same scope. The caller controls when a cut is inactive and when to
rebuild a master after pruning.

`QuadraticWorkspace` retains the assembled constraints and initial ADMM
factor for a fixed Hessian and constraint matrix. It accepts new linear
costs, objective offset and bounds, reuses converged primal and dual
iterates, and refactors if the adaptive penalty changes during a solve.

The Rust v0.36.0 release adds the provisional `lp-decomposition`
capability and the UniFFI LP workspace and verifier exports. The ordinary
LP, MILP and QP entry points remain available.
