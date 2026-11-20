# Phase 65 — Nonlinear decomposition

`solve_nonlinear_decomposition` minimizes a linear master objective plus
weighted convex recourse costs. The master accepts bounded continuous and
integer variables. Each recourse block has its own bounded continuous variables
and selects the master columns it uses. Expression parameters are the selected
master columns in the given order, followed by the private variables in bounds
order. All weights are positive. To model expected cost, use probabilities
as weights.

The solver checks curvature over the full parameter box. Objectives must be
convex, upper-bounded constraint expressions convex, and lower-bounded
constraint expressions concave. Equality constraints therefore need affine
or constant expressions on the box. Graph values and derivatives must be
finite where evaluated. Domain and curvature checks are conservative.

Each block builds a persistent LP outer approximation from nonlinear tangents.
For a proposed master point, equality rows fix the shared parameters. An
audited LP dual yields an affine bound on recourse cost, with the multipliers
of the fixing rows giving the slope in master coordinates. An audited LP
Farkas certificate projects into a feasibility cut by the same construction.
These cuts remain valid over the master box because the block tangents were
constructed from convex or concave sides throughout that box. The master
keeps one epigraph per block and adds cuts to its audited branch-and-bound
tree. The incumbent objective and feasibility are evaluated against the
original nonlinear expressions.

`Converged` means the audited master lower bound and a feasible incumbent
close the requested gap within numerical tolerances. Master limits and
unresolved recourse can return an existing incumbent or lower bound when
available. `max_block_solves` applies to one block evaluation at one master
candidate. Subproblems run serially, and the master generates cuts at integer
candidates when integrality is present.

The first phase covers differentiable, convex, factorable recourse with
continuous private variables. Nonconvex recourse needs spatial relaxations
or logic-based evidence. Mixed-integer private variables, parallel workers,
and multistage nonlinear policy decomposition are future extensions.
