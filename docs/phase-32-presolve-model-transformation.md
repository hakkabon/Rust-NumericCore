# Phase 32: presolve and model transformation

`nc-optimize::presolve_problem` is a solver-independent, reversible sparse
LP/MILP transformation. It performs deterministic row-activity checks,
redundant-row removal, singleton bound propagation with integer rounding, fixed
variable elimination, row-bound shifting, and sparse matrix reconstruction.

The returned `PresolvedProblem` owns the reduced `Problem` when numerical work
remains. It also carries original variable and row mappings, fixed values, an
objective offset, and reduction counts. `restore_point` and
`restore_objective` form the postsolve path. Fully fixed feasible models return
`Solved`; interval contradictions return `Infeasible`, avoiding an unnecessary
solver call.

The API and defaults mirror Swift-NumericCore's `LinearModelPresolver`. This
phase deliberately confines elimination to linear constraints; quadratic
substitution, basis/dual postsolve, stronger coefficient reductions, and
nonlinear graph rewriting are future extensions.
