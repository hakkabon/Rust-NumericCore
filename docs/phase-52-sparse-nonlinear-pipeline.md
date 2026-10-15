# Phase 52 — Sparse nonlinear solver pipeline

Phase 52 replaces the former one-shot sparse-input KKT solve with a reusable
symmetric-indefinite pipeline. `SparseKktWorkspace` performs deterministic
symbolic ordering once, retains that analysis while Hessian and Jacobian values
change, and performs numerical LDLᵀ refactorization plus iterative refinement
for each nonlinear step.

The result reports residuals, factor nonzeros, symbolic-analysis and numeric-
factorization counts, solve count, symbolic reuse, and KKT inertia. Structural
changes fail explicitly instead of silently invalidating the stored ordering.
`solve_sparse_kkt` remains as the compatible one-shot entry point and now uses
the same workspace internally.

`ConstrainedNonlinearProblem::linearize_sparse_kkt` connects the common
nonlinear expression graph to this pipeline. It evaluates the exact Lagrangian
Hessian and reverse-mode sparse constraint Jacobian directly into the KKT
contract, providing the shared linear-system substrate for SQP, interior-point,
sensitivity, circuit, and FEM iterations.

The standard reproducibility corpus now performs two numerical factorizations
with one symbolic analysis and verifies the refined residual and saddle-point
inertia. The current sparse LDLᵀ implementation remains an in-tree direct
solver; parallel multifrontal factorization and matrix-free preconditioned KKT
fallbacks remain future work.
