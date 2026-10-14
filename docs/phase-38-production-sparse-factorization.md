# Phase 38: production sparse factorization

`nc-sparse` now separates structure from values through
`SparseSymbolicAnalysis`. Natural and deterministic minimum-degree orderings
produce explicit forward and inverse permutations plus a structural fill
estimate. The analysis validates that a numerical refactorization retains the
same CSR pattern.

`SparseLuFactorization` provides partial row pivoting after symmetric fill
ordering. `SparseLdlt` handles symmetric-indefinite matrices using complete
1-by-1 or 2-by-2 pivot selection and reports inertia, making zero-diagonal KKT
blocks first-class rather than treating them as failed Cholesky systems.

Both factors are reusable across right-hand sides and numerical updates. They
offer batched solves, iterative refinement, normwise backward error, and
diagnostics for actual and estimated fill, minimum pivot, pivot growth, and
interchanges. Refactorization commits only after a complete replacement factor
has succeeded.

The types remain in the Rust domain layer for this phase. Persistent UniFFI
factor handles are intentionally deferred so the published framework and
generated Swift source continue to move together in one binding-update change.
