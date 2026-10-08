# Phase 16: sparse direct solvers and preconditioners

`nc-sparse` now provides reusable `SparseLu` and `SparseCholesky`
factorizations. LU uses partial row pivoting and dynamic sparse fill;
Cholesky validates symmetry and positive pivots. Their solve reports recompute
the residual against the original CSR matrix and expose factor nonzero counts.

The same crate provides zero-fill `Ilu0` and `IncompleteCholesky` factors.
`nc-iterative` constructs a selected factor once and applies it throughout CG,
BiCGSTAB, or restarted GMRES. CG rejects ILU(0) because a nonsymmetric
preconditioner violates its SPD contract. Factorization failures remain typed
errors rather than being converted into an unconverged iteration.

`nc-ffi` exports the direct solves, the two new preconditioner choices, and all
iterative methods through UniFFI. This lets Swift's `NumericCoreSparse` expose
one coherent CSR solver surface with matching termination and residual
diagnostics.

The implementation is portable and intentionally dependency-free. It is not a
replacement for a future SuiteSparse-class backend: fill-reducing ordering,
symbolic/numeric phase separation across FFI, supernodal kernels, multiple
right-hand-side batching, and advanced pivot strategies remain future work.
