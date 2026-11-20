# Phase 66 — Production sparse solver expansion

`SparseCholeskyFactorization` adds reusable sparse LLᵀ for symmetric positive
definite matrices. It uses the same symbolic analysis as pivoted LU and LDLᵀ.
The Cholesky update traverses active pivot neighbors, and minimum-degree
ordering updates an active degree index as the graph changes. LDLᵀ Schur
updates also use pivot neighbors, and the transpose triangular sweep traverses
stored lower entries. The LDLᵀ pivot search still scans active rows. It scales
2×2 blocks before forming determinants, solving, and evaluating inertia.

LU solves Aᵀx = b using the existing numeric row pivots and symbolic ordering.
Refinement measures the residual and normwise backward error for the requested
orientation against the original matrix. Corrections are accepted while
backward error improves. Reported diagnostics include factor entries, estimated
fill, minimum pivot, growth, and pivot interchanges. LDLᵀ inertia describes the
computed factor at its pivot threshold. With nonzero dropping, use backward
error to assess the solve against the original matrix.

UniFFI exposes a symbolic plan and a persistent numeric workspace for LU,
Cholesky, and LDLᵀ. A plan can create multiple independent factors with the
same CSR pattern. Refactor with a full same-pattern CSR matrix or send values
alone in original CSR storage order. Failed refactors preserve the previous
factor. Read locks keep an entire right-hand-side batch on a single numerical
factor while writers replace factors atomically.

The native tests exercise fill on a sparse star, symbolic and numerical
permutations in a transpose solve, extreme-scale indefinite pivots, and
transactional refactors. This release uses NumericCore's native sparse kernels.
