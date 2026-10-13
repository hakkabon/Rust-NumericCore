# Phase 31: sparse LP/QP infrastructure

Phase 31 establishes explicit sparse representations beneath linear and
quadratic optimization. `SparseLinearWorkspace` validates the existing CSR LP
model and builds a persistent column-oriented transpose once, then supplies
row products and transpose products in time proportional to stored nonzeros.

`SparseSymmetricMatrix` stores one canonical lower triangle, validates sorted
unique structure, performs symmetric matrix-vector products, and reports its
true storage count. `SparseQuadraticProblem` combines that Hessian with the
existing CSR constraint matrix and supports sparse objective evaluation.

The production QP adapter currently materializes the Hessian only at the
existing dense Cholesky/ADMM backend boundary. This keeps solver behavior,
warm starts, and diagnostics conformant while creating a stable seam for a
future sparse KKT or sparse normal-equation factorization. It is therefore an
infrastructure phase, not a claim that every internal factorization is sparse.
