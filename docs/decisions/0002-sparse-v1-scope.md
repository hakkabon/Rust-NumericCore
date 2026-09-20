# 0002 — Sparse v1 scope: CSR only

## Status
Accepted.

## Context
Sparse matrices have several standard formats, each suited to different
operations:
- **COO** (coordinate list) — trivial to build incrementally, poor for
  arithmetic.
- **CSR** (compressed sparse row) — efficient SpMV, natural for
  row-oriented iterative solvers (CG, GMRES).
- **CSC** (compressed sparse column) — needed by some factorization
  algorithms (e.g. certain sparse LU/Cholesky implementations) and for
  column-oriented access.

Three consumers exist or are planned for sparse matrices in this
project: `NumericCoreSparse` (general use), `NumericCoreGraph`
(adjacency/Laplacian, naturally row-oriented for most graph algorithms),
and the AMPL presolve layer (the constraint matrix `A` in `Ax {≤,=,≥} b`,
consumed by iterative or simplex-style solvers that are also
row-oriented).

## Decision
v1 implements **CSR only**, in both `nc-sparse` (Rust) and
`NumericCoreSparse` (Swift, currently a parallel pure-Swift
implementation pending FFI). SpMV is the initial operation.

COO is deferred until the AMPL presolve layer actually needs
incremental construction (at which point a `COO → CSR` conversion
function is the likely addition, not a full parallel COO type with its
own operation set). CSC is deferred until a specific sparse
factorization needs it.

## Consequences
- All three planned consumers (general sparse, graph, AMPL) share one
  format and one code path — less to maintain, easier to keep correct.
- Incrementally building a large sparse matrix (e.g. while parsing an
  AMPL model's constraint list one row at a time) is currently awkward
  with CSR's fixed row-pointer structure — building via
  `Vec<Vec<(col, value)>>` and converting once at the end (as
  `NumericCoreGraph::GraphBridge.adjacencyMatrix` already does) is the
  workaround until COO exists.
- Any future factorization needing CSC will require a real (not
  trivial) addition, not a quick wrapper.

## Update — transpose products for portable sparse statistics

`nc-iterative` now has a concrete statistical consumer: CGLS weighted and
penalized least squares requires both `A·x` and `Aᵀ·r`. CSR remains the right
format: `transpose_spmv` accumulates directly over CSR entries without
materializing a transpose or adding a competing CSC representation. This is
an O(nnz) adjoint product and is sufficient for matrix-free sparse design and
penalty operators.

The CSR constructor now validates the whole row-pointer invariant (starts at
zero, monotonic, bounded by nnz, ends at nnz), not just length. That makes the
new statistical iteration safe against malformed sparse input before either
forward or transpose traversal can index incorrectly. CSC and sparse direct
factorizations remain deferred; the update adds only the adjoint operation a
measured least-squares consumer requires.

## Alternatives considered
- **Implement COO, CSR, and CSC upfront.** Rejected: no current consumer
  needs COO's incremental-build property enough to justify it yet, and
  no current consumer needs CSC at all. Building all three now is
  exactly the kind of speculative generality this project's design
  principles try to avoid.
