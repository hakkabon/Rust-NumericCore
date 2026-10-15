# Phase 48 — LP/QP/MILP production depth

This phase adds a production orchestration layer rather than duplicating solver
kernels. `solve_production_linear` performs reversible presolve, selects the LP
or MILP backend, optionally scales continuous solves, reconstructs the original
point and objective, and certifies the answer against the untouched model.
Diagnostics cover row feasibility, variable feasibility, integrality, objective
recomputation, and finite values.

`polish_convex_qp` infers the active rows and variable bounds of a converged QP,
solves the resulting equality-constrained KKT system with partial pivoting, and
accepts the candidate only when feasibility and objective safeguards pass.
Singular or dependent active sets safely retain the original point.

Swift exposes the same concepts so native and Rust-backed workflows share the
same presolve/solve/postsolve/certify contract and QP refinement semantics.
