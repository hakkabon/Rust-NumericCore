# Phase 26: numerical validation program

Phase 26 adds solver-independent validation primitives to `nc-optimize`: mixed
absolute/relative tolerances, named metrics, aggregate reports, normwise linear
system backward error, and deterministic seeded perturbations. Non-finite or
malformed observations fail closed.

`nc_optimize::run_reference_validation` provides a deterministic smoke corpus
covering LP reference values and scaling invariance, convex QP KKT diagnostics,
an enumerated MILP optimum and certified gap, and nonlinear stationarity. The
report retains every observed value, reference, allowed error and normalized
error for inspection by CI or downstream applications.

This is the foundation of a growing validation corpus, not a performance
benchmark or a claim of coverage equivalent to Netlib, Mittelmann, CUTEst, or
MIPLIB. Pull-request CI runs the entire Rust workspace to keep the evidence live.
