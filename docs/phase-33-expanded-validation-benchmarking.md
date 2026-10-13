# Phase 33: expanded validation and benchmarking

The production reference program now has seven deterministic cases: LP scaling,
QP KKT conditions, MILP certification, nonlinear stationarity, reversible
presolve/postsolve, mixed-scale sparse operators, and repeatable perturbations.
They run as correctness tests with scale-aware tolerances and fail closed on
non-finite results.

`nc-bench/benches/optimization_pipeline.rs` adds Criterion baselines for sparse
row activity at 100, 1,000, and 10,000 variables and presolve at 100 and 1,000
variables. Run:

```sh
cargo bench -p nc-bench --bench optimization_pipeline
```

Criterion output is performance evidence, not a CI timing assertion. Compare
baselines only on a controlled machine and toolchain; retain compiler, target,
CPU, and benchmark parameters with published results. The current size split
also makes presolve scaling visible without making routine benchmark runs
impractically long.
