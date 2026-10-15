# Phase 51 — Benchmark corpus and reproducibility

Phase 51 turns the numerical-validation cases into a versioned benchmark
corpus. Every case has a stable identifier, solver family, execution tier,
metric manifest, and optional external provenance. Identifiers, rather than
display names or array positions, are the durable keys used when comparing
runs.

The tiers are cumulative:

- `smoke` contains the four fast LP, QP, MILP, and nonlinear checks.
- `standard` adds transformations, mixed-scale sparse operations,
  reproducibility, NIST StRD Misra1a, and CUTEst HS71.
- `extended` is the expansion point for larger licensed corpora and scheduled
  validation; it currently includes the complete standard suite.

`ValidationSnapshot` sorts cases by identifier and deliberately records no
clock time, host name, or timestamp. Its JSON is therefore suitable as a CI
artifact or checked-in baseline. `regressions_against` detects removed cases,
removed metrics, pass-to-fail transitions, and normalized-error growth beyond
a caller-selected allowance. Performance timing remains in Criterion because
wall-clock measurements are not reproducible correctness evidence.

Run the report producer with:

```bash
cargo run -p nc-optimize --example numerical_validation -- standard
```

Pull-request CI uploads the resulting standard-tier JSON. Corpus data remains
embedded and offline; adding a case requires a stable ID, an explicit tier and
family, tolerances, and provenance for external material.
