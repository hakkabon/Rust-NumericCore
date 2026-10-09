# Phase 21: feasibility restoration and filter globalization

Phase 21 adds a shared Phase-I solver for graph-represented constrained
nonlinear programs. It minimizes one half of the squared equality and active
inequality violations with projected L-BFGS. An optional positive margin asks
for a strict inequality interior, allowing the nonlinear barrier solver to
recover from boundary or infeasible starts without silently projecting the
model constraints.

SQP can opt into Phase-I restoration and filter globalization. The filter
stores non-dominated `(constraint violation, objective)` pairs and accepts a
trial that makes sufficient progress in feasibility or objective value. Once
the iterate is feasible, the existing merit decrease remains the acceptance
test. This preserves the established solver by default while exposing the new
globalization policy explicitly.

Restoration has its own result and termination contract: already feasible,
converged, iteration limit, or stalled. SQP and nonlinear interior point report
`RestorationFailed` separately from line-search, QP, and infeasible-start
terminations. The complete options and results cross UniFFI, and Swift/Rust
conformance tests exercise all new paths.

This is a first-order restoration method, not a proof that the original
constraints are infeasible. A stalled Phase-I solve means that this local
restoration attempt did not find feasibility. Future strengthening can add
elastic SQP slacks, trust-region restoration, and sparse KKT steps.
