# 0007 — Independently verify optimization solutions

## Status

Accepted.

## Context

`SolveStatus::Optimal` is a solver's internal verdict. It does not independently
show that the final public values satisfy row bounds, variable bounds,
integrality, or the reported objective. Translation and bookkeeping defects can
therefore survive even when an algorithm terminates normally.

## Decision

`Solution::diagnostics(&Problem)` recomputes those invariants solely from the
public problem and returned values. `SolutionDiagnostics::is_verified` applies a
caller-selected tolerance, and `Solver::solve_with_diagnostics` makes the audit
available uniformly without altering existing solver implementations.

The existing `Solution` and UniFFI records are unchanged. This keeps the v0.6.0
ABI stable; a later FFI release can expose diagnostics after the Rust and Swift
contracts have accumulated real usage.

## Consequences

- Native Rust callers can distinguish solver termination from a verified result.
- Tests can detect feasibility, integrality, and objective regressions through a
  solver-independent path.
- Non-optimal statuses may naturally produce non-verifiable diagnostics; the
  report is evidence, not a replacement status enum.
