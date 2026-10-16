# Phase 60 — API stabilization and 1.0 preparation

Phase 60 introduces an explicit compatibility boundary without prematurely
freezing every research-oriented solver. `nc_core::CAPABILITIES` is the shared,
machine-readable contract for capability discovery. Its identifiers, stability
vocabulary, schema version, and ordering match Swift-NumericCore.

## Stability policy

- **Stable** APIs are intended to remain source compatible throughout 1.x.
  Incompatible changes require a major release. Additive options need defaults,
  result fields retain their meaning, and termination states are not silently
  reinterpreted.
- **Provisional** APIs are supported and tested but may receive focused naming
  or representation changes before 1.0. Changes require release notes and a
  migration path whenever practical.
- **Experimental** APIs may evolve as algorithms mature. They carry no 1.x
  source-compatibility promise until promoted.

Stability concerns the API contract, not numerical correctness. Every tier is
subject to validation, deterministic reporting, finite-input checks, and
explicit termination behavior.

## Changes and release discipline

The workspace package version now follows the actual Rust release line instead
of the original `0.1.0` scaffold. All public crates therefore report one
coherent version. The obsolete `StubSolver` remains source compatible for this
release but is deprecated in favor of the production LP solvers.

The 1.0 gates are manifest parity with Swift, documentation and conformance
tests for stable APIs, atomic UniFFI release pins, independent numerical
diagnostics, and a passing standard validation corpus on supported platforms.
The compatibility epoch is `1` and is deliberately independent of the package
version so it can be tested before the stable release.
