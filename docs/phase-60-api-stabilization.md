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

## Maintenance interval

The numerical validation workflow runs weekly and on demand. It uploads
the standard validation snapshot and the Rust and Cargo toolchain versions.
Use the case ID, observed and reference values, tolerance, and toolchain
when recording a numerical regression. Review dependency and toolchain
updates separately so their numerical effects can be assessed.

The `nc-core` compatibility tests guard the established stable capability
IDs, tiers, and contract versions. Coordinate catalog changes with Swift.
Promote provisional capabilities after reviewing documentation and
conformance evidence. Keep maintenance releases focused on corrections
and compatible diagnostics.

Retain the original XCFramework archive and generated Swift binding for
each Rust tag pinned by a supported Swift release. Preserve both files
and the archive checksum in release backups. Swift's stewardship workflow
verifies the published bytes, package pin, and runtime identity. Restore
the original bytes if an asset is lost, or publish a new release and
update the Swift pin.

To collect current evidence locally:

```sh
cargo test --workspace
cargo run -p nc-optimize --example numerical_validation -- standard
```
