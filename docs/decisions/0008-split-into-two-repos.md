# 0008 — Split into `Swift-NumericCore` / `Rust-NumericCore`

## Status
Accepted.

## Context
The original scaffold kept the Rust workspace and Swift package as two
directories (`rust/`, `swift/`) inside one repository. That was fine
while `NumericCore` had exactly one consumer (itself). A second real
consumer showed up — `Swift-DataLens` (a Swift/Rust LOESS/LOCFIT local
regression implementation) — which needs to depend on the Swift API as
a normal SPM package dependency, without pulling in the Rust workspace,
build scripts, or ADRs that are irrelevant to a package consumer.

The `Layout` project already established a working pattern for exactly
this shape: a Rust core repo, consumed by a separate Swift package repo
via a compiled artifact, rather than one monorepo.

## Decision
Split into two repositories, mirroring `Layout`:
- **`Rust-NumericCore`** — the Cargo workspace (`nc-core` through
  `nc-ffi`/`nc-bench`), plus the XCFramework build/release tooling
  (`scripts/build-xcframework.sh`, `.github/workflows/release.yml`).
- **`Swift-NumericCore`** — the Swift package, consumable via
  `.package(url: "https://github.com/hakkabon/Swift-NumericCore", ...)`
  by any client (`Swift-DataLens` included) without any Rust toolchain
  or Cargo workspace in sight.

**Naming**: kept as `Swift-NumericCore` / `Rust-NumericCore`, explicitly
*not* `Swift-Numerics`/`Rust-Numerics`, despite that naming being raised
as an option. `apple/swift-numerics` is an established, widely-used
package (Complex numbers, real-number protocols); a same-named repo
under a different account would create exactly the kind of
package-resolution and documentation confusion this project's own early
`Swift-DataLens` action items already stumbled into once (referring to
"Swift-Numerics" when meaning Apple's package). `NumericCore` is
distinct enough that no one — including future-you skimming a
`Package.swift` a year from now — has to stop and disambiguate.

## Consequences
- `Swift-NumericCore`'s `Package.swift` currently has **no dependency on
  `Rust-NumericCore` at all** — `NCBindings` is a plain empty-ish source
  target, and `RustFallbackBackend` reimplements the Rust logic directly
  in Swift (ADR 0006's accepted debt). This is deliberate: it keeps
  `Swift-NumericCore` buildable as a standalone SPM dependency *today*,
  which is exactly what `Swift-DataLens` needs right now. Wiring a real
  `binaryTarget` against a `Rust-NumericCore` release is a separate,
  later step — see the commented-out scaffold directly in
  `Package.swift` for the intended shape when that happens.
- `Rust-NumericCore` needs its own versioning discipline from day one:
  tag releases (`v0.1.0`, ...) as soon as the XCFramework release
  process is real, so `Swift-NumericCore` can eventually pin a
  compatible range rather than tracking a moving branch.
- Two repos means two READMEs, two sets of ADRs (Rust-side concerns —
  e.g. which Apple platform targets to build for — live in
  `Rust-NumericCore/docs/decisions/`; Swift-API-shape concerns stay
  here). Cross-reference rather than duplicate: this ADR is the one
  place that explains the split itself.

## Alternatives considered
- **Keep the monorepo, use SPM's ability to reference a subdirectory.**
  Rejected: still forces any consumer's dependency graph to include the
  Cargo workspace, build scripts, and Rust-specific tooling/CI even
  though they never build or touch it — exactly the friction `Layout`'s
  split already avoids.
- **`Swift-Numerics`/`Rust-Numerics` naming.** Rejected — see the naming
  discussion above.
