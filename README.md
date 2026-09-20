# Rust-NumericCore

The Rust half of NumericCore — storage, kernels, sparse linear algebra,
iterative solvers, and the AMPL-style optimization interface. Consumed
from Swift via [`Swift-NumericCore`](https://github.com/hakkabon/Swift-NumericCore),
which depends on this repo's tagged releases (an XCFramework built by
`scripts/build-xcframework.sh` / `.github/workflows/release.yml`).

This split mirrors the existing [`Layout`](https://github.com/hakkabon/Layout)
project's structure (Rust core + UniFFI, consumed by a separate Swift
package) — see `docs/decisions/0008-split-into-two-repos.md` in
`Swift-NumericCore` for the reasoning.

**Not a `Swift-Numerics` fork or alternative.** This project is
independent of, and has no relationship to, Apple's
[`apple/swift-numerics`](https://github.com/apple/swift-numerics). The
similar-sounding name is a known, deliberately-avoided collision — see
that same ADR for why `NumericCore` was kept as the name.

## Workspace layout

```
Rust-NumericCore/
├── nc-core/              # strides, shape, dtype — no numerics
├── nc-kernels-generic/   # pure-Rust fallback kernels (matmul, axpy, dot, norm) — implemented
├── nc-kernels-simd/      # SIMD kernels (scaffold; nightly-gated)
├── nc-sparse/            # validated CSR sparse matrices + SpMV / transpose-SpMV — implemented
├── nc-decomp/            # dense Cholesky solve (cholesky_solve), added for nc-optimize's interior-point normal equations
├── nc-iterative/         # CG plus CGLS weighted/penalized sparse statistics; GMRES/BiCGStab/Lanczos deferred
├── nc-optimize/          # LP/MILP/NLP Problem/Solver interface — RevisedSimplexSolver, InteriorPointSolver, BranchAndBoundSolver (MILP) all implemented
├── nc-ffi/               # UniFFI surface — matmul/dot/axpy/norm2/spmv (f64+f32), opt-in FfiVectorF64/F32 zero-copy buffers, solve_lp_simplex/solve_lp_interior_point
├── nc-bench/             # Criterion benchmarks (needs a newer cargo than 1.75 for its deps)
├── scripts/
│   └── build-xcframework.sh   # builds the XCFramework + Swift bindings; used locally and by release.yml
├── .github/workflows/
│   └── release.yml            # on a v*.*.* tag: build, test, publish, dispatch to Swift-NumericCore (ADR 0010)
└── docs/decisions/       # ADRs specific to the Rust side
```

## Building

```bash
cargo test --workspace --exclude nc-bench
```

`nc-bench` (Criterion benchmarks) needs a newer Cargo than what this
was last verified against (1.75) due to its dependency tree; run it
separately once you have a current toolchain: `cargo bench -p nc-bench`.

## Releasing — automated

Pushing a version tag is the whole release process:

```bash
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/release.yml` takes it from there: builds the
XCFramework + UniFFI Swift bindings via `scripts/build-xcframework.sh`,
runs the full Rust test suite, verifies every framework slice has a
`Headers/module.modulemap` (see `build-xcframework.sh`'s comments for
why this specific file matters — its absence causes a silent
`canImport` failure and an undefined-symbols link error on the Swift
side, not a build-time error here), computes the SPM binary-target
checksum, publishes the framework and bindings as release assets, and
notifies `Swift-NumericCore` (via a `repository_dispatch`, authenticated
with the `SWIFT_NUMERICCORE_DISPATCH_TOKEN` repo secret) so its own
`update-ffi.yml` can open a PR pinning `Package.swift` to the new
release. See ADR 0010 for the full pipeline, including what happens if
the dispatch token is missing or expired (degrades gracefully, not a
hard failure).

**No release has been tagged yet as of this writing** — the pipeline
above is fully wired (both workflows exist, both are individually
sound) but has never fired end-to-end as a complete system. `v0.1.0`
will be its first real run; watch both repos' Actions tabs when you
tag it, since a first real run is exactly when an untested interaction
between two otherwise-correct pieces tends to surface.

To build the XCFramework locally without tagging a release (e.g. to
test a change before pushing):

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim \
                   aarch64-apple-darwin x86_64-apple-darwin
./scripts/build-xcframework.sh
```

See the parent project's full architecture write-up and ADRs in
`Swift-NumericCore`'s `docs/` for the reasoning behind this repo's
module boundaries.
