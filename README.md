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
├── nc-sparse/            # CSR sparse matrices + SpMV — implemented
├── nc-decomp/            # LU/QR/SVD/eigen (scaffold — Accelerate covers this on the Swift side)
├── nc-iterative/         # Conjugate Gradient implemented; GMRES/BiCGStab/Lanczos not yet
├── nc-optimize/          # LP/MILP/NLP Problem/Solver interface — the AMPL back end; RevisedSimplexSolver implemented, interior-point next
├── nc-ffi/               # UniFFI surface — matmul/dot/axpy/norm2/spmv (f64+f32), plus opt-in FfiVectorF64/F32 zero-copy buffers
├── nc-bench/             # Criterion benchmarks (needs a newer cargo than 1.75 for its deps)
├── scripts/
│   └── build-xcframework.sh   # builds the XCFramework + Swift bindings for release
└── docs/decisions/       # ADRs specific to the Rust side
```

## Building

```bash
cargo test --workspace --exclude nc-bench
```

`nc-bench` (Criterion benchmarks) needs a newer Cargo than what this
was last verified against (1.75) due to its dependency tree; run it
separately once you have a current toolchain: `cargo bench -p nc-bench`.

## Releasing an XCFramework for Swift-NumericCore

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim \
                   aarch64-apple-darwin x86_64-apple-darwin
./scripts/build-xcframework.sh
```

This is a **scaffold** — it has not been run end-to-end (no macOS host
was available while writing it). Treat it as a documented starting
point, not a verified working release process; debug it against a real
build before wiring `.github/workflows/release.yml` to run
automatically on tag push.

See the parent project's full architecture write-up and ADRs in
`Swift-NumericCore`'s `docs/` for the reasoning behind this repo's
module boundaries.
