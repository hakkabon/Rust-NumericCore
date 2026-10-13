# Rust-NumericCore

The Rust half of NumericCore — storage, kernels, sparse linear algebra,
iterative solvers, and the solver half of the AMPL-style linear and nonlinear
optimization interface. Consumed
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
├── nc-sparse/            # CSR operations; sparse LU/Cholesky; ILU(0)/IC(0) factors
├── nc-decomp/            # portable dense Cholesky and Householder-QR least-squares solves
├── nc-iterative/         # CG/BiCGSTAB/GMRES with Jacobi, ILU(0), or IC(0); CGLS statistics
├── nc-optimize/          # LP/MILP/MINLP/QP; restoration, exact derivatives, sparse KKT systems
├── nc-ffi/               # UniFFI surface — kernels, solvers, nonlinear models and derivative products
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

The release pipeline is in active use. Swift-NumericCore tracks completion
with its root `.ffi-version` file rather than inferring synchronization from
the version already present in `Package.swift`; this prevents a pre-pinned
manifest from suppressing the generated-binding update PR.

To build the XCFramework locally without tagging a release (e.g. to
test a change before pushing):

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim \
                   aarch64-apple-darwin x86_64-apple-darwin
./scripts/build-xcframework.sh
```

Phase 23 also provides matrix-free nonlinear least squares using damped
Gauss–Newton systems and `Jv`/`Jᵀv` Krylov products.
Phase 24 strengthens local MINLP search with rounded-and-polished incumbents,
validated warm incumbents, and best-local-bound node ordering.
Phase 25 deepens LP/QP/MILP production behavior with periodic simplex basis
reconstruction, residual-balanced ADMM penalties, and validated MILP rounding
incumbents with search metrics.
Phase 26 adds structured numerical evidence: scale-aware metrics, independent
backward-error checks, deterministic perturbations, and an audited
LP/QP/MILP/nonlinear reference program run in pull-request CI.
Phase 27 adds conservative interval/convexity analysis and affine graph
relaxations. Phase 28 solves those relaxations inside spatial branch-and-bound,
combining LP lower bounds with local NLP incumbents and certified global gaps
for bounded factorable nonlinear programs. Phase 29 adds forward/backward
nonlinear propagation, node infeasibility proofs, and incumbent-cutoff bound
tightening before each relaxation.
Phase 30 combines the same propagation and graph relaxations with joint
integer/spatial branching and integer-fixed nonlinear polishing for certified
global MINLP search.
Phase 31 adds reusable CSR/transpose LP workspaces and canonical symmetric
sparse Hessians, plus a validated sparse-QP adapter to the production solver.
Phase 32 adds reversible sparse LP/MILP presolve: activity-based infeasibility,
redundant-row removal, singleton bound propagation, fixed-variable elimination,
and primal/objective postsolve reconstruction.
Phase 33 expands deterministic validation with postsolve, mixed-scale sparse,
and perturbation cases, plus opt-in Criterion benchmarks for sparse model
operations (`cargo bench -p nc-bench --bench optimization_pipeline`).
Phase 34 strengthens production MILP search with binary knapsack cover cuts,
iterative general-row bound propagation, and fixed-integer LP repair of rounded
incumbent candidates.
Phase 35 adds matrix-free Steihaug trust-region Newton-CG and Powell's
derivative-free direction-set method as complementary advanced nonlinear paths.

See the parent project's full architecture write-up and ADRs in
`Swift-NumericCore`'s `docs/` for the reasoning behind this repo's
module boundaries.
