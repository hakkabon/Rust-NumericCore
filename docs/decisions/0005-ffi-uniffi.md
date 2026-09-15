# 0005 — FFI via UniFFI, not hand-rolled cbindgen

## Status
Accepted.

## Context
Two realistic approaches exist for exposing Rust to Swift in this
project: hand-rolled C ABI (a `#[no_mangle] extern "C"` surface plus
`cbindgen`-generated headers, consumed via a Swift modulemap), or
UniFFI (proc-macro/UDL-driven codegen that produces both the Rust
scaffolding and idiomatic Swift bindings).

The `Layout` project (Sugiyama-style graph layout engine, Rust core with
a SwiftUI sample app) already uses UniFFI successfully for its
Rust-to-Swift FFI, with GitHub Actions automation already built out for
it.

## Decision
`nc-ffi` uses **UniFFI**, matching `Layout`'s established approach,
rather than hand-rolled `cbindgen` (which was the original sketch in
early scoping conversation, before this precedent was accounted for).

Further specifics (see `nc-ffi`'s module docs): use `uniffi::export`
proc macros rather than a separate `.udl` interface file, so the
interface definition lives next to the Rust code instead of in a file
that can drift out of sync; expose buffers as opaque
reference-counted handles rather than raw pointers, letting UniFFI
manage the reference counting instead of `NCBindings` doing manual
memory management; and derive Swift-side errors directly from each
crate's `thiserror` types rather than hand-maintaining a separate
C-style error-code enum.

## Consequences
- One FFI toolchain and one set of operational lessons (build
  configuration, CI wiring, versioning) shared across `Layout` and
  `NumericCore` instead of two to maintain independently.
- UniFFI has more codegen "magic" than a hand-rolled C ABI — debugging
  a binding-generation issue means understanding UniFFI's proc-macro
  output, not just reading a header file. Accepted given the existing
  `Layout` team (i.e. past-you) already has that context.
- Some very low-level control (e.g. exact struct layout for a
  zero-copy shared-memory buffer between Rust and Accelerate-facing
  Swift code) may be harder to express through UniFFI's ownership model
  than through raw pointers. If that becomes a real bottleneck for the
  performance-critical `Matrix`/`Buffer` path specifically, a narrow
  hand-rolled escape hatch alongside UniFFI (not a wholesale switch back)
  is the likely fix — see the `NCBindings` module docs for where that
  would land.

## Update (first real implementation)
`nc-ffi` now has a working, tested slice of this surface:
`matmul_f64`/`dot_f64`/`axpy_f64`/`norm2_f64` (wrapping
`nc-kernels-generic`) and `spmv_f64` (wrapping `nc-sparse::CsrMatrix`),
using `uniffi::setup_scaffolding!()` and `#[uniffi::export]` — the
proc-macro-only workflow, no `.udl` file and no `build.rs` needed.
Confirmed against `uniffi = "0.27"` on `rustc 1.75`. All values cross by
copy (`Vec<f64>`, plain records) rather than by shared reference — see
that file's module docs for why, and what changes when a zero-copy path
is added later. `NCBindings`/`RustFallbackBackend` on the Swift side are
not yet wired to call through this — that rewiring is the next step,
not done as part of proving the Rust-side surface builds and tests
correctly.

## Update (Swift side wired through, f32 added)
`NCBindings`/`RustFallbackBackend` on the Swift side are now wired
through (`Swift-NumericCore` side — see that repo's ADR 0005/0006
updates). Once that rewiring made `Float`'s remaining gap concrete,
`f32` counterparts (`matmul_f32`, `dot_f32`, `axpy_f32`, `norm2_f32`,
`spmv_f32`) were added here — no new numerics, since
`nc-kernels-generic`'s kernels and `nc-sparse::CsrMatrix` were already
generic over the element type; only the FFI-boundary wrapper functions
and `Ffi*F32` record types were new. Confirmed building and passing
(12/12 tests in `nc-ffi`, full workspace green).

## Update (zero-copy vector buffers)
`FfiVectorF64`/`FfiVectorF32` — `uniffi::Object`s (`Arc`-wrapped,
`Mutex`-guarded `Vec`) with `axpyInPlace`/`dot`/`norm2` methods that
never copy data across the FFI boundary between construction and the
final `toVec()`/`toArray()` read-back. Addresses the copy-per-call cost
this file's module docs (and ADR 0006) flagged from the start, but as
**additive, opt-in infrastructure** — the existing `matmul_f64`-style
functions and `NumericCore`'s default `Matrix`/`Vector` storage are
unchanged. Nothing currently calls this from the Swift side by default;
it's there for a future hot-loop consumer (an iterative solver, or a
`Swift-DataLens` inner loop) to opt into once profiling shows a concrete
need, per this project's "don't build ahead of need" principle.
22/22 tests pass in `nc-ffi` including a chained-axpy cross-check
against the plain kernel function.

## Alternatives considered
- **cbindgen + hand-rolled C ABI** (the original sketch). Rejected once
  the `Layout` precedent was identified — no reason to introduce a
  second FFI approach into this developer's toolchain when one already
  works and has tooling built around it.
