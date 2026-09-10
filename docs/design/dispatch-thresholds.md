# Dispatch thresholds — working notes

`DispatchPolicy.gpuThreshold` (currently a placeholder value of
`250_000` elements) needs to be replaced with a number derived from real
benchmarking, per Apple Silicon generation, once `MPSBackend` has a real
`matmul` implementation to benchmark against `AccelerateBackend`.

## How to fill this in
1. Implement `MPSBackend.matmul`.
2. Run `cargo bench -p nc-bench` for the pure-Rust fallback baseline
   (already has a `matmul_generic` benchmark at 8x8/64x64/256x256).
3. Add an equivalent Swift-side benchmark (XCTest `measure` blocks, or a
   small standalone benchmark harness) comparing `AccelerateBackend` vs
   `MPSBackend` across a range of square matrix sizes, on at least one
   Apple Silicon generation.
4. Record the crossover point(s) below, per machine/chip tested.
5. Update `DispatchPolicy.gpuThreshold`'s default and this file together.

## Results

_(none yet — fill in once step 3 above has been run)_

| Chip | Crossover (elements) | Notes |
|------|----------------------|-------|
| —    | —                    | —     |
