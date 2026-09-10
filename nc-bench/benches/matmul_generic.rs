//! Baseline benchmark for the pure-Rust fallback matmul.
//!
//! Purpose: give the `DispatchPolicy.gpuThreshold` / Accelerate-vs-Rust
//! decision on the Swift side actual numbers to be tuned against, rather
//! than a guessed constant. Run with `cargo bench -p nc-bench` and record
//! results per-machine (Apple Silicon generation matters) in
//! `docs/design/dispatch-thresholds.md`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nc_kernels_generic::matmul;

fn bench_matmul(c: &mut Criterion) {
    for &n in &[8usize, 64, 256] {
        let a = vec![1.0f64; n * n];
        let b = vec![1.0f64; n * n];
        let mut out = vec![0.0f64; n * n];

        c.bench_function(&format!("matmul_generic_{n}x{n}"), |bencher| {
            bencher.iter(|| {
                matmul(black_box(&a), black_box(&b), black_box(&mut out), n, n, n);
            });
        });
    }
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
