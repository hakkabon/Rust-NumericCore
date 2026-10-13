//! Opt-in throughput baselines for sparse model operations and presolve.
//! These measurements are diagnostic evidence, never CI pass/fail gates.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nc_optimize::{presolve_problem, Bound, PresolveOptions, Problem, SparseLinearWorkspace};
use nc_sparse::CsrMatrix;

fn banded_problem(size: usize) -> Problem {
    let mut row_ptr = Vec::with_capacity(size + 1);
    let mut columns = Vec::with_capacity(2 * size);
    let mut values = Vec::with_capacity(2 * size);
    row_ptr.push(0);
    for row in 0..size {
        columns.push(row);
        values.push(2.0);
        if row + 1 < size {
            columns.push(row + 1);
            values.push(-1.0);
        }
        row_ptr.push(values.len());
    }
    Problem {
        objective: vec![1.0; size],
        constraints: CsrMatrix::new(size, size, row_ptr, columns, values).unwrap(),
        row_bounds: vec![
            Bound {
                lower: None,
                upper: Some(3.0)
            };
            size
        ],
        var_bounds: vec![
            Bound {
                lower: Some(0.0),
                upper: Some(1.0)
            };
            size
        ],
        is_integer: vec![false; size],
    }
}

fn bench_optimization_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_pipeline");
    for &size in &[100usize, 1_000, 10_000] {
        let problem = banded_problem(size);
        let workspace = SparseLinearWorkspace::new(problem.clone()).unwrap();
        let point = vec![0.5; size];
        group.bench_with_input(
            BenchmarkId::new("sparse_row_activity", size),
            &size,
            |b, _| b.iter(|| workspace.row_activity(black_box(&point)).unwrap()),
        );
        if size <= 1_000 {
            group.bench_with_input(BenchmarkId::new("presolve", size), &size, |b, _| {
                b.iter(|| {
                    presolve_problem(black_box(&problem), PresolveOptions::default()).unwrap()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_optimization_pipeline);
criterion_main!(benches);
