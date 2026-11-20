use nc_sparse::{
    CooMatrix, CsrMatrix, SparseCholeskyFactorization, SparseDirectError, SparseInertia,
    SparseLdlt, SparseLuFactorization, SparseNumericOptions, SparseOrdering,
    SparseSymbolicAnalysis,
};

fn matrix(n: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for &(r, c, v) in entries {
        coo.push(r, c, v).unwrap();
    }
    coo.to_csr().unwrap()
}

fn close(x: &[f64], expected: &[f64]) {
    assert_eq!(x.len(), expected.len());
    for (&actual, &want) in x.iter().zip(expected) {
        assert!((actual - want).abs() < 1e-10, "{actual} != {want}");
    }
}

#[test]
fn sparse_cholesky_reuses_low_fill_star_pattern_and_rolls_back_failed_refactor() {
    let n = 256;
    let star = |center: f64, leaf: f64| {
        let mut entries = vec![(0, 0, center)];
        for i in 1..n {
            entries.extend([(0, i, 1.0), (i, 0, 1.0), (i, i, leaf)]);
        }
        matrix(n, &entries)
    };
    let a = star(n as f64, 4.0);
    let plan = SparseSymbolicAnalysis::analyze(&a, SparseOrdering::MinimumDegree).unwrap();
    assert_eq!(&plan.permutation()[..3], &[1, 2, 3]);
    assert_eq!(plan.estimated_factor_nonzeros(), 3 * n - 2);
    let mut factor = SparseCholeskyFactorization::factor_with_analysis(
        &a,
        plan,
        SparseNumericOptions::default(),
    )
    .unwrap();
    assert_eq!(factor.diagnostics().factor_nonzeros, 2 * n - 1);
    let x = vec![1.0; n];
    close(&factor.solve(&a.spmv(&x).unwrap()).unwrap(), &x);

    let updated = star(n as f64 + 5.0, 7.0);
    factor
        .refactor_values(updated.iter_entries().map(|(_, _, v)| v).collect())
        .unwrap();
    let rhs = updated.spmv(&x).unwrap();
    let result = factor.solve_refined(&rhs).unwrap();
    close(&result.solution, &x);
    assert!(result.backward_error < 1e-13);
    assert!(matches!(
        factor.refactor(&star(-1.0, 7.0)),
        Err(SparseDirectError::NotPositiveDefinite { .. })
    ));
    assert_eq!(factor.solve_refined(&rhs).unwrap(), result);
}

#[test]
fn lu_transpose_undoes_both_symbolic_order_and_numeric_row_pivots() {
    let a = matrix(
        4,
        &[
            (0, 0, 6.0),
            (0, 1, 1.0),
            (0, 2, 2.0),
            (0, 3, 3.0),
            (1, 0, 4.0),
            (2, 0, -1.0),
            (2, 2, 5.0),
            (3, 0, 2.0),
            (3, 3, 7.0),
        ],
    );
    let factor = SparseLuFactorization::factor(
        &a,
        SparseOrdering::MinimumDegree,
        SparseNumericOptions::default(),
    )
    .unwrap();
    assert_ne!(factor.symbolic_analysis().permutation(), &[0, 1, 2, 3]);
    assert!(factor.diagnostics().row_interchanges > 0);
    let x = [1.0, -2.0, 3.0, 0.5];
    close(&factor.solve(&a.spmv(&x).unwrap()).unwrap(), &x);
    let transpose_rhs = a.transpose_spmv(&x).unwrap();
    let result = factor.solve_transpose_refined(&transpose_rhs).unwrap();
    close(&result.solution, &x);
    assert!(result.relative_residual < 1e-13 && result.backward_error < 1e-13);
}

#[test]
fn indefinite_pivot_scales_large_two_by_two_blocks() {
    let a = matrix(2, &[(0, 1, 1e200), (1, 0, 1e200)]);
    let factor = SparseLdlt::factor(&a, SparseOrdering::Natural, Default::default()).unwrap();
    assert_eq!(
        factor.inertia(),
        SparseInertia {
            positive: 1,
            negative: 1,
            zero: 0
        }
    );
    let result = factor.solve_refined(&[1e200, -2e200]).unwrap();
    close(&result.solution, &[-2.0, 1.0]);
    assert_eq!(result.relative_residual, 0.0);
}
