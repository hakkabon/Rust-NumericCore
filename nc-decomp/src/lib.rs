//! Matrix decompositions.
//!
//! Was a deliberate empty scaffold through v1 — per
//! `docs/decisions/0003-decomposition-scope.md`, LU/QR/SVD are wrapped
//! from Accelerate/LAPACKE on the Swift side (`NumericCoreAccelerate`),
//! not reimplemented here, and this crate was meant to stay empty until
//! a concrete Rust-side need showed up (no Accelerate available in pure
//! Rust).
//!
//! That need arrived: `nc-optimize`'s interior-point solver needs to
//! solve a dense symmetric-positive-definite normal-equations system
//! (`A D⁻¹ Aᵀ dy = rhs`) every Newton iteration, and `nc-optimize` has
//! no access to Accelerate. `cholesky_solve` below is that — a small,
//! dense, pure-Rust Cholesky solve, scoped to exactly the size this one
//! consumer needs (an `m x m` system, `m` = constraint row count), not
//! a general-purpose decomposition library.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionKind {
    Lu,
    Qr,
    Cholesky,
    Svd,
}

/// Solves `A x = b` for symmetric positive-definite `A` via dense
/// Cholesky factorization (`A = L Lᵀ`) and forward/backward
/// substitution.
///
/// `a` is `A`, given as `n` rows of `n` values each (only the lower
/// triangle plus diagonal is read — `A` is assumed symmetric, not
/// checked). Returns `None` if `A` is not positive definite (a
/// non-positive value would appear under a square root during
/// factorization) rather than producing `NaN`s.
///
/// Not tuned for large `n` — `nc-optimize`'s interior-point solver is
/// this function's only consumer today, calling it once per Newton
/// iteration on a matrix sized to the LP's constraint row count, which
/// this whole solve path already scopes to "smallish problems" (see
/// `nc-optimize::simplex`'s module docs for the sibling reasoning on
/// the simplex side).
pub fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }
    debug_assert!(a.iter().all(|row| row.len() == n));
    debug_assert_eq!(b.len(), n);

    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }

    // Backward substitution: Lᵀ x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_known_spd_system() {
        // Same system used in nc-iterative's conjugate_gradient test:
        // [[4, 1], [1, 3]] x = [1, 2] -> x = (1/11, 7/11).
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let x = cholesky_solve(&a, &[1.0, 2.0]).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-9);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn identity_is_a_no_op() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x = cholesky_solve(&a, &[5.0, -3.0]).unwrap();
        assert_eq!(x, vec![5.0, -3.0]);
    }

    #[test]
    fn returns_none_for_non_positive_definite_matrix() {
        // Negative-definite.
        let a = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];
        assert!(cholesky_solve(&a, &[1.0, 1.0]).is_none());
    }

    #[test]
    fn handles_larger_random_looking_spd_system() {
        // A = M^T M + I is guaranteed SPD for any M.
        let m = vec![vec![1.0, 2.0, 0.0], vec![0.0, 1.0, 1.0], vec![2.0, 0.0, 1.0]];
        let mut a = vec![vec![0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += m[k][i] * m[k][j];
                }
                a[i][j] = s + if i == j { 1.0 } else { 0.0 };
            }
        }
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve(&a, &b).unwrap();

        // Verify A x == b directly, rather than trusting a
        // hand-computed x for a 3x3 system.
        for i in 0..3 {
            let lhs: f64 = (0..3).map(|j| a[i][j] * x[j]).sum();
            assert!((lhs - b[i]).abs() < 1e-8, "row {i}: {lhs} != {}", b[i]);
        }
    }

    #[test]
    fn empty_system_returns_empty_solution() {
        assert_eq!(cholesky_solve(&[], &[]), Some(vec![]));
    }
}

