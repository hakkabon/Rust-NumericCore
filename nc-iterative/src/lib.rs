//! Iterative solvers for large sparse linear systems.
//!
//! v1 scope: Conjugate Gradient only (symmetric positive-definite systems).
//! GMRES/BiCGStab (general nonsymmetric) and Lanczos (eigenvalues) are
//! deferred until a concrete need shows up — see module docs in
//! `nc-decomp` for the same "don't build ahead of need" reasoning.

use nc_sparse::CsrMatrix;

/// Result of running an iterative solve.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
}

/// Solve `A x = b` via unpreconditioned Conjugate Gradient.
///
/// `a` must be symmetric positive-definite — this is a precondition, not
/// something this function checks (checking it is as expensive as
/// solving). Passing a non-SPD matrix will not panic but will not
/// converge to a meaningful answer either.
pub fn conjugate_gradient(
    a: &CsrMatrix<f64>,
    b: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Result<SolveResult, nc_sparse::SparseError> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec(); // r = b - A*x0, x0 = 0
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    let b_norm = dot(b, b).sqrt().max(1e-30);

    for iteration in 0..max_iterations {
        let ap = a.spmv(&p)?;
        let alpha = rs_old / dot(&p, &ap).max(1e-300);

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let residual_norm = dot(&r, &r).sqrt();
        if residual_norm / b_norm < tolerance {
            return Ok(SolveResult {
                solution: x,
                iterations: iteration + 1,
                residual_norm,
                converged: true,
            });
        }

        let rs_new = dot(&r, &r);
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    let residual_norm = dot(&r, &r).sqrt();
    Ok(SolveResult { solution: x, iterations: max_iterations, residual_norm, converged: false })
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_simple_spd_system() {
        // A = [[4, 1], [1, 3]], b = [1, 2] -> x = [1/11, 7/11]
        let a = CsrMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![4.0, 1.0, 1.0, 3.0])
            .unwrap();
        let result = conjugate_gradient(&a, &[1.0, 2.0], 100, 1e-10).unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 1.0 / 11.0).abs() < 1e-8);
        assert!((result.solution[1] - 7.0 / 11.0).abs() < 1e-8);
    }
}
