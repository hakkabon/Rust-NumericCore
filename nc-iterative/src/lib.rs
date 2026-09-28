//! Iterative solvers for large sparse linear systems.
//!
//! v1 scope: Conjugate Gradient only (symmetric positive-definite systems).
//! GMRES/BiCGStab (general nonsymmetric) and Lanczos (eigenvalues) are
//! deferred until a concrete need shows up — see module docs in
//! `nc-decomp` for the same "don't build ahead of need" reasoning.

use nc_sparse::CsrMatrix;
use thiserror::Error;

/// Errors for portable sparse statistical iterations.
#[derive(Debug, Error)]
pub enum StatisticalSolveError {
    #[error("design is {rows}x{cols}, but response and weights must each have {rows} entries")]
    ObservationLength { rows: usize, cols: usize },

    #[error("penalty has {actual} columns, but design has {expected} coefficients")]
    PenaltyWidth { actual: usize, expected: usize },

    #[error("penalty weight must be finite and strictly positive")]
    InvalidPenaltyWeight,

    #[error("max_iterations must be positive and tolerance must be finite and positive")]
    InvalidConvergenceSettings,

    #[error("initial solution has {actual} entries, but design has {expected} coefficients")]
    InitialSolutionLength { actual: usize, expected: usize },

    #[error("response, weights, design, and penalty entries must be finite; weights must be non-negative")]
    InvalidNumericInput,

    #[error(transparent)]
    Sparse(#[from] nc_sparse::SparseError),
}

/// Result of sparse weighted or penalized least-squares iteration.
///
/// `converged` means the relative normal residual of the augmented
/// least-squares system met the requested tolerance. Consumers must not treat
/// an unconverged result as a fitted statistical model.
#[derive(Debug, Clone)]
pub struct StatisticalSolveResult {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
    pub weighted_residual_sum_of_squares: f64,
    pub penalty_contribution: f64,
}

/// Diagonal scaling applied to the augmented normal operator during CGLS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatisticalPreconditioner {
    /// Preserve the original, unpreconditioned CGLS iteration.
    None,
    /// Use `diag(XᵀWX + λPᵀP)` without materializing either product.
    #[default]
    Jacobi,
}

/// Controls a sparse statistical solve, including the state reused by IRLS.
#[derive(Debug, Clone)]
pub struct StatisticalSolveOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub initial_solution: Option<Vec<f64>>,
    pub preconditioner: StatisticalPreconditioner,
}

impl Default for StatisticalSolveOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1e-8,
            initial_solution: None,
            preconditioner: StatisticalPreconditioner::Jacobi,
        }
    }
}

impl StatisticalSolveResult {
    pub fn objective(&self) -> f64 {
        self.weighted_residual_sum_of_squares + self.penalty_contribution
    }
}

/// Result of running an iterative solve.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
}

/// Solve `min Σᵢ wᵢ(yᵢ − xᵢᵀβ)²` using CGLS over a CSR design.
///
/// The method uses only `A * v` and `Aᵀ * u`; it never materializes a
/// dense normal-equations matrix. Positive weights scale observation rows by
/// `√wᵢ`; zero weights exclude the row. For a rank-deficient unpenalized
/// design, CGLS starts at zero and returns the minimum-norm Krylov iterate,
/// but callers should require `converged` before consuming the result.
pub fn weighted_least_squares(
    design: &CsrMatrix<f64>,
    response: &[f64],
    weights: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Result<StatisticalSolveResult, StatisticalSolveError> {
    weighted_least_squares_with_options(design, response, weights, StatisticalSolveOptions {
        max_iterations, tolerance, initial_solution: None,
        preconditioner: StatisticalPreconditioner::None,
    })
}

/// Configurable weighted least squares for repeated and ill-scaled fits.
pub fn weighted_least_squares_with_options(
    design: &CsrMatrix<f64>, response: &[f64], weights: &[f64],
    options: StatisticalSolveOptions,
) -> Result<StatisticalSolveResult, StatisticalSolveError> {
    solve_weighted_least_squares(design, response, weights, None, options)
}

/// Solve `min Σᵢ wᵢ(yᵢ − xᵢᵀβ)² + λ‖Pβ‖²` using CGLS over sparse operators.
///
/// `penalty` is the sparse operator `P`; it must have one column per design
/// coefficient. The augmented system is applied matrix-free as
/// `[W½X; √λP]`, so large GAM basis/penalty matrices remain CSR throughout.
/// A non-converged result is returned with `converged == false`, never hidden
/// as a successful fit.
pub fn penalized_weighted_least_squares(
    design: &CsrMatrix<f64>,
    response: &[f64],
    weights: &[f64],
    penalty: &CsrMatrix<f64>,
    penalty_weight: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<StatisticalSolveResult, StatisticalSolveError> {
    penalized_weighted_least_squares_with_options(
        design, response, weights, penalty, penalty_weight,
        StatisticalSolveOptions {
            max_iterations, tolerance, initial_solution: None,
            preconditioner: StatisticalPreconditioner::None,
        },
    )
}

/// Configurable penalized weighted least squares. A previous coefficient
/// vector can be supplied as the starting point for IRLS or smoothing paths.
pub fn penalized_weighted_least_squares_with_options(
    design: &CsrMatrix<f64>, response: &[f64], weights: &[f64],
    penalty: &CsrMatrix<f64>, penalty_weight: f64,
    options: StatisticalSolveOptions,
) -> Result<StatisticalSolveResult, StatisticalSolveError> {
    if !penalty_weight.is_finite() || penalty_weight <= 0.0 {
        return Err(StatisticalSolveError::InvalidPenaltyWeight);
    }
    if penalty.cols() != design.cols() {
        return Err(StatisticalSolveError::PenaltyWidth {
            actual: penalty.cols(), expected: design.cols(),
        });
    }
    solve_weighted_least_squares(design, response, weights, Some((penalty, penalty_weight)), options)
}

fn solve_weighted_least_squares(
    design: &CsrMatrix<f64>,
    response: &[f64],
    weights: &[f64],
    penalty: Option<(&CsrMatrix<f64>, f64)>,
    options: StatisticalSolveOptions,
) -> Result<StatisticalSolveResult, StatisticalSolveError> {
    if response.len() != design.rows() || weights.len() != design.rows() {
        return Err(StatisticalSolveError::ObservationLength {
            rows: design.rows(), cols: design.cols(),
        });
    }
    if options.max_iterations == 0 || !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(StatisticalSolveError::InvalidConvergenceSettings);
    }
    if let Some(initial) = &options.initial_solution {
        if initial.len() != design.cols() {
            return Err(StatisticalSolveError::InitialSolutionLength {
                actual: initial.len(), expected: design.cols(),
            });
        }
    }
    if !response.iter().all(|value| value.is_finite())
        || !weights.iter().all(|weight| weight.is_finite() && *weight >= 0.0)
        || !design.iter_entries().all(|(_, _, value)| value.is_finite())
        || options.initial_solution.as_ref().is_some_and(|values| {
            !values.iter().all(|value| value.is_finite())
        })
        || penalty.is_some_and(|(matrix, _)| {
            !matrix.iter_entries().all(|(_, _, value)| value.is_finite())
        })
    {
        return Err(StatisticalSolveError::InvalidNumericInput);
    }

    let root_weights: Vec<f64> = weights.iter().map(|weight| weight.sqrt()).collect();
    let mut solution = options.initial_solution.unwrap_or_else(|| vec![0.0; design.cols()]);
    let fitted = design.spmv(&solution)?;
    let mut residual_data: Vec<f64> = response.iter().zip(fitted).zip(&root_weights)
        .map(|((observed, fitted), root_weight)| (observed - fitted) * root_weight).collect();
    let mut residual_penalty: Option<Vec<f64>> = match penalty {
        Some((matrix, weight)) => {
            let scale = weight.sqrt();
            Some(matrix.spmv(&solution)?.into_iter().map(|value| -scale * value).collect())
        }
        None => None,
    };
    let mut gradient = adjoint(
        design, &residual_data, &root_weights,
        penalty, residual_penalty.as_deref(),
    )?;
    let inverse_diagonal = inverse_normal_diagonal(design, weights, penalty, options.preconditioner);
    let mut preconditioned_gradient = apply_diagonal(&gradient, &inverse_diagonal);
    let mut direction = preconditioned_gradient.clone();
    let mut gamma = dot(&gradient, &preconditioned_gradient);
    let gradient_norm_sq = dot(&gradient, &gradient);

    if gradient_norm_sq == 0.0 {
        let residual_norm = augmented_norm(&residual_data, residual_penalty.as_deref());
        return finalize_statistical_result(
            design, response, weights, penalty, solution, 0, residual_norm, true,
        );
    }
    let initial_gradient_norm = gradient_norm_sq.sqrt();

    for iteration in 1..=options.max_iterations {
        let projected_data = scaled_forward(design, &direction, &root_weights)?;
        let projected_penalty = match penalty {
            Some((matrix, weight)) => {
                let mut values = matrix.spmv(&direction)?;
                let scale = weight.sqrt();
                for value in &mut values { *value *= scale; }
                Some(values)
            }
            None => None,
        };
        let denominator = dot(&projected_data, &projected_data)
            + projected_penalty.as_ref().map_or(0.0, |values| dot(values, values));
        if !denominator.is_finite() || denominator <= 0.0 {
            let residual_norm = augmented_norm(&residual_data, residual_penalty.as_deref());
            return finalize_statistical_result(
                design, response, weights, penalty, solution, iteration - 1,
                residual_norm, false,
            );
        }

        let alpha = gamma / denominator;
        for (value, delta) in solution.iter_mut().zip(&direction) { *value += alpha * delta; }
        for (value, projected) in residual_data.iter_mut().zip(&projected_data) {
            *value -= alpha * projected;
        }
        if let (Some(residual), Some(projected)) =
            (residual_penalty.as_mut(), projected_penalty.as_ref())
        {
            for (value, projection) in residual.iter_mut().zip(projected) {
                *value -= alpha * projection;
            }
        }

        gradient = adjoint(
            design, &residual_data, &root_weights,
            penalty, residual_penalty.as_deref(),
        )?;
        let next_gradient_norm_sq = dot(&gradient, &gradient);
        let residual_norm = augmented_norm(&residual_data, residual_penalty.as_deref());
        if !next_gradient_norm_sq.is_finite() {
            return finalize_statistical_result(
                design, response, weights, penalty, solution, iteration,
                residual_norm, false,
            );
        }
        // Least-squares optima (especially penalized ones) generally retain
        // a nonzero augmented residual. CGLS therefore converges on the
        // normal residual ||Bᵀr||, not on ||r|| itself.
        if next_gradient_norm_sq.sqrt() / initial_gradient_norm <= options.tolerance {
            return finalize_statistical_result(
                design, response, weights, penalty, solution, iteration,
                residual_norm, true,
            );
        }
        if next_gradient_norm_sq == 0.0 {
            return finalize_statistical_result(
                design, response, weights, penalty, solution, iteration,
                residual_norm, false,
            );
        }
        preconditioned_gradient = apply_diagonal(&gradient, &inverse_diagonal);
        let next_gamma = dot(&gradient, &preconditioned_gradient);
        if !next_gamma.is_finite() || next_gamma <= 0.0 {
            return finalize_statistical_result(
                design, response, weights, penalty, solution, iteration,
                residual_norm, false,
            );
        }
        let beta = next_gamma / gamma;
        for (value, next) in direction.iter_mut().zip(&preconditioned_gradient) {
            *value = next + beta * *value;
        }
        gamma = next_gamma;
    }

    let residual_norm = augmented_norm(&residual_data, residual_penalty.as_deref());
    finalize_statistical_result(
        design, response, weights, penalty, solution, options.max_iterations, residual_norm, false,
    )
}

fn inverse_normal_diagonal(
    design: &CsrMatrix<f64>, weights: &[f64],
    penalty: Option<(&CsrMatrix<f64>, f64)>, preconditioner: StatisticalPreconditioner,
) -> Option<Vec<f64>> {
    if preconditioner == StatisticalPreconditioner::None { return None; }
    let mut diagonal = vec![0.0; design.cols()];
    for (row, column, value) in design.iter_entries() {
        diagonal[column] += weights[row] * value * value;
    }
    if let Some((matrix, weight)) = penalty {
        for (_, column, value) in matrix.iter_entries() {
            diagonal[column] += weight * value * value;
        }
    }
    Some(diagonal.into_iter().map(|value| if value > 0.0 { 1.0 / value } else { 1.0 }).collect())
}

fn apply_diagonal(values: &[f64], inverse_diagonal: &Option<Vec<f64>>) -> Vec<f64> {
    match inverse_diagonal {
        Some(diagonal) => values.iter().zip(diagonal).map(|(value, scale)| value * scale).collect(),
        None => values.to_vec(),
    }
}

fn scaled_forward(
    matrix: &CsrMatrix<f64>, vector: &[f64], scales: &[f64],
) -> Result<Vec<f64>, StatisticalSolveError> {
    let mut values = matrix.spmv(vector)?;
    for (value, scale) in values.iter_mut().zip(scales) { *value *= scale; }
    Ok(values)
}

fn adjoint(
    design: &CsrMatrix<f64>, data: &[f64], root_weights: &[f64],
    penalty: Option<(&CsrMatrix<f64>, f64)>, penalty_residual: Option<&[f64]>,
) -> Result<Vec<f64>, StatisticalSolveError> {
    let weighted_data: Vec<f64> = data.iter().zip(root_weights)
        .map(|(value, scale)| value * scale).collect();
    let mut result = design.transpose_spmv(&weighted_data)?;
    if let (Some((matrix, weight)), Some(residual)) = (penalty, penalty_residual)
    {
        let scale = weight.sqrt();
        let weighted_penalty: Vec<f64> = residual.iter().map(|value| value * scale).collect();
        let contribution = matrix.transpose_spmv(&weighted_penalty)?;
        for (value, extra) in result.iter_mut().zip(contribution) { *value += extra; }
    }
    Ok(result)
}

fn finalize_statistical_result(
    design: &CsrMatrix<f64>, response: &[f64], weights: &[f64],
    penalty: Option<(&CsrMatrix<f64>, f64)>, solution: Vec<f64>,
    iterations: usize, residual_norm: f64, converged: bool,
) -> Result<StatisticalSolveResult, StatisticalSolveError> {
    let fitted = design.spmv(&solution)?;
    let weighted_residual_sum_of_squares: f64 = response.iter().zip(&fitted).zip(weights)
        .map(|((observed, mean), weight)| weight * (observed - mean).powi(2))
        .sum();
    let penalty_contribution: f64 = match penalty {
        Some((matrix, weight)) => matrix.spmv(&solution)?.iter()
            .map(|value| weight * value * value).sum(),
        None => 0.0,
    };
    if !weighted_residual_sum_of_squares.is_finite() || !penalty_contribution.is_finite() {
        return Err(StatisticalSolveError::InvalidNumericInput);
    }
    Ok(StatisticalSolveResult {
        solution, iterations, residual_norm, converged,
        weighted_residual_sum_of_squares, penalty_contribution,
    })
}

fn augmented_norm(data: &[f64], penalty: Option<&[f64]>) -> f64 {
    (dot(data, data) + penalty.map_or(0.0, |values| dot(values, values))).sqrt()
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

    #[test]
    fn weighted_least_squares_excludes_zero_weight_rows() {
        // Rows encode [1, x]. The final response is an excluded outlier.
        let design = CsrMatrix::new(
            4, 2, vec![0, 1, 3, 5, 7],
            vec![0, 0, 1, 0, 1, 0, 1],
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
        ).unwrap();
        let result = weighted_least_squares(
            &design, &[1.0, 3.0, 5.0, 100.0], &[1.0, 1.0, 1.0, 0.0], 20, 1e-12,
        ).unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 1.0).abs() < 1e-10);
        assert!((result.solution[1] - 2.0).abs() < 1e-10);
        assert!(result.weighted_residual_sum_of_squares < 1e-18);
        assert_eq!(result.penalty_contribution, 0.0);
    }

    #[test]
    fn sparse_penalty_stabilizes_rank_deficient_design() {
        // Each observation sees beta0 + beta1 only. I₂ penalty yields the
        // analytic minimizer beta0 = beta1 = 6/7.
        let design = CsrMatrix::new(
            3, 2, vec![0, 2, 4, 6], vec![0, 1, 0, 1, 0, 1], vec![1.0; 6],
        ).unwrap();
        let penalty = CsrMatrix::new(
            2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0],
        ).unwrap();
        let result = penalized_weighted_least_squares(
            &design, &[2.0, 2.0, 2.0], &[1.0, 1.0, 1.0], &penalty, 1.0, 20, 1e-12,
        ).unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 6.0 / 7.0).abs() < 1e-10);
        assert!((result.solution[1] - 6.0 / 7.0).abs() < 1e-10);
        assert!((result.weighted_residual_sum_of_squares - 12.0 / 49.0).abs() < 1e-10);
        assert!((result.penalty_contribution - 72.0 / 49.0).abs() < 1e-10);
        assert!((result.objective() - 12.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn sparse_statistics_reject_invalid_inputs_and_reports_nonconvergence() {
        let design = CsrMatrix::new(
            2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![1.0, 1.0, 1.0],
        ).unwrap();
        assert!(matches!(
            weighted_least_squares(&design, &[1.0], &[1.0], 10, 1e-8),
            Err(StatisticalSolveError::ObservationLength { .. })
        ));
        assert!(matches!(
            weighted_least_squares(&design, &[1.0, 2.0], &[1.0, -1.0], 10, 1e-8),
            Err(StatisticalSolveError::InvalidNumericInput)
        ));
        let bad_penalty = CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
        assert!(matches!(
            penalized_weighted_least_squares(
                &design, &[1.0, 2.0], &[1.0, 1.0], &bad_penalty, 1.0, 10, 1e-8,
            ),
            Err(StatisticalSolveError::PenaltyWidth { .. })
        ));
        let incomplete = weighted_least_squares(
            &design, &[1.0, 2.0], &[1.0, 1.0], 1, 1e-14,
        ).unwrap();
        assert!(!incomplete.converged);
    }

    #[test]
    fn warm_start_reuses_a_converged_statistical_fit() {
        let design = CsrMatrix::new(
            3, 2, vec![0, 1, 3, 5], vec![0, 0, 1, 0, 1],
            vec![1.0, 1.0, 1.0, 1.0, 2.0],
        ).unwrap();
        let result = weighted_least_squares_with_options(
            &design, &[1.0, 3.0, 5.0], &[1.0; 3], StatisticalSolveOptions {
                initial_solution: Some(vec![1.0, 2.0]),
                max_iterations: 10, tolerance: 1e-12,
                preconditioner: StatisticalPreconditioner::Jacobi,
            },
        ).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert!(result.weighted_residual_sum_of_squares < 1e-24);
    }

    #[test]
    fn jacobi_preconditioning_handles_column_scale_disparity() {
        let design = CsrMatrix::new(
            3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1e-4, 1.0, 1e4],
        ).unwrap();
        let result = weighted_least_squares_with_options(
            &design, &[1e-4, 2.0, 3e4], &[1.0; 3], StatisticalSolveOptions {
                max_iterations: 5, tolerance: 1e-12, initial_solution: None,
                preconditioner: StatisticalPreconditioner::Jacobi,
            },
        ).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        for (actual, expected) in result.solution.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn configurable_solver_rejects_invalid_warm_start() {
        let design = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let result = weighted_least_squares_with_options(
            &design, &[1.0], &[1.0], StatisticalSolveOptions {
                initial_solution: Some(vec![0.0]), ..StatisticalSolveOptions::default()
            },
        );
        assert!(matches!(result, Err(StatisticalSolveError::InitialSolutionLength { .. })));
    }
}
