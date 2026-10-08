use crate::CsrMatrix;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SparseDirectError {
    #[error("sparse factorization requires a square matrix, got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },
    #[error("right-hand side has {actual} entries, expected {expected}")]
    RightHandSideLength { actual: usize, expected: usize },
    #[error("matrix and right-hand side must contain only finite values")]
    InvalidNumericInput,
    #[error("matrix is singular at pivot {pivot}")]
    Singular { pivot: usize },
    #[error("matrix is not symmetric within tolerance at ({row}, {column})")]
    NotSymmetric { row: usize, column: usize },
    #[error("matrix is not positive definite at pivot {pivot}")]
    NotPositiveDefinite { pivot: usize },
    #[error("drop tolerance must be finite and non-negative")]
    InvalidDropTolerance,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseDirectReport {
    pub solution: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub factor_nonzeros: usize,
}

fn rows(matrix: &CsrMatrix<f64>) -> Result<Vec<BTreeMap<usize, f64>>, SparseDirectError> {
    if matrix.rows() != matrix.cols() {
        return Err(SparseDirectError::NonSquare {
            rows: matrix.rows(),
            cols: matrix.cols(),
        });
    }
    let mut result = vec![BTreeMap::new(); matrix.rows()];
    for (row, column, value) in matrix.iter_entries() {
        if !value.is_finite() {
            return Err(SparseDirectError::InvalidNumericInput);
        }
        *result[row].entry(column).or_insert(0.0) += value;
    }
    if result
        .iter()
        .any(|row| row.values().any(|value| !value.is_finite()))
    {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    Ok(result)
}

fn validate_rhs(rhs: &[f64], size: usize) -> Result<(), SparseDirectError> {
    if rhs.len() != size {
        return Err(SparseDirectError::RightHandSideLength {
            actual: rhs.len(),
            expected: size,
        });
    }
    if !rhs.iter().all(|value| value.is_finite()) {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    Ok(())
}

fn report(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    solution: Vec<f64>,
    factor_nonzeros: usize,
) -> Result<SparseDirectReport, SparseDirectError> {
    let product = matrix
        .spmv(&solution)
        .map_err(|_| SparseDirectError::InvalidNumericInput)?;
    let residual_norm = rhs
        .iter()
        .zip(product)
        .map(|(b, ax)| (b - ax).powi(2))
        .sum::<f64>()
        .sqrt();
    if !residual_norm.is_finite() {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    Ok(SparseDirectReport {
        solution,
        residual_norm,
        relative_residual: residual_norm / rhs_norm.max(1e-30),
        factor_nonzeros,
    })
}

/// Sparse LU with threshold-free partial pivoting. Fill is stored only when
/// generated, and one factorization may solve multiple right-hand sides.
#[derive(Debug, Clone)]
pub struct SparseLu {
    rows: Vec<BTreeMap<usize, f64>>,
    permutation: Vec<usize>,
    drop_tolerance: f64,
}

impl SparseLu {
    pub fn factor(matrix: &CsrMatrix<f64>, drop_tolerance: f64) -> Result<Self, SparseDirectError> {
        if !drop_tolerance.is_finite() || drop_tolerance < 0.0 {
            return Err(SparseDirectError::InvalidDropTolerance);
        }
        let mut rows = rows(matrix)?;
        let n = rows.len();
        let mut permutation: Vec<usize> = (0..n).collect();
        for pivot in 0..n {
            let selected = (pivot..n)
                .max_by(|&lhs, &rhs| {
                    rows[lhs]
                        .get(&pivot)
                        .copied()
                        .unwrap_or(0.0)
                        .abs()
                        .total_cmp(&rows[rhs].get(&pivot).copied().unwrap_or(0.0).abs())
                })
                .unwrap_or(pivot);
            let pivot_value = rows[selected].get(&pivot).copied().unwrap_or(0.0);
            if !pivot_value.is_finite() || pivot_value.abs() <= drop_tolerance {
                return Err(SparseDirectError::Singular { pivot });
            }
            rows.swap(pivot, selected);
            permutation.swap(pivot, selected);
            let upper: Vec<(usize, f64)> = rows[pivot]
                .range((pivot + 1)..)
                .map(|(&column, &value)| (column, value))
                .collect();
            let diagonal = rows[pivot][&pivot];
            for row in (pivot + 1)..n {
                let entry = rows[row].get(&pivot).copied().unwrap_or(0.0);
                if entry == 0.0 {
                    continue;
                }
                let factor = entry / diagonal;
                rows[row].insert(pivot, factor);
                for &(column, value) in &upper {
                    let updated = rows[row].get(&column).copied().unwrap_or(0.0) - factor * value;
                    if !updated.is_finite() {
                        return Err(SparseDirectError::InvalidNumericInput);
                    }
                    if updated.abs() <= drop_tolerance {
                        rows[row].remove(&column);
                    } else {
                        rows[row].insert(column, updated);
                    }
                }
            }
        }
        Ok(Self {
            rows,
            permutation,
            drop_tolerance,
        })
    }

    pub fn nonzeros(&self) -> usize {
        self.rows.iter().map(BTreeMap::len).sum()
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        let n = self.rows.len();
        validate_rhs(rhs, n)?;
        let mut solution: Vec<f64> = self.permutation.iter().map(|&row| rhs[row]).collect();
        for row in 0..n {
            let contribution: f64 = self.rows[row]
                .range(..row)
                .map(|(&column, &value)| value * solution[column])
                .sum();
            solution[row] -= contribution;
        }
        for row in (0..n).rev() {
            let diagonal = self.rows[row].get(&row).copied().unwrap_or(0.0);
            if diagonal.abs() <= self.drop_tolerance {
                return Err(SparseDirectError::Singular { pivot: row });
            }
            let contribution: f64 = self.rows[row]
                .range((row + 1)..)
                .map(|(&column, &value)| value * solution[column])
                .sum();
            solution[row] = (solution[row] - contribution) / diagonal;
        }
        Ok(solution)
    }

    pub fn solve_report(
        &self,
        matrix: &CsrMatrix<f64>,
        rhs: &[f64],
    ) -> Result<SparseDirectReport, SparseDirectError> {
        report(matrix, rhs, self.solve(rhs)?, self.nonzeros())
    }
}

/// Sparse lower Cholesky factorization for symmetric positive-definite input.
#[derive(Debug, Clone)]
pub struct SparseCholesky {
    lower: Vec<BTreeMap<usize, f64>>,
    drop_tolerance: f64,
}

impl SparseCholesky {
    pub fn factor(matrix: &CsrMatrix<f64>, drop_tolerance: f64) -> Result<Self, SparseDirectError> {
        if !drop_tolerance.is_finite() || drop_tolerance < 0.0 {
            return Err(SparseDirectError::InvalidDropTolerance);
        }
        let source = rows(matrix)?;
        let n = source.len();
        check_symmetric(&source, 1e-12)?;
        let mut lower = vec![BTreeMap::new(); n];
        for row in 0..n {
            for column in 0..=row {
                let mut value = source[row].get(&column).copied().unwrap_or(0.0);
                value -= dot_lower(&lower[row], &lower[column], column);
                if !value.is_finite() {
                    return Err(SparseDirectError::InvalidNumericInput);
                }
                if row == column {
                    if !value.is_finite() || value <= drop_tolerance {
                        return Err(SparseDirectError::NotPositiveDefinite { pivot: row });
                    }
                    lower[row].insert(row, value.sqrt());
                } else if value.abs() > drop_tolerance {
                    let diagonal = lower[column][&column];
                    lower[row].insert(column, value / diagonal);
                }
            }
        }
        Ok(Self {
            lower,
            drop_tolerance,
        })
    }

    pub fn nonzeros(&self) -> usize {
        self.lower.iter().map(BTreeMap::len).sum()
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.lower.len())?;
        triangular_solve(&self.lower, rhs, self.drop_tolerance)
    }

    pub fn solve_report(
        &self,
        matrix: &CsrMatrix<f64>,
        rhs: &[f64],
    ) -> Result<SparseDirectReport, SparseDirectError> {
        report(matrix, rhs, self.solve(rhs)?, self.nonzeros())
    }
}

/// Zero-fill incomplete LU preconditioner preserving the matrix pattern.
#[derive(Debug, Clone)]
pub struct Ilu0 {
    rows: Vec<BTreeMap<usize, f64>>,
    tolerance: f64,
}

impl Ilu0 {
    pub fn factor(matrix: &CsrMatrix<f64>, tolerance: f64) -> Result<Self, SparseDirectError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(SparseDirectError::InvalidDropTolerance);
        }
        let mut rows = rows(matrix)?;
        let n = rows.len();
        for row in 0..n {
            let lower_columns: Vec<usize> =
                rows[row].range(..row).map(|(&column, _)| column).collect();
            for column in lower_columns {
                let diagonal = rows[column].get(&column).copied().unwrap_or(0.0);
                if diagonal.abs() <= tolerance {
                    return Err(SparseDirectError::Singular { pivot: column });
                }
                let factor = rows[row][&column] / diagonal;
                rows[row].insert(column, factor);
                let upper: Vec<(usize, f64)> = rows[column]
                    .range((column + 1)..)
                    .map(|(&index, &value)| (index, value))
                    .collect();
                for (index, value) in upper {
                    if let Some(current) = rows[row].get_mut(&index) {
                        *current -= factor * value;
                        if !current.is_finite() {
                            return Err(SparseDirectError::InvalidNumericInput);
                        }
                    }
                }
            }
            if rows[row].get(&row).copied().unwrap_or(0.0).abs() <= tolerance {
                return Err(SparseDirectError::Singular { pivot: row });
            }
        }
        Ok(Self { rows, tolerance })
    }

    pub fn apply(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.rows.len())?;
        let n = rhs.len();
        let mut result = rhs.to_vec();
        for row in 0..n {
            let contribution: f64 = self.rows[row]
                .range(..row)
                .map(|(&column, &value)| value * result[column])
                .sum();
            result[row] -= contribution;
        }
        for row in (0..n).rev() {
            let diagonal = self.rows[row].get(&row).copied().unwrap_or(0.0);
            if diagonal.abs() <= self.tolerance {
                return Err(SparseDirectError::Singular { pivot: row });
            }
            let contribution: f64 = self.rows[row]
                .range((row + 1)..)
                .map(|(&column, &value)| value * result[column])
                .sum();
            result[row] = (result[row] - contribution) / diagonal;
        }
        Ok(result)
    }
}

/// Zero-fill incomplete Cholesky preconditioner for SPD matrices.
#[derive(Debug, Clone)]
pub struct IncompleteCholesky {
    lower: Vec<BTreeMap<usize, f64>>,
    tolerance: f64,
}

impl IncompleteCholesky {
    pub fn factor(matrix: &CsrMatrix<f64>, tolerance: f64) -> Result<Self, SparseDirectError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(SparseDirectError::InvalidDropTolerance);
        }
        let source = rows(matrix)?;
        check_symmetric(&source, 1e-12)?;
        let n = source.len();
        let mut lower = vec![BTreeMap::new(); n];
        for row in 0..n {
            let pattern: Vec<usize> = source[row]
                .range(..=row)
                .map(|(&column, _)| column)
                .collect();
            for column in pattern {
                let mut value = source[row].get(&column).copied().unwrap_or(0.0);
                value -= dot_lower(&lower[row], &lower[column], column);
                if !value.is_finite() {
                    return Err(SparseDirectError::InvalidNumericInput);
                }
                if row == column {
                    if value <= tolerance || !value.is_finite() {
                        return Err(SparseDirectError::NotPositiveDefinite { pivot: row });
                    }
                    lower[row].insert(row, value.sqrt());
                } else if value != 0.0 {
                    let diagonal = lower[column][&column];
                    lower[row].insert(column, value / diagonal);
                }
            }
        }
        Ok(Self { lower, tolerance })
    }

    pub fn apply(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.lower.len())?;
        triangular_solve(&self.lower, rhs, self.tolerance)
    }
}

fn check_symmetric(rows: &[BTreeMap<usize, f64>], tolerance: f64) -> Result<(), SparseDirectError> {
    for (row, entries) in rows.iter().enumerate() {
        for (&column, &value) in entries {
            let transpose = rows[column].get(&row).copied().unwrap_or(0.0);
            if (value - transpose).abs() > tolerance * value.abs().max(transpose.abs()).max(1.0) {
                return Err(SparseDirectError::NotSymmetric { row, column });
            }
        }
    }
    Ok(())
}

fn dot_lower(lhs: &BTreeMap<usize, f64>, rhs: &BTreeMap<usize, f64>, before: usize) -> f64 {
    lhs.range(..before)
        .filter_map(|(&column, &value)| rhs.get(&column).map(|other| value * other))
        .sum()
}

fn triangular_solve(
    lower: &[BTreeMap<usize, f64>],
    rhs: &[f64],
    tolerance: f64,
) -> Result<Vec<f64>, SparseDirectError> {
    let n = lower.len();
    let mut result = rhs.to_vec();
    for row in 0..n {
        let diagonal = lower[row].get(&row).copied().unwrap_or(0.0);
        if diagonal.abs() <= tolerance {
            return Err(SparseDirectError::Singular { pivot: row });
        }
        let contribution: f64 = lower[row]
            .range(..row)
            .map(|(&column, &value)| value * result[column])
            .sum();
        result[row] = (result[row] - contribution) / diagonal;
    }
    for row in (0..n).rev() {
        let diagonal = lower[row][&row];
        let contribution: f64 = ((row + 1)..n)
            .filter_map(|other| lower[other].get(&row).map(|value| value * result[other]))
            .sum();
        result[row] = (result[row] - contribution) / diagonal;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut maps = vec![BTreeMap::new(); rows];
        for &(row, column, value) in entries {
            maps[row].insert(column, value);
        }
        let mut pointers = vec![0];
        let mut columns = vec![];
        let mut values = vec![];
        for row in maps {
            for (column, value) in row {
                columns.push(column);
                values.push(value);
            }
            pointers.push(values.len());
        }
        CsrMatrix::new(rows, rows, pointers, columns, values).unwrap()
    }

    #[test]
    fn sparse_lu_pivots_and_solves() {
        let a = matrix(
            3,
            &[
                (0, 1, 2.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 1.0),
                (2, 0, 2.0),
                (2, 2, 3.0),
            ],
        );
        let factor = SparseLu::factor(&a, 0.0).unwrap();
        let report = factor.solve_report(&a, &[7.0, 3.0, 11.0]).unwrap();
        assert!(report.relative_residual < 1e-12);
        for (actual, expected) in report.solution.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_cholesky_solves_spd_system() {
        let a = matrix(
            3,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 2.0),
            ],
        );
        let factor = SparseCholesky::factor(&a, 0.0).unwrap();
        let report = factor.solve_report(&a, &[6.0, 10.0, 8.0]).unwrap();
        assert!(report.relative_residual < 1e-12);
        for (actual, expected) in report.solution.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn incomplete_factors_apply_finitely() {
        let a = matrix(
            3,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 2.0),
            ],
        );
        assert!(IncompleteCholesky::factor(&a, 0.0)
            .unwrap()
            .apply(&[1.0; 3])
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        assert!(Ilu0::factor(&a, 0.0)
            .unwrap()
            .apply(&[1.0; 3])
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn direct_factorizations_fail_closed() {
        let singular = matrix(2, &[(0, 0, 1.0), (1, 0, 2.0)]);
        assert!(matches!(
            SparseLu::factor(&singular, 0.0),
            Err(SparseDirectError::Singular { .. })
        ));
        let asymmetric = matrix(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 2.0)]);
        assert!(matches!(
            SparseCholesky::factor(&asymmetric, 0.0),
            Err(SparseDirectError::NotSymmetric { .. })
        ));
        let indefinite = matrix(2, &[(0, 0, 1.0), (1, 1, -1.0)]);
        assert!(matches!(
            SparseCholesky::factor(&indefinite, 0.0),
            Err(SparseDirectError::NotPositiveDefinite { .. })
        ));
    }
}
