//! Sparse matrix formats.
//!
//! v1 scope: CSR only, plus SpMV. COO (convenient for incremental
//! construction, e.g. while parsing an AMPL model's constraint list) and
//! CSC (needed for some factorizations) are deliberately deferred —
//! see `docs/decisions/0002-sparse-v1-scope.md`.
//!
//! This crate is intentionally shared by three consumers:
//! - `NumericCoreSparse` (Swift) — general sparse linear algebra
//! - `NumericCoreGraph` (Swift) — adjacency matrices / graph Laplacians
//! - the AMPL presolve layer — the constraint matrix `A` in `Ax {<=,=,>=} b`
//!   is sparse for any model of realistic size, so this format is the
//!   handoff point between the modeling language and the solver.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SparseError {
    #[error("row_ptr length {actual} does not match rows+1 ({expected})")]
    RowPtrLength { expected: usize, actual: usize },

    #[error("row_ptr must start at zero, found {actual}")]
    RowPtrStart { actual: usize },

    #[error("row_ptr is not monotonic at index {index}: {previous} then {actual}")]
    RowPtrNotMonotonic { index: usize, previous: usize, actual: usize },

    #[error("row_ptr entry {index} is {actual}, beyond nnz {nnz}")]
    RowPtrOutOfBounds { index: usize, actual: usize, nnz: usize },

    #[error("row_ptr must end at nnz {expected}, found {actual}")]
    RowPtrEnd { expected: usize, actual: usize },

    #[error("col_indices length {col_indices} does not match values length {values}")]
    ColValuesMismatch { col_indices: usize, values: usize },

    #[error("column index {index} out of bounds for {cols} columns")]
    ColumnOutOfBounds { index: usize, cols: usize },

    #[error("dimension mismatch: matrix is {rows}x{cols}, vector has length {vec_len}")]
    DimensionMismatch { rows: usize, cols: usize, vec_len: usize },
}

/// Compressed Sparse Row matrix.
///
/// Standard three-array CSR: `row_ptr` has `rows + 1` entries, `col_indices`
/// and `values` are parallel arrays of length `nnz`.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>> CsrMatrix<T> {
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseError> {
        if row_ptr.len() != rows + 1 {
            return Err(SparseError::RowPtrLength { expected: rows + 1, actual: row_ptr.len() });
        }
        if col_indices.len() != values.len() {
            return Err(SparseError::ColValuesMismatch {
                col_indices: col_indices.len(),
                values: values.len(),
            });
        }
        if row_ptr.first().copied() != Some(0) {
            return Err(SparseError::RowPtrStart { actual: row_ptr.first().copied().unwrap_or_default() });
        }
        let nnz = values.len();
        for (index, &pointer) in row_ptr.iter().enumerate() {
            if pointer > nnz {
                return Err(SparseError::RowPtrOutOfBounds { index, actual: pointer, nnz });
            }
            if index > 0 && pointer < row_ptr[index - 1] {
                return Err(SparseError::RowPtrNotMonotonic {
                    index, previous: row_ptr[index - 1], actual: pointer,
                });
            }
        }
        if row_ptr[rows] != nnz {
            return Err(SparseError::RowPtrEnd { expected: nnz, actual: row_ptr[rows] });
        }
        if let Some(&bad) = col_indices.iter().find(|&&c| c >= cols) {
            return Err(SparseError::ColumnOutOfBounds { index: bad, cols });
        }
        Ok(Self { rows, cols, row_ptr, col_indices, values })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate `(row, col, value)` triples in row-major order. Added for
    /// consumers that need to expand a sparse matrix into a different
    /// representation — the first is `nc-optimize`'s dense revised
    /// simplex workspace (see that crate's `simplex` module), which
    /// works with a dense constraint matrix given this project's
    /// "smallish problems" scope for the simplex path (ADR — ties to
    /// the Rust-side sequencing discussion, not yet a numbered ADR in
    /// this repo since it's Swift-NumericCore's ADR set that tracks
    /// these).
    pub fn iter_entries(&self) -> impl Iterator<Item = (usize, usize, T)> + '_
    where
        T: Copy,
    {
        (0..self.rows).flat_map(move |row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            (start..end).map(move |idx| (row, self.col_indices[idx], self.values[idx]))
        })
    }

    /// Sparse matrix-vector product: `y = A * x`.
    pub fn spmv(&self, x: &[T]) -> Result<Vec<T>, SparseError> {
        if x.len() != self.cols {
            return Err(SparseError::DimensionMismatch {
                rows: self.rows,
                cols: self.cols,
                vec_len: x.len(),
            });
        }
        let mut y = vec![T::default(); self.rows];
        for row in 0..self.rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut acc = T::default();
            for idx in start..end {
                acc = acc + self.values[idx] * x[self.col_indices[idx]];
            }
            y[row] = acc;
        }
        Ok(y)
    }

    /// Sparse transpose-matrix-vector product: `y = Aᵀ * x`.
    ///
    /// CSR is row-oriented, so this performs one accumulation per stored
    /// entry rather than materializing a CSC/transpose representation. It is
    /// the adjoint required by portable sparse least-squares iterations.
    pub fn transpose_spmv(&self, x: &[T]) -> Result<Vec<T>, SparseError> {
        if x.len() != self.rows {
            return Err(SparseError::DimensionMismatch {
                rows: self.rows,
                cols: self.cols,
                vec_len: x.len(),
            });
        }
        let mut y = vec![T::default(); self.cols];
        for row in 0..self.rows {
            for index in self.row_ptr[row]..self.row_ptr[row + 1] {
                let column = self.col_indices[index];
                y[column] = y[column] + self.values[index] * x[row];
            }
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [[1, 0, 2],
    ///  [0, 3, 0]]
    fn sample() -> CsrMatrix<f64> {
        CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap()
    }

    #[test]
    fn spmv_matches_hand_computation() {
        let m = sample();
        let y = m.spmv(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(y, vec![3.0, 3.0]);
    }

    #[test]
    fn rejects_bad_row_ptr() {
        assert!(CsrMatrix::<f64>::new(2, 3, vec![0, 2], vec![0], vec![1.0]).is_err());
    }

    #[test]
    fn spmv_rejects_dimension_mismatch() {
        let m = sample();
        assert!(m.spmv(&[1.0, 1.0]).is_err());
    }

    #[test]
    fn transpose_spmv_matches_hand_computation() {
        let m = sample(); // [[1, 0, 2], [0, 3, 0]]
        let y = m.transpose_spmv(&[4.0, 5.0]).unwrap();
        assert_eq!(y, vec![4.0, 15.0, 8.0]);
    }

    #[test]
    fn rejects_invalid_row_pointer_structure_before_multiplication() {
        assert!(matches!(
            CsrMatrix::<f64>::new(1, 1, vec![1, 1], vec![], vec![]),
            Err(SparseError::RowPtrStart { .. })
        ));
        assert!(matches!(
            CsrMatrix::<f64>::new(2, 1, vec![0, 2, 1], vec![0], vec![1.0]),
            Err(SparseError::RowPtrOutOfBounds { .. })
        ));
        assert!(matches!(
            CsrMatrix::<f64>::new(2, 1, vec![0, 1, 0], vec![0], vec![1.0]),
            Err(SparseError::RowPtrNotMonotonic { .. })
        ));
        assert!(matches!(
            CsrMatrix::<f64>::new(1, 1, vec![0, 0], vec![0], vec![1.0]),
            Err(SparseError::RowPtrEnd { .. })
        ));
    }

    #[test]
    fn iter_entries_visits_every_nonzero_in_row_major_order() {
        let m = sample(); // [[1, 0, 2], [0, 3, 0]]
        let entries: Vec<_> = m.iter_entries().collect();
        assert_eq!(entries, vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)]);
    }
}
