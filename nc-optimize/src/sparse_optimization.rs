//! Sparse LP/QP storage and validated adapters to production solvers.

use crate::{
    solve_convex_qp, solve_convex_qp_warm, Bound, OptimizeError, Problem, QuadraticOptions,
    QuadraticProblem, QuadraticResult, QuadraticWarmStart,
};
use nc_sparse::CsrMatrix;

/// A row- and column-oriented view of an LP matrix. The CSC view is stored as
/// CSR of `A^T`, avoiding repeated scans when algorithms need columns.
#[derive(Debug, Clone)]
pub struct SparseLinearWorkspace {
    pub problem: Problem,
    transpose: CsrMatrix<f64>,
}

impl SparseLinearWorkspace {
    pub fn new(problem: Problem) -> Result<Self, OptimizeError> {
        problem.validate()?;
        let transpose = problem.constraints.transpose()?;
        Ok(Self { problem, transpose })
    }
    pub fn nonzeros(&self) -> usize {
        self.problem.constraints.nnz()
    }
    pub fn row_activity(&self, point: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        Ok(self.problem.constraints.spmv(point)?)
    }
    pub fn transpose_product(&self, row_values: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        Ok(self.transpose.spmv(row_values)?)
    }
}

/// Symmetric matrix stored once, as canonical lower-triangular CSR.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseSymmetricMatrix {
    pub dimension: usize,
    pub row_pointers: Vec<usize>,
    pub column_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseSymmetricMatrix {
    pub fn new(
        dimension: usize,
        row_pointers: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, OptimizeError> {
        if dimension == 0
            || row_pointers.len() != dimension + 1
            || row_pointers.first() != Some(&0)
            || row_pointers.last() != Some(&values.len())
            || column_indices.len() != values.len()
            || !values.iter().all(|x| x.is_finite())
        {
            return Err(invalid("invalid sparse symmetric matrix structure"));
        }
        for row in 0..dimension {
            if row_pointers[row] > row_pointers[row + 1] || row_pointers[row + 1] > values.len() {
                return Err(invalid("invalid sparse symmetric row pointers"));
            }
            let columns = &column_indices[row_pointers[row]..row_pointers[row + 1]];
            if columns.iter().any(|&column| column > row)
                || columns.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid(
                    "sparse symmetric rows must be sorted unique lower triangles",
                ));
            }
        }
        Ok(Self {
            dimension,
            row_pointers,
            column_indices,
            values,
        })
    }

    pub fn nonzeros(&self) -> usize {
        self.values.len()
    }
    pub fn multiply(&self, point: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        if point.len() != self.dimension {
            return Err(invalid("sparse Hessian dimension mismatch"));
        }
        let mut result = vec![0.0; self.dimension];
        for row in 0..self.dimension {
            for index in self.row_pointers[row]..self.row_pointers[row + 1] {
                let column = self.column_indices[index];
                let value = self.values[index];
                result[row] += value * point[column];
                if column != row {
                    result[column] += value * point[row];
                }
            }
        }
        Ok(result)
    }

    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.dimension]; self.dimension];
        for row in 0..self.dimension {
            for index in self.row_pointers[row]..self.row_pointers[row + 1] {
                let column = self.column_indices[index];
                let value = self.values[index];
                result[row][column] = value;
                result[column][row] = value;
            }
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct SparseQuadraticProblem {
    pub quadratic: SparseSymmetricMatrix,
    pub linear: Vec<f64>,
    pub constant: f64,
    pub constraints: CsrMatrix<f64>,
    pub row_bounds: Vec<Bound>,
    pub variable_bounds: Vec<Bound>,
}

impl SparseQuadraticProblem {
    pub fn validate(&self, convexity_tolerance: f64) -> Result<(), OptimizeError> {
        self.to_dense().validate(convexity_tolerance)
    }
    pub fn objective(&self, point: &[f64]) -> Result<f64, OptimizeError> {
        if point.len() != self.linear.len() || !point.iter().all(|x| x.is_finite()) {
            return Err(invalid("sparse QP point dimension or values are invalid"));
        }
        let product = self.quadratic.multiply(point)?;
        Ok(self.constant
            + self
                .linear
                .iter()
                .zip(point)
                .map(|(a, b)| a * b)
                .sum::<f64>()
            + 0.5 * point.iter().zip(product).map(|(a, b)| a * b).sum::<f64>())
    }
    pub fn to_dense(&self) -> QuadraticProblem {
        QuadraticProblem {
            quadratic: self.quadratic.to_dense(),
            linear: self.linear.clone(),
            constant: self.constant,
            constraints: self.constraints.clone(),
            row_bounds: self.row_bounds.clone(),
            variable_bounds: self.variable_bounds.clone(),
        }
    }
}

/// Uses sparse storage/evaluation and the current production dense Cholesky
/// backend. This adapter is the seam for a later sparse KKT factorization.
pub fn solve_sparse_convex_qp(
    problem: &SparseQuadraticProblem,
    options: QuadraticOptions,
    warm_start: Option<&QuadraticWarmStart>,
) -> Result<QuadraticResult, OptimizeError> {
    problem.validate(options.convexity_tolerance)?;
    match warm_start {
        Some(warm) => solve_convex_qp_warm(&problem.to_dense(), options, warm),
        None => solve_convex_qp(&problem.to_dense(), options),
    }
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_qp_matches_dense_production_solver() {
        let q = SparseSymmetricMatrix::new(2, vec![0, 1, 3], vec![0, 0, 1], vec![2.0, 0.5, 2.0])
            .unwrap();
        let constraints = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        let sparse = SparseQuadraticProblem {
            quadratic: q,
            linear: vec![-2.0, -4.0],
            constant: 0.0,
            constraints,
            row_bounds: vec![Bound {
                lower: Some(1.0),
                upper: None,
            }],
            variable_bounds: vec![
                Bound {
                    lower: Some(0.0),
                    upper: None
                };
                2
            ],
        };
        let result = solve_sparse_convex_qp(&sparse, QuadraticOptions::default(), None).unwrap();
        assert_eq!(result.termination, crate::QuadraticTermination::Converged);
        assert!((result.objective - sparse.objective(&result.point).unwrap()).abs() < 1e-10);
        assert_eq!(sparse.quadratic.nonzeros(), 3);
    }
    #[test]
    fn lp_workspace_applies_rows_and_columns() {
        let problem = Problem {
            objective: vec![1.0, 2.0],
            constraints: CsrMatrix::new(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![3.0, 4.0, 5.0])
                .unwrap(),
            row_bounds: vec![Bound::free(); 2],
            var_bounds: vec![Bound::free(); 2],
            is_integer: vec![false; 2],
        };
        let workspace = SparseLinearWorkspace::new(problem).unwrap();
        assert_eq!(
            workspace.row_activity(&[2.0, 1.0]).unwrap(),
            vec![6.0, 13.0]
        );
        assert_eq!(
            workspace.transpose_product(&[1.0, 2.0]).unwrap(),
            vec![11.0, 10.0]
        );
    }
}
