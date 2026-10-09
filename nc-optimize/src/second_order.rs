//! Exact second-order graph derivatives and sparse saddle-point KKT solves.

use crate::{ConstrainedNonlinearProblem, NonlinearExpression, NonlinearNode, OptimizeError};
use nc_sparse::{CooMatrix, SparseLu};

#[derive(Debug, Clone, PartialEq)]
pub struct SecondOrderValue {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseHessian {
    pub dimension: usize,
    pub row_pointers: Vec<usize>,
    pub column_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseHessian {
    pub fn multiply(&self, direction: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        if direction.len() != self.dimension {
            return Err(invalid("Hessian direction dimension does not agree"));
        }
        let mut result = vec![0.0; self.dimension];
        for (row, value) in result.iter_mut().enumerate() {
            for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                *value += self.values[entry] * direction[self.column_indices[entry]];
            }
        }
        Ok(result)
    }
}

impl NonlinearExpression {
    pub fn evaluate_second_order(
        &self,
        parameters: &[f64],
    ) -> Result<SecondOrderValue, OptimizeError> {
        self.validate(parameters.len())?;
        if !parameters.iter().all(|value| value.is_finite()) {
            return Err(invalid("parameters must be finite"));
        }
        let n = parameters.len();
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut gradients = Vec::<Vec<f64>>::with_capacity(self.nodes.len());
        let mut hessians = Vec::<Vec<Vec<f64>>>::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let evaluated = second_order_node(node, parameters, &values, &gradients, &hessians, n);
            if !evaluated.0.is_finite()
                || !evaluated.1.iter().all(|value| value.is_finite())
                || !evaluated.2.iter().flatten().all(|value| value.is_finite())
            {
                return Err(invalid(
                    "second-order evaluation produced a non-finite value",
                ));
            }
            values.push(evaluated.0);
            gradients.push(evaluated.1);
            hessians.push(evaluated.2);
        }
        Ok(SecondOrderValue {
            value: values[self.output],
            gradient: gradients[self.output].clone(),
            hessian: hessians[self.output].clone(),
        })
    }

    pub fn hessian_vector_product(
        &self,
        parameters: &[f64],
        direction: &[f64],
    ) -> Result<Vec<f64>, OptimizeError> {
        if parameters.len() != direction.len() || !direction.iter().all(|value| value.is_finite()) {
            return Err(invalid(
                "Hessian direction must be finite and match the parameter dimension",
            ));
        }
        let value = self.evaluate_second_order(parameters)?;
        Ok(matvec(&value.hessian, direction))
    }

    pub fn evaluate_sparse_hessian(
        &self,
        parameters: &[f64],
        zero_tolerance: f64,
    ) -> Result<(f64, Vec<f64>, SparseHessian), OptimizeError> {
        if !zero_tolerance.is_finite() || zero_tolerance < 0.0 {
            return Err(OptimizeError::InvalidConfiguration(
                "invalid Hessian zero tolerance",
            ));
        }
        let value = self.evaluate_second_order(parameters)?;
        let hessian = sparse_hessian(&value.hessian, zero_tolerance);
        Ok((value.value, value.gradient, hessian))
    }
}

impl ConstrainedNonlinearProblem {
    pub fn lagrangian_hessian(
        &self,
        parameters: &[f64],
        weights: &[f64],
    ) -> Result<Vec<Vec<f64>>, OptimizeError> {
        self.validate()?;
        if weights.len() != self.constraints.len() {
            return Err(invalid(
                "Lagrangian weights must match the constraint count",
            ));
        }
        let objective = self
            .model
            .objective
            .as_ref()
            .ok_or_else(|| invalid("objective required"))?
            .evaluate_second_order(parameters)?;
        let mut result = objective.hessian;
        for (constraint, &weight) in self.constraints.iter().zip(weights) {
            if !weight.is_finite() {
                return Err(invalid("Lagrangian weights must be finite"));
            }
            if weight == 0.0 {
                continue;
            }
            let value = constraint.expression.evaluate_second_order(parameters)?;
            add_scaled_matrix(&mut result, &value.hessian, weight);
        }
        Ok(result)
    }

    pub fn lagrangian_hessian_vector_product(
        &self,
        parameters: &[f64],
        weights: &[f64],
        direction: &[f64],
    ) -> Result<Vec<f64>, OptimizeError> {
        if direction.len() != self.model.parameter_count {
            return Err(invalid(
                "Lagrangian Hessian direction dimension does not agree",
            ));
        }
        Ok(matvec(
            &self.lagrangian_hessian(parameters, weights)?,
            direction,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct SparseKktProblem {
    pub hessian: SparseHessian,
    pub jacobian: crate::SparseJacobian,
    pub primal_rhs: Vec<f64>,
    pub constraint_rhs: Vec<f64>,
    pub primal_regularization: f64,
    pub dual_regularization: f64,
    pub drop_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct SparseKktResult {
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub factor_nonzeros: usize,
}

pub fn solve_sparse_kkt(problem: &SparseKktProblem) -> Result<SparseKktResult, OptimizeError> {
    let n = problem.hessian.dimension;
    let m = problem.jacobian.rows;
    if problem.jacobian.columns != n
        || problem.primal_rhs.len() != n
        || problem.constraint_rhs.len() != m
        || !problem.primal_regularization.is_finite()
        || problem.primal_regularization < 0.0
        || !problem.dual_regularization.is_finite()
        || problem.dual_regularization < 0.0
        || !problem.drop_tolerance.is_finite()
        || problem.drop_tolerance < 0.0
    {
        return Err(invalid("invalid sparse KKT dimensions or regularization"));
    }
    let mut matrix = CooMatrix::new(n + m, n + m);
    for row in 0..n {
        for entry in problem.hessian.row_pointers[row]..problem.hessian.row_pointers[row + 1] {
            matrix
                .push(
                    row,
                    problem.hessian.column_indices[entry],
                    problem.hessian.values[entry],
                )
                .map_err(|error| invalid(&error.to_string()))?;
        }
        if problem.primal_regularization != 0.0 {
            matrix
                .push(row, row, problem.primal_regularization)
                .map_err(|error| invalid(&error.to_string()))?;
        }
    }
    for row in 0..m {
        for entry in problem.jacobian.row_pointers[row]..problem.jacobian.row_pointers[row + 1] {
            let column = problem.jacobian.column_indices[entry];
            let value = problem.jacobian.values[entry];
            matrix
                .push(n + row, column, value)
                .map_err(|error| invalid(&error.to_string()))?;
            matrix
                .push(column, n + row, value)
                .map_err(|error| invalid(&error.to_string()))?;
        }
        if problem.dual_regularization != 0.0 {
            matrix
                .push(n + row, n + row, -problem.dual_regularization)
                .map_err(|error| invalid(&error.to_string()))?;
        }
    }
    let matrix = matrix
        .to_csr()
        .map_err(|error| invalid(&error.to_string()))?;
    let rhs: Vec<f64> = problem
        .primal_rhs
        .iter()
        .chain(&problem.constraint_rhs)
        .copied()
        .collect();
    let factor = SparseLu::factor(&matrix, problem.drop_tolerance)
        .map_err(|error| invalid(&format!("sparse KKT factorization failed: {error}")))?;
    let report = factor
        .solve_report(&matrix, &rhs)
        .map_err(|error| invalid(&format!("sparse KKT solve failed: {error}")))?;
    Ok(SparseKktResult {
        primal: report.solution[..n].to_vec(),
        dual: report.solution[n..].to_vec(),
        residual_norm: report.residual_norm,
        relative_residual: report.relative_residual,
        factor_nonzeros: report.factor_nonzeros,
    })
}

fn second_order_node(
    node: &NonlinearNode,
    parameters: &[f64],
    values: &[f64],
    gradients: &[Vec<f64>],
    hessians: &[Vec<Vec<f64>>],
    n: usize,
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    match *node {
        NonlinearNode::Constant(value) => (value, vec![0.0; n], zeros(n)),
        NonlinearNode::Parameter(index) => {
            let mut gradient = vec![0.0; n];
            gradient[index] = 1.0;
            (parameters[index], gradient, zeros(n))
        }
        NonlinearNode::Add(a, b) => binary_linear(a, b, 1.0, values, gradients, hessians),
        NonlinearNode::Subtract(a, b) => binary_linear(a, b, -1.0, values, gradients, hessians),
        NonlinearNode::Multiply(a, b) => binary_second(
            a,
            b,
            values,
            gradients,
            hessians,
            values[b],
            values[a],
            0.0,
            1.0,
            0.0,
            values[a] * values[b],
        ),
        NonlinearNode::Divide(a, b) => binary_second(
            a,
            b,
            values,
            gradients,
            hessians,
            1.0 / values[b],
            -values[a] / values[b].powi(2),
            0.0,
            -1.0 / values[b].powi(2),
            2.0 * values[a] / values[b].powi(3),
            values[a] / values[b],
        ),
        NonlinearNode::Negate(a) => {
            unary_second(a, values, gradients, hessians, -values[a], -1.0, 0.0)
        }
        NonlinearNode::Exp(a) => {
            let value = values[a].exp();
            unary_second(a, values, gradients, hessians, value, value, value)
        }
        NonlinearNode::Log(a) => unary_second(
            a,
            values,
            gradients,
            hessians,
            values[a].ln(),
            1.0 / values[a],
            -1.0 / values[a].powi(2),
        ),
        NonlinearNode::Sqrt(a) => {
            let value = values[a].sqrt();
            unary_second(
                a,
                values,
                gradients,
                hessians,
                value,
                0.5 / value,
                -0.25 / (values[a] * value),
            )
        }
        NonlinearNode::Sin(a) => unary_second(
            a,
            values,
            gradients,
            hessians,
            values[a].sin(),
            values[a].cos(),
            -values[a].sin(),
        ),
        NonlinearNode::Cos(a) => unary_second(
            a,
            values,
            gradients,
            hessians,
            values[a].cos(),
            -values[a].sin(),
            -values[a].cos(),
        ),
        NonlinearNode::Powf(a, p) => unary_second(
            a,
            values,
            gradients,
            hessians,
            values[a].powf(p),
            p * values[a].powf(p - 1.0),
            p * (p - 1.0) * values[a].powf(p - 2.0),
        ),
    }
}

fn binary_linear(
    a: usize,
    b: usize,
    sign: f64,
    values: &[f64],
    gradients: &[Vec<f64>],
    hessians: &[Vec<Vec<f64>>],
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let gradient = gradients[a]
        .iter()
        .zip(&gradients[b])
        .map(|(x, y)| x + sign * y)
        .collect();
    let mut hessian = hessians[a].clone();
    add_scaled_matrix(&mut hessian, &hessians[b], sign);
    (values[a] + sign * values[b], gradient, hessian)
}
fn unary_second(
    a: usize,
    _values: &[f64],
    gradients: &[Vec<f64>],
    hessians: &[Vec<Vec<f64>>],
    value: f64,
    first: f64,
    second: f64,
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let gradient = gradients[a].iter().map(|entry| first * entry).collect();
    let mut hessian = scaled_matrix(&hessians[a], first);
    add_outer(&mut hessian, &gradients[a], &gradients[a], second);
    (value, gradient, hessian)
}
#[allow(clippy::too_many_arguments)]
fn binary_second(
    a: usize,
    b: usize,
    _values: &[f64],
    gradients: &[Vec<f64>],
    hessians: &[Vec<Vec<f64>>],
    fa: f64,
    fb: f64,
    faa: f64,
    fab: f64,
    fbb: f64,
    value: f64,
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let gradient = gradients[a]
        .iter()
        .zip(&gradients[b])
        .map(|(x, y)| fa * x + fb * y)
        .collect();
    let mut hessian = scaled_matrix(&hessians[a], fa);
    add_scaled_matrix(&mut hessian, &hessians[b], fb);
    add_outer(&mut hessian, &gradients[a], &gradients[a], faa);
    add_outer(&mut hessian, &gradients[a], &gradients[b], fab);
    add_outer(&mut hessian, &gradients[b], &gradients[a], fab);
    add_outer(&mut hessian, &gradients[b], &gradients[b], fbb);
    (value, gradient, hessian)
}
fn sparse_hessian(matrix: &[Vec<f64>], tolerance: f64) -> SparseHessian {
    let mut row_pointers = vec![0];
    let mut column_indices = vec![];
    let mut values = vec![];
    for row in matrix {
        for (column, &value) in row.iter().enumerate() {
            if value.abs() > tolerance {
                column_indices.push(column);
                values.push(value);
            }
        }
        row_pointers.push(values.len());
    }
    SparseHessian {
        dimension: matrix.len(),
        row_pointers,
        column_indices,
        values,
    }
}
fn zeros(n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; n]
}
fn scaled_matrix(matrix: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| scale * value).collect())
        .collect()
}
fn add_scaled_matrix(target: &mut [Vec<f64>], source: &[Vec<f64>], scale: f64) {
    for (target, source) in target.iter_mut().zip(source) {
        for (target, source) in target.iter_mut().zip(source) {
            *target += scale * source;
        }
    }
}
fn add_outer(target: &mut [Vec<f64>], left: &[f64], right: &[f64], scale: f64) {
    for i in left.indices() {
        for j in right.indices() {
            target[i][j] += scale * left[i] * right[j];
        }
    }
}
trait Indices {
    fn indices(&self) -> std::ops::Range<usize>;
}
impl Indices for [f64] {
    fn indices(&self) -> std::ops::Range<usize> {
        0..self.len()
    }
}
fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearNode::*, SparseJacobian};

    #[test]
    fn exact_hessian_and_product_match_quadratic() {
        let expression = NonlinearExpression::new(
            vec![
                Parameter(0),
                Parameter(1),
                Multiply(0, 1),
                Powf(0, 2.0),
                Add(2, 3),
            ],
            4,
        );
        let value = expression.evaluate_second_order(&[2.0, 3.0]).unwrap();
        assert_eq!(value.gradient, vec![7.0, 2.0]);
        assert_eq!(value.hessian, vec![vec![2.0, 1.0], vec![1.0, 0.0]]);
        assert_eq!(
            expression
                .hessian_vector_product(&[2.0, 3.0], &[4.0, 5.0])
                .unwrap(),
            vec![13.0, 4.0]
        );
    }

    #[test]
    fn sparse_kkt_solves_equality_newton_system() {
        let problem = SparseKktProblem {
            hessian: SparseHessian {
                dimension: 2,
                row_pointers: vec![0, 1, 2],
                column_indices: vec![0, 1],
                values: vec![2.0, 2.0],
            },
            jacobian: SparseJacobian {
                rows: 1,
                columns: 2,
                row_pointers: vec![0, 2],
                column_indices: vec![0, 1],
                values: vec![1.0, 1.0],
            },
            primal_rhs: vec![-2.0, -4.0],
            constraint_rhs: vec![-1.0],
            primal_regularization: 0.0,
            dual_regularization: 0.0,
            drop_tolerance: 1e-14,
        };
        let result = solve_sparse_kkt(&problem).unwrap();
        assert!((result.primal[0]).abs() < 1e-12);
        assert!((result.primal[1] + 1.0).abs() < 1e-12);
        assert!(result.relative_residual < 1e-12);
    }
}
