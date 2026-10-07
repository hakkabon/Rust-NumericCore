//! Sparse reverse-mode and matrix-free derivatives for nonlinear graphs.

use std::collections::BTreeMap;

use crate::{NonlinearExpression, NonlinearModel, NonlinearNode, OptimizeError};

#[derive(Debug, Clone, PartialEq)]
pub struct SparseDerivative {
    pub dimension: usize,
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseDerivative {
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.dimension];
        for (&index, &value) in self.indices.iter().zip(&self.values) {
            dense[index] = value;
        }
        dense
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseJacobian {
    pub rows: usize,
    pub columns: usize,
    pub row_pointers: Vec<usize>,
    pub column_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseJacobian {
    pub fn multiply(&self, direction: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        if direction.len() != self.columns {
            return Err(invalid("Jacobian direction dimension does not agree"));
        }
        let mut result = vec![0.0; self.rows];
        for (row, value) in result.iter_mut().enumerate() {
            for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                *value += self.values[entry] * direction[self.column_indices[entry]];
            }
        }
        Ok(result)
    }

    pub fn transpose_multiply(&self, weights: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        if weights.len() != self.rows {
            return Err(invalid(
                "Jacobian transpose weight dimension does not agree",
            ));
        }
        let mut result = vec![0.0; self.columns];
        for (row, &weight) in weights.iter().enumerate() {
            for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                result[self.column_indices[entry]] += self.values[entry] * weight;
            }
        }
        Ok(result)
    }
}

impl NonlinearExpression {
    /// Value and sparse gradient via one primal and one reverse graph sweep.
    pub fn evaluate_sparse(
        &self,
        parameters: &[f64],
    ) -> Result<(f64, SparseDerivative), OptimizeError> {
        let values = primal_values(self, parameters)?;
        let output_value = values[self.output];
        let mut adjoints = vec![0.0; self.nodes.len()];
        adjoints[self.output] = 1.0;
        for index in (0..=self.output).rev() {
            let seed = adjoints[index];
            if seed == 0.0 {
                continue;
            }
            match self.nodes[index] {
                NonlinearNode::Constant(_) | NonlinearNode::Parameter(_) => {}
                NonlinearNode::Add(a, b) => {
                    adjoints[a] += seed;
                    adjoints[b] += seed;
                }
                NonlinearNode::Subtract(a, b) => {
                    adjoints[a] += seed;
                    adjoints[b] -= seed;
                }
                NonlinearNode::Multiply(a, b) => {
                    adjoints[a] += seed * values[b];
                    adjoints[b] += seed * values[a];
                }
                NonlinearNode::Divide(a, b) => {
                    adjoints[a] += seed / values[b];
                    adjoints[b] -= seed * values[a] / (values[b] * values[b]);
                }
                NonlinearNode::Negate(a) => adjoints[a] -= seed,
                NonlinearNode::Exp(a) => adjoints[a] += seed * values[index],
                NonlinearNode::Log(a) => adjoints[a] += seed / values[a],
                NonlinearNode::Sqrt(a) => adjoints[a] += seed * 0.5 / values[index],
                NonlinearNode::Sin(a) => adjoints[a] += seed * values[a].cos(),
                NonlinearNode::Cos(a) => adjoints[a] -= seed * values[a].sin(),
                NonlinearNode::Powf(a, power) => {
                    adjoints[a] += seed * power * values[a].powf(power - 1.0)
                }
            }
        }
        let mut entries = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if let NonlinearNode::Parameter(parameter) = *node {
                *entries.entry(parameter).or_insert(0.0) += adjoints[index];
            }
        }
        if !entries.values().all(|value| value.is_finite()) {
            return Err(invalid("expression derivative is non-finite"));
        }
        let (indices, values): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .filter(|(_, value)| *value != 0.0)
            .unzip();
        Ok((
            output_value,
            SparseDerivative {
                dimension: parameters.len(),
                indices,
                values,
            },
        ))
    }

    /// Value and directional derivative without constructing a gradient.
    pub fn evaluate_directional(
        &self,
        parameters: &[f64],
        direction: &[f64],
    ) -> Result<(f64, f64), OptimizeError> {
        if direction.len() != parameters.len() || !direction.iter().all(|v| v.is_finite()) {
            return Err(invalid(
                "direction must be finite and match the parameter dimension",
            ));
        }
        self.validate(parameters.len())?;
        if !parameters.iter().all(|v| v.is_finite()) {
            return Err(invalid("parameters must be finite"));
        }
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut tangents = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let (value, tangent) =
                directional_node(node, &values, &tangents, parameters, direction);
            if !value.is_finite() || !tangent.is_finite() {
                return Err(invalid(
                    "expression evaluation produced a non-finite value or derivative",
                ));
            }
            values.push(value);
            tangents.push(tangent);
            if index == self.output {
                break;
            }
        }
        Ok((values[self.output], tangents[self.output]))
    }
}

impl NonlinearModel {
    pub fn evaluate_sparse_residuals(
        &self,
        parameters: &[f64],
    ) -> Result<(Vec<f64>, SparseJacobian), OptimizeError> {
        self.validate()?;
        if parameters.len() != self.parameter_count || self.residuals.is_empty() {
            return Err(invalid(
                "model residual and parameter dimensions do not agree",
            ));
        }
        let mut residuals = Vec::with_capacity(self.residuals.len());
        let mut row_pointers = vec![0];
        let mut column_indices = Vec::new();
        let mut entries = Vec::new();
        for expression in &self.residuals {
            let (value, derivative) = expression.evaluate_sparse(parameters)?;
            residuals.push(value);
            column_indices.extend(derivative.indices);
            entries.extend(derivative.values);
            row_pointers.push(entries.len());
        }
        Ok((
            residuals,
            SparseJacobian {
                rows: self.residuals.len(),
                columns: self.parameter_count,
                row_pointers,
                column_indices,
                values: entries,
            },
        ))
    }

    pub fn jacobian_vector_product(
        &self,
        parameters: &[f64],
        direction: &[f64],
    ) -> Result<Vec<f64>, OptimizeError> {
        self.validate()?;
        if parameters.len() != self.parameter_count || direction.len() != self.parameter_count {
            return Err(invalid("Jacobian product dimensions do not agree"));
        }
        self.residuals
            .iter()
            .map(|expression| {
                expression
                    .evaluate_directional(parameters, direction)
                    .map(|v| v.1)
            })
            .collect()
    }

    pub fn jacobian_transpose_vector_product(
        &self,
        parameters: &[f64],
        weights: &[f64],
    ) -> Result<Vec<f64>, OptimizeError> {
        self.validate()?;
        if parameters.len() != self.parameter_count
            || weights.len() != self.residuals.len()
            || !weights.iter().all(|v| v.is_finite())
        {
            return Err(invalid(
                "transpose weights must be finite and match the residual dimension",
            ));
        }
        let mut result = vec![0.0; self.parameter_count];
        for (expression, &weight) in self.residuals.iter().zip(weights) {
            let (_, derivative) = expression.evaluate_sparse(parameters)?;
            for (&index, &value) in derivative.indices.iter().zip(&derivative.values) {
                result[index] += weight * value;
            }
        }
        Ok(result)
    }
}

fn primal_values(
    expression: &NonlinearExpression,
    parameters: &[f64],
) -> Result<Vec<f64>, OptimizeError> {
    expression.validate(parameters.len())?;
    if !parameters.iter().all(|v| v.is_finite()) {
        return Err(invalid("parameters must be finite"));
    }
    let mut values = Vec::with_capacity(expression.nodes.len());
    for (index, node) in expression.nodes.iter().enumerate() {
        let value = node_value(node, &values, parameters);
        if !value.is_finite() {
            return Err(invalid("expression evaluation produced a non-finite value"));
        }
        values.push(value);
        if index == expression.output {
            break;
        }
    }
    Ok(values)
}

fn node_value(node: &NonlinearNode, v: &[f64], p: &[f64]) -> f64 {
    match *node {
        NonlinearNode::Constant(x) => x,
        NonlinearNode::Parameter(i) => p[i],
        NonlinearNode::Add(a, b) => v[a] + v[b],
        NonlinearNode::Subtract(a, b) => v[a] - v[b],
        NonlinearNode::Multiply(a, b) => v[a] * v[b],
        NonlinearNode::Divide(a, b) => v[a] / v[b],
        NonlinearNode::Negate(a) => -v[a],
        NonlinearNode::Exp(a) => v[a].exp(),
        NonlinearNode::Log(a) => v[a].ln(),
        NonlinearNode::Sqrt(a) => v[a].sqrt(),
        NonlinearNode::Sin(a) => v[a].sin(),
        NonlinearNode::Cos(a) => v[a].cos(),
        NonlinearNode::Powf(a, x) => v[a].powf(x),
    }
}

fn directional_node(
    node: &NonlinearNode,
    v: &[f64],
    d: &[f64],
    p: &[f64],
    q: &[f64],
) -> (f64, f64) {
    match *node {
        NonlinearNode::Constant(x) => (x, 0.0),
        NonlinearNode::Parameter(i) => (p[i], q[i]),
        NonlinearNode::Add(a, b) => (v[a] + v[b], d[a] + d[b]),
        NonlinearNode::Subtract(a, b) => (v[a] - v[b], d[a] - d[b]),
        NonlinearNode::Multiply(a, b) => (v[a] * v[b], d[a] * v[b] + v[a] * d[b]),
        NonlinearNode::Divide(a, b) => (v[a] / v[b], (d[a] * v[b] - v[a] * d[b]) / (v[b] * v[b])),
        NonlinearNode::Negate(a) => (-v[a], -d[a]),
        NonlinearNode::Exp(a) => {
            let x = v[a].exp();
            (x, x * d[a])
        }
        NonlinearNode::Log(a) => (v[a].ln(), d[a] / v[a]),
        NonlinearNode::Sqrt(a) => {
            let x = v[a].sqrt();
            (x, 0.5 * d[a] / x)
        }
        NonlinearNode::Sin(a) => (v[a].sin(), v[a].cos() * d[a]),
        NonlinearNode::Cos(a) => (v[a].cos(), -v[a].sin() * d[a]),
        NonlinearNode::Powf(a, x) => (v[a].powf(x), x * v[a].powf(x - 1.0) * d[a]),
    }
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bound;
    fn shifted(parameter: usize, constant: f64) -> NonlinearExpression {
        NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(parameter),
                NonlinearNode::Constant(constant),
                NonlinearNode::Subtract(0, 1),
            ],
            2,
        )
    }
    #[test]
    fn sparse_and_matrix_free_products_agree() {
        let model = NonlinearModel::least_squares(
            4,
            vec![Bound::free(); 4],
            vec![shifted(0, 1.0), shifted(3, 2.0)],
        );
        let point = [3.0, 7.0, 8.0, 5.0];
        let direction = [2.0, 4.0, 6.0, 8.0];
        let (residuals, jacobian) = model.evaluate_sparse_residuals(&point).unwrap();
        assert_eq!(residuals, vec![2.0, 3.0]);
        assert_eq!(jacobian.values, vec![1.0, 1.0]);
        assert_eq!(
            jacobian.multiply(&direction).unwrap(),
            model.jacobian_vector_product(&point, &direction).unwrap()
        );
        assert_eq!(
            jacobian.transpose_multiply(&[5.0, 7.0]).unwrap(),
            model
                .jacobian_transpose_vector_product(&point, &[5.0, 7.0])
                .unwrap()
        );
    }

    #[test]
    fn reverse_mode_accumulates_repeated_parameters() {
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Multiply(0, 0),
                NonlinearNode::Parameter(3),
                NonlinearNode::Sin(2),
                NonlinearNode::Add(1, 3),
            ],
            4,
        );
        let point = [2.0, 10.0, 20.0, 0.0, 30.0];
        let dense = expression.evaluate(&point).unwrap();
        let (value, sparse) = expression.evaluate_sparse(&point).unwrap();
        assert_eq!(value, dense.value);
        assert_eq!(sparse.indices, vec![0, 3]);
        assert_eq!(sparse.to_dense(), dense.gradient);
        assert_eq!(
            expression
                .evaluate_directional(&point, &[3.0, 0.0, 0.0, 5.0, 0.0])
                .unwrap()
                .1,
            17.0
        );
    }
}
