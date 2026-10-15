//! Implicit forward and discrete-adjoint engineering sensitivities.

use crate::{EngineeringAssembly, EngineeringModel, OptimizeError};
use nc_sparse::{CooMatrix, CsrMatrix, SparseLu};

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringResponseGradient {
    pub name: String,
    pub state_gradient: Vec<f64>,
    pub direct_parameter_gradient: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringResponseSensitivity {
    pub name: String,
    /// Solution of `A.transpose * adjoint = state_gradient`.
    pub adjoint: Vec<f64>,
    pub parameter_gradient: Vec<f64>,
    /// Independent contraction through the forward state sensitivities.
    pub forward_parameter_gradient: Vec<f64>,
    pub maximum_forward_adjoint_mismatch: f64,
    pub adjoint_residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringSensitivityResult {
    /// One vector per parameter: `dy/dp[j]`.
    pub state_sensitivities: Vec<Vec<f64>>,
    pub responses: Vec<EngineeringResponseSensitivity>,
    pub operating_residual_norm: f64,
    pub maximum_forward_residual_norm: f64,
    pub derivative_sensitivity_factor: f64,
}

impl EngineeringModel {
    /// Analyze sensitivities using `A = F_y + alpha F_ydot`. Steady models
    /// use `alpha = 0`; an implicit DAE step supplies its local
    /// `d(ydot)/dy` coefficient.
    pub fn sensitivities(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        derivative_sensitivity_factor: f64,
        responses: &[EngineeringResponseGradient],
    ) -> Result<EngineeringSensitivityResult, OptimizeError> {
        let alpha = derivative_sensitivity_factor;
        let mut names = std::collections::BTreeSet::new();
        if !alpha.is_finite()
            || self.parameters.is_empty()
            || responses.is_empty()
            || responses.iter().any(|response| {
                response.name.is_empty()
                    || !names.insert(response.name.as_str())
                    || response.state_gradient.len() != self.unknowns.len()
                    || response.direct_parameter_gradient.len() != self.parameters.len()
                    || response
                        .state_gradient
                        .iter()
                        .chain(&response.direct_parameter_gradient)
                        .any(|value| !value.is_finite())
            })
        {
            return Err(invalid("invalid engineering sensitivity request"));
        }
        let assembly = self.assemble(time, state, derivative)?;
        let tangent = effective_tangent(&assembly, alpha)?;
        let factor = SparseLu::factor(&tangent, 0.0).map_err(direct_error)?;
        let mut state_sensitivities = Vec::with_capacity(self.parameters.len());
        let mut maximum_forward_residual_norm: f64 = 0.0;
        for parameter in 0..self.parameters.len() {
            let parameter_column = column(&assembly.parameter_jacobian, parameter);
            let rhs: Vec<f64> = parameter_column.iter().map(|value| -value).collect();
            let sensitivity = factor.solve(&rhs).map_err(direct_error)?;
            let product = tangent.spmv(&sensitivity)?;
            maximum_forward_residual_norm = maximum_forward_residual_norm.max(norm(
                &product
                    .iter()
                    .zip(&parameter_column)
                    .map(|(lhs, rhs)| lhs + rhs)
                    .collect::<Vec<_>>(),
            ));
            state_sensitivities.push(sensitivity);
        }
        let transpose = tangent.transpose()?;
        let transpose_factor = SparseLu::factor(&transpose, 0.0).map_err(direct_error)?;
        let parameter_transpose = assembly.parameter_jacobian.transpose()?;
        let response_results = responses
            .iter()
            .map(|response| {
                let adjoint = transpose_factor
                    .solve(&response.state_gradient)
                    .map_err(direct_error)?;
                let product = transpose.spmv(&adjoint)?;
                let adjoint_residual_norm = norm(
                    &product
                        .iter()
                        .zip(&response.state_gradient)
                        .map(|(lhs, rhs)| lhs - rhs)
                        .collect::<Vec<_>>(),
                );
                let implicit = parameter_transpose.spmv(&adjoint)?;
                let parameter_gradient: Vec<f64> = response
                    .direct_parameter_gradient
                    .iter()
                    .zip(implicit)
                    .map(|(direct, implicit)| direct - implicit)
                    .collect();
                let forward_parameter_gradient: Vec<f64> = state_sensitivities
                    .iter()
                    .enumerate()
                    .map(|(parameter, sensitivity)| {
                        response.direct_parameter_gradient[parameter]
                            + response
                                .state_gradient
                                .iter()
                                .zip(sensitivity)
                                .map(|(gradient, state)| gradient * state)
                                .sum::<f64>()
                    })
                    .collect();
                let maximum_forward_adjoint_mismatch = parameter_gradient
                    .iter()
                    .zip(&forward_parameter_gradient)
                    .map(|(lhs, rhs)| (lhs - rhs).abs())
                    .fold(0.0_f64, f64::max);
                Ok(EngineeringResponseSensitivity {
                    name: response.name.clone(),
                    adjoint,
                    parameter_gradient,
                    forward_parameter_gradient,
                    maximum_forward_adjoint_mismatch,
                    adjoint_residual_norm,
                })
            })
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        Ok(EngineeringSensitivityResult {
            state_sensitivities,
            responses: response_results,
            operating_residual_norm: norm(&assembly.residuals),
            maximum_forward_residual_norm,
            derivative_sensitivity_factor: alpha,
        })
    }
}

fn effective_tangent(
    assembly: &EngineeringAssembly,
    alpha: f64,
) -> Result<CsrMatrix<f64>, OptimizeError> {
    let mut coo = CooMatrix::new(
        assembly.jacobian_state.rows(),
        assembly.jacobian_state.cols(),
    );
    for (row, column, value) in assembly.jacobian_state.iter_entries() {
        coo.push(row, column, value)?;
    }
    if alpha != 0.0 {
        for (row, column, value) in assembly.jacobian_derivative.iter_entries() {
            coo.push(row, column, alpha * value)?;
        }
    }
    Ok(coo.to_csr()?)
}

fn column(matrix: &CsrMatrix<f64>, target: usize) -> Vec<f64> {
    let mut result = vec![0.0; matrix.rows()];
    for (row, column, value) in matrix.iter_entries() {
        if column == target {
            result[row] = value;
        }
    }
    result
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn direct_error(error: nc_sparse::SparseDirectError) -> OptimizeError {
    invalid(&format!(
        "engineering sensitivity factorization failed: {error}"
    ))
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EngineeringElement, EngineeringEquation, EngineeringParameter, EngineeringPort,
        EngineeringUnknown, EngineeringUnknownRole, NonlinearExpression, NonlinearNode::*,
    };

    fn scalar_model(role: EngineeringUnknownRole) -> EngineeringModel {
        // F = C*ydot + k*y - p; inputs=[t,y,ydot,k,C,p].
        let expression = NonlinearExpression::new(
            vec![
                Parameter(4),
                Parameter(2),
                Multiply(0, 1),
                Parameter(3),
                Parameter(1),
                Multiply(3, 4),
                Add(2, 5),
                Parameter(5),
                Subtract(6, 7),
            ],
            8,
        );
        EngineeringModel::new(
            vec![EngineeringUnknown {
                name: "y".into(),
                unit: "m".into(),
                role,
                scale: 1.0,
                initial_value: 0.0,
                initial_derivative: 0.0,
            }],
            vec![
                EngineeringParameter {
                    name: "k".into(),
                    unit: "N/m".into(),
                    value: 4.0,
                },
                EngineeringParameter {
                    name: "C".into(),
                    unit: "N s/m".into(),
                    value: 2.0,
                },
                EngineeringParameter {
                    name: "p".into(),
                    unit: "N".into(),
                    value: 8.0,
                },
            ],
            vec![EngineeringElement {
                name: "balance".into(),
                kind: "scalar".into(),
                ports: vec![EngineeringPort {
                    name: "y".into(),
                    unknown: 0,
                    orientation: 1.0,
                }],
                equations: vec![EngineeringEquation { row: 0, expression }],
            }],
        )
        .unwrap()
    }

    #[test]
    fn steady_forward_and_adjoint_sensitivities_agree() {
        let responses = vec![
            EngineeringResponseGradient {
                name: "displacement".into(),
                state_gradient: vec![1.0],
                direct_parameter_gradient: vec![0.0; 3],
            },
            EngineeringResponseGradient {
                name: "scaled".into(),
                state_gradient: vec![3.0],
                direct_parameter_gradient: vec![1.0, 0.0, 0.0],
            },
        ];
        let result = scalar_model(EngineeringUnknownRole::Algebraic)
            .sensitivities(0.0, &[2.0], &[0.0], 0.0, &responses)
            .unwrap();
        assert!((result.state_sensitivities[0][0] + 0.5).abs() < 1e-12);
        assert_eq!(result.state_sensitivities[1], vec![0.0]);
        assert!((result.state_sensitivities[2][0] - 0.25).abs() < 1e-12);
        assert_eq!(
            result.responses[0].parameter_gradient,
            vec![-0.5, 0.0, 0.25]
        );
        assert!((result.responses[1].parameter_gradient[0] + 0.5).abs() < 1e-12);
        assert!(result.responses[0].maximum_forward_adjoint_mismatch < 1e-12);
        assert!(result.maximum_forward_residual_norm < 1e-12);
    }

    #[test]
    fn implicit_dae_factor_uses_derivative_jacobian() {
        let responses = vec![EngineeringResponseGradient {
            name: "state".into(),
            state_gradient: vec![1.0],
            direct_parameter_gradient: vec![0.0; 3],
        }];
        let result = scalar_model(EngineeringUnknownRole::Differential)
            .sensitivities(0.0, &[1.0], &[2.0], 3.0, &responses)
            .unwrap();
        assert!((result.state_sensitivities[2][0] - 0.1).abs() < 1e-12);
        assert!((result.responses[0].adjoint[0] - 0.1).abs() < 1e-12);
        assert_eq!(result.derivative_sensitivity_factor, 3.0);
    }

    #[test]
    fn malformed_request_and_singular_tangent_fail_closed() {
        let model = scalar_model(EngineeringUnknownRole::Algebraic);
        let bad = vec![EngineeringResponseGradient {
            name: "bad".into(),
            state_gradient: vec![],
            direct_parameter_gradient: vec![0.0; 3],
        }];
        assert!(model.sensitivities(0.0, &[2.0], &[0.0], 0.0, &bad).is_err());
        let singular = EngineeringModel::new(
            vec![EngineeringUnknown {
                name: "y".into(),
                unit: "m".into(),
                role: EngineeringUnknownRole::Algebraic,
                scale: 1.0,
                initial_value: 0.0,
                initial_derivative: 0.0,
            }],
            vec![EngineeringParameter {
                name: "p".into(),
                unit: "N".into(),
                value: 1.0,
            }],
            vec![EngineeringElement {
                name: "load".into(),
                kind: "source".into(),
                ports: vec![EngineeringPort {
                    name: "y".into(),
                    unknown: 0,
                    orientation: 1.0,
                }],
                equations: vec![EngineeringEquation {
                    row: 0,
                    expression: NonlinearExpression::new(vec![Parameter(3)], 0),
                }],
            }],
        )
        .unwrap();
        let response = vec![EngineeringResponseGradient {
            name: "y".into(),
            state_gradient: vec![1.0],
            direct_parameter_gradient: vec![0.0],
        }];
        assert!(singular
            .sensitivities(0.0, &[0.0], &[0.0], 0.0, &response)
            .is_err());
    }
}
