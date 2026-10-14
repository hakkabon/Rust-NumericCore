//! Domain-neutral component representation and sparse residual assembly.

use crate::{DaeEvaluation, NonlinearExpression, OptimizeError};
use nc_sparse::{CooMatrix, CsrMatrix};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineeringUnknownRole {
    Differential,
    Algebraic,
    Multiplier,
    Reference,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringUnknown {
    pub name: String,
    pub unit: String,
    pub role: EngineeringUnknownRole,
    pub scale: f64,
    pub initial_value: f64,
    pub initial_derivative: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringParameter {
    pub name: String,
    pub unit: String,
    pub value: f64,
}

/// A local element port. `orientation` is used only by topology/incidence
/// consumers; constitutive expressions receive the unweighted unknown value.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringPort {
    pub name: String,
    pub unknown: usize,
    pub orientation: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringEquation {
    /// Global residual row that receives this contribution.
    pub row: usize,
    pub expression: NonlinearExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringElement {
    pub name: String,
    pub kind: String,
    /// Port order defines the element-local state and derivative order.
    pub ports: Vec<EngineeringPort>,
    pub equations: Vec<EngineeringEquation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringModel {
    pub unknowns: Vec<EngineeringUnknown>,
    pub parameters: Vec<EngineeringParameter>,
    pub elements: Vec<EngineeringElement>,
}

#[derive(Debug, Clone)]
pub struct EngineeringAssembly {
    pub residuals: Vec<f64>,
    pub jacobian_state: CsrMatrix<f64>,
    pub jacobian_derivative: CsrMatrix<f64>,
    pub parameter_jacobian: CsrMatrix<f64>,
}

impl EngineeringAssembly {
    pub fn to_dae_evaluation(&self) -> DaeEvaluation {
        DaeEvaluation {
            residuals: self.residuals.clone(),
            jacobian_state: dense(&self.jacobian_state),
            jacobian_derivative: dense(&self.jacobian_derivative),
        }
    }
}

impl EngineeringModel {
    pub fn new(
        unknowns: Vec<EngineeringUnknown>,
        parameters: Vec<EngineeringParameter>,
        elements: Vec<EngineeringElement>,
    ) -> Result<Self, OptimizeError> {
        let model = Self {
            unknowns,
            parameters,
            elements,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), OptimizeError> {
        if self.unknowns.is_empty() {
            return Err(invalid("engineering model must contain unknowns"));
        }
        unique_nonempty(
            self.unknowns.iter().map(|item| item.name.as_str()),
            "unknown",
        )?;
        unique_nonempty(
            self.parameters.iter().map(|item| item.name.as_str()),
            "parameter",
        )?;
        unique_nonempty(
            self.elements.iter().map(|item| item.name.as_str()),
            "element",
        )?;
        if self.unknowns.iter().any(|item| {
            item.unit.is_empty()
                || !item.scale.is_finite()
                || item.scale <= 0.0
                || !item.initial_value.is_finite()
                || !item.initial_derivative.is_finite()
        }) {
            return Err(invalid("invalid engineering unknown metadata"));
        }
        if self
            .parameters
            .iter()
            .any(|item| item.unit.is_empty() || !item.value.is_finite())
        {
            return Err(invalid("invalid engineering parameter metadata"));
        }
        let mut covered_rows = vec![false; self.unknowns.len()];
        for element in &self.elements {
            if element.kind.is_empty() || element.ports.is_empty() || element.equations.is_empty() {
                return Err(invalid(
                    "engineering elements need a kind, ports, and equations",
                ));
            }
            unique_nonempty(element.ports.iter().map(|port| port.name.as_str()), "port")?;
            if element
                .ports
                .iter()
                .any(|port| port.unknown >= self.unknowns.len() || !port.orientation.is_finite())
            {
                return Err(invalid("invalid engineering port"));
            }
            let input_count = 1 + 2 * element.ports.len() + self.parameters.len();
            for equation in &element.equations {
                if equation.row >= self.unknowns.len() {
                    return Err(invalid("engineering equation row is out of range"));
                }
                equation.expression.validate(input_count)?;
                covered_rows[equation.row] = true;
            }
        }
        if covered_rows.iter().any(|covered| !covered) {
            return Err(invalid(
                "every engineering residual row needs an element contribution",
            ));
        }
        Ok(())
    }

    pub fn initial_state(&self) -> Vec<f64> {
        self.unknowns
            .iter()
            .map(|item| item.initial_value)
            .collect()
    }

    pub fn initial_derivative(&self) -> Vec<f64> {
        self.unknowns
            .iter()
            .map(|item| item.initial_derivative)
            .collect()
    }

    /// Element-by-unknown signed incidence matrix. Duplicate ports are summed.
    pub fn incidence_matrix(&self) -> Result<CsrMatrix<f64>, OptimizeError> {
        let mut coo = CooMatrix::new(self.elements.len(), self.unknowns.len());
        for (row, element) in self.elements.iter().enumerate() {
            for port in &element.ports {
                coo.push(row, port.unknown, port.orientation)?;
            }
        }
        Ok(coo.to_csr()?)
    }

    pub fn assemble(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
    ) -> Result<EngineeringAssembly, OptimizeError> {
        if !time.is_finite()
            || state.len() != self.unknowns.len()
            || derivative.len() != self.unknowns.len()
            || state
                .iter()
                .chain(derivative)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("invalid engineering evaluation point"));
        }
        let n = self.unknowns.len();
        let p = self.parameters.len();
        let mut residuals = vec![0.0; n];
        let mut state_coo = CooMatrix::new(n, n);
        let mut derivative_coo = CooMatrix::new(n, n);
        let mut parameter_coo = CooMatrix::new(n, p);
        for element in &self.elements {
            let local = element.ports.len();
            let mut input = Vec::with_capacity(1 + 2 * local + p);
            input.push(time);
            input.extend(element.ports.iter().map(|port| state[port.unknown]));
            input.extend(element.ports.iter().map(|port| derivative[port.unknown]));
            input.extend(self.parameters.iter().map(|parameter| parameter.value));
            for equation in &element.equations {
                let value = equation.expression.evaluate(&input)?;
                residuals[equation.row] += value.value;
                for (local_index, port) in element.ports.iter().enumerate() {
                    push_nonzero(
                        &mut state_coo,
                        equation.row,
                        port.unknown,
                        value.gradient[1 + local_index],
                    )?;
                    push_nonzero(
                        &mut derivative_coo,
                        equation.row,
                        port.unknown,
                        value.gradient[1 + local + local_index],
                    )?;
                }
                for parameter in 0..p {
                    push_nonzero(
                        &mut parameter_coo,
                        equation.row,
                        parameter,
                        value.gradient[1 + 2 * local + parameter],
                    )?;
                }
            }
        }
        if residuals.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "engineering assembly produced non-finite residuals",
            ));
        }
        Ok(EngineeringAssembly {
            residuals,
            jacobian_state: state_coo.to_csr()?,
            jacobian_derivative: derivative_coo.to_csr()?,
            parameter_jacobian: parameter_coo.to_csr()?,
        })
    }

    /// Convenience bridge to Phase 39's residual-form DAE integrator.
    pub fn dae_evaluation(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
    ) -> Result<DaeEvaluation, OptimizeError> {
        Ok(self.assemble(time, state, derivative)?.to_dae_evaluation())
    }
}

fn push_nonzero(
    coo: &mut CooMatrix<f64>,
    row: usize,
    column: usize,
    value: f64,
) -> Result<(), OptimizeError> {
    if value != 0.0 {
        coo.push(row, column, value)?;
    }
    Ok(())
}

fn unique_nonempty<'a>(
    names: impl Iterator<Item = &'a str>,
    kind: &str,
) -> Result<(), OptimizeError> {
    let mut seen = BTreeSet::new();
    for name in names {
        if name.is_empty() || !seen.insert(name) {
            return Err(invalid(&format!(
                "{kind} names must be non-empty and unique"
            )));
        }
    }
    Ok(())
}

fn dense(matrix: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; matrix.cols()]; matrix.rows()];
    for (row, column, value) in matrix.iter_entries() {
        result[row][column] = value;
    }
    result
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NonlinearNode::*;

    fn unknown(name: &str, role: EngineeringUnknownRole) -> EngineeringUnknown {
        EngineeringUnknown {
            name: name.into(),
            unit: "V".into(),
            role,
            scale: 1.0,
            initial_value: 0.0,
            initial_derivative: 0.0,
        }
    }

    #[test]
    fn assembles_dynamic_component_and_parameter_sensitivities() {
        // C*vDot + v/R - source, input=[time,v,vDot,R,C,source].
        let expression = NonlinearExpression::new(
            vec![
                Parameter(2),
                Parameter(4),
                Multiply(0, 1),
                Parameter(1),
                Parameter(3),
                Divide(3, 4),
                Add(2, 5),
                Parameter(5),
                Subtract(6, 7),
            ],
            8,
        );
        let model = EngineeringModel::new(
            vec![unknown("v", EngineeringUnknownRole::Differential)],
            vec![
                EngineeringParameter {
                    name: "R".into(),
                    unit: "ohm".into(),
                    value: 2.,
                },
                EngineeringParameter {
                    name: "C".into(),
                    unit: "F".into(),
                    value: 3.,
                },
                EngineeringParameter {
                    name: "source".into(),
                    unit: "A".into(),
                    value: 1.,
                },
            ],
            vec![EngineeringElement {
                name: "load".into(),
                kind: "RC".into(),
                ports: vec![EngineeringPort {
                    name: "p".into(),
                    unknown: 0,
                    orientation: 1.,
                }],
                equations: vec![EngineeringEquation { row: 0, expression }],
            }],
        )
        .unwrap();
        let assembled = model.assemble(0., &[2.], &[0.25]).unwrap();
        assert!((assembled.residuals[0] - 0.75).abs() < 1e-12);
        assert_eq!(assembled.jacobian_state.spmv(&[1.]).unwrap(), vec![0.5]);
        assert_eq!(assembled.jacobian_derivative.spmv(&[1.]).unwrap(), vec![3.]);
        assert_eq!(
            assembled.parameter_jacobian.spmv(&[1., 0., 0.]).unwrap(),
            vec![-0.5]
        );
    }

    #[test]
    fn scatters_two_node_element_and_builds_incidence() {
        // Spring residuals k*(u0-u1), -k*(u0-u1).
        let positive = NonlinearExpression::new(
            vec![
                Parameter(1),
                Parameter(2),
                Subtract(0, 1),
                Parameter(5),
                Multiply(2, 3),
            ],
            4,
        );
        let negative = NonlinearExpression::new(
            vec![
                Parameter(1),
                Parameter(2),
                Subtract(0, 1),
                Parameter(5),
                Multiply(2, 3),
                Negate(4),
            ],
            5,
        );
        let model = EngineeringModel::new(
            vec![
                unknown("u0", EngineeringUnknownRole::Algebraic),
                unknown("u1", EngineeringUnknownRole::Algebraic),
            ],
            vec![EngineeringParameter {
                name: "k".into(),
                unit: "N/m".into(),
                value: 10.,
            }],
            vec![EngineeringElement {
                name: "spring".into(),
                kind: "spring".into(),
                ports: vec![
                    EngineeringPort {
                        name: "a".into(),
                        unknown: 0,
                        orientation: 1.,
                    },
                    EngineeringPort {
                        name: "b".into(),
                        unknown: 1,
                        orientation: -1.,
                    },
                ],
                equations: vec![
                    EngineeringEquation {
                        row: 0,
                        expression: positive,
                    },
                    EngineeringEquation {
                        row: 1,
                        expression: negative,
                    },
                ],
            }],
        )
        .unwrap();
        let assembled = model.assemble(0., &[0.2, 0.1], &[0., 0.]).unwrap();
        assert_eq!(assembled.residuals, vec![1., -1.]);
        assert_eq!(
            assembled.jacobian_state.spmv(&[1., 1.]).unwrap(),
            vec![0., 0.]
        );
        assert_eq!(
            model.incidence_matrix().unwrap().spmv(&[2., 1.]).unwrap(),
            vec![1.]
        );
    }

    #[test]
    fn validation_rejects_uncovered_rows() {
        let result = EngineeringModel::new(
            vec![
                unknown("a", EngineeringUnknownRole::Algebraic),
                unknown("b", EngineeringUnknownRole::Algebraic),
            ],
            vec![],
            vec![EngineeringElement {
                name: "only-a".into(),
                kind: "constraint".into(),
                ports: vec![EngineeringPort {
                    name: "a".into(),
                    unknown: 0,
                    orientation: 1.,
                }],
                equations: vec![EngineeringEquation {
                    row: 0,
                    expression: NonlinearExpression::new(vec![Parameter(1)], 0),
                }],
            }],
        );
        assert!(result.is_err());
    }

    #[test]
    fn engineering_model_drives_dae_integrator() {
        let mut state = unknown("x", EngineeringUnknownRole::Differential);
        state.initial_value = 1.;
        state.initial_derivative = -1.;
        let model = EngineeringModel::new(
            vec![state],
            vec![],
            vec![EngineeringElement {
                name: "decay".into(),
                kind: "first-order".into(),
                ports: vec![EngineeringPort {
                    name: "x".into(),
                    unknown: 0,
                    orientation: 1.,
                }],
                equations: vec![EngineeringEquation {
                    row: 0,
                    expression: NonlinearExpression::new(
                        vec![Parameter(1), Parameter(2), Add(0, 1)],
                        2,
                    ),
                }],
            }],
        )
        .unwrap();
        let result = crate::integrate_dae(
            0.,
            1.,
            &model.initial_state(),
            &model.initial_derivative(),
            Default::default(),
            |t, y, yd| model.dae_evaluation(t, y, yd),
        )
        .unwrap();
        assert!((result.points.last().unwrap().state[0] - (-1f64).exp()).abs() < 5e-4);
    }
}
