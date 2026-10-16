//! Coupled engineering steady solves and adjoint-driven design optimization.

use crate::{
    minimize_lbfgsb, Bound, EngineeringModel, EngineeringParameter, EngineeringResponseGradient,
    LbfgsOptions, LbfgsResult, NonlinearExpression, OptimizeError,
};
use nc_sparse::SparseLu;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiphysicsDiscipline {
    pub name: String,
    pub equation_rows: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiphysicsSteadyTermination {
    ConvergedResidual,
    ConvergedStep,
    IterationLimit,
    LineSearchFailed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiphysicsSteadyOptions {
    pub max_iterations: usize,
    pub residual_tolerance: f64,
    pub step_tolerance: f64,
    pub maximum_line_search_iterations: usize,
    pub backtracking: f64,
}
impl Default for MultiphysicsSteadyOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            residual_tolerance: 1e-10,
            step_tolerance: 1e-10,
            maximum_line_search_iterations: 20,
            backtracking: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiphysicsSteadyResult {
    pub state: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_norm: f64,
    pub discipline_residual_norms: BTreeMap<String, f64>,
    pub iterations: usize,
    pub linear_solves: usize,
    pub termination: MultiphysicsSteadyTermination,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiphysicsModel {
    pub engineering_model: EngineeringModel,
    pub disciplines: Vec<MultiphysicsDiscipline>,
}

impl MultiphysicsModel {
    pub fn new(
        engineering_model: EngineeringModel,
        disciplines: Vec<MultiphysicsDiscipline>,
    ) -> Result<Self, OptimizeError> {
        let model = Self {
            engineering_model,
            disciplines,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), OptimizeError> {
        let names: BTreeSet<_> = self
            .disciplines
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        if self.disciplines.is_empty()
            || names.len() != self.disciplines.len()
            || self
                .disciplines
                .iter()
                .any(|item| item.name.is_empty() || item.equation_rows.is_empty())
        {
            return Err(invalid(
                "multiphysics disciplines must be named and non-empty",
            ));
        }
        let rows: Vec<_> = self
            .disciplines
            .iter()
            .flat_map(|item| &item.equation_rows)
            .copied()
            .collect();
        let unique: BTreeSet<_> = rows.iter().copied().collect();
        if rows.len() != self.engineering_model.unknowns.len()
            || unique.len() != rows.len()
            || unique != (0..self.engineering_model.unknowns.len()).collect()
        {
            return Err(invalid(
                "discipline equation rows must partition the coupled system",
            ));
        }
        Ok(())
    }

    pub fn replacing_parameter_values(&self, values: &[f64]) -> Result<Self, OptimizeError> {
        if values.len() != self.engineering_model.parameters.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("replacement parameter values are invalid"));
        }
        let parameters = self
            .engineering_model
            .parameters
            .iter()
            .zip(values)
            .map(|(parameter, value)| EngineeringParameter {
                name: parameter.name.clone(),
                unit: parameter.unit.clone(),
                value: *value,
            })
            .collect();
        Self::new(
            EngineeringModel::new(
                self.engineering_model.unknowns.clone(),
                parameters,
                self.engineering_model.elements.clone(),
            )?,
            self.disciplines.clone(),
        )
    }

    pub fn solve_steady(
        &self,
        initial: Option<&[f64]>,
        options: MultiphysicsSteadyOptions,
    ) -> Result<MultiphysicsSteadyResult, OptimizeError> {
        validate_options(options)?;
        let mut state =
            initial.map_or_else(|| self.engineering_model.initial_state(), |x| x.to_vec());
        if state.len() != self.engineering_model.unknowns.len()
            || state.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid multiphysics initial state"));
        }
        let derivative = vec![0.0; state.len()];
        let mut linear_solves = 0;
        for iteration in 0..=options.max_iterations {
            let assembly = self.engineering_model.assemble(0.0, &state, &derivative)?;
            let residual_norm = norm(&assembly.residuals);
            if residual_norm <= options.residual_tolerance {
                return Ok(self.finish(
                    state,
                    assembly.residuals,
                    iteration,
                    linear_solves,
                    MultiphysicsSteadyTermination::ConvergedResidual,
                ));
            }
            if iteration == options.max_iterations {
                return Ok(self.finish(
                    state,
                    assembly.residuals,
                    iteration,
                    linear_solves,
                    MultiphysicsSteadyTermination::IterationLimit,
                ));
            }
            let factor = SparseLu::factor(&assembly.jacobian_state, 0.0).map_err(direct_error)?;
            let rhs: Vec<_> = assembly.residuals.iter().map(|value| -value).collect();
            let step = factor.solve(&rhs).map_err(direct_error)?;
            linear_solves += 1;
            if norm(&step) <= options.step_tolerance * (1.0 + norm(&state)) {
                return Ok(self.finish(
                    state,
                    assembly.residuals,
                    iteration,
                    linear_solves,
                    MultiphysicsSteadyTermination::ConvergedStep,
                ));
            }
            let mut scale = 1.0;
            let mut accepted = None;
            for _ in 0..options.maximum_line_search_iterations {
                let candidate: Vec<_> = state
                    .iter()
                    .zip(&step)
                    .map(|(value, delta)| value + scale * delta)
                    .collect();
                if let Ok(next) = self
                    .engineering_model
                    .assemble(0.0, &candidate, &derivative)
                {
                    if norm(&next.residuals) < residual_norm {
                        accepted = Some(candidate);
                        break;
                    }
                }
                scale *= options.backtracking;
            }
            let Some(next) = accepted else {
                return Ok(self.finish(
                    state,
                    assembly.residuals,
                    iteration,
                    linear_solves,
                    MultiphysicsSteadyTermination::LineSearchFailed,
                ));
            };
            state = next;
        }
        unreachable!()
    }

    fn finish(
        &self,
        state: Vec<f64>,
        residuals: Vec<f64>,
        iterations: usize,
        linear_solves: usize,
        termination: MultiphysicsSteadyTermination,
    ) -> MultiphysicsSteadyResult {
        let discipline_residual_norms = self
            .disciplines
            .iter()
            .map(|discipline| {
                (
                    discipline.name.clone(),
                    norm(
                        &discipline
                            .equation_rows
                            .iter()
                            .map(|row| residuals[*row])
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect();
        MultiphysicsSteadyResult {
            residual_norm: norm(&residuals),
            state,
            residuals,
            discipline_residual_norms,
            iterations,
            linear_solves,
            termination,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiphysicsDesignVariable {
    pub parameter: usize,
    pub bound: Bound,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiphysicsDesignProblem {
    pub model: MultiphysicsModel,
    /// Inputs are all steady states followed by all engineering parameters.
    pub response: NonlinearExpression,
    pub design_variables: Vec<MultiphysicsDesignVariable>,
}
impl MultiphysicsDesignProblem {
    pub fn validate(&self) -> Result<(), OptimizeError> {
        self.response.validate(
            self.model.engineering_model.unknowns.len()
                + self.model.engineering_model.parameters.len(),
        )?;
        let indices: BTreeSet<_> = self
            .design_variables
            .iter()
            .map(|item| item.parameter)
            .collect();
        if self.design_variables.is_empty()
            || indices.len() != self.design_variables.len()
            || self.design_variables.iter().any(|item| {
                item.parameter >= self.model.engineering_model.parameters.len()
                    || !valid_bound(item.bound)
            })
        {
            return Err(invalid("invalid multiphysics design variables"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MultiphysicsDesignResult {
    pub design: Vec<f64>,
    pub state: Vec<f64>,
    pub response: f64,
    pub gradient: Vec<f64>,
    pub steady_result: MultiphysicsSteadyResult,
    pub optimization_result: LbfgsResult,
    pub coupled_solves: usize,
}

pub fn optimize_multiphysics_design(
    problem: &MultiphysicsDesignProblem,
    steady_options: MultiphysicsSteadyOptions,
    optimization_options: LbfgsOptions,
) -> Result<MultiphysicsDesignResult, OptimizeError> {
    problem.validate()?;
    let base: Vec<_> = problem
        .model
        .engineering_model
        .parameters
        .iter()
        .map(|item| item.value)
        .collect();
    let initial: Vec<_> = problem
        .design_variables
        .iter()
        .map(|item| base[item.parameter])
        .collect();
    let bounds: Vec<_> = problem
        .design_variables
        .iter()
        .map(|item| item.bound)
        .collect();
    let mut warm_state = problem.model.engineering_model.initial_state();
    let mut coupled_solves = 0;
    let mut evaluate = |design: &[f64]| -> Result<_, OptimizeError> {
        let mut parameters = base.clone();
        for (variable, value) in problem.design_variables.iter().zip(design) {
            parameters[variable.parameter] = *value;
        }
        let model = problem.model.replacing_parameter_values(&parameters)?;
        let mut steady = model.solve_steady(Some(&warm_state), steady_options)?;
        if steady.termination != MultiphysicsSteadyTermination::ConvergedResidual {
            steady = model.solve_steady(None, steady_options)?;
        }
        if steady.termination != MultiphysicsSteadyTermination::ConvergedResidual {
            return Err(invalid("coupled steady solve did not converge"));
        }
        coupled_solves += 1;
        warm_state = steady.state.clone();
        let mut inputs = steady.state.clone();
        inputs.extend(&parameters);
        let response = problem.response.evaluate(&inputs)?;
        let n = steady.state.len();
        let sensitivity = model.engineering_model.sensitivities(
            0.0,
            &steady.state,
            &vec![0.0; n],
            0.0,
            &[EngineeringResponseGradient {
                name: "design-response".into(),
                state_gradient: response.gradient[..n].to_vec(),
                direct_parameter_gradient: response.gradient[n..].to_vec(),
            }],
        )?;
        let total = &sensitivity.responses[0].parameter_gradient;
        let gradient = problem
            .design_variables
            .iter()
            .map(|item| total[item.parameter])
            .collect();
        Ok((response.value, gradient, model, steady))
    };
    let optimized = minimize_lbfgsb(&initial, &bounds, optimization_options, |design| {
        evaluate(design).map(|value| (value.0, value.1))
    })?;
    let (response, gradient, _, steady_result) = evaluate(&optimized.point)?;
    Ok(MultiphysicsDesignResult {
        design: optimized.point.clone(),
        state: steady_result.state.clone(),
        response,
        gradient,
        steady_result,
        optimization_result: optimized,
        coupled_solves,
    })
}

fn validate_options(options: MultiphysicsSteadyOptions) -> Result<(), OptimizeError> {
    if options.max_iterations == 0
        || options.maximum_line_search_iterations == 0
        || !options.residual_tolerance.is_finite()
        || options.residual_tolerance <= 0.0
        || !options.step_tolerance.is_finite()
        || options.step_tolerance <= 0.0
        || !options.backtracking.is_finite()
        || options.backtracking <= 0.0
        || options.backtracking >= 1.0
    {
        return Err(invalid("invalid steady solver options"));
    }
    Ok(())
}
fn valid_bound(bound: Bound) -> bool {
    bound.lower.is_none_or(f64::is_finite)
        && bound.upper.is_none_or(f64::is_finite)
        && bound.lower.unwrap_or(f64::NEG_INFINITY) <= bound.upper.unwrap_or(f64::INFINITY)
}
fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
fn direct_error(error: nc_sparse::SparseDirectError) -> OptimizeError {
    invalid(&format!("multiphysics factorization failed: {error}"))
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EngineeringElement, EngineeringEquation, EngineeringPort, EngineeringUnknown,
        EngineeringUnknownRole, NonlinearNode::*,
    };

    fn model() -> MultiphysicsModel {
        // x + a*T - p = 0, T - b*x = 0; local inputs=[t,x,T,xdot,Tdot,a,b,p].
        let field = NonlinearExpression::new(
            vec![
                Parameter(1),
                Parameter(5),
                Parameter(2),
                Multiply(1, 2),
                Add(0, 3),
                Parameter(7),
                Subtract(4, 5),
            ],
            6,
        );
        let thermal = NonlinearExpression::new(
            vec![
                Parameter(2),
                Parameter(6),
                Parameter(1),
                Multiply(1, 2),
                Subtract(0, 3),
            ],
            4,
        );
        let engineering = EngineeringModel::new(
            vec![
                EngineeringUnknown {
                    name: "x".into(),
                    unit: "m".into(),
                    role: EngineeringUnknownRole::Algebraic,
                    scale: 1.0,
                    initial_value: 0.0,
                    initial_derivative: 0.0,
                },
                EngineeringUnknown {
                    name: "T".into(),
                    unit: "K".into(),
                    role: EngineeringUnknownRole::Algebraic,
                    scale: 1.0,
                    initial_value: 0.0,
                    initial_derivative: 0.0,
                },
            ],
            vec![
                EngineeringParameter {
                    name: "a".into(),
                    unit: "1".into(),
                    value: 0.5,
                },
                EngineeringParameter {
                    name: "b".into(),
                    unit: "1".into(),
                    value: 0.4,
                },
                EngineeringParameter {
                    name: "p".into(),
                    unit: "N".into(),
                    value: 1.0,
                },
            ],
            vec![EngineeringElement {
                name: "coupler".into(),
                kind: "field-thermal".into(),
                ports: vec![
                    EngineeringPort {
                        name: "field".into(),
                        unknown: 0,
                        orientation: 1.0,
                    },
                    EngineeringPort {
                        name: "thermal".into(),
                        unknown: 1,
                        orientation: 1.0,
                    },
                ],
                equations: vec![
                    EngineeringEquation {
                        row: 0,
                        expression: field,
                    },
                    EngineeringEquation {
                        row: 1,
                        expression: thermal,
                    },
                ],
            }],
        )
        .unwrap();
        MultiphysicsModel::new(
            engineering,
            vec![
                MultiphysicsDiscipline {
                    name: "field".into(),
                    equation_rows: vec![0],
                },
                MultiphysicsDiscipline {
                    name: "thermal".into(),
                    equation_rows: vec![1],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn coupled_steady_solve_reports_each_discipline() {
        let result = model().solve_steady(None, Default::default()).unwrap();
        assert_eq!(
            result.termination,
            MultiphysicsSteadyTermination::ConvergedResidual
        );
        assert!((result.state[0] - 1.0 / 1.2).abs() < 1e-12);
        assert!((result.state[1] - 0.4 / 1.2).abs() < 1e-12);
        assert!(result.discipline_residual_norms["field"] < 1e-12);
        assert!(result.discipline_residual_norms["thermal"] < 1e-12);
    }

    #[test]
    fn adjoint_drives_bounded_design_optimization() {
        // 0.5*(x-2)^2 + 0.05*p^2; inputs=[x,T,a,b,p].
        let response = NonlinearExpression::new(
            vec![
                Parameter(0),
                Constant(2.0),
                Subtract(0, 1),
                Powf(2, 2.0),
                Constant(0.5),
                Multiply(3, 4),
                Parameter(4),
                Powf(6, 2.0),
                Constant(0.05),
                Multiply(7, 8),
                Add(5, 9),
            ],
            10,
        );
        let problem = MultiphysicsDesignProblem {
            model: model(),
            response,
            design_variables: vec![MultiphysicsDesignVariable {
                parameter: 2,
                bound: Bound {
                    lower: Some(0.0),
                    upper: Some(5.0),
                },
            }],
        };
        let result =
            optimize_multiphysics_design(&problem, Default::default(), Default::default()).unwrap();
        assert!((result.design[0] - 2.0979020979).abs() < 1e-6);
        assert!((result.state[0] - result.design[0] / 1.2).abs() < 1e-8);
        assert!(result.gradient[0].abs() < 1e-6);
        assert!(result.coupled_solves > 1);
    }

    #[test]
    fn discipline_rows_must_partition_system() {
        let valid = model();
        assert!(MultiphysicsModel::new(
            valid.engineering_model,
            vec![MultiphysicsDiscipline {
                name: "incomplete".into(),
                equation_rows: vec![0]
            }]
        )
        .is_err());
    }
}
