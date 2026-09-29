//! Portable nonlinear expression graph shared by objective and residual models.

use crate::{
    minimize_lbfgs, minimize_lbfgsb, nonlinear_least_squares_configured, Bound, LbfgsOptions,
    LbfgsResult, NonlinearLeastSquaresOptions, NonlinearLeastSquaresResult, OptimizeError,
    RobustLoss,
};

/// One node in a topologically ordered scalar expression graph. Every operand
/// index must refer to an earlier node.
#[derive(Debug, Clone, PartialEq)]
pub enum NonlinearNode {
    Constant(f64),
    Parameter(usize),
    Add(usize, usize),
    Subtract(usize, usize),
    Multiply(usize, usize),
    Divide(usize, usize),
    Negate(usize),
    Exp(usize),
    Log(usize),
    Sqrt(usize),
    Sin(usize),
    Cos(usize),
    Powf(usize, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearExpression {
    pub nodes: Vec<NonlinearNode>,
    pub output: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableValue {
    pub value: f64,
    pub gradient: Vec<f64>,
}

impl NonlinearExpression {
    pub fn new(nodes: Vec<NonlinearNode>, output: usize) -> Self {
        Self { nodes, output }
    }

    pub fn validate(&self, parameter_count: usize) -> Result<(), OptimizeError> {
        if self.nodes.is_empty() || self.output >= self.nodes.len() {
            return Err(invalid("expression must contain a valid output node"));
        }
        for (index, node) in self.nodes.iter().enumerate() {
            let valid_operand = |operand: usize| operand < index;
            let valid = match node {
                NonlinearNode::Constant(value) => value.is_finite(),
                NonlinearNode::Parameter(parameter) => *parameter < parameter_count,
                NonlinearNode::Add(a, b)
                | NonlinearNode::Subtract(a, b)
                | NonlinearNode::Multiply(a, b)
                | NonlinearNode::Divide(a, b) => valid_operand(*a) && valid_operand(*b),
                NonlinearNode::Negate(a)
                | NonlinearNode::Exp(a)
                | NonlinearNode::Log(a)
                | NonlinearNode::Sqrt(a)
                | NonlinearNode::Sin(a)
                | NonlinearNode::Cos(a) => valid_operand(*a),
                NonlinearNode::Powf(a, exponent) => valid_operand(*a) && exponent.is_finite(),
            };
            if !valid {
                return Err(invalid("expression graph is not valid topological form"));
            }
        }
        Ok(())
    }

    pub fn evaluate(&self, parameters: &[f64]) -> Result<DifferentiableValue, OptimizeError> {
        self.validate(parameters.len())?;
        if !parameters.iter().all(|value| value.is_finite()) {
            return Err(invalid("parameters must be finite"));
        }
        let n = parameters.len();
        let mut values = Vec::with_capacity(self.nodes.len());
        let mut derivatives: Vec<Vec<f64>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let (value, gradient) = match *node {
                NonlinearNode::Constant(value) => (value, vec![0.0; n]),
                NonlinearNode::Parameter(parameter) => {
                    let mut gradient = vec![0.0; n];
                    gradient[parameter] = 1.0;
                    (parameters[parameter], gradient)
                }
                NonlinearNode::Add(a, b) => {
                    binary(&values, &derivatives, a, b, |x, y| x + y, |_, _| (1.0, 1.0))
                }
                NonlinearNode::Subtract(a, b) => binary(
                    &values,
                    &derivatives,
                    a,
                    b,
                    |x, y| x - y,
                    |_, _| (1.0, -1.0),
                ),
                NonlinearNode::Multiply(a, b) => {
                    binary(&values, &derivatives, a, b, |x, y| x * y, |x, y| (y, x))
                }
                NonlinearNode::Divide(a, b) => binary(
                    &values,
                    &derivatives,
                    a,
                    b,
                    |x, y| x / y,
                    |x, y| (1.0 / y, -x / (y * y)),
                ),
                NonlinearNode::Negate(a) => unary(&values, &derivatives, a, |x| -x, |_| -1.0),
                NonlinearNode::Exp(a) => unary(&values, &derivatives, a, f64::exp, f64::exp),
                NonlinearNode::Log(a) => unary(&values, &derivatives, a, f64::ln, |x| 1.0 / x),
                NonlinearNode::Sqrt(a) => {
                    unary(&values, &derivatives, a, f64::sqrt, |x| 0.5 / x.sqrt())
                }
                NonlinearNode::Sin(a) => unary(&values, &derivatives, a, f64::sin, f64::cos),
                NonlinearNode::Cos(a) => unary(&values, &derivatives, a, f64::cos, |x| -x.sin()),
                NonlinearNode::Powf(a, p) => unary(
                    &values,
                    &derivatives,
                    a,
                    |x| x.powf(p),
                    |x| p * x.powf(p - 1.0),
                ),
            };
            if !value.is_finite() || !gradient.iter().all(|value| value.is_finite()) {
                return Err(invalid(
                    "expression evaluation produced a non-finite value or derivative",
                ));
            }
            values.push(value);
            derivatives.push(gradient);
        }
        Ok(DifferentiableValue {
            value: values[self.output],
            gradient: derivatives[self.output].clone(),
        })
    }
}

/// Solver-independent nonlinear model. An objective model has one scalar
/// expression; a least-squares model has one or more residual expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearModel {
    pub parameter_count: usize,
    pub bounds: Vec<Bound>,
    pub objective: Option<NonlinearExpression>,
    pub residuals: Vec<NonlinearExpression>,
}

impl NonlinearModel {
    pub fn objective(
        parameter_count: usize,
        bounds: Vec<Bound>,
        objective: NonlinearExpression,
    ) -> Self {
        Self {
            parameter_count,
            bounds,
            objective: Some(objective),
            residuals: Vec::new(),
        }
    }
    pub fn least_squares(
        parameter_count: usize,
        bounds: Vec<Bound>,
        residuals: Vec<NonlinearExpression>,
    ) -> Self {
        Self {
            parameter_count,
            bounds,
            objective: None,
            residuals,
        }
    }
    pub fn validate(&self) -> Result<(), OptimizeError> {
        if self.parameter_count == 0 || self.bounds.len() != self.parameter_count {
            return Err(invalid(
                "model must have a positive parameter count and one bound per parameter",
            ));
        }
        if self.objective.is_some() == !self.residuals.is_empty() {
            return Err(invalid(
                "model must contain either one objective or residual expressions",
            ));
        }
        for bound in &self.bounds {
            if bound.lower.is_some_and(|v| !v.is_finite())
                || bound.upper.is_some_and(|v| !v.is_finite())
                || matches!((bound.lower,bound.upper),(Some(l),Some(u)) if l>u)
            {
                return Err(invalid("parameter bounds are invalid"));
            }
        }
        if let Some(objective) = &self.objective {
            objective.validate(self.parameter_count)?;
        }
        for residual in &self.residuals {
            residual.validate(self.parameter_count)?;
        }
        Ok(())
    }
    pub fn evaluate_objective(
        &self,
        parameters: &[f64],
    ) -> Result<DifferentiableValue, OptimizeError> {
        self.validate()?;
        if parameters.len() != self.parameter_count {
            return Err(invalid("parameter vector length does not match the model"));
        }
        self.objective
            .as_ref()
            .ok_or_else(|| invalid("model does not contain an objective"))?
            .evaluate(parameters)
    }
    pub fn evaluate_residuals(
        &self,
        parameters: &[f64],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError> {
        self.validate()?;
        if parameters.len() != self.parameter_count {
            return Err(invalid("parameter vector length does not match the model"));
        }
        if self.residuals.is_empty() {
            return Err(invalid("model does not contain residuals"));
        }
        let values: Result<Vec<_>, _> = self
            .residuals
            .iter()
            .map(|r| r.evaluate(parameters))
            .collect();
        let values = values?;
        Ok((
            values.iter().map(|v| v.value).collect(),
            values.into_iter().map(|v| v.gradient).collect(),
        ))
    }
}

/// Solve an objective model without bounds. Models containing finite bounds
/// are rejected so constraints cannot be silently ignored.
pub fn minimize_model_lbfgs(
    model: &NonlinearModel,
    initial: &[f64],
    options: LbfgsOptions,
) -> Result<LbfgsResult, OptimizeError> {
    model.validate()?;
    if model
        .bounds
        .iter()
        .any(|b| b.lower.is_some() || b.upper.is_some())
    {
        return Err(invalid("unconstrained L-BFGS cannot ignore model bounds"));
    }
    minimize_lbfgs(initial, options, |x| {
        let value = model.evaluate_objective(x)?;
        Ok((value.value, value.gradient))
    })
}

/// Solve an objective model while enforcing its parameter bounds.
pub fn minimize_model_lbfgsb(
    model: &NonlinearModel,
    initial: &[f64],
    options: LbfgsOptions,
) -> Result<LbfgsResult, OptimizeError> {
    model.validate()?;
    minimize_lbfgsb(initial, &model.bounds, options, |x| {
        let value = model.evaluate_objective(x)?;
        Ok((value.value, value.gradient))
    })
}

/// Solve a residual model using its shared Jacobian representation.
pub fn solve_model_least_squares(
    model: &NonlinearModel,
    initial: &[f64],
    weights: &[f64],
    loss: RobustLoss,
    options: NonlinearLeastSquaresOptions,
) -> Result<NonlinearLeastSquaresResult, OptimizeError> {
    model.validate()?;
    nonlinear_least_squares_configured(initial, &model.bounds, weights, loss, options, |x| {
        model.evaluate_residuals(x)
    })
}

fn unary<V, D>(
    values: &[f64],
    derivatives: &[Vec<f64>],
    a: usize,
    value: V,
    derivative: D,
) -> (f64, Vec<f64>)
where
    V: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let factor = derivative(values[a]);
    (
        value(values[a]),
        derivatives[a].iter().map(|d| factor * d).collect(),
    )
}
fn binary<V, D>(
    values: &[f64],
    derivatives: &[Vec<f64>],
    a: usize,
    b: usize,
    value: V,
    derivative: D,
) -> (f64, Vec<f64>)
where
    V: Fn(f64, f64) -> f64,
    D: Fn(f64, f64) -> (f64, f64),
{
    let (da, db) = derivative(values[a], values[b]);
    (
        value(values[a], values[b]),
        derivatives[a]
            .iter()
            .zip(&derivatives[b])
            .map(|(x, y)| da * x + db * y)
            .collect(),
    )
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn evaluates_shared_value_gradient_and_jacobian() {
        let square = NonlinearExpression::new(
            vec![NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0)],
            1,
        );
        let shifted = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Constant(3.0),
                NonlinearNode::Subtract(0, 1),
            ],
            2,
        );
        let model = NonlinearModel::least_squares(1, vec![Bound::free()], vec![square, shifted]);
        let (r, j) = model.evaluate_residuals(&[2.0]).unwrap();
        assert_eq!(r, vec![4.0, -1.0]);
        assert_eq!(j, vec![vec![4.0], vec![1.0]]);
    }
    #[test]
    fn rejects_forward_references() {
        let expression = NonlinearExpression::new(vec![NonlinearNode::Add(0, 0)], 0);
        assert!(expression.validate(1).is_err());
    }

    #[test]
    fn shared_objective_runs_through_bounded_solver() {
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Constant(3.0),
                NonlinearNode::Subtract(0, 1),
                NonlinearNode::Powf(2, 2.0),
            ],
            3,
        );
        let model = NonlinearModel::objective(
            1,
            vec![Bound {
                lower: None,
                upper: Some(1.0),
            }],
            expression,
        );
        let result = minimize_model_lbfgsb(&model, &[0.0], Default::default()).unwrap();
        assert!((result.point[0] - 1.0).abs() < 1e-10);
        assert!(result.gradient_norm < 1e-10);
    }

    #[test]
    fn shared_residuals_run_through_least_squares_solver() {
        fn shifted(constant: f64) -> NonlinearExpression {
            NonlinearExpression::new(
                vec![
                    NonlinearNode::Parameter(0),
                    NonlinearNode::Constant(constant),
                    NonlinearNode::Subtract(0, 1),
                ],
                2,
            )
        }
        let model =
            NonlinearModel::least_squares(1, vec![Bound::free()], vec![shifted(1.0), shifted(2.0)]);
        let result =
            solve_model_least_squares(&model, &[0.0], &[], RobustLoss::Squared, Default::default())
                .unwrap();
        assert!((result.point[0] - 1.5).abs() < 1e-8);
    }
}
