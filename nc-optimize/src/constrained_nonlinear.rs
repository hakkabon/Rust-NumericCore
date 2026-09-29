//! Augmented-Lagrangian foundation for graph-represented nonlinear constraints.

use crate::{
    minimize_lbfgsb, Bound, LbfgsOptions, NonlinearExpression, NonlinearModel, OptimizeError,
};

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearConstraint {
    pub expression: NonlinearExpression,
    pub bound: Bound,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstrainedNonlinearProblem {
    pub model: NonlinearModel,
    pub constraints: Vec<NonlinearConstraint>,
}

impl ConstrainedNonlinearProblem {
    pub fn validate(&self) -> Result<(), OptimizeError> {
        self.model.validate()?;
        if self.model.objective.is_none() || self.constraints.is_empty() {
            return Err(invalid(
                "constrained problem requires an objective and constraints",
            ));
        }
        for constraint in &self.constraints {
            constraint.expression.validate(self.model.parameter_count)?;
            if constraint.bound.lower.is_none() && constraint.bound.upper.is_none() {
                return Err(invalid(
                    "a nonlinear constraint must have at least one bound",
                ));
            }
            if constraint.bound.lower.is_some_and(|v| !v.is_finite())
                || constraint.bound.upper.is_some_and(|v| !v.is_finite())
                || matches!((constraint.bound.lower,constraint.bound.upper),(Some(l),Some(u)) if l>u)
            {
                return Err(invalid("nonlinear constraint bounds are invalid"));
            }
        }
        Ok(())
    }

    pub fn evaluate_constraints(
        &self,
        point: &[f64],
    ) -> Result<Vec<crate::DifferentiableValue>, OptimizeError> {
        self.validate()?;
        if point.len() != self.model.parameter_count {
            return Err(invalid(
                "parameter vector length does not match the problem",
            ));
        }
        self.constraints
            .iter()
            .map(|c| c.expression.evaluate(point))
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstrainedOptions {
    pub max_outer_iterations: usize,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
    pub initial_penalty: f64,
    pub penalty_increase: f64,
    pub maximum_penalty: f64,
    pub inner_options: LbfgsOptions,
}

impl Default for ConstrainedOptions {
    fn default() -> Self {
        Self {
            max_outer_iterations: 30,
            feasibility_tolerance: 1e-7,
            stationarity_tolerance: 1e-6,
            initial_penalty: 10.0,
            penalty_increase: 10.0,
            maximum_penalty: 1e10,
            inner_options: LbfgsOptions {
                max_iterations: 300,
                ..Default::default()
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrainedTermination {
    Converged,
    IterationLimit,
    PenaltyLimit,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintMultiplier {
    pub lower: f64,
    pub upper: f64,
    pub equality: f64,
}

impl Default for ConstraintMultiplier {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: 0.0,
            equality: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstrainedResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    pub multipliers: Vec<ConstraintMultiplier>,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub outer_iterations: usize,
    pub inner_iterations: usize,
    pub evaluations: usize,
    pub final_penalty: f64,
    pub termination: ConstrainedTermination,
}

#[derive(Debug, Clone, Copy)]
pub struct ConstrainedIteration {
    pub iteration: usize,
    pub objective: f64,
    pub maximum_violation: f64,
    pub stationarity_norm: f64,
    pub penalty: f64,
}

pub fn minimize_constrained(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: ConstrainedOptions,
) -> Result<ConstrainedResult, OptimizeError> {
    minimize_constrained_with_observer(problem, initial, options, |_| true)
}

pub fn minimize_constrained_with_observer<O>(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: ConstrainedOptions,
    mut observer: O,
) -> Result<ConstrainedResult, OptimizeError>
where
    O: FnMut(ConstrainedIteration) -> bool,
{
    problem.validate()?;
    validate_options(options)?;
    if initial.len() != problem.model.parameter_count || !initial.iter().all(|v| v.is_finite()) {
        return Err(invalid("initial point must match the constrained problem"));
    }
    let mut point = initial.to_vec();
    let mut multipliers = vec![ConstraintMultiplier::default(); problem.constraints.len()];
    let mut penalty = options.initial_penalty;
    let mut previous_violation = f64::INFINITY;
    let mut inner_iterations = 0;
    let mut evaluations = 0;

    for outer in 0..options.max_outer_iterations {
        let inner = minimize_lbfgsb(&point, &problem.model.bounds, options.inner_options, |x| {
            evaluations += 1;
            augmented_value(problem, x, &multipliers, penalty)
        })?;
        inner_iterations += inner.iterations;
        point = inner.point;
        let objective = problem.model.evaluate_objective(&point)?;
        let values = problem.evaluate_constraints(&point)?;
        update_multipliers(problem, &values, &mut multipliers, penalty);
        let violation = maximum_violation(problem, &values);
        let stationarity =
            stationarity_norm(problem, &point, &objective.gradient, &values, &multipliers);
        let snapshot = ConstrainedIteration {
            iteration: outer + 1,
            objective: objective.value,
            maximum_violation: violation,
            stationarity_norm: stationarity,
            penalty,
        };
        if !observer(snapshot) {
            return Ok(make_result(
                problem,
                point,
                objective.value,
                values,
                multipliers,
                violation,
                stationarity,
                outer + 1,
                inner_iterations,
                evaluations,
                penalty,
                ConstrainedTermination::Cancelled,
            ));
        }
        if violation <= options.feasibility_tolerance
            && stationarity <= options.stationarity_tolerance
        {
            return Ok(make_result(
                problem,
                point,
                objective.value,
                values,
                multipliers,
                violation,
                stationarity,
                outer + 1,
                inner_iterations,
                evaluations,
                penalty,
                ConstrainedTermination::Converged,
            ));
        }
        if violation > 0.25 * previous_violation {
            penalty *= options.penalty_increase;
            if !penalty.is_finite() || penalty > options.maximum_penalty {
                return Ok(make_result(
                    problem,
                    point,
                    objective.value,
                    values,
                    multipliers,
                    violation,
                    stationarity,
                    outer + 1,
                    inner_iterations,
                    evaluations,
                    penalty,
                    ConstrainedTermination::PenaltyLimit,
                ));
            }
        }
        previous_violation = violation;
    }
    let objective = problem.model.evaluate_objective(&point)?;
    let values = problem.evaluate_constraints(&point)?;
    let violation = maximum_violation(problem, &values);
    let stationarity =
        stationarity_norm(problem, &point, &objective.gradient, &values, &multipliers);
    Ok(make_result(
        problem,
        point,
        objective.value,
        values,
        multipliers,
        violation,
        stationarity,
        options.max_outer_iterations,
        inner_iterations,
        evaluations,
        penalty,
        ConstrainedTermination::IterationLimit,
    ))
}

fn augmented_value(
    problem: &ConstrainedNonlinearProblem,
    x: &[f64],
    multipliers: &[ConstraintMultiplier],
    penalty: f64,
) -> Result<(f64, Vec<f64>), OptimizeError> {
    let objective = problem.model.evaluate_objective(x)?;
    let values = problem.evaluate_constraints(x)?;
    let mut value = objective.value;
    let mut gradient = objective.gradient;
    for ((constraint, evaluated), lambda) in
        problem.constraints.iter().zip(&values).zip(multipliers)
    {
        if let (Some(lower), Some(upper)) = (constraint.bound.lower, constraint.bound.upper) {
            if lower == upper {
                let h = evaluated.value - lower;
                value += lambda.equality * h + 0.5 * penalty * h * h;
                add_scaled(
                    &mut gradient,
                    &evaluated.gradient,
                    lambda.equality + penalty * h,
                );
                continue;
            }
        }
        if let Some(lower) = constraint.bound.lower {
            let g = lower - evaluated.value;
            let shifted = (lambda.lower + penalty * g).max(0.0);
            value += (shifted * shifted - lambda.lower * lambda.lower) / (2.0 * penalty);
            add_scaled(&mut gradient, &evaluated.gradient, -shifted);
        }
        if let Some(upper) = constraint.bound.upper {
            let g = evaluated.value - upper;
            let shifted = (lambda.upper + penalty * g).max(0.0);
            value += (shifted * shifted - lambda.upper * lambda.upper) / (2.0 * penalty);
            add_scaled(&mut gradient, &evaluated.gradient, shifted);
        }
    }
    Ok((value, gradient))
}
fn update_multipliers(
    problem: &ConstrainedNonlinearProblem,
    values: &[crate::DifferentiableValue],
    multipliers: &mut [ConstraintMultiplier],
    penalty: f64,
) {
    for ((c, v), m) in problem.constraints.iter().zip(values).zip(multipliers) {
        if let (Some(l), Some(u)) = (c.bound.lower, c.bound.upper) {
            if l == u {
                m.equality += penalty * (v.value - l);
                continue;
            }
        }
        if let Some(l) = c.bound.lower {
            m.lower = (m.lower + penalty * (l - v.value)).max(0.0)
        }
        if let Some(u) = c.bound.upper {
            m.upper = (m.upper + penalty * (v.value - u)).max(0.0)
        }
    }
}
fn maximum_violation(
    problem: &ConstrainedNonlinearProblem,
    values: &[crate::DifferentiableValue],
) -> f64 {
    problem
        .constraints
        .iter()
        .zip(values)
        .fold(0.0, |maximum, (c, v)| {
            let lower = c.bound.lower.map_or(0.0, |l| (l - v.value).max(0.0));
            let upper = c.bound.upper.map_or(0.0, |u| (v.value - u).max(0.0));
            maximum.max(lower).max(upper)
        })
}
fn stationarity_norm(
    problem: &ConstrainedNonlinearProblem,
    point: &[f64],
    objective_gradient: &[f64],
    values: &[crate::DifferentiableValue],
    multipliers: &[ConstraintMultiplier],
) -> f64 {
    let mut g = objective_gradient.to_vec();
    for ((c, v), m) in problem.constraints.iter().zip(values).zip(multipliers) {
        if matches!((c.bound.lower,c.bound.upper),(Some(l),Some(u)) if l==u) {
            add_scaled(&mut g, &v.gradient, m.equality)
        } else {
            add_scaled(&mut g, &v.gradient, m.upper - m.lower)
        }
    }
    for ((value, bound), gradient) in point.iter().zip(&problem.model.bounds).zip(&mut g) {
        if (bound.lower.is_some_and(|l| *value <= l) && *gradient > 0.0)
            || (bound.upper.is_some_and(|u| *value >= u) && *gradient < 0.0)
        {
            *gradient = 0.0
        }
    }
    g.iter().fold(0.0, |m, v| m.max(v.abs()))
}
fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    for (t, s) in target.iter_mut().zip(source) {
        *t += scale * s
    }
}
fn make_result(
    problem: &ConstrainedNonlinearProblem,
    point: Vec<f64>,
    objective: f64,
    values: Vec<crate::DifferentiableValue>,
    multipliers: Vec<ConstraintMultiplier>,
    maximum_violation: f64,
    stationarity_norm: f64,
    outer_iterations: usize,
    inner_iterations: usize,
    evaluations: usize,
    final_penalty: f64,
    termination: ConstrainedTermination,
) -> ConstrainedResult {
    debug_assert_eq!(values.len(), problem.constraints.len());
    ConstrainedResult {
        point,
        objective,
        constraint_values: values.into_iter().map(|v| v.value).collect(),
        multipliers,
        maximum_violation,
        stationarity_norm,
        outer_iterations,
        inner_iterations,
        evaluations,
        final_penalty,
        termination,
    }
}
fn validate_options(o: ConstrainedOptions) -> Result<(), OptimizeError> {
    if o.max_outer_iterations == 0
        || !o.feasibility_tolerance.is_finite()
        || o.feasibility_tolerance <= 0.0
        || !o.stationarity_tolerance.is_finite()
        || o.stationarity_tolerance <= 0.0
        || !o.initial_penalty.is_finite()
        || o.initial_penalty <= 0.0
        || !o.penalty_increase.is_finite()
        || o.penalty_increase <= 1.0
        || !o.maximum_penalty.is_finite()
        || o.maximum_penalty < o.initial_penalty
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid constrained nonlinear options",
        ));
    }
    Ok(())
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NonlinearNode;
    use crate::NonlinearNode::*;
    fn expression(nodes: Vec<NonlinearNode>) -> NonlinearExpression {
        let output = nodes.len() - 1;
        NonlinearExpression::new(nodes, output)
    }
    #[test]
    fn solves_equality_constrained_quadratic() {
        // min x²+y² subject to x+y=1 => (0.5,0.5)
        let objective = expression(vec![
            Parameter(0),
            Powf(0, 2.0),
            Parameter(1),
            Powf(2, 2.0),
            Add(1, 3),
        ]);
        let equality = expression(vec![Parameter(0), Parameter(1), Add(0, 1)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(2, vec![Bound::free(), Bound::free()], objective),
            constraints: vec![NonlinearConstraint {
                expression: equality,
                bound: Bound::fixed(1.0),
            }],
        };
        let result = minimize_constrained(&problem, &[0.0, 0.0], Default::default()).unwrap();
        assert_eq!(result.termination, ConstrainedTermination::Converged);
        assert!((result.point[0] - 0.5).abs() < 1e-5 && (result.point[1] - 0.5).abs() < 1e-5);
        assert!(result.maximum_violation < 1e-7);
    }
    #[test]
    fn solves_active_nonlinear_inequality() {
        // min (x-2)² subject to x² <= 1 => x=1
        let objective = expression(vec![
            Parameter(0),
            Constant(2.0),
            Subtract(0, 1),
            Powf(2, 2.0),
        ]);
        let constraint = expression(vec![Parameter(0), Powf(0, 2.0)]);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(
                1,
                vec![Bound {
                    lower: Some(0.0),
                    upper: None,
                }],
                objective,
            ),
            constraints: vec![NonlinearConstraint {
                expression: constraint,
                bound: Bound {
                    lower: None,
                    upper: Some(1.0),
                },
            }],
        };
        let result = minimize_constrained(&problem, &[0.5], Default::default()).unwrap();
        assert!((result.point[0] - 1.0).abs() < 1e-5);
        assert!(result.maximum_violation < 1e-7 && result.multipliers[0].upper > 0.0);
    }
}
