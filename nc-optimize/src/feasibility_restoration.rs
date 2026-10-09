//! Phase-I feasibility restoration for graph-represented nonlinear programs.

use crate::{
    minimize_lbfgsb, Bound, ConstrainedNonlinearProblem, LbfgsOptions, NonlinearTermination,
    OptimizeError,
};

#[derive(Debug, Clone, Copy)]
pub struct FeasibilityRestorationOptions {
    pub max_iterations: usize,
    pub feasibility_tolerance: f64,
    /// Positive margins produce a strictly interior point for inequalities.
    pub interior_margin: f64,
}

impl Default for FeasibilityRestorationOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            feasibility_tolerance: 1e-7,
            interior_margin: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeasibilityRestorationTermination {
    AlreadyFeasible,
    Converged,
    IterationLimit,
    Stalled,
}

#[derive(Debug, Clone)]
pub struct FeasibilityRestorationResult {
    pub point: Vec<f64>,
    pub maximum_violation: f64,
    pub squared_violation: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: FeasibilityRestorationTermination,
}

pub fn restore_feasibility(
    problem: &ConstrainedNonlinearProblem,
    initial: &[f64],
    options: FeasibilityRestorationOptions,
) -> Result<FeasibilityRestorationResult, OptimizeError> {
    problem.validate()?;
    validate_options(options)?;
    if initial.len() != problem.model.parameter_count || !initial.iter().all(|x| x.is_finite()) {
        return Err(OptimizeError::InvalidProblem(
            "restoration initial point must match the constrained problem".into(),
        ));
    }
    let bounds = restoration_bounds(&problem.model.bounds, options.interior_margin)?;
    let start = project(initial, &bounds);
    let (initial_merit, _) = violation_value_gradient(problem, &start, options.interior_margin)?;
    let initial_violation = maximum_violation(problem, &start, options.interior_margin)?;
    if initial_violation <= options.feasibility_tolerance {
        return Ok(FeasibilityRestorationResult {
            point: start,
            maximum_violation: initial_violation,
            squared_violation: initial_merit,
            iterations: 0,
            evaluations: 1,
            termination: FeasibilityRestorationTermination::AlreadyFeasible,
        });
    }

    let inner = LbfgsOptions {
        max_iterations: options.max_iterations,
        gradient_tolerance: options.feasibility_tolerance,
        objective_tolerance: options.feasibility_tolerance * options.feasibility_tolerance,
        ..LbfgsOptions::default()
    };
    let solved = minimize_lbfgsb(&start, &bounds, inner, |point| {
        violation_value_gradient(problem, point, options.interior_margin)
    })?;
    let violation = maximum_violation(problem, &solved.point, options.interior_margin)?;
    let termination = if violation <= options.feasibility_tolerance {
        FeasibilityRestorationTermination::Converged
    } else if solved.termination == NonlinearTermination::IterationLimit {
        FeasibilityRestorationTermination::IterationLimit
    } else {
        FeasibilityRestorationTermination::Stalled
    };
    Ok(FeasibilityRestorationResult {
        point: solved.point,
        maximum_violation: violation,
        squared_violation: solved.objective,
        iterations: solved.iterations,
        evaluations: solved.evaluations,
        termination,
    })
}

fn violation_value_gradient(
    problem: &ConstrainedNonlinearProblem,
    point: &[f64],
    margin: f64,
) -> Result<(f64, Vec<f64>), OptimizeError> {
    let values = problem.evaluate_constraints(point)?;
    let mut merit = 0.0;
    let mut gradient = vec![0.0; point.len()];
    for (constraint, value) in problem.constraints.iter().zip(values) {
        let equality = matches!((constraint.bound.lower, constraint.bound.upper),
            (Some(lower), Some(upper)) if lower == upper);
        if equality {
            let residual = value.value - constraint.bound.lower.unwrap();
            merit += 0.5 * residual * residual;
            add_scaled(&mut gradient, &value.gradient, residual);
        } else {
            if let Some(lower) = constraint.bound.lower {
                let residual = (lower + margin - value.value).max(0.0);
                merit += 0.5 * residual * residual;
                add_scaled(&mut gradient, &value.gradient, -residual);
            }
            if let Some(upper) = constraint.bound.upper {
                let residual = (value.value - (upper - margin)).max(0.0);
                merit += 0.5 * residual * residual;
                add_scaled(&mut gradient, &value.gradient, residual);
            }
        }
    }
    Ok((merit, gradient))
}

fn maximum_violation(
    problem: &ConstrainedNonlinearProblem,
    point: &[f64],
    margin: f64,
) -> Result<f64, OptimizeError> {
    let values = problem.evaluate_constraints(point)?;
    Ok(problem.constraints.iter().zip(values).fold(0.0_f64, |maximum, (constraint, value)| {
        let equality = matches!((constraint.bound.lower, constraint.bound.upper),
            (Some(lower), Some(upper)) if lower == upper);
        if equality {
            maximum.max((value.value - constraint.bound.lower.unwrap()).abs())
        } else {
            maximum
                .max(constraint.bound.lower.map_or(0.0, |lower| (lower + margin - value.value).max(0.0)))
                .max(constraint.bound.upper.map_or(0.0, |upper| (value.value - (upper - margin)).max(0.0)))
        }
    }))
}

fn restoration_bounds(bounds: &[Bound], margin: f64) -> Result<Vec<Bound>, OptimizeError> {
    bounds.iter().map(|bound| {
        let lower = bound.lower.map(|value| value + margin);
        let upper = bound.upper.map(|value| value - margin);
        if matches!((lower, upper), (Some(lower), Some(upper)) if lower > upper) {
            return Err(OptimizeError::InvalidProblem(
                "variable bounds have no restoration interior".into(),
            ));
        }
        Ok(Bound { lower, upper })
    }).collect()
}

fn project(point: &[f64], bounds: &[Bound]) -> Vec<f64> {
    point.iter().zip(bounds).map(|(&value, bound)| {
        value.max(bound.lower.unwrap_or(f64::NEG_INFINITY))
            .min(bound.upper.unwrap_or(f64::INFINITY))
    }).collect()
}

fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += scale * source;
    }
}

fn validate_options(options: FeasibilityRestorationOptions) -> Result<(), OptimizeError> {
    if options.max_iterations == 0
        || !options.feasibility_tolerance.is_finite()
        || options.feasibility_tolerance <= 0.0
        || !options.interior_margin.is_finite()
        || options.interior_margin < 0.0
    {
        return Err(OptimizeError::InvalidConfiguration("invalid restoration options"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearConstraint, NonlinearExpression, NonlinearModel, NonlinearNode};

    #[test]
    fn restores_an_infeasible_nonlinear_inequality() {
        let square = NonlinearExpression::new(vec![
            NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0),
        ], 1);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel {
                parameter_count: 1,
                bounds: vec![Bound { lower: Some(-2.0), upper: Some(2.0) }],
                objective: Some(square.clone()), residuals: vec![],
            },
            constraints: vec![NonlinearConstraint {
                expression: square,
                bound: Bound { lower: Some(1.0), upper: None },
            }],
        };
        let result = restore_feasibility(&problem, &[0.2], FeasibilityRestorationOptions::default()).unwrap();
        assert!(result.maximum_violation <= 1e-7);
        assert!(result.point[0].abs() >= 1.0 - 1e-7);
    }
}
