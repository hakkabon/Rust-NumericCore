//! Feasibility-based bound tightening for factorable nonlinear graphs.

use crate::{
    analyze_expression, Bound, Interval, NonlinearConstraint, NonlinearExpression, NonlinearModel,
    NonlinearNode, OptimizeError,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearPropagationOptions {
    pub max_passes: usize,
    pub tolerance: f64,
}

impl Default for NonlinearPropagationOptions {
    fn default() -> Self {
        Self {
            max_passes: 8,
            tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearPropagationResult {
    pub bounds: Vec<Bound>,
    pub passes: usize,
    pub bounds_tightened: usize,
    pub infeasible: bool,
}

/// Repeatedly applies forward interval evaluation and conservative backward
/// contraction. An objective cutoff, when supplied, is treated as `f(x) <= cutoff`.
pub fn propagate_nonlinear_bounds(
    model: &NonlinearModel,
    constraints: &[NonlinearConstraint],
    initial_bounds: &[Bound],
    objective_cutoff: Option<f64>,
    options: NonlinearPropagationOptions,
) -> Result<NonlinearPropagationResult, OptimizeError> {
    model.validate()?;
    if initial_bounds.len() != model.parameter_count
        || options.max_passes == 0
        || !options.tolerance.is_finite()
        || options.tolerance < 0.0
    {
        return Err(invalid("invalid nonlinear propagation configuration"));
    }
    let mut bounds = initial_bounds.to_vec();
    let mut tightened = 0;
    for pass in 0..options.max_passes {
        let before = tightened;
        if let (Some(expression), Some(cutoff)) = (&model.objective, objective_cutoff) {
            if !contract_expression(
                expression,
                Bound {
                    lower: None,
                    upper: Some(cutoff),
                },
                &mut bounds,
                options.tolerance,
                &mut tightened,
            )? {
                return Ok(result(bounds, pass + 1, tightened, true));
            }
        }
        for constraint in constraints {
            constraint.expression.validate(model.parameter_count)?;
            if !contract_expression(
                &constraint.expression,
                constraint.bound,
                &mut bounds,
                options.tolerance,
                &mut tightened,
            )? {
                return Ok(result(bounds, pass + 1, tightened, true));
            }
        }
        if tightened == before {
            return Ok(result(bounds, pass + 1, tightened, false));
        }
    }
    Ok(result(bounds, options.max_passes, tightened, false))
}

fn result(
    bounds: Vec<Bound>,
    passes: usize,
    bounds_tightened: usize,
    infeasible: bool,
) -> NonlinearPropagationResult {
    NonlinearPropagationResult {
        bounds,
        passes,
        bounds_tightened,
        infeasible,
    }
}

fn contract_expression(
    expression: &NonlinearExpression,
    target: Bound,
    bounds: &mut [Bound],
    tolerance: f64,
    tightened: &mut usize,
) -> Result<bool, OptimizeError> {
    let analysis = analyze_expression(expression, bounds)?;
    let mut intervals: Vec<Interval> = analysis.nodes.iter().map(|x| x.interval).collect();
    let target = interval(target);
    if !intersect(&mut intervals[expression.output], target) {
        return Ok(false);
    }
    for index in (0..expression.nodes.len()).rev() {
        let z = intervals[index];
        let ok = match expression.nodes[index] {
            NonlinearNode::Constant(value) => contains(z, value),
            NonlinearNode::Parameter(parameter) => {
                let old = interval(bounds[parameter]);
                let mut next = old;
                if !intersect(&mut next, z) {
                    false
                } else {
                    if changed(old, next, tolerance) {
                        bounds[parameter] = bound(next);
                        *tightened += endpoint_changes(old, next);
                    }
                    true
                }
            }
            NonlinearNode::Add(a, b) => {
                let (x, y) = (intervals[a], intervals[b]);
                intersect(&mut intervals[a], sub(z, y)) && intersect(&mut intervals[b], sub(z, x))
            }
            NonlinearNode::Subtract(a, b) => {
                let (x, y) = (intervals[a], intervals[b]);
                intersect(&mut intervals[a], add(z, y)) && intersect(&mut intervals[b], sub(x, z))
            }
            NonlinearNode::Negate(a) => intersect(
                &mut intervals[a],
                Interval {
                    lower: -z.upper,
                    upper: -z.lower,
                },
            ),
            NonlinearNode::Multiply(a, b) if a == b => {
                if z.upper < 0.0 {
                    false
                } else {
                    let radius = z.upper.max(0.0).sqrt();
                    intersect(
                        &mut intervals[a],
                        Interval {
                            lower: -radius,
                            upper: radius,
                        },
                    )
                }
            }
            NonlinearNode::Multiply(a, b) => {
                let (x, y) = (intervals[a], intervals[b]);
                (!excludes_zero(y) || intersect(&mut intervals[a], divide(z, y)))
                    && (!excludes_zero(x) || intersect(&mut intervals[b], divide(z, x)))
            }
            NonlinearNode::Divide(a, b) => {
                let (x, y) = (intervals[a], intervals[b]);
                intersect(&mut intervals[a], multiply(z, y))
                    && (!excludes_zero(z) || intersect(&mut intervals[b], divide(x, z)))
            }
            NonlinearNode::Exp(a) => {
                z.upper > 0.0
                    && intersect(
                        &mut intervals[a],
                        Interval {
                            lower: if z.lower > 0.0 {
                                z.lower.ln()
                            } else {
                                f64::NEG_INFINITY
                            },
                            upper: z.upper.ln(),
                        },
                    )
            }
            NonlinearNode::Log(a) => intersect(
                &mut intervals[a],
                Interval {
                    lower: z.lower.exp(),
                    upper: z.upper.exp(),
                },
            ),
            NonlinearNode::Sqrt(a) => {
                let mut positive_z = z;
                intersect(
                    &mut positive_z,
                    Interval {
                        lower: 0.0,
                        upper: f64::INFINITY,
                    },
                ) && intersect(
                    &mut intervals[a],
                    Interval {
                        lower: positive_z.lower * positive_z.lower,
                        upper: positive_z.upper * positive_z.upper,
                    },
                )
            }
            NonlinearNode::Powf(a, exponent) if exponent == 1.0 => intersect(&mut intervals[a], z),
            NonlinearNode::Powf(a, exponent) if exponent == 2.0 => {
                if z.upper < 0.0 {
                    false
                } else {
                    let radius = z.upper.max(0.0).sqrt();
                    intersect(
                        &mut intervals[a],
                        Interval {
                            lower: -radius,
                            upper: radius,
                        },
                    )
                }
            }
            _ => true,
        };
        if !ok {
            return Ok(false);
        }
    }
    Ok(true)
}

fn interval(value: Bound) -> Interval {
    Interval {
        lower: value.lower.unwrap_or(f64::NEG_INFINITY),
        upper: value.upper.unwrap_or(f64::INFINITY),
    }
}
fn bound(value: Interval) -> Bound {
    Bound {
        lower: value.lower.is_finite().then_some(value.lower),
        upper: value.upper.is_finite().then_some(value.upper),
    }
}
fn intersect(value: &mut Interval, other: Interval) -> bool {
    value.lower = value.lower.max(other.lower);
    value.upper = value.upper.min(other.upper);
    value.lower <= value.upper && !value.lower.is_nan() && !value.upper.is_nan()
}
fn contains(value: Interval, point: f64) -> bool {
    value.lower <= point && point <= value.upper
}
fn add(a: Interval, b: Interval) -> Interval {
    Interval {
        lower: a.lower + b.lower,
        upper: a.upper + b.upper,
    }
}
fn sub(a: Interval, b: Interval) -> Interval {
    Interval {
        lower: a.lower - b.upper,
        upper: a.upper - b.lower,
    }
}
fn multiply(a: Interval, b: Interval) -> Interval {
    let values = [
        a.lower * b.lower,
        a.lower * b.upper,
        a.upper * b.lower,
        a.upper * b.upper,
    ];
    if values.iter().any(|x| x.is_nan()) {
        return Interval::whole();
    }
    Interval {
        lower: values.iter().copied().fold(f64::INFINITY, f64::min),
        upper: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}
fn divide(a: Interval, b: Interval) -> Interval {
    if !excludes_zero(b) {
        Interval::whole()
    } else {
        multiply(
            a,
            Interval {
                lower: 1.0 / b.upper,
                upper: 1.0 / b.lower,
            },
        )
    }
}
fn excludes_zero(value: Interval) -> bool {
    value.upper < 0.0 || value.lower > 0.0
}
fn changed(a: Interval, b: Interval, tolerance: f64) -> bool {
    b.lower > a.lower + tolerance || b.upper < a.upper - tolerance
}
fn endpoint_changes(a: Interval, b: Interval) -> usize {
    usize::from(b.lower > a.lower) + usize::from(b.upper < a.upper)
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_model() -> NonlinearModel {
        NonlinearModel::objective(
            1,
            vec![Bound {
                lower: Some(-10.0),
                upper: Some(10.0),
            }],
            NonlinearExpression::new(
                vec![NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0)],
                1,
            ),
        )
    }

    #[test]
    fn square_constraint_tightens_both_sides() {
        let model = square_model();
        let constraint = NonlinearConstraint {
            expression: model.objective.clone().unwrap(),
            bound: Bound {
                lower: None,
                upper: Some(4.0),
            },
        };
        let result = propagate_nonlinear_bounds(
            &model,
            &[constraint],
            &model.bounds,
            None,
            NonlinearPropagationOptions::default(),
        )
        .unwrap();
        assert_eq!(
            result.bounds[0],
            Bound {
                lower: Some(-2.0),
                upper: Some(2.0)
            }
        );
        assert!(!result.infeasible && result.bounds_tightened == 2);
    }

    #[test]
    fn detects_impossible_square_constraint() {
        let model = square_model();
        let constraint = NonlinearConstraint {
            expression: model.objective.clone().unwrap(),
            bound: Bound {
                lower: None,
                upper: Some(-1.0),
            },
        };
        assert!(
            propagate_nonlinear_bounds(
                &model,
                &[constraint],
                &model.bounds,
                None,
                NonlinearPropagationOptions::default()
            )
            .unwrap()
            .infeasible
        );
    }
}
