//! Equilibrated active-set KKT sensitivity analysis for nonlinear programs.

use crate::{
    Bound, ConstrainedNonlinearProblem, ConstrainedResult, ConstraintMultiplier, OptimizeError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearActiveSide {
    Equality,
    Lower,
    Upper,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonlinearActiveConstraint {
    pub constraint: Option<usize>,
    pub variable: Option<usize>,
    pub side: NonlinearActiveSide,
}

#[derive(Debug, Clone)]
pub struct NonlinearScalingOptions {
    pub active_tolerance: f64,
    pub minimum_factor: f64,
    pub maximum_factor: f64,
    pub regularization: f64,
    pub variable_factors: Vec<f64>,
}
impl Default for NonlinearScalingOptions {
    fn default() -> Self {
        Self {
            active_tolerance: 1e-6,
            minimum_factor: 1e-8,
            maximum_factor: 1e8,
            regularization: 1e-10,
            variable_factors: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearScalingReport {
    pub variable_factors: Vec<f64>,
    pub constraint_factors: Vec<f64>,
    pub active_set: Vec<NonlinearActiveConstraint>,
    pub unscaled_jacobian_norm_range: (f64, f64),
    pub scaled_jacobian_norm_range: (f64, f64),
    pub pivot_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct NonlinearSensitivityRequest {
    pub objective_gradient_perturbation: Vec<f64>,
    pub constraint_bound_perturbation: Vec<f64>,
    pub variable_bound_perturbation: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NonlinearSensitivityResult {
    pub point_derivative: Vec<f64>,
    pub constraint_multiplier_derivative: Vec<ConstraintMultiplier>,
    pub variable_lower_multiplier_derivative: Vec<f64>,
    pub variable_upper_multiplier_derivative: Vec<f64>,
    pub residual_norm: f64,
    pub scaling: NonlinearScalingReport,
}

pub fn analyze_nonlinear_result_sensitivity(
    problem: &ConstrainedNonlinearProblem,
    result: &ConstrainedResult,
    request: &NonlinearSensitivityRequest,
    options: &NonlinearScalingOptions,
) -> Result<NonlinearSensitivityResult, OptimizeError> {
    analyze_nonlinear_sensitivity(
        problem,
        &result.point,
        &result.multipliers,
        request,
        options,
    )
}

pub fn analyze_nonlinear_sensitivity(
    problem: &ConstrainedNonlinearProblem,
    point: &[f64],
    multipliers: &[ConstraintMultiplier],
    request: &NonlinearSensitivityRequest,
    options: &NonlinearScalingOptions,
) -> Result<NonlinearSensitivityResult, OptimizeError> {
    problem.validate()?;
    let n = problem.model.parameter_count;
    let m = problem.constraints.len();
    if point.len() != n
        || multipliers.len() != m
        || request.objective_gradient_perturbation.len() != n
        || request.constraint_bound_perturbation.len() != m
        || request.variable_bound_perturbation.len() != n
        || !point
            .iter()
            .chain(&request.objective_gradient_perturbation)
            .chain(&request.constraint_bound_perturbation)
            .chain(&request.variable_bound_perturbation)
            .all(|value| value.is_finite())
        || !options.active_tolerance.is_finite()
        || options.active_tolerance <= 0.0
        || !options.minimum_factor.is_finite()
        || options.minimum_factor <= 0.0
        || !options.maximum_factor.is_finite()
        || options.maximum_factor < options.minimum_factor
        || !options.regularization.is_finite()
        || options.regularization < 0.0
        || (!options.variable_factors.is_empty() && options.variable_factors.len() != n)
        || !options
            .variable_factors
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid nonlinear scaling or sensitivity dimensions",
        ));
    }
    let values = problem.evaluate_constraints(point)?;
    let variable_factors = if options.variable_factors.is_empty() {
        automatic_variable_factors(&problem.model.bounds, point, options)
    } else {
        options.variable_factors.clone()
    };
    let mut rows = Vec::new();
    let mut bound_rhs = Vec::new();
    let mut active_set = Vec::new();
    for index in 0..m {
        if let Some(side) = active_side(
            &problem.constraints[index].bound,
            values[index].value,
            options.active_tolerance,
        ) {
            rows.push(values[index].gradient.clone());
            bound_rhs.push(request.constraint_bound_perturbation[index]);
            active_set.push(NonlinearActiveConstraint {
                constraint: Some(index),
                variable: None,
                side,
            });
        }
    }
    for index in 0..n {
        if let Some(side) = active_side(
            &problem.model.bounds[index],
            point[index],
            options.active_tolerance,
        ) {
            let mut row = vec![0.0; n];
            row[index] = 1.0;
            rows.push(row);
            bound_rhs.push(request.variable_bound_perturbation[index]);
            active_set.push(NonlinearActiveConstraint {
                constraint: None,
                variable: Some(index),
                side,
            });
        }
    }
    let weights: Vec<_> = multipliers.iter().enumerate().map(|(index, multiplier)| {
        let bound = &problem.constraints[index].bound;
        if matches!((bound.lower,bound.upper),(Some(lower),Some(upper)) if (upper-lower).abs() <= options.active_tolerance) {
            multiplier.equality
        } else { multiplier.upper - multiplier.lower }
    }).collect();
    let mut hessian = problem.lagrangian_hessian(point, &weights)?;
    for (index, row) in hessian.iter_mut().enumerate() {
        row[index] += options.regularization;
    }
    let unscaled_norms: Vec<_> = rows.iter().map(|row| row_norm(row)).collect();
    let row_factors: Vec<_> = rows
        .iter()
        .map(|row| {
            let scaled: Vec<_> = row
                .iter()
                .zip(&variable_factors)
                .map(|(a, b)| a * b)
                .collect();
            (1.0 / row_norm(&scaled).max(1e-300))
                .clamp(options.minimum_factor, options.maximum_factor)
        })
        .collect();
    let size = n + rows.len();
    let mut matrix = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];
    for i in 0..n {
        rhs[i] = -variable_factors[i] * request.objective_gradient_perturbation[i];
        for j in 0..n {
            matrix[i][j] = variable_factors[i] * hessian[i][j] * variable_factors[j];
        }
    }
    for row in 0..rows.len() {
        rhs[n + row] = row_factors[row] * bound_rhs[row];
        for column in 0..n {
            let value = row_factors[row] * rows[row][column] * variable_factors[column];
            matrix[column][n + row] = value;
            matrix[n + row][column] = value;
        }
    }
    let solved = solve_pivoted(matrix.clone(), rhs.clone()).ok_or_else(|| {
        OptimizeError::InvalidProblem("active nonlinear KKT system is singular".to_owned())
    })?;
    let point_derivative: Vec<_> = solved[..n]
        .iter()
        .zip(&variable_factors)
        .map(|(a, b)| a * b)
        .collect();
    let active_dual: Vec<_> = (0..rows.len())
        .map(|row| row_factors[row] * solved[n + row])
        .collect();
    let mut constraint_derivative = vec![ConstraintMultiplier::default(); m];
    let mut lower_derivative = vec![0.0; n];
    let mut upper_derivative = vec![0.0; n];
    for (index, active) in active_set.iter().enumerate() {
        let dual = active_dual[index];
        if let Some(constraint) = active.constraint {
            match active.side {
                NonlinearActiveSide::Equality => constraint_derivative[constraint].equality = dual,
                NonlinearActiveSide::Lower => constraint_derivative[constraint].lower = -dual,
                NonlinearActiveSide::Upper => constraint_derivative[constraint].upper = dual,
            }
        } else if let Some(variable) = active.variable {
            match active.side {
                NonlinearActiveSide::Lower => lower_derivative[variable] = -dual,
                _ => upper_derivative[variable] = dual,
            }
        }
    }
    let residual_norm = matvec(&matrix, &solved)
        .iter()
        .zip(&rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let scaled_norms: Vec<_> = rows
        .iter()
        .zip(&row_factors)
        .map(|(row, factor)| {
            let scaled: Vec<_> = row
                .iter()
                .zip(&variable_factors)
                .map(|(a, b)| a * b)
                .collect();
            row_norm(&scaled) * factor
        })
        .collect();
    Ok(NonlinearSensitivityResult {
        point_derivative,
        constraint_multiplier_derivative: constraint_derivative,
        variable_lower_multiplier_derivative: lower_derivative,
        variable_upper_multiplier_derivative: upper_derivative,
        residual_norm,
        scaling: NonlinearScalingReport {
            variable_factors,
            constraint_factors: row_factors,
            active_set,
            unscaled_jacobian_norm_range: value_range(&unscaled_norms),
            scaled_jacobian_norm_range: value_range(&scaled_norms),
            pivot_ratio: pivot_ratio(matrix),
        },
    })
}

fn active_side(bound: &Bound, value: f64, tolerance: f64) -> Option<NonlinearActiveSide> {
    if matches!((bound.lower,bound.upper),(Some(lower),Some(upper)) if (upper-lower).abs() <= tolerance)
    {
        Some(NonlinearActiveSide::Equality)
    } else if bound
        .lower
        .is_some_and(|lower| (value - lower).abs() <= tolerance)
    {
        Some(NonlinearActiveSide::Lower)
    } else if bound
        .upper
        .is_some_and(|upper| (value - upper).abs() <= tolerance)
    {
        Some(NonlinearActiveSide::Upper)
    } else {
        None
    }
}
fn automatic_variable_factors(
    bounds: &[Bound],
    point: &[f64],
    options: &NonlinearScalingOptions,
) -> Vec<f64> {
    point
        .iter()
        .zip(bounds)
        .map(|(&value, bound)| {
            let width = match (bound.lower, bound.upper) {
                (Some(lower), Some(upper)) => upper - lower,
                _ => 0.0,
            };
            value
                .abs()
                .max(width)
                .max(1.0)
                .clamp(options.minimum_factor, options.maximum_factor)
        })
        .collect()
}
fn row_norm(row: &[f64]) -> f64 {
    row.iter().map(|value| value * value).sum::<f64>().sqrt()
}
fn value_range(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        (0.0, 0.0)
    } else {
        (
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}
fn matvec(matrix: &[Vec<f64>], point: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(point).map(|(a, b)| a * b).sum())
        .collect()
}
fn pivot_ratio(mut matrix: Vec<Vec<f64>>) -> f64 {
    let mut minimum = f64::INFINITY;
    let mut maximum: f64 = 0.0;
    for column in 0..matrix.len() {
        let pivot = (column..matrix.len())
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        let value = matrix[pivot][column].abs();
        if value <= 1e-14 {
            return f64::INFINITY;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        matrix.swap(column, pivot);
        for row in (column + 1)..matrix.len() {
            let factor = matrix[row][column] / matrix[column][column];
            for index in column..matrix.len() {
                matrix[row][index] -= factor * matrix[column][index];
            }
        }
    }
    maximum / minimum
}
fn solve_pivoted(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        if matrix[pivot][column].abs() <= 1e-14 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in (column + 1)..n {
            let factor = matrix[row][column] / matrix[column][column];
            for index in column..n {
                matrix[row][index] -= factor * matrix[column][index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = ((row + 1)..n)
            .map(|column| matrix[row][column] * result[column])
            .sum();
        result[row] = (rhs[row] - tail) / matrix[row][row];
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearConstraint, NonlinearExpression, NonlinearModel, NonlinearNode::*};

    #[test]
    fn equality_right_hand_side_sensitivity_is_exact() {
        let objective = NonlinearExpression::new(
            vec![
                Parameter(0),
                Powf(0, 2.0),
                Parameter(1),
                Powf(2, 2.0),
                Add(1, 3),
            ],
            4,
        );
        let equality = NonlinearExpression::new(vec![Parameter(0), Parameter(1), Add(0, 1)], 2);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(2, vec![Bound::free(), Bound::free()], objective),
            constraints: vec![NonlinearConstraint {
                expression: equality,
                bound: Bound::fixed(1.0),
            }],
        };
        let result = analyze_nonlinear_sensitivity(
            &problem,
            &[0.5, 0.5],
            &[ConstraintMultiplier {
                equality: -1.0,
                ..Default::default()
            }],
            &NonlinearSensitivityRequest {
                objective_gradient_perturbation: vec![0.0, 0.0],
                constraint_bound_perturbation: vec![1.0],
                variable_bound_perturbation: vec![0.0, 0.0],
            },
            &Default::default(),
        )
        .unwrap();
        assert!((result.point_derivative[0] - 0.5).abs() < 1e-9);
        assert!((result.point_derivative[1] - 0.5).abs() < 1e-9);
        assert!(result.residual_norm < 1e-10);
    }

    #[test]
    fn explicit_scaling_equilibrates_active_row() {
        let objective = NonlinearExpression::new(vec![Parameter(0), Powf(0, 2.0)], 1);
        let row = NonlinearExpression::new(vec![Parameter(0)], 0);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(1, vec![Bound::free()], objective),
            constraints: vec![NonlinearConstraint {
                expression: row,
                bound: Bound::fixed(2.0),
            }],
        };
        let options = NonlinearScalingOptions {
            variable_factors: vec![1_000.0],
            ..Default::default()
        };
        let result = analyze_nonlinear_sensitivity(
            &problem,
            &[2.0],
            &[ConstraintMultiplier {
                equality: -4.0,
                ..Default::default()
            }],
            &NonlinearSensitivityRequest {
                objective_gradient_perturbation: vec![0.0],
                constraint_bound_perturbation: vec![1.0],
                variable_bound_perturbation: vec![0.0],
            },
            &options,
        )
        .unwrap();
        assert_eq!(result.scaling.constraint_factors, vec![0.001]);
        assert!((result.scaling.scaled_jacobian_norm_range.0 - 1.0).abs() < 1e-12);
        assert!((result.point_derivative[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn active_variable_bound_sensitivity_has_consistent_multiplier_sign() {
        let objective = NonlinearExpression::new(
            vec![
                Parameter(0),
                Powf(0, 2.0),
                Constant(-4.0),
                Multiply(0, 2),
                Add(1, 3),
            ],
            4,
        );
        let row = NonlinearExpression::new(vec![Parameter(0)], 0);
        let problem = ConstrainedNonlinearProblem {
            model: NonlinearModel::objective(
                1,
                vec![Bound {
                    lower: None,
                    upper: Some(1.0),
                }],
                objective,
            ),
            constraints: vec![NonlinearConstraint {
                expression: row,
                bound: Bound {
                    lower: None,
                    upper: Some(10.0),
                },
            }],
        };
        let result = analyze_nonlinear_sensitivity(
            &problem,
            &[1.0],
            &[Default::default()],
            &NonlinearSensitivityRequest {
                objective_gradient_perturbation: vec![0.0],
                constraint_bound_perturbation: vec![0.0],
                variable_bound_perturbation: vec![1.0],
            },
            &Default::default(),
        )
        .unwrap();
        assert!((result.point_derivative[0] - 1.0).abs() < 1e-12);
        assert!((result.variable_upper_multiplier_derivative[0] + 2.0).abs() < 1e-9);
        assert_eq!(result.scaling.active_set[0].variable, Some(0));
    }
}
