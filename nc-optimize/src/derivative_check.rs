//! Scale-aware central-difference checks for nonlinear callbacks.

use crate::OptimizeError;

#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeCheckReport {
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
    pub worst_index: (usize, usize),
    pub passed: bool,
    pub evaluations: usize,
}

pub fn check_gradient<F>(
    point: &[f64],
    analytic: &[f64],
    relative_step: f64,
    tolerance: f64,
    mut objective: F,
) -> Result<DerivativeCheckReport, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    if point.is_empty() || analytic.len() != point.len() {
        return Err(OptimizeError::InvalidProblem(
            "gradient check dimensions do not agree".into(),
        ));
    }
    let numerical = central_difference(point, 1, relative_step, |x| Ok(vec![objective(x)?]))?;
    Ok(compare(
        &[analytic.to_vec()],
        &numerical,
        tolerance,
        2 * point.len(),
    ))
}

pub fn check_jacobian<F>(
    point: &[f64],
    analytic: &[Vec<f64>],
    relative_step: f64,
    tolerance: f64,
    mut residuals: F,
) -> Result<DerivativeCheckReport, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    if point.is_empty()
        || analytic.is_empty()
        || analytic.iter().any(|row| row.len() != point.len())
    {
        return Err(OptimizeError::InvalidProblem(
            "Jacobian check dimensions do not agree".into(),
        ));
    }
    let rows = analytic.len();
    let numerical = central_difference(point, rows, relative_step, |x| residuals(x))?;
    Ok(compare(analytic, &numerical, tolerance, 2 * point.len()))
}

fn central_difference<F>(
    point: &[f64],
    rows: usize,
    relative_step: f64,
    mut values: F,
) -> Result<Vec<Vec<f64>>, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    if !relative_step.is_finite() || relative_step <= 0.0 {
        return Err(OptimizeError::InvalidConfiguration(
            "derivative-check step must be finite and positive",
        ));
    }
    let mut derivative = vec![vec![0.0; point.len()]; rows];
    for column in 0..point.len() {
        let step = relative_step * point[column].abs().max(1.0);
        let mut plus = point.to_vec();
        let mut minus = point.to_vec();
        plus[column] += step;
        minus[column] -= step;
        let high = values(&plus)?;
        let low = values(&minus)?;
        if high.len() != rows
            || low.len() != rows
            || !high.iter().chain(&low).all(|v| v.is_finite())
        {
            return Err(OptimizeError::InvalidProblem(
                "finite-difference callback shape changed or returned non-finite values".into(),
            ));
        }
        for row in 0..rows {
            derivative[row][column] = (high[row] - low[row]) / (2.0 * step);
        }
    }
    Ok(derivative)
}

fn compare(
    analytic: &[Vec<f64>],
    numerical: &[Vec<f64>],
    tolerance: f64,
    evaluations: usize,
) -> DerivativeCheckReport {
    let mut max_abs: f64 = 0.0;
    let mut max_rel: f64 = 0.0;
    let mut worst = (0, 0);
    for row in 0..analytic.len() {
        for column in 0..analytic[row].len() {
            let absolute = (analytic[row][column] - numerical[row][column]).abs();
            let relative = absolute
                / analytic[row][column]
                    .abs()
                    .max(numerical[row][column].abs())
                    .max(1.0);
            max_abs = max_abs.max(absolute);
            if relative > max_rel {
                max_rel = relative;
                worst = (row, column);
            }
        }
    }
    DerivativeCheckReport {
        maximum_absolute_error: max_abs,
        maximum_relative_error: max_rel,
        worst_index: worst,
        passed: tolerance.is_finite() && tolerance >= 0.0 && max_rel <= tolerance,
        evaluations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_correct_and_incorrect_rosenbrock_gradients() {
        let point = [-1.2, 1.0];
        let value = |x: &[f64]| Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2));
        let good = check_gradient(&point, &[-215.6, -88.0], 1e-6, 1e-5, value).unwrap();
        assert!(good.passed);
        let bad = check_gradient(&point, &[215.6, -88.0], 1e-6, 1e-5, value).unwrap();
        assert!(!bad.passed);
        assert_eq!(bad.worst_index, (0, 0));
    }

    #[test]
    fn verifies_exponential_residual_jacobian() {
        let point = [2.0, -0.5];
        let xs = [0.0_f64, 0.5, 1.0];
        let jacobian: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| {
                let e = (point[1] * x).exp();
                vec![e, point[0] * x * e]
            })
            .collect();
        let report = check_jacobian(&point, &jacobian, 1e-6, 1e-5, |p| {
            Ok(xs.iter().map(|x| p[0] * (p[1] * x).exp()).collect())
        })
        .unwrap();
        assert!(report.passed, "{report:?}");
    }
}
