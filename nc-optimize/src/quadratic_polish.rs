//! Active-set polishing for converged convex quadratic programs.

use crate::{OptimizeError, QuadraticProblem, QuadraticResult, QuadraticTermination};

#[derive(Debug, Clone)]
pub struct QuadraticPolishResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub attempted: bool,
    pub accepted: bool,
    pub active_constraints: usize,
    pub objective_improvement: f64,
    pub maximum_row_violation: f64,
    pub maximum_variable_violation: f64,
}

/// Solves the equality-constrained KKT system for the active set inferred at `result`.
pub fn polish_convex_qp(
    problem: &QuadraticProblem,
    result: &QuadraticResult,
    active_tolerance: f64,
) -> Result<QuadraticPolishResult, OptimizeError> {
    problem.validate(1e-10)?;
    if !active_tolerance.is_finite() || active_tolerance <= 0.0 {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid QP polish tolerance",
        ));
    }
    if result.termination != QuadraticTermination::Converged
        || result.point.len() != problem.linear.len()
    {
        return Ok(make_result(
            problem,
            result.point.clone(),
            result.objective,
            false,
            false,
            0,
            result.objective,
        ));
    }
    let n = problem.linear.len();
    let mut dense = vec![vec![0.0; n]; problem.row_bounds.len()];
    for (row, column, value) in problem.constraints.iter_entries() {
        dense[row][column] += value;
    }
    let activity = matvec(&dense, &result.point);
    let mut active = Vec::new();
    let mut targets = Vec::new();
    for row in 0..dense.len() {
        let target = problem.row_bounds[row]
            .lower
            .filter(|value| (activity[row] - value).abs() <= active_tolerance)
            .or_else(|| {
                problem.row_bounds[row]
                    .upper
                    .filter(|value| (activity[row] - value).abs() <= active_tolerance)
            });
        if let Some(target) = target {
            active.push(dense[row].clone());
            targets.push(target);
        }
    }
    for column in 0..n {
        let target = problem.variable_bounds[column]
            .lower
            .filter(|value| (result.point[column] - value).abs() <= active_tolerance)
            .or_else(|| {
                problem.variable_bounds[column]
                    .upper
                    .filter(|value| (result.point[column] - value).abs() <= active_tolerance)
            });
        if let Some(target) = target {
            let mut row = vec![0.0; n];
            row[column] = 1.0;
            active.push(row);
            targets.push(target);
        }
    }
    let Some(candidate) = solve_kkt(problem, &active, &targets) else {
        return Ok(make_result(
            problem,
            result.point.clone(),
            result.objective,
            true,
            false,
            active.len(),
            result.objective,
        ));
    };
    let row_violation = maximum_violation(&matvec(&dense, &candidate), &problem.row_bounds);
    let variable_violation = maximum_violation(&candidate, &problem.variable_bounds);
    let objective = problem.objective(&candidate);
    let accepted = candidate.iter().all(|value| value.is_finite())
        && row_violation <= active_tolerance
        && variable_violation <= active_tolerance
        && objective <= result.objective + active_tolerance;
    Ok(make_result(
        problem,
        if accepted {
            candidate
        } else {
            result.point.clone()
        },
        if accepted {
            objective
        } else {
            result.objective
        },
        true,
        accepted,
        active.len(),
        result.objective,
    ))
}

fn make_result(
    problem: &QuadraticProblem,
    point: Vec<f64>,
    objective: f64,
    attempted: bool,
    accepted: bool,
    active_constraints: usize,
    original_objective: f64,
) -> QuadraticPolishResult {
    let activity = problem.constraints.spmv(&point).unwrap_or_default();
    QuadraticPolishResult {
        maximum_row_violation: maximum_violation(&activity, &problem.row_bounds),
        maximum_variable_violation: maximum_violation(&point, &problem.variable_bounds),
        objective_improvement: original_objective - objective,
        point,
        objective,
        attempted,
        accepted,
        active_constraints,
    }
}

fn solve_kkt(problem: &QuadraticProblem, active: &[Vec<f64>], targets: &[f64]) -> Option<Vec<f64>> {
    if active.is_empty() {
        return None;
    }
    let n = problem.linear.len();
    let size = n + active.len();
    let mut matrix = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];
    for row in 0..n {
        rhs[row] = -problem.linear[row];
        matrix[row][..n].copy_from_slice(&problem.quadratic[row]);
    }
    for (row, coefficients) in active.iter().enumerate() {
        rhs[n + row] = targets[row];
        for column in 0..n {
            matrix[column][n + row] = coefficients[column];
            matrix[n + row][column] = coefficients[column];
        }
    }
    solve_pivoted(matrix, rhs).map(|solution| solution[..n].to_vec())
}

fn solve_pivoted(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for column in 0..n {
        let pivot = (column..n).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= 1e-12 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in (column + 1)..n {
            let scale = matrix[row][column] / matrix[column][column];
            for index in column..n {
                matrix[row][index] -= scale * matrix[column][index];
            }
            rhs[row] -= scale * rhs[column];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = ((row + 1)..n)
            .map(|column| matrix[row][column] * solution[column])
            .sum();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

fn matvec(matrix: &[Vec<f64>], point: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(point).map(|(a, b)| a * b).sum())
        .collect()
}

fn maximum_violation(values: &[f64], bounds: &[crate::Bound]) -> f64 {
    values
        .iter()
        .zip(bounds)
        .fold(0.0_f64, |maximum, (&value, bound)| {
            maximum
                .max(bound.lower.map_or(0.0, |lower| (lower - value).max(0.0)))
                .max(bound.upper.map_or(0.0, |upper| (value - upper).max(0.0)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{solve_convex_qp, Bound};
    use nc_sparse::CsrMatrix;

    #[test]
    fn polishes_equality_constrained_qp() {
        let problem = QuadraticProblem {
            quadratic: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            linear: vec![0.0, 0.0],
            constant: 0.0,
            constraints: CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0]).unwrap(),
            row_bounds: vec![Bound::fixed(1.0)],
            variable_bounds: vec![Bound::free(), Bound::free()],
        };
        let solved = solve_convex_qp(&problem, Default::default()).unwrap();
        let polished = polish_convex_qp(&problem, &solved, 1e-6).unwrap();
        assert!(polished.attempted && polished.accepted);
        assert!((polished.point[0] - 0.5).abs() < 1e-10);
        assert!((polished.point[1] - 0.5).abs() < 1e-10);
        assert!(polished.objective <= solved.objective + 1e-6);
        assert!(polished.maximum_row_violation <= 1e-10);
    }
}
