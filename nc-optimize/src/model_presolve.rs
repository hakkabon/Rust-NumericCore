//! Reversible sparse LP/MILP presolve and postsolve reconstruction.

use crate::{Bound, OptimizeError, Problem};
use nc_sparse::CooMatrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresolveOptions {
    pub max_passes: usize,
    pub tolerance: f64,
}
impl Default for PresolveOptions {
    fn default() -> Self {
        Self {
            max_passes: 8,
            tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresolveStatus {
    Reduced,
    Solved,
    Infeasible,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresolveReport {
    pub passes: usize,
    pub fixed_variables: usize,
    pub removed_rows: usize,
    pub tightened_bounds: usize,
    pub original_nonzeros: usize,
    pub reduced_nonzeros: usize,
}

#[derive(Debug, Clone)]
pub struct PresolvedProblem {
    pub problem: Option<Problem>,
    pub status: PresolveStatus,
    pub active_variables: Vec<usize>,
    pub fixed_values: Vec<Option<f64>>,
    pub kept_rows: Vec<usize>,
    pub objective_offset: f64,
    pub report: PresolveReport,
}

impl PresolvedProblem {
    pub fn restore_point(&self, reduced: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        if reduced.len() != self.active_variables.len() {
            return Err(invalid("postsolve point dimension mismatch"));
        }
        let mut point = vec![0.0; self.fixed_values.len()];
        for (i, value) in self.fixed_values.iter().enumerate() {
            if let Some(value) = value {
                point[i] = *value;
            }
        }
        for (i, &original) in self.active_variables.iter().enumerate() {
            point[original] = reduced[i];
        }
        Ok(point)
    }
    pub fn restore_objective(&self, reduced_objective: f64) -> f64 {
        reduced_objective + self.objective_offset
    }
}

pub fn presolve_problem(
    problem: &Problem,
    options: PresolveOptions,
) -> Result<PresolvedProblem, OptimizeError> {
    problem.validate()?;
    if options.max_passes == 0 || !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(invalid("invalid presolve options"));
    }
    let n = problem.objective.len();
    let mut bounds = problem.var_bounds.clone();
    let mut removed = vec![false; problem.row_bounds.len()];
    let mut tightened = 0;
    let mut passes = 0;
    let mut infeasible = false;
    for pass in 0..options.max_passes {
        passes = pass + 1;
        let before = tightened;
        for row in 0..problem.row_bounds.len() {
            if removed[row] {
                continue;
            }
            let entries: Vec<_> = problem
                .constraints
                .iter_entries()
                .filter(|x| x.0 == row)
                .collect();
            let (minimum, maximum) = activity_interval(&entries, &bounds);
            let target = problem.row_bounds[row];
            if target
                .lower
                .is_some_and(|x| maximum < x - options.tolerance)
                || target
                    .upper
                    .is_some_and(|x| minimum > x + options.tolerance)
            {
                infeasible = true;
                break;
            }
            if target
                .lower
                .is_none_or(|x| minimum >= x - options.tolerance)
                && target
                    .upper
                    .is_none_or(|x| maximum <= x + options.tolerance)
            {
                removed[row] = true;
                continue;
            }
            let mut free = None;
            let mut fixed_sum = 0.0;
            let mut multiple = false;
            for &(_, column, value) in &entries {
                if let Some(x) = fixed(bounds[column], options.tolerance) {
                    fixed_sum += value * x
                } else if free.is_none() {
                    free = Some((column, value))
                } else {
                    multiple = true;
                    break;
                }
            }
            if !multiple {
                if let Some((column, coefficient)) = free {
                    let mut next = bounds[column];
                    if let Some(lower) = target.lower {
                        let value = (lower - fixed_sum) / coefficient;
                        if coefficient > 0.0 {
                            next.lower = Some(next.lower.unwrap_or(f64::NEG_INFINITY).max(value))
                        } else {
                            next.upper = Some(next.upper.unwrap_or(f64::INFINITY).min(value))
                        }
                    }
                    if let Some(upper) = target.upper {
                        let value = (upper - fixed_sum) / coefficient;
                        if coefficient > 0.0 {
                            next.upper = Some(next.upper.unwrap_or(f64::INFINITY).min(value))
                        } else {
                            next.lower = Some(next.lower.unwrap_or(f64::NEG_INFINITY).max(value))
                        }
                    }
                    if problem.is_integer[column] {
                        next.lower = next.lower.map(|x| (x - options.tolerance).ceil());
                        next.upper = next.upper.map(|x| (x + options.tolerance).floor());
                    }
                    if next.lower.unwrap_or(f64::NEG_INFINITY)
                        > next.upper.unwrap_or(f64::INFINITY) + options.tolerance
                    {
                        infeasible = true;
                        break;
                    }
                    if next != bounds[column] {
                        tightened += usize::from(next.lower != bounds[column].lower)
                            + usize::from(next.upper != bounds[column].upper);
                        bounds[column] = next;
                    }
                } else if free.is_none()
                    && (target
                        .lower
                        .is_some_and(|x| fixed_sum < x - options.tolerance)
                        || target
                            .upper
                            .is_some_and(|x| fixed_sum > x + options.tolerance))
                {
                    infeasible = true;
                    break;
                }
            }
        }
        if infeasible || tightened == before {
            break;
        }
    }
    if infeasible {
        return Ok(PresolvedProblem {
            problem: None,
            status: PresolveStatus::Infeasible,
            active_variables: vec![],
            fixed_values: vec![None; n],
            kept_rows: vec![],
            objective_offset: 0.0,
            report: PresolveReport {
                passes,
                fixed_variables: 0,
                removed_rows: 0,
                tightened_bounds: tightened,
                original_nonzeros: problem.constraints.nnz(),
                reduced_nonzeros: 0,
            },
        });
    }
    let fixed_values: Vec<_> = bounds
        .iter()
        .map(|b| fixed(*b, options.tolerance))
        .collect();
    let active: Vec<_> = (0..n).filter(|&i| fixed_values[i].is_none()).collect();
    let mut remap = vec![None; n];
    for (i, &x) in active.iter().enumerate() {
        remap[x] = Some(i)
    }
    let mut coo = CooMatrix::new(problem.row_bounds.len(), active.len());
    let mut shifted = vec![0.0; problem.row_bounds.len()];
    for (row, column, value) in problem.constraints.iter_entries() {
        if let Some(x) = fixed_values[column] {
            shifted[row] += value * x
        } else {
            coo.push(row, remap[column].unwrap(), value)?
        }
    }
    let full = coo.to_csr()?;
    let mut kept = Vec::new();
    // Rebuild with only nonredundant rows after final bound propagation.
    let mut rows_data = Vec::new();
    let mut row_bounds = Vec::new();
    for row in 0..problem.row_bounds.len() {
        if removed[row] {
            continue;
        }
        let entries: Vec<_> = full.iter_entries().filter(|x| x.0 == row).collect();
        let adjusted = Bound {
            lower: problem.row_bounds[row].lower.map(|x| x - shifted[row]),
            upper: problem.row_bounds[row].upper.map(|x| x - shifted[row]),
        };
        let (lo, hi) = activity_interval(
            &entries,
            &active.iter().map(|&i| bounds[i]).collect::<Vec<_>>(),
        );
        if adjusted.lower.is_none_or(|x| lo >= x - options.tolerance)
            && adjusted.upper.is_none_or(|x| hi <= x + options.tolerance)
        {
            continue;
        }
        kept.push(row);
        row_bounds.push(adjusted);
        rows_data.push(entries);
    }
    let mut final_coo = CooMatrix::new(rows_data.len(), active.len());
    for (new_row, entries) in rows_data.iter().enumerate() {
        for &(_, column, value) in entries {
            final_coo.push(new_row, column, value)?
        }
    }
    let matrix = final_coo.to_csr()?;
    let objective_offset = (0..n)
        .filter_map(|i| fixed_values[i].map(|x| problem.objective[i] * x))
        .sum();
    let reduced = if active.is_empty() {
        None
    } else {
        Some(Problem {
            objective: active.iter().map(|&i| problem.objective[i]).collect(),
            constraints: matrix.clone(),
            row_bounds,
            var_bounds: active.iter().map(|&i| bounds[i]).collect(),
            is_integer: active.iter().map(|&i| problem.is_integer[i]).collect(),
        })
    };
    Ok(PresolvedProblem {
        problem: reduced,
        status: if active.is_empty() {
            PresolveStatus::Solved
        } else {
            PresolveStatus::Reduced
        },
        active_variables: active,
        fixed_values,
        kept_rows: kept,
        objective_offset,
        report: PresolveReport {
            passes,
            fixed_variables: n - remap.iter().filter(|x| x.is_some()).count(),
            removed_rows: problem.row_bounds.len() - rows_data.len(),
            tightened_bounds: tightened,
            original_nonzeros: problem.constraints.nnz(),
            reduced_nonzeros: matrix.nnz(),
        },
    })
}

fn fixed(bound: Bound, tolerance: f64) -> Option<f64> {
    match (bound.lower, bound.upper) {
        (Some(a), Some(b)) if (b - a).abs() <= tolerance => Some(0.5 * (a + b)),
        _ => None,
    }
}
fn activity_interval(entries: &[(usize, usize, f64)], bounds: &[Bound]) -> (f64, f64) {
    let (mut lo, mut hi) = (0.0, 0.0);
    for &(_, column, a) in entries {
        let b = bounds[column];
        let (x, y) = if a >= 0.0 {
            (
                b.lower.unwrap_or(f64::NEG_INFINITY),
                b.upper.unwrap_or(f64::INFINITY),
            )
        } else {
            (
                b.upper.unwrap_or(f64::INFINITY),
                b.lower.unwrap_or(f64::NEG_INFINITY),
            )
        };
        lo += a * x;
        hi += a * y
    }
    (lo, hi)
}
fn invalid(s: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(s.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_sparse::CsrMatrix;
    #[test]
    fn fixes_tightens_removes_and_restores() {
        let p = Problem {
            objective: vec![2.0, 1.0],
            constraints: CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 1.0, 1.0])
                .unwrap(),
            row_bounds: vec![
                Bound {
                    lower: Some(3.0),
                    upper: Some(3.0),
                },
                Bound {
                    lower: None,
                    upper: Some(10.0),
                },
            ],
            var_bounds: vec![
                Bound::fixed(1.0),
                Bound {
                    lower: Some(0.0),
                    upper: Some(10.0),
                },
            ],
            is_integer: vec![false, false],
        };
        let r = presolve_problem(&p, PresolveOptions::default()).unwrap();
        assert_eq!(r.status, PresolveStatus::Solved);
        assert_eq!(r.restore_point(&[]).unwrap(), vec![1.0, 2.0]);
        assert_eq!(r.restore_objective(0.0), 4.0);
        assert_eq!(r.report.fixed_variables, 2);
    }
    #[test]
    fn proves_row_infeasible() {
        let p = Problem {
            objective: vec![1.0],
            constraints: CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
            row_bounds: vec![Bound {
                lower: Some(2.0),
                upper: None,
            }],
            var_bounds: vec![Bound {
                lower: Some(0.0),
                upper: Some(1.0),
            }],
            is_integer: vec![false],
        };
        assert_eq!(
            presolve_problem(&p, PresolveOptions::default())
                .unwrap()
                .status,
            PresolveStatus::Infeasible
        );
    }
}
