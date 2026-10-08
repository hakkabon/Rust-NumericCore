//! Reversible diagonal equilibration for linear optimization problems.
//!
//! Scaling is kept outside individual algorithms so simplex, interior point,
//! and branch-and-bound all see exactly the same transformed model.  The
//! transformation uses `x = D z` and positive row multipliers `R`, preserving
//! feasibility, integrality, and the objective while improving coefficient
//! balance.

use crate::{Bound, OptimizeError, Problem, Solution, Solver};
use nc_sparse::CooMatrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingOptions {
    /// Number of alternating row/column max-norm equilibration passes.
    pub iterations: usize,
    /// Smallest/largest factor permitted in one accumulated scaling.
    pub minimum_factor: f64,
    pub maximum_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RevisedSimplexSolver, SolveStatus};

    #[test]
    fn equilibrates_and_restores_an_ill_scaled_lp() {
        let mut coo = CooMatrix::new(1, 2);
        coo.push(0, 0, 1e-6).unwrap();
        coo.push(0, 1, 1e6).unwrap();
        let problem = Problem {
            objective: vec![-1.0, 0.0],
            constraints: coo.to_csr().unwrap(),
            row_bounds: vec![Bound { lower: None, upper: Some(2e-6) }],
            var_bounds: vec![Bound { lower: Some(0.0), upper: None }, Bound::fixed(0.0)],
            is_integer: vec![false, false],
        };
        let scaled = ScaledProblem::new(&problem, ScalingOptions::default()).unwrap();
        assert_ne!(scaled.report.variable_factors[0], 1.0);
        let solution = ScaledSolver {
            inner: RevisedSimplexSolver::default(),
            options: ScalingOptions::default(),
        }
        .solve(&problem)
        .unwrap();
        assert_eq!(solution.status, SolveStatus::Optimal);
        assert!((solution.variable_values[0] - 2.0).abs() < 1e-7, "{:?}", solution);
        assert!((solution.objective_value + 2.0).abs() < 1e-7);
    }

    #[test]
    fn integer_columns_keep_unit_variable_scaling() {
        let mut coo = CooMatrix::new(1, 1);
        coo.push(0, 0, 1e9).unwrap();
        let problem = Problem {
            objective: vec![1.0],
            constraints: coo.to_csr().unwrap(),
            row_bounds: vec![Bound { lower: None, upper: Some(1e9) }],
            var_bounds: vec![Bound { lower: Some(0.0), upper: None }],
            is_integer: vec![true],
        };
        let scaled = ScaledProblem::new(&problem, ScalingOptions::default()).unwrap();
        assert_eq!(scaled.report.variable_factors, vec![1.0]);
    }
}

impl Default for ScalingOptions {
    fn default() -> Self {
        Self { iterations: 3, minimum_factor: 1e-8, maximum_factor: 1e8 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalingReport {
    pub row_factors: Vec<f64>,
    pub variable_factors: Vec<f64>,
}

/// A solver decorator applying reversible max-norm equilibration.
pub struct ScaledSolver<S> {
    pub inner: S,
    pub options: ScalingOptions,
}

impl<S: Solver> Solver for ScaledSolver<S> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn solve(&self, problem: &Problem) -> Result<Solution, OptimizeError> {
        let scaled = ScaledProblem::new(problem, self.options)?;
        let solution = self.inner.solve(&scaled.problem)?;
        Ok(scaled.restore(solution, problem))
    }
}

pub struct ScaledProblem {
    pub problem: Problem,
    pub report: ScalingReport,
}

impl ScaledProblem {
    pub fn new(original: &Problem, options: ScalingOptions) -> Result<Self, OptimizeError> {
        original.validate()?;
        if options.iterations == 0
            || !options.minimum_factor.is_finite()
            || !options.maximum_factor.is_finite()
            || options.minimum_factor <= 0.0
            || options.minimum_factor > options.maximum_factor
        {
            return Err(OptimizeError::InvalidConfiguration(
                "scaling requires iterations > 0 and finite 0 < minimum_factor <= maximum_factor",
            ));
        }

        let m = original.constraints.rows();
        let n = original.constraints.cols();
        let mut entries: Vec<(usize, usize, f64)> = original.constraints.iter_entries().collect();
        let mut row_factors = vec![1.0; m];
        let mut variable_factors = vec![1.0; n];

        for _ in 0..options.iterations {
            let mut row_norm = vec![0.0_f64; m];
            for &(row, _, value) in &entries {
                row_norm[row] = row_norm[row].max(value.abs());
            }
            let row_step: Vec<f64> = row_norm
                .into_iter()
                .enumerate()
                .map(|(row, norm)| {
                    if norm == 0.0 {
                        1.0
                    } else {
                        let requested = 1.0 / norm.sqrt();
                        (row_factors[row] * requested)
                            .clamp(options.minimum_factor, options.maximum_factor)
                            / row_factors[row]
                    }
                })
                .collect();
            for entry in &mut entries {
                entry.2 *= row_step[entry.0];
            }
            for i in 0..m {
                row_factors[i] *= row_step[i];
            }

            let mut column_norm = vec![0.0_f64; n];
            for &(_, column, value) in &entries {
                column_norm[column] = column_norm[column].max(value.abs());
            }
            let column_step: Vec<f64> = column_norm
                .into_iter()
                .enumerate()
                .map(|(column, norm)| {
                    // A non-unit change of variables would turn an integer lattice
                    // into a scaled lattice. Keep integer columns in source units.
                    if original.is_integer[column] || norm == 0.0 {
                        1.0
                    } else {
                        let requested = 1.0 / norm.sqrt();
                        (variable_factors[column] * requested)
                            .clamp(options.minimum_factor, options.maximum_factor)
                            / variable_factors[column]
                    }
                })
                .collect();
            for entry in &mut entries {
                entry.2 *= column_step[entry.1];
            }
            for j in 0..n {
                variable_factors[j] *= column_step[j];
            }
        }

        let mut coo = CooMatrix::with_capacity(m, n, entries.len());
        for (row, column, value) in entries {
            coo.push(row, column, value)?;
        }
        let scale_bound = |bound: Bound, factor: f64| Bound {
            lower: bound.lower.map(|v| v * factor),
            upper: bound.upper.map(|v| v * factor),
        };
        let problem = Problem {
            objective: original
                .objective
                .iter()
                .zip(&variable_factors)
                .map(|(c, d)| c * d)
                .collect(),
            constraints: coo.to_csr()?,
            row_bounds: original
                .row_bounds
                .iter()
                .zip(&row_factors)
                .map(|(&b, &r)| scale_bound(b, r))
                .collect(),
            var_bounds: original
                .var_bounds
                .iter()
                .zip(&variable_factors)
                .map(|(&b, &d)| scale_bound(b, 1.0 / d))
                .collect(),
            is_integer: original.is_integer.clone(),
        };
        Ok(Self { problem, report: ScalingReport { row_factors, variable_factors } })
    }

    pub fn restore(&self, mut solution: Solution, original: &Problem) -> Solution {
        if solution.variable_values.len() == self.report.variable_factors.len() {
            for (value, factor) in
                solution.variable_values.iter_mut().zip(&self.report.variable_factors)
            {
                *value *= factor;
            }
            solution.objective_value = original
                .objective
                .iter()
                .zip(&solution.variable_values)
                .map(|(coefficient, value)| coefficient * value)
                .sum();
        }
        solution
    }
}
