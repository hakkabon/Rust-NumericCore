//! Audited presolve/solve/postsolve orchestration for LP and MILP models.

use crate::{
    presolve_problem, BranchAndBoundSolver, InteriorPointSolver, OptimizeError, PresolveOptions,
    PresolveReport, PresolveStatus, Problem, RevisedSimplexSolver, ScaledSolver, ScalingOptions,
    Solution, SolutionDiagnostics, SolveStatus, Solver,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionLinearMethod {
    Automatic,
    Simplex,
    InteriorPoint,
    BranchAndBound,
}

#[derive(Debug, Clone, Copy)]
pub struct ProductionLinearOptions {
    pub method: ProductionLinearMethod,
    pub presolve: PresolveOptions,
    pub scaling: bool,
    pub verification_tolerance: f64,
}
impl Default for ProductionLinearOptions {
    fn default() -> Self {
        Self {
            method: ProductionLinearMethod::Automatic,
            presolve: Default::default(),
            scaling: true,
            verification_tolerance: 1e-7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProductionLinearResult {
    pub solution: Solution,
    pub diagnostics: SolutionDiagnostics,
    pub presolve_report: PresolveReport,
    pub original_variables: usize,
    pub reduced_variables: usize,
    pub used_milp: bool,
    pub verified: bool,
}

pub fn solve_production_linear(
    problem: &Problem,
    options: ProductionLinearOptions,
) -> Result<ProductionLinearResult, OptimizeError> {
    if !options.verification_tolerance.is_finite() || options.verification_tolerance < 0. {
        return Err(OptimizeError::InvalidConfiguration(
            "verification tolerance must be finite and nonnegative",
        ));
    }
    let reduced = presolve_problem(problem, options.presolve)?;
    let used_milp = match options.method {
        ProductionLinearMethod::BranchAndBound => true,
        ProductionLinearMethod::Automatic => problem.is_integer.iter().any(|&v| v),
        _ => false,
    };
    let mut solution = match reduced.status {
        PresolveStatus::Infeasible => Solution {
            variable_values: vec![],
            objective_value: 0.,
            status: SolveStatus::Infeasible,
        },
        PresolveStatus::Solved => {
            let point = reduced.restore_point(&[])?;
            Solution {
                objective_value: reduced.restore_objective(0.),
                variable_values: point,
                status: SolveStatus::Optimal,
            }
        }
        PresolveStatus::Reduced => {
            let model = reduced.problem.as_ref().unwrap();
            let solved = match options.method {
                ProductionLinearMethod::Automatic if model.is_integer.iter().any(|&v| v) => {
                    solve_scaled(model, BranchAndBoundSolver::default(), options.scaling)?
                }
                ProductionLinearMethod::BranchAndBound => {
                    solve_scaled(model, BranchAndBoundSolver::default(), options.scaling)?
                }
                ProductionLinearMethod::InteriorPoint => {
                    solve_scaled(model, InteriorPointSolver::default(), options.scaling)?
                }
                _ => solve_scaled(model, RevisedSimplexSolver::default(), options.scaling)?,
            };
            if solved.variable_values.is_empty() {
                solved
            } else {
                Solution {
                    variable_values: reduced.restore_point(&solved.variable_values)?,
                    objective_value: reduced.restore_objective(solved.objective_value),
                    status: solved.status,
                }
            }
        }
    };
    if solution.status == SolveStatus::Infeasible {
        solution.variable_values.clear();
    }
    let diagnostics = solution.diagnostics(problem);
    let verified = solution.status == SolveStatus::Optimal
        && diagnostics.is_verified(options.verification_tolerance);
    Ok(ProductionLinearResult {
        solution,
        diagnostics,
        presolve_report: reduced.report,
        original_variables: problem.objective.len(),
        reduced_variables: reduced.active_variables.len(),
        used_milp,
        verified,
    })
}

fn solve_scaled<S: Solver>(
    problem: &Problem,
    solver: S,
    scaling: bool,
) -> Result<Solution, OptimizeError> {
    if scaling {
        ScaledSolver {
            inner: solver,
            options: ScalingOptions::default(),
        }
        .solve(problem)
    } else {
        solver.solve(problem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bound;
    use nc_sparse::CooMatrix;
    fn model(integer: bool) -> Problem {
        let mut matrix = CooMatrix::new(1, 2);
        matrix.push(0, 0, 1.).unwrap();
        matrix.push(0, 1, 1.).unwrap();
        Problem {
            objective: vec![2., 1.],
            constraints: matrix.to_csr().unwrap(),
            row_bounds: vec![Bound {
                lower: Some(3.),
                upper: Some(3.),
            }],
            var_bounds: vec![
                Bound {
                    lower: Some(1.),
                    upper: Some(1.),
                },
                Bound {
                    lower: Some(0.),
                    upper: Some(10.),
                },
            ],
            is_integer: vec![integer, integer],
        }
    }
    #[test]
    fn presolves_postsolves_and_certifies_lp() {
        let result = solve_production_linear(&model(false), Default::default()).unwrap();
        assert_eq!(result.solution.variable_values, vec![1., 2.]);
        assert_eq!(result.solution.objective_value, 4.);
        assert!(result.verified);
        assert_eq!(result.reduced_variables, 0);
    }
    #[test]
    fn automatic_honors_integrality_after_presolve() {
        let mut problem = model(true);
        problem.row_bounds[0] = Bound {
            lower: Some(3.5),
            upper: None,
        };
        problem.var_bounds[0] = Bound {
            lower: Some(0.),
            upper: Some(3.),
        };
        let result = solve_production_linear(&problem, Default::default()).unwrap();
        assert!(result.used_milp && result.verified);
        assert!(result
            .solution
            .variable_values
            .iter()
            .all(|v| (v - v.round()).abs() < 1e-9));
    }
}
