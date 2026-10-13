//! Deterministic reference program spanning the production solver families.

use crate::{
    minimize_lbfgs, presolve_problem, solve_convex_qp, Bound, BranchAndBoundSolver, LbfgsOptions,
    OptimizeError, PresolveOptions, Problem, QuadraticOptions, QuadraticProblem,
    RevisedSimplexSolver, ScaledSolver, ScalingOptions, Solver, SparseLinearWorkspace,
};
use crate::{ValidationCase, ValidationMetric, ValidationReport, ValidationTolerance};
use nc_sparse::CsrMatrix;

/// Run small, hand-checkable LP, QP, MILP, and nonlinear cases. The returned
/// report is structured so CI and downstream applications can retain evidence,
/// rather than relying only on assertions or solver termination flags.
pub fn run_reference_validation() -> Result<ValidationReport, OptimizeError> {
    let tight = ValidationTolerance::new(1e-7, 1e-7);
    let constraints = CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 0], vec![1.0, 1.0, 1.0])?;
    let lp = Problem {
        objective: vec![-3.0, -2.0],
        constraints,
        row_bounds: vec![
            Bound {
                lower: None,
                upper: Some(4.0),
            },
            Bound {
                lower: None,
                upper: Some(3.0),
            },
        ],
        var_bounds: vec![
            Bound {
                lower: Some(0.0),
                upper: None,
            },
            Bound {
                lower: Some(0.0),
                upper: Some(10.0),
            },
        ],
        is_integer: vec![false, false],
    };
    let simplex = RevisedSimplexSolver::default().solve(&lp)?;
    let scaled = ScaledSolver {
        inner: RevisedSimplexSolver::default(),
        options: ScalingOptions::default(),
    }
    .solve(&lp)?;
    let lp_diagnostics = simplex.diagnostics(&lp);
    let lp_case = ValidationCase::new(
        "LP reference and scaling invariance",
        vec![
            ValidationMetric::compare("LP objective", simplex.objective_value, -11.0, tight),
            ValidationMetric::compare(
                "scaled LP objective",
                scaled.objective_value,
                simplex.objective_value,
                tight,
            ),
            ValidationMetric::compare(
                "LP row violation",
                lp_diagnostics.maximum_row_violation,
                0.0,
                tight,
            ),
        ],
    );

    let qp = QuadraticProblem {
        quadratic: vec![vec![2.0]],
        linear: vec![-4.0],
        constant: 0.0,
        constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![])?,
        row_bounds: vec![],
        variable_bounds: vec![Bound {
            lower: Some(0.0),
            upper: Some(1.0),
        }],
    };
    let qp_result = solve_convex_qp(&qp, QuadraticOptions::default())?;
    let qp_tolerance = ValidationTolerance::new(2e-6, 1e-7);
    let qp_case = ValidationCase::new(
        "convex QP KKT reference",
        vec![
            ValidationMetric::compare("QP point", qp_result.point[0], 1.0, qp_tolerance),
            ValidationMetric::compare(
                "QP bound violation",
                qp_result.maximum_variable_violation,
                0.0,
                qp_tolerance,
            ),
            ValidationMetric::compare(
                "QP stationarity",
                qp_result.stationarity_norm,
                0.0,
                ValidationTolerance::new(1e-5, 0.0),
            ),
        ],
    );

    let milp = Problem {
        objective: vec![-2.0, -1.0],
        constraints: CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0])?,
        row_bounds: vec![Bound {
            lower: None,
            upper: Some(1.4),
        }],
        var_bounds: vec![
            Bound {
                lower: Some(0.0),
                upper: Some(1.0)
            };
            2
        ],
        is_integer: vec![true, true],
    };
    let milp_result = BranchAndBoundSolver::default().solve_with_report(&milp)?;
    let milp_diagnostics = milp_result.solution.diagnostics(&milp);
    let milp_case = ValidationCase::new(
        "MILP enumerated optimum",
        vec![
            ValidationMetric::compare(
                "MILP objective",
                milp_result.solution.objective_value,
                -2.0,
                tight,
            ),
            ValidationMetric::compare(
                "MILP integrality",
                milp_diagnostics.maximum_integrality_violation,
                0.0,
                tight,
            ),
            ValidationMetric::compare(
                "MILP certified gap",
                milp_result.absolute_gap.unwrap_or(f64::INFINITY),
                0.0,
                tight,
            ),
        ],
    );

    let nonlinear = minimize_lbfgs(&[8.0, -8.0], LbfgsOptions::default(), |point| {
        let dx = point[0] - 2.0;
        let dy = point[1] + 1.0;
        Ok((dx * dx + dy * dy, vec![2.0 * dx, 2.0 * dy]))
    })?;
    let nonlinear_case = ValidationCase::new(
        "nonlinear stationary reference",
        vec![
            ValidationMetric::compare("nonlinear x", nonlinear.point[0], 2.0, tight),
            ValidationMetric::compare("nonlinear y", nonlinear.point[1], -1.0, tight),
            ValidationMetric::compare(
                "nonlinear gradient norm",
                nonlinear.gradient_norm,
                0.0,
                ValidationTolerance::new(1e-7, 0.0),
            ),
        ],
    );

    let transform_problem = Problem {
        objective: vec![2.0, 1.0],
        constraints: CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 1.0, 1.0])?,
        row_bounds: vec![
            Bound::fixed(3.0),
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
    let presolved = presolve_problem(&transform_problem, PresolveOptions::default())?;
    let restored = presolved.restore_point(&[])?;
    let presolve_case = ValidationCase::new(
        "presolve and postsolve equivalence",
        vec![
            ValidationMetric::compare("restored x", restored[0], 1.0, tight),
            ValidationMetric::compare("restored y", restored[1], 2.0, tight),
            ValidationMetric::compare(
                "restored objective",
                presolved.restore_objective(0.0),
                4.0,
                tight,
            ),
            ValidationMetric::compare(
                "fixed variables",
                presolved.report.fixed_variables as f64,
                2.0,
                ValidationTolerance::new(0.0, 0.0),
            ),
        ],
    );

    let sparse_problem = Problem {
        objective: vec![0.0; 3],
        constraints: CsrMatrix::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 2, 0, 1, 1, 2],
            vec![1e-8, 2.0, -3.0, 4e8, 5.0, -6.0],
        )?,
        row_bounds: vec![Bound::free(); 3],
        var_bounds: vec![Bound::free(); 3],
        is_integer: vec![false; 3],
    };
    let workspace = SparseLinearWorkspace::new(sparse_problem)?;
    let activity = workspace.row_activity(&[2.0, -1.0, 3.0])?;
    let transpose = workspace.transpose_product(&[1.0, -2.0, 0.5])?;
    let sparse_case = ValidationCase::new(
        "sparse operator scale coverage",
        vec![
            ValidationMetric::compare("scaled row 0", activity[0], 6.00000002, tight),
            ValidationMetric::compare("scaled row 1", activity[1], -400000006.0, tight),
            ValidationMetric::compare("transpose column 0", transpose[0], 6.00000001, tight),
            ValidationMetric::compare("transpose column 2", transpose[2], -1.0, tight),
        ],
    );

    let first = crate::deterministic_perturbation(&[1.0, -2.0, 3.0], 42, 1e-6);
    let second = crate::deterministic_perturbation(&[1.0, -2.0, 3.0], 42, 1e-6);
    let perturbation_case = ValidationCase::new(
        "deterministic perturbation reproducibility",
        first
            .iter()
            .zip(second)
            .enumerate()
            .map(|(index, (observed, reference))| {
                ValidationMetric::compare(
                    format!("perturbation {index}"),
                    *observed,
                    reference,
                    ValidationTolerance::new(0.0, 0.0),
                )
            })
            .collect(),
    );

    Ok(ValidationReport::new(vec![
        lp_case,
        qp_case,
        milp_case,
        nonlinear_case,
        presolve_case,
        sparse_case,
        perturbation_case,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_reference_program_passes() {
        let report = run_reference_validation().unwrap();
        assert!(report.passed(), "{report:#?}");
        assert_eq!(report.cases.len(), 7);
        assert_eq!(report.failure_count(), 0);
        assert!(report.maximum_normalized_error() <= 1.0);
    }
}
