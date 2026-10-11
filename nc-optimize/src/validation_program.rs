//! Deterministic reference program spanning the production solver families.

use crate::{minimize_lbfgs, solve_convex_qp, Bound, BranchAndBoundSolver,
    LbfgsOptions, OptimizeError, Problem, QuadraticOptions, QuadraticProblem,
    RevisedSimplexSolver, ScaledSolver, ScalingOptions, Solver};
use crate::{ValidationCase, ValidationMetric, ValidationReport, ValidationTolerance};
use nc_sparse::CsrMatrix;

/// Run small, hand-checkable LP, QP, MILP, and nonlinear cases. The returned
/// report is structured so CI and downstream applications can retain evidence,
/// rather than relying only on assertions or solver termination flags.
pub fn run_reference_validation() -> Result<ValidationReport, OptimizeError> {
    let tight = ValidationTolerance::new(1e-7, 1e-7);
    let constraints = CsrMatrix::new(
        2, 2, vec![0, 2, 3], vec![0, 1, 0], vec![1.0, 1.0, 1.0])?;
    let lp = Problem {
        objective: vec![-3.0, -2.0], constraints,
        row_bounds: vec![Bound { lower: None, upper: Some(4.0) },
                         Bound { lower: None, upper: Some(3.0) }],
        var_bounds: vec![Bound { lower: Some(0.0), upper: None },
                         Bound { lower: Some(0.0), upper: Some(10.0) }],
        is_integer: vec![false, false],
    };
    let simplex = RevisedSimplexSolver::default().solve(&lp)?;
    let scaled = ScaledSolver { inner: RevisedSimplexSolver::default(),
        options: ScalingOptions::default() }.solve(&lp)?;
    let lp_diagnostics = simplex.diagnostics(&lp);
    let lp_case = ValidationCase::new("LP reference and scaling invariance", vec![
        ValidationMetric::compare("LP objective", simplex.objective_value, -11.0, tight),
        ValidationMetric::compare("scaled LP objective", scaled.objective_value,
                                  simplex.objective_value, tight),
        ValidationMetric::compare("LP row violation", lp_diagnostics.maximum_row_violation,
                                  0.0, tight),
    ]);

    let qp = QuadraticProblem {
        quadratic: vec![vec![2.0]], linear: vec![-4.0], constant: 0.0,
        constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![])?, row_bounds: vec![],
        variable_bounds: vec![Bound { lower: Some(0.0), upper: Some(1.0) }],
    };
    let qp_result = solve_convex_qp(&qp, QuadraticOptions::default())?;
    let qp_tolerance = ValidationTolerance::new(2e-6, 1e-7);
    let qp_case = ValidationCase::new("convex QP KKT reference", vec![
        ValidationMetric::compare("QP point", qp_result.point[0], 1.0, qp_tolerance),
        ValidationMetric::compare("QP bound violation", qp_result.maximum_variable_violation,
                                  0.0, qp_tolerance),
        ValidationMetric::compare("QP stationarity", qp_result.stationarity_norm, 0.0,
                                  ValidationTolerance::new(1e-5, 0.0)),
    ]);

    let milp = Problem {
        objective: vec![-2.0, -1.0],
        constraints: CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0])?,
        row_bounds: vec![Bound { lower: None, upper: Some(1.4) }],
        var_bounds: vec![Bound { lower: Some(0.0), upper: Some(1.0) }; 2],
        is_integer: vec![true, true],
    };
    let milp_result = BranchAndBoundSolver::default().solve_with_report(&milp)?;
    let milp_diagnostics = milp_result.solution.diagnostics(&milp);
    let milp_case = ValidationCase::new("MILP enumerated optimum", vec![
        ValidationMetric::compare("MILP objective", milp_result.solution.objective_value,
                                  -2.0, tight),
        ValidationMetric::compare("MILP integrality",
                                  milp_diagnostics.maximum_integrality_violation, 0.0, tight),
        ValidationMetric::compare("MILP certified gap", milp_result.absolute_gap.unwrap_or(f64::INFINITY),
                                  0.0, tight),
    ]);

    let nonlinear = minimize_lbfgs(&[8.0, -8.0], LbfgsOptions::default(), |point| {
        let dx = point[0] - 2.0;
        let dy = point[1] + 1.0;
        Ok((dx * dx + dy * dy, vec![2.0 * dx, 2.0 * dy]))
    })?;
    let nonlinear_case = ValidationCase::new("nonlinear stationary reference", vec![
        ValidationMetric::compare("nonlinear x", nonlinear.point[0], 2.0, tight),
        ValidationMetric::compare("nonlinear y", nonlinear.point[1], -1.0, tight),
        ValidationMetric::compare("nonlinear gradient norm", nonlinear.gradient_norm, 0.0,
                                  ValidationTolerance::new(1e-7, 0.0)),
    ]);

    Ok(ValidationReport::new(vec![lp_case, qp_case, milp_case, nonlinear_case]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_reference_program_passes() {
        let report = run_reference_validation().unwrap();
        assert!(report.passed(), "{report:#?}");
        assert_eq!(report.cases.len(), 4);
        assert_eq!(report.failure_count(), 0);
    }
}
