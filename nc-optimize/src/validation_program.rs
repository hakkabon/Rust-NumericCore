//! Deterministic reference program spanning the production solver families.

use crate::{
    minimize_lbfgs, presolve_problem, solve_convex_qp, Bound, BranchAndBoundSolver, LbfgsOptions,
    OptimizeError, PresolveOptions, Problem, QuadraticOptions, QuadraticProblem,
    RevisedSimplexSolver, ScaledSolver, ScalingOptions, Solver, SparseLinearWorkspace,
    SparseHessian, SparseJacobian, SparseKktProblem, SparseKktWorkspace,
    certify_convex_qp, parse_spice_netlist, CertificateClaim, CertificateTolerance, SpiceAnalysis,
};
use crate::{
    ValidationCase, ValidationMetric, ValidationReport, ValidationTier, ValidationTolerance,
};
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
    )
    .classified(
        "internal.lp.reference",
        "linear-programming",
        ValidationTier::Smoke,
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
    )
    .classified(
        "internal.qp.kkt",
        "quadratic-programming",
        ValidationTier::Smoke,
    );
    let qp_certificate = certify_convex_qp(&qp, &qp_result, CertificateTolerance::default())?;
    let certificate_case = ValidationCase::new("independent convex QP certificate", vec![
        ValidationMetric::compare("QP certificate verified",
            if qp_certificate.verified() { 1.0 } else { 0.0 }, 1.0,
            ValidationTolerance::new(0.0, 0.0)),
        ValidationMetric::compare("QP optimality claim",
            if qp_certificate.claim == CertificateClaim::Optimality { 1.0 } else { 0.0 }, 1.0,
            ValidationTolerance::new(0.0, 0.0)),
        ValidationMetric::compare("QP certificate failures",
            qp_certificate.metrics.iter().filter(|metric| !metric.passed).count() as f64, 0.0,
            ValidationTolerance::new(0.0, 0.0)),
    ]).classified("internal.certificate.qp-kkt", "optimization-certification",
        ValidationTier::Standard);

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
    )
    .classified(
        "internal.milp.enumerated",
        "mixed-integer-programming",
        ValidationTier::Smoke,
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
    )
    .classified(
        "internal.nlp.stationary",
        "nonlinear-optimization",
        ValidationTier::Smoke,
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
    )
    .classified(
        "internal.presolve.postsolve",
        "model-transformation",
        ValidationTier::Standard,
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
    )
    .classified(
        "internal.sparse.mixed-scale",
        "sparse-linear-algebra",
        ValidationTier::Standard,
    );

    let kkt_problem = SparseKktProblem {
        hessian: SparseHessian { dimension: 2, row_pointers: vec![0, 1, 2],
            column_indices: vec![0, 1], values: vec![2.0, 2.0] },
        jacobian: SparseJacobian { rows: 1, columns: 2, row_pointers: vec![0, 2],
            column_indices: vec![0, 1], values: vec![1.0, 1.0] },
        primal_rhs: vec![-2.0, -4.0], constraint_rhs: vec![-1.0],
        primal_regularization: 0.0, dual_regularization: 0.0, drop_tolerance: 0.0,
    };
    let mut kkt_workspace = SparseKktWorkspace::new(
        &kkt_problem, nc_sparse::SparseOrdering::MinimumDegree)?;
    kkt_workspace.solve(&kkt_problem)?;
    let mut updated_kkt = kkt_problem.clone();
    updated_kkt.hessian.values = vec![4.0, 3.0];
    updated_kkt.primal_rhs = vec![-4.0, -3.0];
    let kkt = kkt_workspace.solve(&updated_kkt)?;
    let kkt_case = ValidationCase::new("reusable sparse nonlinear KKT pipeline", vec![
        ValidationMetric::compare("KKT relative residual", kkt.relative_residual, 0.0,
            ValidationTolerance::new(1e-12, 0.0)),
        ValidationMetric::compare("KKT symbolic analyses", kkt.symbolic_analyses as f64, 1.0,
            ValidationTolerance::new(0.0, 0.0)),
        ValidationMetric::compare("KKT numeric factorizations", kkt.numeric_factorizations as f64,
            2.0, ValidationTolerance::new(0.0, 0.0)),
        ValidationMetric::compare("KKT negative inertia", kkt.inertia.negative as f64, 1.0,
            ValidationTolerance::new(0.0, 0.0)),
    ]).classified("internal.sparse.nonlinear-kkt", "sparse-nonlinear",
        ValidationTier::Standard);

    let netlist = parse_spice_netlist("validation RC low pass\nV1 in 0 DC 1 AC 1\nR1 in out 1k\nC1 out 0 1u\n.OP\n.AC DEC 10 10 10k\n.END")
        .map_err(|error| OptimizeError::InvalidProblem(error.to_string()))?;
    let operating = netlist.circuit.dc_operating_point(None, Default::default())?;
    let ac_request = netlist.analyses.iter().find_map(|analysis| match analysis {
        SpiceAnalysis::Ac(request) => Some(request), _ => None,
    }).ok_or_else(|| OptimizeError::InvalidProblem("SPICE validation lost AC request".into()))?;
    let sweep = netlist.circuit.ac_analysis(&operating.state, &ac_request.frequencies(),
        &netlist.ac_excitations)?;
    let maximum_ac_residual = sweep.points.iter().map(|point| point.residual_norm)
        .fold(0.0_f64, f64::max);
    let spice_case = ValidationCase::new("SPICE netlist DC and AC integration", vec![
        ValidationMetric::compare("SPICE DC output", operating.node_voltages[2], 1.0, tight),
        ValidationMetric::compare("SPICE AC point count", sweep.points.len() as f64, 31.0,
            ValidationTolerance::new(0.0, 0.0)),
        ValidationMetric::compare("SPICE AC maximum residual", maximum_ac_residual, 0.0,
            ValidationTolerance::new(1e-9, 0.0)),
    ]).classified("internal.circuit.spice-netlist", "circuit-analysis",
        ValidationTier::Standard);

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
    )
    .classified(
        "internal.reproducibility.perturbation",
        "validation-infrastructure",
        ValidationTier::Standard,
    );

    let mut cases = vec![
        lp_case,
        qp_case,
        certificate_case,
        milp_case,
        nonlinear_case,
        presolve_case,
        sparse_case,
        kkt_case,
        spice_case,
        perturbation_case,
    ];
    cases.extend(crate::run_external_reference_validation()?);
    Ok(ValidationReport::new(cases))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_reference_program_passes() {
        let report = run_reference_validation().unwrap();
        assert!(report.passed(), "{report:#?}");
        assert_eq!(report.cases.len(), 12);
        assert_eq!(report.failure_count(), 0);
        assert!(report.maximum_normalized_error() <= 1.0);
        assert_eq!(report.for_tier(ValidationTier::Smoke).cases.len(), 4);
        assert_eq!(report.for_tier(ValidationTier::Standard).cases.len(), 12);

        let first = report.snapshot("rust", ValidationTier::Standard);
        let second = run_reference_validation()
            .unwrap()
            .snapshot("rust", ValidationTier::Standard);
        assert_eq!(first.to_json_pretty(), second.to_json_pretty());
        assert!(second.regressions_against(&first, 0.0).is_empty());
    }
}
