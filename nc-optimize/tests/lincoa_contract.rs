use nc_optimize::{
    minimize_lincoa, Bound, LincoaOptions, LincoaTermination, OptimizeError, Problem,
};
use nc_sparse::CsrMatrix;

fn model(scale: f64) -> Problem {
    Problem {
        // The callback, not this linear objective, sets the minimizer.
        objective: vec![-100.0, 10.0, 0.0],
        constraints: CsrMatrix::new(
            3,
            3,
            vec![0, 3, 5, 6],
            vec![0, 1, 2, 0, 1, 0],
            vec![1.0, 1.0, 2.0, scale, scale, 1.0],
        )
        .unwrap(),
        row_bounds: vec![
            Bound {
                lower: Some(8.25),
                upper: Some(10.0),
            },
            Bound {
                lower: None,
                upper: Some(2.0 * scale),
            },
            Bound {
                lower: None,
                upper: Some(0.75),
            },
        ],
        var_bounds: vec![
            Bound {
                lower: Some(0.0),
                upper: None,
            },
            Bound {
                lower: Some(0.0),
                upper: None,
            },
            Bound::fixed(4.0),
        ],
        is_integer: vec![false; 3],
    }
}

#[test]
fn recovers_two_active_constraints_with_a_dependent_scaled_row() {
    for scale in [1.0, 1e-160, 1e160] {
        let mut samples = Vec::new();
        let report = minimize_lincoa(
            &[0.0, 0.25, 9.0],
            &model(scale),
            LincoaOptions::default(),
            |x| {
                assert_eq!(x[2], 4.0);
                let value = (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
                samples.push((x.to_vec(), value));
                Ok(value)
            },
        )
        .unwrap();

        eprintln!("row scale {scale:e}, {} evaluations", report.evaluations);
        assert_eq!(report.termination, LincoaTermination::RadiusReached);
        assert_eq!(report.evaluations as usize, samples.len());
        assert!((report.point[0] - 0.75).abs() < 1e-6, "{report:?}");
        assert!((report.point[1] - 1.25).abs() < 1e-6, "{report:?}");
        assert!((report.objective - 2.125).abs() < 1e-6, "{report:?}");
        assert!(report.maximum_violation <= 1e-8);
        assert!(samples.contains(&(report.point, report.objective)));
    }
}

#[test]
fn initialization_budget_counts_infeasible_samples_without_returning_them() {
    let problem = model(1.0);
    let mut values = Vec::new();
    let report = minimize_lincoa(
        &[0.0, 0.25, 4.0],
        &problem,
        LincoaOptions {
            max_evaluations: 2,
            ..LincoaOptions::default()
        },
        |x| {
            values.push(x.to_vec());
            Ok(-x[0])
        },
    )
    .unwrap();
    assert_eq!(report.termination, LincoaTermination::EvaluationLimit);
    assert_eq!(report.evaluations, 2);
    assert_eq!(values.len(), 2);
    assert!(values[1][0] > 0.75, "{values:?}");
    assert_eq!(report.point, vec![0.0, 0.25, 4.0]);
    assert_eq!(report.maximum_violation, 0.0);

    let report = minimize_lincoa(
        &[0.0, 0.25, 4.0],
        &problem,
        LincoaOptions {
            max_iterations: 1,
            ..LincoaOptions::default()
        },
        |x| Ok(-x[0]),
    )
    .unwrap();
    assert_eq!(report.termination, LincoaTermination::IterationLimit);
}

#[test]
fn validates_before_callback_and_propagates_callback_errors() {
    let initial = [0.0, 0.25, 4.0];
    let options = LincoaOptions::default();
    let mut problem = model(1.0);
    let unused = |_: &[f64]| panic!("invalid model must not call the objective");
    assert!(minimize_lincoa(&[0.0, 0.0, 4.0], &problem, options, unused).is_err());
    problem.is_integer[0] = true;
    assert!(minimize_lincoa(&initial, &problem, options, unused).is_err());
    problem.is_integer[0] = false;
    problem.row_bounds[0] = Bound::fixed(8.25);
    assert!(minimize_lincoa(&initial, &problem, options, unused).is_err());
    problem.row_bounds[0] = Bound::free();
    let err = minimize_lincoa(&initial, &problem, options, |_| {
        Err(OptimizeError::InvalidConfiguration("callback error"))
    })
    .unwrap_err();
    assert!(matches!(
        err,
        OptimizeError::InvalidConfiguration("callback error")
    ));
    assert!(minimize_lincoa(&initial, &problem, options, |_| Ok(f64::NAN)).is_err());

    problem.var_bounds = initial.into_iter().map(Bound::fixed).collect();
    let report = minimize_lincoa(&[9.0; 3], &problem, options, |x| Ok(x.iter().sum())).unwrap();
    assert_eq!(report.termination, LincoaTermination::AllVariablesFixed);
    assert_eq!(report.evaluations, 1);
    assert_eq!(report.point, initial);
    assert_eq!(report.objective, 4.25);
    problem.row_bounds[0] = Bound::fixed(8.0);
    assert!(minimize_lincoa(&initial, &problem, options, unused).is_err());
}

#[test]
fn supports_a_single_free_variable_without_explicit_rows() {
    let problem = Problem {
        objective: vec![0.0],
        constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![]).unwrap(),
        row_bounds: vec![],
        var_bounds: vec![Bound::free()],
        is_integer: vec![false],
    };
    let report = minimize_lincoa(&[0.0], &problem, LincoaOptions::default(), |x| {
        Ok((x[0] - 3.0).powi(2))
    })
    .unwrap();
    assert!((report.point[0] - 3.0).abs() < 1e-6, "{report:?}");
}

#[test]
fn compares_with_cobyla_at_the_same_objective_budget() {
    use nc_optimize::{
        minimize_cobyla, CobylaOptions, ConstrainedDerivativeFreeEvaluation,
        DerivativeFreeConstraintKind,
    };

    let problem = model(1.0);
    let initial = [0.0, 0.25, 4.0];
    let budget = 100;
    let tolerance = 1e-8;
    let objective = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
    let lincoa = minimize_lincoa(
        &initial,
        &problem,
        LincoaOptions {
            max_evaluations: budget,
            feasibility_tolerance: tolerance,
            ..LincoaOptions::default()
        },
        |x| Ok(objective(x)),
    )
    .unwrap();
    let cobyla = minimize_cobyla(
        &initial,
        Some(&problem.var_bounds),
        &[DerivativeFreeConstraintKind::LessEqual; 4],
        CobylaOptions {
            max_evaluations: budget,
            feasibility_tolerance: tolerance,
            ..CobylaOptions::default()
        },
        |x| {
            let y = problem.constraints.spmv(x)?;
            Ok(ConstrainedDerivativeFreeEvaluation {
                objective: objective(x),
                constraints: vec![8.25 - y[0], y[0] - 10.0, y[1] - 2.0, y[2] - 0.75],
            })
        },
    )
    .unwrap();

    assert!(lincoa.maximum_violation <= tolerance);
    assert!(cobyla.feasible, "{cobyla:?}");
    assert!((lincoa.objective - 2.125).abs() < 1e-6, "{lincoa:?}");
    assert!((cobyla.objective - 2.125).abs() < 1e-6, "{cobyla:?}");
    assert!(lincoa.evaluations <= budget && cobyla.evaluations <= budget);
    eprintln!(
        "LINCOA {} calls, COBYLA {} calls",
        lincoa.evaluations, cobyla.evaluations
    );
}
