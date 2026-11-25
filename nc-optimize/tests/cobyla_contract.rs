use nc_optimize::{
    minimize_cobyla, minimize_constrained_derivative_free, Bound, CobylaOptions, CobylaTermination,
    ConstrainedDerivativeFreeEvaluation as Evaluation, ConstrainedDerivativeFreeOptions,
    DerivativeFreeConstraintKind::{Equality, LessEqual},
    OptimizeError,
};

fn disk(p: &[f64]) -> Result<Evaluation, OptimizeError> {
    Ok(Evaluation {
        objective: p[0].powi(2) + p[1].powi(2),
        constraints: vec![(p[0] - 2.0).powi(2) + p[1].powi(2) - 1.0],
    })
}

#[test]
fn reaches_a_curved_active_constraint_from_an_infeasible_start() {
    let start = [0.0, 2.0];
    let budget = 220;
    let mut calls = 0;
    let result = minimize_cobyla(
        &start,
        None,
        &[LessEqual],
        CobylaOptions {
            max_evaluations: budget,
            initial_radius: 0.75,
            ..Default::default()
        },
        |p| {
            calls += 1;
            disk(p)
        },
    )
    .unwrap();
    let prior = minimize_constrained_derivative_free(
        &start,
        None,
        &[LessEqual],
        ConstrainedDerivativeFreeOptions {
            max_evaluations: budget as usize,
            initial_radius: 0.75,
            ..Default::default()
        },
        disk,
    )
    .unwrap();
    println!("Common budget {budget}: COBYLA f={:.9e} violation={:.3e} calls={}; existing method f={:.9e} violation={:.3e} calls={}",
             result.objective, result.maximum_violation, result.evaluations,
             prior.objective, prior.maximum_violation, prior.evaluations);
    assert_eq!(result.termination, CobylaTermination::RadiusReached);
    assert_eq!(result.evaluations, calls);
    assert!(calls <= budget && result.feasible, "{result:?}");
    assert!((result.point[0] - 1.0).abs() < 1e-5);
    assert!(result.point[1].abs() < 1e-4);
    assert!((result.objective - 1.0).abs() < 1e-6);
    assert_eq!(
        result.constraint_values,
        disk(&result.point).unwrap().constraints
    );
}

#[test]
fn folds_equalities_and_restores_fixed_coordinates() {
    let result = minimize_cobyla(
        &[0.0, 0.0],
        Some(&[
            Bound {
                lower: Some(-1.0),
                upper: Some(1.0),
            },
            Bound::fixed(0.25),
        ]),
        &[Equality],
        CobylaOptions {
            initial_radius: 0.25,
            ..Default::default()
        },
        |p| {
            assert_eq!(p[1], 0.25);
            Ok(Evaluation {
                objective: p[0] * p[0],
                constraints: vec![p[0] + p[1] - 1.0],
            })
        },
    )
    .unwrap();
    assert!(result.feasible, "{result:?}");
    assert!((result.point[0] - 0.75).abs() <= 1e-7);
    assert_eq!(result.constraint_values.len(), 1);

    let fixed = minimize_cobyla(
        &[0.0],
        Some(&[Bound::fixed(2.0)]),
        &[LessEqual],
        Default::default(),
        |p| {
            Ok(Evaluation {
                objective: 0.0,
                constraints: vec![p[0] - 1.0],
            })
        },
    )
    .unwrap();
    assert_eq!(fixed.termination, CobylaTermination::AllVariablesFixed);
    assert_eq!(
        (fixed.feasible, fixed.maximum_violation, fixed.evaluations),
        (false, 1.0, 1)
    );
}

#[test]
fn budget_is_exact_even_during_simplex_initialization() {
    let mut samples = Vec::new();
    let result = minimize_cobyla(
        &[0.0, 0.0, 0.0],
        None,
        &[LessEqual],
        CobylaOptions {
            max_evaluations: 2,
            ..Default::default()
        },
        |p| {
            samples.push(p.to_vec());
            Ok(Evaluation {
                objective: p[0] + p[1],
                constraints: vec![2.0 - p[0]],
            })
        },
    )
    .unwrap();
    assert_eq!(result.termination, CobylaTermination::EvaluationLimit);
    assert_eq!(result.evaluations, 2);
    assert_eq!(samples.len(), 2);
    assert!(samples.contains(&result.point));
    assert!(!result.feasible && result.maximum_violation > 0.0);
}

#[test]
fn malformed_and_failed_evaluations_stop_the_solver() {
    let malformed = minimize_cobyla(&[0.0], None, &[LessEqual], Default::default(), |_| {
        Ok(Evaluation {
            objective: 0.0,
            constraints: vec![],
        })
    });
    assert!(malformed.is_err());
    for invalid in [f64::NAN, f64::INFINITY, f64::MAX] {
        let result = minimize_cobyla(&[0.0], None, &[], Default::default(), |_| {
            Ok(Evaluation {
                objective: invalid,
                constraints: vec![],
            })
        });
        assert!(result.is_err());
    }
    let mut calls = 0;
    let failed = minimize_cobyla(&[0.0], None, &[], Default::default(), |_| {
        calls += 1;
        Err(OptimizeError::InvalidConfiguration("callback failed"))
    })
    .unwrap_err();
    assert!(matches!(
        failed,
        OptimizeError::InvalidConfiguration("callback failed")
    ));
    assert_eq!(calls, 1);
}
