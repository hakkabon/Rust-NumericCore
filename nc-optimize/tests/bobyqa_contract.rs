use nc_optimize::{
    minimize_bobyqa, minimize_model_based, BobyqaOptions, BobyqaTermination, Bound,
    ModelBasedOptions, OptimizeError,
};

#[test]
fn active_bound_on_rotated_quadratic() {
    let mut calls = 0;
    let solution = minimize_bobyqa(
        &[-3.0, 2.0],
        &[
            Bound {
                lower: Some(-4.0),
                upper: Some(1.0),
            },
            Bound {
                lower: Some(-4.0),
                upper: Some(4.0),
            },
        ],
        BobyqaOptions {
            max_evaluations: 80,
            ..Default::default()
        },
        |p| {
            calls += 1;
            assert!((-4.0..=1.0).contains(&p[0]) && (-4.0..=4.0).contains(&p[1]));
            Ok((p[0] + p[1] - 2.0).powi(2) + 3.0 * (p[0] - p[1] - 1.0).powi(2))
        },
    )
    .unwrap();
    assert_eq!(solution.termination, BobyqaTermination::RadiusReached);
    assert_eq!(solution.evaluations, calls);
    assert!(calls <= 80);
    assert_eq!(solution.point[0], 1.0);
    assert!((solution.point[1] - 0.25).abs() < 1e-5);
    assert!((solution.objective - 0.75).abs() < 1e-9);
}

#[test]
fn fixed_coordinates_and_budgets_smaller_than_the_initial_model() {
    let mut samples = Vec::new();
    let bounds = [
        Bound {
            lower: Some(0.0),
            upper: Some(0.02),
        },
        Bound::fixed(7.0),
    ];
    let result = minimize_bobyqa(
        &[5.0, 2.0],
        &bounds,
        BobyqaOptions {
            max_evaluations: 2,
            ..Default::default()
        },
        |p| {
            assert!((0.0..=0.02).contains(&p[0]));
            assert_eq!(p[1], 7.0);
            samples.push(p.to_vec());
            Ok((p[0] - 0.007).powi(2))
        },
    )
    .unwrap();
    assert_eq!(result.termination, BobyqaTermination::EvaluationLimit);
    assert_eq!(result.evaluations, 2);
    assert_eq!(samples.len(), 2);
    assert!(samples.contains(&result.point));
    let refined = minimize_bobyqa(&[5.0, 2.0], &bounds, Default::default(), |p| {
        Ok((p[0] - 0.007).powi(2))
    })
    .unwrap();
    assert!((refined.point[0] - 0.007).abs() < 1e-5);
    assert_eq!(refined.point[1], 7.0);
    let fixed = minimize_bobyqa(&[0.0], &[Bound::fixed(4.0)], Default::default(), |p| {
        Ok(p[0])
    })
    .unwrap();
    assert_eq!(fixed.termination, BobyqaTermination::AllVariablesFixed);
    assert_eq!(
        (fixed.point.as_slice(), fixed.objective, fixed.evaluations),
        (&[4.0][..], 4.0, 1)
    );
}

#[test]
fn objective_error_stops_callbacks_and_bad_options_are_rejected() {
    let mut calls = 0;
    let error = minimize_bobyqa(&[0.0], &[Bound::free()], Default::default(), |_| {
        calls += 1;
        Err(OptimizeError::InvalidConfiguration("callback failure"))
    })
    .unwrap_err();
    assert!(matches!(
        error,
        OptimizeError::InvalidConfiguration("callback failure")
    ));
    assert_eq!(calls, 1);
    assert!(
        minimize_bobyqa(&[0.0], &[Bound::free()], Default::default(), |_| Ok(
            f64::NAN
        ))
        .is_err()
    );
    assert!(minimize_bobyqa(
        &[0.0],
        &[Bound::free()],
        BobyqaOptions {
            interpolation_points: Some(4),
            ..Default::default()
        },
        |_| unreachable!("invalid options must fail before evaluating"),
    )
    .is_err());
}

#[test]
fn calibrates_an_exponential_response_at_a_shared_budget() {
    let bounds = [Bound {
        lower: Some(0.0),
        upper: Some(4.0),
    }; 3];
    let initial = [1.0, 1.3, 1.0];
    let budget = 160;
    let objective = |p: &[f64]| {
        Ok((0..12)
            .map(|i| {
                let time = f64::from(i) * 0.4;
                let target = 0.4 + 2.5 * (-0.7 * time).exp();
                (p[2] + p[0] * (-p[1] * time).exp() - target).powi(2)
            })
            .sum())
    };
    let result = minimize_bobyqa(
        &initial,
        &bounds,
        BobyqaOptions {
            max_evaluations: budget,
            initial_radius: 0.3,
            ..Default::default()
        },
        objective,
    )
    .unwrap();
    let diagonal = minimize_model_based(
        &initial,
        Some(&bounds),
        ModelBasedOptions {
            max_evaluations: budget,
            initial_radius: 0.3,
            final_radius: 1e-6,
            ..Default::default()
        },
        objective,
    )
    .unwrap();
    println!("Equal budget {budget}: BOBYQA f={:.6e} after {} calls, diagonal model f={:.6e} after {} calls",
             result.objective, result.evaluations, diagonal.objective, diagonal.evaluations);
    assert!(result.evaluations <= budget && diagonal.evaluations <= budget);
    assert!(result.objective < 1e-8, "{result:?}");
}
