use nc_optimize::{
    minimize_newuoa, minimize_uobyqa, OptimizeError, PowellUnconstrainedOptions as Options,
    PowellUnconstrainedTermination as Stop,
};

#[test]
fn compares_objective_calls_at_a_shared_budget() {
    fn coupled(x: &[f64]) -> f64 {
        let a = x[0] - 2.0;
        let b = x[1] + 1.0;
        let c = x[2] - 3.0;
        (a + b).powi(2) + (b + c).powi(2) + 0.1 * (a - c).powi(2) + c.powi(2)
    }
    fn rosenbrock(x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    }
    type Objective = fn(&[f64]) -> f64;
    let problems: [(&str, &[f64], &[f64], Objective); 2] = [
        ("coupled", &[0.0; 3], &[2.0, -1.0, 3.0], coupled),
        ("rosenbrock", &[-1.2, 1.0], &[1.0, 1.0], rosenbrock),
    ];
    for (label, initial, expected, cost) in problems {
        let options = Options {
            max_evaluations: 500,
            initial_radius: 0.5,
            ..Options::default()
        };
        let mut samples = Vec::new();
        let mut objective = |x: &[f64]| {
            let value = cost(x);
            samples.push((x.to_vec(), value));
            Ok(value)
        };
        let newuoa = minimize_newuoa(initial, options, &mut objective).unwrap();
        let newuoa_samples = std::mem::take(&mut samples);
        let uobyqa = minimize_uobyqa(initial, options, |x| {
            let value = cost(x);
            samples.push((x.to_vec(), value));
            Ok(value)
        })
        .unwrap();

        eprintln!("{label}: NEWUOA {newuoa:?}, UOBYQA {uobyqa:?}");
        for (report, history) in [(newuoa, newuoa_samples), (uobyqa, samples)] {
            assert_eq!(report.termination, Stop::RadiusReached);
            assert_eq!(report.evaluations as usize, history.len());
            assert!(report.evaluations <= options.max_evaluations);
            assert!(report
                .point
                .iter()
                .zip(expected)
                .all(|(x, want)| (x - want).abs() < 2e-5));
            assert!(report.objective < 1e-10);
            assert!(history.contains(&(report.point, report.objective)));
        }
    }
}

#[test]
fn interpolation_setup_counts_each_callback_and_returns_observed_best() {
    let options = Options {
        max_evaluations: 2,
        ..Options::default()
    };
    for newuoa in [true, false] {
        let mut samples = Vec::new();
        let objective = |x: &[f64]| {
            samples.push(x.to_vec());
            Ok(-x[0])
        };
        let initial = [0.0; 3];
        let result = if newuoa {
            minimize_newuoa(&initial, options, objective)
        } else {
            minimize_uobyqa(&initial, options, objective)
        }
        .unwrap();
        assert_eq!(result.termination, Stop::EvaluationLimit);
        assert_eq!(result.evaluations, 2);
        assert_eq!(samples, [vec![0.0; 3], vec![1.0, 0.0, 0.0]]);
        assert_eq!(result.point, samples[1]);
        assert_eq!(result.objective, -1.0);
    }
}

#[test]
fn rejects_invalid_inputs_and_preserves_the_original_callback_error() {
    let options = Options::default();
    for solve in [minimize_newuoa, minimize_uobyqa] {
        assert!(solve(&[], options, |_: &[f64]| unreachable!()).is_err());
    }
    for solve in [minimize_newuoa, minimize_uobyqa] {
        assert!(solve(&[0.0], options, |_: &[f64]| Ok(f64::NAN)).is_err());
    }
    for solve in [minimize_newuoa, minimize_uobyqa] {
        let error = solve(&[0.0], options, |_: &[f64]| {
            Err(OptimizeError::InvalidConfiguration("objective error"))
        })
        .unwrap_err();
        assert!(matches!(
            error,
            OptimizeError::InvalidConfiguration("objective error")
        ));
    }
    assert!(minimize_uobyqa(&[0.0; 17], options, |_| unreachable!()).is_err());
}

#[test]
fn supports_one_dimension_and_distinguishes_iteration_limit() {
    for solve in [minimize_newuoa, minimize_uobyqa] {
        let report = solve(&[0.0], Options::default(), |x: &[f64]| {
            Ok((x[0] - 3.0).powi(2))
        })
        .unwrap();
        assert_eq!(report.termination, Stop::RadiusReached);
        assert!((report.point[0] - 3.0).abs() < 1e-6);
    }
    for solve in [minimize_newuoa, minimize_uobyqa] {
        let options = Options {
            max_iterations: 1,
            ..Options::default()
        };
        let report = solve(&[0.0], options, |x: &[f64]| Ok((x[0] - 3.0).powi(2))).unwrap();
        assert_eq!(report.termination, Stop::IterationLimit);
    }
}
