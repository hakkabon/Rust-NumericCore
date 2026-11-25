use nc_optimize::{
    solve_nonlinear_system, solve_powell_hybrid, solve_powell_hybrid_model, NonlinearEquationModel,
    NonlinearExpression, NonlinearNode, NonlinearSystemEvaluation, NonlinearSystemOptions,
    NonlinearSystemTermination, OptimizeError, PowellHybridOptions as Options,
    PowellHybridTermination as Stop,
};

#[test]
fn solves_minpack_tridiagonal_system_and_compares_to_damped_newton() {
    // Nine-variable example in the MINPACK HYBRJ documentation.
    let residual = |x: &[f64]| {
        Ok((0..x.len())
            .map(|i| {
                let lower = if i > 0 { x[i - 1] } else { 0.0 };
                let upper = x.get(i + 1).copied().unwrap_or(0.0);
                (3.0 - 2.0 * x[i]) * x[i] - lower - 2.0 * upper + 1.0
            })
            .collect())
    };
    let jacobian = |x: &[f64]| {
        let mut j = vec![vec![0.0; x.len()]; x.len()];
        for (i, row) in j.iter_mut().enumerate() {
            row[i] = 3.0 - 4.0 * x[i];
            if i > 0 {
                row[i - 1] = -1.0;
            }
            if i + 1 < x.len() {
                row[i + 1] = -2.0;
            }
        }
        Ok(j)
    };
    let initial = [-1.0; 9];
    let mut residual_calls = 0;
    let mut jacobian_calls = 0;
    let hybrid = solve_powell_hybrid(
        &initial,
        Options::default(),
        |x| {
            residual_calls += 1;
            residual(x)
        },
        |x| {
            jacobian_calls += 1;
            jacobian(x)
        },
    )
    .unwrap();
    let mut newton_calls = 0;
    let newton = solve_nonlinear_system(&initial, NonlinearSystemOptions::default(), |x| {
        newton_calls += 1;
        Ok(NonlinearSystemEvaluation {
            residuals: residual(x)?,
            jacobian: jacobian(x)?,
        })
    })
    .unwrap();
    eprintln!("MINPACK tridiagonal: hybrid {residual_calls} residuals, {jacobian_calls} Jacobians; damped Newton {newton_calls} combined calls");
    assert_eq!(hybrid.termination, Stop::ConvergedResidual);
    assert_eq!(
        newton.termination,
        NonlinearSystemTermination::ConvergedResidual
    );
    assert_eq!(hybrid.residual_evaluations, residual_calls);
    assert_eq!(hybrid.jacobian_evaluations, jacobian_calls);
    assert!(hybrid.rank_one_updates > 0);
    assert!(jacobian_calls < newton_calls);
    assert!(hybrid.residual_norm <= 1e-9);
    assert!(hybrid
        .point
        .iter()
        .zip(newton.point)
        .all(|(a, b)| (a - b).abs() < 1e-8));
    assert!((hybrid.point[0] + 0.5706545).abs() < 1e-6);
    assert!((hybrid.point[8] + 0.4164121).abs() < 1e-6);
}

#[test]
fn globalizes_rosenbrock_equations() {
    let result = solve_powell_hybrid(
        &[-1.2, 1.0],
        Options::default(),
        |x| Ok(vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]),
        |x| Ok(vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]]),
    )
    .unwrap();
    assert_eq!(result.termination, Stop::ConvergedResidual, "{result:?}");
    assert!(result.point.iter().all(|p| (p - 1.0).abs() < 1e-8));
    assert!(result.rank_one_updates > 2);
    assert!(result.accepted_steps < result.iterations);
}

#[test]
fn scaled_rank_deficient_system_can_converge_and_nonroot_stationarity_stalls() {
    let scaled = Options {
        variable_scale: vec![1e6, 1e-6],
        residual_scale: vec![1e-8, 1e-8],
        ..Options::default()
    };
    let root = solve_powell_hybrid(
        &[0.0, 0.0],
        scaled,
        |x| {
            let eq = 1e8 * (x[0] / 1e6 + x[1] / 1e-6 - 2.0);
            Ok(vec![eq, eq])
        },
        |_| Ok(vec![vec![1e2, 1e14]; 2]),
    )
    .unwrap();
    assert_eq!(root.termination, Stop::ConvergedResidual);
    assert!((root.point[0] / 1e6 - 1.0).abs() < 1e-9);
    assert!((root.point[1] / 1e-6 - 1.0).abs() < 1e-9);

    let stalled = solve_powell_hybrid(
        &[0.0],
        Options::default(),
        |x| Ok(vec![1.0 + x[0] * x[0]]),
        |x| Ok(vec![vec![2.0 * x[0]]]),
    )
    .unwrap();
    assert_eq!(stalled.termination, Stop::Stalled);
    assert_eq!(stalled.residual_norm, 1.0);
    assert_eq!(stalled.residual_evaluations, 1);
}

#[test]
fn enforces_budgets_and_preserves_callback_errors() {
    let residual = |x: &[f64]| Ok(vec![x[0] - 2.0]);
    let jacobian = |_: &[f64]| Ok(vec![vec![1.0]]);
    let report = solve_powell_hybrid(
        &[0.0],
        Options {
            max_residual_evaluations: 1,
            ..Options::default()
        },
        residual,
        |_| panic!("no Jacobian refresh after evaluation budget"),
    )
    .unwrap();
    assert_eq!(report.termination, Stop::EvaluationLimit);
    assert_eq!(report.jacobian_evaluations, 0);
    let report = solve_powell_hybrid(
        &[0.0],
        Options {
            max_iterations: 1,
            ..Options::default()
        },
        residual,
        jacobian,
    )
    .unwrap();
    assert_eq!(report.termination, Stop::IterationLimit);
    assert_eq!(report.residual_evaluations, 2);
    let root = solve_powell_hybrid(&[2.0], Options::default(), residual, |_| {
        panic!("no Jacobian at an initially converged root")
    })
    .unwrap();
    assert_eq!(root.termination, Stop::ConvergedResidual);
    let error = solve_powell_hybrid(&[0.0], Options::default(), residual, |_| {
        Err(OptimizeError::InvalidConfiguration("callback error"))
    })
    .unwrap_err();
    assert!(matches!(
        error,
        OptimizeError::InvalidConfiguration("callback error")
    ));
    assert!(
        solve_powell_hybrid(&[0.0], Options::default(), residual, |_| Ok(vec![vec![]])).is_err()
    );
    assert!(
        solve_powell_hybrid(&[0.0], Options::default(), |_| Ok(vec![f64::NAN]), jacobian).is_err()
    );
}

#[test]
fn solves_expression_graph_without_requesting_derivative_at_an_initial_root() {
    let expression =
        NonlinearExpression::new(vec![NonlinearNode::Parameter(0), NonlinearNode::Sqrt(0)], 1);
    let model = NonlinearEquationModel::new(1, vec![expression]).unwrap();
    let result = solve_powell_hybrid_model(&model, &[0.0], Options::default()).unwrap();
    assert_eq!(result.termination, Stop::ConvergedResidual);
    assert_eq!(result.jacobian_evaluations, 0);
}

#[test]
fn poor_steps_force_a_jacobian_refresh_at_an_accepted_point() {
    let initial = [0.05];
    let mut refreshes = Vec::new();
    let result = solve_powell_hybrid(
        &initial,
        Options {
            initial_radius: 10.0,
            ..Options::default()
        },
        |x| Ok(vec![x[0] * x[0] - 1.0]),
        |x| {
            refreshes.push(x[0]);
            Ok(vec![vec![2.0 * x[0]]])
        },
    )
    .unwrap();
    assert_eq!(result.termination, Stop::ConvergedResidual, "{result:?}");
    assert!((result.point[0] - 1.0).abs() < 1e-8);
    assert!(refreshes.len() > 1, "{result:?}");
    assert_eq!(refreshes[0], initial[0]);
    assert!(refreshes[1] > initial[0] && refreshes[1] < 1.0);
    assert_eq!(refreshes.len() as u64, result.jacobian_evaluations);
}
