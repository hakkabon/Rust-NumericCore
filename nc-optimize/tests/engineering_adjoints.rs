use nc_optimize::{NonlinearNode::*, *};

fn equation(row: usize, nodes: Vec<NonlinearNode>) -> EngineeringEquation {
    EngineeringEquation {
        row,
        expression: NonlinearExpression::new(nodes.clone(), nodes.len() - 1),
    }
}

fn unknown(name: String, role: EngineeringUnknownRole) -> EngineeringUnknown {
    EngineeringUnknown {
        name,
        unit: "1".into(),
        role,
        scale: 1.0,
        initial_value: 0.0,
        initial_derivative: 0.0,
    }
}

fn port(i: usize) -> EngineeringPort {
    EngineeringPort {
        name: format!("p{i}"),
        unknown: i,
        orientation: 1.0,
    }
}

fn dae(p: f64) -> EngineeringModel {
    EngineeringModel::new(
        vec![
            unknown("x".into(), EngineeringUnknownRole::Differential),
            unknown("z".into(), EngineeringUnknownRole::Algebraic),
        ],
        vec![EngineeringParameter {
            name: "load".into(),
            unit: "1".into(),
            value: p,
        }],
        vec![EngineeringElement {
            name: "model".into(),
            kind: "index-1".into(),
            ports: vec![port(0), port(1)],
            equations: vec![
                // Inputs are t, x, z, xdot, zdot, p.
                equation(
                    0,
                    vec![
                        Parameter(3),
                        Parameter(1),
                        Add(0, 1),
                        Parameter(5),
                        Subtract(2, 3),
                    ],
                ),
                equation(
                    1,
                    vec![
                        Parameter(2),
                        Parameter(1),
                        Constant(2.0),
                        Multiply(1, 2),
                        Subtract(0, 3),
                    ],
                ),
            ],
        }],
    )
    .unwrap()
}

fn responses() -> Vec<EngineeringTrajectoryResponse> {
    vec![
        EngineeringTrajectoryResponse {
            name: "response".into(),
            // Running term x² + 0.1 zdot. Terminal term z + 0.25 xdot + p².
            running: Some(
                equation(
                    0,
                    vec![
                        Parameter(1),
                        Multiply(0, 0),
                        Constant(0.1),
                        Parameter(4),
                        Multiply(2, 3),
                        Add(1, 4),
                    ],
                )
                .expression,
            ),
            terminal: Some(
                equation(
                    0,
                    vec![
                        Parameter(2),
                        Parameter(3),
                        Constant(0.25),
                        Multiply(1, 2),
                        Add(0, 3),
                        Parameter(5),
                        Multiply(5, 5),
                        Add(4, 6),
                    ],
                )
                .expression,
            ),
        },
        EngineeringTrajectoryResponse {
            name: "terminal-state".into(),
            running: None,
            terminal: Some(NonlinearExpression::new(vec![Parameter(1)], 0)),
        },
    ]
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

#[test]
fn checkpointed_dae_adjoint_matches_full_tape_and_central_differences() {
    let times = [0.0, 0.05, 0.2, 0.38, 0.5, 0.71, 1.0];
    let y0 = [0.4, 0.8];
    let run = |p, y: &[f64], stride| {
        EngineeringTrajectory::integrate(
            dae(p),
            &times,
            y,
            EngineeringTrajectoryOptions {
                checkpoint_stride: stride,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let p = 1.7;
    let compact = run(p, &y0, 2);
    let full = run(p, &y0, 1);
    assert_eq!(
        compact
            .checkpoints()
            .iter()
            .map(|c| c.step)
            .collect::<Vec<_>>(),
        [0, 2, 4, 6]
    );
    let objectives = responses();
    let result = compact.adjoints(&objectives).unwrap();
    let replay_free = full.adjoints(&objectives).unwrap();
    assert_eq!(result.responses, replay_free.responses);
    assert_eq!(result.replayed_steps, 3);
    assert_eq!(result.peak_segment_states, 3);
    assert_eq!(result.transpose_solves, 12);
    assert_eq!(replay_free.replayed_steps, 0);
    assert!(result.maximum_primal_residual < 1e-12);
    assert!(result.responses[0].maximum_adjoint_backward_error < 1e-12);

    let mut x = y0[0];
    let mut value = 0.0;
    for pair in times.windows(2) {
        let h = pair[1] - pair[0];
        let next = (x + h * p) / (1.0 + h);
        value += h * next * next + 0.2 * (next - x);
        x = next;
    }
    near(
        result.responses[0].value,
        value + 2.0 * x + 0.25 * (p - x) + p * p,
        1e-12,
    );
    let step = 1e-5;
    let value_at = |p, y: &[f64]| run(p, y, 2).adjoints(&objectives).unwrap().responses[0].value;
    near(
        result.responses[0].parameter_gradient[0],
        (value_at(p + step, &y0) - value_at(p - step, &y0)) / (2.0 * step),
        1e-8,
    );
    for i in 0..y0.len() {
        let (mut upper, mut lower) = (y0, y0);
        upper[i] += step;
        lower[i] -= step;
        near(
            result.responses[0].initial_state_gradient[i],
            (value_at(p, &upper) - value_at(p, &lower)) / (2.0 * step),
            1e-8,
        );
    }
}

#[test]
fn sparse_point_adjoint_handles_one_parameter_per_state() {
    let n = 1024;
    let model = EngineeringModel::new(
        (0..n)
            .map(|i| unknown(format!("y{i}"), EngineeringUnknownRole::Algebraic))
            .collect(),
        (0..n)
            .map(|i| EngineeringParameter {
                name: format!("load{i}"),
                unit: "1".into(),
                value: if i == 0 { 2.0 } else { 1.0 },
            })
            .collect(),
        (0..n)
            .map(|i| {
                let (ports, nodes) = if i == 0 {
                    (
                        vec![port(i)],
                        vec![
                            Constant(2.0),
                            Parameter(1),
                            Multiply(0, 1),
                            Parameter(3),
                            Subtract(2, 3),
                        ],
                    )
                } else {
                    (
                        vec![port(i), port(i - 1)],
                        vec![
                            Constant(2.0),
                            Parameter(1),
                            Multiply(0, 1),
                            Parameter(2),
                            Subtract(2, 3),
                            Parameter(5 + i),
                            Subtract(4, 5),
                        ],
                    )
                };
                EngineeringElement {
                    name: format!("row{i}"),
                    kind: "banded".into(),
                    ports,
                    equations: vec![equation(i, nodes)],
                }
            })
            .collect(),
    )
    .unwrap();
    let mut seed = vec![0.0; n];
    seed[n - 1] = 1.0;
    let point = vec![1.0; n];
    let result = model
        .adjoint_sensitivities(
            0.0,
            &point,
            &vec![0.0; n],
            0.0,
            &[EngineeringResponseGradient {
                name: "last-state".into(),
                state_gradient: seed,
                direct_parameter_gradient: vec![0.0; n],
            }],
        )
        .unwrap();
    assert_eq!(result.operating_residual_norm, 0.0);
    let response = &result.responses[0];
    assert_eq!(
        &response.parameter_gradient[n - 4..],
        &[0.0625, 0.125, 0.25, 0.5]
    );
    assert_eq!(response.adjoint_backward_error, 0.0);
}

#[test]
fn invalid_grid_and_singular_transpose_fail_closed() {
    assert!(EngineeringTrajectory::integrate(
        dae(1.0),
        &[0.0, 0.0],
        &[0.0, 0.0],
        Default::default()
    )
    .is_err());
    let mut model = dae(1.0);
    model.elements[0].equations[1] = equation(1, vec![Constant(0.0)]);
    let trajectory =
        EngineeringTrajectory::integrate(model, &[0.0, 1.0], &[1.0, 0.0], Default::default())
            .unwrap();
    assert!(trajectory.adjoints(&responses()).is_err());
}
