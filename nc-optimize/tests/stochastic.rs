use nc_optimize::*;
use nc_sparse::{CooMatrix, CsrMatrix};

fn matrix(rows: usize) -> CsrMatrix<f64> {
    let mut matrix = CooMatrix::new(rows, 1);
    if rows > 0 {
        matrix.push(0, 0, 1.0).unwrap();
    }
    matrix.to_csr().unwrap()
}

fn block(cost: f64, demand: Option<f64>) -> Problem {
    let row_bounds: Vec<_> = demand
        .into_iter()
        .map(|lower| Bound {
            lower: Some(lower),
            upper: None,
        })
        .collect();
    Problem {
        objective: vec![cost],
        constraints: matrix(row_bounds.len()),
        row_bounds,
        var_bounds: vec![Bound {
            lower: Some(0.0),
            upper: Some(10.0),
        }],
        is_integer: vec![false],
    }
}

fn problem() -> BendersProblem {
    BendersProblem {
        first_stage: block(1.5, None),
        scenarios: [("normal", 0.8, 2.0), ("peak", 0.2, 6.0)]
            .into_iter()
            .map(|(name, probability, demand)| BendersScenario {
                id: LpSubproblemID {
                    name: name.into(),
                    scenario: None,
                },
                probability,
                recourse: block(4.0, Some(demand)),
                coupling: matrix(1),
            })
            .collect(),
    }
}

fn close(observed: f64, expected: f64, tol: f64) {
    assert!(
        (observed - expected).abs() < tol,
        "{observed} differs from {expected}"
    );
}

#[test]
fn expected_value_and_cvar_choose_different_plans() {
    let tree = ScenarioTree::two_stage(&problem());
    let expected =
        solve_scenario_tree(&tree, ScenarioRisk::ExpectedValue, Default::default()).unwrap();
    assert_eq!(expected.termination, ScenarioTermination::Converged);
    close(expected.lower_bound.unwrap(), 6.2, 1e-7);
    let policy = expected.policy.unwrap();
    close(policy.decisions[0][0], 2.0, 1e-7);
    close(policy.expected_cost, 6.2, 1e-7);
    let risk_averse = solve_scenario_tree(
        &tree,
        ScenarioRisk::MeanCvar {
            alpha: 0.8,
            weight: 1.0,
        },
        Default::default(),
    )
    .unwrap();
    assert_eq!(risk_averse.termination, ScenarioTermination::Converged);
    close(risk_averse.lower_bound.unwrap(), 9.0, 1e-7);
    let policy = risk_averse.policy.unwrap();
    close(policy.decisions[0][0], 6.0, 1e-7);
    close(policy.conditional_value_at_risk.unwrap(), 9.0, 1e-7);
    let diagnostic = solve_scenario_tree(
        &tree,
        ScenarioRisk::MeanCvar {
            alpha: 0.5,
            weight: 0.0,
        },
        Default::default(),
    )
    .unwrap()
    .policy
    .unwrap();
    close(diagnostic.value_at_risk.unwrap(), 3.0, 1e-7);
    close(diagnostic.conditional_value_at_risk.unwrap(), 9.4, 1e-7);
}

#[test]
fn multistage_decisions_are_shared_and_ancestor_links_are_checked() {
    let mut root = block(1.0, None);
    root.var_bounds[0] = Bound::fixed(1.0);
    let mut tree = ScenarioTree {
        nodes: vec![
            ScenarioNode {
                id: "root".into(),
                parent: None,
                conditional_probability: 1.0,
                block: root,
                links: vec![],
            },
            ScenarioNode {
                id: "shared".into(),
                parent: Some(0),
                conditional_probability: 1.0,
                block: block(1.0, None),
                links: vec![],
            },
        ],
    };
    for (id, probability, demand) in [("low", 0.8, 3.0), ("high", 0.2, 5.0)] {
        tree.nodes.push(ScenarioNode {
            id: id.into(),
            parent: Some(1),
            conditional_probability: probability,
            block: block(4.0, Some(demand)),
            links: [0, 1]
                .into_iter()
                .map(|ancestor| ScenarioLink {
                    ancestor,
                    coefficients: matrix(1),
                })
                .collect(),
        });
    }
    let run = |tree: &ScenarioTree| {
        solve_scenario_tree(tree, ScenarioRisk::ExpectedValue, Default::default())
    };
    let result = run(&tree).unwrap();
    assert_eq!(result.termination, ScenarioTermination::Converged);
    let policy = result.policy.unwrap();
    assert_eq!(
        policy.decisions,
        vec![vec![1.0], vec![2.0], vec![0.0], vec![2.0]]
    );
    close(policy.expected_cost, 4.6, 1e-7);
    tree.nodes[3].links[0].ancestor = 2;
    assert!(run(&tree).is_err());
    tree.nodes[3].links[0].ancestor = 0;
    tree.nodes[3].conditional_probability = 0.3;
    assert!(run(&tree).is_err());
}

#[test]
fn extensive_form_branches_on_bounded_integers() {
    let mut problem = problem();
    problem.first_stage.is_integer[0] = true;
    problem.scenarios.truncate(1);
    problem.scenarios[0].probability = 1.0;
    problem.scenarios[0].recourse = block(4.0, Some(3.5));
    let tree = ScenarioTree::two_stage(&problem);
    let limited = solve_scenario_tree(
        &tree,
        ScenarioRisk::ExpectedValue,
        ScenarioSolveOptions {
            max_nodes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(limited.termination, ScenarioTermination::NodeLimit);
    close(limited.lower_bound.unwrap(), 5.25, 1e-7);
    assert!(limited.policy.is_none());
    let result =
        solve_scenario_tree(&tree, ScenarioRisk::ExpectedValue, Default::default()).unwrap();
    assert_eq!(result.termination, ScenarioTermination::Converged);
    assert_eq!(result.policy.unwrap().decisions[0], vec![4.0]);
    close(result.lower_bound.unwrap(), 6.0, 1e-7);
}

#[test]
fn sampling_reweights_draws_and_estimates_a_fixed_policy_mean() {
    let problem = problem();
    let samples = draw_scenarios(&[0.8, 0.2], 1000, 123).unwrap();
    assert_eq!(samples, draw_scenarios(&[0.8, 0.2], 1000, 123).unwrap());
    assert_ne!(samples, draw_scenarios(&[0.8, 0.2], 1000, 456).unwrap());
    assert!(draw_scenarios(&[1.0, 1.0], 2, 123).is_err());
    assert!(draw_scenarios(&[1.0], usize::MAX, 123).is_err());
    let sampled = sample_average_problem(&problem, &[0, 1, 1, 1]).unwrap();
    assert_eq!(sampled.scenarios[0].probability, 0.25);
    assert_eq!(sampled.scenarios[1].probability, 0.75);
    let stats =
        evaluate_scenario_sample(&problem, &[2.0], &[0, 0, 0, 1], Default::default()).unwrap();
    close(stats.mean, 7.0, 1e-7);
    close(stats.standard_error, 4.0, 1e-7);
    assert!(stats.confidence_interval_95.0 < stats.mean);
    assert!(evaluate_scenario_sample(&problem, &[11.0], &[0, 1], Default::default()).is_err());
}

#[test]
fn reduction_preserves_mass_and_reports_distortion() {
    let reduced =
        reduce_scenarios(&[vec![0.0], vec![1.0], vec![10.0]], &[0.4, 0.4, 0.2], 2).unwrap();
    assert_eq!(reduced.retained, vec![(1, 0.8), (2, 0.2)]);
    assert_eq!(reduced.assignment, vec![1, 1, 2]);
    close(reduced.weighted_distance, 0.4, 1e-7);
}

#[test]
fn progressive_hedging_replays_across_checkpoints_and_worker_counts() {
    let problem = problem();
    let options = ProgressiveHedgingOptions {
        rho: 1.5,
        tolerance: 1e-5,
        ..Default::default()
    };
    let full = solve_progressive_hedging(&problem, options, None).unwrap();
    assert_eq!(full.termination, HedgingTermination::Converged);
    let first = solve_progressive_hedging(
        &problem,
        ProgressiveHedgingOptions {
            max_iterations: 3,
            ..options
        },
        None,
    )
    .unwrap();
    assert_eq!(first.termination, HedgingTermination::IterationLimit);
    let resumed = solve_progressive_hedging(
        &problem,
        ProgressiveHedgingOptions {
            scenario_threads: 2,
            ..options
        },
        first.checkpoint.clone(),
    )
    .unwrap();
    assert_eq!(resumed.termination, HedgingTermination::Converged);
    assert_eq!(resumed.primal_residual, full.primal_residual);
    let checkpoint = resumed.checkpoint.unwrap();
    assert_eq!(checkpoint.iterations, full.checkpoint.unwrap().iterations);
    close(checkpoint.consensus[0], 2.0, 1e-4);
    assert!(solve_progressive_hedging(
        &problem,
        ProgressiveHedgingOptions {
            rho: 2.0,
            ..options
        },
        first.checkpoint
    )
    .is_err());
}
