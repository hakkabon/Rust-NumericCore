use nc_optimize::*;
use nc_sparse::{CooMatrix, CsrMatrix};

fn lower(value: f64) -> Bound {
    Bound {
        lower: Some(value),
        upper: None,
    }
}
fn upper(value: f64) -> Bound {
    Bound {
        lower: None,
        upper: Some(value),
    }
}
fn boxed(lower: f64, upper: f64) -> Bound {
    Bound {
        lower: Some(lower),
        upper: Some(upper),
    }
}

fn matrix(rows: usize, columns: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut result = CooMatrix::new(rows, columns);
    for &(row, col, value) in entries {
        result.push(row, col, value).unwrap();
    }
    result.to_csr().unwrap()
}

fn lp(c: Vec<f64>, a: CsrMatrix<f64>, rows: Vec<Bound>, variables: Vec<Bound>) -> Problem {
    Problem {
        is_integer: vec![false; c.len()],
        objective: c,
        constraints: a,
        row_bounds: rows,
        var_bounds: variables,
    }
}

fn first_stage(objective: f64, bound: Bound) -> Problem {
    lp(vec![objective], matrix(0, 1, &[]), vec![], vec![bound])
}

fn scenario(name: &str, probability: f64, demand: f64, variable: Bound) -> BendersScenario {
    BendersScenario {
        id: LpSubproblemID {
            name: name.into(),
            scenario: None,
        },
        probability,
        recourse: lp(
            vec![4.0],
            matrix(1, 1, &[(0, 0, 1.0)]),
            vec![lower(demand)],
            vec![variable],
        ),
        coupling: matrix(1, 1, &[(0, 0, 1.0)]),
    }
}

fn reference(problem: &BendersProblem) -> Solution {
    let first = &problem.first_stage;
    let mut cost = first.objective.clone();
    let mut vars = first.var_bounds.clone();
    let mut rows = first.row_bounds.clone();
    let mut entries: Vec<_> = first.constraints.iter_entries().collect();
    for scenario in &problem.scenarios {
        let offset_col = vars.len();
        let offset_row = rows.len();
        cost.extend(
            scenario
                .recourse
                .objective
                .iter()
                .map(|v| scenario.probability * v),
        );
        vars.extend(&scenario.recourse.var_bounds);
        rows.extend(&scenario.recourse.row_bounds);
        entries.extend(
            scenario
                .coupling
                .iter_entries()
                .map(|(i, j, v)| (offset_row + i, j, v)),
        );
        entries.extend(
            scenario
                .recourse
                .constraints
                .iter_entries()
                .map(|(i, j, v)| (offset_row + i, offset_col + j, v)),
        );
    }
    RevisedSimplexSolver::default()
        .solve(&lp(
            cost,
            matrix(rows.len(), vars.len(), &entries),
            rows,
            vars,
        ))
        .unwrap()
}

#[test]
fn weighted_single_and_multi_cut_match_extensive_form() {
    let problem = BendersProblem {
        first_stage: first_stage(1.5, boxed(0.0, 10.0)),
        scenarios: vec![
            scenario("low", 0.5, 2.0, lower(0.0)),
            scenario("high", 0.5, 6.0, lower(0.0)),
        ],
    };
    let reference = reference(&problem);
    assert_eq!(reference.status, SolveStatus::Optimal);
    for cut_strategy in [BendersCutStrategy::Single, BendersCutStrategy::Multi] {
        let options = BendersOptions {
            cut_strategy,
            ..Default::default()
        };
        let serial = solve_benders(&problem, options).unwrap();
        let parallel = solve_benders(
            &problem,
            BendersOptions {
                scenario_threads: 2,
                ..options
            },
        )
        .unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(
            serial.termination,
            BendersTermination::Converged,
            "{serial:#?}"
        );
        let incumbent = serial.incumbent.as_ref().unwrap();
        assert!((incumbent.objective - reference.objective_value).abs() < 1e-7);
        assert!((incumbent.first_stage[0] - 6.0).abs() < 1e-7);
        assert!(serial.history.iter().any(|row| row.warm_subproblems == 2));
        for pair in serial.history.windows(2) {
            assert!(pair[1].lower_bound >= pair[0].lower_bound);
            if let (Some(old), Some(new)) = (pair[0].upper_bound, pair[1].upper_bound) {
                assert!(new <= old);
            }
        }
    }
    let stabilized = solve_benders(
        &problem,
        BendersOptions {
            trust_region_radius: Some(1.0),
            maximum_inactive_rounds: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        stabilized.termination,
        BendersTermination::Converged,
        "{stabilized:#?}"
    );
    assert!((stabilized.upper_bound.unwrap() - reference.objective_value).abs() < 1e-7);
    assert!(stabilized.history.iter().any(|row| row.used_trust_region));
}

#[test]
fn ages_inactive_cuts_without_losing_certified_bounds() {
    let problem = BendersProblem::classical(
        first_stage(1.5, boxed(0.0, 10.0)),
        lp(
            vec![1.0],
            matrix(3, 1, &[(0, 0, 1.0), (1, 0, 1.0), (2, 0, 1.0)]),
            vec![lower(12.0), lower(9.0), lower(5.0)],
            vec![lower(0.0)],
        ),
        matrix(3, 1, &[(0, 0, 4.0), (1, 0, 2.0), (2, 0, 1.0)]),
    );
    let result = solve_benders(
        &problem,
        BendersOptions {
            maximum_inactive_rounds: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        result.termination,
        BendersTermination::Converged,
        "{result:#?}"
    );
    assert!(result.history.iter().any(|row| row.cuts_pruned > 0));
    assert!(result
        .history
        .windows(2)
        .all(|pair| pair[1].lower_bound >= pair[0].lower_bound));
    assert!((result.upper_bound.unwrap() - reference(&problem).objective_value).abs() < 1e-8);
}

#[test]
fn classical_feasibility_cuts_and_infeasible_master() {
    for (capacity, expected) in [
        (10.0, BendersTermination::Converged),
        (1.0, BendersTermination::Infeasible),
    ] {
        let s = scenario("shortfall", 1.0, 3.0, boxed(0.0, 1.0));
        let problem = BendersProblem::classical(
            first_stage(1.0, boxed(0.0, capacity)),
            s.recourse,
            s.coupling,
        );
        let result = solve_benders(&problem, Default::default()).unwrap();
        assert_eq!(result.termination, expected, "{result:#?}");
        assert!(result.history[0].feasibility_cuts > 0);
        if expected == BendersTermination::Converged {
            assert!(
                (result.upper_bound.unwrap() - reference(&problem).objective_value).abs() < 1e-8
            );
        } else {
            assert!(result.incumbent.is_none());
            assert_eq!(reference(&problem).status, SolveStatus::Infeasible);
        }
    }
}

#[test]
fn uses_auxiliary_floor_for_negative_cost_and_upper_recourse_row() {
    let problem = BendersProblem::classical(
        first_stage(0.5, boxed(0.0, 5.0)),
        lp(
            vec![-1.0],
            matrix(1, 1, &[(0, 0, 1.0)]),
            vec![upper(3.0)],
            vec![lower(0.0)],
        ),
        matrix(1, 1, &[(0, 0, -1.0)]),
    );
    let result = solve_benders(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        BendersTermination::Converged,
        "{result:#?}"
    );
    assert!(
        (result.incumbent.unwrap().objective - reference(&problem).objective_value).abs() < 1e-8
    );
}

#[test]
fn incomplete_runs_preserve_bounds_and_make_no_optimality_claim() {
    let problem = BendersProblem {
        first_stage: first_stage(1.5, boxed(0.0, 10.0)),
        scenarios: vec![scenario("high", 1.0, 6.0, lower(0.0))],
    };
    let result = solve_benders(
        &problem,
        BendersOptions {
            max_iterations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.termination, BendersTermination::IterationLimit);
    assert_eq!(result.lower_bound, Some(0.0));
    assert_eq!(result.upper_bound, Some(24.0));
    assert_eq!(result.absolute_gap, Some(24.0));
    let lp_limited = solve_benders(
        &problem,
        BendersOptions {
            lp_iteration_limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(lp_limited.termination, BendersTermination::LpIterationLimit);
    assert_eq!(
        lp_limited.blocking_scenario,
        Some(problem.scenarios[0].id.clone())
    );
    assert!(lp_limited.upper_bound.is_none());
    let unbounded_master = BendersProblem {
        first_stage: first_stage(-1.0, lower(0.0)),
        ..problem.clone()
    };
    assert_eq!(
        solve_benders(&unbounded_master, Default::default())
            .unwrap()
            .termination,
        BendersTermination::MasterUnbounded
    );
    let mut invalid = problem;
    invalid.scenarios[0].probability = 0.9;
    assert!(solve_benders(&invalid, Default::default()).is_err());
}
