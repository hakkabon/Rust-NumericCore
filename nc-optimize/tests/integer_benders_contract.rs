use nc_optimize::*;
use nc_sparse::{CooMatrix, CsrMatrix};

fn bound(lower: f64, upper: f64) -> Bound {
    Bound {
        lower: Some(lower),
        upper: Some(upper),
    }
}
fn lower(value: f64) -> Bound {
    Bound {
        lower: Some(value),
        upper: None,
    }
}

fn csr(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut matrix = CooMatrix::new(rows, cols);
    for &(i, j, v) in entries {
        matrix.push(i, j, v).unwrap();
    }
    matrix.to_csr().unwrap()
}

fn first(c: Vec<f64>, upper: f64) -> Problem {
    let cols = c.len();
    Problem {
        objective: c,
        is_integer: vec![true; cols],
        row_bounds: vec![],
        constraints: csr(0, cols, &[]),
        var_bounds: vec![bound(0., upper); cols],
    }
}

fn demand(master: Problem, demand: f64, integer: bool) -> BendersProblem {
    BendersProblem::classical(
        master,
        Problem {
            objective: vec![5.],
            is_integer: vec![integer],
            row_bounds: vec![lower(demand)],
            var_bounds: vec![bound(0., 10.)],
            constraints: csr(1, 1, &[(0, 0, 1.)]),
        },
        csr(1, 1, &[(0, 0, 1.)]),
    )
}

fn extensive(problem: &BendersProblem) -> Solution {
    let scenario = &problem.scenarios[0];
    let master = &problem.first_stage;
    let sub = &scenario.recourse;
    let joint = Problem {
        objective: [master.objective[0], sub.objective[0]].to_vec(),
        constraints: csr(1, 2, &[(0, 0, 1.), (0, 1, 1.)]),
        row_bounds: sub.row_bounds.clone(),
        var_bounds: vec![master.var_bounds[0], sub.var_bounds[0]],
        is_integer: vec![true, sub.is_integer[0]],
    };
    BranchAndBoundSolver::default().solve(&joint).unwrap()
}

#[test]
fn mixed_master_branches_with_shared_lp_cuts() {
    let problem = demand(first(vec![2.], 4.), 2.5, false);
    let result = solve_integer_benders(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        IntegerBendersTermination::Converged,
        "{result:#?}"
    );
    assert_eq!(
        result.upper_bound,
        Some(extensive(&problem).objective_value)
    );
    assert_eq!(result.incumbent.unwrap().first_stage, vec![3.]);
    assert!(result.master_nodes > result.candidate_evaluations);
    assert!(result.optimality_cuts >= 1);
    assert!(result.lower_bound.unwrap() <= result.upper_bound.unwrap());
}

#[test]
fn integer_recourse_uses_assignment_evidence_and_distinguishes_limits() {
    let problem = demand(first(vec![1.], 1.), 0.5, true);
    let result = solve_integer_benders(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        IntegerBendersTermination::Converged,
        "{result:#?}"
    );
    assert_eq!(
        result.upper_bound,
        Some(extensive(&problem).objective_value)
    );
    let plan = result.incumbent.unwrap();
    assert_eq!(plan.first_stage, vec![1.]);
    assert_eq!(plan.recourse[0].point, vec![0.]);
    assert!(result.optimality_cuts >= 1);
    assert!(!result.uses_logic_oracle);
    let result = solve_integer_benders(
        &problem,
        IntegerBendersOptions {
            max_recourse_nodes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        result.termination,
        IntegerBendersTermination::RecourseNodeLimit
    );
    assert_eq!(
        result.blocking_scenario,
        Some(problem.scenarios[0].id.clone())
    );
    assert!(result.incumbent.is_none());
    assert_eq!(result.optimality_cuts, 0);
    assert!(
        solve_integer_benders(&demand(first(vec![1.], 3.), 0.5, true), Default::default()).is_err()
    );
}

#[test]
fn integer_recourse_infeasibility_adds_a_no_good_cut() {
    let mut problem = demand(first(vec![1.], 1.), 0.5, true);
    problem.scenarios[0].recourse.row_bounds = vec![Bound::fixed(0.5)];
    problem.scenarios[0].coupling = csr(1, 1, &[(0, 0, 0.5)]);
    let result = solve_integer_benders(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        IntegerBendersTermination::Converged,
        "{result:#?}"
    );
    assert_eq!(result.incumbent.unwrap().first_stage, vec![1.]);
    assert_eq!(result.upper_bound, Some(1.0));
    assert_eq!(result.feasibility_cuts, 1);
}

fn logic_problem() -> IntegerBendersProblem {
    IntegerBendersProblem {
        first_stage: first(vec![1., 2.], 1.),
        scenarios: vec![IntegerBendersScenario::Logic(LogicBendersScenario {
            id: LpSubproblemID {
                name: "schedule".into(),
                scenario: Some("peak".into()),
            },
            probability: 1.0,
            binary_columns: vec![0, 1],
            objective_lower_bound: 0.0,
        })],
    }
}

#[test]
fn logic_conflict_excludes_all_extensions_of_a_partial_assignment() {
    let problem = logic_problem();
    let mut visited = Vec::new();
    let result = solve_logic_benders(&problem, Default::default(), |id, bits| {
        assert_eq!(id.name, "schedule");
        visited.push(bits.to_vec());
        Ok(if !bits[0] {
            LogicBendersEvaluation::Infeasible {
                conflict: Some(vec![0]),
            }
        } else {
            LogicBendersEvaluation::Optimal {
                objective: if bits[1] { 0. } else { 5. },
                point: vec![],
                validity: Some(vec![1]),
            }
        })
    })
    .unwrap();
    assert_eq!(
        result.termination,
        IntegerBendersTermination::Converged,
        "{result:#?}"
    );
    assert_eq!(result.incumbent.unwrap().first_stage, vec![1., 1.]);
    assert_eq!(result.upper_bound, Some(3.0));
    assert_eq!(visited.iter().filter(|bits| !bits[0]).count(), 1);
    assert_eq!(result.feasibility_cuts, 1);
    assert!(result.uses_logic_oracle);
}

#[test]
fn unresolved_oracles_and_unsearched_nodes_make_no_optimality_claim() {
    let problem = logic_problem();
    let stopped = solve_logic_benders(&problem, Default::default(), |_, _| {
        Ok(LogicBendersEvaluation::Unresolved)
    })
    .unwrap();
    assert_eq!(
        stopped.termination,
        IntegerBendersTermination::OracleUnresolved
    );
    assert_eq!(stopped.lower_bound, Some(0.0));
    assert!(stopped.upper_bound.is_none());
    assert_eq!(stopped.feasibility_cuts, 0);
    let limited = solve_logic_benders(
        &problem,
        IntegerBendersOptions {
            max_master_nodes: 1,
            ..Default::default()
        },
        |_, _| {
            Ok(LogicBendersEvaluation::Optimal {
                objective: 5.,
                point: vec![],
                validity: None,
            })
        },
    )
    .unwrap();
    assert_eq!(
        limited.termination,
        IntegerBendersTermination::MasterNodeLimit
    );
    assert_eq!(limited.lower_bound, Some(0.0));
    assert_eq!(limited.upper_bound, Some(5.0));
    let candidates = solve_logic_benders(
        &problem,
        IntegerBendersOptions {
            max_candidate_evaluations: 1,
            ..Default::default()
        },
        |_, _| {
            Ok(LogicBendersEvaluation::Optimal {
                objective: 5.,
                point: vec![],
                validity: None,
            })
        },
    )
    .unwrap();
    assert_eq!(
        candidates.termination,
        IntegerBendersTermination::CandidateLimit
    );
    assert_eq!(candidates.lower_bound, Some(1.));
    assert_eq!(candidates.upper_bound, Some(5.));
    assert!(solve_logic_benders(&problem, Default::default(), |_, _| {
        Ok(LogicBendersEvaluation::Infeasible {
            conflict: Some(vec![2]),
        })
    })
    .is_err());
}
