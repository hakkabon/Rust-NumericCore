use nc_optimize::*;
use nc_sparse::CooMatrix;
use std::collections::BTreeSet;

const TOL: f64 = 1e-8;

#[test]
fn quadratic_workspace_reuses_converged_iterates_across_rhs_updates() {
    let problem = QuadraticProblem {
        quadratic: vec![vec![2.0]],
        linear: vec![-4.0],
        constant: 0.0,
        constraints: CooMatrix::new(0, 1).to_csr().unwrap(),
        row_bounds: vec![],
        variable_bounds: vec![lower(0.0)],
    };
    let mut workspace = QuadraticWorkspace::new(problem, Default::default()).unwrap();
    let first = workspace.solve().unwrap();
    assert_eq!(first.termination, QuadraticTermination::Converged);
    assert!((first.point[0] - 2.0).abs() < 1e-5);
    assert!(workspace.solve().unwrap().iterations < first.iterations);
    workspace
        .update(vec![-6.0], 1.0, vec![], vec![lower(0.0)])
        .unwrap();
    assert!((workspace.solve().unwrap().point[0] - 3.0).abs() < 1e-5);
    assert!(workspace.update(vec![99.0], 1.0, vec![], vec![]).is_err());
    assert_eq!(workspace.problem().linear, vec![-6.0]);
}

fn lower(v: f64) -> Bound {
    Bound {
        lower: Some(v),
        upper: None,
    }
}

fn upper(v: f64) -> Bound {
    Bound {
        lower: None,
        upper: Some(v),
    }
}

fn lp(objective: Vec<f64>, variable_bounds: Vec<Bound>, rows: Vec<LpRow>) -> Problem {
    let mut a = CooMatrix::new(rows.len(), objective.len());
    for (row, spec) in rows.iter().enumerate() {
        for &(column, value) in &spec.entries {
            a.push(row, column, value).unwrap();
        }
    }
    Problem {
        is_integer: vec![false; objective.len()],
        constraints: a.to_csr().unwrap(),
        objective,
        var_bounds: variable_bounds,
        row_bounds: rows.into_iter().map(|row| row.bound).collect(),
    }
}

#[test]
fn reports_ranged_row_duals_and_reuses_basis_across_updates() {
    let problem = lp(
        vec![1.0],
        vec![lower(0.0)],
        vec![LpRow {
            entries: vec![(0, 1.0)],
            bound: Bound {
                lower: Some(2.0),
                upper: Some(6.0),
            },
        }],
    );
    let id = LpSubproblemID {
        name: "recourse".into(),
        scenario: Some("winter".into()),
    };
    let mut ws = LpWorkspace::new(problem.clone(), Some(id.clone())).unwrap();
    let solver = RevisedSimplexSolver::default();
    let first = ws.solve(&solver, TOL).unwrap();
    assert_eq!(first.subproblem, Some(id));
    assert_eq!(first.solution.variable_values, vec![2.0]);
    assert_eq!(first.dual.unwrap().row_multipliers, vec![1.0]);
    assert!(!first.used_warm_start);

    let mut replay = LpWorkspace::new(problem, None).unwrap();
    replay.install_basis(&first.basis).unwrap();
    assert!(replay.solve(&solver, TOL).unwrap().used_warm_start);
    assert!(replay
        .install_basis(&LpBasis {
            variables: vec![],
            rows: vec![]
        })
        .is_err());

    ws.update(LpUpdate {
        objective: vec![(0, -1.0)],
        row_bounds: vec![(0, upper(5.0))],
        variable_bounds: vec![(0, lower(1.0))],
        ..Default::default()
    })
    .unwrap();
    let second = ws.solve(&solver, TOL).unwrap();
    assert!(second.used_warm_start);
    assert_eq!(second.solution.variable_values, vec![5.0]);
    assert_eq!(second.dual.as_ref().unwrap().row_multipliers, vec![-1.0]);
    assert_eq!(second.dual.unwrap().objective_bound, -5.0);

    ws.update(LpUpdate {
        append_rows: vec![LpRow {
            entries: vec![(0, 1.0)],
            bound: upper(3.0),
        }],
        append_columns: vec![LpColumn {
            entries: vec![(1, -1.0)],
            bound: lower(0.0),
            objective: 2.0,
        }],
        ..Default::default()
    })
    .unwrap();
    let third = ws.solve(&solver, TOL).unwrap();
    assert!(third.used_warm_start);
    assert_eq!(third.solution.variable_values, vec![3.0, 0.0]);
    assert_eq!(third.dual.unwrap().reduced_costs, vec![0.0, 1.0]);
    assert!(ws
        .update(LpUpdate {
            objective: vec![(0, 77.0)],
            append_rows: vec![LpRow {
                entries: vec![(2, 1.0)],
                bound: upper(0.0)
            }],
            ..Default::default()
        })
        .is_err());
    assert_eq!(ws.problem().objective, vec![-1.0, 2.0]);
    assert_eq!(ws.problem().row_bounds.len(), 2);
}

#[test]
fn validates_farkas_and_recession_evidence() {
    let solver = RevisedSimplexSolver::default();
    let infeasible = lp(
        vec![1.0],
        vec![Bound::fixed(0.0)],
        vec![LpRow {
            entries: vec![(0, 1.0)],
            bound: lower(1.0),
        }],
    );
    let result = LpWorkspace::new(infeasible.clone(), None)
        .unwrap()
        .solve(&solver, TOL)
        .unwrap();
    assert_eq!(result.solution.status, SolveStatus::Infeasible);
    assert_eq!(result.infeasibility.unwrap().contradiction_margin, 1.0);
    assert!(verify_lp_farkas(&infeasible, &[-1.0], TOL).is_none());

    let unbounded = lp(
        vec![-1.0, 0.0],
        vec![lower(0.0), lower(0.0)],
        vec![LpRow {
            entries: vec![(0, 1.0), (1, -1.0)],
            bound: Bound::fixed(0.0),
        }],
    );
    let result = LpWorkspace::new(unbounded.clone(), None)
        .unwrap()
        .solve(&solver, TOL)
        .unwrap();
    assert_eq!(result.solution.status, SolveStatus::Unbounded);
    assert_eq!(result.unbounded_ray.unwrap().direction, vec![1.0, 1.0]);
    assert!(verify_lp_unbounded_ray(&unbounded, &[0.0, 0.0], &[1.0, 1.0 - 1e-10], TOL).is_none());
    assert!(verify_lp_unbounded_ray(&unbounded, &[0.0, 1.0], &[1.0, 1.0], TOL).is_none());
    assert!(verify_lp_unbounded_ray(&unbounded, &[0.0, 1e-10], &[1.0, 1.0], TOL).is_none());
}

#[test]
fn postsolve_rechecks_dropped_rows_and_tightened_bounds() {
    let original = lp(
        vec![1.0, 2.0],
        vec![lower(0.0), Bound::fixed(3.0)],
        vec![LpRow {
            entries: vec![(1, 1.0)],
            bound: lower(1.0),
        }],
    );
    let reduced = presolve_problem(&original, Default::default()).unwrap();
    assert_eq!(
        reduced
            .lift_lp_dual(&original, &[0.0], &[], TOL)
            .unwrap()
            .objective_bound,
        6.0
    );

    let tightened = lp(
        vec![1.0],
        vec![lower(0.0)],
        vec![LpRow {
            entries: vec![(0, 1.0)],
            bound: lower(1.0),
        }],
    );
    let reduced = presolve_problem(&tightened, Default::default()).unwrap();
    assert!(reduced.lift_lp_dual(&tightened, &[1.0], &[], TOL).is_none());
}

#[test]
fn cut_pool_keeps_nearby_inequalities_and_separate_scopes() {
    let id = LpSubproblemID {
        name: "recourse".into(),
        scenario: None,
    };
    let cut = DecompositionCut {
        entries: vec![(1, 2.0), (0, 1.0)],
        upper: 3.0,
        origin: CutOrigin {
            source: id.clone(),
            iteration: 0,
            kind: CutKind::Optimality,
        },
        scope: CutScope::Global,
    };
    let mut pool = DecompositionCutPool::default();
    assert_eq!(pool.insert(cut.clone()).unwrap(), CutInsertion::Inserted(0));
    let mut duplicate = cut.clone();
    duplicate.entries.reverse();
    assert_eq!(pool.insert(duplicate).unwrap(), CutInsertion::Duplicate(0));
    assert_eq!(
        pool.insert(DecompositionCut {
            upper: 3.0 - 1e-10,
            ..cut.clone()
        })
        .unwrap(),
        CutInsertion::Inserted(1)
    );
    assert_eq!(
        pool.insert(DecompositionCut {
            scope: CutScope::Subproblem(id),
            ..cut
        })
        .unwrap(),
        CutInsertion::Inserted(2)
    );
    pool.age(&BTreeSet::from([1]));
    assert_eq!(pool.prune(0), vec![0, 2]);
    assert_eq!(
        pool.entries().next().unwrap().cut.lp_row().bound.upper,
        Some(3.0 - 1e-10)
    );
}
