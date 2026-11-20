use nc_optimize::*;
use nc_sparse::CooMatrix;

fn box_bound(lower: f64, upper: f64) -> Bound {
    Bound {
        lower: Some(lower),
        upper: Some(upper),
    }
}

fn master(cost: f64, upper: f64, integer: bool) -> Problem {
    Problem {
        objective: vec![cost],
        constraints: CooMatrix::new(0, 1).to_csr().unwrap(),
        row_bounds: vec![],
        var_bounds: vec![box_bound(0.0, upper)],
        is_integer: vec![integer],
    }
}

fn square_from_target(column: usize, target: f64) -> NonlinearExpression {
    use NonlinearNode::*;
    NonlinearExpression::new(
        vec![
            Parameter(column),
            Constant(target),
            Subtract(0, 1),
            Powf(2, 2.0),
        ],
        3,
    )
}

fn production_block() -> NonlinearRecourseBlock {
    use NonlinearNode::*;
    NonlinearRecourseBlock {
        id: "production".into(),
        shared_columns: vec![0],
        local_bounds: vec![box_bound(0.0, 4.0)],
        weight: 1.0,
        objective: square_from_target(1, 3.0),
        constraints: vec![NonlinearConstraint {
            expression: NonlinearExpression::new(
                vec![Parameter(1), Parameter(0), Subtract(0, 1)],
                2,
            ),
            bound: Bound {
                lower: None,
                upper: Some(0.0),
            },
        }],
    }
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} differs from {b}");
}

#[test]
fn continuous_master_projects_local_recourse_and_matches_analytical_optimum() {
    let mut constant = production_block();
    constant.id = "independent".into();
    constant.shared_columns.clear();
    constant.local_bounds.clear();
    constant.objective = NonlinearExpression::new(vec![NonlinearNode::Constant(2.0)], 0);
    constant.constraints.clear();
    constant.weight = 0.5;
    let problem = NonlinearDecompositionProblem {
        master: master(2.0, 4.0, false),
        blocks: vec![production_block(), constant],
    };
    let result = solve_nonlinear_decomposition(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        NonlinearDecompositionTermination::Converged,
        "{result:?}"
    );
    let best = result.incumbent.unwrap();
    assert_eq!(best.recourse.len(), 2);
    close(best.master_point[0], 2.0, 0.004);
    close(best.recourse[0].local_point[0], 2.0, 0.004);
    close(best.objective, 6.0, 1e-5);
    close(result.lower_bound.unwrap(), 6.0, 1e-5);
    assert!(result.optimality_cuts > 0);
    assert!(best.recourse[0].maximum_violation <= 1e-8);
}

fn nonlinear_feasibility_problem() -> NonlinearDecompositionProblem {
    use NonlinearNode::*;
    NonlinearDecompositionProblem {
        master: master(2.0, 1.0, true),
        blocks: vec![NonlinearRecourseBlock {
            id: "capacity".into(),
            shared_columns: vec![0],
            local_bounds: vec![box_bound(0.0, 1.0)],
            weight: 1.0,
            objective: square_from_target(1, 0.0),
            constraints: vec![NonlinearConstraint {
                // x >= 0.5 + y²
                expression: NonlinearExpression::new(
                    vec![
                        Parameter(0),
                        Parameter(1),
                        Powf(1, 2.0),
                        Constant(0.5),
                        Add(2, 3),
                        Subtract(0, 4),
                    ],
                    5,
                ),
                bound: Bound {
                    lower: Some(0.0),
                    upper: None,
                },
            }],
        }],
    }
}

#[test]
fn convex_recourse_infeasibility_produces_a_global_cut_for_integer_master() {
    let mut problem = nonlinear_feasibility_problem();
    let result = solve_nonlinear_decomposition(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        NonlinearDecompositionTermination::Converged,
        "{result:?}"
    );
    assert!(result.feasibility_cuts > 0);
    let best = result.incumbent.unwrap();
    assert_eq!(best.master_point, vec![1.0]);
    close(best.objective, 2.0, 1e-5);
    close(result.lower_bound.unwrap(), 2.0, 1e-5);

    problem.master.var_bounds[0] = Bound::fixed(0.0);
    let infeasible = solve_nonlinear_decomposition(&problem, Default::default()).unwrap();
    assert_eq!(
        infeasible.termination,
        NonlinearDecompositionTermination::Infeasible,
        "{infeasible:?}"
    );
    assert!(infeasible.incumbent.is_none());
}

#[test]
fn recourse_limit_preserves_lower_bound_and_curvature_is_checked() {
    let mut block = production_block();
    block.constraints.clear();
    block.objective = square_from_target(1, 1.0);
    let mut problem = NonlinearDecompositionProblem {
        master: master(1.0, 1.0, false),
        blocks: vec![block],
    };
    let options = NonlinearDecompositionOptions {
        max_block_solves: 1,
        ..Default::default()
    };
    let result = solve_nonlinear_decomposition(&problem, options).unwrap();
    assert_eq!(
        result.termination,
        NonlinearDecompositionTermination::RecourseLimit,
        "{result:?}"
    );
    assert_eq!(result.blocking_block.as_deref(), Some("production"));
    assert!(result.lower_bound.unwrap() < 0.0);
    assert!(result.incumbent.is_none());
    problem.blocks[0]
        .objective
        .nodes
        .push(NonlinearNode::Negate(3));
    problem.blocks[0].objective.output = 4;
    assert!(solve_nonlinear_decomposition(&problem, Default::default()).is_err());
}

#[test]
fn projected_slopes_follow_shared_column_order() {
    use NonlinearNode::*;
    let problem = NonlinearDecompositionProblem {
        master: Problem {
            objective: vec![0.5, 1.0],
            constraints: CooMatrix::new(0, 2).to_csr().unwrap(),
            row_bounds: vec![],
            var_bounds: vec![box_bound(0.0, 3.0); 2],
            is_integer: vec![false; 2],
        },
        blocks: vec![NonlinearRecourseBlock {
            id: "penalty".into(),
            shared_columns: vec![1, 0],
            local_bounds: vec![],
            weight: 0.5,
            objective: NonlinearExpression::new(
                vec![
                    Parameter(0),
                    Parameter(1),
                    Constant(2.0),
                    Subtract(0, 2),
                    Subtract(1, 2),
                    Powf(3, 2.0),
                    Powf(4, 2.0),
                    Add(5, 6),
                ],
                7,
            ),
            constraints: vec![],
        }],
    };
    let result = solve_nonlinear_decomposition(&problem, Default::default()).unwrap();
    assert_eq!(
        result.termination,
        NonlinearDecompositionTermination::Converged,
        "{result:?}"
    );
    let best = result.incumbent.unwrap();
    close(best.master_point[0], 1.5, 0.005);
    close(best.master_point[1], 1.0, 0.005);
    close(best.objective, 2.375, 1e-5);

    let limited = solve_nonlinear_decomposition(
        &problem,
        NonlinearDecompositionOptions {
            max_master_solves: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        limited.termination,
        NonlinearDecompositionTermination::MasterLimit
    );
    assert!(limited.incumbent.unwrap().objective > 2.375);
    assert!(limited.lower_bound.unwrap() <= 2.375);
}
