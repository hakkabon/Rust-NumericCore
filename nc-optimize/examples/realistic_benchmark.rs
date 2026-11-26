use nc_optimize::{
    Bound, BranchAndBoundSolver, LpWorkspace, Problem, ProductionLinearOptions,
    RevisedSimplexSolver, Solution, SolveStatus, solve_production_linear, verify_lp_dual,
};
use nc_sparse::CsrMatrix;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{error::Error, fs::File, io::BufReader, time::Instant};

#[derive(Deserialize)]
struct Input {
    cost: Vec<f64>,
    row_start: Vec<usize>,
    columns: Vec<usize>,
    coefficients: Vec<f64>,
    rows: Vec<[Option<f64>; 2]>,
    bounds: Vec<[Option<f64>; 2]>,
    integer: Vec<bool>,
    offset: f64,
    sense: f64,
}

const AUDIT_TOLERANCE: f64 = 1e-7;

fn bounds(data: Vec<[Option<f64>; 2]>) -> Vec<Bound> {
    data.into_iter()
        .map(|[lower, upper]| Bound { lower, upper })
        .collect()
}

fn primal(solution: &Solution, problem: &Problem, restore_objective: impl Fn(f64) -> f64) -> Value {
    let diagnostics = solution.diagnostics(problem);
    json!({
        "objective": restore_objective(diagnostics.recomputed_objective),
        "row_violation": diagnostics.maximum_row_violation,
        "variable_violation": diagnostics.maximum_variable_bound_violation,
        "integrality_violation": diagnostics.maximum_integrality_violation,
        "objective_error": diagnostics.objective_error,
        "verified": diagnostics.is_verified(AUDIT_TOLERANCE),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("missing benchmark input")?;
    let mode = args.next().ok_or("missing benchmark mode")?;
    let node_limit = args.next().map(|value| value.parse()).transpose()?;
    let input: Input = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let restore_objective = |value| input.sense * value + input.offset;
    let problem = Problem {
        constraints: CsrMatrix::new(
            input.rows.len(),
            input.cost.len(),
            input.row_start,
            input.columns,
            input.coefficients,
        )?,
        objective: input.cost,
        row_bounds: bounds(input.rows),
        var_bounds: bounds(input.bounds),
        is_integer: input.integer,
    };
    problem.validate()?;
    let start = Instant::now();

    let (seconds, report) = match (mode.as_str(), node_limit) {
        ("lp", None) => {
            let mut workspace = LpWorkspace::new(problem.clone(), None)?;
            let report = workspace.solve(&RevisedSimplexSolver::default(), AUDIT_TOLERANCE)?;
            let seconds = start.elapsed().as_secs_f64();
            let x = &report.solution;
            let dual = report.dual.as_ref().and_then(|candidate| {
                verify_lp_dual(
                    &problem,
                    &x.variable_values,
                    &candidate.row_multipliers,
                    AUDIT_TOLERANCE,
                )
            });
            (
                seconds,
                json!({
                    "status": format!("{:?}", x.status),
                    "iterations": report.iterations,
                    "pivots": report.pivots,
                    "primal": (x.status == SolveStatus::Optimal).then(|| primal(x, &problem, restore_objective)),
                    "dual_bound": dual.as_ref().map(|d| restore_objective(d.objective_bound)),
                    "dual_gap": dual.map(|d| d.absolute_gap),
                    "verified_farkas": report.infeasibility.is_some(),
                }),
            )
        }
        ("scaled", None) => {
            let report = solve_production_linear(&problem, ProductionLinearOptions::default())?;
            let seconds = start.elapsed().as_secs_f64();
            (
                seconds,
                json!({
                    "status": format!("{:?}", report.solution.status),
                    "reduced_columns": report.reduced_variables,
                    "primal": (report.solution.status == SolveStatus::Optimal)
                        .then(|| primal(&report.solution, &problem, restore_objective)),
                }),
            )
        }
        ("milp", Some(max_nodes)) => {
            let report = BranchAndBoundSolver {
                max_nodes,
                ..Default::default()
            }
            .solve_with_report(&problem)?;
            let seconds = start.elapsed().as_secs_f64();
            (
                seconds,
                json!({
                    "status": format!("{:?}", report.solution.status),
                    "termination": format!("{:?}", report.termination),
                    "nodes": report.nodes_explored,
                    "relaxations": report.relaxations_solved,
                    "incumbents": report.incumbents_found,
                    "heuristic_attempts": report.heuristic_attempts,
                    "heuristic_successes": report.heuristic_successes,
                    "reported_bound": report.best_bound.map(restore_objective),
                    "reported_gap": report.relative_gap,
                    "primal": (report.incumbents_found > 0)
                        .then(|| primal(&report.solution, &problem, restore_objective)),
                }),
            )
        }
        _ => return Err("expected lp, scaled, or milp with a node limit".into()),
    };
    println!("{}", json!({"solver_seconds": seconds, "result": report}));
    Ok(())
}
