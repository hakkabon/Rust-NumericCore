use crate::distributed_benders::{
    audit_benders_proof, solve_recourse_job, BendersProof, DistributedBendersClaim,
    DistributedBendersCoordinator, DistributedBendersOptions, DistributedBendersStatus,
};
use crate::engineering_sensitivity::direct_error;
use crate::{
    certify_linear_solution, solve_benders, BendersOptions, BendersProblem, BendersTermination,
    Bound, CertificateClaim, EngineeringElement, EngineeringEquation, EngineeringModel,
    EngineeringParameter, EngineeringPort, EngineeringTrajectory, EngineeringTrajectoryOptions,
    EngineeringTrajectoryResponse, EngineeringUnknown, EngineeringUnknownRole, NonlinearExpression,
    NonlinearNode, OptimizeError, Problem, Solution, SolveStatus, ValidationCase, ValidationMetric,
    ValidationTier, ValidationTolerance,
};
use nc_sparse::{CsrMatrix, SparseLuFactorization, SparseNumericOptions, SparseOrdering};

pub(super) fn run() -> Result<[ValidationCase; 5], OptimizeError> {
    let [limit_case, proof_case] = decomposition_cases()?;
    Ok([
        supplied_gap_case()?,
        limit_case,
        proof_case,
        scaled_sparse_solve_case()?,
        checkpointed_adjoint_case()?,
    ])
}

fn supplied_gap_case() -> Result<ValidationCase, OptimizeError> {
    let model = Problem {
        objective: vec![1.0],
        constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![])?,
        row_bounds: vec![],
        var_bounds: vec![Bound {
            lower: Some(0.0),
            upper: Some(2.0),
        }],
        is_integer: vec![false],
    };
    let solution = Solution {
        variable_values: vec![2.0],
        objective_value: 2.0,
        status: SolveStatus::Optimal,
    };
    let certificate = certify_linear_solution(&model, &solution, Some(0.0), Default::default())?;
    Ok(ValidationCase::new(
        "Supplied gap on a feasible suboptimal LP point",
        vec![
            exact_check(
                "feasibility claim verified",
                certificate.claim == CertificateClaim::Feasibility && certificate.verified(),
            ),
            exact_check(
                "supplied gap passed scalar check",
                certificate
                    .metrics
                    .iter()
                    .any(|metric| metric.name == "caller-supplied gap" && metric.passed),
            ),
            ValidationMetric::compare(
                "objective above known minimum",
                solution.objective_value,
                2.0,
                Default::default(),
            ),
        ],
    )
    .classified(
        "internal.certificate.supplied-gap",
        "optimization-certification",
        ValidationTier::Standard,
    ))
}

fn decomposition_cases() -> Result<[ValidationCase; 2], OptimizeError> {
    let capacity = Bound {
        lower: Some(0.0),
        upper: Some(10.0),
    };
    let first = Problem {
        objective: vec![1.0],
        constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![])?,
        row_bounds: vec![],
        var_bounds: vec![capacity],
        is_integer: vec![false],
    };
    let recourse = Problem {
        objective: vec![4.0],
        constraints: CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0])?,
        row_bounds: vec![Bound {
            lower: Some(2.0),
            upper: None,
        }],
        var_bounds: vec![capacity],
        is_integer: vec![false],
    };
    let coupling = CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0])?;
    let model = BendersProblem::classical(first, recourse, coupling);
    let limited = solve_benders(
        &model,
        BendersOptions {
            max_iterations: 1,
            ..Default::default()
        },
    )?;
    let limit_options = DistributedBendersOptions {
        max_rounds: 1,
        ..Default::default()
    };
    let mut partial =
        DistributedBendersCoordinator::start(model.clone(), "release-limit".into(), limit_options)?;
    complete_benders_rounds(&mut partial, limit_options.max_rounds)?;
    let bounds_only = audit_benders_proof(&model, &partial.proof(), limit_options)?;
    let limit_case = ValidationCase::new(
        "Benders limits retain partial evidence",
        vec![
            exact_check(
                "local iteration limit",
                limited.termination == BendersTermination::IterationLimit,
            ),
            exact_check(
                "local bounds bracket the optimum",
                limited.lower_bound.is_some_and(|value| value <= 2.0)
                    && limited.upper_bound.is_some_and(|value| value > 2.0),
            ),
            exact_check(
                "distributed round limit",
                partial.status() == DistributedBendersStatus::RoundLimit,
            ),
            exact_check(
                "partial distributed proof replays bounds",
                bounds_only.claim == DistributedBendersClaim::Bounds,
            ),
        ],
    )
    .classified(
        "internal.benders.limits",
        "decomposition",
        ValidationTier::Standard,
    );

    let options = DistributedBendersOptions {
        max_rounds: 10,
        ..Default::default()
    };
    let mut coordinator =
        DistributedBendersCoordinator::start(model.clone(), "release-replay".into(), options)?;
    complete_benders_rounds(&mut coordinator, options.max_rounds)?;
    let proof = coordinator.proof();
    let audit = audit_benders_proof(&model, &proof, options)?;
    let altered = altered_benders_proof(&proof)?;
    let proof_case = ValidationCase::new(
        "Replay and rejection of altered Benders evidence",
        vec![
            exact_check(
                "valid proof replays optimality",
                audit.claim == DistributedBendersClaim::Optimality
                    && audit.verified_subproblems > 0,
            ),
            ValidationMetric::compare(
                "replayed lower bound",
                audit.lower_bound.unwrap_or(f64::INFINITY),
                2.0,
                Default::default(),
            ),
            ValidationMetric::compare(
                "replayed upper bound",
                audit.upper_bound.unwrap_or(f64::INFINITY),
                2.0,
                Default::default(),
            ),
            exact_check(
                "altered witness rejected",
                audit_benders_proof(&model, &altered, options).is_err(),
            ),
        ],
    )
    .classified(
        "internal.benders.proof-replay",
        "decomposition",
        ValidationTier::Standard,
    );
    Ok([limit_case, proof_case])
}

fn complete_benders_rounds(
    coordinator: &mut DistributedBendersCoordinator,
    maximum: usize,
) -> Result<(), OptimizeError> {
    for _ in 0..maximum {
        if coordinator.status() != DistributedBendersStatus::AwaitingReplies {
            break;
        }
        for job in coordinator.pending_jobs()? {
            coordinator.submit(solve_recourse_job(job)?)?;
        }
    }
    Ok(())
}

fn altered_benders_proof(proof: &BendersProof) -> Result<BendersProof, OptimizeError> {
    let mut document = serde_json::to_value(proof)
        .map_err(|error| OptimizeError::InvalidProblem(error.to_string()))?;
    let point = document
        .pointer_mut("/rounds/0/master/point/0")
        .ok_or_else(|| {
            OptimizeError::InvalidProblem("validation proof omitted its first master point".into())
        })?;
    *point = serde_json::json!(1e6);
    serde_json::from_value(document)
        .map_err(|error| OptimizeError::InvalidProblem(error.to_string()))
}

fn scaled_sparse_solve_case() -> Result<ValidationCase, OptimizeError> {
    let matrix = CsrMatrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4e-6, 1e-6, 1.0, 3.0, 1.0, 1e6, 2e6],
    )?;
    let options = SparseNumericOptions {
        pivot_tolerance: 1e-15,
        ..Default::default()
    };
    let factor = SparseLuFactorization::factor(&matrix, SparseOrdering::MinimumDegree, options)
        .map_err(direct_error)?;
    let solve = factor
        .solve_refined(&[2e-6, -2.0, 4e6])
        .map_err(direct_error)?;
    let forward_error = solve
        .solution
        .iter()
        .zip([1.0, -2.0, 3.0])
        .map(|(observed, reference)| (observed - reference).abs())
        .fold(0.0, f64::max);
    Ok(ValidationCase::new(
        "Sparse LU with row scales spanning twelve orders",
        vec![
            ValidationMetric::compare(
                "forward infinity error",
                forward_error,
                0.0,
                ValidationTolerance::new(1e-9, 0.0),
            ),
            ValidationMetric::compare(
                "normwise backward error",
                solve.backward_error,
                0.0,
                ValidationTolerance::new(1e-12, 0.0),
            ),
        ],
    )
    .classified(
        "internal.sparse.scaled-solve",
        "sparse-linear-algebra",
        ValidationTier::Standard,
    ))
}

fn checkpointed_adjoint_case() -> Result<ValidationCase, OptimizeError> {
    // ẏ + y = p on a fixed, nonuniform grid. J = y(final) + Σ h·y².
    use NonlinearNode::{Add, Multiply, Parameter, Subtract};
    let times = [0.0, 0.1, 0.27, 0.45, 0.7];
    let (load, initial, epsilon) = (2.0, 0.25, 1e-5);
    let residual = NonlinearExpression::new(
        vec![
            Parameter(2),
            Parameter(1),
            Add(0, 1),
            Parameter(3),
            Subtract(2, 3),
        ],
        4,
    );
    let model = EngineeringModel {
        unknowns: vec![EngineeringUnknown {
            name: "y".into(),
            unit: "1".into(),
            role: EngineeringUnknownRole::Differential,
            scale: 1.0,
            initial_value: initial,
            initial_derivative: load - initial,
        }],
        parameters: vec![EngineeringParameter {
            name: "p".into(),
            unit: "1".into(),
            value: load,
        }],
        elements: vec![EngineeringElement {
            name: "balance".into(),
            kind: "scalar".into(),
            ports: vec![EngineeringPort {
                name: "y".into(),
                unknown: 0,
                orientation: 1.0,
            }],
            equations: vec![EngineeringEquation {
                row: 0,
                expression: residual,
            }],
        }],
    };
    let options = EngineeringTrajectoryOptions {
        checkpoint_stride: 2,
        ..Default::default()
    };
    let trajectory = EngineeringTrajectory::integrate(model, &times, &[initial], options)?;
    let result = trajectory.adjoints(&[EngineeringTrajectoryResponse {
        name: "response".into(),
        running: Some(NonlinearExpression::new(
            vec![Parameter(1), Multiply(0, 0)],
            1,
        )),
        terminal: Some(NonlinearExpression::new(vec![Parameter(1)], 0)),
    }])?;
    let gradient = &result.responses[0];
    let reference_objective = |load: f64, initial: f64| {
        let mut state = initial;
        let mut running = 0.0;
        for endpoints in times.windows(2) {
            let step = endpoints[1] - endpoints[0];
            state = (state + step * load) / (1.0 + step);
            running += step * state * state;
        }
        running + state
    };
    let load_difference = (reference_objective(load + epsilon, initial)
        - reference_objective(load - epsilon, initial))
        / (2.0 * epsilon);
    let initial_difference = (reference_objective(load, initial + epsilon)
        - reference_objective(load, initial - epsilon))
        / (2.0 * epsilon);
    let tolerance = ValidationTolerance::new(1e-8, 0.0);
    Ok(ValidationCase::new(
        "Checkpointed backward-Euler adjoint",
        vec![
            ValidationMetric::compare(
                "objective",
                gradient.value,
                reference_objective(load, initial),
                tolerance,
            ),
            ValidationMetric::compare(
                "load derivative",
                gradient.parameter_gradient[0],
                load_difference,
                tolerance,
            ),
            ValidationMetric::compare(
                "initial-state derivative",
                gradient.initial_state_gradient[0],
                initial_difference,
                tolerance,
            ),
            exact_check(
                "checkpoint replay and transpose solves",
                result.replayed_steps == 2 && result.transpose_solves == 4,
            ),
        ],
    )
    .classified(
        "internal.adjoint.checkpointed",
        "engineering-sensitivity",
        ValidationTier::Standard,
    ))
}

fn exact_check(name: &str, passed: bool) -> ValidationMetric {
    ValidationMetric::compare(
        name,
        if passed { 1.0 } else { 0.0 },
        1.0,
        ValidationTolerance::new(0.0, 0.0),
    )
}
