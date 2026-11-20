use crate::{
    nonlinear::{bound, expression},
    to_domain_problem, FfiBound, FfiError, FfiNonlinearConstraint, FfiNonlinearExpression,
    FfiProblem,
};
use nc_optimize as opt;

#[derive(uniffi::Record)]
pub struct FfiNonlinearRecourseBlock {
    pub id: String,
    pub shared_columns: Vec<u64>,
    pub local_bounds: Vec<FfiBound>,
    pub weight: f64,
    pub objective: FfiNonlinearExpression,
    pub constraints: Vec<FfiNonlinearConstraint>,
}

#[derive(uniffi::Record)]
pub struct FfiNonlinearDecompositionOptions {
    pub max_master_solves: u64,
    pub max_block_solves: u64,
    pub lp_iteration_limit: u64,
    pub feasibility_tolerance: f64,
    pub absolute_gap: f64,
    pub relative_gap: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiNonlinearDecompositionTermination {
    Converged,
    Infeasible,
    MasterLimit,
    RecourseLimit,
    LpIterationLimit,
    MissingEvidence,
    EvaluationFailure,
    Stalled,
}

#[derive(uniffi::Record)]
pub struct FfiNonlinearRecourseSolution {
    pub id: String,
    pub local_point: Vec<f64>,
    pub objective: f64,
    pub maximum_violation: f64,
}

#[derive(uniffi::Record)]
pub struct FfiNonlinearDecompositionIncumbent {
    pub master_point: Vec<f64>,
    pub recourse: Vec<FfiNonlinearRecourseSolution>,
    pub objective: f64,
}

#[derive(uniffi::Record)]
pub struct FfiNonlinearDecompositionResult {
    pub termination: FfiNonlinearDecompositionTermination,
    pub incumbent: Option<FfiNonlinearDecompositionIncumbent>,
    pub lower_bound: Option<f64>,
    pub master_solves: u64,
    pub recourse_solves: u64,
    pub feasibility_cuts: u64,
    pub optimality_cuts: u64,
    pub blocking_block: Option<String>,
}

#[uniffi::export]
pub fn solve_nonlinear_decomposition(
    master: FfiProblem,
    blocks: Vec<FfiNonlinearRecourseBlock>,
    options: FfiNonlinearDecompositionOptions,
) -> Result<FfiNonlinearDecompositionResult, FfiError> {
    let size = |value| {
        usize::try_from(value).map_err(|_| FfiError::SolverError {
            message: "nonlinear decomposition index or option outside platform range".into(),
        })
    };
    let master = to_domain_problem(master)?;
    let blocks = blocks
        .into_iter()
        .map(|block| {
            Ok(opt::NonlinearRecourseBlock {
                id: block.id,
                shared_columns: block
                    .shared_columns
                    .into_iter()
                    .map(size)
                    .collect::<Result<_, _>>()?,
                local_bounds: block.local_bounds.into_iter().map(bound).collect(),
                weight: block.weight,
                objective: expression(block.objective),
                constraints: block
                    .constraints
                    .into_iter()
                    .map(|c| opt::NonlinearConstraint {
                        expression: expression(c.expression),
                        bound: bound(c.bound),
                    })
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, FfiError>>()?;
    let result = opt::solve_nonlinear_decomposition(
        &opt::NonlinearDecompositionProblem { master, blocks },
        opt::NonlinearDecompositionOptions {
            max_master_solves: size(options.max_master_solves)?,
            max_block_solves: size(options.max_block_solves)?,
            lp_iteration_limit: size(options.lp_iteration_limit)?,
            feasibility_tolerance: options.feasibility_tolerance,
            absolute_gap: options.absolute_gap,
            relative_gap: options.relative_gap,
        },
    )?;
    use opt::NonlinearDecompositionTermination as Term;
    use FfiNonlinearDecompositionTermination as FfiTerm;
    Ok(FfiNonlinearDecompositionResult {
        termination: match result.termination {
            Term::Converged => FfiTerm::Converged,
            Term::Infeasible => FfiTerm::Infeasible,
            Term::MasterLimit => FfiTerm::MasterLimit,
            Term::RecourseLimit => FfiTerm::RecourseLimit,
            Term::LpIterationLimit => FfiTerm::LpIterationLimit,
            Term::MissingEvidence => FfiTerm::MissingEvidence,
            Term::EvaluationFailure => FfiTerm::EvaluationFailure,
            Term::Stalled => FfiTerm::Stalled,
        },
        incumbent: result
            .incumbent
            .map(|incumbent| FfiNonlinearDecompositionIncumbent {
                master_point: incumbent.master_point,
                objective: incumbent.objective,
                recourse: incumbent
                    .recourse
                    .into_iter()
                    .map(|s| FfiNonlinearRecourseSolution {
                        id: s.id,
                        local_point: s.local_point,
                        objective: s.objective,
                        maximum_violation: s.maximum_violation,
                    })
                    .collect(),
            }),
        lower_bound: result.lower_bound,
        master_solves: result.master_solves as u64,
        recourse_solves: result.recourse_solves as u64,
        feasibility_cuts: result.feasibility_cuts as u64,
        optimality_cuts: result.optimality_cuts as u64,
        blocking_block: result.blocking_block,
    })
}
