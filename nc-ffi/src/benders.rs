use crate::{
    to_csr_f64, to_domain_problem, FfiCsrMatrixF64, FfiError, FfiLpSubproblemId, FfiProblem,
};
use nc_optimize as opt;

#[derive(uniffi::Record)]
pub struct FfiBendersScenario {
    pub id: FfiLpSubproblemId,
    pub probability: f64,
    pub recourse: FfiProblem,
    pub coupling: FfiCsrMatrixF64,
}

pub(crate) fn to_benders_scenario(
    scenario: FfiBendersScenario,
) -> Result<opt::BendersScenario, FfiError> {
    Ok(opt::BendersScenario {
        id: opt::LpSubproblemID {
            name: scenario.id.name,
            scenario: scenario.id.scenario,
        },
        probability: scenario.probability,
        recourse: to_domain_problem(scenario.recourse)?,
        coupling: to_csr_f64(scenario.coupling)?,
    })
}

#[derive(uniffi::Enum)]
pub enum FfiBendersCutStrategy {
    Single,
    Multi,
}

#[derive(uniffi::Record)]
pub struct FfiBendersOptions {
    pub cut_strategy: FfiBendersCutStrategy,
    pub max_iterations: u64,
    pub lp_iteration_limit: u64,
    pub simplex_tolerance: f64,
    pub evidence_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub scenario_threads: u64,
    pub maximum_inactive_rounds: Option<u64>,
    pub trust_region_radius: Option<f64>,
}

#[derive(uniffi::Enum)]
pub enum FfiBendersTermination {
    Converged,
    Infeasible,
    IterationLimit,
    LpIterationLimit,
    MissingEvidence,
    RecourseBoundUnavailable,
    MasterUnbounded,
    Stalled,
}

#[derive(uniffi::Record)]
pub struct FfiBendersRecourseSolution {
    pub id: FfiLpSubproblemId,
    pub point: Vec<f64>,
    pub objective: f64,
}

#[derive(uniffi::Record)]
pub struct FfiBendersIncumbent {
    pub first_stage: Vec<f64>,
    pub recourse: Vec<FfiBendersRecourseSolution>,
    pub objective: f64,
}

#[derive(uniffi::Record)]
pub struct FfiBendersIteration {
    pub iteration: u64,
    pub lower_bound: f64,
    pub upper_bound: Option<f64>,
    pub feasibility_cuts: u64,
    pub optimality_cuts: u64,
    pub cuts_pruned: u64,
    pub used_trust_region: bool,
    pub warm_subproblems: u64,
}

#[derive(uniffi::Record)]
pub struct FfiBendersResult {
    pub termination: FfiBendersTermination,
    pub incumbent: Option<FfiBendersIncumbent>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub history: Vec<FfiBendersIteration>,
    pub blocking_scenario: Option<FfiLpSubproblemId>,
}

#[uniffi::export]
pub fn solve_benders_decomposition(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    options: FfiBendersOptions,
) -> Result<FfiBendersResult, FfiError> {
    let first_stage = to_domain_problem(first_stage)?;
    let scenarios = scenarios
        .into_iter()
        .map(to_benders_scenario)
        .collect::<Result<Vec<_>, FfiError>>()?;
    let size = |value| {
        usize::try_from(value).map_err(|_| FfiError::SolverError {
            message: "Benders option outside platform range".into(),
        })
    };
    let result = opt::solve_benders(
        &opt::BendersProblem {
            first_stage,
            scenarios,
        },
        opt::BendersOptions {
            cut_strategy: match options.cut_strategy {
                FfiBendersCutStrategy::Single => opt::BendersCutStrategy::Single,
                FfiBendersCutStrategy::Multi => opt::BendersCutStrategy::Multi,
            },
            max_iterations: size(options.max_iterations)?,
            lp_iteration_limit: size(options.lp_iteration_limit)?,
            simplex_tolerance: options.simplex_tolerance,
            evidence_tolerance: options.evidence_tolerance,
            absolute_gap_tolerance: options.absolute_gap_tolerance,
            relative_gap_tolerance: options.relative_gap_tolerance,
            scenario_threads: size(options.scenario_threads)?,
            maximum_inactive_rounds: options.maximum_inactive_rounds.map(size).transpose()?,
            trust_region_radius: options.trust_region_radius,
        },
    )?;
    let ffi_id = |id: opt::LpSubproblemID| FfiLpSubproblemId {
        name: id.name,
        scenario: id.scenario,
    };
    Ok(FfiBendersResult {
        termination: match result.termination {
            opt::BendersTermination::Converged => FfiBendersTermination::Converged,
            opt::BendersTermination::Infeasible => FfiBendersTermination::Infeasible,
            opt::BendersTermination::IterationLimit => FfiBendersTermination::IterationLimit,
            opt::BendersTermination::LpIterationLimit => FfiBendersTermination::LpIterationLimit,
            opt::BendersTermination::MissingEvidence => FfiBendersTermination::MissingEvidence,
            opt::BendersTermination::RecourseBoundUnavailable => {
                FfiBendersTermination::RecourseBoundUnavailable
            }
            opt::BendersTermination::MasterUnbounded => FfiBendersTermination::MasterUnbounded,
            opt::BendersTermination::Stalled => FfiBendersTermination::Stalled,
        },
        incumbent: result.incumbent.map(|incumbent| FfiBendersIncumbent {
            first_stage: incumbent.first_stage,
            objective: incumbent.objective,
            recourse: incumbent
                .recourse
                .into_iter()
                .map(|scenario| FfiBendersRecourseSolution {
                    id: ffi_id(scenario.id),
                    point: scenario.point,
                    objective: scenario.objective,
                })
                .collect(),
        }),
        lower_bound: result.lower_bound,
        upper_bound: result.upper_bound,
        absolute_gap: result.absolute_gap,
        relative_gap: result.relative_gap,
        blocking_scenario: result.blocking_scenario.map(ffi_id),
        history: result
            .history
            .into_iter()
            .map(|record| FfiBendersIteration {
                iteration: record.iteration as u64,
                lower_bound: record.lower_bound,
                upper_bound: record.upper_bound,
                feasibility_cuts: record.feasibility_cuts as u64,
                optimality_cuts: record.optimality_cuts as u64,
                cuts_pruned: record.cuts_pruned as u64,
                used_trust_region: record.used_trust_region,
                warm_subproblems: record.warm_subproblems as u64,
            })
            .collect(),
    })
}
