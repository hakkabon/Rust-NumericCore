use crate::{
    benders::to_benders_scenario, to_domain_problem, FfiBendersIncumbent,
    FfiBendersRecourseSolution, FfiBendersScenario, FfiError, FfiLpSubproblemId, FfiProblem,
};
use nc_optimize as opt;

#[derive(uniffi::Record)]
pub struct FfiLogicBendersScenario {
    pub id: FfiLpSubproblemId,
    pub probability: f64,
    pub binary_columns: Vec<u32>,
    pub objective_lower_bound: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiIntegerBendersScenario {
    Model { scenario: FfiBendersScenario },
    Logic { scenario: FfiLogicBendersScenario },
}

#[derive(uniffi::Enum)]
pub enum FfiLogicBendersEvaluation {
    Optimal {
        objective: f64,
        point: Vec<f64>,
        validity: Option<Vec<u32>>,
    },
    Infeasible {
        conflict: Option<Vec<u32>>,
    },
    Unresolved,
}

#[uniffi::export(callback_interface)]
pub trait FfiLogicBendersOracle: Send + Sync {
    fn evaluate(&self, id: FfiLpSubproblemId, bits: Vec<bool>) -> FfiLogicBendersEvaluation;
}

#[derive(uniffi::Record)]
pub struct FfiIntegerBendersOptions {
    pub max_master_nodes: u64,
    pub max_candidate_evaluations: u64,
    pub max_recourse_nodes: u64,
    pub lp_iteration_limit: u64,
    pub simplex_tolerance: f64,
    pub evidence_tolerance: f64,
    pub integrality_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiIntegerBendersTermination {
    Converged,
    Infeasible,
    MasterNodeLimit,
    CandidateLimit,
    LpIterationLimit,
    RecourseNodeLimit,
    OracleUnresolved,
    MissingEvidence,
    RecourseBoundUnavailable,
    MasterUnbounded,
    Stalled,
}

#[derive(uniffi::Record)]
pub struct FfiIntegerBendersResult {
    pub termination: FfiIntegerBendersTermination,
    pub incumbent: Option<FfiBendersIncumbent>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub master_nodes: u64,
    pub candidate_evaluations: u64,
    pub recourse_evaluations: u64,
    pub assignment_cache_hits: u64,
    pub feasibility_cuts: u64,
    pub optimality_cuts: u64,
    pub blocking_scenario: Option<FfiLpSubproblemId>,
    pub uses_logic_oracle: bool,
}

#[uniffi::export]
pub fn solve_integer_benders_decomposition(
    first_stage: FfiProblem,
    scenarios: Vec<FfiIntegerBendersScenario>,
    options: FfiIntegerBendersOptions,
    oracle: Box<dyn FfiLogicBendersOracle>,
) -> Result<FfiIntegerBendersResult, FfiError> {
    let first_stage = to_domain_problem(first_stage)?;
    let scenarios = scenarios
        .into_iter()
        .map(|item| {
            Ok(match item {
                FfiIntegerBendersScenario::Model { scenario } => {
                    opt::IntegerBendersScenario::Model(to_benders_scenario(scenario)?)
                }
                FfiIntegerBendersScenario::Logic { scenario } => {
                    opt::IntegerBendersScenario::Logic(opt::LogicBendersScenario {
                        id: opt::LpSubproblemID {
                            name: scenario.id.name,
                            scenario: scenario.id.scenario,
                        },
                        probability: scenario.probability,
                        binary_columns: scenario
                            .binary_columns
                            .into_iter()
                            .map(|i| i as usize)
                            .collect(),
                        objective_lower_bound: scenario.objective_lower_bound,
                    })
                }
            })
        })
        .collect::<Result<Vec<_>, FfiError>>()?;
    let size = |value| {
        usize::try_from(value).map_err(|_| FfiError::SolverError {
            message: "integer Benders option outside platform range".into(),
        })
    };
    let result = opt::solve_logic_benders(
        &opt::IntegerBendersProblem {
            first_stage,
            scenarios,
        },
        opt::IntegerBendersOptions {
            max_master_nodes: size(options.max_master_nodes)?,
            max_candidate_evaluations: size(options.max_candidate_evaluations)?,
            max_recourse_nodes: size(options.max_recourse_nodes)?,
            lp_iteration_limit: size(options.lp_iteration_limit)?,
            simplex_tolerance: options.simplex_tolerance,
            evidence_tolerance: options.evidence_tolerance,
            integrality_tolerance: options.integrality_tolerance,
            absolute_gap_tolerance: options.absolute_gap_tolerance,
            relative_gap_tolerance: options.relative_gap_tolerance,
        },
        |id, bits| {
            let positions = |scope: Option<Vec<u32>>| {
                scope.map(|v| v.into_iter().map(|i| i as usize).collect())
            };
            Ok(match oracle.evaluate(ffi_id(id.clone()), bits.to_vec()) {
                FfiLogicBendersEvaluation::Optimal {
                    objective,
                    point,
                    validity,
                } => opt::LogicBendersEvaluation::Optimal {
                    objective,
                    point,
                    validity: positions(validity),
                },
                FfiLogicBendersEvaluation::Infeasible { conflict } => {
                    opt::LogicBendersEvaluation::Infeasible {
                        conflict: positions(conflict),
                    }
                }
                FfiLogicBendersEvaluation::Unresolved => opt::LogicBendersEvaluation::Unresolved,
            })
        },
    )?;
    use opt::IntegerBendersTermination as Stop;
    Ok(FfiIntegerBendersResult {
        termination: match result.termination {
            Stop::Converged => FfiIntegerBendersTermination::Converged,
            Stop::Infeasible => FfiIntegerBendersTermination::Infeasible,
            Stop::MasterNodeLimit => FfiIntegerBendersTermination::MasterNodeLimit,
            Stop::CandidateLimit => FfiIntegerBendersTermination::CandidateLimit,
            Stop::LpIterationLimit => FfiIntegerBendersTermination::LpIterationLimit,
            Stop::RecourseNodeLimit => FfiIntegerBendersTermination::RecourseNodeLimit,
            Stop::OracleUnresolved => FfiIntegerBendersTermination::OracleUnresolved,
            Stop::MissingEvidence => FfiIntegerBendersTermination::MissingEvidence,
            Stop::RecourseBoundUnavailable => {
                FfiIntegerBendersTermination::RecourseBoundUnavailable
            }
            Stop::MasterUnbounded => FfiIntegerBendersTermination::MasterUnbounded,
            Stop::Stalled => FfiIntegerBendersTermination::Stalled,
        },
        incumbent: result.incumbent.map(|i| FfiBendersIncumbent {
            first_stage: i.first_stage,
            objective: i.objective,
            recourse: i
                .recourse
                .into_iter()
                .map(|s| FfiBendersRecourseSolution {
                    id: ffi_id(s.id),
                    point: s.point,
                    objective: s.objective,
                })
                .collect(),
        }),
        lower_bound: result.lower_bound,
        upper_bound: result.upper_bound,
        absolute_gap: result.absolute_gap,
        relative_gap: result.relative_gap,
        master_nodes: result.master_nodes as u64,
        candidate_evaluations: result.candidate_evaluations as u64,
        recourse_evaluations: result.recourse_evaluations as u64,
        assignment_cache_hits: result.assignment_cache_hits as u64,
        feasibility_cuts: result.feasibility_cuts as u64,
        optimality_cuts: result.optimality_cuts as u64,
        blocking_scenario: result.blocking_scenario.map(ffi_id),
        uses_logic_oracle: result.uses_logic_oracle,
    })
}

fn ffi_id(id: opt::LpSubproblemID) -> FfiLpSubproblemId {
    FfiLpSubproblemId {
        name: id.name,
        scenario: id.scenario,
    }
}
