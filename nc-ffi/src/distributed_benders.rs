use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    benders::ffi_benders_incumbent, to_benders_scenario, to_domain_problem, FfiBendersIncumbent,
    FfiBendersScenario, FfiError, FfiProblem,
};
use nc_optimize::{distributed_benders as db, BendersProblem};

#[derive(uniffi::Record)]
pub struct FfiDistributedBendersOptions {
    pub max_rounds: u64,
    pub lp_iteration_limit: u64,
    pub evidence_tolerance: f64,
    pub absolute_gap: f64,
    pub relative_gap: f64,
}

#[derive(uniffi::Record)]
pub struct FfiDistributedBendersJob {
    pub round: u64,
    pub scenario: u64,
    pub json: String,
}

#[derive(uniffi::Enum)]
pub enum FfiDistributedBendersClaim {
    Bounds,
    Optimality,
    Infeasibility,
}

#[derive(uniffi::Enum)]
pub enum FfiDistributedBendersStatus {
    AwaitingReplies,
    Converged,
    Infeasible,
    RoundLimit,
    MasterUnresolved,
    Stalled,
}

#[derive(uniffi::Enum)]
pub enum FfiDistributedReplyDisposition {
    Accepted,
    Duplicate,
    Retry,
}

#[derive(uniffi::Record)]
pub struct FfiDistributedBendersAudit {
    pub claim: FfiDistributedBendersClaim,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub incumbent: Option<FfiBendersIncumbent>,
    pub verified_rounds: u64,
    pub verified_subproblems: u64,
    pub cuts: u64,
}

#[derive(uniffi::Record)]
pub struct FfiDistributedBendersProgress {
    pub status: FfiDistributedBendersStatus,
    pub audit: FfiDistributedBendersAudit,
}

#[derive(uniffi::Object)]
pub struct FfiDistributedBendersCoordinator {
    coordinator: Mutex<db::DistributedBendersCoordinator>,
}

#[uniffi::export]
pub fn start_distributed_benders(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    run_id: String,
    options: FfiDistributedBendersOptions,
) -> Result<Arc<FfiDistributedBendersCoordinator>, FfiError> {
    let coordinator = db::DistributedBendersCoordinator::start(
        problem(first_stage, scenarios)?,
        run_id,
        config(options)?,
    )?;
    Ok(Arc::new(FfiDistributedBendersCoordinator {
        coordinator: Mutex::new(coordinator),
    }))
}

#[uniffi::export]
pub fn resume_distributed_benders(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    checkpoint: String,
    options: FfiDistributedBendersOptions,
) -> Result<Arc<FfiDistributedBendersCoordinator>, FfiError> {
    let coordinator = db::DistributedBendersCoordinator::resume(
        problem(first_stage, scenarios)?,
        decode(&checkpoint)?,
        config(options)?,
    )?;
    Ok(Arc::new(FfiDistributedBendersCoordinator {
        coordinator: Mutex::new(coordinator),
    }))
}

#[uniffi::export]
pub fn verify_distributed_benders(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    proof: String,
    options: FfiDistributedBendersOptions,
) -> Result<FfiDistributedBendersAudit, FfiError> {
    let audit = db::audit_benders_proof(
        &problem(first_stage, scenarios)?,
        &decode(&proof)?,
        config(options)?,
    )?;
    Ok(ffi_audit(audit))
}

#[uniffi::export]
pub fn execute_distributed_benders_job(job: String) -> Result<String, FfiError> {
    encode(&db::solve_recourse_job(decode(&job)?)?)
}

#[uniffi::export]
impl FfiDistributedBendersCoordinator {
    pub fn pending_jobs(&self) -> Result<Vec<FfiDistributedBendersJob>, FfiError> {
        self.lock()?
            .pending_jobs()?
            .iter()
            .map(|job| {
                Ok(FfiDistributedBendersJob {
                    round: job.key.round as u64,
                    scenario: job.key.scenario as u64,
                    json: encode(job)?,
                })
            })
            .collect()
    }

    pub fn submit(&self, reply: String) -> Result<FfiDistributedReplyDisposition, FfiError> {
        let reply = decode(&reply)?;
        Ok(match self.lock()?.submit(reply)? {
            db::ReplyDisposition::Accepted => FfiDistributedReplyDisposition::Accepted,
            db::ReplyDisposition::Duplicate => FfiDistributedReplyDisposition::Duplicate,
            db::ReplyDisposition::Retry => FfiDistributedReplyDisposition::Retry,
        })
    }

    pub fn progress(&self) -> Result<FfiDistributedBendersProgress, FfiError> {
        let state = self.lock()?;
        Ok(FfiDistributedBendersProgress {
            audit: ffi_audit(state.audit().clone()),
            status: match state.status() {
                db::DistributedBendersStatus::AwaitingReplies => {
                    FfiDistributedBendersStatus::AwaitingReplies
                }
                db::DistributedBendersStatus::Converged => FfiDistributedBendersStatus::Converged,
                db::DistributedBendersStatus::Infeasible => FfiDistributedBendersStatus::Infeasible,
                db::DistributedBendersStatus::RoundLimit => FfiDistributedBendersStatus::RoundLimit,
                db::DistributedBendersStatus::MasterUnresolved => {
                    FfiDistributedBendersStatus::MasterUnresolved
                }
                db::DistributedBendersStatus::Stalled => FfiDistributedBendersStatus::Stalled,
            },
        })
    }

    pub fn export_proof(&self) -> Result<String, FfiError> {
        encode(&self.lock()?.proof())
    }

    pub fn export_checkpoint(&self) -> Result<String, FfiError> {
        encode(&self.lock()?.checkpoint())
    }
}

impl FfiDistributedBendersCoordinator {
    fn lock(&self) -> Result<MutexGuard<'_, db::DistributedBendersCoordinator>, FfiError> {
        self.coordinator
            .lock()
            .map_err(|_| error("distributed coordinator lock is poisoned"))
    }
}

fn problem(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
) -> Result<BendersProblem, FfiError> {
    Ok(BendersProblem {
        first_stage: to_domain_problem(first_stage)?,
        scenarios: scenarios
            .into_iter()
            .map(to_benders_scenario)
            .collect::<Result<_, _>>()?,
    })
}

fn config(
    options: FfiDistributedBendersOptions,
) -> Result<db::DistributedBendersOptions, FfiError> {
    let size = |v| {
        usize::try_from(v).map_err(|_| error("distributed Benders option outside platform range"))
    };
    Ok(db::DistributedBendersOptions {
        max_rounds: size(options.max_rounds)?,
        lp_iteration_limit: size(options.lp_iteration_limit)?,
        evidence_tolerance: options.evidence_tolerance,
        absolute_gap: options.absolute_gap,
        relative_gap: options.relative_gap,
    })
}

fn ffi_audit(audit: db::DistributedBendersAudit) -> FfiDistributedBendersAudit {
    FfiDistributedBendersAudit {
        claim: match audit.claim {
            db::DistributedBendersClaim::Bounds => FfiDistributedBendersClaim::Bounds,
            db::DistributedBendersClaim::Optimality => FfiDistributedBendersClaim::Optimality,
            db::DistributedBendersClaim::Infeasibility => FfiDistributedBendersClaim::Infeasibility,
        },
        lower_bound: audit.lower_bound,
        upper_bound: audit.upper_bound,
        absolute_gap: audit.absolute_gap,
        relative_gap: audit.relative_gap,
        incumbent: audit.incumbent.map(ffi_benders_incumbent),
        verified_rounds: audit.verified_rounds as u64,
        verified_subproblems: audit.verified_subproblems as u64,
        cuts: audit.cuts as u64,
    }
}

fn decode<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, FfiError> {
    serde_json::from_str(text).map_err(|e| error(&format!("invalid distributed Benders JSON: {e}")))
}

fn encode(value: &impl serde::Serialize) -> Result<String, FfiError> {
    serde_json::to_string(value).map_err(|e| {
        error(&format!(
            "cannot serialize distributed Benders evidence: {e}"
        ))
    })
}

fn error(message: &str) -> FfiError {
    FfiError::SolverError {
        message: message.into(),
    }
}
