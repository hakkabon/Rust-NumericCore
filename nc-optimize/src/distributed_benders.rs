//! Transport-neutral continuous Benders with independently replayable numerical evidence.

use crate::{
    benders::{affine, box_minimum, make_master, solve_lp, solve_master_lp, validate, violation},
    verify_lp_dual, verify_lp_farkas, BendersCutStrategy, BendersIncumbent, BendersOptions,
    BendersProblem, BendersRecourseSolution, BendersScenario, Bound, CutInsertion, CutKind,
    DecompositionCut, DecompositionCutPool, LpSolveReport, LpUpdate, LpWorkspace, OptimizeError,
    Problem, RevisedSimplexSolver,
};
use nc_sparse::CsrMatrix;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct DistributedBendersOptions {
    pub max_rounds: usize,
    pub lp_iteration_limit: usize,
    pub evidence_tolerance: f64,
    pub absolute_gap: f64,
    pub relative_gap: f64,
}

impl Default for DistributedBendersOptions {
    fn default() -> Self {
        Self {
            max_rounds: 100,
            lp_iteration_limit: 10_000,
            evidence_tolerance: 1e-8,
            absolute_gap: 1e-7,
            relative_gap: 1e-7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobKey {
    pub run_id: String,
    /// Starts at one and advances only after the current batch is complete.
    pub round: usize,
    /// Original scenario index in the problem.
    pub scenario: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecourseJob {
    pub version: u32,
    pub key: JobKey,
    pub lp_iteration_limit: usize,
    pub evidence_tolerance: f64,
    lp: WireLp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecourseReply {
    pub version: u32,
    pub key: JobKey,
    pub evidence: LpWitness,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LpWitness {
    Optimal {
        point: Vec<f64>,
        row_multipliers: Vec<f64>,
    },
    Infeasible {
        row_multipliers: Vec<f64>,
    },
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireLp {
    objective: Vec<f64>,
    row_bounds: Vec<[Option<f64>; 2]>,
    variable_bounds: Vec<[Option<f64>; 2]>,
    row_pointers: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

impl WireLp {
    fn encode(lp: &Problem) -> Self {
        let mut row_pointers = vec![0; lp.row_bounds.len() + 1];
        let mut column_indices = Vec::with_capacity(lp.constraints.nnz());
        let mut values = Vec::with_capacity(lp.constraints.nnz());
        for (r, c, v) in lp.constraints.iter_entries() {
            row_pointers[r + 1] += 1;
            column_indices.push(c);
            values.push(v);
        }
        for row in 1..row_pointers.len() {
            row_pointers[row] += row_pointers[row - 1];
        }
        let bounds = |b: &[Bound]| b.iter().map(|v| [v.lower, v.upper]).collect();
        Self {
            objective: lp.objective.clone(),
            row_bounds: bounds(&lp.row_bounds),
            variable_bounds: bounds(&lp.var_bounds),
            row_pointers,
            column_indices,
            values,
        }
    }

    fn decode(self) -> Result<Problem, OptimizeError> {
        let n = self.objective.len();
        let bounds = |b: Vec<[Option<f64>; 2]>| {
            b.into_iter()
                .map(|[lower, upper]| Bound { lower, upper })
                .collect()
        };
        let problem = Problem {
            constraints: CsrMatrix::new(
                self.row_bounds.len(),
                n,
                self.row_pointers,
                self.column_indices,
                self.values,
            )?,
            row_bounds: bounds(self.row_bounds),
            var_bounds: bounds(self.variable_bounds),
            objective: self.objective,
            is_integer: vec![false; n],
        };
        problem.validate()?;
        Ok(problem)
    }
}

/// Stateless worker entry point. A reply carries raw witnesses only.
pub fn solve_recourse_job(job: RecourseJob) -> Result<RecourseReply, OptimizeError> {
    if job.version != PROTOCOL_VERSION
        || job.key.run_id.trim().is_empty()
        || job.key.round == 0
        || job.lp_iteration_limit == 0
        || !valid_evidence_tolerance(job.evidence_tolerance)
    {
        return Err(invalid("invalid distributed recourse job"));
    }
    let mut workspace = LpWorkspace::new(job.lp.decode()?, None)?;
    let solver = solver(job.lp_iteration_limit, job.evidence_tolerance);
    let evidence = witness(solve_lp(&mut workspace, &solver, job.evidence_tolerance)?);
    Ok(RecourseReply {
        version: PROTOCOL_VERSION,
        key: job.key,
        evidence,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CertifiedRound {
    master: LpWitness,
    replies: Vec<RecourseReply>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersProof {
    version: u32,
    run_id: String,
    rounds: Vec<CertifiedRound>,
    closing_master: Option<LpWitness>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingRound {
    master: LpWitness,
    replies: Vec<Option<RecourseReply>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedBendersCheckpoint {
    proof: BendersProof,
    pending: Option<PendingRound>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedBendersClaim {
    Bounds,
    Optimality,
    Infeasibility,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistributedBendersAudit {
    pub claim: DistributedBendersClaim,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub incumbent: Option<BendersIncumbent>,
    pub verified_rounds: usize,
    pub verified_subproblems: usize,
    pub cuts: usize,
}

impl DistributedBendersAudit {
    fn new() -> Self {
        Self {
            claim: DistributedBendersClaim::Bounds,
            lower_bound: None,
            upper_bound: None,
            absolute_gap: None,
            relative_gap: None,
            incumbent: None,
            verified_rounds: 0,
            verified_subproblems: 0,
            cuts: 0,
        }
    }

    fn check_gap(&mut self, options: DistributedBendersOptions) -> Result<(), OptimizeError> {
        if let (Some(lower), Some(upper)) = (self.lower_bound, self.upper_bound) {
            let scale = lower.abs().max(upper.abs()).max(1.0);
            if (lower > upper) && (lower - upper) / scale > options.evidence_tolerance {
                return Err(invalid(
                    "master lower bound conflicts with the verified incumbent",
                ));
            }
            let lower = lower.min(upper);
            self.lower_bound = Some(lower);
            let gap = upper - lower;
            self.absolute_gap = gap.is_finite().then_some(gap);
            let relative = self
                .absolute_gap
                .map_or_else(|| upper / scale - lower / scale, |gap| gap / scale);
            self.relative_gap = Some(relative);
            if self
                .absolute_gap
                .is_some_and(|gap| gap <= options.absolute_gap)
                || relative <= options.relative_gap
            {
                self.claim = DistributedBendersClaim::Optimality;
            }
        }
        Ok(())
    }
}

/// Replay an exported proof against the original model without invoking simplex.
pub fn audit_benders_proof(
    problem: &BendersProblem,
    proof: &BendersProof,
    options: DistributedBendersOptions,
) -> Result<DistributedBendersAudit, OptimizeError> {
    Ok(AuditState::replay(problem.clone(), proof, options)?.audit)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedBendersStatus {
    AwaitingReplies,
    Converged,
    Infeasible,
    RoundLimit,
    MasterUnresolved,
    Stalled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyDisposition {
    Accepted,
    Duplicate,
    Retry,
}

pub struct DistributedBendersCoordinator {
    state: AuditState,
    proof: BendersProof,
    pending: Option<PendingRound>,
    remaining_replies: usize,
    status: DistributedBendersStatus,
}

impl DistributedBendersCoordinator {
    pub fn start(
        problem: BendersProblem,
        run_id: String,
        options: DistributedBendersOptions,
    ) -> Result<Self, OptimizeError> {
        let proof = BendersProof {
            version: PROTOCOL_VERSION,
            run_id,
            rounds: vec![],
            closing_master: None,
        };
        Self::resume(
            problem,
            DistributedBendersCheckpoint {
                proof,
                pending: None,
            },
            options,
        )
    }

    /// Recheck every imported witness, including replies to an unfinished batch.
    pub fn resume(
        problem: BendersProblem,
        mut checkpoint: DistributedBendersCheckpoint,
        options: DistributedBendersOptions,
    ) -> Result<Self, OptimizeError> {
        let mut state = AuditState::replay(problem, &checkpoint.proof, options)?;
        let mut status = state.status();
        let mut remaining_replies = 0;
        if let Some(batch) = &checkpoint.pending {
            remaining_replies = batch.replies.iter().filter(|r| r.is_none()).count();
            if checkpoint.proof.closing_master.is_some()
                || status.is_some()
                || batch.replies.len() != state.problem.scenarios.len()
                || remaining_replies == 0
                || state.audit.verified_rounds >= options.max_rounds
            {
                return Err(invalid("invalid pending distributed Benders batch"));
            }
            let master = state.inspect(&batch.master)?;
            let point = master
                .point
                .ok_or_else(|| invalid("pending batch requires an optimal master witness"))?;
            if master.audit.claim != DistributedBendersClaim::Bounds {
                return Err(invalid("pending batch follows a terminal master witness"));
            }
            for (i, reply) in batch.replies.iter().enumerate() {
                if let Some(reply) = reply {
                    state.check_reply(reply, i, &point)?;
                }
            }
            state.audit = master.audit;
            status = Some(DistributedBendersStatus::AwaitingReplies);
        }
        if status.is_none() && checkpoint.proof.closing_master.take().is_some() {
            // A tighter target can reopen a converged checkpoint. The
            // discarded endpoint's bound must be discarded from memory too.
            state = AuditState::replay(state.problem, &checkpoint.proof, options)?;
        }
        let mut result = Self {
            state,
            proof: checkpoint.proof,
            pending: checkpoint.pending,
            remaining_replies,
            status: status.unwrap_or(DistributedBendersStatus::MasterUnresolved),
        };
        if status.is_none() {
            result.advance();
        }
        Ok(result)
    }

    pub fn status(&self) -> DistributedBendersStatus {
        self.status
    }
    pub fn audit(&self) -> &DistributedBendersAudit {
        &self.state.audit
    }

    pub fn pending_jobs(&self) -> Result<Vec<RecourseJob>, OptimizeError> {
        let Some(batch) = &self.pending else {
            return Ok(vec![]);
        };
        let LpWitness::Optimal { point, .. } = &batch.master else {
            unreachable!()
        };
        self.state
            .problem
            .scenarios
            .iter()
            .zip(&batch.replies)
            .enumerate()
            .filter(|(_, (_, reply))| reply.is_none())
            .map(|(i, (scenario, _))| {
                let lp = shifted(scenario, &point[..self.state.columns()])?;
                Ok(RecourseJob {
                    version: PROTOCOL_VERSION,
                    key: self.state.key(i),
                    lp_iteration_limit: self.state.options.lp_iteration_limit,
                    evidence_tolerance: self.state.options.evidence_tolerance,
                    lp: WireLp::encode(&lp),
                })
            })
            .collect()
    }

    pub fn submit(&mut self, reply: RecourseReply) -> Result<ReplyDisposition, OptimizeError> {
        let i = reply.key.scenario;
        if reply.version != PROTOCOL_VERSION
            || i >= self.state.problem.scenarios.len()
            || reply.key.run_id != self.proof.run_id
        {
            return Err(invalid("unexpected distributed recourse reply identity"));
        }
        // Network retries can arrive after the coordinator has advanced.
        if let Some(round) = reply
            .key
            .round
            .checked_sub(1)
            .and_then(|r| self.proof.rounds.get(r))
        {
            return duplicate(&round.replies[i], &reply);
        }
        if reply.key != self.state.key(i) {
            return Err(invalid("stale or unexpected distributed recourse round"));
        }
        let batch = self
            .pending
            .as_mut()
            .ok_or_else(|| invalid("no pending distributed recourse batch"))?;
        if let Some(previous) = &batch.replies[i] {
            return duplicate(previous, &reply);
        }
        if reply.evidence == LpWitness::Unresolved {
            return Ok(ReplyDisposition::Retry);
        }
        let LpWitness::Optimal { point, .. } = &batch.master else {
            unreachable!()
        };
        self.state.check_reply(&reply, i, point)?;
        if self.remaining_replies == 1 {
            let round = CertifiedRound {
                master: batch.master.clone(),
                replies: batch
                    .replies
                    .iter()
                    .enumerate()
                    .map(|(j, r)| {
                        if i == j {
                            reply.clone()
                        } else {
                            r.clone().unwrap()
                        }
                    })
                    .collect(),
            };
            self.state.apply(&round)?;
            self.pending = None;
            self.remaining_replies = 0;
            self.proof.rounds.push(round);
            self.advance();
        } else {
            batch.replies[i] = Some(reply);
            self.remaining_replies -= 1;
        }
        Ok(ReplyDisposition::Accepted)
    }

    /// Includes a pending master bound. Only completed recourse batches enter the proof.
    pub fn proof(&self) -> BendersProof {
        let mut proof = self.proof.clone();
        if let Some(batch) = &self.pending {
            proof.closing_master = Some(batch.master.clone());
        }
        proof
    }

    pub fn checkpoint(&self) -> DistributedBendersCheckpoint {
        DistributedBendersCheckpoint {
            proof: self.proof.clone(),
            pending: self.pending.clone(),
        }
    }

    fn advance(&mut self) {
        if let Some(status) = self.state.status() {
            self.status = status;
            return;
        }
        if self.state.audit.verified_rounds >= self.state.options.max_rounds {
            self.status = DistributedBendersStatus::RoundLimit;
            return;
        }
        let options = self.state.options;
        let first_stage_columns = self.state.columns();
        let Ok(report) = solve_master_lp(
            &mut self.state.master,
            &solver(options.lp_iteration_limit, options.evidence_tolerance),
            options.evidence_tolerance,
            first_stage_columns,
        ) else {
            self.status = DistributedBendersStatus::MasterUnresolved;
            return;
        };
        let evidence = witness(report);
        let Ok(inspected) = self.state.inspect(&evidence) else {
            self.status = DistributedBendersStatus::MasterUnresolved;
            return;
        };
        self.state.audit = inspected.audit;
        if let Some(status) = self.state.status() {
            self.status = status;
            self.proof.closing_master = Some(evidence);
        } else {
            self.status = DistributedBendersStatus::AwaitingReplies;
            self.remaining_replies = self.state.problem.scenarios.len();
            self.pending = Some(PendingRound {
                master: evidence,
                replies: vec![None; self.state.problem.scenarios.len()],
            });
        }
    }
}

fn duplicate(
    previous: &RecourseReply,
    reply: &RecourseReply,
) -> Result<ReplyDisposition, OptimizeError> {
    if previous == reply {
        Ok(ReplyDisposition::Duplicate)
    } else {
        Err(invalid(
            "conflicting reply for an accepted distributed recourse job",
        ))
    }
}

struct AuditState {
    problem: BendersProblem,
    weights: Vec<f64>,
    options: DistributedBendersOptions,
    run_id: String,
    master: LpWorkspace,
    cuts: DecompositionCutPool,
    audit: DistributedBendersAudit,
    stalled: bool,
}

struct MasterAudit<'a> {
    audit: DistributedBendersAudit,
    point: Option<&'a [f64]>,
}

impl AuditState {
    fn replay(
        problem: BendersProblem,
        proof: &BendersProof,
        options: DistributedBendersOptions,
    ) -> Result<Self, OptimizeError> {
        if proof.version != PROTOCOL_VERSION || proof.run_id.trim().is_empty() {
            return Err(invalid(
                "invalid distributed Benders proof version or run ID",
            ));
        }
        if !valid_evidence_tolerance(options.evidence_tolerance) || options.max_rounds == 0 {
            return Err(invalid("invalid distributed Benders options"));
        }
        let weights = validate(
            &problem,
            BendersOptions {
                lp_iteration_limit: options.lp_iteration_limit,
                evidence_tolerance: options.evidence_tolerance,
                absolute_gap_tolerance: options.absolute_gap,
                relative_gap_tolerance: options.relative_gap,
                ..Default::default()
            },
        )?;
        let floors = problem.scenarios.iter().map(|s| {
            box_minimum(&s.recourse.objective, &s.recourse.var_bounds)
                .ok_or_else(|| invalid("distributed Benders requires a finite recourse objective floor from variable bounds"))
        }).collect::<Result<Vec<_>, _>>()?;
        let master = make_master(
            &problem.first_stage,
            &weights,
            &floors,
            BendersCutStrategy::Multi,
        )?;
        let mut state = Self {
            problem,
            weights,
            options,
            run_id: proof.run_id.clone(),
            master: LpWorkspace::new(master, None)?,
            cuts: DecompositionCutPool::default(),
            audit: DistributedBendersAudit::new(),
            stalled: false,
        };
        for round in &proof.rounds {
            state.apply(round)?;
        }
        if let Some(endpoint) = &proof.closing_master {
            state.audit = state.inspect(endpoint)?.audit;
        }
        Ok(state)
    }

    fn columns(&self) -> usize {
        self.problem.first_stage.objective.len()
    }

    fn key(&self, scenario: usize) -> JobKey {
        JobKey {
            run_id: self.run_id.clone(),
            round: self.audit.verified_rounds + 1,
            scenario,
        }
    }

    fn status(&self) -> Option<DistributedBendersStatus> {
        match self.audit.claim {
            DistributedBendersClaim::Optimality => Some(DistributedBendersStatus::Converged),
            DistributedBendersClaim::Infeasibility => Some(DistributedBendersStatus::Infeasible),
            DistributedBendersClaim::Bounds => {
                self.stalled.then_some(DistributedBendersStatus::Stalled)
            }
        }
    }

    fn inspect<'a>(&self, evidence: &'a LpWitness) -> Result<MasterAudit<'a>, OptimizeError> {
        let tol = self.options.evidence_tolerance;
        let mut audit = self.audit.clone();
        let point = match evidence {
            LpWitness::Optimal {
                point,
                row_multipliers,
            } => {
                let dual = verify_lp_dual(self.master.problem(), point, row_multipliers, tol)
                    .ok_or_else(|| {
                        invalid("distributed master primal-dual witness failed verification")
                    })?;
                audit.lower_bound = Some(
                    audit
                        .lower_bound
                        .map_or(dual.objective_bound, |old| old.max(dual.objective_bound)),
                );
                audit.check_gap(self.options)?;
                Some(point.as_slice())
            }
            LpWitness::Infeasible { row_multipliers } => {
                if audit.incumbent.is_some()
                    || verify_lp_farkas(self.master.problem(), row_multipliers, tol).is_none()
                {
                    return Err(invalid(
                        "distributed master infeasibility witness failed verification",
                    ));
                }
                audit.claim = DistributedBendersClaim::Infeasibility;
                None
            }
            LpWitness::Unresolved => {
                return Err(invalid(
                    "unresolved master cannot appear in a Benders proof",
                ))
            }
        };
        Ok(MasterAudit { audit, point })
    }

    fn check_reply(
        &self,
        reply: &RecourseReply,
        i: usize,
        point: &[f64],
    ) -> Result<(DecompositionCut, Option<BendersRecourseSolution>), OptimizeError> {
        if reply.version != PROTOCOL_VERSION || reply.key != self.key(i) {
            return Err(invalid(
                "distributed proof contains an unexpected reply key",
            ));
        }
        let scenario = &self.problem.scenarios[i];
        let model = shifted(scenario, &point[..self.columns()])?;
        let tol = self.options.evidence_tolerance;
        match &reply.evidence {
            LpWitness::Optimal {
                point: local,
                row_multipliers,
            } => {
                let dual =
                    verify_lp_dual(&model, local, row_multipliers, tol).ok_or_else(|| {
                        invalid("distributed recourse primal-dual witness failed verification")
                    })?;
                let plane = affine(scenario, &dual.row_multipliers, &dual.reduced_costs)?;
                let cut = plane.cut(
                    scenario.id.clone(),
                    reply.key.round,
                    CutKind::Optimality,
                    Some(self.columns() + i),
                );
                Ok((
                    cut,
                    Some(BendersRecourseSolution {
                        id: scenario.id.clone(),
                        point: local.clone(),
                        objective: dot(&model.objective, local),
                    }),
                ))
            }
            LpWitness::Infeasible { row_multipliers } => {
                let proof = verify_lp_farkas(&model, row_multipliers, tol).ok_or_else(|| {
                    invalid("distributed recourse Farkas witness failed verification")
                })?;
                let plane = affine(
                    scenario,
                    &proof.row_multipliers,
                    &proof.variable_multipliers,
                )?;
                let cut = plane.cut(
                    scenario.id.clone(),
                    reply.key.round,
                    CutKind::Feasibility,
                    None,
                );
                if violation(&cut, point)? <= tol {
                    return Err(invalid(
                        "projected Farkas witness does not separate the master point",
                    ));
                }
                Ok((cut, None))
            }
            LpWitness::Unresolved => Err(invalid(
                "unresolved recourse cannot appear in a Benders proof",
            )),
        }
    }

    fn apply(&mut self, round: &CertifiedRound) -> Result<(), OptimizeError> {
        if self.audit.claim == DistributedBendersClaim::Infeasibility
            || round.replies.len() != self.weights.len()
        {
            return Err(invalid("invalid completed distributed Benders round"));
        }
        let MasterAudit { mut audit, point } = self.inspect(&round.master)?;
        let point =
            point.ok_or_else(|| invalid("a completed round requires an optimal master witness"))?;
        let mut proposed = Vec::with_capacity(round.replies.len());
        let mut recourse = Vec::new();
        for (i, reply) in round.replies.iter().enumerate() {
            let (cut, solution) = self.check_reply(reply, i, point)?;
            proposed.push(cut);
            recourse.extend(solution);
        }
        if recourse.len() == self.weights.len() {
            let first = &point[..self.columns()];
            let objective = dot(&self.problem.first_stage.objective, first)
                + recourse
                    .iter()
                    .zip(&self.weights)
                    .map(|(r, w)| r.objective * w)
                    .sum::<f64>();
            if !objective.is_finite() {
                return Err(invalid(
                    "non-finite distributed Benders incumbent objective",
                ));
            }
            if audit.upper_bound.is_none_or(|old| objective < old) {
                audit.upper_bound = Some(objective);
                audit.incumbent = Some(BendersIncumbent {
                    first_stage: first.to_vec(),
                    recourse,
                    objective,
                });
            }
        }
        audit.check_gap(self.options)?;
        let mut cuts = self.cuts.clone();
        let mut append = Vec::new();
        for cut in proposed {
            let row = cut.lp_row();
            if matches!(cuts.insert(cut)?, CutInsertion::Inserted(_)) {
                append.push(row);
            }
        }
        self.stalled = append.is_empty();
        audit.cuts += append.len();
        audit.verified_rounds += 1;
        audit.verified_subproblems += round.replies.len();
        self.master.update(LpUpdate {
            append_rows: append,
            ..Default::default()
        })?;
        self.cuts = cuts;
        self.audit = audit;
        Ok(())
    }
}

fn shifted(scenario: &BendersScenario, x: &[f64]) -> Result<Problem, OptimizeError> {
    let mut lp = scenario.recourse.clone();
    for (bound, shift) in lp.row_bounds.iter_mut().zip(scenario.coupling.spmv(x)?) {
        if !shift.is_finite() {
            return Err(invalid("non-finite distributed Benders RHS shift"));
        }
        bound.lower = bound.lower.map(|v| v - shift);
        bound.upper = bound.upper.map(|v| v - shift);
    }
    lp.validate()?;
    Ok(lp)
}

fn solver(max_iterations: usize, tolerance: f64) -> RevisedSimplexSolver {
    RevisedSimplexSolver {
        max_iterations,
        tolerance: tolerance.min(1e-9),
    }
}

fn valid_evidence_tolerance(tolerance: f64) -> bool {
    tolerance.is_finite() && tolerance > 0.0
}

fn witness(report: LpSolveReport) -> LpWitness {
    if let Some(dual) = report.dual {
        LpWitness::Optimal {
            point: report.solution.variable_values,
            row_multipliers: dual.row_multipliers,
        }
    } else if let Some(proof) = report.infeasibility {
        LpWitness::Infeasible {
            row_multipliers: proof.row_multipliers,
        }
    } else {
        LpWitness::Unresolved
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
