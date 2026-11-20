//! Integer and logic-based branch-and-Benders-cut.

use crate::{
    benders::{affine, recourse_floor, solve_lp, violation},
    benders_tree::{round_integer_bounds, search, Candidate, TreeOptions, TreeStop},
    BendersIncumbent, BendersProblem, BendersRecourseSolution, BendersScenario, BendersTermination,
    Bound, CutInsertion, CutKind, CutOrigin, CutScope, DecompositionCut, DecompositionCutPool,
    LpColumn, LpSubproblemID, LpUpdate, LpWorkspace, OptimizeError, Problem, RevisedSimplexSolver,
    SolveStatus,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct LogicBendersScenario {
    pub id: LpSubproblemID,
    pub probability: f64,
    /// Ordered first-stage binary columns passed to the oracle.
    pub binary_columns: Vec<usize>,
    /// Caller-proven lower bound over every feasible assignment.
    pub objective_lower_bound: f64,
}

#[derive(Debug, Clone)]
pub enum IntegerBendersScenario {
    Model(BendersScenario),
    Logic(LogicBendersScenario),
}

#[derive(Debug, Clone)]
pub struct IntegerBendersProblem {
    pub first_stage: Problem,
    pub scenarios: Vec<IntegerBendersScenario>,
}

/// Assertions from a caller-supplied recourse solver. A scope is expressed
/// in positions of the input bit vector. None means the entire assignment.
#[derive(Debug, Clone)]
pub enum LogicBendersEvaluation {
    Optimal {
        objective: f64,
        point: Vec<f64>,
        /// The returned objective must lower-bound the value at every
        /// assignment matching these bits, and be attained at the trial.
        validity: Option<Vec<usize>>,
    },
    Infeasible {
        /// Every assignment matching this conflict must be infeasible.
        conflict: Option<Vec<usize>>,
    },
    Unresolved,
}

#[derive(Debug, Clone, Copy)]
pub struct IntegerBendersOptions {
    pub max_master_nodes: usize,
    pub max_candidate_evaluations: usize,
    pub max_recourse_nodes: usize,
    pub lp_iteration_limit: usize,
    pub simplex_tolerance: f64,
    pub evidence_tolerance: f64,
    pub integrality_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
}

impl Default for IntegerBendersOptions {
    fn default() -> Self {
        Self {
            max_master_nodes: 10_000,
            max_candidate_evaluations: 1_000,
            max_recourse_nodes: 10_000,
            lp_iteration_limit: 10_000,
            simplex_tolerance: 1e-9,
            evidence_tolerance: 1e-8,
            integrality_tolerance: 1e-7,
            absolute_gap_tolerance: 1e-7,
            relative_gap_tolerance: 1e-7,
        }
    }
}

impl IntegerBendersOptions {
    fn tree(self, recourse: bool) -> TreeOptions {
        TreeOptions {
            solve_limit: if recourse {
                self.max_recourse_nodes
            } else {
                self.max_master_nodes
            },
            candidate_limit: if recourse {
                self.max_recourse_nodes
            } else {
                self.max_candidate_evaluations
            },
            lp_iteration_limit: self.lp_iteration_limit,
            simplex_tolerance: self.simplex_tolerance,
            evidence_tolerance: self.evidence_tolerance,
            integrality_tolerance: self.integrality_tolerance,
            absolute_gap: if recourse {
                self.absolute_gap_tolerance.min(self.evidence_tolerance)
            } else {
                self.absolute_gap_tolerance
            },
            relative_gap: if recourse {
                self.relative_gap_tolerance.min(self.evidence_tolerance)
            } else {
                self.relative_gap_tolerance
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerBendersTermination {
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

#[derive(Debug, Clone, PartialEq)]
pub struct IntegerBendersResult {
    pub termination: IntegerBendersTermination,
    pub incumbent: Option<BendersIncumbent>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub master_nodes: usize,
    pub candidate_evaluations: usize,
    pub recourse_evaluations: usize,
    pub assignment_cache_hits: usize,
    pub feasibility_cuts: usize,
    pub optimality_cuts: usize,
    pub blocking_scenario: Option<LpSubproblemID>,
    /// The numerical bound and cut claims are conditional on the oracle's assertions.
    pub uses_logic_oracle: bool,
}

/// Model-based convenience entry point. First-stage variables may be mixed
/// integer. An integer recourse block may link only to binary master columns.
pub fn solve_integer_benders(
    problem: &BendersProblem,
    options: IntegerBendersOptions,
) -> Result<IntegerBendersResult, OptimizeError> {
    solve_logic_benders(
        &IntegerBendersProblem {
            first_stage: problem.first_stage.clone(),
            scenarios: problem
                .scenarios
                .iter()
                .cloned()
                .map(IntegerBendersScenario::Model)
                .collect(),
        },
        options,
        |_, _| Ok(LogicBendersEvaluation::Unresolved),
    )
}

/// Run one master search with globally shared cuts. The oracle runs
/// synchronously on candidate binary projections, in scenario input order.
pub fn solve_logic_benders(
    problem: &IntegerBendersProblem,
    options: IntegerBendersOptions,
    mut oracle: impl FnMut(&LpSubproblemID, &[bool]) -> Result<LogicBendersEvaluation, OptimizeError>,
) -> Result<IntegerBendersResult, OptimizeError> {
    let total = validate(problem, options)?;
    let mut result = IntegerBendersResult {
        termination: IntegerBendersTermination::MissingEvidence,
        incumbent: None,
        lower_bound: None,
        upper_bound: None,
        absolute_gap: None,
        relative_gap: None,
        master_nodes: 0,
        candidate_evaluations: 0,
        recourse_evaluations: 0,
        assignment_cache_hits: 0,
        feasibility_cuts: 0,
        optimality_cuts: 0,
        blocking_scenario: None,
        uses_logic_oracle: problem
            .scenarios
            .iter()
            .any(|s| matches!(s, IntegerBendersScenario::Logic(_))),
    };
    let mut first = problem.first_stage.clone();
    if !round_integer_bounds(&mut first)? {
        result.termination = IntegerBendersTermination::Infeasible;
        return Ok(result);
    }
    let mut relaxed_first = first.clone();
    relaxed_first.is_integer.fill(false);
    let lp_solver = RevisedSimplexSolver {
        max_iterations: options.lp_iteration_limit,
        tolerance: options.simplex_tolerance,
    };
    let mut prepared = Vec::new();
    for scenario in &problem.scenarios {
        let (kind, floor) = match scenario {
            IntegerBendersScenario::Logic(s) => {
                require_binary_columns(&first, &s.binary_columns)?;
                (Recourse::Logic(s), s.objective_lower_bound)
            }
            IntegerBendersScenario::Model(s) => {
                let mut lp = s.clone();
                let mixed = s.recourse.is_integer.iter().any(|&flag| flag);
                let columns = if mixed {
                    let columns: Vec<_> = s
                        .coupling
                        .iter_entries()
                        .filter_map(|(_, col, value)| (value != 0.0).then_some(col))
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    require_binary_columns(&first, &columns)?;
                    if !round_integer_bounds(&mut lp.recourse)? {
                        result.termination = IntegerBendersTermination::Infeasible;
                        result.blocking_scenario = Some(s.id.clone());
                        return Ok(result);
                    }
                    columns
                } else {
                    Vec::new()
                };
                lp.recourse.is_integer.fill(false);
                let floor = match recourse_floor(
                    &relaxed_first,
                    &lp,
                    &lp_solver,
                    options.evidence_tolerance,
                )? {
                    Ok(bound) => bound,
                    Err(reason) => {
                        result.termination = match reason {
                            BendersTermination::Infeasible => IntegerBendersTermination::Infeasible,
                            BendersTermination::LpIterationLimit => {
                                IntegerBendersTermination::LpIterationLimit
                            }
                            BendersTermination::RecourseBoundUnavailable => {
                                IntegerBendersTermination::RecourseBoundUnavailable
                            }
                            _ => IntegerBendersTermination::MissingEvidence,
                        };
                        result.blocking_scenario = Some(s.id.clone());
                        return Ok(result);
                    }
                };
                if mixed {
                    (Recourse::Integer(s, columns), floor)
                } else {
                    (
                        Recourse::Linear(
                            s,
                            Box::new(LpWorkspace::new(lp.recourse, Some(s.id.clone()))?),
                        ),
                        floor,
                    )
                }
            }
        };
        prepared.push(Prepared {
            kind,
            floor,
            weight: scenario.probability() / total,
            cache: BTreeMap::new(),
        });
    }

    let n = first.objective.len();
    let mut model = LpWorkspace::new(relaxed_first, None)?;
    model.update(LpUpdate {
        append_columns: prepared
            .iter()
            .map(|s| LpColumn {
                entries: Vec::new(),
                objective: s.weight,
                bound: Bound {
                    lower: Some(s.floor),
                    upper: None,
                },
            })
            .collect(),
        ..Default::default()
    })?;
    let mut master = model.problem().clone();
    master.is_integer[..n].copy_from_slice(&first.is_integer);
    let mut cuts = DecompositionCutPool::default();
    let mut blocked = None;
    let tree = search(master, &options.tree(false), |trial, _| {
        let x = &trial[..n];
        let mut objective = x
            .iter()
            .zip(&first.objective)
            .map(|(x, c)| x * c)
            .sum::<f64>();
        let mut plans = Vec::new();
        let mut rows = Vec::new();
        let iteration = result.candidate_evaluations + 1;
        result.candidate_evaluations = iteration;
        for (index, scenario) in prepared.iter_mut().enumerate() {
            let id = scenario.kind.id().clone();
            let evaluation = evaluate(
                scenario,
                x,
                &lp_solver,
                options,
                &mut oracle,
                &mut result.recourse_evaluations,
                &mut result.assignment_cache_hits,
            )?;
            let (kind, cut) = match evaluation {
                Evaluation::ContinuousOptimal { solution, plane } => {
                    objective += scenario.weight * solution.objective;
                    plans.push(solution);
                    (
                        CutKind::Optimality,
                        plane.cut(id.clone(), iteration, CutKind::Optimality, Some(n + index)),
                    )
                }
                Evaluation::ContinuousInfeasible(plane) => (
                    CutKind::Feasibility,
                    plane.cut(id.clone(), iteration, CutKind::Feasibility, None),
                ),
                Evaluation::Discrete(Discrete::Optimal {
                    objective: cost,
                    lower,
                    point,
                    scope,
                }) => {
                    if (scenario.floor - cost) / scenario.floor.abs().max(cost.abs()).max(1.0)
                        > options.evidence_tolerance
                    {
                        blocked = Some((IntegerBendersTermination::MissingEvidence, id));
                        return Ok(Candidate::Stop);
                    }
                    objective += scenario.weight * cost;
                    plans.push(BendersRecourseSolution {
                        id: id.clone(),
                        point,
                        objective: cost,
                    });
                    let cut = assignment_cut(
                        scope,
                        x,
                        id.clone(),
                        iteration,
                        Some((n + index, scenario.floor, lower)),
                    )?;
                    (CutKind::Optimality, cut)
                }
                Evaluation::Discrete(Discrete::Infeasible(scope)) => (
                    CutKind::Feasibility,
                    assignment_cut(scope, x, id.clone(), iteration, None)?,
                ),
                Evaluation::Blocked(reason) => {
                    blocked = Some((reason, id));
                    return Ok(Candidate::Stop);
                }
            };
            let excess = violation(&cut, trial)?;
            if kind == CutKind::Feasibility && excess <= options.evidence_tolerance {
                blocked = Some((IntegerBendersTermination::MissingEvidence, id));
                return Ok(Candidate::Stop);
            }
            if excess > options.evidence_tolerance {
                let row = cut.lp_row();
                if let CutInsertion::Inserted(_) = cuts.insert(cut)? {
                    rows.push((kind, row));
                }
            }
        }
        for (kind, _) in &rows {
            match kind {
                CutKind::Feasibility => result.feasibility_cuts += 1,
                CutKind::Optimality => result.optimality_cuts += 1,
            }
        }
        let incumbent = (plans.len() == prepared.len()).then(|| {
            (
                objective,
                BendersIncumbent {
                    first_stage: x.to_vec(),
                    recourse: plans,
                    objective,
                },
            )
        });
        Ok(Candidate::Evaluated {
            incumbent,
            rows: rows.into_iter().map(|(_, row)| row).collect(),
        })
    })?;
    result.master_nodes = tree.solves;
    result.candidate_evaluations = tree.candidates;
    result.lower_bound = tree.lower;
    if let Some((value, incumbent)) = tree.incumbent {
        result.upper_bound = Some(value);
        result.incumbent = Some(incumbent);
        if let Some(lower) = tree.lower {
            let scale = value.abs().max(lower.abs()).max(1.0);
            if (lower - value) / scale <= options.evidence_tolerance {
                let gap = (value - lower).max(0.0);
                result.absolute_gap = gap.is_finite().then_some(gap);
                result.relative_gap = Some(
                    result
                        .absolute_gap
                        .map_or_else(|| value / scale - lower / scale, |g| g / scale),
                );
            }
        }
    }
    result.termination = match tree.stop {
        TreeStop::Converged => IntegerBendersTermination::Converged,
        TreeStop::Infeasible => IntegerBendersTermination::Infeasible,
        TreeStop::NodeLimit => IntegerBendersTermination::MasterNodeLimit,
        TreeStop::CandidateLimit => IntegerBendersTermination::CandidateLimit,
        TreeStop::LpIterationLimit => IntegerBendersTermination::LpIterationLimit,
        TreeStop::UnboundedRelaxation => IntegerBendersTermination::MasterUnbounded,
        TreeStop::Stalled => IntegerBendersTermination::Stalled,
        TreeStop::MissingEvidence | TreeStop::EvaluationStopped => {
            IntegerBendersTermination::MissingEvidence
        }
    };
    if let Some((reason, id)) = blocked {
        result.termination = reason;
        result.blocking_scenario = Some(id);
    }
    Ok(result)
}

struct Prepared<'a> {
    kind: Recourse<'a>,
    floor: f64,
    weight: f64,
    cache: BTreeMap<Vec<bool>, Discrete>,
}

enum Recourse<'a> {
    Linear(&'a BendersScenario, Box<LpWorkspace>),
    Integer(&'a BendersScenario, Vec<usize>),
    Logic(&'a LogicBendersScenario),
}

impl Recourse<'_> {
    fn id(&self) -> &LpSubproblemID {
        match self {
            Self::Linear(s, _) | Self::Integer(s, _) => &s.id,
            Self::Logic(s) => &s.id,
        }
    }
}

#[derive(Clone)]
enum Discrete {
    Optimal {
        objective: f64,
        lower: f64,
        point: Vec<f64>,
        scope: Vec<usize>,
    },
    Infeasible(Vec<usize>),
}

enum Evaluation {
    ContinuousOptimal {
        solution: BendersRecourseSolution,
        plane: crate::benders::Affine,
    },
    ContinuousInfeasible(crate::benders::Affine),
    Discrete(Discrete),
    Blocked(IntegerBendersTermination),
}

fn evaluate(
    prepared: &mut Prepared<'_>,
    x: &[f64],
    solver: &RevisedSimplexSolver,
    options: IntegerBendersOptions,
    oracle: &mut impl FnMut(&LpSubproblemID, &[bool]) -> Result<LogicBendersEvaluation, OptimizeError>,
    evaluations: &mut usize,
    cache_hits: &mut usize,
) -> Result<Evaluation, OptimizeError> {
    if let Recourse::Linear(s, workspace) = &mut prepared.kind {
        *evaluations += 1;
        let row_bounds = shifted_bounds(s, x)?.into_iter().enumerate().collect();
        workspace.update(LpUpdate {
            row_bounds,
            ..Default::default()
        })?;
        let report = solve_lp(workspace, solver, options.evidence_tolerance)?;
        if let Some(dual) = report.dual {
            return Ok(Evaluation::ContinuousOptimal {
                solution: BendersRecourseSolution {
                    id: s.id.clone(),
                    point: report.solution.variable_values,
                    objective: report.solution.objective_value,
                },
                plane: affine(s, &dual.row_multipliers, &dual.reduced_costs)?,
            });
        }
        if let Some(farkas) = report.infeasibility {
            return Ok(Evaluation::ContinuousInfeasible(affine(
                s,
                &farkas.row_multipliers,
                &farkas.variable_multipliers,
            )?));
        }
        return Ok(Evaluation::Blocked(
            if report.solution.status == SolveStatus::IterationLimit {
                IntegerBendersTermination::LpIterationLimit
            } else {
                IntegerBendersTermination::MissingEvidence
            },
        ));
    }
    let columns = match &prepared.kind {
        Recourse::Integer(_, columns) => columns,
        Recourse::Logic(s) => &s.binary_columns,
        Recourse::Linear(_, _) => unreachable!(),
    };
    let bits: Vec<_> = columns.iter().map(|&i| x[i] == 1.0).collect();
    if let Some(previous) = prepared.cache.get(&bits) {
        *cache_hits += 1;
        return Ok(Evaluation::Discrete(previous.clone()));
    }
    *evaluations += 1;
    let outcome = match &prepared.kind {
        Recourse::Logic(s) => match oracle(&s.id, &bits)? {
            LogicBendersEvaluation::Optimal {
                objective,
                point,
                validity,
            } => {
                if !objective.is_finite() || point.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("non-finite logic recourse result"));
                }
                Discrete::Optimal {
                    objective,
                    lower: objective,
                    point,
                    scope: scope(columns, validity)?,
                }
            }
            LogicBendersEvaluation::Infeasible { conflict } => {
                Discrete::Infeasible(scope(columns, conflict)?)
            }
            LogicBendersEvaluation::Unresolved => {
                return Ok(Evaluation::Blocked(
                    IntegerBendersTermination::OracleUnresolved,
                ))
            }
        },
        Recourse::Integer(s, _) => {
            let mut problem = s.recourse.clone();
            problem.row_bounds = shifted_bounds(s, x)?;
            let report = search(problem, &options.tree(true), |trial, objective| {
                Ok(Candidate::Evaluated {
                    incumbent: Some((objective, trial.to_vec())),
                    rows: Vec::new(),
                })
            })?;
            match (report.stop, report.incumbent, report.lower) {
                (TreeStop::Converged, Some((objective, point)), Some(lower)) => Discrete::Optimal {
                    objective,
                    lower: lower.min(objective),
                    point,
                    scope: columns.to_vec(),
                },
                (TreeStop::Infeasible, _, _) => Discrete::Infeasible(columns.to_vec()),
                (TreeStop::NodeLimit | TreeStop::CandidateLimit, _, _) => {
                    return Ok(Evaluation::Blocked(
                        IntegerBendersTermination::RecourseNodeLimit,
                    ))
                }
                (TreeStop::LpIterationLimit, _, _) => {
                    return Ok(Evaluation::Blocked(
                        IntegerBendersTermination::LpIterationLimit,
                    ))
                }
                _ => {
                    return Ok(Evaluation::Blocked(
                        IntegerBendersTermination::MissingEvidence,
                    ))
                }
            }
        }
        Recourse::Linear(_, _) => unreachable!(),
    };
    prepared.cache.insert(bits, outcome.clone());
    Ok(Evaluation::Discrete(outcome))
}

fn shifted_bounds(s: &BendersScenario, x: &[f64]) -> Result<Vec<Bound>, OptimizeError> {
    let shifts = s.coupling.spmv(x)?;
    if shifts.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite recourse RHS shift"));
    }
    Ok(s.recourse
        .row_bounds
        .iter()
        .zip(shifts)
        .map(|(bound, shift)| Bound {
            lower: bound.lower.map(|value| value - shift),
            upper: bound.upper.map(|value| value - shift),
        })
        .collect())
}

fn scope(columns: &[usize], positions: Option<Vec<usize>>) -> Result<Vec<usize>, OptimizeError> {
    let Some(positions) = positions else {
        return Ok(columns.to_vec());
    };
    let mut seen = BTreeSet::new();
    for position in positions {
        let index = *columns
            .get(position)
            .ok_or_else(|| invalid("oracle scope is outside its binary projection"))?;
        if !seen.insert(index) {
            return Err(invalid("oracle scope contains duplicate positions"));
        }
    }
    Ok(seen.into_iter().collect())
}

fn assignment_cut(
    columns: Vec<usize>,
    x: &[f64],
    source: LpSubproblemID,
    iteration: usize,
    optimal: Option<(usize, f64, f64)>,
) -> Result<DecompositionCut, OptimizeError> {
    let ones = columns.iter().filter(|&&i| x[i] == 1.0).count() as f64;
    let (coefficient, upper, kind) = if let Some((_, floor, lower)) = optimal {
        let delta = (lower - floor).max(0.0);
        (delta, delta * ones - lower, CutKind::Optimality)
    } else {
        (1.0, ones - 1.0, CutKind::Feasibility)
    };
    if !coefficient.is_finite() || !upper.is_finite() {
        return Err(invalid("non-finite assignment cut"));
    }
    let mut entries: Vec<_> = columns
        .into_iter()
        .map(|i| {
            (
                i,
                if x[i] == 1.0 {
                    coefficient
                } else {
                    -coefficient
                },
            )
        })
        .collect();
    if let Some((theta, _, _)) = optimal {
        entries.push((theta, -1.0));
    }
    Ok(DecompositionCut {
        entries,
        upper,
        origin: CutOrigin {
            source,
            iteration,
            kind,
        },
        scope: CutScope::Global,
    })
}

impl IntegerBendersScenario {
    fn probability(&self) -> f64 {
        match self {
            Self::Model(s) => s.probability,
            Self::Logic(s) => s.probability,
        }
    }
}

fn require_binary_columns(first: &Problem, columns: &[usize]) -> Result<(), OptimizeError> {
    let mut seen = BTreeSet::new();
    for &column in columns {
        if !seen.insert(column)
            || first.is_integer.get(column) != Some(&true)
            || !first.var_bounds.get(column).is_some_and(|b| {
                b.lower.is_some_and(|value| value >= 0.0)
                    && b.upper.is_some_and(|value| value <= 1.0)
            })
        {
            return Err(invalid(
                "discrete recourse must link to distinct binary master columns",
            ));
        }
    }
    Ok(())
}

fn validate(
    problem: &IntegerBendersProblem,
    options: IntegerBendersOptions,
) -> Result<f64, OptimizeError> {
    if [
        options.max_master_nodes,
        options.max_candidate_evaluations,
        options.max_recourse_nodes,
        options.lp_iteration_limit,
    ]
    .contains(&0)
        || ![
            options.simplex_tolerance,
            options.evidence_tolerance,
            options.integrality_tolerance,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0)
        || options.integrality_tolerance >= 0.5
        || ![
            options.absolute_gap_tolerance,
            options.relative_gap_tolerance,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid integer Benders options",
        ));
    }
    problem.first_stage.validate()?;
    let mut identities = BTreeSet::new();
    let mut total = 0.0;
    for s in &problem.scenarios {
        let id = match s {
            IntegerBendersScenario::Model(s) => {
                s.recourse.validate()?;
                if s.coupling.cols() != problem.first_stage.objective.len()
                    || s.coupling.rows() != s.recourse.row_bounds.len()
                    || s.coupling
                        .iter_entries()
                        .any(|(_, _, value)| !value.is_finite())
                {
                    return Err(invalid("invalid integer Benders coupling matrix"));
                }
                &s.id
            }
            IntegerBendersScenario::Logic(s) => {
                if !s.objective_lower_bound.is_finite() {
                    return Err(invalid("logic scenario needs a finite floor"));
                }
                &s.id
            }
        };
        if !identities.insert(id)
            || id.name.trim().is_empty()
            || !s.probability().is_finite()
            || s.probability() <= 0.0
        {
            return Err(invalid(
                "invalid integer Benders scenario identity or probability",
            ));
        }
        total += s.probability();
    }
    if (total - 1.0).abs() > 1e-10 {
        return Err(invalid("scenario probabilities must sum to one"));
    }
    Ok(total)
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
