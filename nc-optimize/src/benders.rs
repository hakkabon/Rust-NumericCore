//! Classical Benders and the two-stage L-shaped method for continuous LPs.

use crate::{
    Bound, CutInsertion, CutKind, CutOrigin, CutScope, DecompositionCut, DecompositionCutPool,
    LpColumn, LpSolveReport, LpSubproblemID, LpUpdate, LpWorkspace, OptimizeError, Problem,
    RevisedSimplexSolver, SolveStatus,
};
use nc_sparse::{CooMatrix, CsrMatrix};
use std::{collections::BTreeSet, thread};

/// Recourse rows are l ≤ W y + T x ≤ u. The recourse LP stores W and the
/// unshifted bounds. The coupling T has one column per first-stage variable.
#[derive(Debug, Clone)]
pub struct BendersScenario {
    pub id: LpSubproblemID,
    pub probability: f64,
    pub recourse: Problem,
    pub coupling: CsrMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct BendersProblem {
    pub first_stage: Problem,
    pub scenarios: Vec<BendersScenario>,
}

impl BendersProblem {
    /// Classical Benders with a single, deterministic recourse block.
    pub fn classical(first_stage: Problem, recourse: Problem, coupling: CsrMatrix<f64>) -> Self {
        Self {
            first_stage,
            scenarios: vec![BendersScenario {
                id: LpSubproblemID {
                    name: "recourse".into(),
                    scenario: None,
                },
                probability: 1.0,
                recourse,
                coupling,
            }],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BendersCutStrategy {
    Single,
    Multi,
}

#[derive(Debug, Clone, Copy)]
pub struct BendersOptions {
    pub cut_strategy: BendersCutStrategy,
    pub max_iterations: usize,
    pub lp_iteration_limit: usize,
    pub simplex_tolerance: f64,
    pub evidence_tolerance: f64,
    pub absolute_gap_tolerance: f64,
    pub relative_gap_tolerance: f64,
    pub scenario_threads: usize,
    pub maximum_inactive_rounds: Option<usize>,
    /// At most every other iteration, solve an additional restricted master.
    pub trust_region_radius: Option<f64>,
}

impl Default for BendersOptions {
    fn default() -> Self {
        Self {
            cut_strategy: BendersCutStrategy::Multi,
            max_iterations: 100,
            lp_iteration_limit: 10_000,
            simplex_tolerance: 1e-9,
            evidence_tolerance: 1e-8,
            absolute_gap_tolerance: 1e-7,
            relative_gap_tolerance: 1e-7,
            scenario_threads: 1,
            maximum_inactive_rounds: None,
            trust_region_radius: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BendersTermination {
    Converged,
    Infeasible,
    IterationLimit,
    LpIterationLimit,
    MissingEvidence,
    /// A recourse floor could not be established. This is not a global claim.
    RecourseBoundUnavailable,
    /// The LP relaxation is unbounded. Further cuts could still bound it.
    MasterUnbounded,
    Stalled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BendersRecourseSolution {
    pub id: LpSubproblemID,
    pub point: Vec<f64>,
    pub objective: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BendersIncumbent {
    pub first_stage: Vec<f64>,
    pub recourse: Vec<BendersRecourseSolution>,
    pub objective: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BendersIteration {
    pub iteration: usize,
    pub lower_bound: f64,
    pub upper_bound: Option<f64>,
    pub feasibility_cuts: usize,
    pub optimality_cuts: usize,
    pub cuts_pruned: usize,
    pub used_trust_region: bool,
    pub warm_subproblems: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BendersResult {
    pub termination: BendersTermination,
    pub incumbent: Option<BendersIncumbent>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub absolute_gap: Option<f64>,
    pub relative_gap: Option<f64>,
    pub history: Vec<BendersIteration>,
    pub blocking_scenario: Option<LpSubproblemID>,
}

impl BendersResult {
    fn new() -> Self {
        Self {
            termination: BendersTermination::IterationLimit,
            incumbent: None,
            lower_bound: None,
            upper_bound: None,
            absolute_gap: None,
            relative_gap: None,
            history: Vec::new(),
            blocking_scenario: None,
        }
    }

    fn update_bounds(&mut self, lower: f64, candidate: Option<BendersIncumbent>) {
        self.lower_bound = Some(self.lower_bound.map_or(lower, |old| old.max(lower)));
        if let Some(candidate) = candidate {
            if self
                .upper_bound
                .is_none_or(|upper| candidate.objective < upper)
            {
                self.upper_bound = Some(candidate.objective);
                self.incumbent = Some(candidate);
            }
        }
        if let (Some(lower), Some(upper)) = (self.lower_bound, self.upper_bound) {
            let gap = (upper - lower).max(0.0);
            self.absolute_gap = gap.is_finite().then_some(gap);
            let scale = upper.abs().max(lower.abs()).max(1.0);
            let relative = self
                .absolute_gap
                .map_or_else(|| upper / scale - lower / scale, |gap| gap / scale);
            self.relative_gap = Some(relative.max(0.0));
        }
    }
}

/// Solve the continuous two-stage problem with deterministic scenario order.
pub fn solve_benders(
    problem: &BendersProblem,
    options: BendersOptions,
) -> Result<BendersResult, OptimizeError> {
    let weights = validate(problem, options)?;
    let solver = RevisedSimplexSolver {
        max_iterations: options.lp_iteration_limit,
        tolerance: options.simplex_tolerance,
    };
    let mut result = BendersResult::new();
    let mut floors = Vec::with_capacity(problem.scenarios.len());
    for scenario in &problem.scenarios {
        match recourse_floor(
            &problem.first_stage,
            scenario,
            &solver,
            options.evidence_tolerance,
        )? {
            Ok(bound) => floors.push(bound),
            Err(termination) => {
                result.termination = termination;
                result.blocking_scenario = Some(scenario.id.clone());
                return Ok(result);
            }
        }
    }

    let n = problem.first_stage.objective.len();
    let base = make_master(
        &problem.first_stage,
        &weights,
        &floors,
        options.cut_strategy,
    )?;
    let mut master = LpWorkspace::new(base.clone(), None)?;
    let mut pool = DecompositionCutPool::default();
    let mut scenarios = problem
        .scenarios
        .iter()
        .map(|scenario| LpWorkspace::new(scenario.recourse.clone(), Some(scenario.id.clone())))
        .collect::<Result<Vec<_>, _>>()?;

    for iteration in 1..=options.max_iterations {
        let global = solve_lp(&mut master, &solver, options.evidence_tolerance)?;
        let Some(dual) = &global.dual else {
            result.termination = match global.solution.status {
                SolveStatus::Infeasible
                    if global.infeasibility.is_some() && result.incumbent.is_none() =>
                {
                    BendersTermination::Infeasible
                }
                SolveStatus::Unbounded if global.unbounded_ray.is_some() => {
                    BendersTermination::MasterUnbounded
                }
                SolveStatus::IterationLimit => BendersTermination::LpIterationLimit,
                _ => BendersTermination::MissingEvidence,
            };
            return Ok(result);
        };
        result.update_bounds(dual.objective_bound, None);
        let mut trial = std::borrow::Cow::Borrowed(global.solution.variable_values.as_slice());
        let mut stabilized = false;
        if let (Some(radius), Some(center)) = (options.trust_region_radius, &result.incumbent) {
            if iteration % 2 == 0 {
                if let Some(point) = trust_trial(
                    &mut master,
                    &base,
                    &center.first_stage,
                    radius,
                    &solver,
                    options.evidence_tolerance,
                )? {
                    stabilized = true;
                    trial = std::borrow::Cow::Owned(point);
                }
            }
        }
        let reports = solve_scenarios(
            &mut scenarios,
            &problem.scenarios,
            &trial[..n],
            &solver,
            options,
        )?;
        let warm_subproblems = reports
            .iter()
            .filter(|report| report.used_warm_start)
            .count();
        let mut planes = Vec::new();
        let mut proposed = Vec::new();
        let mut recourse = Vec::new();
        let mut objective = dot(&problem.first_stage.objective, &trial[..n]);
        for (index, ((scenario, report), &weight)) in problem
            .scenarios
            .iter()
            .zip(reports)
            .zip(&weights)
            .enumerate()
        {
            if let Some(dual) = report.dual {
                let value = report.solution.objective_value;
                objective += weight * value;
                recourse.push(BendersRecourseSolution {
                    id: scenario.id.clone(),
                    point: report.solution.variable_values,
                    objective: value,
                });
                planes.push((
                    index,
                    affine(scenario, &dual.row_multipliers, &dual.reduced_costs)?,
                ));
            } else if let Some(farkas) = report.infeasibility {
                let plane = affine(
                    scenario,
                    &farkas.row_multipliers,
                    &farkas.variable_multipliers,
                )?;
                let cut = plane.cut(scenario.id.clone(), iteration, CutKind::Feasibility, None);
                if violation(&cut, &trial)? <= options.evidence_tolerance {
                    result.termination = BendersTermination::MissingEvidence;
                    result.blocking_scenario = Some(scenario.id.clone());
                    return Ok(result);
                }
                proposed.push(cut);
            } else {
                result.termination = if report.solution.status == SolveStatus::IterationLimit {
                    BendersTermination::LpIterationLimit
                } else {
                    BendersTermination::MissingEvidence
                };
                result.blocking_scenario = Some(scenario.id.clone());
                return Ok(result);
            }
        }

        let all_feasible = recourse.len() == weights.len();
        match options.cut_strategy {
            BendersCutStrategy::Single if all_feasible => {
                let plane = Affine::weighted_sum(n, planes.iter().map(|(_, p)| p).zip(&weights))?;
                let origin = LpSubproblemID {
                    name: "weighted recourse".into(),
                    scenario: None,
                };
                proposed.push(plane.cut(origin, iteration, CutKind::Optimality, Some(n)));
            }
            BendersCutStrategy::Multi => {
                for (i, plane) in planes {
                    proposed.push(plane.cut(
                        problem.scenarios[i].id.clone(),
                        iteration,
                        CutKind::Optimality,
                        Some(n + i),
                    ));
                }
            }
            _ => {}
        }
        if all_feasible {
            if !objective.is_finite() {
                return Err(invalid("non-finite Benders incumbent objective"));
            }
            result.update_bounds(
                dual.objective_bound,
                Some(BendersIncumbent {
                    first_stage: trial[..n].to_vec(),
                    recourse,
                    objective,
                }),
            );
        }
        if let (Some(lower), Some(upper)) = (result.lower_bound, result.upper_bound) {
            if lower > upper
                && (lower - upper) / upper.abs().max(lower.abs()).max(1.0)
                    > options.evidence_tolerance
            {
                result.termination = BendersTermination::MissingEvidence;
                result.absolute_gap = None;
                result.relative_gap = None;
                return Ok(result);
            }
        }

        let mut rows = Vec::new();
        let mut entry = BendersIteration {
            iteration,
            lower_bound: result.lower_bound.unwrap(),
            upper_bound: result.upper_bound,
            feasibility_cuts: 0,
            optimality_cuts: 0,
            cuts_pruned: 0,
            used_trust_region: stabilized,
            warm_subproblems,
        };
        let mut active = BTreeSet::new();
        if options.maximum_inactive_rounds.is_some() {
            for stored in pool.entries() {
                if violation(&stored.cut, &global.solution.variable_values)?
                    >= -options.evidence_tolerance
                    || (stabilized
                        && violation(&stored.cut, &trial)? >= -options.evidence_tolerance)
                {
                    active.insert(stored.id);
                }
            }
            pool.age(&active);
        }
        for cut in proposed {
            if violation(&cut, &trial)? <= options.evidence_tolerance {
                continue;
            }
            let kind = cut.origin.kind;
            let row = cut.lp_row();
            if let CutInsertion::Inserted(_) = pool.insert(cut)? {
                match kind {
                    CutKind::Feasibility => entry.feasibility_cuts += 1,
                    CutKind::Optimality => entry.optimality_cuts += 1,
                }
                rows.push(row);
            }
        }
        // Preserve a strong master near stagnation. Prior certified bounds
        // remain valid if the active cut set is subsequently pruned.
        if let Some(age) = options.maximum_inactive_rounds.filter(|_| !rows.is_empty()) {
            entry.cuts_pruned = pool.prune(age).len();
        }
        let pruned = entry.cuts_pruned > 0;
        result.history.push(entry);
        if result
            .absolute_gap
            .is_some_and(|gap| gap <= options.absolute_gap_tolerance)
            || result
                .relative_gap
                .is_some_and(|gap| gap <= options.relative_gap_tolerance)
        {
            result.termination = BendersTermination::Converged;
            return Ok(result);
        }
        if rows.is_empty() && !stabilized {
            result.termination = BendersTermination::Stalled;
            return Ok(result);
        }
        if pruned {
            let mut rebuilt = LpWorkspace::new(base.clone(), None)?;
            let cut_rows = pool.entries().map(|entry| entry.cut.lp_row()).collect();
            rebuilt.update(LpUpdate {
                append_rows: cut_rows,
                ..Default::default()
            })?;
            master = rebuilt;
        } else if !rows.is_empty() {
            master.update(LpUpdate {
                append_rows: rows,
                ..Default::default()
            })?;
        }
    }
    Ok(result)
}

fn validate(problem: &BendersProblem, options: BendersOptions) -> Result<Vec<f64>, OptimizeError> {
    let continuous = |p: &Problem| -> Result<(), OptimizeError> {
        p.validate()?;
        if p.is_integer.iter().any(|&flag| flag) {
            return Err(invalid("Benders requires continuous LPs"));
        }
        Ok(())
    };
    continuous(&problem.first_stage)?;
    if problem.scenarios.is_empty() {
        return Err(invalid("Benders requires at least one scenario"));
    }
    if options.max_iterations == 0
        || options.lp_iteration_limit == 0
        || options.scenario_threads == 0
        || ![options.simplex_tolerance, options.evidence_tolerance]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
        || ![
            options.absolute_gap_tolerance,
            options.relative_gap_tolerance,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v >= 0.0)
        || options
            .trust_region_radius
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid Benders options",
        ));
    }
    let mut ids = BTreeSet::new();
    for s in &problem.scenarios {
        continuous(&s.recourse)?;
        if s.id.name.trim().is_empty()
            || !ids.insert(&s.id)
            || !s.probability.is_finite()
            || s.probability <= 0.0
            || s.coupling.cols() != problem.first_stage.objective.len()
            || s.coupling.rows() != s.recourse.row_bounds.len()
            || !s.coupling.iter_entries().all(|(_, _, v)| v.is_finite())
        {
            return Err(invalid(
                "invalid Benders scenario identity, probability, or coupling",
            ));
        }
    }
    let sum: f64 = problem.scenarios.iter().map(|s| s.probability).sum();
    if !sum.is_finite() || (sum - 1.0).abs() > 1e-10 {
        return Err(invalid("scenario probabilities must sum to one"));
    }
    Ok(problem
        .scenarios
        .iter()
        .map(|s| s.probability / sum)
        .collect())
}

type Floor = Result<f64, BendersTermination>;

fn recourse_floor(
    first: &Problem,
    scenario: &BendersScenario,
    solver: &RevisedSimplexSolver,
    tolerance: f64,
) -> Result<Floor, OptimizeError> {
    if let Some(bound) = box_minimum(&scenario.recourse.objective, &scenario.recourse.var_bounds) {
        return Ok(Ok(bound));
    }
    // One auxiliary LP certifies the floor over all feasible (x, y) for
    // this scenario, independently of the other scenarios.
    let n = first.objective.len();
    let m = first.row_bounds.len();
    let columns = n + scenario.recourse.objective.len();
    let mut constraints = CooMatrix::new(m + scenario.recourse.row_bounds.len(), columns);
    for (row, col, value) in first.constraints.iter_entries() {
        constraints.push(row, col, value)?;
    }
    for (row, col, value) in scenario.coupling.iter_entries() {
        constraints.push(m + row, col, value)?;
    }
    for (row, col, value) in scenario.recourse.constraints.iter_entries() {
        constraints.push(m + row, n + col, value)?;
    }
    let problem = Problem {
        objective: std::iter::repeat_n(0.0, n)
            .chain(scenario.recourse.objective.iter().copied())
            .collect(),
        constraints: constraints.to_csr()?,
        row_bounds: first
            .row_bounds
            .iter()
            .chain(&scenario.recourse.row_bounds)
            .copied()
            .collect(),
        var_bounds: first
            .var_bounds
            .iter()
            .chain(&scenario.recourse.var_bounds)
            .copied()
            .collect(),
        is_integer: vec![false; columns],
    };
    let report = LpWorkspace::new(problem, None)?.solve(solver, tolerance)?;
    Ok(match report.solution.status {
        SolveStatus::Optimal => report
            .dual
            .map(|dual| dual.objective_bound)
            .ok_or(BendersTermination::MissingEvidence),
        SolveStatus::Infeasible if report.infeasibility.is_some() => {
            Err(BendersTermination::Infeasible)
        }
        SolveStatus::Unbounded if report.unbounded_ray.is_some() => {
            Err(BendersTermination::RecourseBoundUnavailable)
        }
        SolveStatus::IterationLimit => Err(BendersTermination::LpIterationLimit),
        _ => Err(BendersTermination::MissingEvidence),
    })
}

fn make_master(
    first: &Problem,
    weights: &[f64],
    floors: &[f64],
    strategy: BendersCutStrategy,
) -> Result<Problem, OptimizeError> {
    let columns: Vec<_> = match strategy {
        BendersCutStrategy::Single => vec![(1.0, dot(weights, floors))],
        BendersCutStrategy::Multi => weights
            .iter()
            .copied()
            .zip(floors.iter().copied())
            .collect(),
    };
    let mut workspace = LpWorkspace::new(first.clone(), None)?;
    workspace.update(LpUpdate {
        append_columns: columns
            .into_iter()
            .map(|(objective, floor)| LpColumn {
                entries: vec![],
                bound: Bound {
                    lower: Some(floor),
                    upper: None,
                },
                objective,
            })
            .collect(),
        ..Default::default()
    })?;
    Ok(workspace.problem().clone())
}

fn trust_trial(
    master: &mut LpWorkspace,
    base: &Problem,
    center: &[f64],
    radius: f64,
    solver: &RevisedSimplexSolver,
    tolerance: f64,
) -> Result<Option<Vec<f64>>, OptimizeError> {
    let original: Vec<_> = base
        .var_bounds
        .iter()
        .copied()
        .take(center.len())
        .enumerate()
        .collect();
    let restricted: Vec<_> = original
        .iter()
        .zip(center)
        .map(|(&(i, bound), &value)| {
            let lower = bound.lower.unwrap_or(f64::NEG_INFINITY).max(value - radius);
            let upper = bound.upper.unwrap_or(f64::INFINITY).min(value + radius);
            (
                i,
                Bound {
                    lower: lower.is_finite().then_some(lower),
                    upper: upper.is_finite().then_some(upper),
                },
            )
        })
        .collect();
    if restricted
        .iter()
        .any(|(_, b)| matches!((b.lower, b.upper), (Some(l), Some(u)) if l > u))
    {
        return Ok(None);
    }
    master.update(LpUpdate {
        variable_bounds: restricted,
        ..Default::default()
    })?;
    let result = solve_lp(master, solver, tolerance);
    master.update(LpUpdate {
        variable_bounds: original,
        ..Default::default()
    })?;
    let report = result?;
    Ok(report
        .dual
        .is_some()
        .then_some(report.solution.variable_values))
}

fn solve_scenarios(
    workspaces: &mut [LpWorkspace],
    scenarios: &[BendersScenario],
    x: &[f64],
    solver: &RevisedSimplexSolver,
    options: BendersOptions,
) -> Result<Vec<LpSolveReport>, OptimizeError> {
    let solve_chunk = |workspaces: &mut [LpWorkspace], scenarios: &[BendersScenario]| {
        workspaces
            .iter_mut()
            .zip(scenarios)
            .map(|(workspace, scenario)| {
                let shifts = scenario.coupling.spmv(x)?;
                if shifts.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("non-finite Benders RHS shift"));
                }
                let row_bounds = scenario
                    .recourse
                    .row_bounds
                    .iter()
                    .zip(shifts)
                    .enumerate()
                    .map(|(i, (b, shift))| {
                        (
                            i,
                            Bound {
                                lower: b.lower.map(|l| l - shift),
                                upper: b.upper.map(|u| u - shift),
                            },
                        )
                    })
                    .collect();
                workspace.update(LpUpdate {
                    row_bounds,
                    ..Default::default()
                })?;
                solve_lp(workspace, solver, options.evidence_tolerance)
            })
            .collect::<Result<Vec<_>, OptimizeError>>()
    };
    let workers = options.scenario_threads.min(scenarios.len());
    if workers == 1 {
        return solve_chunk(workspaces, scenarios);
    }
    let size = scenarios.len().div_ceil(workers);
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (workspaces, scenarios) in workspaces.chunks_mut(size).zip(scenarios.chunks(size)) {
            handles.push(
                thread::Builder::new()
                    .spawn_scoped(scope, move || solve_chunk(workspaces, scenarios))
                    .map_err(|_| invalid("could not start Benders scenario worker"))?,
            );
        }
        let mut reports = Vec::new();
        for handle in handles {
            reports.extend(
                handle
                    .join()
                    .map_err(|_| invalid("Benders scenario worker failed"))??,
            );
        }
        Ok(reports)
    })
}

fn solve_lp(
    workspace: &mut LpWorkspace,
    solver: &RevisedSimplexSolver,
    tolerance: f64,
) -> Result<LpSolveReport, OptimizeError> {
    let report = workspace.solve(solver, tolerance)?;
    if report.used_warm_start
        && report.solution.status != SolveStatus::IterationLimit
        && report.dual.is_none()
        && report.infeasibility.is_none()
        && report.unbounded_ray.is_none()
    {
        let mut cold = LpWorkspace::new(workspace.problem().clone(), report.subproblem)?;
        let result = cold.solve(solver, tolerance)?;
        *workspace = cold;
        return Ok(result);
    }
    Ok(report)
}

struct Affine {
    constant: f64,
    slope: Vec<f64>,
}

impl Affine {
    fn weighted_sum<'a>(
        columns: usize,
        values: impl Iterator<Item = (&'a Affine, &'a f64)>,
    ) -> Result<Self, OptimizeError> {
        let mut sum = Self {
            constant: 0.0,
            slope: vec![0.0; columns],
        };
        for (value, &weight) in values {
            sum.constant += weight * value.constant;
            for (sum, &slope) in sum.slope.iter_mut().zip(&value.slope) {
                *sum += weight * slope;
            }
        }
        if !sum.constant.is_finite() || sum.slope.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite weighted recourse cut"));
        }
        Ok(sum)
    }

    fn cut(
        self,
        source: LpSubproblemID,
        iteration: usize,
        kind: CutKind,
        theta: Option<usize>,
    ) -> DecompositionCut {
        let mut entries: Vec<_> = self
            .slope
            .into_iter()
            .enumerate()
            .filter(|(_, a)| *a != 0.0)
            .collect();
        if let Some(theta) = theta {
            entries.push((theta, -1.0));
        }
        DecompositionCut {
            entries,
            upper: -self.constant,
            origin: CutOrigin {
                source,
                iteration,
                kind,
            },
            scope: CutScope::Global,
        }
    }
}

fn affine(s: &BendersScenario, row: &[f64], column: &[f64]) -> Result<Affine, OptimizeError> {
    let constant = box_minimum(row, &s.recourse.row_bounds)
        .zip(box_minimum(column, &s.recourse.var_bounds))
        .map(|(row, col)| row + col)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid("recourse certificate has no finite cut intercept"))?;
    let slope: Vec<_> = s
        .coupling
        .transpose_spmv(row)?
        .into_iter()
        .map(|v| -v)
        .collect();
    if slope.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite recourse cut slope"));
    }
    Ok(Affine { constant, slope })
}

fn box_minimum(coefficients: &[f64], bounds: &[Bound]) -> Option<f64> {
    let mut minimum = 0.0;
    for (&coefficient, bound) in coefficients.iter().zip(bounds) {
        if coefficient > 0.0 {
            minimum += coefficient * bound.lower?;
        } else if coefficient < 0.0 {
            minimum += coefficient * bound.upper?;
        }
    }
    minimum.is_finite().then_some(minimum)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn violation(cut: &DecompositionCut, x: &[f64]) -> Result<f64, OptimizeError> {
    let activity: f64 = cut.entries.iter().map(|&(i, value)| value * x[i]).sum();
    let excess = activity - cut.upper;
    if !excess.is_finite() {
        return Err(invalid("non-finite Benders cut activity"));
    }
    Ok(excess)
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
