//! Projected Benders cuts from convex nonlinear recourse outer approximations.

use std::collections::BTreeSet;

use nc_sparse::CooMatrix;

use crate::{
    analyze_expression,
    benders::{solve_lp, violation},
    benders_tree::{search, Candidate, TreeOptions, TreeStop},
    lp_decomposition::accumulate_support,
    Bound, CutInsertion, CutKind, CutOrigin, CutScope, DecompositionCut, DecompositionCutPool,
    DifferentiableValue, LpColumn, LpRow, LpSubproblemID, LpUpdate, LpWorkspace,
    NonlinearConstraint, NonlinearExpression, NonlinearNode, OptimizeError, Problem,
    RevisedSimplexSolver, SolveStatus,
};

/// Objective and constraints use parameters in shared-column order followed
/// by private-variable order. Recourse variables are continuous.
#[derive(Debug, Clone)]
pub struct NonlinearRecourseBlock {
    pub id: String,
    pub shared_columns: Vec<usize>,
    pub local_bounds: Vec<Bound>,
    pub weight: f64,
    pub objective: NonlinearExpression,
    pub constraints: Vec<NonlinearConstraint>,
}

#[derive(Debug, Clone)]
pub struct NonlinearDecompositionProblem {
    pub master: Problem,
    pub blocks: Vec<NonlinearRecourseBlock>,
}

#[derive(Debug, Clone, Copy)]
pub struct NonlinearDecompositionOptions {
    pub max_master_solves: usize,
    pub max_block_solves: usize,
    pub lp_iteration_limit: usize,
    pub feasibility_tolerance: f64,
    pub absolute_gap: f64,
    pub relative_gap: f64,
}

impl Default for NonlinearDecompositionOptions {
    fn default() -> Self {
        Self {
            max_master_solves: 1000,
            max_block_solves: 100,
            lp_iteration_limit: 10_000,
            feasibility_tolerance: 1e-8,
            absolute_gap: 1e-6,
            relative_gap: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearDecompositionTermination {
    Converged,
    Infeasible,
    MasterLimit,
    RecourseLimit,
    LpIterationLimit,
    MissingEvidence,
    EvaluationFailure,
    Stalled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearRecourseSolution {
    pub id: String,
    pub local_point: Vec<f64>,
    pub objective: f64,
    pub maximum_violation: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearDecompositionIncumbent {
    pub master_point: Vec<f64>,
    pub recourse: Vec<NonlinearRecourseSolution>,
    pub objective: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearDecompositionResult {
    pub termination: NonlinearDecompositionTermination,
    pub incumbent: Option<NonlinearDecompositionIncumbent>,
    pub lower_bound: Option<f64>,
    pub master_solves: usize,
    pub recourse_solves: usize,
    pub feasibility_cuts: usize,
    pub optimality_cuts: usize,
    pub blocking_block: Option<String>,
}

pub fn solve_nonlinear_decomposition(
    problem: &NonlinearDecompositionProblem,
    options: NonlinearDecompositionOptions,
) -> Result<NonlinearDecompositionResult, OptimizeError> {
    use NonlinearDecompositionTermination as Term;
    if options.max_master_solves == 0
        || options.max_block_solves == 0
        || options.lp_iteration_limit == 0
        || !(1e-12..=1e-3).contains(&options.feasibility_tolerance)
        || !options.absolute_gap.is_finite()
        || options.absolute_gap < 0.0
        || !options.relative_gap.is_finite()
        || options.relative_gap < 0.0
        || options.absolute_gap.max(options.relative_gap) == 0.0
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid nonlinear decomposition options",
        ));
    }
    problem.master.validate()?;
    let n = problem.master.objective.len();
    if n == 0 || !problem.master.var_bounds.iter().all(finite_box) || problem.blocks.is_empty() {
        return Err(invalid(
            "nonlinear decomposition requires a bounded master and at least one recourse block",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut blocks = problem
        .blocks
        .iter()
        .map(|block| {
            if block.id.trim().is_empty()
                || !ids.insert(&block.id)
                || !block.weight.is_finite()
                || block.weight <= 0.0
            {
                return Err(invalid(
                    "recourse names must be distinct and weights must be positive and finite",
                ));
            }
            Block::new(block, &problem.master.var_bounds)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total_weight: f64 = problem.blocks.iter().map(|b| b.weight).sum();
    if !total_weight.is_finite() {
        return Err(invalid("non-finite total recourse weight"));
    }
    let mut relaxation = problem.master.clone();
    relaxation.is_integer.fill(false);
    let mut master = LpWorkspace::new(relaxation, None)?;
    master.update(LpUpdate {
        append_columns: blocks
            .iter()
            .map(|block| LpColumn {
                entries: vec![],
                bound: block.epigraph_bound(),
                objective: block.model.weight,
            })
            .collect(),
        ..Default::default()
    })?;
    let tree_options = TreeOptions {
        solve_limit: options.max_master_solves,
        candidate_limit: options.max_master_solves,
        lp_iteration_limit: options.lp_iteration_limit,
        simplex_tolerance: options.feasibility_tolerance.min(1e-9),
        evidence_tolerance: options.feasibility_tolerance,
        integrality_tolerance: options.feasibility_tolerance,
        absolute_gap: options.absolute_gap,
        relative_gap: options.relative_gap,
    };
    let solver = RevisedSimplexSolver {
        max_iterations: options.lp_iteration_limit,
        tolerance: tree_options.simplex_tolerance,
    };
    let mut pool = DecompositionCutPool::default();
    let mut candidate_number = 0;
    let (mut feasibility_cuts, mut optimality_cuts) = (0, 0);
    let mut blocked = None;
    let mut master = master.problem().clone();
    master.is_integer[..n].copy_from_slice(&problem.master.is_integer);
    let result = search(master, &tree_options, |point, master_objective| {
        candidate_number += 1;
        let mut rows = Vec::new();
        let mut recourse = Vec::with_capacity(blocks.len());
        let mut objective: f64 = problem
            .master
            .objective
            .iter()
            .zip(point)
            .map(|(c, x)| c * x)
            .sum();
        for (i, block) in blocks.iter_mut().enumerate() {
            let theta = n + i;
            let inner_gap = options
                .absolute_gap
                .max(options.relative_gap * master_objective.abs().max(1.0))
                * 0.1
                / total_weight;
            let reply = block.solve(point, &solver, options, inner_gap)?;
            let (support, kind, column) = match reply {
                Reply::Feasible(solution, support) => {
                    objective += block.model.weight * solution.objective;
                    recourse.push(solution);
                    (support, CutKind::Optimality, Some(theta))
                }
                Reply::Infeasible(support) => (support, CutKind::Feasibility, None),
                Reply::Stopped(status) => {
                    blocked = Some((status, block.model.id.clone()));
                    // Keep global cuts derived by earlier blocks.
                    return Ok(if rows.is_empty() {
                        Candidate::Stop
                    } else {
                        Candidate::Evaluated {
                            incumbent: None,
                            rows,
                        }
                    });
                }
            };
            let cut = support.cut(block.model, candidate_number, kind, column);
            if violation(&cut, point)? > 0.0
                && matches!(pool.insert(cut.clone())?, CutInsertion::Inserted(_))
            {
                rows.push(cut.lp_row());
                match kind {
                    CutKind::Feasibility => feasibility_cuts += 1,
                    CutKind::Optimality => optimality_cuts += 1,
                }
            }
            if kind == CutKind::Feasibility {
                break;
            }
        }
        blocked = None;
        let incumbent = if recourse.len() == blocks.len() {
            if !objective.is_finite() {
                return Err(invalid("non-finite incumbent objective"));
            }
            let candidate = NonlinearDecompositionIncumbent {
                master_point: point[..n].to_vec(),
                recourse,
                objective,
            };
            Some((objective, candidate))
        } else {
            None
        };
        Ok(Candidate::Evaluated { incumbent, rows })
    })?;
    let termination = match result.stop {
        TreeStop::Converged => Term::Converged,
        TreeStop::Infeasible => Term::Infeasible,
        TreeStop::NodeLimit | TreeStop::CandidateLimit => Term::MasterLimit,
        TreeStop::LpIterationLimit => Term::LpIterationLimit,
        TreeStop::EvaluationStopped => blocked
            .as_ref()
            .map_or(Term::MissingEvidence, |(status, _)| *status),
        TreeStop::Stalled => Term::Stalled,
        _ => Term::MissingEvidence,
    };
    Ok(NonlinearDecompositionResult {
        termination,
        incumbent: result.incumbent.map(|(_, candidate)| candidate),
        lower_bound: result.lower,
        master_solves: result.solves,
        recourse_solves: blocks.iter().map(|b| b.solves).sum(),
        feasibility_cuts,
        optimality_cuts,
        blocking_block: if result.stop == TreeStop::EvaluationStopped {
            blocked.map(|(_, id)| id)
        } else {
            None
        },
    })
}

struct Block<'a> {
    model: &'a NonlinearRecourseBlock,
    bounds: Vec<Bound>,
    lp: LpWorkspace,
    solves: usize,
}

enum Reply {
    Feasible(NonlinearRecourseSolution, Support),
    Infeasible(Support),
    Stopped(NonlinearDecompositionTermination),
}

impl<'a> Block<'a> {
    fn new(
        model: &'a NonlinearRecourseBlock,
        master_bounds: &[Bound],
    ) -> Result<Self, OptimizeError> {
        let mut seen = BTreeSet::new();
        let mut bounds = Vec::new();
        for &column in &model.shared_columns {
            if column >= master_bounds.len() || !seen.insert(column) {
                return Err(invalid(
                    "recourse shared columns must be distinct valid master indices",
                ));
            }
            bounds.push(master_bounds[column]);
        }
        bounds.extend(&model.local_bounds);
        if !bounds.iter().all(finite_box) {
            return Err(invalid("recourse requires finite box bounds"));
        }
        let objective = check_expression(&model.objective, &bounds)?;
        if !objective.convexity.is_convex() {
            return Err(invalid(
                "recourse objective curvature must be convex on the full box",
            ));
        }
        for constraint in &model.constraints {
            let bound = constraint.bound;
            let c = check_expression(&constraint.expression, &bounds)?;
            if (bound.lower.is_none() && bound.upper.is_none())
                || bound
                    .lower
                    .is_some_and(|l| !l.is_finite() || !c.convexity.is_concave())
                || bound
                    .upper
                    .is_some_and(|u| !u.is_finite() || !c.convexity.is_convex())
                || matches!((bound.lower, bound.upper), (Some(l), Some(u)) if l > u)
            {
                return Err(invalid(
                    "recourse constraints require convex upper and concave lower sides",
                ));
            }
        }
        let m = model.shared_columns.len();
        let n = bounds.len();
        let mut matrix = CooMatrix::new(m, n + 1);
        for i in 0..m {
            matrix.push(i, i, 1.0)?;
        }
        let middle: Vec<_> = bounds
            .iter()
            .map(|b| b.lower.unwrap().midpoint(b.upper.unwrap()))
            .collect();
        let evaluation = model.objective.evaluate(&middle)?;
        let tangent = tangent(&evaluation, &middle, &bounds, 1.0, 0.0)?;
        let floor = tangent
            .entries
            .iter()
            .fold(-tangent.bound.upper.unwrap(), |floor, &(i, g)| {
                let b = bounds[i];
                floor
                    + g * if g > 0.0 {
                        b.lower.unwrap()
                    } else {
                        b.upper.unwrap()
                    }
            });
        let mut var_bounds = bounds.clone();
        var_bounds.push(Bound {
            lower: Some(floor),
            upper: None,
        });
        let mut objective_linear = vec![0.0; n + 1];
        objective_linear[n] = 1.0;
        let problem = Problem {
            objective: objective_linear,
            constraints: matrix.to_csr()?,
            row_bounds: middle[..m].iter().copied().map(Bound::fixed).collect(),
            var_bounds,
            is_integer: vec![false; n + 1],
        };
        let lp = LpWorkspace::new(problem, None)?;
        let mut block = Self {
            model,
            bounds,
            lp,
            solves: 0,
        };
        let (_, _, rows) = block.linearize(&middle, None, 0.0)?;
        block.lp.update(LpUpdate {
            append_rows: rows,
            ..Default::default()
        })?;
        Ok(block)
    }

    fn epigraph_bound(&self) -> Bound {
        *self.lp.problem().var_bounds.last().unwrap()
    }

    fn solve(
        &mut self,
        point: &[f64],
        solver: &RevisedSimplexSolver,
        options: NonlinearDecompositionOptions,
        inner_gap: f64,
    ) -> Result<Reply, OptimizeError> {
        use NonlinearDecompositionTermination as Term;
        let fixed: Vec<_> = self
            .model
            .shared_columns
            .iter()
            .map(|&i| point[i])
            .collect();
        let m = fixed.len();
        let n = self.bounds.len();
        if fixed
            .iter()
            .zip(&self.bounds)
            .any(|(&x, b)| x < b.lower.unwrap() || x > b.upper.unwrap())
        {
            return Ok(Reply::Stopped(Term::MissingEvidence));
        }
        self.lp.update(LpUpdate {
            row_bounds: fixed
                .iter()
                .copied()
                .map(Bound::fixed)
                .enumerate()
                .collect(),
            ..Default::default()
        })?;
        let mut best: Option<NonlinearRecourseSolution> = None;
        for _ in 0..options.max_block_solves {
            self.solves += 1;
            let report = solve_lp(&mut self.lp, solver, options.feasibility_tolerance)?;
            if let Some(farkas) = report.infeasibility {
                return Ok(self
                    .support(&farkas.row_multipliers, &farkas.variable_multipliers)
                    .map_or(Reply::Stopped(Term::MissingEvidence), Reply::Infeasible));
            }
            let Some(dual) = report.dual else {
                return Ok(Reply::Stopped(match report.solution.status {
                    SolveStatus::IterationLimit => Term::LpIterationLimit,
                    _ => Term::MissingEvidence,
                }));
            };
            let mut x = report.solution.variable_values[..n].to_vec();
            x[..m].copy_from_slice(&fixed);
            for (value, bound) in x[m..].iter_mut().zip(&self.bounds[m..]) {
                *value = value.clamp(bound.lower.unwrap(), bound.upper.unwrap());
            }
            let epigraph = report.solution.variable_values[n];
            let Ok((value, maximum_violation, rows)) =
                self.linearize(&x, Some(epigraph), options.feasibility_tolerance)
            else {
                return Ok(Reply::Stopped(Term::EvaluationFailure));
            };
            if maximum_violation <= options.feasibility_tolerance
                && best.as_ref().is_none_or(|p| value < p.objective)
            {
                best = Some(NonlinearRecourseSolution {
                    id: self.model.id.clone(),
                    local_point: x[m..].to_vec(),
                    objective: value,
                    maximum_violation,
                });
            }
            let Some(support) = self.support(&dual.row_multipliers, &dual.reduced_costs) else {
                return Ok(Reply::Stopped(Term::MissingEvidence));
            };
            let lower = support.at(&fixed);
            if let Some(solution) = best.as_ref().filter(|p| p.objective - lower <= inner_gap) {
                return Ok(Reply::Feasible(solution.clone(), support));
            }
            if rows.is_empty()
                || rows
                    .iter()
                    .all(|r| !self.excludes(r, &report.solution.variable_values, solver.tolerance))
            {
                return Ok(Reply::Stopped(Term::Stalled));
            }
            self.lp.update(LpUpdate {
                append_rows: rows,
                ..Default::default()
            })?;
        }
        Ok(Reply::Stopped(Term::RecourseLimit))
    }

    fn excludes(&self, row: &LpRow, x: &[f64], tolerance: f64) -> bool {
        let activity: f64 = row.entries.iter().map(|&(i, a)| a * x[i]).sum();
        activity - row.bound.upper.unwrap() > tolerance
    }

    fn linearize(
        &self,
        x: &[f64],
        epigraph: Option<f64>,
        tolerance: f64,
    ) -> Result<(f64, f64, Vec<LpRow>), OptimizeError> {
        let mut rows = Vec::new();
        let f = self.model.objective.evaluate(x)?;
        if epigraph.is_none_or(|t| f.value > t) {
            let mut row = tangent(&f, x, &self.bounds, 1.0, 0.0)?;
            row.entries.push((x.len(), -1.0));
            rows.push(row);
        }
        let mut maximum_violation = 0.0_f64;
        for constraint in &self.model.constraints {
            let c = constraint.expression.evaluate(x)?;
            for (sign, limit) in [
                (1.0, constraint.bound.upper),
                (-1.0, constraint.bound.lower),
            ] {
                if let Some(limit) = limit {
                    let excess = sign * (c.value - limit);
                    maximum_violation = maximum_violation.max(excess);
                    if epigraph.is_none() || excess > tolerance {
                        rows.push(tangent(&c, x, &self.bounds, sign, limit)?);
                    }
                }
            }
        }
        Ok((f.value, maximum_violation, rows))
    }

    fn support(&self, row: &[f64], column: &[f64]) -> Option<Support> {
        let m = self.model.shared_columns.len();
        let mut constant = 0.0;
        let mut scale = 1.0;
        // The first rows are z = x. Their verified multipliers multiply
        // the shared parameters. The remaining support terms are fixed.
        accumulate_support(
            &mut constant,
            &mut scale,
            &row[m..],
            &self.lp.problem().row_bounds[m..],
        )?;
        accumulate_support(
            &mut constant,
            &mut scale,
            column,
            &self.lp.problem().var_bounds,
        )?;
        for (&a, b) in row[..m].iter().zip(&self.bounds) {
            scale += a.abs() * radius(b);
        }
        let lp = self.lp.problem();
        for (r, c, a) in lp.constraints.iter_entries() {
            // Each entry participates in Aᵀy before reduced costs are formed.
            // Include its magnitude even when those products cancel.
            if let Some(b) = self.bounds.get(c) {
                scale += row[r].abs() * a.abs() * radius(b);
            }
        }
        let constant = constant - roundoff(scale, row.len() + column.len());
        constant.is_finite().then(|| Support {
            constant,
            slope: row[..m].to_vec(),
        })
    }
}

struct Support {
    constant: f64,
    slope: Vec<f64>,
}

impl Support {
    fn at(&self, point: &[f64]) -> f64 {
        self.constant
            + self
                .slope
                .iter()
                .zip(point)
                .map(|(a, x)| a * x)
                .sum::<f64>()
    }

    fn cut(
        self,
        model: &NonlinearRecourseBlock,
        iteration: usize,
        kind: CutKind,
        theta: Option<usize>,
    ) -> DecompositionCut {
        let mut entries: Vec<_> = model
            .shared_columns
            .iter()
            .copied()
            .zip(self.slope)
            .filter(|(_, a)| *a != 0.0)
            .collect();
        if let Some(theta) = theta {
            entries.push((theta, -1.0));
        }
        DecompositionCut {
            entries,
            upper: -self.constant,
            origin: CutOrigin {
                source: LpSubproblemID {
                    name: model.id.clone(),
                    scenario: None,
                },
                iteration,
                kind,
            },
            scope: CutScope::Global,
        }
    }
}

fn tangent(
    value: &DifferentiableValue,
    at: &[f64],
    bounds: &[Bound],
    sign: f64,
    limit: f64,
) -> Result<LpRow, OptimizeError> {
    let mut upper = sign * (limit - value.value);
    let mut scale = 1.0 + limit.abs() + value.value.abs();
    let mut entries = Vec::new();
    for (i, ((&g, &x), b)) in value.gradient.iter().zip(at).zip(bounds).enumerate() {
        if g != 0.0 {
            upper += sign * g * x;
            scale += g.abs() * radius(b);
            entries.push((i, sign * g));
        }
    }
    upper += roundoff(scale, at.len());
    if !upper.is_finite() {
        return Err(invalid("non-finite nonlinear tangent"));
    }
    Ok(LpRow {
        entries,
        bound: Bound {
            lower: None,
            upper: Some(upper),
        },
    })
}

fn roundoff(scale: f64, terms: usize) -> f64 {
    16.0 * f64::EPSILON * (terms as f64 + 1.0) * scale
}

fn radius(b: &Bound) -> f64 {
    b.lower.unwrap().abs().max(b.upper.unwrap().abs())
}

fn finite_box(b: &Bound) -> bool {
    matches!((b.lower, b.upper), (Some(l), Some(u)) if l <= u && (u - l).is_finite())
}

fn check_expression(
    expression: &NonlinearExpression,
    bounds: &[Bound],
) -> Result<crate::NodeAnalysis, OptimizeError> {
    let analysis = analyze_expression(expression, bounds)?;
    for (node, item) in expression.nodes.iter().zip(&analysis.nodes) {
        let positive = |i: usize| analysis.nodes[i].interval.lower > 0.0;
        let smooth = match *node {
            NonlinearNode::Sqrt(i) | NonlinearNode::Log(i) => positive(i),
            NonlinearNode::Powf(i, p) if p.fract() != 0.0 || p <= 0.0 => positive(i),
            _ => true,
        };
        if !smooth || !item.interval.is_finite() || item.interval.lower > item.interval.upper {
            return Err(invalid(
                "recourse expression must have finite smooth values on the full box",
            ));
        }
    }
    Ok(analysis.output_analysis().clone())
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
