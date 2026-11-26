//! Continuous LP reoptimization and original-coordinate numerical evidence.

use crate::{
    simplex::Workspace, Bound, OptimizeError, Problem, RevisedSimplexSolver, Solution, SolveStatus,
};
use nc_sparse::CooMatrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpBasisStatus {
    Basic,
    AtLower,
    AtUpper,
    Free,
}

/// One status per original column and per row slack, in model order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LpBasis {
    pub variables: Vec<LpBasisStatus>,
    pub rows: Vec<LpBasisStatus>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpDualSolution {
    /// For minimization, positive multipliers select a row lower bound and
    /// negative multipliers select its upper bound.
    pub row_multipliers: Vec<f64>,
    /// c − Aᵀy. The same sign convention applies to variable bounds.
    pub reduced_costs: Vec<f64>,
    pub objective_bound: f64,
    pub absolute_gap: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpFarkasCertificate {
    /// Normalized to unit infinity norm. The variable coefficients are −Aᵀy.
    pub row_multipliers: Vec<f64>,
    pub variable_multipliers: Vec<f64>,
    /// Minimum of yᵀs − yᵀAx over the row and variable boxes.
    pub contradiction_margin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpUnboundedRay {
    pub direction: Vec<f64>,
    pub objective_slope: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LpSubproblemID {
    pub name: String,
    pub scenario: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LpSolveReport {
    pub solution: Solution,
    pub basis: LpBasis,
    pub dual: Option<LpDualSolution>,
    pub infeasibility: Option<LpFarkasCertificate>,
    pub unbounded_ray: Option<LpUnboundedRay>,
    pub subproblem: Option<LpSubproblemID>,
    pub used_warm_start: bool,
    /// Total pricing iterations across both simplex phases.
    pub iterations: usize,
    pub pivots: usize,
    pub maximum_basic_reduced_cost: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LpRow {
    pub entries: Vec<(usize, f64)>,
    pub bound: Bound,
}

#[derive(Debug, Clone)]
pub struct LpColumn {
    pub entries: Vec<(usize, f64)>,
    pub bound: Bound,
    pub objective: f64,
}

/// Existing coefficients and bounds are edited by index, with the last edit
/// at any repeated index taking precedence. New rows address existing columns.
/// New columns address existing and newly appended rows.
#[derive(Debug, Clone, Default)]
pub struct LpUpdate {
    pub objective: Vec<(usize, f64)>,
    pub variable_bounds: Vec<(usize, Bound)>,
    pub row_bounds: Vec<(usize, Bound)>,
    pub append_rows: Vec<LpRow>,
    pub append_columns: Vec<LpColumn>,
}

/// Retains a simplex workspace for an unscaled, unpresolved continuous LP.
/// Numerical evidence refers to the exact egress model (after edits).
pub struct LpWorkspace {
    problem: Problem,
    workspace: Option<Workspace>,
    subproblem: Option<LpSubproblemID>,
}

impl LpWorkspace {
    pub fn new(
        problem: Problem,
        subproblem: Option<LpSubproblemID>,
    ) -> Result<Self, OptimizeError> {
        problem.validate()?;
        if problem.is_integer.iter().any(|&integer| integer) {
            return Err(invalid("LP reoptimization requires continuous variables"));
        }
        Ok(Self {
            problem,
            workspace: None,
            subproblem,
        })
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Import a compatible basis and refactor it using this model's matrix.
    pub fn install_basis(&mut self, basis: &LpBasis) -> Result<(), OptimizeError> {
        let mut working = Workspace::new(&self.problem)?;
        working.install_basis(basis)?;
        self.workspace = Some(working);
        Ok(())
    }

    /// Apply a batch atomically. Invalid data leaves the input model intact.
    pub fn update(&mut self, edits: LpUpdate) -> Result<(), OptimizeError> {
        let n = self.problem.objective.len();
        let m = self.problem.row_bounds.len();
        let mut objective = self.problem.objective.clone();
        let mut variable_bounds = self.problem.var_bounds.clone();
        let mut row_bounds = self.problem.row_bounds.clone();
        apply_indexed(&mut objective, edits.objective)?;
        apply_indexed(&mut variable_bounds, edits.variable_bounds)?;
        apply_indexed(&mut row_bounds, edits.row_bounds)?;
        if objective.iter().any(|x| !x.is_finite())
            || !variable_bounds.iter().chain(&row_bounds).all(valid_bound)
        {
            return Err(invalid("non-finite objective or invalid LP bound"));
        }
        if edits.append_rows.is_empty() && edits.append_columns.is_empty() {
            self.problem.objective = objective;
            self.problem.var_bounds = variable_bounds;
            self.problem.row_bounds = row_bounds;
            // A failed refresh is retried cold when the new model is solved.
            if self
                .workspace
                .as_mut()
                .is_some_and(|ws| ws.refresh(&self.problem).is_err())
            {
                self.workspace = None;
            }
            return Ok(());
        }

        let rows = m + edits.append_rows.len();
        let columns = n + edits.append_columns.len();
        let mut matrix = CooMatrix::new(rows, columns);
        for (row, column, value) in self.problem.constraints.iter_entries() {
            matrix.push(row, column, value)?;
        }
        for (i, row) in edits.append_rows.into_iter().enumerate() {
            row_bounds.push(row.bound);
            for (column, value) in row.entries {
                if column >= n {
                    return Err(invalid("appended row must reference existing columns"));
                }
                matrix.push(m + i, column, value)?;
            }
        }
        for (j, column) in edits.append_columns.into_iter().enumerate() {
            variable_bounds.push(column.bound);
            objective.push(column.objective);
            for (row, value) in column.entries {
                matrix.push(row, n + j, value)?;
            }
        }
        let problem = Problem {
            constraints: matrix.to_csr()?,
            objective,
            row_bounds,
            var_bounds: variable_bounds,
            is_integer: vec![false; columns],
        };
        problem.validate()?;
        if let Some(previous) = &self.workspace {
            let mut basis = previous.basis(n);
            basis.rows.resize(rows, LpBasisStatus::Basic);
            basis.variables.resize(columns, LpBasisStatus::Free);
            let mut workspace = Workspace::new(&problem)?;
            self.workspace = workspace.install_basis(&basis).is_ok().then_some(workspace);
        }
        self.problem = problem;
        Ok(())
    }

    pub fn solve(
        &mut self,
        solver: &RevisedSimplexSolver,
        evidence_tolerance: f64,
    ) -> Result<LpSolveReport, OptimizeError> {
        self.solve_with_dual_repair(solver, evidence_tolerance, |_, _| None)
    }

    /// A caller can propose multipliers for a structured LP. Every proposal
    /// must pass the same original-model audit before it becomes evidence.
    pub(crate) fn solve_with_dual_repair(
        &mut self,
        solver: &RevisedSimplexSolver,
        evidence_tolerance: f64,
        repair: impl FnOnce(&Problem, &[f64]) -> Option<Vec<f64>>,
    ) -> Result<LpSolveReport, OptimizeError> {
        solver.validate()?;
        if !valid_tolerance(evidence_tolerance) {
            return Err(OptimizeError::InvalidConfiguration(
                "invalid LP evidence tolerance",
            ));
        }
        let used_warm_start = self.workspace.is_some();
        let mut workspace = match self.workspace.take() {
            Some(workspace) => workspace,
            None => Workspace::new(&self.problem)?,
        };
        let outcome = solver.run(&mut workspace)?;
        let n = self.problem.objective.len();
        let solution = workspace.solution(n, outcome.status, &self.problem.objective);
        if !solution.objective_value.is_finite()
            || solution
                .variable_values
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(invalid("non-finite simplex result"));
        }
        let mut report = LpSolveReport {
            solution,
            basis: workspace.basis(n),
            dual: None,
            infeasibility: None,
            unbounded_ray: None,
            subproblem: self.subproblem.clone(),
            used_warm_start,
            iterations: outcome.iterations,
            pivots: outcome.pivots,
            maximum_basic_reduced_cost: None,
        };
        match outcome.status {
            SolveStatus::Optimal => {
                let multipliers = workspace.row_multipliers(None);
                let residual = workspace.maximum_basic_reduced_cost(&multipliers);
                report.maximum_basic_reduced_cost = residual.is_finite().then_some(residual);
                let verify = |candidate: &[f64]| {
                    verify_lp_dual(
                        &self.problem,
                        &report.solution.variable_values,
                        candidate,
                        evidence_tolerance,
                    )
                };
                report.dual = verify(&multipliers)
                    .or_else(|| verify(&repair(&self.problem, &multipliers)?))
                    .or_else(|| verify(&adjust_upper_support(&self.problem, &multipliers)?));
            }
            SolveStatus::Infeasible => {
                report.infeasibility = verify_lp_farkas(
                    &self.problem,
                    &workspace.row_multipliers(Some(solver.tolerance)),
                    evidence_tolerance,
                );
            }
            SolveStatus::Unbounded => {
                if let Some(direction) = outcome.direction {
                    report.unbounded_ray = verify_lp_unbounded_ray(
                        &self.problem,
                        &report.solution.variable_values,
                        &direction,
                        evidence_tolerance,
                    );
                }
            }
            SolveStatus::IterationLimit => {}
        }
        self.workspace = Some(workspace);
        Ok(report)
    }
}

/// Independently evaluate weak duality and the primal-dual gap.
/// Missing bounds for nonzero reduced costs or multipliers invalidate evidence.
pub fn verify_lp_dual(
    problem: &Problem,
    point: &[f64],
    multipliers: &[f64],
    tolerance: f64,
) -> Option<LpDualSolution> {
    if !valid_tolerance(tolerance) || !feasible(problem, point, tolerance) {
        return None;
    }
    let at_y = problem.constraints.transpose_spmv(multipliers).ok()?;
    let reduced: Vec<_> = problem
        .objective
        .iter()
        .zip(at_y)
        .map(|(c, a)| c - a)
        .collect();
    let mut bound = 0.0;
    let mut scale = 1.0;
    accumulate_support(&mut bound, &mut scale, multipliers, &problem.row_bounds)?;
    accumulate_support(&mut bound, &mut scale, &reduced, &problem.var_bounds)?;
    let primal: f64 = problem
        .objective
        .iter()
        .zip(point)
        .map(|(c, x)| c * x)
        .sum();
    let gap = (primal - bound).abs();
    if !primal.is_finite() || !gap.is_finite() || gap / scale.max(primal.abs()) > tolerance {
        return None;
    }
    Some(LpDualSolution {
        row_multipliers: multipliers.to_vec(),
        reduced_costs: reduced,
        objective_bound: bound,
        absolute_gap: gap,
    })
}

// A small negative reduced cost on a one-sided epigraph leaves its support
// unbounded. Scale the witness before re-auditing it against the original LP.
fn adjust_upper_support(problem: &Problem, multipliers: &[f64]) -> Option<Vec<f64>> {
    let roundoff = 64.0 * f64::EPSILON;
    let mut scale = 1.0_f64;
    let at_y = problem.constraints.transpose_spmv(multipliers).ok()?;
    for ((&c, a), bound) in problem.objective.iter().zip(at_y).zip(&problem.var_bounds) {
        if bound.upper.is_none() && a > c {
            if bound.lower.is_none() || c <= 0.0 || (a - c) > roundoff * c {
                return None;
            }
            scale = scale.min((c / a) * (1.0 - roundoff));
        }
    }
    (scale < 1.0).then(|| multipliers.iter().map(|y| scale * y).collect())
}

/// Verify a separating linear functional against the original row/variable boxes.
pub fn verify_lp_farkas(
    problem: &Problem,
    multipliers: &[f64],
    tolerance: f64,
) -> Option<LpFarkasCertificate> {
    if !valid_tolerance(tolerance) || !continuous_model(problem) {
        return None;
    }
    let row_multipliers = normalize(multipliers)?;
    let mut column = problem.constraints.transpose_spmv(&row_multipliers).ok()?;
    column.iter_mut().for_each(|value| *value = -*value);
    let mut margin = 0.0;
    let mut scale = 1.0;
    accumulate_support(
        &mut margin,
        &mut scale,
        &row_multipliers,
        &problem.row_bounds,
    )?;
    accumulate_support(&mut margin, &mut scale, &column, &problem.var_bounds)?;
    (margin / scale > tolerance).then_some(LpFarkasCertificate {
        row_multipliers,
        variable_multipliers: column,
        contradiction_margin: margin,
    })
}

/// Require a feasible anchor and a strictly descending recession direction.
/// The anchor and recession signs are checked without a bound tolerance:
/// an infeasible anchor or even small outward motion invalidates a ray.
pub fn verify_lp_unbounded_ray(
    problem: &Problem,
    anchor: &[f64],
    direction: &[f64],
    tolerance: f64,
) -> Option<LpUnboundedRay> {
    if !valid_tolerance(tolerance) || !feasible(problem, anchor, 0.0) {
        return None;
    }
    let direction = normalize(direction)?;
    let activity = problem.constraints.spmv(&direction).ok()?;
    if !direction
        .iter()
        .zip(&problem.var_bounds)
        .chain(activity.iter().zip(&problem.row_bounds))
        .all(|(&v, b)| {
            v.is_finite() && (b.lower.is_none() || v >= 0.0) && (b.upper.is_none() || v <= 0.0)
        })
    {
        return None;
    }
    let slope: f64 = problem
        .objective
        .iter()
        .zip(&direction)
        .map(|(c, d)| c * d)
        .sum();
    (slope.is_finite() && slope < -tolerance).then_some(LpUnboundedRay {
        direction,
        objective_slope: slope,
    })
}

pub(crate) fn accumulate_support(
    sum: &mut f64,
    scale: &mut f64,
    coefficients: &[f64],
    bounds: &[Bound],
) -> Option<()> {
    if coefficients.len() != bounds.len() {
        return None;
    }
    for (&coefficient, bound) in coefficients.iter().zip(bounds) {
        if !coefficient.is_finite() {
            return None;
        }
        let term = if coefficient > 0.0 {
            coefficient * bound.lower?
        } else if coefficient < 0.0 {
            coefficient * bound.upper?
        } else {
            0.0
        };
        *sum += term;
        *scale += term.abs();
    }
    (sum.is_finite() && scale.is_finite()).then_some(())
}

fn feasible(problem: &Problem, point: &[f64], tolerance: f64) -> bool {
    if !continuous_model(problem) {
        return false;
    }
    let Ok(activity) = problem.constraints.spmv(point) else {
        return false;
    };
    point
        .iter()
        .zip(&problem.var_bounds)
        .chain(activity.iter().zip(&problem.row_bounds))
        .all(|(&v, b)| {
            v.is_finite()
                && b.lower.is_none_or(|lower| lower - v <= tolerance)
                && b.upper.is_none_or(|upper| v - upper <= tolerance)
        })
}

fn continuous_model(problem: &Problem) -> bool {
    problem.validate().is_ok() && !problem.is_integer.iter().any(|&integer| integer)
}

fn normalize(values: &[f64]) -> Option<Vec<f64>> {
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let maximum = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    (maximum > 0.0).then(|| values.iter().map(|value| value / maximum).collect())
}

fn valid_tolerance(tolerance: f64) -> bool {
    tolerance.is_finite() && tolerance >= 0.0
}

fn valid_bound(bound: &Bound) -> bool {
    bound.lower.is_none_or(f64::is_finite)
        && bound.upper.is_none_or(f64::is_finite)
        && !matches!((bound.lower, bound.upper), (Some(lo), Some(hi)) if lo > hi)
}

fn apply_indexed<T>(items: &mut [T], edits: Vec<(usize, T)>) -> Result<(), OptimizeError> {
    for (index, value) in edits {
        *items
            .get_mut(index)
            .ok_or_else(|| invalid("LP update index out of range"))? = value;
    }
    Ok(())
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
