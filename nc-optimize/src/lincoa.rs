//! Powell's quadratic interpolation method for a linearly constrained objective.

use crate::{bound_violation, Bound, OptimizeError, Problem};
use basin::{
    Constrained, ConstraintSet, ConstraintSolveError, CostFunction, DenseMatrix, ExecutionLimit,
    Executor, LinearBlock, Termination,
};
use std::cell::RefCell;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LincoaOptions {
    pub max_iterations: u64,
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
    /// Absolute tolerance in the units of the original rows and variable bounds.
    pub feasibility_tolerance: f64,
}

impl Default for LincoaOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 1_000,
            initial_radius: 1.0,
            final_radius: 1e-7,
            feasibility_tolerance: 1e-8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LincoaTermination {
    /// The algorithm completed its final trust-region stage.
    RadiusReached,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
    AllVariablesFixed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LincoaResult {
    pub point: Vec<f64>,
    pub objective: f64,
    /// Maximum violation recomputed against the original linear rows and variable bounds.
    pub maximum_violation: f64,
    pub evaluations: u64,
    pub termination: LincoaTermination,
}

/// Minimize a callback objective over the continuous linear inequalities in
/// `problem`. Its linear objective coefficients are ignored after validation.
/// Fixed variables are eliminated. Nonconstant equality rows are unsupported.
///
/// The starting point must satisfy the supplied rows and bounds after replacing
/// fixed coordinates by their bound values. The callback receives full points,
/// including possibly infeasible interpolation samples, and must return finite
/// values. Only evaluations feasible to the requested tolerance are eligible
/// for the returned incumbent.
pub fn minimize_lincoa<F>(
    initial: &[f64],
    problem: &Problem,
    options: LincoaOptions,
    evaluate: F,
) -> Result<LincoaResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    problem.validate()?;
    if initial.is_empty()
        || initial.len() != problem.var_bounds.len()
        || initial.iter().any(|value| !value.is_finite())
        || options.max_iterations == 0
        || options.max_evaluations == 0
        || !options.initial_radius.is_finite()
        || !options.final_radius.is_finite()
        || options.final_radius <= 0.0
        || options.final_radius >= options.initial_radius
        || !options.feasibility_tolerance.is_finite()
        || options.feasibility_tolerance < 0.0
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid LINCOA dimensions, radii, tolerance, or budget",
        ));
    }
    if problem.is_integer.iter().any(|&integer| integer) {
        return Err(invalid("LINCOA requires continuous variables"));
    }

    let mut base = initial.to_vec();
    let mut free = Vec::new();
    let mut reduced_column = vec![None; base.len()];
    for (column, bound) in problem.var_bounds.iter().enumerate() {
        match bound.lower {
            Some(fixed) if bound.upper == Some(fixed) => base[column] = fixed,
            _ => {
                reduced_column[column] = Some(free.len());
                free.push(column);
            }
        }
    }
    let activities = problem.constraints.spmv(&base)?;
    if violation(problem, &base, &activities) != 0.0 {
        return Err(invalid("LINCOA requires a feasible initial point"));
    }

    let mut matrix = Vec::new();
    let mut rhs = Vec::new();
    let mut normal = vec![0.0; free.len()];
    let mut entries = problem.constraints.iter_entries().peekable();
    for (row, (&activity, &bound)) in activities.iter().zip(&problem.row_bounds).enumerate() {
        normal.fill(0.0);
        while entries.peek().is_some_and(|entry| entry.0 == row) {
            let (_, column, coefficient) = entries.next().unwrap();
            if let Some(index) = reduced_column[column] {
                normal[index] += coefficient;
            }
        }
        append_inequalities(&normal, activity, bound, &mut matrix, &mut rhs)?;
    }
    for (index, &column) in free.iter().enumerate() {
        let bound = problem.var_bounds[column];
        if bound == Bound::free() {
            continue;
        }
        normal.fill(0.0);
        normal[index] = 1.0;
        append_inequalities(&normal, base[column], bound, &mut matrix, &mut rhs)?;
    }

    let dimension = free.len();
    let mut oracle = Oracle {
        evaluate,
        problem,
        options,
        base,
        free,
        best: None,
        evaluations: 0,
    };
    let termination = if dimension == 0 {
        oracle.sample(&[]).map_err(|err| match err {
            EvaluationError::User(err) => err,
            EvaluationError::Budget => unreachable!("validated positive evaluation budget"),
        })?;
        LincoaTermination::AllVariablesFixed
    } else {
        let inequalities = LinearBlock::new(
            DenseMatrix::from_row_slice(rhs.len(), dimension, &matrix),
            rhs,
        );
        let problem = Constrained::new(
            OracleAdapter(RefCell::new(&mut oracle)),
            ConstraintSet::new().with_linear_inequalities(inequalities),
        );
        let solver = basin::Lincoa::new()
            .with_initial_radius(options.initial_radius)
            .with_final_radius(options.final_radius);
        match Executor::from_start(problem, solver, vec![0.0; dimension])
            .max_iter(options.max_iterations)
            .max_cost_evals(options.max_evaluations)
            .run()
        {
            Err(ConstraintSolveError::Evaluation(EvaluationError::User(err))) => return Err(err),
            Err(ConstraintSolveError::Evaluation(EvaluationError::Budget)) => {
                LincoaTermination::EvaluationLimit
            }
            Err(err) => return Err(OptimizeError::InvalidProblem(format!("LINCOA: {err}"))),
            Ok(run) => match run.report.termination {
                Termination::Converged(_) => LincoaTermination::RadiusReached,
                Termination::Limit(limits)
                    if limits
                        .iter()
                        .any(|limit| matches!(limit, ExecutionLimit::CostEvaluations { .. })) =>
                {
                    LincoaTermination::EvaluationLimit
                }
                Termination::Limit(_) => LincoaTermination::IterationLimit,
                _ => LincoaTermination::NumericalFailure,
            },
        }
    };

    let (point, objective, maximum_violation) = oracle
        .best
        .ok_or_else(|| invalid("LINCOA returned without a feasible objective evaluation"))?;
    Ok(LincoaResult {
        point,
        objective,
        maximum_violation,
        evaluations: oracle.evaluations,
        termination,
    })
}

/// Translate rows to displacements about the start and scale their coefficients
/// before Basin computes their Euclidean norms.
fn append_inequalities(
    normal: &[f64],
    activity: f64,
    bound: Bound,
    matrix: &mut Vec<f64>,
    rhs: &mut Vec<f64>,
) -> Result<(), OptimizeError> {
    if bound == Bound::free() {
        return Ok(());
    }
    if normal.iter().any(|value| !value.is_finite()) {
        return Err(invalid("LINCOA row coefficients overflow when combined"));
    }
    let scale = normal.iter().map(|a| a.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(()); // Constant rows were checked at the starting point.
    }
    if bound.lower.is_some() && bound.lower == bound.upper {
        return Err(invalid("LINCOA does not support nonconstant equality rows"));
    }
    for (sign, limit) in [(-1.0, bound.lower), (1.0, bound.upper)] {
        if let Some(limit) = limit {
            let slack = sign * (limit - activity);
            let scaled = if slack.is_finite() {
                slack / scale
            } else {
                sign * (limit / scale - activity / scale)
            };
            if !scaled.is_finite() || scaled < 0.0 {
                return Err(invalid(
                    "LINCOA row or bound exceeds the scaled finite range",
                ));
            }
            rhs.push(scaled);
            matrix.extend(normal.iter().map(|&a| sign * (a / scale)));
        }
    }
    Ok(())
}

fn violation(problem: &Problem, point: &[f64], activities: &[f64]) -> f64 {
    activities
        .iter()
        .zip(&problem.row_bounds)
        .chain(point.iter().zip(&problem.var_bounds))
        .filter(|&(_, bound)| *bound != Bound::free())
        .map(|(&value, &bound)| {
            if value.is_finite() {
                bound_violation(value, bound)
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

struct Oracle<'a, F> {
    evaluate: F,
    problem: &'a Problem,
    options: LincoaOptions,
    base: Vec<f64>,
    free: Vec<usize>,
    best: Option<(Vec<f64>, f64, f64)>,
    evaluations: u64,
}

#[derive(Debug, thiserror::Error)]
enum EvaluationError {
    #[error(transparent)]
    User(#[from] OptimizeError),
    #[error("LINCOA evaluation budget exhausted")]
    Budget,
}

impl<F> Oracle<'_, F>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    fn sample(&mut self, offsets: &[f64]) -> Result<f64, EvaluationError> {
        if self.evaluations == self.options.max_evaluations {
            return Err(EvaluationError::Budget);
        }
        let mut point = self.base.clone();
        for (&column, &offset) in self.free.iter().zip(offsets) {
            point[column] += offset;
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(invalid("LINCOA produced a non-finite trial point").into());
        }
        self.evaluations += 1;
        let objective = (self.evaluate)(&point)?;
        if !objective.is_finite() {
            return Err(invalid("LINCOA objective must be finite").into());
        }
        let activities = self
            .problem
            .constraints
            .spmv(&point)
            .map_err(OptimizeError::from)?;
        let violation = violation(self.problem, &point, &activities);
        if violation <= self.options.feasibility_tolerance
            && self
                .best
                .as_ref()
                .is_none_or(|(_, best, _)| objective < *best)
        {
            self.best = Some((point, objective, violation));
        }
        Ok(objective)
    }
}

struct OracleAdapter<'a, 'b, F>(RefCell<&'a mut Oracle<'b, F>>);

impl<F> CostFunction for OracleAdapter<'_, '_, F>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    type Param = Vec<f64>;
    type Output = f64;
    type Error = EvaluationError;

    fn cost(&self, offsets: &Self::Param) -> Result<f64, Self::Error> {
        self.0.borrow_mut().sample(offsets)
    }
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
