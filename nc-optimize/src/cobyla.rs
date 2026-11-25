//! Powell's constrained optimization by linear approximations.

use crate::{
    Bound, ConstrainedDerivativeFreeEvaluation, DerivativeFreeConstraintKind, OptimizeError,
};
use basin::{
    Constrained, ConstraintSet, ConstraintSolveError, CostFunction, Executor,
    NonlinearInequalityValues, Termination,
};
use std::cell::RefCell;

// Basin's PRIMA filter moderates objective and residual values outside this range.
const MODEL_LIMIT: f64 = (1_u128 << 100) as f64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CobylaOptions {
    pub max_iterations: u64,
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
    pub feasibility_tolerance: f64,
}

impl Default for CobylaOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 1_000,
            initial_radius: 1.0,
            final_radius: 1e-7,
            feasibility_tolerance: 1e-7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CobylaTermination {
    RadiusReached,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
    AllVariablesFixed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CobylaResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub constraint_values: Vec<f64>,
    /// Maximum violation of the original constraints and variable bounds.
    pub maximum_violation: f64,
    pub feasible: bool,
    pub evaluations: u64,
    pub termination: CobylaTermination,
}

/// Optimize a value-only objective subject to residuals `c(x) <= 0` and
/// optional equalities `c(x) == 0`. The callback receives the full point
/// and must accept trial points outside the supplied constraints and bounds.
pub fn minimize_cobyla<F>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    kinds: &[DerivativeFreeConstraintKind],
    options: CobylaOptions,
    evaluate: F,
) -> Result<CobylaResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
{
    if initial.is_empty()
        || !initial.iter().all(|x| x.is_finite())
        || bounds.is_some_and(|b| b.len() != initial.len())
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
            "invalid COBYLA dimensions, radii, tolerance, or budget",
        ));
    }
    let mut point = initial.to_vec();
    let mut free = Vec::new();
    let mut start = Vec::new();
    let mut free_bounds = Vec::new();
    for (index, value) in point.iter_mut().enumerate() {
        let bound = bounds.map_or(Bound::free(), |b| b[index]);
        let lo = bound.lower.unwrap_or(f64::NEG_INFINITY);
        let hi = bound.upper.unwrap_or(f64::INFINITY);
        if bound
            .lower
            .into_iter()
            .chain(bound.upper)
            .any(|b| !b.is_finite())
            || lo > hi
        {
            return Err(OptimizeError::InvalidConfiguration(
                "COBYLA requires finite ordered bounds when supplied",
            ));
        }
        *value = value.clamp(lo, hi);
        if lo != hi {
            free.push(index);
            start.push(*value);
            free_bounds.push(bound);
        }
    }
    let folded_count = kinds
        .iter()
        .map(|kind| match kind {
            DerivativeFreeConstraintKind::LessEqual => 1,
            DerivativeFreeConstraintKind::Equality => 2,
        })
        .sum::<usize>()
        + free_bounds
            .iter()
            .map(|b| usize::from(b.lower.is_some()) + usize::from(b.upper.is_some()))
            .sum::<usize>();
    let oracle = RefCell::new(Oracle {
        evaluate,
        kinds,
        options,
        point,
        free,
        free_bounds,
        last: None,
        best: None,
        evaluations: 0,
    });
    let termination = if start.is_empty() {
        oracle.borrow_mut().sample(&[]).map_err(|err| match err {
            EvaluationError::User(inner) => inner,
            EvaluationError::Budget => OptimizeError::InvalidConfiguration("empty COBYLA budget"),
        })?;
        CobylaTermination::AllVariablesFixed
    } else {
        let constraints = ConstraintSet::new().with_nonlinear_inequalities(folded_count);
        let problem = Constrained::new(OracleAdapter(&oracle), constraints);
        let solver = basin::Cobyla::new()
            .with_initial_radius(options.initial_radius)
            .with_final_radius(options.final_radius);
        let result = Executor::from_start(problem, solver, start)
            .max_iter(options.max_iterations)
            .max_cost_evals(options.max_evaluations)
            .run();
        match result {
            Err(ConstraintSolveError::Evaluation(EvaluationError::User(err))) => return Err(err),
            Err(ConstraintSolveError::Evaluation(EvaluationError::Budget)) => {
                CobylaTermination::EvaluationLimit
            }
            Err(error) => return Err(OptimizeError::InvalidProblem(format!("COBYLA: {error}"))),
            Ok(result) => match result.report.termination {
                Termination::Converged(_) => CobylaTermination::RadiusReached,
                Termination::Limit(_) if oracle.borrow().evaluations == options.max_evaluations => {
                    CobylaTermination::EvaluationLimit
                }
                Termination::Limit(_) => CobylaTermination::IterationLimit,
                _ => CobylaTermination::NumericalFailure,
            },
        }
    };
    let oracle = oracle.into_inner();
    let best = oracle
        .best
        .ok_or_else(|| OptimizeError::InvalidProblem("COBYLA did not evaluate a point".into()))?;
    Ok(CobylaResult {
        point: best.point,
        objective: best.value.objective,
        constraint_values: best.value.constraints,
        maximum_violation: best.violation,
        feasible: best.violation <= options.feasibility_tolerance,
        evaluations: oracle.evaluations,
        termination,
    })
}

struct Sample {
    point: Vec<f64>,
    value: ConstrainedDerivativeFreeEvaluation,
    violation: f64,
}

impl Sample {
    fn outranks(&self, other: &Self, tolerance: f64) -> bool {
        match (self.violation <= tolerance, other.violation <= tolerance) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => self.value.objective < other.value.objective,
            (false, false) => {
                self.violation < other.violation
                    || (self.violation == other.violation
                        && self.value.objective < other.value.objective)
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum EvaluationError {
    #[error(transparent)]
    User(#[from] OptimizeError),
    #[error("COBYLA evaluation budget exhausted")]
    Budget,
}

struct Oracle<'a, F> {
    evaluate: F,
    kinds: &'a [DerivativeFreeConstraintKind],
    options: CobylaOptions,
    point: Vec<f64>,
    free: Vec<usize>,
    free_bounds: Vec<Bound>,
    last: Option<(Vec<f64>, Vec<f64>)>,
    best: Option<Sample>,
    evaluations: u64,
}

impl<F> Oracle<'_, F>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
{
    fn sample(&mut self, reduced: &[f64]) -> Result<f64, EvaluationError> {
        if self.evaluations == self.options.max_evaluations {
            return Err(EvaluationError::Budget);
        }
        if reduced.len() != self.free.len() || reduced.iter().any(|p| !p.is_finite()) {
            return Err(OptimizeError::InvalidProblem(
                "COBYLA produced an invalid trial point".into(),
            )
            .into());
        }
        for (&index, &item) in self.free.iter().zip(reduced) {
            self.point[index] = item;
        }
        self.evaluations += 1;
        let value = (self.evaluate)(&self.point)?;
        if value.constraints.len() != self.kinds.len()
            || !(value.objective.abs() <= MODEL_LIMIT)
            || !value.constraints.iter().all(|c| c.abs() <= MODEL_LIMIT)
        {
            return Err(OptimizeError::InvalidProblem(
                "COBYLA returned an invalid residual count or a value outside the finite model range".into(),
            )
            .into());
        }
        let mut folded = Vec::new();
        for (&value, &kind) in value.constraints.iter().zip(self.kinds) {
            folded.push(value);
            if kind == DerivativeFreeConstraintKind::Equality {
                folded.push(-value);
            }
        }
        for (&point, bound) in reduced.iter().zip(&self.free_bounds) {
            folded.extend(bound.lower.map(|lo| lo - point));
            folded.extend(bound.upper.map(|hi| point - hi));
        }
        if folded.iter().any(|c| c.abs() > MODEL_LIMIT) {
            return Err(OptimizeError::InvalidProblem(
                "COBYLA bound residual exceeds the model range".into(),
            )
            .into());
        }
        let violation = folded.iter().copied().fold(0.0, f64::max);
        let objective = value.objective;
        let sample = Sample {
            point: self.point.clone(),
            value,
            violation,
        };
        if self
            .best
            .as_ref()
            .is_none_or(|old| sample.outranks(old, self.options.feasibility_tolerance))
        {
            self.best = Some(sample);
        }
        self.last = Some((reduced.to_vec(), folded));
        Ok(objective)
    }
}

struct OracleAdapter<'a, 'b, F>(&'a RefCell<Oracle<'b, F>>);

impl<F> CostFunction for OracleAdapter<'_, '_, F>
where
    F: FnMut(&[f64]) -> Result<ConstrainedDerivativeFreeEvaluation, OptimizeError>,
{
    type Param = Vec<f64>;
    type Output = f64;
    type Error = EvaluationError;

    fn cost(&self, point: &Self::Param) -> Result<f64, Self::Error> {
        self.0.borrow_mut().sample(point)
    }
}

impl<F> NonlinearInequalityValues for OracleAdapter<'_, '_, F> {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = EvaluationError;

    fn nonlinear_inequalities(
        &self,
        point: &Self::Param,
    ) -> Result<Self::Values, ConstraintSolveError<Self::Error>> {
        // Basin evaluates cost first and then its constraint block at the same trial point.
        let mut oracle = self.0.borrow_mut();
        match oracle.last.take() {
            Some((sampled, residuals)) if &sampled == point => Ok(residuals),
            _ => Err(ConstraintSolveError::Evaluation(
                OptimizeError::InvalidProblem("unexpected COBYLA callback sequence".into()).into(),
            )),
        }
    }
}
