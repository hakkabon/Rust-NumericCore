//! Value-only unconstrained optimization with quadratic interpolation.

use basin::{CostFunction, ExecutionLimit, Executor, Newuoa, Termination};
use std::cell::RefCell;

use crate::OptimizeError;

mod uobyqa;

/// Shared controls for NEWUOA and the dense UOBYQA reference method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowellUnconstrainedOptions {
    pub max_iterations: u64,
    pub max_evaluations: u64,
    pub initial_radius: f64,
    pub final_radius: f64,
}

impl Default for PowellUnconstrainedOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 1_000,
            initial_radius: 1.0,
            final_radius: 1e-7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowellUnconstrainedTermination {
    /// The requested trust-region resolution was reached.
    RadiusReached,
    EvaluationLimit,
    IterationLimit,
    NumericalFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowellUnconstrainedResult {
    /// The best point actually evaluated by the objective callback.
    pub point: Vec<f64>,
    pub objective: f64,
    pub evaluations: u64,
    pub termination: PowellUnconstrainedTermination,
}

/// Minimize with Powell's NEWUOA algorithm using Basin and `2n + 1`
/// interpolation points. The evaluation budget includes initialization.
pub fn minimize_newuoa<F>(
    initial: &[f64],
    options: PowellUnconstrainedOptions,
    evaluate: F,
) -> Result<PowellUnconstrainedResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    let mut oracle = Oracle::new(initial, options, evaluate)?;
    let solver = Newuoa::new()
        .with_initial_radius(options.initial_radius)
        .with_final_radius(options.final_radius);
    // Center the internal variables to avoid building a model in coordinates
    // with a large common offset. The oracle returns absolute points.
    let run = Executor::from_start(
        Adapter(RefCell::new(&mut oracle), initial),
        solver,
        vec![0.0; initial.len()],
    )
    .max_iter(options.max_iterations)
    .max_cost_evals(options.max_evaluations)
    .run();
    match run {
        Err(stop) => oracle.stopped(stop),
        Ok(run) => oracle.finish(match run.report.termination {
            Termination::Converged(_) => PowellUnconstrainedTermination::RadiusReached,
            Termination::Limit(limits)
                if limits
                    .iter()
                    .any(|limit| matches!(limit, ExecutionLimit::CostEvaluations { .. })) =>
            {
                PowellUnconstrainedTermination::EvaluationLimit
            }
            Termination::Limit(_) => PowellUnconstrainedTermination::IterationLimit,
            _ => PowellUnconstrainedTermination::NumericalFailure,
        }),
    }
}

/// Minimize with the dense UOBYQA reference variant. It uses the full
/// quadratic interpolation set and updates its geometry using its Lagrange
/// polynomials. Intended for small problems of at most 16 variables.
pub fn minimize_uobyqa<F>(
    initial: &[f64],
    options: PowellUnconstrainedOptions,
    evaluate: F,
) -> Result<PowellUnconstrainedResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    if initial.len() > 16 {
        return Err(OptimizeError::InvalidConfiguration(
            "UOBYQA supports at most 16 variables",
        ));
    }
    let mut oracle = Oracle::new(initial, options, evaluate)?;
    match uobyqa::run(initial, options, &mut oracle) {
        Ok(reason) => oracle.finish(reason),
        Err(stop) => oracle.stopped(stop),
    }
}

struct Sample {
    point: Vec<f64>,
    value: f64,
}

#[derive(Debug, thiserror::Error)]
enum EvaluationStop {
    #[error(transparent)]
    User(#[from] OptimizeError),
    #[error("Powell objective evaluation budget exhausted")]
    Budget,
    #[error("Powell solver produced a non-finite trial point")]
    Numerical,
}

struct Oracle<F> {
    evaluate: F,
    budget: u64,
    calls: u64,
    best: Option<Sample>,
}

impl<F: FnMut(&[f64]) -> Result<f64, OptimizeError>> Oracle<F> {
    fn new(
        initial: &[f64],
        options: PowellUnconstrainedOptions,
        evaluate: F,
    ) -> Result<Self, OptimizeError> {
        if initial.is_empty()
            || initial.iter().any(|x| !x.is_finite())
            || options.max_iterations == 0
            || options.max_evaluations == 0
            || !options.final_radius.is_finite()
            || !options.initial_radius.is_finite()
            || options.final_radius <= 0.0
            || options.final_radius >= options.initial_radius
            || !options.initial_radius.powi(4).is_normal()
            || !options.final_radius.powi(4).is_normal()
        {
            return Err(OptimizeError::InvalidConfiguration(
                "invalid unconstrained Powell dimensions, initial point, radii, or budget",
            ));
        }
        Ok(Self {
            evaluate,
            budget: options.max_evaluations,
            calls: 0,
            best: None,
        })
    }

    fn sample(&mut self, point: Vec<f64>) -> Result<Sample, EvaluationStop> {
        if self.calls >= self.budget {
            return Err(EvaluationStop::Budget);
        }
        if point.iter().any(|x| !x.is_finite()) {
            return Err(EvaluationStop::Numerical);
        }
        self.calls += 1;
        let value = (self.evaluate)(&point)?;
        if !value.is_finite() {
            return Err(
                OptimizeError::InvalidProblem("Powell objective must be finite".into()).into(),
            );
        }
        if self.best.as_ref().is_none_or(|best| value < best.value) {
            self.best = Some(Sample {
                point: point.clone(),
                value,
            });
        }
        Ok(Sample { point, value })
    }

    fn stopped(self, stop: EvaluationStop) -> Result<PowellUnconstrainedResult, OptimizeError> {
        match stop {
            EvaluationStop::User(err) => Err(err),
            EvaluationStop::Budget => self.finish(PowellUnconstrainedTermination::EvaluationLimit),
            EvaluationStop::Numerical => {
                self.finish(PowellUnconstrainedTermination::NumericalFailure)
            }
        }
    }

    fn finish(
        self,
        termination: PowellUnconstrainedTermination,
    ) -> Result<PowellUnconstrainedResult, OptimizeError> {
        let best = self.best.ok_or_else(|| {
            OptimizeError::InvalidProblem(
                "Powell solver returned without an objective evaluation".into(),
            )
        })?;
        Ok(PowellUnconstrainedResult {
            point: best.point,
            objective: best.value,
            evaluations: self.calls,
            termination,
        })
    }
}

struct Adapter<'a, F>(RefCell<&'a mut Oracle<F>>, &'a [f64]);

impl<F: FnMut(&[f64]) -> Result<f64, OptimizeError>> CostFunction for Adapter<'_, F> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = EvaluationStop;

    fn cost(&self, offsets: &Self::Param) -> Result<f64, Self::Error> {
        let point = self
            .1
            .iter()
            .zip(offsets)
            .map(|(base, offset)| base + offset)
            .collect();
        Ok(self.0.borrow_mut().sample(point)?.value)
    }
}
