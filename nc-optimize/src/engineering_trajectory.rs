//! Checkpointed discrete adjoints of sparse backward-Euler trajectories.

use crate::{
    engineering_sensitivity::{direct_error, effective_tangent, factor_tangent},
    EngineeringAssembly, EngineeringModel, NonlinearExpression, OptimizeError,
};
use nc_sparse::{CsrMatrix, SparseLuFactorization};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy)]
pub struct EngineeringTrajectoryOptions {
    /// Retain the initial/final states and every checkpoint_stride-th state.
    pub checkpoint_stride: usize,
    pub max_newton_iterations: usize,
    pub residual_tolerance: f64,
    pub linear_backward_error_tolerance: f64,
}

impl Default for EngineeringTrajectoryOptions {
    fn default() -> Self {
        Self {
            checkpoint_stride: 32,
            max_newton_iterations: 16,
            residual_tolerance: 1e-10,
            linear_backward_error_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringTrajectoryCheckpoint {
    pub step: usize,
    pub time: f64,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EngineeringTrajectory {
    model: EngineeringModel,
    times: Vec<f64>,
    options: EngineeringTrajectoryOptions,
    checkpoints: Vec<EngineeringTrajectoryCheckpoint>,
    maximum_primal_residual: f64,
}

/// Scalar objective with right-endpoint running quadrature and a terminal term.
/// Expression inputs are time, all states, all derivatives, then parameters.
#[derive(Debug, Clone)]
pub struct EngineeringTrajectoryResponse {
    pub name: String,
    pub running: Option<NonlinearExpression>,
    pub terminal: Option<NonlinearExpression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringTrajectoryGradient {
    pub name: String,
    pub value: f64,
    /// Partial derivative of the discrete objective with the initial state fixed.
    pub parameter_gradient: Vec<f64>,
    /// Pullback to the prescribed initial state, for parameterized initialization.
    pub initial_state_gradient: Vec<f64>,
    pub maximum_adjoint_backward_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringTrajectoryAdjoints {
    pub responses: Vec<EngineeringTrajectoryGradient>,
    pub maximum_primal_residual: f64,
    pub replayed_steps: usize,
    pub peak_segment_states: usize,
    pub transpose_solves: usize,
    pub symbolic_analyses: usize,
}

impl EngineeringTrajectory {
    /// Integrate on a prescribed, strictly increasing grid with sparse Newton
    /// solves. The initial state is an independent input to the discrete problem.
    pub fn integrate(
        model: EngineeringModel,
        times: &[f64],
        initial_state: &[f64],
        options: EngineeringTrajectoryOptions,
    ) -> Result<Self, OptimizeError> {
        model.validate()?;
        if times.len() < 2
            || times.iter().any(|t| !t.is_finite())
            || times.windows(2).any(|w| {
                let h = w[1] - w[0];
                h <= 0.0 || !h.is_finite() || !h.recip().is_finite()
            })
            || initial_state.len() != model.unknowns.len()
            || initial_state.iter().any(|x| !x.is_finite())
            || options.checkpoint_stride == 0
            || options.max_newton_iterations == 0
            || options.residual_tolerance <= 0.0
            || !options.residual_tolerance.is_finite()
            || options.linear_backward_error_tolerance <= 0.0
            || !options.linear_backward_error_tolerance.is_finite()
        {
            return Err(invalid("invalid engineering trajectory configuration"));
        }
        let mut trajectory = Self {
            model,
            times: times.to_vec(),
            options,
            checkpoints: Vec::new(),
            maximum_primal_residual: 0.0,
        };
        let mut state = initial_state.to_vec();
        trajectory
            .checkpoints
            .push(trajectory.checkpoint(0, state.clone()));
        let mut tangent = Tangent::default();
        for k in 1..times.len() {
            let (next, residual) = trajectory.step(k, &state, &mut tangent)?;
            trajectory.maximum_primal_residual = trajectory.maximum_primal_residual.max(residual);
            state = next;
            if k % options.checkpoint_stride == 0 || k + 1 == times.len() {
                trajectory
                    .checkpoints
                    .push(trajectory.checkpoint(k, state.clone()));
            }
        }
        Ok(trajectory)
    }

    pub fn checkpoints(&self) -> &[EngineeringTrajectoryCheckpoint] {
        &self.checkpoints
    }

    pub fn maximum_primal_residual(&self) -> f64 {
        self.maximum_primal_residual
    }

    pub fn adjoints(
        &self,
        responses: &[EngineeringTrajectoryResponse],
    ) -> Result<EngineeringTrajectoryAdjoints, OptimizeError> {
        let n = self.model.unknowns.len();
        let p = self.model.parameters.len();
        let dimension = 1 + 2 * n + p;
        let mut names = BTreeSet::new();
        if responses.is_empty()
            || responses.iter().any(|r| {
                r.name.is_empty()
                    || !names.insert(&r.name)
                    || (r.running.is_none() && r.terminal.is_none())
            })
        {
            return Err(invalid(
                "each trajectory response requires a distinct name and an objective expression",
            ));
        }
        for response in responses {
            for expression in response.running.iter().chain(&response.terminal) {
                expression.validate(dimension)?;
            }
        }
        let mut result = EngineeringTrajectoryAdjoints {
            responses: responses
                .iter()
                .map(|r| EngineeringTrajectoryGradient {
                    name: r.name.clone(),
                    value: 0.0,
                    parameter_gradient: vec![0.0; p],
                    initial_state_gradient: vec![0.0; n],
                    maximum_adjoint_backward_error: 0.0,
                })
                .collect(),
            maximum_primal_residual: self.maximum_primal_residual,
            replayed_steps: 0,
            peak_segment_states: 0,
            transpose_solves: 0,
            symbolic_analyses: 0,
        };
        let mut tangent = Tangent::default();
        let mut replay_tangent = Tangent::default();
        for interval in self.checkpoints.windows(2).rev() {
            let mut states = vec![interval[0].state.clone()];
            for k in interval[0].step + 1..interval[1].step {
                let (state, _) = self.step(k, states.last().unwrap(), &mut replay_tangent)?;
                states.push(state);
                result.replayed_steps += 1;
            }
            states.push(interval[1].state.clone());
            result.peak_segment_states = result.peak_segment_states.max(states.len());
            for i in (1..states.len()).rev() {
                let k = interval[0].step + i;
                let h = self.times[k] - self.times[k - 1];
                let current = &states[i];
                let (derivative, assembly) = self.assembly(k, &states[i - 1], current)?;
                let residual = infinity(&assembly.residuals);
                if residual > self.options.residual_tolerance {
                    return Err(invalid(&format!(
                        "checkpoint replay residual exceeds tolerance at step {k}"
                    )));
                }
                result.maximum_primal_residual = result.maximum_primal_residual.max(residual);
                let factor = tangent.update(&effective_tangent(&assembly, h.recip())?)?;
                for (response, gradient) in responses.iter().zip(&mut result.responses) {
                    let mut derivative_seed = vec![0.0; n];
                    let is_final = k + 1 == self.times.len();
                    let terms = response.running.iter().map(|expr| (expr, h)).chain(
                        response
                            .terminal
                            .iter()
                            .filter(|_| is_final)
                            .map(|expr| (expr, 1.0)),
                    );
                    for (expression, weight) in terms {
                        let (value, partials) =
                            expression.evaluate_sparse_at(dimension, |i| match i {
                                0 => self.times[k],
                                i if i <= n => current[i - 1],
                                i if i <= 2 * n => derivative[i - n - 1],
                                i => self.model.parameters[i - 2 * n - 1].value,
                            })?;
                        gradient.value += weight * value;
                        for (index, partial) in partials.indices.into_iter().zip(partials.values) {
                            let partial = weight * partial;
                            match index {
                                0 => {}
                                i if i <= n => gradient.initial_state_gradient[i - 1] += partial,
                                i if i <= 2 * n => derivative_seed[i - n - 1] += partial / h,
                                i => gradient.parameter_gradient[i - 2 * n - 1] += partial,
                            }
                        }
                    }
                    for (seed, derivative) in gradient
                        .initial_state_gradient
                        .iter_mut()
                        .zip(&derivative_seed)
                    {
                        *seed += derivative;
                    }
                    let report = factor
                        .solve_transpose_refined(&gradient.initial_state_gradient)
                        .map_err(direct_error)?;
                    if report.backward_error > self.options.linear_backward_error_tolerance {
                        return Err(invalid(&format!(
                            "adjoint solve exceeds backward-error tolerance at step {k}"
                        )));
                    }
                    gradient.maximum_adjoint_backward_error = gradient
                        .maximum_adjoint_backward_error
                        .max(report.backward_error);
                    for (row, parameter, partial) in assembly.parameter_jacobian.iter_entries() {
                        gradient.parameter_gradient[parameter] -= partial * report.solution[row];
                    }
                    let carry = assembly
                        .jacobian_derivative
                        .transpose_spmv(&report.solution)?;
                    for ((seed, implicit), direct) in gradient
                        .initial_state_gradient
                        .iter_mut()
                        .zip(carry)
                        .zip(derivative_seed)
                    {
                        *seed = implicit / h - direct;
                    }
                    if !gradient.value.is_finite()
                        || gradient
                            .initial_state_gradient
                            .iter()
                            .any(|v| !v.is_finite())
                    {
                        return Err(invalid(&format!(
                            "non-finite trajectory gradient at step {k}"
                        )));
                    }
                    result.transpose_solves += 1;
                }
            }
        }
        if result
            .responses
            .iter()
            .any(|r| r.parameter_gradient.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("non-finite trajectory parameter gradient"));
        }
        result.symbolic_analyses = tangent.symbolic_analyses + replay_tangent.symbolic_analyses;
        Ok(result)
    }

    fn checkpoint(&self, step: usize, state: Vec<f64>) -> EngineeringTrajectoryCheckpoint {
        EngineeringTrajectoryCheckpoint {
            step,
            time: self.times[step],
            state,
        }
    }

    fn assembly(
        &self,
        k: usize,
        previous: &[f64],
        state: &[f64],
    ) -> Result<(Vec<f64>, EngineeringAssembly), OptimizeError> {
        let h = self.times[k] - self.times[k - 1];
        let derivative: Vec<_> = state
            .iter()
            .zip(previous)
            .map(|(v, old)| (v - old) / h)
            .collect();
        let assembly = self.model.assemble(self.times[k], state, &derivative)?;
        Ok((derivative, assembly))
    }

    fn step(
        &self,
        k: usize,
        previous: &[f64],
        tangent: &mut Tangent,
    ) -> Result<(Vec<f64>, f64), OptimizeError> {
        let alpha = (self.times[k] - self.times[k - 1]).recip();
        let mut state = previous.to_vec();
        let mut assembly = self.assembly(k, previous, &state)?.1;
        let mut residual = infinity(&assembly.residuals);
        for _ in 0..self.options.max_newton_iterations {
            if residual <= self.options.residual_tolerance {
                return Ok((state, residual));
            }
            let factor = tangent.update(&effective_tangent(&assembly, alpha)?)?;
            let rhs: Vec<_> = assembly.residuals.iter().map(|v| -v).collect();
            let correction = factor.solve_refined(&rhs).map_err(direct_error)?;
            if correction.backward_error > self.options.linear_backward_error_tolerance {
                return Err(invalid(&format!(
                    "Newton solve exceeds backward-error tolerance at step {k}"
                )));
            }
            let mut accepted = false;
            for trial in 0..=8 {
                let scale = 0.5_f64.powi(trial);
                let candidate: Vec<_> = state
                    .iter()
                    .zip(&correction.solution)
                    .map(|(x, d)| x + scale * d)
                    .collect();
                if let Ok((_, next)) = self.assembly(k, previous, &candidate) {
                    let next_residual = infinity(&next.residuals);
                    if next_residual < residual {
                        state = candidate;
                        assembly = next;
                        residual = next_residual;
                        accepted = true;
                        break;
                    }
                }
            }
            if !accepted {
                return Err(invalid(&format!(
                    "sparse Newton step stalled at trajectory step {k}"
                )));
            }
        }
        if residual <= self.options.residual_tolerance {
            Ok((state, residual))
        } else {
            Err(invalid(&format!(
                "sparse Newton iteration limit at trajectory step {k}"
            )))
        }
    }
}

#[derive(Default)]
struct Tangent {
    factor: Option<SparseLuFactorization>,
    symbolic_analyses: usize,
}

impl Tangent {
    fn update(&mut self, matrix: &CsrMatrix<f64>) -> Result<&SparseLuFactorization, OptimizeError> {
        match &mut self.factor {
            Some(factor) if factor.symbolic_analysis().validate_pattern(matrix).is_ok() => {
                factor.refactor(matrix).map_err(direct_error)?;
            }
            _ => {
                self.factor = Some(factor_tangent(matrix)?);
                self.symbolic_analyses += 1;
            }
        }
        Ok(self.factor.as_ref().unwrap())
    }
}

fn infinity(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |norm, &v| norm.max(v.abs()))
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
