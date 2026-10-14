//! Adaptive initial-value integration for ODEs and index-1 DAEs.

use crate::OptimizeError;

#[derive(Debug, Clone, PartialEq)]
pub struct TimePoint {
    pub time: f64,
    pub state: Vec<f64>,
    pub derivative: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDirection {
    Any,
    Rising,
    Falling,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimeEvent {
    pub index: usize,
    pub time: f64,
    pub state: Vec<f64>,
    pub value: f64,
    pub terminal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeIntegrationTermination {
    ReachedEnd,
    TerminalEvent,
    StepLimit,
    StepSizeUnderflow,
    NewtonFailure,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimeIntegrationOptions {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub initial_step: f64,
    pub minimum_step: f64,
    pub maximum_step: f64,
    pub max_steps: usize,
    pub max_rejected_steps: usize,
    pub max_newton_iterations: usize,
    pub newton_tolerance: f64,
}

impl Default for TimeIntegrationOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-8,
            relative_tolerance: 1e-6,
            initial_step: 1e-3,
            minimum_step: 1e-12,
            maximum_step: 0.1,
            max_steps: 100_000,
            max_rejected_steps: 10_000,
            max_newton_iterations: 12,
            newton_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimeIntegrationIteration {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub time: f64,
    pub step: f64,
    pub error_norm: f64,
    pub newton_iterations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimeIntegrationResult {
    pub points: Vec<TimePoint>,
    pub events: Vec<TimeEvent>,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub evaluations: usize,
    pub jacobian_evaluations: usize,
    pub newton_iterations: usize,
    pub termination: TimeIntegrationTermination,
}

impl TimeIntegrationResult {
    /// Piecewise cubic Hermite interpolation over accepted steps.
    pub fn evaluate(&self, time: f64) -> Option<Vec<f64>> {
        let first = self.points.first()?;
        let last = self.points.last()?;
        let increasing = last.time >= first.time;
        if (increasing && (time < first.time || time > last.time))
            || (!increasing && (time > first.time || time < last.time))
        {
            return None;
        }
        if time == last.time {
            return Some(last.state.clone());
        }
        let index = self.points.windows(2).position(|pair| {
            if increasing {
                time >= pair[0].time && time <= pair[1].time
            } else {
                time <= pair[0].time && time >= pair[1].time
            }
        })?;
        Some(hermite(&self.points[index], &self.points[index + 1], time))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DaeEvaluation {
    pub residuals: Vec<f64>,
    pub jacobian_state: Vec<Vec<f64>>,
    pub jacobian_derivative: Vec<Vec<f64>>,
}

impl DaeEvaluation {
    /// Build `F = M*yDot - f(t,y)` and its two Jacobian blocks. `M` may be
    /// singular, which is the usual semi-explicit index-1 DAE case.
    pub fn from_mass_matrix(
        mass_matrix: Vec<Vec<f64>>,
        rhs: &[f64],
        rhs_jacobian: &[Vec<f64>],
        derivative: &[f64],
    ) -> Self {
        let residuals = mass_matrix
            .iter()
            .enumerate()
            .map(|(row, values)| {
                values
                    .iter()
                    .zip(derivative)
                    .map(|(m, d)| m * d)
                    .sum::<f64>()
                    - rhs.get(row).copied().unwrap_or(f64::NAN)
            })
            .collect();
        Self {
            residuals,
            jacobian_state: rhs_jacobian
                .iter()
                .map(|row| row.iter().map(|value| -value).collect())
                .collect(),
            jacobian_derivative: mass_matrix,
        }
    }
}

pub fn integrate_ode<F>(
    start: f64,
    end: f64,
    initial: &[f64],
    options: TimeIntegrationOptions,
    rhs: F,
) -> Result<TimeIntegrationResult, OptimizeError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    integrate_ode_with_events(
        start,
        end,
        initial,
        options,
        rhs,
        &[],
        &[],
        |_, _| Ok(vec![]),
        |_| true,
    )
}

pub fn integrate_ode_with_events<F, E, O>(
    start: f64,
    end: f64,
    initial: &[f64],
    options: TimeIntegrationOptions,
    mut rhs: F,
    directions: &[EventDirection],
    terminal: &[bool],
    mut event: E,
    mut observer: O,
) -> Result<TimeIntegrationResult, OptimizeError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, OptimizeError>,
    E: FnMut(f64, &[f64]) -> Result<Vec<f64>, OptimizeError>,
    O: FnMut(&TimeIntegrationIteration) -> bool,
{
    validate(start, end, initial, &options)?;
    if directions.len() != terminal.len() {
        return Err(invalid("event metadata lengths differ"));
    }
    let direction = if end >= start { 1.0 } else { -1.0 };
    let mut time = start;
    let mut state = initial.to_vec();
    let mut evaluations = 0;
    let mut derivative = checked_vector(rhs(time, &state)?, state.len())?;
    evaluations += 1;
    let mut previous_events = checked_events(event(time, &state)?, directions.len())?;
    let mut step = direction * options.initial_step.min(options.maximum_step);
    let mut points = vec![TimePoint {
        time,
        state: state.clone(),
        derivative: derivative.clone(),
    }];
    let mut events = vec![];
    let mut accepted = 0;
    let mut rejected = 0;
    while !reached(time, end, direction) {
        if accepted >= options.max_steps {
            return Ok(finish(
                points,
                events,
                accepted,
                rejected,
                evaluations,
                0,
                0,
                TimeIntegrationTermination::StepLimit,
            ));
        }
        if rejected > options.max_rejected_steps {
            return Ok(finish(
                points,
                events,
                accepted,
                rejected,
                evaluations,
                0,
                0,
                TimeIntegrationTermination::StepSizeUnderflow,
            ));
        }
        step = direction * step.abs().min((end - time).abs()).min(options.maximum_step);
        if step.abs() < options.minimum_step {
            return Ok(finish(
                points,
                events,
                accepted,
                rejected,
                evaluations,
                0,
                0,
                TimeIntegrationTermination::StepSizeUnderflow,
            ));
        }
        let trial = rk45_step(time, &state, &derivative, step, &mut rhs)?;
        evaluations += 6;
        let error = scaled_error(&trial.high, &trial.low, &state, &options);
        if error <= 1.0 {
            let next_time = time + step;
            let next_events = checked_events(event(next_time, &trial.high)?, directions.len())?;
            let next_point = TimePoint {
                time: next_time,
                state: trial.high.clone(),
                derivative: trial.derivative,
            };
            let prior_point = points.last().unwrap().clone();
            for index in 0..directions.len() {
                if crossed(
                    previous_events[index],
                    next_events[index],
                    directions[index],
                ) {
                    let (root_time, root_state, root_value) = locate_event(
                        &prior_point,
                        &next_point,
                        index,
                        previous_events[index],
                        next_events[index],
                        &mut event,
                    )?;
                    let found = TimeEvent {
                        index,
                        time: root_time,
                        state: root_state.clone(),
                        value: root_value,
                        terminal: terminal[index],
                    };
                    events.push(found);
                    if terminal[index] {
                        let root_derivative =
                            checked_vector(rhs(root_time, &root_state)?, state.len())?;
                        evaluations += 1;
                        points.push(TimePoint {
                            time: root_time,
                            state: root_state,
                            derivative: root_derivative,
                        });
                        return Ok(finish(
                            points,
                            events,
                            accepted + 1,
                            rejected,
                            evaluations,
                            0,
                            0,
                            TimeIntegrationTermination::TerminalEvent,
                        ));
                    }
                }
            }
            time = next_time;
            state = trial.high;
            derivative = next_point.derivative.clone();
            points.push(next_point);
            previous_events = next_events;
            accepted += 1;
            if !observer(&TimeIntegrationIteration {
                accepted_steps: accepted,
                rejected_steps: rejected,
                time,
                step,
                error_norm: error,
                newton_iterations: 0,
            }) {
                return Ok(finish(
                    points,
                    events,
                    accepted,
                    rejected,
                    evaluations,
                    0,
                    0,
                    TimeIntegrationTermination::Cancelled,
                ));
            }
            step *= factor(error, 5.0);
        } else {
            rejected += 1;
            step *= factor(error, 5.0).min(0.9);
        }
    }
    Ok(finish(
        points,
        events,
        accepted,
        rejected,
        evaluations,
        0,
        0,
        TimeIntegrationTermination::ReachedEnd,
    ))
}

pub fn integrate_dae<F>(
    start: f64,
    end: f64,
    initial_state: &[f64],
    initial_derivative: &[f64],
    options: TimeIntegrationOptions,
    system: F,
) -> Result<TimeIntegrationResult, OptimizeError>
where
    F: FnMut(f64, &[f64], &[f64]) -> Result<DaeEvaluation, OptimizeError>,
{
    integrate_dae_with_observer(
        start,
        end,
        initial_state,
        initial_derivative,
        options,
        system,
        |_| true,
    )
}

pub fn integrate_dae_with_observer<F, O>(
    start: f64,
    end: f64,
    initial_state: &[f64],
    initial_derivative: &[f64],
    options: TimeIntegrationOptions,
    mut system: F,
    mut observer: O,
) -> Result<TimeIntegrationResult, OptimizeError>
where
    F: FnMut(f64, &[f64], &[f64]) -> Result<DaeEvaluation, OptimizeError>,
    O: FnMut(&TimeIntegrationIteration) -> bool,
{
    validate(start, end, initial_state, &options)?;
    if initial_derivative.len() != initial_state.len()
        || initial_derivative.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid initial DAE derivative"));
    }
    let initial_eval = checked_dae(
        system(start, initial_state, initial_derivative)?,
        initial_state.len(),
    )?;
    if inf_norm(&initial_eval.residuals) > options.newton_tolerance.max(options.absolute_tolerance)
    {
        return Err(invalid("inconsistent initial DAE state and derivative"));
    }
    let direction = if end >= start { 1.0 } else { -1.0 };
    let mut time = start;
    let mut state = initial_state.to_vec();
    let mut derivative = initial_derivative.to_vec();
    let mut step = direction * options.initial_step.min(options.maximum_step);
    let mut points = vec![TimePoint {
        time,
        state: state.clone(),
        derivative: derivative.clone(),
    }];
    let mut accepted = 0;
    let mut rejected = 0;
    let mut evaluations = 1;
    let mut jacobians = 1;
    let mut newton_total = 0;
    while !reached(time, end, direction) {
        if accepted >= options.max_steps {
            return Ok(finish(
                points,
                vec![],
                accepted,
                rejected,
                evaluations,
                jacobians,
                newton_total,
                TimeIntegrationTermination::StepLimit,
            ));
        }
        step = direction * step.abs().min((end - time).abs()).min(options.maximum_step);
        if step.abs() < options.minimum_step || rejected > options.max_rejected_steps {
            return Ok(finish(
                points,
                vec![],
                accepted,
                rejected,
                evaluations,
                jacobians,
                newton_total,
                TimeIntegrationTermination::StepSizeUnderflow,
            ));
        }
        let full = dae_step(time, &state, &derivative, step, &options, &mut system)?;
        evaluations += full.evaluations;
        jacobians += full.jacobians;
        newton_total += full.iterations;
        let half1 = dae_step(time, &state, &derivative, step * 0.5, &options, &mut system)?;
        evaluations += half1.evaluations;
        jacobians += half1.jacobians;
        newton_total += half1.iterations;
        let half2 = dae_step(
            time + step * 0.5,
            &half1.state,
            &half1.derivative,
            step * 0.5,
            &options,
            &mut system,
        )?;
        evaluations += half2.evaluations;
        jacobians += half2.jacobians;
        newton_total += half2.iterations;
        if !(full.converged && half1.converged && half2.converged) {
            rejected += 1;
            step *= 0.5;
            if step.abs() < options.minimum_step {
                return Ok(finish(
                    points,
                    vec![],
                    accepted,
                    rejected,
                    evaluations,
                    jacobians,
                    newton_total,
                    TimeIntegrationTermination::NewtonFailure,
                ));
            }
            continue;
        }
        let error = scaled_error(&half2.state, &full.state, &state, &options);
        if error <= 1.0 {
            time += step;
            state = half2.state;
            derivative = half2.derivative;
            accepted += 1;
            points.push(TimePoint {
                time,
                state: state.clone(),
                derivative: derivative.clone(),
            });
            if !observer(&TimeIntegrationIteration {
                accepted_steps: accepted,
                rejected_steps: rejected,
                time,
                step,
                error_norm: error,
                newton_iterations: full.iterations + half1.iterations + half2.iterations,
            }) {
                return Ok(finish(
                    points,
                    vec![],
                    accepted,
                    rejected,
                    evaluations,
                    jacobians,
                    newton_total,
                    TimeIntegrationTermination::Cancelled,
                ));
            }
            step *= factor(error, 2.0);
        } else {
            rejected += 1;
            step *= factor(error, 2.0).min(0.9);
        }
    }
    Ok(finish(
        points,
        vec![],
        accepted,
        rejected,
        evaluations,
        jacobians,
        newton_total,
        TimeIntegrationTermination::ReachedEnd,
    ))
}

struct RkTrial {
    high: Vec<f64>,
    low: Vec<f64>,
    derivative: Vec<f64>,
}
fn rk45_step<F>(
    t: f64,
    y: &[f64],
    k1: &[f64],
    h: f64,
    rhs: &mut F,
) -> Result<RkTrial, OptimizeError>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    let n = y.len();
    let stage = |terms: &[(f64, &[f64])]| {
        (0..n)
            .map(|i| y[i] + h * terms.iter().map(|(a, k)| a * k[i]).sum::<f64>())
            .collect::<Vec<_>>()
    };
    let y2 = stage(&[(1. / 5., k1)]);
    let k2 = checked_vector(rhs(t + h / 5., &y2)?, n)?;
    let y3 = stage(&[(3. / 40., k1), (9. / 40., &k2)]);
    let k3 = checked_vector(rhs(t + 3. * h / 10., &y3)?, n)?;
    let y4 = stage(&[(44. / 45., k1), (-56. / 15., &k2), (32. / 9., &k3)]);
    let k4 = checked_vector(rhs(t + 4. * h / 5., &y4)?, n)?;
    let y5 = stage(&[
        (19372. / 6561., k1),
        (-25360. / 2187., &k2),
        (64448. / 6561., &k3),
        (-212. / 729., &k4),
    ]);
    let k5 = checked_vector(rhs(t + 8. * h / 9., &y5)?, n)?;
    let y6 = stage(&[
        (9017. / 3168., k1),
        (-355. / 33., &k2),
        (46732. / 5247., &k3),
        (49. / 176., &k4),
        (-5103. / 18656., &k5),
    ]);
    let k6 = checked_vector(rhs(t + h, &y6)?, n)?;
    let high = stage(&[
        (35. / 384., k1),
        (500. / 1113., &k3),
        (125. / 192., &k4),
        (-2187. / 6784., &k5),
        (11. / 84., &k6),
    ]);
    let k7 = checked_vector(rhs(t + h, &high)?, n)?;
    let low = stage(&[
        (5179. / 57600., k1),
        (7571. / 16695., &k3),
        (393. / 640., &k4),
        (-92097. / 339200., &k5),
        (187. / 2100., &k6),
        (1. / 40., &k7),
    ]);
    Ok(RkTrial {
        high,
        low,
        derivative: k7,
    })
}

struct DaeStep {
    state: Vec<f64>,
    derivative: Vec<f64>,
    iterations: usize,
    evaluations: usize,
    jacobians: usize,
    converged: bool,
}
fn dae_step<F>(
    t: f64,
    old: &[f64],
    old_derivative: &[f64],
    h: f64,
    options: &TimeIntegrationOptions,
    system: &mut F,
) -> Result<DaeStep, OptimizeError>
where
    F: FnMut(f64, &[f64], &[f64]) -> Result<DaeEvaluation, OptimizeError>,
{
    let n = old.len();
    let mut state: Vec<f64> = old
        .iter()
        .zip(old_derivative)
        .map(|(y, d)| y + h * d)
        .collect();
    for iteration in 0..options.max_newton_iterations {
        let derivative: Vec<_> = state.iter().zip(old).map(|(y, o)| (y - o) / h).collect();
        let eval = checked_dae(system(t + h, &state, &derivative)?, n)?;
        if inf_norm(&eval.residuals) <= options.newton_tolerance {
            return Ok(DaeStep {
                state,
                derivative,
                iterations: iteration,
                evaluations: iteration + 1,
                jacobians: iteration + 1,
                converged: true,
            });
        }
        let matrix: Vec<Vec<_>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| eval.jacobian_state[i][j] + eval.jacobian_derivative[i][j] / h)
                    .collect()
            })
            .collect();
        let Some(delta) = dense_solve(matrix, eval.residuals.iter().map(|v| -v).collect()) else {
            return Ok(DaeStep {
                state,
                derivative,
                iterations: iteration + 1,
                evaluations: iteration + 1,
                jacobians: iteration + 1,
                converged: false,
            });
        };
        for i in 0..n {
            state[i] += delta[i];
        }
    }
    let derivative = state.iter().zip(old).map(|(y, o)| (y - o) / h).collect();
    Ok(DaeStep {
        state,
        derivative,
        iterations: options.max_newton_iterations,
        evaluations: options.max_newton_iterations,
        jacobians: options.max_newton_iterations,
        converged: false,
    })
}

fn locate_event<E>(
    left: &TimePoint,
    right: &TimePoint,
    index: usize,
    mut lv: f64,
    mut rv: f64,
    event: &mut E,
) -> Result<(f64, Vec<f64>, f64), OptimizeError>
where
    E: FnMut(f64, &[f64]) -> Result<Vec<f64>, OptimizeError>,
{
    let mut a = left.time;
    let mut b = right.time;
    for _ in 0..50 {
        let mid = 0.5 * (a + b);
        let state = hermite(left, right, mid);
        let values = event(mid, &state)?;
        if index >= values.len() || !values[index].is_finite() {
            return Err(invalid("invalid event evaluation"));
        }
        let value = values[index];
        if value == 0. || (b - a).abs() <= 1e-12 * (1. + mid.abs()) {
            return Ok((mid, state, value));
        }
        if (lv <= 0. && value >= 0.) || (lv >= 0. && value <= 0.) {
            b = mid;
            rv = value;
        } else {
            a = mid;
            lv = value;
        }
    }
    let time = 0.5 * (a + b);
    let state = hermite(left, right, time);
    Ok((time, state, 0.5 * (lv + rv)))
}
fn hermite(a: &TimePoint, b: &TimePoint, time: f64) -> Vec<f64> {
    let h = b.time - a.time;
    if h == 0. {
        return a.state.clone();
    }
    let s = (time - a.time) / h;
    let h00 = 2. * s.powi(3) - 3. * s.powi(2) + 1.;
    let h10 = s.powi(3) - 2. * s.powi(2) + s;
    let h01 = -2. * s.powi(3) + 3. * s.powi(2);
    let h11 = s.powi(3) - s.powi(2);
    (0..a.state.len())
        .map(|i| {
            h00 * a.state[i]
                + h10 * h * a.derivative[i]
                + h01 * b.state[i]
                + h11 * h * b.derivative[i]
        })
        .collect()
}
fn crossed(a: f64, b: f64, direction: EventDirection) -> bool {
    match direction {
        EventDirection::Any => (a < 0. && b >= 0.) || (a > 0. && b <= 0.),
        EventDirection::Rising => a < 0. && b >= 0.,
        EventDirection::Falling => a > 0. && b <= 0.,
    }
}
fn reached(time: f64, end: f64, direction: f64) -> bool {
    direction * (end - time) <= 16. * f64::EPSILON * (1. + end.abs())
}
fn factor(error: f64, order: f64) -> f64 {
    if error == 0. {
        5.
    } else {
        (0.9 * error.powf(-1. / order)).clamp(0.2, 5.)
    }
}
fn scaled_error(high: &[f64], low: &[f64], old: &[f64], options: &TimeIntegrationOptions) -> f64 {
    let sum = (0..high.len())
        .map(|i| {
            let scale = options.absolute_tolerance
                + options.relative_tolerance * old[i].abs().max(high[i].abs());
            ((high[i] - low[i]) / scale).powi(2)
        })
        .sum::<f64>();
    (sum / high.len().max(1) as f64).sqrt()
}
fn validate(
    start: f64,
    end: f64,
    initial: &[f64],
    o: &TimeIntegrationOptions,
) -> Result<(), OptimizeError> {
    if !start.is_finite()
        || !end.is_finite()
        || initial.is_empty()
        || initial.iter().any(|v| !v.is_finite())
        || !o.absolute_tolerance.is_finite()
        || o.absolute_tolerance <= 0.
        || !o.relative_tolerance.is_finite()
        || o.relative_tolerance < 0.
        || !o.initial_step.is_finite()
        || o.initial_step <= 0.
        || !o.minimum_step.is_finite()
        || o.minimum_step <= 0.
        || !o.maximum_step.is_finite()
        || o.maximum_step < o.minimum_step
        || o.max_steps == 0
        || o.max_newton_iterations == 0
        || !o.newton_tolerance.is_finite()
        || o.newton_tolerance <= 0.
    {
        return Err(invalid("invalid time integration configuration"));
    }
    Ok(())
}
fn checked_vector(values: Vec<f64>, n: usize) -> Result<Vec<f64>, OptimizeError> {
    if values.len() != n || values.iter().any(|v| !v.is_finite()) {
        Err(invalid("invalid differential equation evaluation"))
    } else {
        Ok(values)
    }
}
fn checked_events(values: Vec<f64>, n: usize) -> Result<Vec<f64>, OptimizeError> {
    if values.len() != n || values.iter().any(|v| !v.is_finite()) {
        Err(invalid("invalid event evaluation"))
    } else {
        Ok(values)
    }
}
fn checked_dae(e: DaeEvaluation, n: usize) -> Result<DaeEvaluation, OptimizeError> {
    let matrix = |m: &Vec<Vec<f64>>| {
        m.len() == n
            && m.iter()
                .all(|r| r.len() == n && r.iter().all(|v| v.is_finite()))
    };
    if e.residuals.len() != n
        || e.residuals.iter().any(|v| !v.is_finite())
        || !matrix(&e.jacobian_state)
        || !matrix(&e.jacobian_derivative)
    {
        Err(invalid("invalid DAE evaluation"))
    } else {
        Ok(e)
    }
}
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0., |m, x| m.max(x.abs()))
}
fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[p][k].abs() <= 1e-14 {
            return None;
        }
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] -= a[i][j] * b[j];
        }
        b[i] /= a[i][i];
    }
    Some(b)
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
fn finish(
    points: Vec<TimePoint>,
    events: Vec<TimeEvent>,
    accepted_steps: usize,
    rejected_steps: usize,
    evaluations: usize,
    jacobian_evaluations: usize,
    newton_iterations: usize,
    termination: TimeIntegrationTermination,
) -> TimeIntegrationResult {
    TimeIntegrationResult {
        points,
        events,
        accepted_steps,
        rejected_steps,
        evaluations,
        jacobian_evaluations,
        newton_iterations,
        termination,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rk45_dense_output_and_terminal_event() {
        let result = integrate_ode_with_events(
            0.,
            2.,
            &[1.],
            TimeIntegrationOptions::default(),
            |_, y| Ok(vec![-y[0]]),
            &[EventDirection::Falling],
            &[true],
            |_, y| Ok(vec![y[0] - 0.5]),
            |_| true,
        )
        .unwrap();
        assert_eq!(
            result.termination,
            TimeIntegrationTermination::TerminalEvent
        );
        assert!((result.events[0].time - 2f64.ln()).abs() < 2e-5);
        assert!((result.evaluate(0.25).unwrap()[0] - (-0.25f64).exp()).abs() < 2e-5);
    }
    #[test]
    fn integrates_index_one_dae() {
        let options = TimeIntegrationOptions {
            initial_step: 0.02,
            maximum_step: 0.05,
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            ..Default::default()
        };
        let result = integrate_dae(0., 1., &[1., 0.], &[-1., 1.], options, |_, y, yd| {
            Ok(DaeEvaluation::from_mass_matrix(
                vec![vec![1., 0.], vec![0., 0.]],
                &[-y[0], 1. - y[0] - y[1]],
                &[vec![-1., 0.], vec![-1., -1.]],
                yd,
            ))
        })
        .unwrap();
        assert_eq!(result.termination, TimeIntegrationTermination::ReachedEnd);
        let last = &result.points.last().unwrap().state;
        assert!((last[0] - (-1f64).exp()).abs() < 3e-3);
        assert!((last[0] + last[1] - 1.).abs() < 1e-8);
        assert!(result.newton_iterations > 0);
    }
    #[test]
    fn rejects_inconsistent_dae_initial_conditions() {
        let error = integrate_dae(0., 1., &[1.], &[0.], Default::default(), |_, y, yd| {
            Ok(DaeEvaluation {
                residuals: vec![yd[0] + y[0]],
                jacobian_state: vec![vec![1.]],
                jacobian_derivative: vec![vec![1.]],
            })
        });
        assert!(error.is_err());
    }

    #[test]
    fn reverse_time_and_cancellation_are_explicit() {
        let backward = integrate_ode(
            1.,
            0.,
            &[std::f64::consts::E],
            Default::default(),
            |_, y| Ok(vec![y[0]]),
        )
        .unwrap();
        assert_eq!(backward.termination, TimeIntegrationTermination::ReachedEnd);
        assert!((backward.points.last().unwrap().state[0] - 1.).abs() < 2e-6);
        let cancelled = integrate_ode_with_events(
            0.,
            1.,
            &[1.],
            Default::default(),
            |_, y| Ok(vec![-y[0]]),
            &[],
            &[],
            |_, _| Ok(vec![]),
            |_| false,
        )
        .unwrap();
        assert_eq!(cancelled.termination, TimeIntegrationTermination::Cancelled);
        assert_eq!(cancelled.accepted_steps, 1);
    }
}
