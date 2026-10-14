//! Value-only optimization for nonsmooth or unavailable-derivative models.

use crate::{Bound, OptimizeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeFreeTermination {
    Converged,
    IterationLimit,
    EvaluationLimit,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct DerivativeFreeIteration {
    pub iteration: usize,
    pub point: Vec<f64>,
    pub objective: f64,
    pub evaluations: usize,
    pub size: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NelderMeadOptions {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub initial_step: f64,
    pub point_tolerance: f64,
    pub objective_tolerance: f64,
    pub reflection: f64,
    pub expansion: f64,
    pub contraction: f64,
    pub shrink: f64,
}
impl Default for NelderMeadOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 10_000,
            initial_step: 0.05,
            point_tolerance: 1e-8,
            objective_tolerance: 1e-10,
            reflection: 1.0,
            expansion: 2.0,
            contraction: 0.5,
            shrink: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub simplex_diameter: f64,
    pub objective_spread: f64,
    pub termination: DerivativeFreeTermination,
}

pub fn minimize_nelder_mead<F>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    options: NelderMeadOptions,
    evaluate: F,
) -> Result<NelderMeadResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    minimize_nelder_mead_with_observer(initial, bounds, options, evaluate, |_| true)
}

pub fn minimize_nelder_mead_with_observer<F, O>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    options: NelderMeadOptions,
    mut evaluate: F,
    mut observer: O,
) -> Result<NelderMeadResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
    O: FnMut(&DerivativeFreeIteration) -> bool,
{
    validate_nelder(initial, bounds, options)?;
    let box_bounds = owned_bounds(initial.len(), bounds);
    let first = project(initial, &box_bounds);
    let mut simplex = Vec::with_capacity(initial.len() + 1);
    simplex.push((first.clone(), checked(&mut evaluate, &first)?));
    for dimension in 0..initial.len() {
        let mut vertex = first.clone();
        let delta = options.initial_step * vertex[dimension].abs().max(1.0);
        vertex[dimension] += delta;
        vertex = project(&vertex, &box_bounds);
        if vertex[dimension] == first[dimension] {
            vertex[dimension] -= delta;
            vertex = project(&vertex, &box_bounds);
        }
        if vertex == first {
            return Err(invalid("Nelder-Mead requires non-fixed parameter bounds"));
        }
        let value = checked(&mut evaluate, &vertex)?;
        simplex.push((vertex, value));
    }
    let mut evaluations = simplex.len();
    for iteration in 0..options.max_iterations {
        sort_simplex(&mut simplex);
        let diameter = diameter(&simplex);
        let spread = simplex.last().unwrap().1 - simplex[0].1;
        if diameter <= options.point_tolerance && spread <= options.objective_tolerance {
            return Ok(nelder_result(
                simplex,
                iteration,
                evaluations,
                DerivativeFreeTermination::Converged,
            ));
        }
        if evaluations >= options.max_evaluations {
            return Ok(nelder_result(
                simplex,
                iteration,
                evaluations,
                DerivativeFreeTermination::EvaluationLimit,
            ));
        }
        if !observer(&DerivativeFreeIteration {
            iteration,
            point: simplex[0].0.clone(),
            objective: simplex[0].1,
            evaluations,
            size: diameter,
        }) {
            return Ok(nelder_result(
                simplex,
                iteration,
                evaluations,
                DerivativeFreeTermination::Cancelled,
            ));
        }
        let centroid = centroid(&simplex[..simplex.len() - 1]);
        let last = simplex.len() - 1;
        let reflected = project(
            &affine(&centroid, &simplex[last].0, options.reflection),
            &box_bounds,
        );
        let reflected_value = checked(&mut evaluate, &reflected)?;
        evaluations += 1;
        if reflected_value < simplex[0].1 && evaluations < options.max_evaluations {
            let expanded = project(
                &affine(&centroid, &reflected, -options.expansion),
                &box_bounds,
            );
            let expanded_value = checked(&mut evaluate, &expanded)?;
            evaluations += 1;
            simplex[last] = if expanded_value < reflected_value {
                (expanded, expanded_value)
            } else {
                (reflected, reflected_value)
            };
        } else if reflected_value < simplex[last - 1].1 {
            simplex[last] = (reflected, reflected_value);
        } else if evaluations < options.max_evaluations {
            let outside = reflected_value < simplex[last].1;
            let target = if outside {
                &reflected
            } else {
                &simplex[last].0
            };
            let contracted: Vec<_> = centroid
                .iter()
                .zip(target)
                .map(|(c, x)| c + options.contraction * (x - c))
                .collect();
            let contracted = project(&contracted, &box_bounds);
            let contracted_value = checked(&mut evaluate, &contracted)?;
            evaluations += 1;
            let threshold = if outside {
                reflected_value
            } else {
                simplex[last].1
            };
            if contracted_value < threshold {
                simplex[last] = (contracted, contracted_value);
            } else {
                let best = simplex[0].0.clone();
                for item in simplex.iter_mut().skip(1) {
                    if evaluations >= options.max_evaluations {
                        break;
                    }
                    let candidate: Vec<_> = best
                        .iter()
                        .zip(&item.0)
                        .map(|(b, x)| b + options.shrink * (x - b))
                        .collect();
                    let candidate = project(&candidate, &box_bounds);
                    let value = checked(&mut evaluate, &candidate)?;
                    evaluations += 1;
                    *item = (candidate, value);
                }
            }
        }
    }
    sort_simplex(&mut simplex);
    Ok(nelder_result(
        simplex,
        options.max_iterations,
        evaluations,
        DerivativeFreeTermination::IterationLimit,
    ))
}

#[derive(Debug, Clone, Copy)]
pub struct PatternSearchOptions {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub initial_step: f64,
    pub minimum_step: f64,
    pub contraction: f64,
    pub expansion: f64,
    pub objective_tolerance: f64,
}
impl Default for PatternSearchOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 10_000,
            initial_step: 1.0,
            minimum_step: 1e-8,
            contraction: 0.5,
            expansion: 1.25,
            objective_tolerance: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternSearchResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub successful_polls: usize,
    pub final_step: f64,
    pub termination: DerivativeFreeTermination,
}

/// Options for bound-constrained quadratic-model trust-region optimization.
#[derive(Debug, Clone, Copy)]
pub struct ModelBasedOptions {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub initial_radius: f64,
    pub final_radius: f64,
    pub maximum_radius: f64,
    pub acceptance_threshold: f64,
    pub contraction: f64,
    pub expansion: f64,
    pub objective_tolerance: f64,
}
impl Default for ModelBasedOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            max_evaluations: 20_000,
            initial_radius: 1.0,
            final_radius: 1e-7,
            maximum_radius: 100.0,
            acceptance_threshold: 0.1,
            contraction: 0.5,
            expansion: 2.0,
            objective_tolerance: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelBasedResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub model_builds: usize,
    pub accepted_steps: usize,
    pub one_sided_stencils: usize,
    pub final_radius: f64,
    pub final_projected_gradient_norm: f64,
    pub last_reduction_ratio: f64,
    pub termination: DerivativeFreeTermination,
}

/// Minimizes a value-only objective with local diagonal-quadratic interpolation models.
pub fn minimize_model_based<F>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    options: ModelBasedOptions,
    evaluate: F,
) -> Result<ModelBasedResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    minimize_model_based_with_observer(initial, bounds, options, evaluate, |_| true)
}

pub fn minimize_model_based_with_observer<F, O>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    options: ModelBasedOptions,
    mut evaluate: F,
    mut observer: O,
) -> Result<ModelBasedResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
    O: FnMut(&DerivativeFreeIteration) -> bool,
{
    validate_model_based(initial, bounds, options)?;
    let box_bounds = owned_bounds(initial.len(), bounds);
    let mut point = project(initial, &box_bounds);
    let mut value = checked(&mut evaluate, &point)?;
    let mut evaluations = 1;
    let mut radius = options.initial_radius.min(options.maximum_radius);
    let mut model_builds = 0;
    let mut accepted_steps = 0;
    let mut one_sided_stencils = 0;
    let mut projected_gradient_norm = f64::INFINITY;
    let mut last_ratio = f64::NAN;

    for iteration in 0..options.max_iterations {
        if radius <= options.final_radius {
            return Ok(model_based_result(
                point,
                value,
                iteration,
                evaluations,
                model_builds,
                accepted_steps,
                one_sided_stencils,
                radius,
                projected_gradient_norm,
                last_ratio,
                DerivativeFreeTermination::Converged,
            ));
        }
        if evaluations + 2 * point.len() + 1 > options.max_evaluations {
            return Ok(model_based_result(
                point,
                value,
                iteration,
                evaluations,
                model_builds,
                accepted_steps,
                one_sided_stencils,
                radius,
                projected_gradient_norm,
                last_ratio,
                DerivativeFreeTermination::EvaluationLimit,
            ));
        }
        if !observer(&DerivativeFreeIteration {
            iteration,
            point: point.clone(),
            objective: value,
            evaluations,
            size: radius,
        }) {
            return Ok(model_based_result(
                point,
                value,
                iteration,
                evaluations,
                model_builds,
                accepted_steps,
                one_sided_stencils,
                radius,
                projected_gradient_norm,
                last_ratio,
                DerivativeFreeTermination::Cancelled,
            ));
        }

        let (gradient, curvature, used_one_sided) = interpolation_model(
            &point,
            value,
            radius,
            &box_bounds,
            &mut evaluate,
            &mut evaluations,
        )?;
        model_builds += 1;
        one_sided_stencils += used_one_sided;
        projected_gradient_norm = projected_gradient(&point, &gradient, &box_bounds)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let step = model_step(&point, &gradient, &curvature, radius, &box_bounds);
        let step_norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= options.final_radius * 0.1 {
            radius *= options.contraction;
            continue;
        }
        let predicted = -gradient
            .iter()
            .zip(&curvature)
            .zip(&step)
            .map(|((g, h), s)| g * s + 0.5 * h * s * s)
            .sum::<f64>();
        if !predicted.is_finite() || predicted <= 0. {
            radius *= options.contraction;
            continue;
        }
        let candidate = project(
            &point
                .iter()
                .zip(&step)
                .map(|(x, s)| x + s)
                .collect::<Vec<_>>(),
            &box_bounds,
        );
        let candidate_value = checked(&mut evaluate, &candidate)?;
        evaluations += 1;
        let actual = value - candidate_value;
        last_ratio = actual / predicted;
        let accepted =
            last_ratio >= options.acceptance_threshold && actual > options.objective_tolerance;
        if accepted {
            point = candidate;
            value = candidate_value;
            accepted_steps += 1;
        }
        if !accepted || last_ratio < 0.1 {
            radius *= options.contraction;
        } else if last_ratio > 0.75 && step_norm >= 0.8 * radius {
            radius = (options.expansion * radius).min(options.maximum_radius);
        }
    }
    Ok(model_based_result(
        point,
        value,
        options.max_iterations,
        evaluations,
        model_builds,
        accepted_steps,
        one_sided_stencils,
        radius,
        projected_gradient_norm,
        last_ratio,
        DerivativeFreeTermination::IterationLimit,
    ))
}

fn interpolation_model<F>(
    point: &[f64],
    center_value: f64,
    radius: f64,
    bounds: &[Bound],
    evaluate: &mut F,
    evaluations: &mut usize,
) -> Result<(Vec<f64>, Vec<f64>, usize), OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    let mut gradient = vec![0.; point.len()];
    let mut curvature = vec![0.; point.len()];
    let mut one_sided = 0;
    for i in 0..point.len() {
        let plus = radius.min(bounds[i].upper.unwrap_or(f64::INFINITY) - point[i]);
        let minus = radius.min(point[i] - bounds[i].lower.unwrap_or(f64::NEG_INFINITY));
        if plus > 1e-14 && minus > 1e-14 {
            let mut xp = point.to_vec();
            xp[i] += plus;
            let fp = checked(evaluate, &xp)?;
            let mut xm = point.to_vec();
            xm[i] -= minus;
            let fm = checked(evaluate, &xm)?;
            *evaluations += 2;
            curvature[i] =
                2. * ((fp - center_value) / plus + (fm - center_value) / minus) / (plus + minus);
            gradient[i] = (fp - center_value) / plus - 0.5 * curvature[i] * plus;
        } else {
            one_sided += 1;
            let direction = if plus > 1e-14 { 1. } else { -1. };
            let available = if direction > 0. { plus } else { minus };
            let h = available.max(1e-14);
            let half = 0.5 * h;
            let mut x1 = point.to_vec();
            x1[i] += direction * h;
            let f1 = checked(evaluate, &x1)?;
            let mut x2 = point.to_vec();
            x2[i] += direction * half;
            let f2 = checked(evaluate, &x2)?;
            *evaluations += 2;
            let signed_h = direction * h;
            curvature[i] = (f1 - 2. * f2 + center_value) / (half * half);
            gradient[i] = (f1 - center_value - 0.5 * curvature[i] * h * h) / signed_h;
        }
    }
    Ok((gradient, curvature, one_sided))
}

fn projected_gradient(point: &[f64], gradient: &[f64], bounds: &[Bound]) -> Vec<f64> {
    point
        .iter()
        .zip(gradient)
        .zip(bounds)
        .map(|((x, g), b)| {
            if (*x <= b.lower.unwrap_or(f64::NEG_INFINITY) + 1e-12 && *g > 0.)
                || (*x >= b.upper.unwrap_or(f64::INFINITY) - 1e-12 && *g < 0.)
            {
                0.
            } else {
                *g
            }
        })
        .collect()
}

fn model_step(
    point: &[f64],
    gradient: &[f64],
    curvature: &[f64],
    radius: f64,
    bounds: &[Bound],
) -> Vec<f64> {
    let mut step = gradient
        .iter()
        .zip(curvature)
        .enumerate()
        .map(|(i, (g, h))| {
            let raw = if *h > 1e-12 {
                -g / h
            } else if *g > 0. {
                -radius
            } else if *g < 0. {
                radius
            } else {
                0.
            };
            raw.max(bounds[i].lower.unwrap_or(f64::NEG_INFINITY) - point[i])
                .min(bounds[i].upper.unwrap_or(f64::INFINITY) - point[i])
        })
        .collect::<Vec<_>>();
    let norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > radius {
        for value in &mut step {
            *value *= radius / norm;
        }
    }
    let predicted = -gradient
        .iter()
        .zip(curvature)
        .zip(&step)
        .map(|((g, h), s)| g * s + 0.5 * h * s * s)
        .sum::<f64>();
    if predicted <= 0. || !predicted.is_finite() {
        step = projected_gradient(point, gradient, bounds)
            .iter()
            .map(|g| -g)
            .collect();
        let norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > radius {
            for value in &mut step {
                *value *= radius / norm;
            }
        }
        for i in 0..step.len() {
            step[i] = step[i]
                .max(bounds[i].lower.unwrap_or(f64::NEG_INFINITY) - point[i])
                .min(bounds[i].upper.unwrap_or(f64::INFINITY) - point[i]);
        }
    }
    step
}

#[allow(clippy::too_many_arguments)]
fn model_based_result(
    point: Vec<f64>,
    objective: f64,
    iterations: usize,
    evaluations: usize,
    model_builds: usize,
    accepted_steps: usize,
    one_sided_stencils: usize,
    final_radius: f64,
    final_projected_gradient_norm: f64,
    last_reduction_ratio: f64,
    termination: DerivativeFreeTermination,
) -> ModelBasedResult {
    ModelBasedResult {
        point,
        objective,
        iterations,
        evaluations,
        model_builds,
        accepted_steps,
        one_sided_stencils,
        final_radius,
        final_projected_gradient_norm,
        last_reduction_ratio,
        termination,
    }
}

pub fn minimize_pattern_search<F>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    options: PatternSearchOptions,
    evaluate: F,
) -> Result<PatternSearchResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    minimize_pattern_search_with_observer(initial, bounds, options, evaluate, |_| true)
}

pub fn minimize_pattern_search_with_observer<F, O>(
    initial: &[f64],
    bounds: Option<&[Bound]>,
    options: PatternSearchOptions,
    mut evaluate: F,
    mut observer: O,
) -> Result<PatternSearchResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
    O: FnMut(&DerivativeFreeIteration) -> bool,
{
    validate_pattern(initial, bounds, options)?;
    let box_bounds = owned_bounds(initial.len(), bounds);
    let mut point = project(initial, &box_bounds);
    let mut value = checked(&mut evaluate, &point)?;
    let mut evaluations = 1;
    let mut successful_polls = 0;
    let mut step = options.initial_step;
    for iteration in 0..options.max_iterations {
        if step <= options.minimum_step {
            return Ok(pattern_result(
                point,
                value,
                iteration,
                evaluations,
                successful_polls,
                step,
                DerivativeFreeTermination::Converged,
            ));
        }
        if evaluations >= options.max_evaluations {
            return Ok(pattern_result(
                point,
                value,
                iteration,
                evaluations,
                successful_polls,
                step,
                DerivativeFreeTermination::EvaluationLimit,
            ));
        }
        if !observer(&DerivativeFreeIteration {
            iteration,
            point: point.clone(),
            objective: value,
            evaluations,
            size: step,
        }) {
            return Ok(pattern_result(
                point,
                value,
                iteration,
                evaluations,
                successful_polls,
                step,
                DerivativeFreeTermination::Cancelled,
            ));
        }
        let mut best_point = point.clone();
        let mut best_value = value;
        'poll: for dimension in 0..point.len() {
            for sign in [1.0, -1.0] {
                if evaluations >= options.max_evaluations {
                    break 'poll;
                }
                let mut candidate = point.clone();
                candidate[dimension] += sign * step;
                candidate = project(&candidate, &box_bounds);
                if candidate == point {
                    continue;
                }
                let candidate_value = checked(&mut evaluate, &candidate)?;
                evaluations += 1;
                if candidate_value < best_value {
                    best_point = candidate;
                    best_value = candidate_value;
                }
            }
        }
        if best_value + options.objective_tolerance < value {
            point = best_point;
            value = best_value;
            successful_polls += 1;
            step *= options.expansion;
        } else {
            step *= options.contraction;
        }
    }
    Ok(pattern_result(
        point,
        value,
        options.max_iterations,
        evaluations,
        successful_polls,
        step,
        DerivativeFreeTermination::IterationLimit,
    ))
}

fn validate_nelder(
    x: &[f64],
    bounds: Option<&[Bound]>,
    o: NelderMeadOptions,
) -> Result<(), OptimizeError> {
    validate_problem(x, bounds)?;
    if o.max_iterations == 0
        || o.max_evaluations < x.len() + 1
        || !positive(o.initial_step)
        || !positive(o.point_tolerance)
        || !o.objective_tolerance.is_finite()
        || o.objective_tolerance < 0.0
        || !positive(o.reflection)
        || o.expansion <= 1.0
        || !o.expansion.is_finite()
        || !(0.0..1.0).contains(&o.contraction)
        || !(0.0..1.0).contains(&o.shrink)
    {
        Err(invalid("invalid Nelder-Mead options"))
    } else {
        Ok(())
    }
}
fn validate_pattern(
    x: &[f64],
    bounds: Option<&[Bound]>,
    o: PatternSearchOptions,
) -> Result<(), OptimizeError> {
    validate_problem(x, bounds)?;
    if o.max_iterations == 0
        || o.max_evaluations == 0
        || !positive(o.initial_step)
        || !positive(o.minimum_step)
        || o.minimum_step >= o.initial_step
        || !(0.0..1.0).contains(&o.contraction)
        || !o.expansion.is_finite()
        || o.expansion < 1.0
        || !o.objective_tolerance.is_finite()
        || o.objective_tolerance < 0.0
    {
        Err(invalid("invalid pattern-search options"))
    } else {
        Ok(())
    }
}
fn validate_model_based(
    x: &[f64],
    bounds: Option<&[Bound]>,
    o: ModelBasedOptions,
) -> Result<(), OptimizeError> {
    validate_problem(x, bounds)?;
    if o.max_iterations == 0
        || o.max_evaluations < 2 * x.len() + 2
        || !positive(o.initial_radius)
        || !positive(o.final_radius)
        || o.final_radius >= o.initial_radius
        || !positive(o.maximum_radius)
        || o.maximum_radius < o.initial_radius
        || !(0.0..1.0).contains(&o.acceptance_threshold)
        || !(0.0..1.0).contains(&o.contraction)
        || !o.expansion.is_finite()
        || o.expansion <= 1.
        || !o.objective_tolerance.is_finite()
        || o.objective_tolerance < 0.
        || bounds.is_some_and(|values| {
            values
                .iter()
                .any(|b| b.lower.is_some() && b.lower == b.upper)
        })
    {
        Err(invalid(
            "invalid model-based derivative-free options or fixed bounds",
        ))
    } else {
        Ok(())
    }
}
fn validate_problem(x: &[f64], bounds: Option<&[Bound]>) -> Result<(), OptimizeError> {
    if x.is_empty()
        || !x.iter().all(|v| v.is_finite())
        || bounds.is_some_and(|b| {
            b.len() != x.len()
                || b.iter().any(|v| {
                    v.lower.is_some_and(|x| !x.is_finite())
                        || v.upper.is_some_and(|x| !x.is_finite())
                        || v.lower.unwrap_or(f64::NEG_INFINITY) > v.upper.unwrap_or(f64::INFINITY)
                })
        })
    {
        Err(invalid(
            "invalid derivative-free problem or parameter bounds",
        ))
    } else {
        Ok(())
    }
}
fn owned_bounds(n: usize, bounds: Option<&[Bound]>) -> Vec<Bound> {
    bounds
        .map(<[Bound]>::to_vec)
        .unwrap_or_else(|| (0..n).map(|_| Bound::free()).collect())
}
fn project(x: &[f64], bounds: &[Bound]) -> Vec<f64> {
    x.iter()
        .zip(bounds)
        .map(|(x, b)| {
            x.max(b.lower.unwrap_or(f64::NEG_INFINITY))
                .min(b.upper.unwrap_or(f64::INFINITY))
        })
        .collect()
}
fn checked<F>(f: &mut F, x: &[f64]) -> Result<f64, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    let value = f(x)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("non-finite derivative-free objective"))
    }
}
fn sort_simplex(simplex: &mut [(Vec<f64>, f64)]) {
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
}
fn centroid(simplex: &[(Vec<f64>, f64)]) -> Vec<f64> {
    let mut c = vec![0.0; simplex[0].0.len()];
    for (x, _) in simplex {
        for (v, x) in c.iter_mut().zip(x) {
            *v += x;
        }
    }
    for v in &mut c {
        *v /= simplex.len() as f64;
    }
    c
}
fn affine(center: &[f64], other: &[f64], factor: f64) -> Vec<f64> {
    center
        .iter()
        .zip(other)
        .map(|(c, x)| c + factor * (c - x))
        .collect()
}
fn diameter(simplex: &[(Vec<f64>, f64)]) -> f64 {
    simplex
        .iter()
        .skip(1)
        .map(|(x, _)| {
            x.iter()
                .zip(&simplex[0].0)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max)
}
fn nelder_result(
    mut simplex: Vec<(Vec<f64>, f64)>,
    iterations: usize,
    evaluations: usize,
    termination: DerivativeFreeTermination,
) -> NelderMeadResult {
    sort_simplex(&mut simplex);
    let d = diameter(&simplex);
    NelderMeadResult {
        point: simplex[0].0.clone(),
        objective: simplex[0].1,
        iterations,
        evaluations,
        simplex_diameter: d,
        objective_spread: simplex.last().unwrap().1 - simplex[0].1,
        termination,
    }
}
fn pattern_result(
    point: Vec<f64>,
    objective: f64,
    iterations: usize,
    evaluations: usize,
    successful_polls: usize,
    final_step: f64,
    termination: DerivativeFreeTermination,
) -> PatternSearchResult {
    PatternSearchResult {
        point,
        objective,
        iterations,
        evaluations,
        successful_polls,
        final_step,
        termination,
    }
}
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nelder_mead_solves_bounded_rosenbrock() {
        let bounds = [
            Bound {
                lower: Some(-2.0),
                upper: Some(2.0),
            },
            Bound {
                lower: Some(-1.0),
                upper: Some(3.0),
            },
        ];
        let result = minimize_nelder_mead(
            &[-1.2, 1.0],
            Some(&bounds),
            NelderMeadOptions::default(),
            |x| Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)),
        )
        .unwrap();
        assert_eq!(result.termination, DerivativeFreeTermination::Converged);
        assert!((result.point[0] - 1.0).abs() < 1e-4 && (result.point[1] - 1.0).abs() < 1e-4);
        assert!(result.objective < 1e-9);
    }

    #[test]
    fn pattern_search_handles_nonsmooth_bounded_objective() {
        let bounds = [
            Bound {
                lower: Some(-1.0),
                upper: Some(3.0),
            },
            Bound {
                lower: Some(-2.0),
                upper: Some(2.0),
            },
        ];
        let result = minimize_pattern_search(
            &[-3.0, 4.0],
            Some(&bounds),
            PatternSearchOptions::default(),
            |x| Ok((x[0] - 0.75).abs() + 2.0 * (x[1] + 0.5).abs()),
        )
        .unwrap();
        assert_eq!(result.termination, DerivativeFreeTermination::Converged);
        assert!((result.point[0] - 0.75).abs() < 1e-6 && (result.point[1] + 0.5).abs() < 1e-6);
        assert!(result.successful_polls > 0);
    }

    #[test]
    fn budgets_cancellation_and_invalid_values_are_explicit() {
        let options = PatternSearchOptions {
            max_evaluations: 2,
            ..Default::default()
        };
        let limited = minimize_pattern_search(&[5.0], None, options, |x| Ok(x[0] * x[0])).unwrap();
        assert_eq!(
            limited.termination,
            DerivativeFreeTermination::EvaluationLimit
        );
        let cancelled = minimize_nelder_mead_with_observer(
            &[2.0],
            None,
            NelderMeadOptions::default(),
            |x| Ok(x[0] * x[0]),
            |_| false,
        )
        .unwrap();
        assert_eq!(cancelled.termination, DerivativeFreeTermination::Cancelled);
        assert!(
            minimize_nelder_mead(&[0.0], None, NelderMeadOptions::default(), |_| Ok(f64::NAN))
                .is_err()
        );
    }

    #[test]
    fn model_based_trust_region_solves_rotated_bounded_quadratic() {
        let bounds = [
            Bound {
                lower: Some(-2.),
                upper: Some(4.),
            },
            Bound {
                lower: Some(-3.),
                upper: Some(3.),
            },
        ];
        let result = minimize_model_based(
            &[-2., 3.],
            Some(&bounds),
            ModelBasedOptions::default(),
            |x| {
                let a = x[0] - 1.;
                let b = x[1] + 0.5;
                Ok(2. * a * a + 1.5 * a * b + 3. * b * b)
            },
        )
        .unwrap();
        assert_eq!(result.termination, DerivativeFreeTermination::Converged);
        assert!((result.point[0] - 1.).abs() < 2e-4);
        assert!((result.point[1] + 0.5).abs() < 2e-4);
        assert!(result.objective < 1e-7 && result.accepted_steps > 0 && result.model_builds > 0);
    }

    #[test]
    fn model_based_uses_one_sided_geometry_at_a_bound() {
        let bounds = [Bound {
            lower: Some(0.),
            upper: Some(4.),
        }];
        let result =
            minimize_model_based(&[0.], Some(&bounds), ModelBasedOptions::default(), |x| {
                Ok((x[0] - 2.).powi(2))
            })
            .unwrap();
        assert!((result.point[0] - 2.).abs() < 1e-5);
        assert!(result.one_sided_stencils > 0);
    }
}
