//! Dense UOBYQA reference variant. Rebuilds the Lagrange basis after each
//! replacement, and solves the quadratic subproblems in an eigenbasis.

use super::{
    EvaluationStop, Oracle, PowellUnconstrainedOptions, PowellUnconstrainedTermination as Stop,
    Sample,
};
use crate::OptimizeError;
use basin::{DenseMatrix, MatTransposeVec, MatVec, SymmetricEigen};

pub(super) fn run<F>(
    initial: &[f64],
    options: PowellUnconstrainedOptions,
    oracle: &mut Oracle<F>,
) -> Result<Stop, EvaluationStop>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    let mut resolution = options.initial_radius;
    let mut radius = resolution;
    let mut samples = stencil(initial, radius, oracle)?;

    for _ in 0..options.max_iterations {
        if oracle.calls == oracle.budget {
            return Ok(Stop::EvaluationLimit);
        }
        let mut center = best_index(&samples);
        let mut model = match Model::fit(&samples, center, radius) {
            Some(model) => model,
            None => {
                samples = stencil(&samples[center].point, radius, oracle)?;
                center = best_index(&samples);
                let Some(model) = Model::fit(&samples, center, radius) else {
                    return Ok(Stop::NumericalFailure);
                };
                model
            }
        };

        let Some(step) = step_in_ball(
            &model.coefficients,
            initial.len(),
            radius / model.scale,
            1.0,
        ) else {
            return Ok(Stop::NumericalFailure);
        };
        let length = norm(&step) * model.scale;
        let Some((point, terms)) = model.trial(&samples[center], &step) else {
            return Ok(Stop::NumericalFailure);
        };
        let prediction = -dot(&model.coefficients, &terms);
        if !prediction.is_finite() {
            return Ok(Stop::NumericalFailure);
        }

        let short = length < 0.5 * resolution;
        if !short && prediction > 0.0 && point != samples[center].point {
            let old_value = samples[center].value;
            let trial = evaluate_trial(&samples, point.clone(), oracle)?;
            let ratio = (old_value - trial.value) / prediction;
            let improves = trial.value < old_value;
            let replacement = model.replacement(&terms, center, resolution, improves);
            if let Some(index) = replacement {
                samples[index] = trial;
            } else if improves {
                // Preserve an improving point even when its Lagrange values
                // are too small for a stable one-point exchange.
                samples = stencil(&trial.point, radius, oracle)?;
            }
            radius = if ratio < 0.1 {
                0.5 * length
            } else if ratio <= 0.7 {
                (0.5 * radius).max(length)
            } else {
                (0.5 * radius).max(2.0 * length)
            }
            .min(f64::MAX.sqrt());
            if radius <= 1.5 * resolution {
                radius = resolution;
            }
            if improves {
                continue;
            }
            if replacement.is_some() {
                // The exchange changed the basis used to choose a geometry step.
                let Some(updated) = Model::fit(&samples, center, radius) else {
                    samples = stencil(&samples[center].point, radius, oracle)?;
                    continue;
                };
                model = updated;
            }
        }

        let farthest = (0..samples.len())
            .filter(|&i| i != center)
            .max_by(|&i, &j| model.distances[i].total_cmp(&model.distances[j]))
            .expect("full quadratic interpolation has more than one point");
        if model.distances[farthest] > 2.0 * resolution {
            let n = initial.len();
            let coefficients = &model.lagrange[farthest];
            let geometry_radius = (0.5 * radius)
                .min(0.1 * model.distances[farthest])
                .max(resolution);
            let scaled_radius = geometry_radius / model.scale;
            let Some(step) = [1.0, -1.0]
                .into_iter()
                .filter_map(|sign| step_in_ball(coefficients, n, scaled_radius, sign))
                .max_by(|a, b| {
                    dot(coefficients, &features(a))
                        .abs()
                        .total_cmp(&dot(coefficients, &features(b)).abs())
                })
            else {
                return Ok(Stop::NumericalFailure);
            };
            let Some((point, terms)) = model.trial(&samples[center], &step) else {
                return Ok(Stop::NumericalFailure);
            };
            // The replacement determinant is the Lagrange value at the trial.
            if dot(coefficients, &terms).abs() > 1e-8
                && samples.iter().all(|sample| sample.point != point)
            {
                samples[farthest] = oracle.sample(point)?;
                continue;
            }
            samples = stencil(&samples[center].point, radius, oracle)?;
            continue;
        }
        if radius > resolution {
            radius = (0.5 * radius).max(resolution);
        } else {
            if resolution == options.final_radius {
                if short && prediction > 0.0 && point != samples[center].point {
                    // The last short step can improve the point even though
                    // it is too small to start another trust-region iteration.
                    evaluate_trial(&samples, point, oracle)?;
                }
                return Ok(Stop::RadiusReached);
            }
            let relative = resolution / options.final_radius;
            radius = 0.5 * resolution;
            resolution = if relative <= 16.0 {
                options.final_radius
            } else if relative <= 250.0 {
                relative.sqrt() * options.final_radius
            } else {
                0.1 * resolution
            };
            radius = radius.max(resolution);
        }
    }

    Ok(if oracle.calls == oracle.budget {
        Stop::EvaluationLimit
    } else {
        Stop::IterationLimit
    })
}

fn best_index(samples: &[Sample]) -> usize {
    (0..samples.len())
        .min_by(|&i, &j| samples[i].value.total_cmp(&samples[j].value))
        .unwrap()
}

fn stencil<F>(
    center: &[f64],
    radius: f64,
    oracle: &mut Oracle<F>,
) -> Result<Vec<Sample>, EvaluationStop>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    let n = center.len();
    let mut samples = Vec::with_capacity((n + 1) * (n + 2) / 2);
    let mut signs = vec![1.0; n];
    samples.push(evaluate_trial(&samples, center.to_vec(), oracle)?);
    for (i, sign) in signs.iter_mut().enumerate() {
        let mut plus = center.to_vec();
        plus[i] += radius;
        let mut minus = center.to_vec();
        minus[i] -= radius;
        if plus[i] == center[i] || minus[i] == center[i] {
            return Err(EvaluationStop::Numerical);
        }
        let positive = evaluate_trial(&samples, plus, oracle)?;
        let negative = evaluate_trial(&samples, minus, oracle)?;
        if negative.value < positive.value {
            *sign = -1.0;
        }
        samples.push(positive);
        samples.push(negative);
    }
    for i in 0..n {
        for j in 0..i {
            let mut point = center.to_vec();
            point[i] += signs[i] * radius;
            point[j] += signs[j] * radius;
            samples.push(evaluate_trial(&samples, point, oracle)?);
        }
    }
    Ok(samples)
}

fn evaluate_trial<F>(
    samples: &[Sample],
    point: Vec<f64>,
    oracle: &mut Oracle<F>,
) -> Result<Sample, EvaluationStop>
where
    F: FnMut(&[f64]) -> Result<f64, OptimizeError>,
{
    // A repeated interpolation point already has a value for this run's
    // deterministic objective. Never charge the budget for a cache hit.
    match samples.iter().find(|sample| sample.point == point) {
        Some(sample) => Ok(Sample {
            point,
            value: sample.value,
        }),
        None => oracle.sample(point),
    }
}

struct Model {
    coefficients: Vec<f64>,
    // All noncenter Lagrange polynomials have zero constant term.
    lagrange: Vec<Vec<f64>>,
    distances: Vec<f64>,
    scale: f64,
}

impl Model {
    fn fit(samples: &[Sample], center: usize, radius: f64) -> Option<Self> {
        let origin = &samples[center];
        let offsets: Vec<Vec<f64>> = samples
            .iter()
            .map(|sample| {
                sample
                    .point
                    .iter()
                    .zip(&origin.point)
                    .map(|(x, base)| x - base)
                    .collect()
            })
            .collect();
        let distances: Vec<_> = offsets.iter().map(|d| norm(d)).collect();
        let scale = distances.iter().copied().fold(radius, f64::max);
        if !scale.is_finite() {
            return None;
        }
        let fit_indices: Vec<_> = (0..samples.len()).filter(|&i| i != center).collect();
        let design = fit_indices
            .iter()
            .map(|&i| {
                features(
                    &offsets[i]
                        .iter()
                        .map(|value| value / scale)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let inverse = invert(design)?;
        let mut lagrange = vec![Vec::new(); samples.len()];
        let mut coefficients = vec![0.0; fit_indices.len()];
        for (column, &i) in fit_indices.iter().enumerate() {
            let value = samples[i].value - origin.value;
            lagrange[i] = inverse.iter().map(|row| row[column]).collect();
            for (coefficient, &weight) in coefficients.iter_mut().zip(&lagrange[i]) {
                *coefficient = weight.mul_add(value, *coefficient);
            }
        }
        coefficients.iter().all(|v| v.is_finite()).then_some(Self {
            coefficients,
            lagrange,
            distances,
            scale,
        })
    }

    fn trial(&self, origin: &Sample, step: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let point: Vec<_> = origin
            .point
            .iter()
            .zip(step)
            .map(|(&x, &d)| self.scale.mul_add(d, x))
            .collect();
        if point.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let rounded: Vec<_> = point
            .iter()
            .zip(&origin.point)
            .map(|(x, base)| (x - base) / self.scale)
            .collect();
        Some((point, features(&rounded)))
    }

    fn replacement(
        &self,
        terms: &[f64],
        center: usize,
        radius: f64,
        improves: bool,
    ) -> Option<usize> {
        self.lagrange
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != center)
            .filter_map(|(i, polynomial)| {
                let value = dot(terms, polynomial).abs();
                if value <= 1e-8 {
                    return None;
                }
                let weight = (self.distances[i] / radius).max(1.0).powi(3);
                let score = value * weight;
                (improves || score > 1.0).then_some((i, score))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }
}

// Linear terms, half the squared terms, then mixed terms.
fn features(step: &[f64]) -> Vec<f64> {
    let n = step.len();
    let mut terms = Vec::with_capacity(n * (n + 3) / 2);
    terms.extend_from_slice(step);
    terms.extend(step.iter().map(|d| 0.5 * d * d));
    for (i, &di) in step.iter().enumerate() {
        terms.extend(step[..i].iter().map(|dj| di * dj));
    }
    terms
}

// One pivoted elimination obtains all Lagrange polynomials. The design
// coordinates are normalized before this absolute pivot check.
fn invert(mut matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    for (i, row) in matrix.iter_mut().enumerate() {
        row.resize(2 * n, 0.0);
        row[n + i] = 1.0;
    }
    for i in 0..n {
        let pivot = (i..n).max_by(|&a, &b| matrix[a][i].abs().total_cmp(&matrix[b][i].abs()))?;
        matrix.swap(i, pivot);
        let divisor = matrix[i][i];
        if divisor.abs() <= 32.0 * f64::EPSILON * n as f64 || !divisor.is_finite() {
            return None;
        }
        for entry in &mut matrix[i][i..] {
            *entry /= divisor;
        }
        let row = matrix[i][i..].to_vec();
        for (j, other) in matrix.iter_mut().enumerate() {
            if j != i {
                let factor = other[i];
                for (entry, pivot) in other[i..].iter_mut().zip(&row) {
                    *entry = (-factor).mul_add(*pivot, *entry);
                }
            }
        }
    }
    Some(matrix.into_iter().map(|mut row| row.split_off(n)).collect())
}

// Minimize the quadratic in a ball by solving the secular equation. At a
// singular endpoint, complete the boundary step in the leftmost eigenspace.
fn step_in_ball(coefficients: &[f64], n: usize, radius: f64, sign: f64) -> Option<Vec<f64>> {
    let scale = coefficients.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Some(vec![0.0; n]);
    }
    let g: Vec<_> = coefficients[..n]
        .iter()
        .map(|v| sign * (v / scale))
        .collect();
    let mut hessian = vec![0.0; n * n];
    let mut k = 2 * n;
    for i in 0..n {
        hessian[i * n + i] = sign * (coefficients[n + i] / scale);
        for j in 0..i {
            let value = sign * (coefficients[k] / scale);
            hessian[i * n + j] = value;
            hessian[j * n + i] = value;
            k += 1;
        }
    }
    let (basis, eigenvalues) = DenseMatrix::from_row_slice(n, n, &hessian)
        .try_eigh()
        .ok()?;
    let gamma = basis.mat_transpose_vec(&g);
    let left = (0..n).min_by(|&i, &j| eigenvalues[i].total_cmp(&eigenvalues[j]))?;
    let floor = (-eigenvalues[left]).max(0.0);
    let tolerance = 32.0 * f64::EPSILON * (1.0 + floor);
    let regular = |shift: f64| -> Vec<f64> {
        gamma
            .iter()
            .zip(&eigenvalues)
            .map(|(&g, &lambda)| -g / (lambda + shift))
            .collect()
    };
    let mut endpoint = vec![0.0; n];
    let mut compatible = true;
    for i in 0..n {
        let denominator = eigenvalues[i] + floor;
        if denominator > tolerance {
            endpoint[i] = -gamma[i] / denominator;
        } else {
            compatible &= gamma[i].abs() <= tolerance * norm(&gamma);
        }
    }
    let length = norm(&endpoint);
    if compatible && length <= radius {
        if floor > 0.0 {
            endpoint[left] = radius * (1.0 - (length / radius).powi(2)).max(0.0).sqrt();
        }
        return Some(basis.matvec(&endpoint));
    }

    let mut lower = floor;
    let mut upper = floor + (norm(&gamma) / radius).max(tolerance);
    if !upper.is_finite() {
        return None;
    }
    let mut step = regular(upper);
    // At the analytic upper bound the computed norm can overshoot by
    // roundoff. Expand once before bisecting to preserve feasibility.
    if norm(&step) > radius {
        upper += upper - floor;
        step = regular(upper);
    }
    for _ in 0..80 {
        let middle = lower + 0.5 * (upper - lower);
        if middle == lower || middle == upper {
            break;
        }
        let candidate = regular(middle);
        if norm(&candidate) <= radius {
            upper = middle;
            step = candidate;
        } else {
            lower = middle;
        }
    }
    step.iter()
        .all(|d| d.is_finite())
        .then(|| basis.matvec(&step))
}

fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |norm, &value| norm.hypot(value))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_region_hard_case_uses_negative_curvature_orthogonal_to_gradient() {
        // q(x,y) = -x² + y² + y. At lambda = 2, y = -1/4
        // and the leftmost eigendirection fills the remaining radius.
        let coefficients = [0.0, 1.0, -2.0, 2.0, 0.0];
        let step = step_in_ball(&coefficients, 2, 1.0, 1.0).unwrap();
        assert!((norm(&step) - 1.0).abs() < 1e-12);
        assert!((step[1] + 0.25).abs() < 1e-12);
        assert!((dot(&coefficients, &features(&step)) + 1.125).abs() < 1e-12);
    }
}
