//! Matrix-free Levenberg-Marquardt using Jacobian products and CG.

use crate::{
    Bound, NonlinearLeastSquaresOptions, NonlinearLeastSquaresResult, NonlinearModel,
    NonlinearTermination, OptimizeError, RobustLoss,
};

#[derive(Debug, Clone, Copy)]
pub struct MatrixFreeLeastSquaresOptions {
    pub outer: NonlinearLeastSquaresOptions,
    pub max_krylov_iterations: usize,
    pub krylov_tolerance: f64,
}

impl Default for MatrixFreeLeastSquaresOptions {
    fn default() -> Self {
        Self {
            outer: Default::default(),
            max_krylov_iterations: 200,
            krylov_tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatrixFreeLeastSquaresResult {
    pub solution: NonlinearLeastSquaresResult,
    pub krylov_iterations: usize,
    pub jacobian_products: usize,
    pub transpose_jacobian_products: usize,
}

/// Solve a residual model without assembling its Jacobian or normal matrix.
pub fn solve_model_least_squares_matrix_free(
    model: &NonlinearModel,
    initial: &[f64],
    weights: &[f64],
    loss: RobustLoss,
    options: MatrixFreeLeastSquaresOptions,
) -> Result<MatrixFreeLeastSquaresResult, OptimizeError> {
    model.validate()?;
    validate(model, initial, weights, loss, options)?;
    let bounds = &model.bounds;
    let mut point: Vec<_> = initial
        .iter()
        .zip(bounds)
        .map(|(&x, b)| project(x, b))
        .collect();
    let mut residuals = residual_values(model, &point)?;
    let mut evaluations = 1;
    let mut cost = robust_cost(&residuals, weights, loss);
    let mut damping = options.outer.initial_damping;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut krylov_iterations = 0;
    let mut j_products = 0;
    let mut jt_products = 0;

    for iteration in 0..options.outer.max_iterations {
        let scales = scales(&residuals, weights, loss);
        let weighted_r: Vec<_> = residuals
            .iter()
            .zip(&scales)
            .map(|(r, s)| r * s * s)
            .collect();
        let gradient = model.jacobian_transpose_vector_product(&point, &weighted_r)?;
        jt_products += 1;
        let gradient_norm = infinity_norm(&projected_gradient(&point, &gradient, bounds));
        if gradient_norm <= options.outer.gradient_tolerance {
            return Ok(finish(
                point,
                residuals,
                cost,
                gradient_norm,
                iteration,
                evaluations,
                NonlinearTermination::ConvergedGradient,
                damping,
                accepted_steps,
                rejected_steps,
                krylov_iterations,
                j_products,
                jt_products,
            ));
        }
        let mut accepted = None;
        for _ in 0..options.outer.max_damping_iterations {
            let rhs: Vec<_> = gradient.iter().map(|g| -g).collect();
            let (step, its, jp, jtp) = cg_normal(
                model,
                &point,
                &scales,
                damping,
                &rhs,
                options.max_krylov_iterations,
                options.krylov_tolerance,
            )?;
            krylov_iterations += its;
            j_products += jp;
            jt_products += jtp;
            let candidate: Vec<_> = point
                .iter()
                .zip(&step)
                .zip(bounds)
                .map(|((&x, &s), b)| project(x + s, b))
                .collect();
            let actual_step: Vec<_> = candidate.iter().zip(&point).map(|(a, b)| a - b).collect();
            if candidate.iter().all(|v| v.is_finite()) {
                evaluations += 1;
                if let Ok(next_residuals) = residual_values(model, &candidate) {
                    if next_residuals.len() != residuals.len() {
                        return Err(invalid("residual count must remain constant"));
                    }
                    let next_cost = robust_cost(&next_residuals, weights, loss);
                    let js = model.jacobian_vector_product(&point, &actual_step)?;
                    j_products += 1;
                    let quadratic: f64 = js.iter().zip(&scales).map(|(v, s)| (s * v).powi(2)).sum();
                    let predicted = -dot(&gradient, &actual_step) - 0.5 * quadratic;
                    let ratio = if predicted > 0.0 {
                        (cost - next_cost) / predicted
                    } else {
                        f64::NEG_INFINITY
                    };
                    if ratio > 0.0 && next_cost < cost {
                        accepted = Some((candidate, next_residuals, next_cost, norm(&actual_step)));
                        if ratio > 0.75 {
                            damping =
                                (damping * options.outer.damping_decrease).max(f64::MIN_POSITIVE);
                        } else if ratio < 0.25 {
                            damping *= options.outer.damping_increase;
                        }
                        accepted_steps += 1;
                        break;
                    }
                }
            }
            damping *= options.outer.damping_increase;
            rejected_steps += 1;
            if !damping.is_finite() {
                break;
            }
        }
        let Some((next_point, next_residuals, next_cost, step_norm)) = accepted else {
            return Ok(finish(
                point,
                residuals,
                cost,
                gradient_norm,
                iteration,
                evaluations,
                NonlinearTermination::DampingLimit,
                damping,
                accepted_steps,
                rejected_steps,
                krylov_iterations,
                j_products,
                jt_products,
            ));
        };
        let cost_change = cost - next_cost;
        point = next_point;
        residuals = next_residuals;
        cost = next_cost;
        if step_norm <= options.outer.step_tolerance * (1.0 + norm(&point)) {
            return final_at(
                model,
                point,
                residuals,
                cost,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedStep,
                damping,
                accepted_steps,
                rejected_steps,
                krylov_iterations,
                j_products,
                jt_products,
                weights,
                loss,
            );
        }
        if cost_change <= options.outer.cost_tolerance * (1.0 + cost) {
            return final_at(
                model,
                point,
                residuals,
                cost,
                iteration + 1,
                evaluations,
                NonlinearTermination::ConvergedObjective,
                damping,
                accepted_steps,
                rejected_steps,
                krylov_iterations,
                j_products,
                jt_products,
                weights,
                loss,
            );
        }
    }
    final_at(
        model,
        point,
        residuals,
        cost,
        options.outer.max_iterations,
        evaluations,
        NonlinearTermination::IterationLimit,
        damping,
        accepted_steps,
        rejected_steps,
        krylov_iterations,
        j_products,
        jt_products,
        weights,
        loss,
    )
}

fn cg_normal(
    model: &NonlinearModel,
    x: &[f64],
    scales: &[f64],
    lambda: f64,
    b: &[f64],
    max: usize,
    tol: f64,
) -> Result<(Vec<f64>, usize, usize, usize), OptimizeError> {
    let mut z = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rr = dot(&r, &r);
    let target = tol * norm(b).max(1.0);
    let mut jp = 0;
    let mut jtp = 0;
    if rr.sqrt() <= target {
        return Ok((z, 0, jp, jtp));
    }
    for k in 0..max {
        let j = model.jacobian_vector_product(x, &p)?;
        jp += 1;
        let weighted: Vec<_> = j.iter().zip(scales).map(|(v, s)| v * s * s).collect();
        let mut ap = model.jacobian_transpose_vector_product(x, &weighted)?;
        jtp += 1;
        for i in 0..ap.len() {
            ap[i] += lambda * p[i];
        }
        let denom = dot(&p, &ap);
        if !denom.is_finite() || denom <= 0.0 {
            return Ok((z, k + 1, jp, jtp));
        }
        let alpha = rr / denom;
        for i in 0..z.len() {
            z[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let next = dot(&r, &r);
        if next.sqrt() <= target {
            return Ok((z, k + 1, jp, jtp));
        }
        let beta = next / rr;
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rr = next;
    }
    Ok((z, max, jp, jtp))
}

fn final_at(
    model: &NonlinearModel,
    point: Vec<f64>,
    residuals: Vec<f64>,
    cost: f64,
    it: usize,
    ev: usize,
    term: NonlinearTermination,
    damping: f64,
    accepted: usize,
    rejected: usize,
    ki: usize,
    jp: usize,
    jtp: usize,
    weights: &[f64],
    loss: RobustLoss,
) -> Result<MatrixFreeLeastSquaresResult, OptimizeError> {
    let s = scales(&residuals, weights, loss);
    let wr: Vec<_> = residuals.iter().zip(s).map(|(r, s)| r * s * s).collect();
    let g = model.jacobian_transpose_vector_product(&point, &wr)?;
    let gradient_norm = infinity_norm(&projected_gradient(&point, &g, &model.bounds));
    Ok(finish(
        point,
        residuals,
        cost,
        gradient_norm,
        it,
        ev,
        term,
        damping,
        accepted,
        rejected,
        ki,
        jp,
        jtp + 1,
    ))
}
fn finish(
    point: Vec<f64>,
    residuals: Vec<f64>,
    cost: f64,
    gn: f64,
    it: usize,
    ev: usize,
    term: NonlinearTermination,
    damping: f64,
    accepted: usize,
    rejected: usize,
    ki: usize,
    jp: usize,
    jtp: usize,
) -> MatrixFreeLeastSquaresResult {
    MatrixFreeLeastSquaresResult {
        solution: NonlinearLeastSquaresResult {
            point,
            residuals,
            cost,
            gradient_norm: gn,
            iterations: it,
            evaluations: ev,
            termination: term,
            final_damping: damping,
            accepted_steps: accepted,
            rejected_steps: rejected,
        },
        krylov_iterations: ki,
        jacobian_products: jp,
        transpose_jacobian_products: jtp,
    }
}
fn residual_values(m: &NonlinearModel, x: &[f64]) -> Result<Vec<f64>, OptimizeError> {
    if m.residuals.is_empty() || x.len() != m.parameter_count {
        return Err(invalid("model does not contain compatible residuals"));
    }
    m.residuals
        .iter()
        .map(|e| e.evaluate_directional(x, &vec![0.0; x.len()]).map(|v| v.0))
        .collect()
}
fn scales(r: &[f64], w: &[f64], loss: RobustLoss) -> Vec<f64> {
    r.iter()
        .enumerate()
        .map(|(i, &v)| {
            let rw = match loss {
                RobustLoss::Squared => 1.0,
                RobustLoss::Huber { scale } => {
                    if v.abs() <= scale || v == 0.0 {
                        1.0
                    } else {
                        scale / v.abs()
                    }
                }
                RobustLoss::Cauchy { scale } => 1.0 / (1.0 + (v / scale).powi(2)),
            };
            ((if w.is_empty() { 1.0 } else { w[i] }) * rw).sqrt()
        })
        .collect()
}
fn robust_cost(r: &[f64], w: &[f64], loss: RobustLoss) -> f64 {
    r.iter()
        .enumerate()
        .map(|(i, &v)| {
            let rho = match loss {
                RobustLoss::Squared => 0.5 * v * v,
                RobustLoss::Huber { scale } => {
                    if v.abs() <= scale {
                        0.5 * v * v
                    } else {
                        scale * (v.abs() - 0.5 * scale)
                    }
                }
                RobustLoss::Cauchy { scale } => {
                    0.5 * scale * scale * (1.0 + (v / scale).powi(2)).ln()
                }
            };
            (if w.is_empty() { 1.0 } else { w[i] }) * rho
        })
        .sum()
}
fn validate(
    m: &NonlinearModel,
    x: &[f64],
    w: &[f64],
    loss: RobustLoss,
    o: MatrixFreeLeastSquaresOptions,
) -> Result<(), OptimizeError> {
    if x.len() != m.parameter_count || x.is_empty() || !x.iter().all(|v| v.is_finite()) {
        return Err(invalid("initial point must be finite and match the model"));
    }
    if !w.is_empty()
        && (w.len() != m.residuals.len() || w.iter().any(|v| !v.is_finite() || *v < 0.0))
    {
        return Err(invalid("weights must match residuals and be non-negative"));
    }
    let scale = match loss {
        RobustLoss::Squared => 1.0,
        RobustLoss::Huber { scale } | RobustLoss::Cauchy { scale } => scale,
    };
    let q = o.outer;
    if q.max_iterations == 0
        || q.max_damping_iterations == 0
        || o.max_krylov_iterations == 0
        || !o.krylov_tolerance.is_finite()
        || o.krylov_tolerance <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
        || !q.gradient_tolerance.is_finite()
        || q.gradient_tolerance <= 0.0
        || !q.step_tolerance.is_finite()
        || q.step_tolerance <= 0.0
        || !q.cost_tolerance.is_finite()
        || q.cost_tolerance <= 0.0
        || !q.initial_damping.is_finite()
        || q.initial_damping <= 0.0
        || !q.damping_increase.is_finite()
        || q.damping_increase <= 1.0
        || !q.damping_decrease.is_finite()
        || !(0.0..1.0).contains(&q.damping_decrease)
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid matrix-free least-squares options",
        ));
    }
    Ok(())
}
fn project(x: f64, b: &Bound) -> f64 {
    b.lower
        .map_or(x, |l| x.max(l))
        .min(b.upper.unwrap_or(f64::INFINITY))
}
fn projected_gradient(x: &[f64], g: &[f64], b: &[Bound]) -> Vec<f64> {
    x.iter()
        .zip(g)
        .zip(b)
        .map(|((&x, &g), b)| {
            if (b.lower.is_some_and(|l| x <= l) && g > 0.0)
                || (b.upper.is_some_and(|u| x >= u) && g < 0.0)
            {
                0.0
            } else {
                g
            }
        })
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0, f64::max)
}
fn invalid(s: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(s.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NonlinearExpression as E, NonlinearNode as N};
    #[test]
    fn matrix_free_curve_fit() {
        let r1 = E {
            nodes: vec![N::Parameter(0), N::Constant(2.0), N::Subtract(0, 1)],
            output: 2,
        };
        let r2 = E {
            nodes: vec![
                N::Parameter(0),
                N::Parameter(1),
                N::Add(0, 1),
                N::Constant(5.0),
                N::Subtract(2, 3),
            ],
            output: 4,
        };
        let m = NonlinearModel {
            parameter_count: 2,
            bounds: vec![Bound::free(); 2],
            objective: None,
            residuals: vec![r1, r2],
        };
        let out = solve_model_least_squares_matrix_free(
            &m,
            &[0.0, 0.0],
            &[],
            RobustLoss::Squared,
            Default::default(),
        )
        .unwrap();
        assert!((out.solution.point[0] - 2.0).abs() < 1e-6);
        assert!((out.solution.point[1] - 3.0).abs() < 1e-6);
        assert!(out.jacobian_products > 0);
    }
}
