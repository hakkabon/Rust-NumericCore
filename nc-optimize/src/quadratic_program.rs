//! Convex quadratic programming with sparse linear constraints.

use crate::{Bound, OptimizeError};
use nc_sparse::CsrMatrix;

/// Minimize `0.5 x'Qx + c'x + constant` subject to sparse row and variable bounds.
#[derive(Debug, Clone)]
pub struct QuadraticProblem {
    pub quadratic: Vec<Vec<f64>>,
    pub linear: Vec<f64>,
    pub constant: f64,
    pub constraints: CsrMatrix<f64>,
    pub row_bounds: Vec<Bound>,
    pub variable_bounds: Vec<Bound>,
}

impl QuadraticProblem {
    pub fn validate(&self, tolerance: f64) -> Result<(), OptimizeError> {
        let n = self.linear.len();
        if n == 0
            || self.quadratic.len() != n
            || self.quadratic.iter().any(|row| row.len() != n)
            || self.constraints.cols() != n
            || self.constraints.rows() != self.row_bounds.len()
            || self.variable_bounds.len() != n
        {
            return Err(invalid(
                "QP objective, matrix, and bounds dimensions do not agree",
            ));
        }
        if !tolerance.is_finite()
            || tolerance < 0.0
            || !self.constant.is_finite()
            || !self.linear.iter().all(|v| v.is_finite())
            || !self.quadratic.iter().flatten().all(|v| v.is_finite())
            || !self
                .constraints
                .iter_entries()
                .all(|(_, _, v)| v.is_finite())
        {
            return Err(invalid(
                "QP coefficients and validation tolerance must be finite",
            ));
        }
        for i in 0..n {
            for j in 0..i {
                let scale = 1.0_f64
                    .max(self.quadratic[i][j].abs())
                    .max(self.quadratic[j][i].abs());
                if (self.quadratic[i][j] - self.quadratic[j][i]).abs() > tolerance * scale {
                    return Err(invalid("QP Hessian must be symmetric"));
                }
            }
        }
        for bound in self.row_bounds.iter().chain(&self.variable_bounds) {
            validate_bound(bound)?;
        }
        let shifted: Vec<Vec<f64>> = self
            .quadratic
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| if i == j { v + tolerance.max(1e-14) } else { v })
                    .collect()
            })
            .collect();
        if cholesky(&shifted).is_none() {
            return Err(invalid("QP Hessian must be positive semidefinite"));
        }
        Ok(())
    }
    pub fn objective(&self, x: &[f64]) -> f64 {
        self.constant
            + dot(&self.linear, x)
            + 0.5
                * x.iter()
                    .enumerate()
                    .map(|(i, &xi)| xi * dot(&self.quadratic[i], x))
                    .sum::<f64>()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuadraticOptions {
    pub max_iterations: usize,
    pub rho: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub convexity_tolerance: f64,
}
impl Default for QuadraticOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            rho: 1.0,
            absolute_tolerance: 1e-7,
            relative_tolerance: 1e-6,
            convexity_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuadraticWarmStart {
    pub primal: Vec<f64>,
    pub row_dual: Vec<f64>,
    pub variable_dual: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadraticTermination {
    Converged,
    IterationLimit,
    NumericalFailure,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct QuadraticResult {
    pub point: Vec<f64>,
    pub objective: f64,
    pub row_activity: Vec<f64>,
    pub row_dual: Vec<f64>,
    pub variable_dual: Vec<f64>,
    pub iterations: usize,
    pub termination: QuadraticTermination,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub stationarity_norm: f64,
    pub maximum_row_violation: f64,
    pub maximum_variable_violation: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QuadraticIteration {
    pub iteration: usize,
    pub objective: f64,
    pub primal_residual: f64,
    pub dual_residual: f64,
}

pub fn solve_convex_qp(
    problem: &QuadraticProblem,
    options: QuadraticOptions,
) -> Result<QuadraticResult, OptimizeError> {
    solve_convex_qp_with_observer(problem, options, None, |_| true)
}
pub fn solve_convex_qp_warm(
    problem: &QuadraticProblem,
    options: QuadraticOptions,
    warm: &QuadraticWarmStart,
) -> Result<QuadraticResult, OptimizeError> {
    solve_convex_qp_with_observer(problem, options, Some(warm), |_| true)
}
pub fn solve_convex_qp_with_observer<O>(
    problem: &QuadraticProblem,
    options: QuadraticOptions,
    warm: Option<&QuadraticWarmStart>,
    mut observer: O,
) -> Result<QuadraticResult, OptimizeError>
where
    O: FnMut(QuadraticIteration) -> bool,
{
    validate_options(options)?;
    problem.validate(options.convexity_tolerance)?;
    let n = problem.linear.len();
    let m = problem.row_bounds.len();
    let p = m + n;
    let mut c = vec![vec![0.0; n]; p];
    for (row, col, value) in problem.constraints.iter_entries() {
        c[row][col] += value;
    }
    for j in 0..n {
        c[m + j][j] = 1.0
    }
    let bounds: Vec<Bound> = problem
        .row_bounds
        .iter()
        .chain(&problem.variable_bounds)
        .copied()
        .collect();
    let mut system = problem.quadratic.clone();
    for i in 0..n {
        for j in 0..n {
            system[i][j] += options.rho * (0..p).map(|r| c[r][i] * c[r][j]).sum::<f64>();
        }
    }
    let Some(factor) = cholesky(&system) else {
        return Ok(failure(
            problem,
            vec![0.0; n],
            0,
            QuadraticTermination::NumericalFailure,
        ));
    };
    let (mut x, mut y) = if let Some(w) = warm {
        if w.primal.len() != n
            || w.row_dual.len() != m
            || w.variable_dual.len() != n
            || !w
                .primal
                .iter()
                .chain(&w.row_dual)
                .chain(&w.variable_dual)
                .all(|v| v.is_finite())
        {
            return Err(invalid("QP warm start dimensions or values are invalid"));
        }
        (
            w.primal.clone(),
            w.row_dual
                .iter()
                .chain(&w.variable_dual)
                .map(|v| v / options.rho)
                .collect(),
        )
    } else {
        (vec![0.0; n], vec![0.0; p])
    };
    let mut cx = matvec(&c, &x);
    let mut z: Vec<f64> = cx
        .iter()
        .zip(&y)
        .zip(&bounds)
        .map(|((&v, &d), b)| project(v + d, b))
        .collect();
    let mut last_primal = f64::INFINITY;
    let mut last_dual = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        let rhs: Vec<f64> = (0..n)
            .map(|j| {
                -problem.linear[j]
                    + options.rho * (0..p).map(|r| c[r][j] * (z[r] - y[r])).sum::<f64>()
            })
            .collect();
        x = solve_cholesky(&factor, &rhs);
        if !x.iter().all(|v| v.is_finite()) {
            return Ok(failure(
                problem,
                x,
                iteration,
                QuadraticTermination::NumericalFailure,
            ));
        }
        cx = matvec(&c, &x);
        let old_z = z.clone();
        z = cx
            .iter()
            .zip(&y)
            .zip(&bounds)
            .map(|((&v, &d), b)| project(v + d, b))
            .collect();
        for r in 0..p {
            y[r] += cx[r] - z[r]
        }
        last_primal = inf_norm(&cx.iter().zip(&z).map(|(a, b)| a - b).collect::<Vec<_>>());
        let z_change: Vec<f64> = z.iter().zip(&old_z).map(|(a, b)| a - b).collect();
        last_dual = options.rho * inf_norm(&transpose_matvec(&c, &z_change));
        let eps_primal = options.absolute_tolerance
            + options.relative_tolerance * inf_norm(&cx).max(inf_norm(&z));
        let lambda: Vec<f64> = y.iter().map(|v| options.rho * v).collect();
        let qx = matvec(&problem.quadratic, &x);
        let ctl = transpose_matvec(&c, &lambda);
        let eps_dual = options.absolute_tolerance
            + options.relative_tolerance
                * inf_norm(&qx)
                    .max(inf_norm(&ctl))
                    .max(inf_norm(&problem.linear));
        if !observer(QuadraticIteration {
            iteration,
            objective: problem.objective(&x),
            primal_residual: last_primal,
            dual_residual: last_dual,
        }) {
            return Ok(make_result(
                problem,
                x,
                &lambda,
                iteration,
                QuadraticTermination::Cancelled,
                last_primal,
                last_dual,
            ));
        }
        if last_primal <= eps_primal && last_dual <= eps_dual {
            return Ok(make_result(
                problem,
                x,
                &lambda,
                iteration,
                QuadraticTermination::Converged,
                last_primal,
                last_dual,
            ));
        }
    }
    let lambda: Vec<f64> = y.iter().map(|v| options.rho * v).collect();
    Ok(make_result(
        problem,
        x,
        &lambda,
        options.max_iterations,
        QuadraticTermination::IterationLimit,
        last_primal,
        last_dual,
    ))
}

fn make_result(
    problem: &QuadraticProblem,
    x: Vec<f64>,
    lambda: &[f64],
    iterations: usize,
    termination: QuadraticTermination,
    primal_residual: f64,
    dual_residual: f64,
) -> QuadraticResult {
    let m = problem.row_bounds.len();
    let row_activity = problem
        .constraints
        .spmv(&x)
        .unwrap_or_else(|_| vec![f64::NAN; m]);
    let qx = matvec(&problem.quadratic, &x);
    let mut stationarity: Vec<f64> = qx.iter().zip(&problem.linear).map(|(a, b)| a + b).collect();
    let mut combined = vec![vec![0.0; x.len()]; m + x.len()];
    for (r, c, v) in problem.constraints.iter_entries() {
        combined[r][c] += v
    }
    for j in 0..x.len() {
        combined[m + j][j] = 1.0
    }
    let ctl = transpose_matvec(&combined, lambda);
    for j in 0..x.len() {
        stationarity[j] += ctl[j]
    }
    QuadraticResult {
        objective: problem.objective(&x),
        maximum_row_violation: max_violation(&row_activity, &problem.row_bounds),
        maximum_variable_violation: max_violation(&x, &problem.variable_bounds),
        stationarity_norm: inf_norm(&stationarity),
        point: x,
        row_activity,
        row_dual: lambda[..m].to_vec(),
        variable_dual: lambda[m..].to_vec(),
        iterations,
        termination,
        primal_residual,
        dual_residual,
    }
}
fn failure(
    problem: &QuadraticProblem,
    x: Vec<f64>,
    iterations: usize,
    termination: QuadraticTermination,
) -> QuadraticResult {
    make_result(
        problem,
        x,
        &vec![0.0; problem.row_bounds.len() + problem.linear.len()],
        iterations,
        termination,
        f64::INFINITY,
        f64::INFINITY,
    )
}
fn validate_options(o: QuadraticOptions) -> Result<(), OptimizeError> {
    if o.max_iterations == 0
        || !o.rho.is_finite()
        || o.rho <= 0.0
        || !o.absolute_tolerance.is_finite()
        || o.absolute_tolerance <= 0.0
        || !o.relative_tolerance.is_finite()
        || o.relative_tolerance <= 0.0
        || !o.convexity_tolerance.is_finite()
        || o.convexity_tolerance < 0.0
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid convex QP options",
        ));
    }
    Ok(())
}
fn validate_bound(b: &Bound) -> Result<(), OptimizeError> {
    if b.lower.is_some_and(|v| !v.is_finite())
        || b.upper.is_some_and(|v| !v.is_finite())
        || matches!((b.lower,b.upper),(Some(l),Some(u))if l>u)
    {
        return Err(invalid("QP bounds must be finite when present and ordered"));
    }
    Ok(())
}
fn project(v: f64, b: &Bound) -> f64 {
    b.lower
        .map_or(v, |l| v.max(l))
        .min(b.upper.unwrap_or(f64::INFINITY))
}
fn max_violation(v: &[f64], b: &[Bound]) -> f64 {
    v.iter().zip(b).fold(0.0, |m, (&v, b)| {
        m.max(b.lower.map_or(0.0, |l| (l - v).max(0.0)))
            .max(b.upper.map_or(0.0, |u| (v - u).max(0.0)))
    })
}
fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|r| dot(r, x)).collect()
}
fn transpose_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.first().map_or(0, Vec::len);
    let mut y = vec![0.0; n];
    for (r, &xr) in a.iter().zip(x) {
        for j in 0..n {
            y[j] += r[j] * xr
        }
    }
    y
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn inf_norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt()
            } else {
                l[i][j] = s / l[j][j]
            }
        }
    }
    Some(l)
}
fn solve_cholesky(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>()) / l[i][i]
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - ((i + 1)..n).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i]
    }
    x
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn empty(n: usize) -> CsrMatrix<f64> {
        CsrMatrix::new(0, n, vec![0], vec![], vec![]).unwrap()
    }
    #[test]
    fn solves_bound_constrained_qp() {
        let p = QuadraticProblem {
            quadratic: vec![vec![2.0]],
            linear: vec![-4.0],
            constant: 0.0,
            constraints: empty(1),
            row_bounds: vec![],
            variable_bounds: vec![Bound {
                lower: Some(0.0),
                upper: Some(1.0),
            }],
        };
        let r = solve_convex_qp(&p, Default::default()).unwrap();
        assert_eq!(r.termination, QuadraticTermination::Converged);
        assert!((r.point[0] - 1.0).abs() < 1e-5);
        assert!(r.maximum_variable_violation < 1e-5)
    }
    #[test]
    fn solves_equality_constrained_qp() {
        let a = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        let p = QuadraticProblem {
            quadratic: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            linear: vec![0.0, 0.0],
            constant: 0.0,
            constraints: a,
            row_bounds: vec![Bound::fixed(1.0)],
            variable_bounds: vec![Bound::free(), Bound::free()],
        };
        let r = solve_convex_qp(&p, Default::default()).unwrap();
        assert!((r.point[0] - 0.5).abs() < 1e-5 && (r.point[1] - 0.5).abs() < 1e-5);
        assert!(r.maximum_row_violation < 1e-5 && r.stationarity_norm < 1e-5)
    }
    #[test]
    fn rejects_indefinite_hessian() {
        let p = QuadraticProblem {
            quadratic: vec![vec![-1.0]],
            linear: vec![0.0],
            constant: 0.0,
            constraints: empty(1),
            row_bounds: vec![],
            variable_bounds: vec![Bound::free()],
        };
        assert!(solve_convex_qp(&p, Default::default()).is_err())
    }
}
