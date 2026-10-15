//! Independent, fail-closed certificates for public optimization results.

use crate::{Bound, OptimizeError, Problem, QuadraticProblem, QuadraticResult, Solution, SolveStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertificateClaim { Optimality, Feasibility, Candidate, Rejected }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertificateTolerance {
    pub feasibility: f64,
    pub integrality: f64,
    pub stationarity: f64,
    pub complementarity: f64,
    pub objective: f64,
    pub gap: f64,
    pub active_set: f64,
}
impl Default for CertificateTolerance {
    fn default() -> Self { Self { feasibility: 1e-5, integrality: 1e-7,
        stationarity: 1e-6, complementarity: 1e-5, objective: 1e-8,
        gap: 1e-7, active_set: 1e-5 } }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CertificateMetric {
    pub name: String,
    pub value: f64,
    pub tolerance: f64,
    pub passed: bool,
}
impl CertificateMetric {
    fn new(name: &str, value: f64, tolerance: f64) -> Self {
        Self { name: name.into(), value, tolerance,
            passed: value.is_finite() && value <= tolerance }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationCertificate {
    pub claim: CertificateClaim,
    pub finite: bool,
    pub metrics: Vec<CertificateMetric>,
    pub warnings: Vec<String>,
}
impl OptimizationCertificate {
    pub fn verified(&self) -> bool {
        self.finite && matches!(self.claim, CertificateClaim::Optimality | CertificateClaim::Feasibility)
            && !self.metrics.is_empty() && self.metrics.iter().all(|metric| metric.passed)
    }
}

pub fn certify_linear_solution(problem: &Problem, solution: &Solution,
    global_gap: Option<f64>, tolerance: CertificateTolerance)
    -> Result<OptimizationCertificate, OptimizeError> {
    validate_tolerance(tolerance)?;
    let diagnostics = solution.diagnostics(problem);
    let mut metrics = vec![
        CertificateMetric::new("row feasibility", diagnostics.maximum_row_violation, tolerance.feasibility),
        CertificateMetric::new("variable feasibility", diagnostics.maximum_variable_bound_violation, tolerance.feasibility),
        CertificateMetric::new("integrality", diagnostics.maximum_integrality_violation, tolerance.integrality),
        CertificateMetric::new("objective reconstruction", diagnostics.objective_error, tolerance.objective),
    ];
    if let Some(gap) = global_gap {
        metrics.push(CertificateMetric::new("global bound gap", gap.abs(), tolerance.gap));
    }
    let finite = diagnostics.finite && metrics.iter().all(|metric| metric.value.is_finite());
    let passed = finite && metrics.iter().all(|metric| metric.passed);
    let mut warnings = Vec::new();
    let claim = if solution.status != SolveStatus::Optimal || !passed {
        CertificateClaim::Rejected
    } else if global_gap.is_some() {
        CertificateClaim::Optimality
    } else {
        warnings.push("dual or global-bound evidence unavailable; certifying feasibility only".into());
        CertificateClaim::Feasibility
    };
    Ok(OptimizationCertificate { claim, finite, metrics, warnings })
}

pub fn certify_convex_qp(problem: &QuadraticProblem, result: &QuadraticResult,
    tolerance: CertificateTolerance) -> Result<OptimizationCertificate, OptimizeError> {
    validate_tolerance(tolerance)?;
    problem.validate(1e-10)?;
    let n = problem.linear.len();
    let m = problem.row_bounds.len();
    if result.point.len() != n || result.row_dual.len() != m || result.variable_dual.len() != n {
        return Ok(rejected("QP primal/dual dimensions do not agree"));
    }
    let activities = problem.constraints.spmv(&result.point)?;
    let mut gradient = vec![0.0; n];
    for i in 0..n {
        gradient[i] = problem.linear[i] + problem.quadratic[i].iter()
            .zip(&result.point).map(|(a, x)| a * x).sum::<f64>();
    }
    for (row, column, value) in problem.constraints.iter_entries() {
        gradient[column] += value * result.row_dual[row];
    }
    for i in 0..n { gradient[i] += result.variable_dual[i]; }
    let stationarity = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let row_violation = activities.iter().zip(&problem.row_bounds)
        .map(|(&value, &bound)| violation(value, bound)).fold(0.0, f64::max);
    let variable_violation = result.point.iter().zip(&problem.variable_bounds)
        .map(|(&value, &bound)| violation(value, bound)).fold(0.0, f64::max);
    let mut complementarity: f64 = 0.0;
    let mut dual_violation: f64 = 0.0;
    let mut active_rows = Vec::new();
    for row in 0..m {
        let (product, sign_error, active) = dual_evidence(
            activities[row], problem.row_bounds[row], result.row_dual[row], tolerance.active_set);
        complementarity = complementarity.max(product); dual_violation = dual_violation.max(sign_error);
        if active { active_rows.push(row_vector(problem, row)); }
    }
    for i in 0..n {
        let (product, sign_error, active) = dual_evidence(
            result.point[i], problem.variable_bounds[i], result.variable_dual[i], tolerance.active_set);
        complementarity = complementarity.max(product); dual_violation = dual_violation.max(sign_error);
        if active { let mut row = vec![0.0; n]; row[i] = 1.0; active_rows.push(row); }
    }
    let objective_error = (problem.objective(&result.point) - result.objective).abs();
    let metrics = vec![
        CertificateMetric::new("row feasibility", row_violation, tolerance.feasibility),
        CertificateMetric::new("variable feasibility", variable_violation, tolerance.feasibility),
        CertificateMetric::new("stationarity", stationarity, tolerance.stationarity),
        CertificateMetric::new("complementarity", complementarity, tolerance.complementarity),
        CertificateMetric::new("dual sign feasibility", dual_violation, tolerance.complementarity),
        CertificateMetric::new("objective reconstruction", objective_error, tolerance.objective),
    ];
    let finite = result.point.iter().chain(&result.row_dual).chain(&result.variable_dual)
        .all(|value| value.is_finite()) && metrics.iter().all(|metric| metric.value.is_finite());
    let mut warnings = Vec::new();
    let active_count = active_rows.len();
    let rank = numerical_rank(active_rows, tolerance.active_set);
    if rank < active_count { warnings.push("active constraints are linearly dependent".into()); }
    if active_count > n { warnings.push("active set is overdetermined".into()); }
    let claim = if finite && metrics.iter().all(|metric| metric.passed) {
        CertificateClaim::Optimality
    } else { CertificateClaim::Rejected };
    Ok(OptimizationCertificate { claim, finite, metrics, warnings })
}

fn dual_evidence(value: f64, bound: Bound, dual: f64, active_tolerance: f64) -> (f64, f64, bool) {
    let lower_slack = bound.lower.map(|limit| value - limit);
    let upper_slack = bound.upper.map(|limit| limit - value);
    if dual == 0.0 {
        let slack = lower_slack.into_iter().chain(upper_slack).fold(f64::INFINITY, f64::min);
        return (0.0, 0.0, slack.is_finite() && slack.abs() <= active_tolerance);
    }
    let slack = if dual < 0.0 { lower_slack.unwrap_or(f64::INFINITY) }
        else if dual > 0.0 { upper_slack.unwrap_or(f64::INFINITY) }
        else { f64::INFINITY };
    let sign_error = if (dual < 0.0 && bound.lower.is_none()) || (dual > 0.0 && bound.upper.is_none()) {
        dual.abs()
    } else { 0.0 };
    ((dual * slack).abs(), sign_error, slack.abs() <= active_tolerance)
}

fn row_vector(problem: &QuadraticProblem, selected: usize) -> Vec<f64> {
    let mut row = vec![0.0; problem.linear.len()];
    for (r, c, value) in problem.constraints.iter_entries() { if r == selected { row[c] = value; } }
    row
}
fn numerical_rank(mut rows: Vec<Vec<f64>>, tolerance: f64) -> usize {
    if rows.is_empty() { return 0; }
    let columns = rows[0].len(); let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows.len()).max_by(|&a, &b| rows[a][column].abs().total_cmp(&rows[b][column].abs())) else { break };
        if rows[pivot][column].abs() <= tolerance { continue; }
        rows.swap(rank, pivot); let diagonal = rows[rank][column];
        for row in rank + 1..rows.len() { let factor = rows[row][column] / diagonal;
            for item in column..columns { rows[row][item] -= factor * rows[rank][item]; } }
        rank += 1; if rank == rows.len() { break; }
    }
    rank
}
fn violation(value: f64, bound: Bound) -> f64 {
    bound.lower.map_or(0.0, |limit| (limit - value).max(0.0))
        .max(bound.upper.map_or(0.0, |limit| (value - limit).max(0.0)))
}
fn rejected(message: &str) -> OptimizationCertificate {
    OptimizationCertificate { claim: CertificateClaim::Rejected, finite: false, metrics: vec![],
        warnings: vec![message.into()] }
}
fn validate_tolerance(value: CertificateTolerance) -> Result<(), OptimizeError> {
    let values = [value.feasibility, value.integrality, value.stationarity,
        value.complementarity, value.objective, value.gap, value.active_set];
    if values.iter().all(|value| value.is_finite() && *value >= 0.0) { Ok(()) }
    else { Err(OptimizeError::InvalidConfiguration("certificate tolerances must be finite and nonnegative")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{solve_convex_qp, QuadraticOptions};
    use nc_sparse::CsrMatrix;

    #[test]
    fn convex_qp_certificate_recomputes_kkt_evidence() {
        let problem = QuadraticProblem { quadratic: vec![vec![2.0]], linear: vec![-4.0],
            constant: 0.0, constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![]).unwrap(),
            row_bounds: vec![], variable_bounds: vec![Bound { lower: Some(0.0), upper: Some(1.0) }] };
        let result = solve_convex_qp(&problem, QuadraticOptions { rho: 1e-8, ..Default::default() }).unwrap();
        let certificate = certify_convex_qp(&problem, &result, CertificateTolerance::default()).unwrap();
        assert_eq!(certificate.claim, CertificateClaim::Optimality, "{certificate:#?}");
        assert!(certificate.verified());

        let mut corrupted = result.clone();
        corrupted.point[0] = 2.0;
        let rejected = certify_convex_qp(&problem, &corrupted, CertificateTolerance::default()).unwrap();
        assert_eq!(rejected.claim, CertificateClaim::Rejected);
    }

    #[test]
    fn linear_certificate_does_not_invent_dual_evidence() {
        let problem = Problem { objective: vec![1.0],
            constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![]).unwrap(),
            row_bounds: vec![], var_bounds: vec![Bound::fixed(2.0)], is_integer: vec![false] };
        let solution = Solution { variable_values: vec![2.0], objective_value: 2.0,
            status: SolveStatus::Optimal };
        let feasible = certify_linear_solution(&problem, &solution, None, Default::default()).unwrap();
        assert_eq!(feasible.claim, CertificateClaim::Feasibility);
        assert!(feasible.verified());
        let optimal = certify_linear_solution(&problem, &solution, Some(0.0), Default::default()).unwrap();
        assert_eq!(optimal.claim, CertificateClaim::Optimality);
        let rejected = certify_linear_solution(&problem, &solution, Some(f64::NAN), Default::default()).unwrap();
        assert_eq!(rejected.claim, CertificateClaim::Rejected);
        assert!(!rejected.finite);
    }
}
