//! Reproducible, solver-independent numerical validation primitives.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationTolerance { pub absolute: f64, pub relative: f64 }

impl Default for ValidationTolerance {
    fn default() -> Self { Self { absolute: 1e-10, relative: 1e-8 } }
}

impl ValidationTolerance {
    pub fn new(absolute: f64, relative: f64) -> Self { Self { absolute, relative } }
    pub fn is_valid(self) -> bool {
        self.absolute.is_finite() && self.absolute >= 0.0
            && self.relative.is_finite() && self.relative >= 0.0
    }
    pub fn threshold(self, observed: f64, reference: f64) -> f64 {
        if !self.is_valid() || !observed.is_finite() || !reference.is_finite() { return f64::NAN; }
        self.absolute + self.relative * observed.abs().max(reference.abs())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationMetric {
    pub name: String,
    pub observed: f64,
    pub reference: f64,
    pub absolute_error: f64,
    pub allowed_error: f64,
    pub normalized_error: f64,
    pub passed: bool,
}

impl ValidationMetric {
    pub fn compare(name: impl Into<String>, observed: f64, reference: f64,
                   tolerance: ValidationTolerance) -> Self {
        let absolute_error = (observed - reference).abs();
        let allowed_error = tolerance.threshold(observed, reference);
        let normalized_error = if allowed_error > 0.0 { absolute_error / allowed_error }
            else if absolute_error == 0.0 { 0.0 } else { f64::INFINITY };
        let passed = absolute_error.is_finite() && allowed_error.is_finite()
            && absolute_error <= allowed_error;
        Self { name: name.into(), observed, reference, absolute_error,
            allowed_error, normalized_error, passed }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationCase { pub name: String, pub metrics: Vec<ValidationMetric> }

impl ValidationCase {
    pub fn new(name: impl Into<String>, metrics: Vec<ValidationMetric>) -> Self {
        Self { name: name.into(), metrics }
    }
    pub fn passed(&self) -> bool { !self.metrics.is_empty() && self.metrics.iter().all(|m| m.passed) }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport { pub cases: Vec<ValidationCase> }

impl ValidationReport {
    pub fn new(cases: Vec<ValidationCase>) -> Self { Self { cases } }
    pub fn passed(&self) -> bool { !self.cases.is_empty() && self.cases.iter().all(ValidationCase::passed) }
    pub fn failure_count(&self) -> usize { self.cases.iter().filter(|case| !case.passed()).count() }
    pub fn maximum_normalized_error(&self) -> f64 {
        self.cases.iter().flat_map(|case| &case.metrics)
            .map(|metric| metric.normalized_error).fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearSystemValidation {
    pub finite: bool,
    pub residual_infinity_norm: f64,
    pub relative_residual: f64,
    pub backward_error: f64,
}

impl LinearSystemValidation {
    pub fn is_verified(self, tolerance: f64) -> bool {
        self.finite && tolerance.is_finite() && tolerance >= 0.0
            && self.backward_error <= tolerance
    }
}

/// Independently recompute normwise residual and backward error for `A x = b`.
pub fn validate_linear_system(matrix: &[Vec<f64>], solution: &[f64], rhs: &[f64])
    -> LinearSystemValidation {
    let valid = matrix.len() == rhs.len() && matrix.iter().all(|row| row.len() == solution.len())
        && matrix.iter().flatten().chain(solution).chain(rhs).all(|value| value.is_finite());
    if !valid {
        return LinearSystemValidation { finite: false, residual_infinity_norm: f64::INFINITY,
            relative_residual: f64::INFINITY, backward_error: f64::INFINITY };
    }
    let mut residual_norm: f64 = 0.0;
    let mut matrix_norm: f64 = 0.0;
    for (row, target) in matrix.iter().zip(rhs) {
        let fitted: f64 = row.iter().zip(solution).map(|(a, x)| a * x).sum();
        residual_norm = residual_norm.max((fitted - target).abs());
        matrix_norm = matrix_norm.max(row.iter().map(|value| value.abs()).sum());
    }
    let solution_norm = solution.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let rhs_norm = rhs.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let relative_residual = residual_norm / rhs_norm.max(1e-300);
    let denominator = matrix_norm * solution_norm + rhs_norm;
    let backward_error = residual_norm / denominator.max(1e-300);
    LinearSystemValidation { finite: residual_norm.is_finite() && backward_error.is_finite(),
        residual_infinity_norm: residual_norm, relative_residual, backward_error }
}

/// Stable pseudo-random relative perturbations for repeatable sensitivity tests.
pub fn deterministic_perturbation(values: &[f64], seed: u64, relative_magnitude: f64) -> Vec<f64> {
    if !relative_magnitude.is_finite() || relative_magnitude < 0.0 { return vec![f64::NAN; values.len()]; }
    let mut state = seed;
    values.iter().map(|value| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        value + (2.0 * unit - 1.0) * relative_magnitude * value.abs().max(1.0)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_aware_metrics_and_reports_fail_closed() {
        let tolerance = ValidationTolerance::new(1e-12, 1e-6);
        let good = ValidationMetric::compare("large", 1e9 + 100.0, 1e9, tolerance);
        let bad = ValidationMetric::compare("nonfinite", f64::NAN, 0.0, tolerance);
        assert!(good.passed);
        assert!(!bad.passed);
        let report = ValidationReport::new(vec![ValidationCase::new("case", vec![good, bad])]);
        assert!(!report.passed());
        assert_eq!(report.failure_count(), 1);
    }

    #[test]
    fn independently_checks_backward_error() {
        let exact = validate_linear_system(&[vec![3.0, 1.0], vec![1.0, 2.0]], &[2.0, 3.0], &[9.0, 8.0]);
        assert!(exact.is_verified(1e-15));
        let wrong = validate_linear_system(&[vec![3.0, 1.0], vec![1.0, 2.0]], &[2.0, 2.0], &[9.0, 8.0]);
        assert!(!wrong.is_verified(1e-3));
    }

    #[test]
    fn perturbations_are_reproducible_and_bounded() {
        let values = vec![0.0, 2.0, -3.0];
        let first = deterministic_perturbation(&values, 42, 1e-4);
        assert_eq!(first, deterministic_perturbation(&values, 42, 1e-4));
        assert_eq!(first, vec![1.3646065328781521e-5, 1.9998901853715791, -3.0000522970087022]);
        for (perturbed, original) in first.iter().zip(values) {
            assert!((perturbed - original).abs() <= 1e-4 * original.abs().max(1.0));
        }
    }
}
