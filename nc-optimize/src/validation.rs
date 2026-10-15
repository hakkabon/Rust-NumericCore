//! Reproducible, solver-independent numerical validation primitives.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationTier {
    Smoke,
    Standard,
    Extended,
}

impl ValidationTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Standard => "standard",
            Self::Extended => "extended",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationTolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl Default for ValidationTolerance {
    fn default() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-8,
        }
    }
}

impl ValidationTolerance {
    pub fn new(absolute: f64, relative: f64) -> Self {
        Self { absolute, relative }
    }
    pub fn is_valid(self) -> bool {
        self.absolute.is_finite()
            && self.absolute >= 0.0
            && self.relative.is_finite()
            && self.relative >= 0.0
    }
    pub fn threshold(self, observed: f64, reference: f64) -> f64 {
        if !self.is_valid() || !observed.is_finite() || !reference.is_finite() {
            return f64::NAN;
        }
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
    pub fn compare(
        name: impl Into<String>,
        observed: f64,
        reference: f64,
        tolerance: ValidationTolerance,
    ) -> Self {
        let absolute_error = (observed - reference).abs();
        let allowed_error = tolerance.threshold(observed, reference);
        let normalized_error = if allowed_error > 0.0 {
            absolute_error / allowed_error
        } else if absolute_error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        let passed = absolute_error.is_finite()
            && allowed_error.is_finite()
            && absolute_error <= allowed_error;
        Self {
            name: name.into(),
            observed,
            reference,
            absolute_error,
            allowed_error,
            normalized_error,
            passed,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationProvenance {
    pub collection: String,
    pub problem: String,
    pub citation: String,
    pub source_url: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationCase {
    pub id: String,
    pub family: String,
    pub tier: ValidationTier,
    pub name: String,
    pub metrics: Vec<ValidationMetric>,
    pub provenance: Option<ValidationProvenance>,
}

impl ValidationCase {
    pub fn new(name: impl Into<String>, metrics: Vec<ValidationMetric>) -> Self {
        let name = name.into();
        Self {
            id: stable_identifier(&name),
            family: "core".into(),
            tier: ValidationTier::Smoke,
            name,
            metrics,
            provenance: None,
        }
    }
    pub fn external(
        name: impl Into<String>,
        metrics: Vec<ValidationMetric>,
        provenance: ValidationProvenance,
    ) -> Self {
        let name = name.into();
        Self {
            id: stable_identifier(&name),
            family: "nonlinear-optimization".into(),
            tier: ValidationTier::Standard,
            name,
            metrics,
            provenance: Some(provenance),
        }
    }
    pub fn classified(
        mut self,
        id: impl Into<String>,
        family: impl Into<String>,
        tier: ValidationTier,
    ) -> Self {
        self.id = id.into();
        self.family = family.into();
        self.tier = tier;
        self
    }
    pub fn passed(&self) -> bool {
        !self.metrics.is_empty() && self.metrics.iter().all(|m| m.passed)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub cases: Vec<ValidationCase>,
}

impl ValidationReport {
    pub fn new(cases: Vec<ValidationCase>) -> Self {
        Self { cases }
    }
    pub fn passed(&self) -> bool {
        !self.cases.is_empty() && self.cases.iter().all(ValidationCase::passed)
    }
    pub fn failure_count(&self) -> usize {
        self.cases.iter().filter(|case| !case.passed()).count()
    }
    pub fn maximum_normalized_error(&self) -> f64 {
        self.cases
            .iter()
            .flat_map(|case| &case.metrics)
            .map(|metric| metric.normalized_error)
            .fold(0.0, f64::max)
    }
    pub fn for_tier(&self, tier: ValidationTier) -> Self {
        Self::new(
            self.cases
                .iter()
                .filter(|case| case.tier <= tier)
                .cloned()
                .collect(),
        )
    }
    pub fn snapshot(
        &self,
        implementation: impl Into<String>,
        tier: ValidationTier,
    ) -> ValidationSnapshot {
        ValidationSnapshot::new(implementation.into(), tier, self.for_tier(tier).cases)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationSnapshot {
    pub schema_version: u32,
    pub implementation: String,
    pub tier: ValidationTier,
    pub cases: Vec<ValidationCase>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationRegression {
    pub case_id: String,
    pub metric: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationManifestEntry {
    pub id: String,
    pub family: String,
    pub tier: ValidationTier,
    pub name: String,
    pub metric_names: Vec<String>,
    pub provenance: Option<ValidationProvenance>,
}

impl ValidationSnapshot {
    pub fn new(
        implementation: String,
        tier: ValidationTier,
        mut cases: Vec<ValidationCase>,
    ) -> Self {
        cases.sort_by(|left, right| left.id.cmp(&right.id));
        Self {
            schema_version: 1,
            implementation,
            tier,
            cases,
        }
    }

    pub fn manifest(&self) -> Vec<ValidationManifestEntry> {
        self.cases
            .iter()
            .map(|case| ValidationManifestEntry {
                id: case.id.clone(),
                family: case.family.clone(),
                tier: case.tier,
                name: case.name.clone(),
                metric_names: case
                    .metrics
                    .iter()
                    .map(|metric| metric.name.clone())
                    .collect(),
                provenance: case.provenance.clone(),
            })
            .collect()
    }

    pub fn manifest_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for (index, case) in self.cases.iter().enumerate() {
            if case.id.is_empty() {
                issues.push(format!("case at index {index} has an empty identifier"));
            }
            if self.cases[..index].iter().any(|other| other.id == case.id) {
                issues.push(format!("duplicate case identifier: {}", case.id));
            }
            if case.family.is_empty() {
                issues.push(format!("case {} has an empty family", case.id));
            }
        }
        issues
    }

    /// A deterministic, timestamp-free JSON artifact suitable for checked-in baselines and CI.
    pub fn to_json_pretty(&self) -> String {
        let mut output = format!(
            "{{\n  \"schemaVersion\": {},\n  \"implementation\": \"{}\",\n  \"tier\": \"{}\",\n  \"cases\": [",
            self.schema_version, json_escape(&self.implementation), self.tier.as_str());
        for (case_index, case) in self.cases.iter().enumerate() {
            if case_index > 0 {
                output.push(',');
            }
            output.push_str(&format!(
                "\n    {{\"id\":\"{}\",\"family\":\"{}\",\"tier\":\"{}\",\"name\":\"{}\",\"passed\":{},\"metrics\":[",
                json_escape(&case.id), json_escape(&case.family), case.tier.as_str(),
                json_escape(&case.name), case.passed()));
            for (metric_index, metric) in case.metrics.iter().enumerate() {
                if metric_index > 0 {
                    output.push(',');
                }
                output.push_str(&format!(
                    "{{\"name\":\"{}\",\"observed\":{},\"reference\":{},\"absoluteError\":{},\"allowedError\":{},\"normalizedError\":{},\"passed\":{}}}",
                    json_escape(&metric.name), json_number(metric.observed),
                    json_number(metric.reference), json_number(metric.absolute_error),
                    json_number(metric.allowed_error), json_number(metric.normalized_error),
                    metric.passed));
            }
            output.push_str("]");
            if let Some(source) = &case.provenance {
                output.push_str(&format!(
                    ",\"provenance\":{{\"collection\":\"{}\",\"problem\":\"{}\",\"citation\":\"{}\",\"sourceURL\":\"{}\"}}",
                    json_escape(&source.collection), json_escape(&source.problem),
                    json_escape(&source.citation), json_escape(&source.source_url)));
            }
            output.push('}');
        }
        output.push_str("\n  ]\n}\n");
        output
    }

    /// Compare a current run with an earlier in-memory snapshot. New cases are allowed; removed
    /// cases, missing metrics, pass-to-fail transitions, and material normalized-error growth are
    /// regressions. Runtime is intentionally excluded because it is not reproducible in shared CI.
    pub fn regressions_against(
        &self,
        baseline: &Self,
        normalized_error_slack: f64,
    ) -> Vec<ValidationRegression> {
        let slack = if normalized_error_slack.is_finite() && normalized_error_slack >= 0.0 {
            normalized_error_slack
        } else {
            0.0
        };
        let mut regressions = Vec::new();
        for old_case in &baseline.cases {
            let Some(new_case) = self.cases.iter().find(|case| case.id == old_case.id) else {
                regressions.push(ValidationRegression {
                    case_id: old_case.id.clone(),
                    metric: None,
                    message: "case missing from current snapshot".into(),
                });
                continue;
            };
            for old_metric in &old_case.metrics {
                let Some(new_metric) = new_case
                    .metrics
                    .iter()
                    .find(|metric| metric.name == old_metric.name)
                else {
                    regressions.push(ValidationRegression {
                        case_id: old_case.id.clone(),
                        metric: Some(old_metric.name.clone()),
                        message: "metric missing from current snapshot".into(),
                    });
                    continue;
                };
                if old_metric.passed && !new_metric.passed {
                    regressions.push(ValidationRegression {
                        case_id: old_case.id.clone(),
                        metric: Some(old_metric.name.clone()),
                        message: "metric changed from pass to fail".into(),
                    });
                } else if new_metric.normalized_error > old_metric.normalized_error + slack {
                    regressions.push(ValidationRegression {
                        case_id: old_case.id.clone(),
                        metric: Some(old_metric.name.clone()),
                        message: "normalized error exceeded baseline slack".into(),
                    });
                }
            }
        }
        regressions
    }
}

fn stable_identifier(name: &str) -> String {
    let mut id = String::new();
    let mut separator = false;
    for character in name.chars().flat_map(char::to_lowercase) {
        if character.is_ascii_alphanumeric() {
            id.push(character);
            separator = false;
        } else if !separator && !id.is_empty() {
            id.push('-');
            separator = true;
        }
    }
    while id.ends_with('-') {
        id.pop();
    }
    id
}

fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            value if value.is_control() => '?'.to_string().chars().collect(),
            value => vec![value],
        })
        .collect()
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.17}")
    } else {
        "null".into()
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
        self.finite && tolerance.is_finite() && tolerance >= 0.0 && self.backward_error <= tolerance
    }
}

/// Independently recompute normwise residual and backward error for `A x = b`.
pub fn validate_linear_system(
    matrix: &[Vec<f64>],
    solution: &[f64],
    rhs: &[f64],
) -> LinearSystemValidation {
    let valid = matrix.len() == rhs.len()
        && matrix.iter().all(|row| row.len() == solution.len())
        && matrix
            .iter()
            .flatten()
            .chain(solution)
            .chain(rhs)
            .all(|value| value.is_finite());
    if !valid {
        return LinearSystemValidation {
            finite: false,
            residual_infinity_norm: f64::INFINITY,
            relative_residual: f64::INFINITY,
            backward_error: f64::INFINITY,
        };
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
    LinearSystemValidation {
        finite: residual_norm.is_finite() && backward_error.is_finite(),
        residual_infinity_norm: residual_norm,
        relative_residual,
        backward_error,
    }
}

/// Stable pseudo-random relative perturbations for repeatable sensitivity tests.
pub fn deterministic_perturbation(values: &[f64], seed: u64, relative_magnitude: f64) -> Vec<f64> {
    if !relative_magnitude.is_finite() || relative_magnitude < 0.0 {
        return vec![f64::NAN; values.len()];
    }
    let mut state = seed;
    values
        .iter()
        .map(|value| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            value + (2.0 * unit - 1.0) * relative_magnitude * value.abs().max(1.0)
        })
        .collect()
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
        let exact =
            validate_linear_system(&[vec![3.0, 1.0], vec![1.0, 2.0]], &[2.0, 3.0], &[9.0, 8.0]);
        assert!(exact.is_verified(1e-15));
        let wrong =
            validate_linear_system(&[vec![3.0, 1.0], vec![1.0, 2.0]], &[2.0, 2.0], &[9.0, 8.0]);
        assert!(!wrong.is_verified(1e-3));
    }

    #[test]
    fn perturbations_are_reproducible_and_bounded() {
        let values = vec![0.0, 2.0, -3.0];
        let first = deterministic_perturbation(&values, 42, 1e-4);
        assert_eq!(first, deterministic_perturbation(&values, 42, 1e-4));
        assert_eq!(
            first,
            vec![
                1.3646065328781521e-5,
                1.9998901853715791,
                -3.0000522970087022
            ]
        );
        for (perturbed, original) in first.iter().zip(values) {
            assert!((perturbed - original).abs() <= 1e-4 * original.abs().max(1.0));
        }
    }

    #[test]
    fn snapshots_are_stable_and_detect_regressions() {
        let baseline = ValidationReport::new(vec![ValidationCase::new(
            "quoted \"case\"",
            vec![ValidationMetric::compare(
                "metric",
                1.0,
                1.0,
                ValidationTolerance::default(),
            )],
        )
        .classified("stable.case", "test", ValidationTier::Smoke)])
        .snapshot("rust", ValidationTier::Smoke);
        assert!(baseline.to_json_pretty().contains("quoted \\\"case\\\""));
        assert_eq!(baseline.manifest()[0].id, "stable.case");

        let current = ValidationReport::new(vec![ValidationCase::new(
            "quoted case",
            vec![ValidationMetric::compare(
                "metric",
                2.0,
                1.0,
                ValidationTolerance::new(0.1, 0.0),
            )],
        )
        .classified("stable.case", "test", ValidationTier::Smoke)])
        .snapshot("rust", ValidationTier::Smoke);
        let regressions = current.regressions_against(&baseline, 0.0);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].metric.as_deref(), Some("metric"));
    }
}
