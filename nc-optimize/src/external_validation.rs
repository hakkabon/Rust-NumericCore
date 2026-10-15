//! Offline, deterministic fixtures from independently maintained numerical corpora.

use crate::{
    minimize_sqp, solve_model_least_squares, Bound, ConstrainedNonlinearProblem,
    NonlinearConstraint, NonlinearExpression, NonlinearLeastSquaresOptions, NonlinearModel,
    NonlinearNode::*, OptimizeError, RobustLoss, SqpOptions, ValidationCase, ValidationMetric,
    ValidationProvenance, ValidationTier, ValidationTolerance,
};

pub fn run_external_reference_validation() -> Result<Vec<ValidationCase>, OptimizeError> {
    Ok(vec![nist_misra1a()?, cutest_hs71()?])
}

fn nist_misra1a() -> Result<ValidationCase, OptimizeError> {
    let observations = [
        (10.07, 77.6),
        (14.73, 114.9),
        (17.94, 141.1),
        (23.93, 190.8),
        (29.61, 239.9),
        (35.18, 289.0),
        (40.02, 332.8),
        (44.82, 378.4),
        (50.76, 434.8),
        (55.05, 477.3),
        (61.01, 536.8),
        (66.40, 593.1),
        (75.47, 689.1),
        (81.78, 760.0),
    ];
    let residuals = observations
        .iter()
        .map(|&(response, predictor)| {
            NonlinearExpression::new(
                vec![
                    Parameter(0),
                    Constant(1.0),
                    Parameter(1),
                    Constant(-predictor),
                    Multiply(2, 3),
                    Exp(4),
                    Subtract(1, 5),
                    Multiply(0, 6),
                    Constant(response),
                    Subtract(7, 8),
                ],
                9,
            )
        })
        .collect();
    let model = NonlinearModel::least_squares(2, vec![Bound::free(), Bound::free()], residuals);
    let result = solve_model_least_squares(
        &model,
        &[250.0, 0.0005],
        &[],
        RobustLoss::Squared,
        NonlinearLeastSquaresOptions::default(),
    )?;
    let rss: f64 = result.residuals.iter().map(|value| value * value).sum();
    Ok(ValidationCase::external(
        "NIST StRD Misra1a nonlinear regression",
        vec![
            ValidationMetric::compare(
                "Misra1a b1",
                result.point[0],
                2.3894212918e2,
                ValidationTolerance::new(3e-3, 2e-5),
            ),
            ValidationMetric::compare(
                "Misra1a b2",
                result.point[1],
                5.5015643181e-4,
                ValidationTolerance::new(2e-8, 2e-5),
            ),
            ValidationMetric::compare(
                "Misra1a residual sum of squares",
                rss,
                1.2455138894e-1,
                ValidationTolerance::new(2e-5, 2e-5),
            ),
        ],
        ValidationProvenance {
            collection: "NIST Statistical Reference Datasets".into(),
            problem: "Misra1a".into(),
            citation: "Misra, D., NIST (1978), Dental Research Monomolecular Adsorption Study"
                .into(),
            source_url: "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat"
                .into(),
        },
    )
    .classified(
        "nist.strd.misra1a",
        "nonlinear-least-squares",
        ValidationTier::Standard,
    ))
}

fn cutest_hs71() -> Result<ValidationCase, OptimizeError> {
    let objective = NonlinearExpression::new(
        vec![
            Parameter(0),
            Parameter(1),
            Parameter(2),
            Parameter(3),
            Add(0, 1),
            Add(4, 2),
            Multiply(0, 3),
            Multiply(6, 5),
            Add(7, 2),
        ],
        8,
    );
    let product = NonlinearExpression::new(
        vec![
            Parameter(0),
            Parameter(1),
            Parameter(2),
            Parameter(3),
            Multiply(0, 1),
            Multiply(4, 2),
            Multiply(5, 3),
        ],
        6,
    );
    let squares = NonlinearExpression::new(
        vec![
            Parameter(0),
            Powf(0, 2.0),
            Parameter(1),
            Powf(2, 2.0),
            Parameter(2),
            Powf(4, 2.0),
            Parameter(3),
            Powf(6, 2.0),
            Add(1, 3),
            Add(8, 5),
            Add(9, 7),
        ],
        10,
    );
    let problem = ConstrainedNonlinearProblem {
        model: NonlinearModel::objective(
            4,
            vec![
                Bound {
                    lower: Some(1.0),
                    upper: Some(5.0)
                };
                4
            ],
            objective,
        ),
        constraints: vec![
            NonlinearConstraint {
                expression: product,
                bound: Bound {
                    lower: Some(25.0),
                    upper: None,
                },
            },
            NonlinearConstraint {
                expression: squares,
                bound: Bound::fixed(40.0),
            },
        ],
    };
    let mut options = SqpOptions::default();
    options.restoration = true;
    options.max_iterations = 300;
    let result = minimize_sqp(&problem, &[1.0, 5.0, 5.0, 1.0], options)?;
    let reference = [1.0, 4.74299963, 3.82114998, 1.37940829];
    let tolerance = ValidationTolerance::new(3e-4, 3e-5);
    let mut metrics: Vec<_> = result
        .point
        .iter()
        .zip(reference)
        .enumerate()
        .map(|(index, (&value, expected))| {
            ValidationMetric::compare(format!("HS71 x{}", index + 1), value, expected, tolerance)
        })
        .collect();
    metrics.push(ValidationMetric::compare(
        "HS71 objective",
        result.objective,
        17.0140173,
        ValidationTolerance::new(3e-4, 3e-5),
    ));
    metrics.push(ValidationMetric::compare(
        "HS71 feasibility",
        result.maximum_violation,
        0.0,
        ValidationTolerance::new(2e-5, 0.0),
    ));
    Ok(ValidationCase::external("CUTEst HS71 constrained nonlinear program", metrics,
        ValidationProvenance { collection: "CUTEst SIF problem set".into(), problem: "HS71".into(),
            citation: "Hock and Schittkowski (1981), Test Examples for Nonlinear Programming Codes, problem 71".into(),
            source_url: "https://github.com/ralna/SIF/blob/master/HS71.SIF".into() })
        .classified("cutest.hs71", "constrained-nonlinear", ValidationTier::Standard))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_reference_cases_pass_and_retain_provenance() {
        let cases = run_external_reference_validation().unwrap();
        assert_eq!(cases.len(), 2);
        assert!(
            cases
                .iter()
                .all(|case| case.passed() && case.provenance.is_some()),
            "{cases:#?}"
        );
    }
}
