//! Reproducible finite-distribution sampling and scenario reduction.

use crate::{
    solve_scenario_tree,
    stochastic::{invalid, probabilities},
    BendersProblem, Bound, OptimizeError, ScenarioRisk, ScenarioSolveOptions, ScenarioTermination,
    ScenarioTree,
};

/// IID draws with replacement. Indices refer to the input probability vector.
pub fn draw_scenarios(
    weights: &[f64],
    count: usize,
    seed: u64,
) -> Result<Vec<usize>, OptimizeError> {
    if count == 0 {
        return Err(invalid("scenario sample must have at least one draw"));
    }
    let mut total = 0.0;
    let mut cdf: Vec<_> = probabilities(weights.iter().copied())?
        .iter()
        .map(|p| {
            total += p;
            total.min(1.0)
        })
        .collect();
    *cdf.last_mut().unwrap() = 1.0;
    let mut state = seed;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| invalid("scenario sample is too large"))?;
    for _ in 0..count {
        // SplitMix64, using the high 53 bits for the unit interval.
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut bits = state;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
        let u = ((bits ^ (bits >> 31)) >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
        indices.push(cdf.partition_point(|&p| p <= u));
    }
    Ok(indices)
}

/// Collapse duplicate draws into empirical probabilities in source order.
pub fn sample_average_problem(
    problem: &BendersProblem,
    indices: &[usize],
) -> Result<BendersProblem, OptimizeError> {
    if indices.is_empty() {
        return Err(invalid("scenario sample must have at least one draw"));
    }
    let mut counts = vec![0; problem.scenarios.len()];
    for &i in indices {
        *counts
            .get_mut(i)
            .ok_or_else(|| invalid("scenario draw index out of range"))? += 1;
    }
    let scenarios = problem
        .scenarios
        .iter()
        .zip(counts)
        .filter(|(_, n)| *n > 0)
        .map(|(s, n)| {
            let mut scenario = s.clone();
            scenario.probability = n as f64 / indices.len() as f64;
            scenario
        })
        .collect();
    Ok(BendersProblem {
        first_stage: problem.first_stage.clone(),
        scenarios,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioEvaluation {
    pub sample_count: usize,
    pub mean: f64,
    pub standard_error: f64,
    /// Asymptotic normal interval for a fixed policy and independent draws.
    pub confidence_interval_95: (f64, f64),
}

/// Evaluate a fixed two-stage decision on a separate IID sample. Any sampled
/// scenario with uncertified recourse prevents statistical reporting.
pub fn evaluate_scenario_sample(
    problem: &BendersProblem,
    first_stage: &[f64],
    indices: &[usize],
    options: ScenarioSolveOptions,
) -> Result<ScenarioEvaluation, OptimizeError> {
    let mut sampled = sample_average_problem(problem, indices)?;
    sampled.first_stage.validate()?;
    let root = &mut sampled.first_stage;
    if indices.len() < 2
        || first_stage.len() != root.objective.len()
        || first_stage.iter().enumerate().any(|(i, &x)| {
            !x.is_finite()
                || root.var_bounds[i].lower.is_some_and(|lo| x < lo)
                || root.var_bounds[i].upper.is_some_and(|hi| x > hi)
                || (root.is_integer[i] && x != x.round())
        })
    {
        return Err(invalid(
            "evaluation requires a valid first-stage point and at least two draws",
        ));
    }
    root.var_bounds = first_stage.iter().copied().map(Bound::fixed).collect();
    let result = solve_scenario_tree(
        &ScenarioTree::two_stage(&sampled),
        ScenarioRisk::ExpectedValue,
        options,
    )?;
    let policy = match (result.termination, result.policy) {
        (ScenarioTermination::Converged, Some(policy)) => policy,
        (termination, _) => {
            return Err(invalid(&format!(
                "policy evaluation stopped with {termination:?}"
            )))
        }
    };
    let n = indices.len() as f64;
    let mean = policy.expected_cost;
    let variance: f64 = policy
        .terminal_costs
        .iter()
        .map(|s| s.probability * (s.cost - mean).powi(2))
        .sum();
    let standard_error = (variance / (n - 1.0)).sqrt();
    let radius = 1.959_963_984_540_054 * standard_error;
    let confidence_interval_95 = (mean - radius, mean + radius);
    if !confidence_interval_95.0.is_finite() || !confidence_interval_95.1.is_finite() {
        return Err(invalid("non-finite policy evaluation interval"));
    }
    Ok(ScenarioEvaluation {
        sample_count: indices.len(),
        mean,
        standard_error,
        confidence_interval_95,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioReduction {
    /// Input indices and their redistributed probabilities, in input order.
    pub retained: Vec<(usize, f64)>,
    /// One retained input index per original scenario.
    pub assignment: Vec<usize>,
    pub weighted_distance: f64,
}

/// Greedy forward selection under probability-weighted Euclidean distance.
/// Features should be scaled by the caller to reflect the intended metric.
pub fn reduce_scenarios(
    features: &[Vec<f64>],
    weights: &[f64],
    keep: usize,
) -> Result<ScenarioReduction, OptimizeError> {
    let n = features.len();
    if weights.len() != n
        || keep == 0
        || keep > n
        || features[0].is_empty()
        || features
            .iter()
            .any(|f| f.len() != features[0].len() || f.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid("invalid scenario reduction dimensions or features"));
    }
    let weights = probabilities(weights.iter().copied())?;
    let distance = |i: usize, j: usize| {
        features[i]
            .iter()
            .zip(&features[j])
            .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b))
    };
    let mut selected = vec![false; n];
    let mut nearest = vec![f64::INFINITY; n];
    let mut assignment = vec![0; n];
    for _ in 0..keep {
        let (index, cost) = (0..n)
            .filter(|&j| !selected[j])
            .map(|j| {
                let cost: f64 = (0..n)
                    .map(|i| weights[i] * nearest[i].min(distance(i, j)))
                    .sum();
                (j, cost)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .unwrap();
        if !cost.is_finite() {
            return Err(invalid("non-finite scenario reduction distance"));
        }
        selected[index] = true;
        for i in 0..n {
            let candidate = distance(i, index);
            if i == index || candidate < nearest[i] {
                nearest[i] = candidate;
                assignment[i] = index;
            }
        }
    }
    let mut mass = vec![0.0; n];
    for (&target, weight) in assignment.iter().zip(&weights) {
        mass[target] += weight;
    }
    Ok(ScenarioReduction {
        retained: mass
            .into_iter()
            .enumerate()
            .filter(|(i, _)| selected[*i])
            .collect(),
        assignment,
        weighted_distance: nearest.iter().zip(weights).map(|(d, w)| d * w).sum(),
    })
}
