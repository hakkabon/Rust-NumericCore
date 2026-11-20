use crate::{
    to_benders_scenario, to_csr_f64, to_domain_problem, FfiBendersScenario, FfiCsrMatrixF64,
    FfiError, FfiProblem,
};
use nc_optimize as opt;

#[derive(uniffi::Record)]
pub struct FfiScenarioLink {
    pub ancestor: u64,
    pub coefficients: FfiCsrMatrixF64,
}

#[derive(uniffi::Record)]
pub struct FfiScenarioNode {
    pub id: String,
    pub parent: Option<u64>,
    pub conditional_probability: f64,
    pub block: FfiProblem,
    pub links: Vec<FfiScenarioLink>,
}

#[derive(uniffi::Enum)]
pub enum FfiScenarioRisk {
    ExpectedValue,
    MeanCvar { alpha: f64, weight: f64 },
}

#[derive(uniffi::Record)]
pub struct FfiScenarioOptions {
    pub max_nodes: u64,
    pub lp_iteration_limit: u64,
    pub tolerance: f64,
}

#[derive(uniffi::Enum)]
pub enum FfiScenarioTermination {
    Converged,
    Infeasible,
    NodeLimit,
    LpIterationLimit,
    MissingEvidence,
    UnboundedRelaxation,
}

#[derive(uniffi::Record)]
pub struct FfiScenarioCost {
    pub id: String,
    pub probability: f64,
    pub cost: f64,
}

#[derive(uniffi::Record)]
pub struct FfiScenarioPolicy {
    pub decisions: Vec<Vec<f64>>,
    pub terminal_costs: Vec<FfiScenarioCost>,
    pub expected_cost: f64,
    pub value_at_risk: Option<f64>,
    pub conditional_value_at_risk: Option<f64>,
    pub objective: f64,
}

#[derive(uniffi::Record)]
pub struct FfiScenarioSolveResult {
    pub termination: FfiScenarioTermination,
    pub policy: Option<FfiScenarioPolicy>,
    pub lower_bound: Option<f64>,
    pub nodes: u64,
}

#[uniffi::export]
pub fn solve_scenario_tree(
    nodes: Vec<FfiScenarioNode>,
    risk: FfiScenarioRisk,
    options: FfiScenarioOptions,
) -> Result<FfiScenarioSolveResult, FfiError> {
    let nodes = nodes
        .into_iter()
        .map(|node| {
            Ok(opt::ScenarioNode {
                id: node.id,
                parent: node.parent.map(size).transpose()?,
                conditional_probability: node.conditional_probability,
                block: to_domain_problem(node.block)?,
                links: node
                    .links
                    .into_iter()
                    .map(|link| {
                        Ok(opt::ScenarioLink {
                            ancestor: size(link.ancestor)?,
                            coefficients: to_csr_f64(link.coefficients)?,
                        })
                    })
                    .collect::<Result<Vec<_>, FfiError>>()?,
            })
        })
        .collect::<Result<Vec<_>, FfiError>>()?;
    solve(&opt::ScenarioTree { nodes }, risk, options)
}

#[uniffi::export]
pub fn draw_scenario_indices(
    weights: Vec<f64>,
    count: u64,
    seed: u64,
) -> Result<Vec<u64>, FfiError> {
    Ok(opt::draw_scenarios(&weights, size(count)?, seed)?
        .into_iter()
        .map(|i| i as u64)
        .collect())
}

#[uniffi::export]
pub fn solve_scenario_sample(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    indices: Vec<u64>,
    risk: FfiScenarioRisk,
    options: FfiScenarioOptions,
) -> Result<FfiScenarioSolveResult, FfiError> {
    let problem = two_stage(first_stage, scenarios)?;
    let sampled = opt::sample_average_problem(&problem, &sample_indices(indices)?)?;
    solve(&opt::ScenarioTree::two_stage(&sampled), risk, options)
}

#[derive(uniffi::Record)]
pub struct FfiScenarioEvaluation {
    pub sample_count: u64,
    pub mean: f64,
    pub standard_error: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
}

#[uniffi::export]
pub fn evaluate_scenario_policy(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    decision: Vec<f64>,
    indices: Vec<u64>,
    options: FfiScenarioOptions,
) -> Result<FfiScenarioEvaluation, FfiError> {
    let result = opt::evaluate_scenario_sample(
        &two_stage(first_stage, scenarios)?,
        &decision,
        &sample_indices(indices)?,
        solve_options(options)?,
    )?;
    Ok(FfiScenarioEvaluation {
        sample_count: result.sample_count as u64,
        mean: result.mean,
        standard_error: result.standard_error,
        confidence_lower: result.confidence_interval_95.0,
        confidence_upper: result.confidence_interval_95.1,
    })
}

#[derive(uniffi::Record)]
pub struct FfiReducedScenario {
    pub index: u64,
    pub probability: f64,
}

#[derive(uniffi::Record)]
pub struct FfiScenarioReduction {
    pub retained: Vec<FfiReducedScenario>,
    pub assignment: Vec<u64>,
    pub weighted_distance: f64,
}

#[uniffi::export]
pub fn reduce_scenario_features(
    features: Vec<Vec<f64>>,
    weights: Vec<f64>,
    keep: u64,
) -> Result<FfiScenarioReduction, FfiError> {
    let result = opt::reduce_scenarios(&features, &weights, size(keep)?)?;
    Ok(FfiScenarioReduction {
        retained: result
            .retained
            .into_iter()
            .map(|(index, probability)| FfiReducedScenario {
                index: index as u64,
                probability,
            })
            .collect(),
        assignment: result.assignment.into_iter().map(|i| i as u64).collect(),
        weighted_distance: result.weighted_distance,
    })
}

#[derive(uniffi::Record)]
pub struct FfiHedgingOptions {
    pub rho: f64,
    pub max_iterations: u64,
    pub tolerance: f64,
    pub qp_iteration_limit: u64,
    pub scenario_threads: u64,
}

#[derive(uniffi::Record)]
pub struct FfiHedgingWarmStart {
    pub primal: Vec<f64>,
    pub row_dual: Vec<f64>,
    pub variable_dual: Vec<f64>,
}

#[derive(uniffi::Record)]
pub struct FfiHedgingCheckpoint {
    pub version: u32,
    pub model_signature: u64,
    pub iterations: u64,
    pub consensus: Vec<f64>,
    pub multipliers: Vec<Vec<f64>>,
    pub warm_starts: Vec<FfiHedgingWarmStart>,
}

#[derive(uniffi::Enum)]
pub enum FfiHedgingTermination {
    Converged,
    IterationLimit,
    SubproblemFailure,
}

#[derive(uniffi::Record)]
pub struct FfiHedgingResult {
    pub termination: FfiHedgingTermination,
    pub checkpoint: Option<FfiHedgingCheckpoint>,
    pub primal_residual: Option<f64>,
    pub dual_residual: Option<f64>,
    pub failed_scenario: Option<u64>,
}

#[uniffi::export]
pub fn run_progressive_hedging(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
    options: FfiHedgingOptions,
    checkpoint: Option<FfiHedgingCheckpoint>,
) -> Result<FfiHedgingResult, FfiError> {
    let checkpoint = checkpoint
        .map(|state| {
            Ok::<_, FfiError>(opt::HedgingCheckpoint {
                version: state.version,
                model_signature: state.model_signature,
                iterations: size(state.iterations)?,
                consensus: state.consensus,
                multipliers: state.multipliers,
                warm_starts: state
                    .warm_starts
                    .into_iter()
                    .map(|w| opt::QuadraticWarmStart {
                        primal: w.primal,
                        row_dual: w.row_dual,
                        variable_dual: w.variable_dual,
                    })
                    .collect(),
            })
        })
        .transpose()?;
    let result = opt::solve_progressive_hedging(
        &two_stage(first_stage, scenarios)?,
        opt::ProgressiveHedgingOptions {
            rho: options.rho,
            max_iterations: size(options.max_iterations)?,
            tolerance: options.tolerance,
            qp_iteration_limit: size(options.qp_iteration_limit)?,
            scenario_threads: size(options.scenario_threads)?,
        },
        checkpoint,
    )?;
    Ok(FfiHedgingResult {
        termination: match result.termination {
            opt::HedgingTermination::Converged => FfiHedgingTermination::Converged,
            opt::HedgingTermination::IterationLimit => FfiHedgingTermination::IterationLimit,
            opt::HedgingTermination::SubproblemFailure => FfiHedgingTermination::SubproblemFailure,
        },
        checkpoint: result.checkpoint.map(|state| FfiHedgingCheckpoint {
            version: state.version,
            model_signature: state.model_signature,
            iterations: state.iterations as u64,
            consensus: state.consensus,
            multipliers: state.multipliers,
            warm_starts: state
                .warm_starts
                .into_iter()
                .map(|w| FfiHedgingWarmStart {
                    primal: w.primal,
                    row_dual: w.row_dual,
                    variable_dual: w.variable_dual,
                })
                .collect(),
        }),
        primal_residual: result.primal_residual,
        dual_residual: result.dual_residual,
        failed_scenario: result.failed_scenario.map(|i| i as u64),
    })
}

fn size(value: u64) -> Result<usize, FfiError> {
    usize::try_from(value).map_err(|_| FfiError::SolverError {
        message: "scenario index or option outside platform range".into(),
    })
}

fn sample_indices(indices: Vec<u64>) -> Result<Vec<usize>, FfiError> {
    indices.into_iter().map(size).collect()
}

fn two_stage(
    first_stage: FfiProblem,
    scenarios: Vec<FfiBendersScenario>,
) -> Result<opt::BendersProblem, FfiError> {
    Ok(opt::BendersProblem {
        first_stage: to_domain_problem(first_stage)?,
        scenarios: scenarios
            .into_iter()
            .map(to_benders_scenario)
            .collect::<Result<_, _>>()?,
    })
}

fn solve_options(options: FfiScenarioOptions) -> Result<opt::ScenarioSolveOptions, FfiError> {
    Ok(opt::ScenarioSolveOptions {
        max_nodes: size(options.max_nodes)?,
        lp_iteration_limit: size(options.lp_iteration_limit)?,
        tolerance: options.tolerance,
    })
}

fn solve(
    tree: &opt::ScenarioTree,
    risk: FfiScenarioRisk,
    options: FfiScenarioOptions,
) -> Result<FfiScenarioSolveResult, FfiError> {
    let risk = match risk {
        FfiScenarioRisk::ExpectedValue => opt::ScenarioRisk::ExpectedValue,
        FfiScenarioRisk::MeanCvar { alpha, weight } => {
            opt::ScenarioRisk::MeanCvar { alpha, weight }
        }
    };
    let result = opt::solve_scenario_tree(tree, risk, solve_options(options)?)?;
    Ok(FfiScenarioSolveResult {
        termination: match result.termination {
            opt::ScenarioTermination::Converged => FfiScenarioTermination::Converged,
            opt::ScenarioTermination::Infeasible => FfiScenarioTermination::Infeasible,
            opt::ScenarioTermination::NodeLimit => FfiScenarioTermination::NodeLimit,
            opt::ScenarioTermination::LpIterationLimit => FfiScenarioTermination::LpIterationLimit,
            opt::ScenarioTermination::MissingEvidence => FfiScenarioTermination::MissingEvidence,
            opt::ScenarioTermination::UnboundedRelaxation => {
                FfiScenarioTermination::UnboundedRelaxation
            }
        },
        policy: result.policy.map(|p| FfiScenarioPolicy {
            decisions: p.decisions,
            expected_cost: p.expected_cost,
            objective: p.objective,
            value_at_risk: p.value_at_risk,
            conditional_value_at_risk: p.conditional_value_at_risk,
            terminal_costs: p
                .terminal_costs
                .into_iter()
                .map(|s| FfiScenarioCost {
                    id: s.id,
                    probability: s.probability,
                    cost: s.cost,
                })
                .collect(),
        }),
        lower_bound: result.lower_bound,
        nodes: result.nodes as u64,
    })
}
