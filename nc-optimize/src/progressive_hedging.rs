//! Progressive hedging for continuous two-stage linear models.

use crate::{
    stochastic::{invalid, probabilities},
    BendersProblem, Bound, OptimizeError, Problem, QuadraticOptions, QuadraticProblem,
    QuadraticTermination, QuadraticWarmStart, QuadraticWorkspace,
};
use nc_sparse::CooMatrix;

const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct ProgressiveHedgingOptions {
    pub rho: f64,
    /// Additional iterations to perform on this invocation.
    pub max_iterations: usize,
    pub tolerance: f64,
    pub qp_iteration_limit: usize,
    pub scenario_threads: usize,
}

impl Default for ProgressiveHedgingOptions {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iterations: 250,
            tolerance: 1e-5,
            qp_iteration_limit: 10_000,
            scenario_threads: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HedgingCheckpoint {
    pub version: u32,
    pub model_signature: u64,
    pub iterations: usize,
    pub consensus: Vec<f64>,
    pub multipliers: Vec<Vec<f64>>,
    pub warm_starts: Vec<QuadraticWarmStart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HedgingTermination {
    Converged,
    IterationLimit,
    SubproblemFailure,
}

#[derive(Debug, Clone)]
pub struct ProgressiveHedgingResult {
    pub termination: HedgingTermination,
    pub checkpoint: Option<HedgingCheckpoint>,
    pub primal_residual: Option<f64>,
    pub dual_residual: Option<f64>,
    pub failed_scenario: Option<usize>,
}

/// Convergence uses weighted first-stage consensus and the change in consensus.
/// This iterative result carries no objective bound.
pub fn solve_progressive_hedging(
    problem: &BendersProblem,
    options: ProgressiveHedgingOptions,
    checkpoint: Option<HedgingCheckpoint>,
) -> Result<ProgressiveHedgingResult, OptimizeError> {
    if options.rho <= 0.0
        || !options.rho.is_finite()
        || !(1e-10..=0.1).contains(&options.tolerance)
        || options.max_iterations == 0
        || options.qp_iteration_limit == 0
        || options.scenario_threads == 0
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid progressive hedging options",
        ));
    }
    let root = &problem.first_stage;
    root.validate()?;
    let n = root.objective.len();
    if n == 0 || root.is_integer.iter().any(|v| *v) {
        return Err(invalid("hedging requires continuous first-stage variables"));
    }
    let weights = probabilities(problem.scenarios.iter().map(|s| s.probability))?;
    let signature = model_signature(problem, options);
    let mut workspaces = problem
        .scenarios
        .iter()
        .map(|s| {
            s.recourse.validate()?;
            if s.recourse.is_integer.iter().any(|v| *v)
                || s.coupling.cols() != n
                || s.coupling.rows() != s.recourse.row_bounds.len()
            {
                return Err(invalid(
                    "hedging requires continuous recourse and valid coupling dimensions",
                ));
            }
            let recourse = &s.recourse;
            let columns = n + recourse.objective.len();
            let rows = root.row_bounds.len() + recourse.row_bounds.len();
            let mut matrix = CooMatrix::new(rows, columns);
            for (r, c, v) in root.constraints.iter_entries() {
                matrix.push(r, c, v)?;
            }
            for (r, c, v) in s.coupling.iter_entries() {
                matrix.push(root.row_bounds.len() + r, c, v)?;
            }
            for (r, c, v) in recourse.constraints.iter_entries() {
                matrix.push(root.row_bounds.len() + r, n + c, v)?;
            }
            let mut quadratic = vec![vec![0.0; columns]; columns];
            for (i, row) in quadratic.iter_mut().take(n).enumerate() {
                row[i] = options.rho;
            }
            let objective = [root.objective.as_slice(), &recourse.objective].concat();
            let qp = QuadraticProblem {
                quadratic,
                linear: objective.clone(),
                constant: 0.0,
                constraints: matrix.to_csr()?,
                row_bounds: [root.row_bounds.as_slice(), &recourse.row_bounds].concat(),
                variable_bounds: [root.var_bounds.as_slice(), &recourse.var_bounds].concat(),
            };
            let workspace = QuadraticWorkspace::new(
                qp,
                QuadraticOptions {
                    max_iterations: options.qp_iteration_limit,
                    absolute_tolerance: options.tolerance * 0.01,
                    relative_tolerance: options.tolerance * 0.001,
                    ..Default::default()
                },
            )?;
            Ok((workspace, objective))
        })
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let mut consensus = vec![0.0; n];
    let mut multipliers = vec![vec![0.0; n]; weights.len()];
    let mut result = ProgressiveHedgingResult {
        termination: HedgingTermination::IterationLimit,
        checkpoint,
        primal_residual: None,
        dual_residual: None,
        failed_scenario: None,
    };
    let previous = if let Some(state) = &result.checkpoint {
        validate_checkpoint(state, signature, n, &workspaces)?;
        consensus.clone_from(&state.consensus);
        multipliers.clone_from(&state.multipliers);
        for ((workspace, _), warm) in workspaces.iter_mut().zip(&state.warm_starts) {
            workspace.set_warm_start(Some(warm.clone()));
        }
        state.iterations
    } else {
        0
    };
    if previous.checked_add(options.max_iterations).is_none() {
        return Err(invalid("hedging iteration count overflow"));
    }
    let batch_size = workspaces.len().div_ceil(options.scenario_threads);
    for step in 1..=options.max_iterations {
        let solve_batch = |batch: &mut [(QuadraticWorkspace, Vec<f64>)], dual: &[Vec<f64>]| {
            batch
                .iter_mut()
                .zip(dual)
                .map(|((workspace, objective), dual)| {
                    let mut linear = objective.clone();
                    for i in 0..n {
                        linear[i] += dual[i] - options.rho * consensus[i];
                    }
                    workspace.update(
                        linear,
                        0.0,
                        workspace.problem().row_bounds.clone(),
                        workspace.problem().variable_bounds.clone(),
                    )?;
                    workspace.solve()
                })
                .collect::<Result<Vec<_>, OptimizeError>>()
        };
        let solved = if options.scenario_threads == 1 || workspaces.len() == 1 {
            solve_batch(&mut workspaces, &multipliers)?
        } else {
            std::thread::scope(|scope| {
                let tasks = workspaces
                    .chunks_mut(batch_size)
                    .zip(multipliers.chunks(batch_size))
                    .map(|(batch, dual)| scope.spawn(|| solve_batch(batch, dual)))
                    .collect::<Vec<_>>();
                let mut solved = Vec::with_capacity(weights.len());
                for task in tasks {
                    solved.extend(
                        task.join()
                            .map_err(|_| invalid("hedging worker panicked"))??,
                    );
                }
                Ok::<_, OptimizeError>(solved)
            })?
        };
        if let Some(i) = solved.iter().position(|s| {
            s.termination != QuadraticTermination::Converged
                || s.maximum_row_violation > options.tolerance
                || s.maximum_variable_violation > options.tolerance
        }) {
            result.termination = HedgingTermination::SubproblemFailure;
            result.failed_scenario = Some(i);
            return Ok(result);
        }
        let mut next = vec![0.0; n];
        for (s, &p) in solved.iter().zip(&weights) {
            for (mean, &x) in next.iter_mut().zip(&s.point) {
                *mean += p * x;
            }
        }
        let dual_residual = next
            .iter()
            .zip(&consensus)
            .fold(0.0_f64, |norm, (x, z)| norm.hypot(options.rho * (x - z)));
        let mut primal_residual = 0.0_f64;
        for ((s, &p), dual) in solved.iter().zip(&weights).zip(&mut multipliers) {
            for ((&x, &z), w) in s.point.iter().zip(&next).zip(dual) {
                let delta = x - z;
                *w += options.rho * delta;
                primal_residual = primal_residual.hypot(p.sqrt() * delta);
            }
        }
        if !primal_residual.is_finite()
            || !dual_residual.is_finite()
            || multipliers.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("non-finite progressive hedging iterate"));
        }
        consensus = next;
        result.primal_residual = Some(primal_residual);
        result.dual_residual = Some(dual_residual);
        if primal_residual <= options.tolerance && dual_residual <= options.tolerance {
            result.termination = HedgingTermination::Converged;
        }
        // Save the last completed iteration if a later scenario subproblem fails.
        result.checkpoint = Some(HedgingCheckpoint {
            version: CHECKPOINT_VERSION,
            model_signature: signature,
            iterations: previous + step,
            consensus: consensus.clone(),
            multipliers: multipliers.clone(),
            warm_starts: solved.iter().map(|s| s.warm_start()).collect(),
        });
        if result.termination == HedgingTermination::Converged {
            break;
        }
    }
    Ok(result)
}

fn validate_checkpoint(
    checkpoint: &HedgingCheckpoint,
    signature: u64,
    n: usize,
    workspaces: &[(QuadraticWorkspace, Vec<f64>)],
) -> Result<(), OptimizeError> {
    let valid = |v: &[f64], size| v.len() == size && v.iter().all(|x| x.is_finite());
    if checkpoint.version != CHECKPOINT_VERSION
        || checkpoint.iterations == 0
        || checkpoint.model_signature != signature
        || !valid(&checkpoint.consensus, n)
        || checkpoint.multipliers.len() != workspaces.len()
        || checkpoint.multipliers.iter().any(|v| !valid(v, n))
        || checkpoint.warm_starts.len() != workspaces.len()
        || checkpoint
            .warm_starts
            .iter()
            .zip(workspaces)
            .any(|(warm, (state, _))| {
                let qp = state.problem();
                !valid(&warm.primal, qp.linear.len())
                    || !valid(&warm.variable_dual, qp.linear.len())
                    || !valid(&warm.row_dual, qp.row_bounds.len())
            })
    {
        return Err(invalid("incompatible progressive hedging checkpoint"));
    }
    Ok(())
}

fn model_signature(problem: &BendersProblem, options: ProgressiveHedgingOptions) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    let mut add = |x: u64| {
        hash = (hash ^ x).wrapping_mul(0x100000001b3);
    };
    fn block(b: &Problem, add: &mut impl FnMut(u64)) {
        for &x in &b.objective {
            add(x.to_bits());
        }
        for &Bound { lower, upper } in b.var_bounds.iter().chain(&b.row_bounds) {
            add(lower.unwrap_or(f64::NEG_INFINITY).to_bits());
            add(upper.unwrap_or(f64::INFINITY).to_bits());
        }
        matrix(&b.constraints, add);
    }
    fn matrix(a: &nc_sparse::CsrMatrix<f64>, add: &mut impl FnMut(u64)) {
        add(a.rows() as u64);
        add(a.cols() as u64);
        add(a.nnz() as u64);
        for (r, c, x) in a.iter_entries() {
            add(r as u64);
            add(c as u64);
            add(x.to_bits());
        }
    }
    add(options.rho.to_bits());
    add(options.tolerance.to_bits());
    add(problem.scenarios.len() as u64);
    block(&problem.first_stage, &mut add);
    for s in &problem.scenarios {
        add(s.probability.to_bits());
        block(&s.recourse, &mut add);
        matrix(&s.coupling, &mut add);
    }
    hash
}
