//! Scenario-tree assembly and expected-cost / mean-CVaR optimization.

use crate::{
    benders_tree::{search, Candidate, TreeOptions, TreeStop},
    BendersProblem, Bound, OptimizeError, Problem,
};
use nc_sparse::{CooMatrix, CsrMatrix};
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct ScenarioLink {
    pub ancestor: usize,
    pub coefficients: CsrMatrix<f64>,
}

/// The LP block at an information node. Ancestor terms are added to the
/// activities of the node's rows. Root probability is one.
#[derive(Debug, Clone)]
pub struct ScenarioNode {
    pub id: String,
    pub parent: Option<usize>,
    pub conditional_probability: f64,
    pub block: Problem,
    pub links: Vec<ScenarioLink>,
}

#[derive(Debug, Clone)]
pub struct ScenarioTree {
    pub nodes: Vec<ScenarioNode>,
}

impl ScenarioTree {
    pub fn two_stage(problem: &BendersProblem) -> Self {
        let mut nodes = vec![ScenarioNode {
            id: "first-stage".into(),
            parent: None,
            conditional_probability: 1.0,
            block: problem.first_stage.clone(),
            links: Vec::new(),
        }];
        nodes.extend(
            problem
                .scenarios
                .iter()
                .enumerate()
                .map(|(i, s)| ScenarioNode {
                    id: format!("{}:{}", i + 1, s.id.name),
                    parent: Some(0),
                    conditional_probability: s.probability,
                    block: s.recourse.clone(),
                    links: vec![ScenarioLink {
                        ancestor: 0,
                        coefficients: s.coupling.clone(),
                    }],
                }),
        );
        Self { nodes }
    }

    /// Assemble the extensive form with one copy of each information-node decision.
    pub fn extensive_form(&self, risk: ScenarioRisk) -> Result<Problem, OptimizeError> {
        Ok(assemble(self, risk)?.problem)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScenarioRisk {
    ExpectedValue,
    /// Minimize (1 - weight) times the mean plus weight times upper-tail CVaR.
    MeanCvar {
        alpha: f64,
        weight: f64,
    },
}

impl ScenarioRisk {
    fn validate(self) -> Result<(), OptimizeError> {
        if let Self::MeanCvar { alpha, weight } = self {
            if !alpha.is_finite()
                || !(0.0..1.0).contains(&alpha)
                || !weight.is_finite()
                || !(0.0..=1.0).contains(&weight)
            {
                return Err(OptimizeError::InvalidConfiguration(
                    "invalid mean-CVaR parameters",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScenarioSolveOptions {
    pub max_nodes: usize,
    pub lp_iteration_limit: usize,
    pub tolerance: f64,
}

impl Default for ScenarioSolveOptions {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            lp_iteration_limit: 10_000,
            tolerance: 1e-8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioTermination {
    Converged,
    Infeasible,
    NodeLimit,
    LpIterationLimit,
    MissingEvidence,
    UnboundedRelaxation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioCost {
    pub id: String,
    pub probability: f64,
    pub cost: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioPolicy {
    /// One decision vector per input node, in input order.
    pub decisions: Vec<Vec<f64>>,
    pub terminal_costs: Vec<ScenarioCost>,
    pub expected_cost: f64,
    pub value_at_risk: Option<f64>,
    pub conditional_value_at_risk: Option<f64>,
    pub objective: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioSolveResult {
    pub termination: ScenarioTermination,
    pub policy: Option<ScenarioPolicy>,
    pub lower_bound: Option<f64>,
    pub nodes: usize,
}

pub fn solve_scenario_tree(
    tree: &ScenarioTree,
    risk: ScenarioRisk,
    options: ScenarioSolveOptions,
) -> Result<ScenarioSolveResult, OptimizeError> {
    if options.max_nodes == 0
        || options.lp_iteration_limit == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.tolerance >= 0.5
    {
        return Err(OptimizeError::InvalidConfiguration(
            "invalid scenario solve options",
        ));
    }
    let assembly = assemble(tree, risk)?;
    let options = TreeOptions {
        solve_limit: options.max_nodes,
        candidate_limit: options.max_nodes,
        lp_iteration_limit: options.lp_iteration_limit,
        simplex_tolerance: options.tolerance.min(1e-9),
        evidence_tolerance: options.tolerance,
        integrality_tolerance: options.tolerance,
        absolute_gap: options.tolerance,
        relative_gap: options.tolerance,
    };
    let result = search(assembly.problem, &options, |point, objective| {
        Ok(Candidate::Evaluated {
            incumbent: Some((
                objective,
                point[..assembly.offsets[tree.nodes.len()]].to_vec(),
            )),
            rows: vec![],
        })
    })?;
    let policy = result
        .incumbent
        .map(|(_, point)| {
            let decisions: Vec<Vec<f64>> = assembly
                .offsets
                .windows(2)
                .map(|window| point[window[0]..window[1]].to_vec())
                .collect();
            policy_costs(tree, &assembly.mass, &assembly.leaves, decisions, risk)
        })
        .transpose()?;
    Ok(ScenarioSolveResult {
        termination: match result.stop {
            TreeStop::Converged => ScenarioTermination::Converged,
            TreeStop::Infeasible => ScenarioTermination::Infeasible,
            TreeStop::NodeLimit | TreeStop::CandidateLimit => ScenarioTermination::NodeLimit,
            TreeStop::LpIterationLimit => ScenarioTermination::LpIterationLimit,
            TreeStop::UnboundedRelaxation => ScenarioTermination::UnboundedRelaxation,
            _ => ScenarioTermination::MissingEvidence,
        },
        policy,
        lower_bound: result.lower,
        nodes: result.solves,
    })
}

struct Assembly {
    problem: Problem,
    mass: Vec<f64>,
    leaves: Vec<usize>,
    offsets: Vec<usize>,
}

fn assemble(tree: &ScenarioTree, risk: ScenarioRisk) -> Result<Assembly, OptimizeError> {
    risk.validate()?;
    let (mass, leaves) = tree_probabilities(tree)?;
    let (alpha, weight) = match risk {
        ScenarioRisk::ExpectedValue => (0.0, 0.0),
        ScenarioRisk::MeanCvar { alpha, weight } if alpha > 0.0 => (alpha, weight),
        ScenarioRisk::MeanCvar { .. } => (0.0, 0.0),
    };
    let mut offsets = vec![0];
    let mut objective = Vec::new();
    let mut row_bounds = Vec::new();
    let mut var_bounds = Vec::new();
    let mut is_integer = Vec::new();
    for (node, &probability) in tree.nodes.iter().zip(&mass) {
        objective.extend(
            node.block
                .objective
                .iter()
                .map(|c| (1.0 - weight) * probability * c),
        );
        row_bounds.extend(&node.block.row_bounds);
        var_bounds.extend(&node.block.var_bounds);
        is_integer.extend(&node.block.is_integer);
        offsets.push(objective.len());
    }
    let threshold = objective.len();
    let mut matrix = CooMatrix::new(
        row_bounds.len() + if weight > 0.0 { leaves.len() } else { 0 },
        threshold + if weight > 0.0 { leaves.len() + 1 } else { 0 },
    );
    let mut next_row = 0;
    for (i, node) in tree.nodes.iter().enumerate() {
        for (r, c, v) in node.block.constraints.iter_entries() {
            matrix.push(next_row + r, offsets[i] + c, v)?;
        }
        for link in &node.links {
            for (r, c, v) in link.coefficients.iter_entries() {
                matrix.push(next_row + r, offsets[link.ancestor] + c, v)?;
            }
        }
        next_row += node.block.row_bounds.len();
    }
    if weight > 0.0 {
        objective.push(weight);
        var_bounds.push(Bound::free());
        is_integer.push(false);
        for (i, &leaf) in leaves.iter().enumerate() {
            let tail = threshold + 1 + i;
            objective.push((weight * mass[leaf]) / (1.0 - alpha));
            var_bounds.push(Bound {
                lower: Some(0.0),
                upper: None,
            });
            is_integer.push(false);
            row_bounds.push(Bound {
                lower: None,
                upper: Some(0.0),
            });
            for node in path(tree, leaf) {
                for (j, &c) in tree.nodes[node].block.objective.iter().enumerate() {
                    matrix.push(next_row, offsets[node] + j, c)?;
                }
            }
            matrix.push(next_row, threshold, -1.0)?;
            matrix.push(next_row, tail, -1.0)?;
            next_row += 1;
        }
    }
    let problem = Problem {
        objective,
        constraints: matrix.to_csr()?,
        row_bounds,
        var_bounds,
        is_integer,
    };
    problem.validate()?;
    Ok(Assembly {
        problem,
        mass,
        leaves,
        offsets,
    })
}

fn tree_probabilities(tree: &ScenarioTree) -> Result<(Vec<f64>, Vec<usize>), OptimizeError> {
    let n = tree.nodes.len();
    if n == 0 {
        return Err(invalid("scenario tree requires a root"));
    }
    let mut children = vec![Vec::new(); n];
    let mut ids = BTreeSet::new();
    for (i, node) in tree.nodes.iter().enumerate() {
        node.block.validate()?;
        if node.id.trim().is_empty() || !ids.insert(&node.id) {
            return Err(invalid("scenario node IDs must be nonempty and unique"));
        }
        match node.parent {
            None if i == 0 && node.conditional_probability == 1.0 => {}
            Some(parent) if parent < i => children[parent].push(i),
            _ => return Err(invalid("scenario parents must precede their children")),
        }
        let ancestors: BTreeSet<_> = path(tree, i).skip(1).collect();
        let mut linked = BTreeSet::new();
        for link in &node.links {
            if !ancestors.contains(&link.ancestor)
                || !linked.insert(link.ancestor)
                || link.coefficients.rows() != node.block.row_bounds.len()
                || link.coefficients.cols() != tree.nodes[link.ancestor].block.objective.len()
                || link
                    .coefficients
                    .iter_entries()
                    .any(|(_, _, v)| !v.is_finite())
            {
                return Err(invalid("invalid scenario ancestor link"));
            }
        }
    }
    let mut mass = vec![0.0; n];
    mass[0] = 1.0;
    for (parent, children) in children.iter().enumerate().filter(|(_, c)| !c.is_empty()) {
        let weights = probabilities(
            children
                .iter()
                .map(|&i| tree.nodes[i].conditional_probability),
        )?;
        for (&child, weight) in children.iter().zip(weights) {
            mass[child] = mass[parent] * weight;
            if mass[child] <= 0.0 {
                return Err(invalid("scenario path probability underflow"));
            }
        }
    }
    let leaves = children
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.is_empty().then_some(i))
        .collect();
    Ok((mass, leaves))
}

fn path(tree: &ScenarioTree, leaf: usize) -> impl Iterator<Item = usize> + '_ {
    std::iter::successors(Some(leaf), |&i| tree.nodes[i].parent)
}

fn policy_costs(
    tree: &ScenarioTree,
    mass: &[f64],
    leaves: &[usize],
    decisions: Vec<Vec<f64>>,
    risk: ScenarioRisk,
) -> Result<ScenarioPolicy, OptimizeError> {
    let mut costs = Vec::new();
    for (node, x) in tree.nodes.iter().zip(&decisions) {
        let parent = node.parent.map_or(0.0, |i| costs[i]);
        let cost = parent
            + node
                .block
                .objective
                .iter()
                .zip(x)
                .map(|(c, x)| c * x)
                .sum::<f64>();
        if !cost.is_finite() {
            return Err(invalid("non-finite scenario path cost"));
        }
        costs.push(cost);
    }
    let terminal_costs: Vec<_> = leaves
        .iter()
        .map(|&i| ScenarioCost {
            id: tree.nodes[i].id.clone(),
            probability: mass[i],
            cost: costs[i],
        })
        .collect();
    let expected_cost = terminal_costs
        .iter()
        .map(|leaf| leaf.probability * leaf.cost)
        .sum::<f64>();
    let (var, cvar, objective) = match risk {
        ScenarioRisk::ExpectedValue => (None, None, expected_cost),
        ScenarioRisk::MeanCvar { alpha, weight } => {
            let (var, cvar) = tail_metrics(&terminal_costs, alpha);
            (
                Some(var),
                Some(cvar),
                (1.0 - weight) * expected_cost + weight * cvar,
            )
        }
    };
    if !expected_cost.is_finite() || !objective.is_finite() || cvar.is_some_and(|v| !v.is_finite())
    {
        return Err(invalid("non-finite scenario risk"));
    }
    Ok(ScenarioPolicy {
        decisions,
        terminal_costs,
        expected_cost,
        value_at_risk: var,
        conditional_value_at_risk: cvar,
        objective,
    })
}

fn tail_metrics(costs: &[ScenarioCost], alpha: f64) -> (f64, f64) {
    let mut sorted: Vec<_> = costs.iter().collect();
    sorted.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    let mut cumulative = 0.0;
    let var = sorted
        .iter()
        .find(|leaf| {
            cumulative += leaf.probability;
            cumulative >= alpha
        })
        .unwrap_or_else(|| sorted.last().unwrap())
        .cost;
    let tail = 1.0 - alpha;
    let mut remaining = tail;
    let mut cvar = 0.0;
    for leaf in sorted.into_iter().rev() {
        let amount = remaining.min(leaf.probability);
        cvar += (amount / tail) * leaf.cost;
        remaining -= amount;
        if remaining == 0.0 {
            break;
        }
    }
    (var, cvar + (remaining / tail) * var)
}

pub(crate) fn probabilities(values: impl Iterator<Item = f64>) -> Result<Vec<f64>, OptimizeError> {
    let values: Vec<_> = values.collect();
    let sum: f64 = values.iter().sum();
    if values.iter().any(|p| !p.is_finite() || *p <= 0.0) || (sum - 1.0).abs() > 1e-10 {
        return Err(invalid("positive scenario probabilities must sum to one"));
    }
    Ok(values.into_iter().map(|p| p / sum).collect())
}

pub(crate) fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
