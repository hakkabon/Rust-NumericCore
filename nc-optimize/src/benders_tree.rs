//! Branch-and-bound with audited LP evidence and globally shared lazy rows.

use crate::{
    benders::solve_lp, Bound, LpRow, LpUpdate, LpWorkspace, OptimizeError, Problem,
    RevisedSimplexSolver, Solution, SolveStatus,
};

pub(crate) struct TreeOptions {
    pub solve_limit: usize,
    pub candidate_limit: usize,
    pub lp_iteration_limit: usize,
    pub simplex_tolerance: f64,
    pub evidence_tolerance: f64,
    pub integrality_tolerance: f64,
    pub absolute_gap: f64,
    pub relative_gap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeStop {
    Converged,
    Infeasible,
    NodeLimit,
    CandidateLimit,
    LpIterationLimit,
    MissingEvidence,
    UnboundedRelaxation,
    Stalled,
    EvaluationStopped,
}

pub(crate) struct TreeResult<T> {
    pub stop: TreeStop,
    pub incumbent: Option<(f64, T)>,
    pub lower: Option<f64>,
    pub solves: usize,
    pub candidates: usize,
}

pub(crate) enum Candidate<T> {
    Evaluated {
        incumbent: Option<(f64, T)>,
        rows: Vec<LpRow>,
    },
    Stop,
}

struct Node {
    bounds: Vec<Bound>,
    lower: Option<f64>,
}

pub(crate) fn round_integer_bounds(problem: &mut Problem) -> Result<bool, OptimizeError> {
    const LATTICE_LIMIT: f64 = (1u64 << 52) as f64;
    let mut possible = true;
    for (bound, &integer) in problem.var_bounds.iter_mut().zip(&problem.is_integer) {
        if !integer {
            continue;
        }
        let (Some(lo), Some(hi)) = (bound.lower, bound.upper) else {
            return Err(invalid(
                "integer Benders requires finite bounds on integer variables",
            ));
        };
        if lo.abs().max(hi.abs()) > LATTICE_LIMIT {
            return Err(invalid(
                "integer bounds exceed the supported exact lattice range",
            ));
        }
        let lo = lo.ceil();
        let hi = hi.floor();
        possible &= lo <= hi;
        *bound = Bound {
            lower: Some(lo),
            upper: Some(hi),
        };
    }
    Ok(possible)
}

pub(crate) fn search<T>(
    mut problem: Problem,
    options: &TreeOptions,
    mut evaluate: impl FnMut(&[f64], f64) -> Result<Candidate<T>, OptimizeError>,
) -> Result<TreeResult<T>, OptimizeError> {
    problem.validate()?;
    let mut result = TreeResult {
        stop: TreeStop::Infeasible,
        incumbent: None,
        lower: None,
        solves: 0,
        candidates: 0,
    };
    if !round_integer_bounds(&mut problem)? {
        return Ok(result);
    }
    let integer = std::mem::replace(
        &mut problem.is_integer,
        vec![false; problem.objective.len()],
    );
    let mut frontier = vec![Node {
        bounds: problem.var_bounds.clone(),
        lower: None,
    }];
    let mut closed_lower: Option<f64> = None;
    let mut workspace = LpWorkspace::new(problem, None)?;
    let solver = RevisedSimplexSolver {
        max_iterations: options.lp_iteration_limit,
        tolerance: options.simplex_tolerance,
    };
    loop {
        result.lower = global_lower(&frontier, closed_lower);
        if let (Some((upper, _)), Some(lower)) = (&result.incumbent, result.lower) {
            if gap_closes(*upper, lower, options) {
                result.stop = TreeStop::Converged;
                return Ok(result);
            }
        }
        if frontier.is_empty() {
            result.stop = if result.incumbent.is_some() {
                TreeStop::Stalled
            } else {
                TreeStop::Infeasible
            };
            return Ok(result);
        }
        if result.solves == options.solve_limit {
            result.stop = TreeStop::NodeLimit;
            return Ok(result);
        }

        let next = frontier
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.lower
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&b.lower.unwrap_or(f64::NEG_INFINITY))
            })
            .unwrap()
            .0;
        let mut node = frontier.swap_remove(next);
        if let (Some(lower), Some((upper, _))) = (node.lower, &result.incumbent) {
            if lower >= *upper {
                keep_closed(&mut closed_lower, lower);
                continue;
            }
        }
        let changed = node
            .bounds
            .iter()
            .enumerate()
            .filter(|&(index, bound)| workspace.problem().var_bounds[index] != *bound)
            .map(|(index, &bound)| (index, bound))
            .collect::<Vec<_>>();
        if !changed.is_empty() {
            workspace.update(LpUpdate {
                variable_bounds: changed,
                ..Default::default()
            })?;
        }
        let report = solve_lp(&mut workspace, &solver, options.evidence_tolerance)?;
        result.solves += 1;
        let Some(dual) = report.dual else {
            if report.infeasibility.is_some() {
                continue;
            }
            result.stop = match report.solution.status {
                SolveStatus::IterationLimit => TreeStop::LpIterationLimit,
                SolveStatus::Unbounded if report.unbounded_ray.is_some() => {
                    TreeStop::UnboundedRelaxation
                }
                _ => TreeStop::MissingEvidence,
            };
            frontier.push(node);
            result.lower = global_lower(&frontier, closed_lower);
            return Ok(result);
        };
        let lower = node.lower.map_or(dual.objective_bound, |bound| {
            bound.max(dual.objective_bound)
        });
        node.lower = Some(lower);
        if result
            .incumbent
            .as_ref()
            .is_some_and(|(upper, _)| lower >= *upper)
        {
            keep_closed(&mut closed_lower, lower);
            continue;
        }
        let raw = report.solution.variable_values;
        let fractional = raw
            .iter()
            .zip(&integer)
            .enumerate()
            .filter_map(|(i, (&v, &integer))| {
                let delta = (v - v.round()).abs();
                (integer
                    && delta > 0.0
                    && v.floor() >= node.bounds[i].lower.unwrap()
                    && v.ceil() <= node.bounds[i].upper.unwrap())
                .then_some((i, delta))
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b));
        if let Some((index, _)) =
            fractional.filter(|(_, delta)| *delta > options.integrality_tolerance)
        {
            branch(&mut frontier, node, index, raw[index]);
            continue;
        }

        let point = raw
            .iter()
            .zip(&integer)
            .map(
                |(&value, &integer)| {
                    if integer {
                        value.round()
                    } else {
                        value
                    }
                },
            )
            .collect::<Vec<_>>();
        let objective: f64 = point
            .iter()
            .zip(&workspace.problem().objective)
            .map(|(x, c)| x * c)
            .sum();
        let trial = Solution {
            variable_values: point,
            objective_value: objective,
            status: SolveStatus::Optimal,
        };
        if !trial
            .diagnostics(workspace.problem())
            .is_verified(options.evidence_tolerance)
            || trial
                .variable_values
                .iter()
                .zip(&integer)
                .zip(&node.bounds)
                .any(|((&v, &integer), b)| {
                    integer
                        && (b.lower.is_some_and(|lo| v < lo) || b.upper.is_some_and(|hi| v > hi))
                })
        {
            if let Some((index, _)) = fractional {
                branch(&mut frontier, node, index, raw[index]);
                continue;
            }
            result.stop = TreeStop::MissingEvidence;
        } else if result.candidates == options.candidate_limit {
            result.stop = TreeStop::CandidateLimit;
        } else {
            result.candidates += 1;
            match evaluate(&trial.variable_values, objective)? {
                Candidate::Stop => result.stop = TreeStop::EvaluationStopped,
                Candidate::Evaluated { incumbent, rows } => {
                    if let Some((value, candidate)) = incumbent {
                        if !value.is_finite() {
                            return Err(invalid("non-finite Benders incumbent"));
                        }
                        if (lower - value) / gap_scale(value, lower) > options.evidence_tolerance {
                            result.stop = TreeStop::MissingEvidence;
                            frontier.push(node);
                            result.lower = global_lower(&frontier, closed_lower);
                            return Ok(result);
                        }
                        if result
                            .incumbent
                            .as_ref()
                            .is_none_or(|(old, _)| value < *old)
                        {
                            result.incumbent = Some((value, candidate));
                        }
                    }
                    let added_cuts = !rows.is_empty();
                    if added_cuts {
                        workspace.update(LpUpdate {
                            append_rows: rows,
                            ..Default::default()
                        })?;
                    }
                    if result
                        .incumbent
                        .as_ref()
                        .is_some_and(|(upper, _)| gap_closes(*upper, lower, options))
                    {
                        keep_closed(&mut closed_lower, lower);
                        continue;
                    }
                    if let Some((index, _)) = fractional {
                        branch(&mut frontier, node, index, raw[index]);
                        continue;
                    }
                    if added_cuts {
                        frontier.push(node);
                        continue;
                    }
                    result.stop = TreeStop::Stalled;
                }
            }
        }
        frontier.push(node);
        result.lower = global_lower(&frontier, closed_lower);
        return Ok(result);
    }
}

fn branch(frontier: &mut Vec<Node>, node: Node, index: usize, value: f64) {
    let mut right = Node {
        bounds: node.bounds.clone(),
        lower: node.lower,
    };
    let mut left = node;
    left.bounds[index].upper = Some(value.floor());
    right.bounds[index].lower = Some(value.ceil());
    frontier.push(left);
    frontier.push(right);
}

fn keep_closed(current: &mut Option<f64>, lower: f64) {
    *current = Some(current.map_or(lower, |bound| bound.min(lower)));
}

fn global_lower(frontier: &[Node], mut known: Option<f64>) -> Option<f64> {
    for node in frontier {
        keep_closed(&mut known, node.lower?);
    }
    known
}

fn gap_scale(upper: f64, lower: f64) -> f64 {
    upper.abs().max(lower.abs()).max(1.0)
}

fn gap_closes(upper: f64, lower: f64, options: &TreeOptions) -> bool {
    let scale = gap_scale(upper, lower);
    let difference = upper - lower;
    let relative = difference / scale;
    if relative < -options.evidence_tolerance {
        return false;
    }
    difference <= options.absolute_gap
        || (if difference.is_finite() {
            relative
        } else {
            upper / scale - lower / scale
        }) <= options.relative_gap
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}
