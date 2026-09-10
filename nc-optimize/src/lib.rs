//! Optimization layer — the numeric target the AMPL-style modeling
//! language (`NumericCoreAMPL`, Swift side) compiles down to.
//!
//! Mirrors AMPL's own architecture: the modeling language and the solver
//! never talk directly. A `Problem` (this crate's ASL-equivalent handoff
//! format) sits between them, so any solver can consume any presolved
//! model without knowing anything about modeling-language syntax.
//!
//! v1 scope: the `Problem` data model and the `Solver` trait exist now;
//! only a stub solver is implemented today. Real solving is sequenced as
//! LP → MILP → NLP, each a substantial, separately-scoped effort — see
//! `docs/decisions/0004-optimize-sequencing.md`. Don't be surprised that
//! `StubSolver` is the only thing that runs right now; that's intentional
//! so the modeling-language → presolve → solver-interface pipeline can be
//! wired and tested end to end before any real solving exists.

use nc_sparse::CsrMatrix;
use thiserror::Error;

/// Bound on a variable or constraint. `None` means unbounded in that
/// direction (`-inf` / `+inf`).
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl Bound {
    pub fn free() -> Self {
        Self { lower: None, upper: None }
    }

    pub fn fixed(value: f64) -> Self {
        Self { lower: Some(value), upper: Some(value) }
    }
}

/// Linear program in standard presolved form:
///
/// ```text
/// minimize    c^T x
/// subject to  row_bounds.lower <= A x <= row_bounds.upper
///             var_bounds.lower <= x   <= var_bounds.upper
/// ```
///
/// This is the artifact the AMPL presolve layer produces and the only
/// thing a `Solver` implementation needs to understand — it has no
/// knowledge of sets, indexing expressions, or any modeling-language
/// syntax. Row/variable bounds subsume `<=`, `>=`, `=`, and range
/// constraints uniformly, rather than the solver needing a case per
/// constraint sense.
#[derive(Debug, Clone)]
pub struct Problem {
    pub objective: Vec<f64>,
    pub constraints: CsrMatrix<f64>,
    pub row_bounds: Vec<Bound>,
    pub var_bounds: Vec<Bound>,
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub variable_values: Vec<f64>,
    pub objective_value: f64,
    pub status: SolveStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
}

#[derive(Debug, Error)]
pub enum OptimizeError {
    #[error("solver does not yet implement this problem class: {0}")]
    NotImplemented(&'static str),

    #[error(transparent)]
    Sparse(#[from] nc_sparse::SparseError),
}

/// The ASL-equivalent seam: any solver plugs in here without the
/// modeling layer or presolve code needing to change.
pub trait Solver {
    fn name(&self) -> &'static str;
    fn solve(&self, problem: &Problem) -> Result<Solution, OptimizeError>;
}

/// Placeholder solver so the pipeline (parse → presolve → `Problem` →
/// `Solver` → `Solution`) can be built and tested end to end before a
/// real simplex/interior-point implementation lands.
pub struct StubSolver;

impl Solver for StubSolver {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn solve(&self, _problem: &Problem) -> Result<Solution, OptimizeError> {
        Err(OptimizeError::NotImplemented("LP simplex/interior-point solver"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_solver_reports_not_implemented() {
        let problem = Problem {
            objective: vec![1.0],
            constraints: CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
            row_bounds: vec![Bound { lower: None, upper: Some(10.0) }],
            var_bounds: vec![Bound { lower: Some(0.0), upper: None }],
        };
        let result = StubSolver.solve(&problem);
        assert!(matches!(result, Err(OptimizeError::NotImplemented(_))));
    }
}
