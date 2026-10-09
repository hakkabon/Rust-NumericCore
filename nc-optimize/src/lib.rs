//! Optimization layer — the numeric target the AMPL-style modeling
//! language (`NumericCoreAMPL`, Swift side) compiles down to.
//!
//! Mirrors AMPL's own architecture: the modeling language and the solver
//! never talk directly. A `Problem` (this crate's ASL-equivalent handoff
//! format) sits between them, so any solver can consume any presolved
//! model without knowing anything about modeling-language syntax.
//!
//! The linear `Problem`/`Solver` seam is consumed by revised-simplex,
//! interior-point, and branch-and-bound. Smooth nonlinear objectives use
//! closure-based L-BFGS and Levenberg-Marquardt APIs because their callback
//! structure is materially different from a coefficient-matrix problem.

use nc_sparse::CsrMatrix;
use thiserror::Error;

pub mod simplex;
pub use simplex::RevisedSimplexSolver;

pub mod interior_point;
pub use interior_point::InteriorPointSolver;

pub mod branch_and_bound;
pub use branch_and_bound::{
    BranchAndBoundReport, BranchAndBoundSolver, BranchAndBoundTermination, BranchingStrategy,
    NodeSelection,
};

pub mod lbfgs;
pub use lbfgs::{
    minimize_lbfgs, minimize_lbfgs_with_observer, LbfgsIteration, LbfgsOptions,
    LbfgsResult, NonlinearTermination,
};

pub mod lbfgsb;
pub use lbfgsb::{minimize_lbfgsb, minimize_lbfgsb_with_observer};

pub mod nonlinear_least_squares;
pub use nonlinear_least_squares::{
    nonlinear_least_squares, nonlinear_least_squares_configured,
    nonlinear_least_squares_configured_with_observer, nonlinear_least_squares_with_observer,
    NonlinearLeastSquaresIteration, NonlinearLeastSquaresOptions, NonlinearLeastSquaresResult,
    RobustLoss,
};

pub mod derivative_check;
pub use derivative_check::{check_gradient, check_jacobian, DerivativeCheckReport};

pub mod nonlinear_model;
pub use nonlinear_model::{
    minimize_model_lbfgs, minimize_model_lbfgsb, solve_model_least_squares,
    DifferentiableValue, NonlinearExpression, NonlinearModel, NonlinearNode,
};

pub mod sparse_derivatives;
pub use sparse_derivatives::{SparseDerivative, SparseJacobian};

pub mod second_order;
pub use second_order::{
    solve_sparse_kkt, SecondOrderValue, SparseHessian, SparseKktProblem, SparseKktResult,
};

pub mod sqp;
pub use sqp::{
    minimize_sqp, minimize_sqp_with_observer, SqpCurvature, SqpGlobalization, SqpIteration,
    SqpOptions, SqpResult, SqpTermination,
};

pub mod feasibility_restoration;
pub use feasibility_restoration::{
    restore_feasibility, FeasibilityRestorationOptions, FeasibilityRestorationResult,
    FeasibilityRestorationTermination,
};

pub mod constrained_nonlinear;
pub use constrained_nonlinear::{
    minimize_constrained, minimize_constrained_with_observer, ConstrainedIteration,
    ConstrainedNonlinearProblem, ConstrainedOptions, ConstrainedResult, ConstrainedTermination,
    ConstraintMultiplier, NonlinearConstraint,
};

pub mod nonlinear_interior_point;
pub use nonlinear_interior_point::{
    minimize_nonlinear_interior_point, minimize_nonlinear_interior_point_with_observer,
    NonlinearInteriorPointIteration, NonlinearInteriorPointOptions,
    NonlinearInteriorPointResult, NonlinearInteriorPointTermination,
};

pub mod mixed_integer_nonlinear;
pub use mixed_integer_nonlinear::{
    minimize_mixed_integer_nonlinear, minimize_mixed_integer_nonlinear_with_observer,
    MixedIntegerNonlinearIteration, MixedIntegerNonlinearOptions, MixedIntegerNonlinearProblem,
    MixedIntegerNonlinearResult, MixedIntegerNonlinearTermination, NonlinearRelaxationStrategy,
};

pub mod quadratic_program;
pub use quadratic_program::{
    solve_convex_qp, solve_convex_qp_warm, solve_convex_qp_with_observer,
    QuadraticIteration, QuadraticOptions, QuadraticProblem, QuadraticResult,
    QuadraticTermination, QuadraticWarmStart,
};

pub mod scaling;
pub use scaling::{ScaledProblem, ScaledSolver, ScalingOptions, ScalingReport};

/// Bound on a variable or constraint. `None` means unbounded in that
/// direction (`-inf` / `+inf`).
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// `is_integer[j]` restricts variable `j` to integer values.
    /// Length must match `objective`/`var_bounds`. All-`false` recovers
    /// a pure LP — `RevisedSimplexSolver`/`InteriorPointSolver` both
    /// ignore this field entirely (they solve the LP relaxation
    /// regardless of its contents); only `BranchAndBoundSolver` reads
    /// it. Added alongside `BranchAndBoundSolver` rather than
    /// speculatively with the original `Problem` definition — see
    /// `docs/decisions/0004-optimize-sequencing.md`'s update for why
    /// this is exactly the "breaking addition" that ADR anticipated
    /// MILP would eventually need.
    pub is_integer: Vec<bool>,
}

impl Problem {
    /// Validate the solver-independent LP/MILP handoff before any algorithm
    /// indexes its parallel vectors or relies on finite arithmetic.
    pub fn validate(&self) -> Result<(), OptimizeError> {
        let variables = self.objective.len();
        if self.constraints.cols() != variables
            || self.var_bounds.len() != variables
            || self.is_integer.len() != variables
            || self.row_bounds.len() != self.constraints.rows()
        {
            return Err(OptimizeError::InvalidProblem(
                "objective, matrix, bounds, and integrality dimensions do not agree".to_owned(),
            ));
        }
        if !self.objective.iter().all(|value| value.is_finite())
            || !self.constraints.iter_entries().all(|(_, _, value)| value.is_finite())
        {
            return Err(OptimizeError::InvalidProblem(
                "objective and constraint coefficients must be finite".to_owned(),
            ));
        }
        for bound in self.row_bounds.iter().chain(&self.var_bounds) {
            if bound.lower.is_some_and(|value| !value.is_finite())
                || bound.upper.is_some_and(|value| !value.is_finite())
                || matches!((bound.lower, bound.upper), (Some(lower), Some(upper)) if lower > upper)
            {
                return Err(OptimizeError::InvalidProblem(
                    "bounds must be finite when present and lower must not exceed upper".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub variable_values: Vec<f64>,
    pub objective_value: f64,
    pub status: SolveStatus,
}

/// Independently recomputed correctness diagnostics for a returned solution.
///
/// Solvers remain free to use different internal representations. This report
/// is deliberately computed from the public `Problem` and final values so it
/// can detect translation, presolve, and solver bookkeeping errors as well as
/// ordinary numerical drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolutionDiagnostics {
    pub finite: bool,
    pub maximum_row_violation: f64,
    pub maximum_variable_bound_violation: f64,
    pub maximum_integrality_violation: f64,
    pub recomputed_objective: f64,
    pub objective_error: f64,
}

impl SolutionDiagnostics {
    pub fn is_verified(&self, tolerance: f64) -> bool {
        tolerance.is_finite()
            && tolerance >= 0.0
            && self.finite
            && self.maximum_row_violation <= tolerance
            && self.maximum_variable_bound_violation <= tolerance
            && self.maximum_integrality_violation <= tolerance
            && self.objective_error <= tolerance
    }
}

impl Solution {
    /// Re-evaluate feasibility, integrality, and the objective independently
    /// of the solver that produced this solution.
    pub fn diagnostics(&self, problem: &Problem) -> SolutionDiagnostics {
        if self.variable_values.len() != problem.objective.len()
            || problem.var_bounds.len() != problem.objective.len()
            || problem.constraints.cols() != problem.objective.len()
            || problem.row_bounds.len() != problem.constraints.rows()
            || (!problem.is_integer.is_empty()
                && problem.is_integer.len() != problem.objective.len())
        {
            return SolutionDiagnostics {
                finite: false,
                maximum_row_violation: f64::INFINITY,
                maximum_variable_bound_violation: f64::INFINITY,
                maximum_integrality_violation: f64::INFINITY,
                recomputed_objective: f64::NAN,
                objective_error: f64::INFINITY,
            };
        }

        let recomputed_objective: f64 = problem
            .objective
            .iter()
            .zip(&self.variable_values)
            .map(|(coefficient, value)| coefficient * value)
            .sum();
        let mut row_values = vec![0.0; problem.constraints.rows()];
        for (row, column, value) in problem.constraints.iter_entries() {
            row_values[row] += value * self.variable_values[column];
        }

        let maximum_row_violation = row_values
            .iter()
            .zip(&problem.row_bounds)
            .map(|(&value, bound)| bound_violation(value, *bound))
            .fold(0.0, f64::max);
        let maximum_variable_bound_violation = self
            .variable_values
            .iter()
            .zip(&problem.var_bounds)
            .map(|(&value, bound)| bound_violation(value, *bound))
            .fold(0.0, f64::max);
        let maximum_integrality_violation = self
            .variable_values
            .iter()
            .enumerate()
            .filter(|(index, _)| problem.is_integer.get(*index).copied().unwrap_or(false))
            .map(|(_, value)| (value - value.round()).abs())
            .fold(0.0, f64::max);
        let objective_error = (self.objective_value - recomputed_objective).abs();
        let finite = self.objective_value.is_finite()
            && self.variable_values.iter().all(|value| value.is_finite())
            && row_values.iter().all(|value| value.is_finite())
            && recomputed_objective.is_finite()
            && maximum_row_violation.is_finite()
            && maximum_variable_bound_violation.is_finite()
            && maximum_integrality_violation.is_finite()
            && objective_error.is_finite();

        SolutionDiagnostics {
            finite,
            maximum_row_violation,
            maximum_variable_bound_violation,
            maximum_integrality_violation,
            recomputed_objective,
            objective_error,
        }
    }
}

fn bound_violation(value: f64, bound: Bound) -> f64 {
    let lower = bound.lower.map_or(0.0, |limit| (limit - value).max(0.0));
    let upper = bound.upper.map_or(0.0, |limit| (value - limit).max(0.0));
    lower.max(upper)
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

    #[error("invalid solver configuration: {0}")]
    InvalidConfiguration(&'static str),

    #[error("invalid optimization problem: {0}")]
    InvalidProblem(String),

    #[error(transparent)]
    Sparse(#[from] nc_sparse::SparseError),
}

/// The ASL-equivalent seam: any solver plugs in here without the
/// modeling layer or presolve code needing to change.
pub trait Solver {
    fn name(&self) -> &'static str;
    fn solve(&self, problem: &Problem) -> Result<Solution, OptimizeError>;

    /// Solve and independently recompute the final numerical invariants.
    /// This is additive to `solve` so existing solver implementations and FFI
    /// callers retain their current ABI.
    fn solve_with_diagnostics(
        &self,
        problem: &Problem,
    ) -> Result<(Solution, SolutionDiagnostics), OptimizeError> {
        let solution = self.solve(problem)?;
        let diagnostics = solution.diagnostics(problem);
        Ok((solution, diagnostics))
    }
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
            is_integer: vec![false],
        };
        let result = StubSolver.solve(&problem);
        assert!(matches!(result, Err(OptimizeError::NotImplemented(_))));
    }

    #[test]
    fn diagnostics_independently_verify_a_feasible_integer_solution() {
        let problem = Problem {
            objective: vec![3.0, 2.0],
            constraints: CsrMatrix::new(
                1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0],
            )
            .unwrap(),
            row_bounds: vec![Bound { lower: None, upper: Some(4.0) }],
            var_bounds: vec![
                Bound { lower: Some(0.0), upper: None },
                Bound { lower: Some(0.0), upper: None },
            ],
            is_integer: vec![true, true],
        };
        let solution = Solution {
            variable_values: vec![2.0, 2.0],
            objective_value: 10.0,
            status: SolveStatus::Optimal,
        };

        let diagnostics = solution.diagnostics(&problem);
        assert!(diagnostics.is_verified(1e-12));
        assert_eq!(diagnostics.recomputed_objective, 10.0);
    }

    #[test]
    fn diagnostics_expose_each_incorrectness_category() {
        let problem = Problem {
            objective: vec![1.0],
            constraints: CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
            row_bounds: vec![Bound { lower: None, upper: Some(1.0) }],
            var_bounds: vec![Bound { lower: Some(0.0), upper: Some(1.0) }],
            is_integer: vec![true],
        };
        let solution = Solution {
            variable_values: vec![1.5],
            objective_value: 9.0,
            status: SolveStatus::Optimal,
        };

        let diagnostics = solution.diagnostics(&problem);
        assert_eq!(diagnostics.maximum_row_violation, 0.5);
        assert_eq!(diagnostics.maximum_variable_bound_violation, 0.5);
        assert_eq!(diagnostics.maximum_integrality_violation, 0.5);
        assert_eq!(diagnostics.objective_error, 7.5);
        assert!(!diagnostics.is_verified(1e-9));
    }

    #[test]
    fn solvers_reject_invalid_numerical_configuration() {
        let problem = Problem {
            objective: vec![1.0],
            constraints: CsrMatrix::new(0, 1, vec![0], vec![], vec![]).unwrap(),
            row_bounds: vec![],
            var_bounds: vec![Bound { lower: Some(0.0), upper: Some(1.0) }],
            is_integer: vec![true],
        };

        let simplex = RevisedSimplexSolver { max_iterations: 0, tolerance: 1e-9 };
        assert!(matches!(
            simplex.solve(&problem),
            Err(OptimizeError::InvalidConfiguration(_))
        ));

        let mut interior = InteriorPointSolver::default();
        interior.tolerance = f64::NAN;
        assert!(matches!(
            interior.solve(&problem),
            Err(OptimizeError::InvalidConfiguration(_))
        ));

        let mut branch = BranchAndBoundSolver::default();
        branch.integer_tolerance = 0.0;
        assert!(matches!(
            branch.solve(&problem),
            Err(OptimizeError::InvalidConfiguration(_))
        ));
    }
}
