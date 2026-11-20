use crate::{
    from_domain_solution, to_domain_bound, to_domain_problem, FfiBound, FfiError, FfiProblem,
    FfiSolution,
};
use nc_optimize as opt;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, uniffi::Enum)]
pub enum FfiLpBasisStatus {
    Basic,
    AtLower,
    AtUpper,
    Free,
}

impl From<opt::LpBasisStatus> for FfiLpBasisStatus {
    fn from(status: opt::LpBasisStatus) -> Self {
        match status {
            opt::LpBasisStatus::Basic => Self::Basic,
            opt::LpBasisStatus::AtLower => Self::AtLower,
            opt::LpBasisStatus::AtUpper => Self::AtUpper,
            opt::LpBasisStatus::Free => Self::Free,
        }
    }
}

impl From<FfiLpBasisStatus> for opt::LpBasisStatus {
    fn from(status: FfiLpBasisStatus) -> Self {
        match status {
            FfiLpBasisStatus::Basic => Self::Basic,
            FfiLpBasisStatus::AtLower => Self::AtLower,
            FfiLpBasisStatus::AtUpper => Self::AtUpper,
            FfiLpBasisStatus::Free => Self::Free,
        }
    }
}

#[derive(uniffi::Record)]
pub struct FfiLpBasis {
    pub variables: Vec<FfiLpBasisStatus>,
    pub rows: Vec<FfiLpBasisStatus>,
}

#[derive(uniffi::Record)]
pub struct FfiLpSubproblemId {
    pub name: String,
    pub scenario: Option<String>,
}

#[derive(uniffi::Record)]
pub struct FfiLpEntry {
    pub index: u32,
    pub value: f64,
}

#[derive(uniffi::Record)]
pub struct FfiLpBoundEdit {
    pub index: u32,
    pub bound: FfiBound,
}

#[derive(uniffi::Record)]
pub struct FfiLpRow {
    pub entries: Vec<FfiLpEntry>,
    pub bound: FfiBound,
}

#[derive(uniffi::Record)]
pub struct FfiLpColumn {
    pub entries: Vec<FfiLpEntry>,
    pub bound: FfiBound,
    pub objective: f64,
}

#[derive(uniffi::Record)]
pub struct FfiLpUpdate {
    pub objective: Vec<FfiLpEntry>,
    pub variable_bounds: Vec<FfiLpBoundEdit>,
    pub row_bounds: Vec<FfiLpBoundEdit>,
    pub append_rows: Vec<FfiLpRow>,
    pub append_columns: Vec<FfiLpColumn>,
}

impl From<FfiLpUpdate> for opt::LpUpdate {
    fn from(update: FfiLpUpdate) -> Self {
        fn entries(values: Vec<FfiLpEntry>) -> Vec<(usize, f64)> {
            values
                .into_iter()
                .map(|e| (e.index as usize, e.value))
                .collect()
        }
        fn bounds(values: Vec<FfiLpBoundEdit>) -> Vec<(usize, opt::Bound)> {
            values
                .into_iter()
                .map(|e| (e.index as usize, to_domain_bound(e.bound)))
                .collect()
        }
        Self {
            objective: entries(update.objective),
            variable_bounds: bounds(update.variable_bounds),
            row_bounds: bounds(update.row_bounds),
            append_rows: update
                .append_rows
                .into_iter()
                .map(|row| opt::LpRow {
                    entries: entries(row.entries),
                    bound: to_domain_bound(row.bound),
                })
                .collect(),
            append_columns: update
                .append_columns
                .into_iter()
                .map(|column| opt::LpColumn {
                    entries: entries(column.entries),
                    bound: to_domain_bound(column.bound),
                    objective: column.objective,
                })
                .collect(),
        }
    }
}

#[derive(uniffi::Record)]
pub struct FfiLpDualSolution {
    pub row_multipliers: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub objective_bound: f64,
    pub absolute_gap: f64,
}

impl From<opt::LpDualSolution> for FfiLpDualSolution {
    fn from(value: opt::LpDualSolution) -> Self {
        Self {
            row_multipliers: value.row_multipliers,
            reduced_costs: value.reduced_costs,
            objective_bound: value.objective_bound,
            absolute_gap: value.absolute_gap,
        }
    }
}

#[derive(uniffi::Record)]
pub struct FfiLpFarkasCertificate {
    pub row_multipliers: Vec<f64>,
    pub variable_multipliers: Vec<f64>,
    pub contradiction_margin: f64,
}

impl From<opt::LpFarkasCertificate> for FfiLpFarkasCertificate {
    fn from(value: opt::LpFarkasCertificate) -> Self {
        Self {
            row_multipliers: value.row_multipliers,
            variable_multipliers: value.variable_multipliers,
            contradiction_margin: value.contradiction_margin,
        }
    }
}

#[derive(uniffi::Record)]
pub struct FfiLpUnboundedRay {
    pub direction: Vec<f64>,
    pub objective_slope: f64,
}

impl From<opt::LpUnboundedRay> for FfiLpUnboundedRay {
    fn from(value: opt::LpUnboundedRay) -> Self {
        Self {
            direction: value.direction,
            objective_slope: value.objective_slope,
        }
    }
}

#[derive(uniffi::Record)]
pub struct FfiLpSolveReport {
    pub solution: FfiSolution,
    pub basis: FfiLpBasis,
    pub dual: Option<FfiLpDualSolution>,
    pub infeasibility: Option<FfiLpFarkasCertificate>,
    pub unbounded_ray: Option<FfiLpUnboundedRay>,
    pub subproblem: Option<FfiLpSubproblemId>,
    pub used_warm_start: bool,
    pub iterations: u64,
    pub pivots: u64,
    pub maximum_basic_reduced_cost: Option<f64>,
}

#[derive(uniffi::Object)]
pub struct FfiLpWorkspace(Mutex<opt::LpWorkspace>);

impl FfiLpWorkspace {
    fn lock(&self) -> Result<MutexGuard<'_, opt::LpWorkspace>, FfiError> {
        self.0.lock().map_err(|_| FfiError::SolverError {
            message: "LP workspace unavailable".into(),
        })
    }
}

#[uniffi::export]
impl FfiLpWorkspace {
    #[uniffi::constructor]
    pub fn new(
        problem: FfiProblem,
        subproblem: Option<FfiLpSubproblemId>,
    ) -> Result<Arc<Self>, FfiError> {
        let problem = to_domain_problem(problem)?;
        let id = subproblem.map(|id| opt::LpSubproblemID {
            name: id.name,
            scenario: id.scenario,
        });
        Ok(Arc::new(Self(Mutex::new(opt::LpWorkspace::new(
            problem, id,
        )?))))
    }

    pub fn install_basis(&self, basis: FfiLpBasis) -> Result<(), FfiError> {
        self.lock()?.install_basis(&opt::LpBasis {
            variables: basis.variables.into_iter().map(Into::into).collect(),
            rows: basis.rows.into_iter().map(Into::into).collect(),
        })?;
        Ok(())
    }

    pub fn update(&self, update: FfiLpUpdate) -> Result<(), FfiError> {
        self.lock()?.update(update.into())?;
        Ok(())
    }

    pub fn solve(
        &self,
        max_iterations: u64,
        tolerance: f64,
        evidence_tolerance: f64,
    ) -> Result<FfiLpSolveReport, FfiError> {
        let solver = opt::RevisedSimplexSolver {
            max_iterations: usize::try_from(max_iterations).map_err(|_| FfiError::SolverError {
                message: "LP iteration limit outside platform range".into(),
            })?,
            tolerance,
        };
        let report = self.lock()?.solve(&solver, evidence_tolerance)?;
        Ok(FfiLpSolveReport {
            solution: from_domain_solution(report.solution),
            basis: FfiLpBasis {
                variables: report.basis.variables.into_iter().map(Into::into).collect(),
                rows: report.basis.rows.into_iter().map(Into::into).collect(),
            },
            dual: report.dual.map(Into::into),
            infeasibility: report.infeasibility.map(Into::into),
            unbounded_ray: report.unbounded_ray.map(Into::into),
            subproblem: report.subproblem.map(|id| FfiLpSubproblemId {
                name: id.name,
                scenario: id.scenario,
            }),
            used_warm_start: report.used_warm_start,
            iterations: report.iterations as u64,
            pivots: report.pivots as u64,
            maximum_basic_reduced_cost: report.maximum_basic_reduced_cost,
        })
    }
}

#[uniffi::export]
pub fn verify_lp_dual_evidence(
    problem: FfiProblem,
    point: Vec<f64>,
    row_multipliers: Vec<f64>,
    tolerance: f64,
) -> Result<Option<FfiLpDualSolution>, FfiError> {
    Ok(opt::verify_lp_dual(
        &to_domain_problem(problem)?,
        &point,
        &row_multipliers,
        tolerance,
    )
    .map(Into::into))
}

#[uniffi::export]
pub fn verify_lp_farkas_evidence(
    problem: FfiProblem,
    row_multipliers: Vec<f64>,
    tolerance: f64,
) -> Result<Option<FfiLpFarkasCertificate>, FfiError> {
    Ok(
        opt::verify_lp_farkas(&to_domain_problem(problem)?, &row_multipliers, tolerance)
            .map(Into::into),
    )
}

#[uniffi::export]
pub fn verify_lp_unbounded_evidence(
    problem: FfiProblem,
    anchor: Vec<f64>,
    direction: Vec<f64>,
    tolerance: f64,
) -> Result<Option<FfiLpUnboundedRay>, FfiError> {
    Ok(
        opt::verify_lp_unbounded_ray(&to_domain_problem(problem)?, &anchor, &direction, tolerance)
            .map(Into::into),
    )
}
