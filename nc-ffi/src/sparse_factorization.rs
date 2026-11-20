use std::sync::{Arc, RwLock};

use nc_sparse::{
    SparseCholeskyFactorization, SparseDirectError, SparseFactorizationDiagnostics, SparseInertia,
    SparseLdlt, SparseLuFactorization, SparseNumericOptions, SparseOrdering, SparseRefinedReport,
    SparseSymbolicAnalysis,
};

use crate::{to_csr_f64, FfiCsrMatrixF64, FfiError};

#[derive(Clone, Copy, uniffi::Enum)]
pub enum FfiSparseOrdering {
    Natural,
    MinimumDegree,
}

#[derive(uniffi::Enum)]
pub enum FfiSparseFactorMethod {
    Lu,
    Cholesky,
    Ldlt,
}

#[derive(uniffi::Record)]
pub struct FfiSparseNumericOptions {
    pub pivot_tolerance: f64,
    pub drop_tolerance: f64,
    pub refinement_steps: u64,
    pub refinement_tolerance: f64,
}

#[derive(uniffi::Record)]
pub struct FfiSparsePatternInfo {
    pub size: u64,
    pub ordering: FfiSparseOrdering,
    pub permutation: Vec<u64>,
    pub inverse_permutation: Vec<u64>,
    pub original_nonzeros: u64,
    pub estimated_factor_nonzeros: u64,
}

#[derive(uniffi::Object)]
pub struct FfiSparseFactorizationPlan {
    analysis: SparseSymbolicAnalysis,
}

#[uniffi::export]
pub fn analyze_sparse_factorization(
    matrix: FfiCsrMatrixF64,
    ordering: FfiSparseOrdering,
) -> Result<Arc<FfiSparseFactorizationPlan>, FfiError> {
    let matrix = to_csr_f64(matrix)?;
    let order = match ordering {
        FfiSparseOrdering::Natural => SparseOrdering::Natural,
        FfiSparseOrdering::MinimumDegree => SparseOrdering::MinimumDegree,
    };
    Ok(Arc::new(FfiSparseFactorizationPlan {
        analysis: SparseSymbolicAnalysis::analyze(&matrix, order)?,
    }))
}

#[uniffi::export]
impl FfiSparseFactorizationPlan {
    pub fn info(&self) -> FfiSparsePatternInfo {
        let a = &self.analysis;
        FfiSparsePatternInfo {
            size: a.size() as u64,
            ordering: ffi_ordering(a.ordering()),
            permutation: a.permutation().iter().map(|&i| i as u64).collect(),
            inverse_permutation: a.inverse_permutation().iter().map(|&i| i as u64).collect(),
            original_nonzeros: a.original_nonzeros() as u64,
            estimated_factor_nonzeros: a.estimated_factor_nonzeros() as u64,
        }
    }

    pub fn factor(
        &self,
        matrix: FfiCsrMatrixF64,
        method: FfiSparseFactorMethod,
        options: FfiSparseNumericOptions,
    ) -> Result<Arc<FfiSparseFactorization>, FfiError> {
        let matrix = to_csr_f64(matrix)?;
        let analysis = self.analysis.clone();
        let options = SparseNumericOptions {
            pivot_tolerance: options.pivot_tolerance,
            drop_tolerance: options.drop_tolerance,
            refinement_steps: usize::try_from(options.refinement_steps).map_err(|_| {
                FfiError::SolverError {
                    message: "refinement steps exceed platform range".into(),
                }
            })?,
            refinement_tolerance: options.refinement_tolerance,
        };
        let factor = match method {
            FfiSparseFactorMethod::Lu => Factor::Lu(SparseLuFactorization::factor_with_analysis(
                &matrix, analysis, options,
            )?),
            FfiSparseFactorMethod::Cholesky => Factor::Cholesky(
                SparseCholeskyFactorization::factor_with_analysis(&matrix, analysis, options)?,
            ),
            FfiSparseFactorMethod::Ldlt => Factor::Ldlt(SparseLdlt::factor_with_analysis(
                &matrix, analysis, options,
            )?),
        };
        Ok(Arc::new(FfiSparseFactorization {
            factor: RwLock::new(factor),
        }))
    }
}

#[derive(uniffi::Record)]
pub struct FfiSparseFactorDiagnostics {
    pub ordering: FfiSparseOrdering,
    pub original_nonzeros: u64,
    pub factor_nonzeros: u64,
    pub estimated_factor_nonzeros: u64,
    pub fill_ratio: f64,
    pub minimum_abs_pivot: f64,
    pub pivot_growth: f64,
    pub interchanges: u64,
}

#[derive(uniffi::Record)]
pub struct FfiSparseInertia {
    pub positive: u64,
    pub negative: u64,
    pub zero: u64,
}

#[derive(uniffi::Record)]
pub struct FfiSparseFactorStatus {
    pub diagnostics: FfiSparseFactorDiagnostics,
    pub inertia: Option<FfiSparseInertia>,
}

#[derive(uniffi::Record)]
pub struct FfiSparseRefinedSolution {
    pub solution: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub backward_error: f64,
    pub refinement_steps: u64,
    pub diagnostics: FfiSparseFactorDiagnostics,
}

enum Factor {
    Lu(SparseLuFactorization),
    Cholesky(SparseCholeskyFactorization),
    Ldlt(SparseLdlt),
}

#[derive(uniffi::Object)]
pub struct FfiSparseFactorization {
    factor: RwLock<Factor>,
}

#[uniffi::export]
impl FfiSparseFactorization {
    pub fn refactor(&self, matrix: FfiCsrMatrixF64) -> Result<(), FfiError> {
        let matrix = to_csr_f64(matrix)?;
        let mut factor = self.factor.write().map_err(|_| lock_error())?;
        match &mut *factor {
            Factor::Lu(f) => f.refactor(&matrix)?,
            Factor::Cholesky(f) => f.refactor(&matrix)?,
            Factor::Ldlt(f) => f.refactor(&matrix)?,
        }
        Ok(())
    }

    pub fn refactor_values(&self, values: Vec<f64>) -> Result<(), FfiError> {
        let mut factor = self.factor.write().map_err(|_| lock_error())?;
        match &mut *factor {
            Factor::Lu(f) => f.refactor_values(values)?,
            Factor::Cholesky(f) => f.refactor_values(values)?,
            Factor::Ldlt(f) => f.refactor_values(values)?,
        }
        Ok(())
    }

    pub fn status(&self) -> Result<FfiSparseFactorStatus, FfiError> {
        let factor = self.factor.read().map_err(|_| lock_error())?;
        let (d, inertia) = match &*factor {
            Factor::Lu(f) => (f.diagnostics(), None),
            Factor::Cholesky(f) => (f.diagnostics(), None),
            Factor::Ldlt(f) => (f.diagnostics(), Some(f.inertia())),
        };
        Ok(FfiSparseFactorStatus {
            diagnostics: diagnostics(d),
            inertia: inertia.map(
                |SparseInertia {
                     positive,
                     negative,
                     zero,
                 }| FfiSparseInertia {
                    positive: positive as u64,
                    negative: negative as u64,
                    zero: zero as u64,
                },
            ),
        })
    }

    pub fn solve_batch(
        &self,
        rhs: Vec<Vec<f64>>,
        transpose: bool,
    ) -> Result<Vec<Vec<f64>>, FfiError> {
        let factor = self.factor.read().map_err(|_| lock_error())?;
        Ok(match &*factor {
            Factor::Lu(f) if transpose => rhs
                .iter()
                .map(|r| f.solve_transpose(r))
                .collect::<Result<_, _>>()?,
            Factor::Lu(f) => f.solve_multiple(&rhs)?,
            Factor::Cholesky(f) => f.solve_multiple(&rhs)?,
            Factor::Ldlt(f) => f.solve_multiple(&rhs)?,
        })
    }

    pub fn solve_batch_refined(
        &self,
        rhs: Vec<Vec<f64>>,
        transpose: bool,
    ) -> Result<Vec<FfiSparseRefinedSolution>, FfiError> {
        let factor = self.factor.read().map_err(|_| lock_error())?;
        let solve = |r: &[f64]| -> Result<SparseRefinedReport, SparseDirectError> {
            match (&*factor, transpose) {
                (Factor::Lu(f), false) => f.solve_refined(r),
                (Factor::Lu(f), true) => f.solve_transpose_refined(r),
                (Factor::Cholesky(f), false) => f.solve_refined(r),
                (Factor::Cholesky(f), true) => f.solve_transpose_refined(r),
                (Factor::Ldlt(f), false) => f.solve_refined(r),
                (Factor::Ldlt(f), true) => f.solve_transpose_refined(r),
            }
        };
        rhs.iter()
            .map(|r| {
                let report = solve(r)?;
                Ok(FfiSparseRefinedSolution {
                    solution: report.solution,
                    residual_norm: report.residual_norm,
                    relative_residual: report.relative_residual,
                    backward_error: report.backward_error,
                    refinement_steps: report.refinement_steps as u64,
                    diagnostics: diagnostics(&report.diagnostics),
                })
            })
            .collect()
    }
}

fn lock_error() -> FfiError {
    FfiError::SolverError {
        message: "sparse factorization lock is poisoned".into(),
    }
}

fn ffi_ordering(value: SparseOrdering) -> FfiSparseOrdering {
    match value {
        SparseOrdering::Natural => FfiSparseOrdering::Natural,
        SparseOrdering::MinimumDegree => FfiSparseOrdering::MinimumDegree,
    }
}

fn diagnostics(d: &SparseFactorizationDiagnostics) -> FfiSparseFactorDiagnostics {
    FfiSparseFactorDiagnostics {
        ordering: ffi_ordering(d.ordering),
        original_nonzeros: d.original_nonzeros as u64,
        factor_nonzeros: d.factor_nonzeros as u64,
        estimated_factor_nonzeros: d.estimated_factor_nonzeros as u64,
        fill_ratio: d.fill_ratio,
        minimum_abs_pivot: d.minimum_abs_pivot,
        pivot_growth: d.pivot_growth,
        interchanges: d.row_interchanges as u64,
    }
}
