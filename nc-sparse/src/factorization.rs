//! Reusable sparse factorizations with an explicit symbolic/numeric split.

use crate::{CsrMatrix, SparseDirectError};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseOrdering {
    Natural,
    MinimumDegree,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseNumericOptions {
    pub pivot_tolerance: f64,
    pub drop_tolerance: f64,
    pub refinement_steps: usize,
    pub refinement_tolerance: f64,
}

impl Default for SparseNumericOptions {
    fn default() -> Self {
        Self {
            pivot_tolerance: 1e-12,
            drop_tolerance: 0.0,
            refinement_steps: 3,
            refinement_tolerance: 1e-12,
        }
    }
}

impl SparseNumericOptions {
    fn validate(&self) -> Result<(), SparseDirectError> {
        if !self.pivot_tolerance.is_finite()
            || self.pivot_tolerance < 0.0
            || !self.drop_tolerance.is_finite()
            || self.drop_tolerance < 0.0
            || !self.refinement_tolerance.is_finite()
            || self.refinement_tolerance < 0.0
        {
            return Err(SparseDirectError::InvalidOptions);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseSymbolicAnalysis {
    size: usize,
    ordering: SparseOrdering,
    permutation: Vec<usize>, // ordered index -> original index
    inverse_permutation: Vec<usize>,
    pattern: Vec<(usize, usize)>,
    original_nonzeros: usize,
    estimated_factor_nonzeros: usize,
}

impl SparseSymbolicAnalysis {
    pub fn analyze(
        matrix: &CsrMatrix<f64>,
        ordering: SparseOrdering,
    ) -> Result<Self, SparseDirectError> {
        let rows = matrix_rows(matrix)?;
        let size = rows.len();
        let mut adjacency = vec![BTreeSet::new(); size];
        for (row, columns) in rows.iter().enumerate() {
            for &column in columns.keys() {
                if row != column {
                    adjacency[row].insert(column);
                    adjacency[column].insert(row);
                }
            }
        }
        let (permutation, estimated_factor_nonzeros) = match ordering {
            SparseOrdering::Natural => symbolic_eliminate(adjacency, Some((0..size).collect())),
            SparseOrdering::MinimumDegree => symbolic_eliminate(adjacency, None),
        };
        let mut inverse_permutation = vec![0; size];
        for (ordered, &original) in permutation.iter().enumerate() {
            inverse_permutation[original] = ordered;
        }
        let pattern = matrix.iter_entries().map(|(r, c, _)| (r, c)).collect();
        Ok(Self {
            size,
            ordering,
            permutation,
            inverse_permutation,
            pattern,
            original_nonzeros: matrix.nnz(),
            estimated_factor_nonzeros,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }
    pub fn ordering(&self) -> SparseOrdering {
        self.ordering
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn inverse_permutation(&self) -> &[usize] {
        &self.inverse_permutation
    }
    pub fn original_nonzeros(&self) -> usize {
        self.original_nonzeros
    }
    pub fn estimated_factor_nonzeros(&self) -> usize {
        self.estimated_factor_nonzeros
    }

    pub fn validate_pattern(&self, matrix: &CsrMatrix<f64>) -> Result<(), SparseDirectError> {
        if matrix.rows() != self.size || matrix.cols() != self.size {
            return Err(SparseDirectError::PatternMismatch);
        }
        if !matrix
            .iter_entries()
            .map(|(r, c, _)| (r, c))
            .eq(self.pattern.iter().copied())
        {
            return Err(SparseDirectError::PatternMismatch);
        }
        Ok(())
    }
}

fn symbolic_eliminate(
    mut graph: Vec<BTreeSet<usize>>,
    prescribed: Option<Vec<usize>>,
) -> (Vec<usize>, usize) {
    let n = graph.len();
    // This index contains one entry per active vertex. Updating its key only
    // when neighbors change avoids a full scan for every minimum-degree pivot.
    let mut degrees: BTreeSet<_> = if prescribed.is_none() {
        graph
            .iter()
            .enumerate()
            .map(|(v, edges)| (edges.len(), v))
            .collect()
    } else {
        BTreeSet::new()
    };
    let mut order = Vec::with_capacity(n);
    let mut estimate = n;
    for step in 0..n {
        let vertex = prescribed
            .as_ref()
            .map(|p| p[step])
            .unwrap_or_else(|| degrees.pop_first().unwrap().1);
        let neighbors = std::mem::take(&mut graph[vertex]);
        estimate += 2 * neighbors.len();
        if prescribed.is_none() {
            for &neighbor in &neighbors {
                degrees.remove(&(graph[neighbor].len(), neighbor));
            }
        }
        let neighbors: Vec<_> = neighbors.into_iter().collect();
        for (position, &left) in neighbors.iter().enumerate() {
            for &right in &neighbors[position + 1..] {
                graph[left].insert(right);
                graph[right].insert(left);
            }
            graph[left].remove(&vertex);
            if prescribed.is_none() {
                // All newly inserted neighbors of left are now present.
                degrees.insert((graph[left].len(), left));
            }
        }
        order.push(vertex);
    }
    (order, estimate)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseFactorizationDiagnostics {
    pub ordering: SparseOrdering,
    pub original_nonzeros: usize,
    pub factor_nonzeros: usize,
    pub estimated_factor_nonzeros: usize,
    pub fill_ratio: f64,
    pub minimum_abs_pivot: f64,
    pub pivot_growth: f64,
    pub row_interchanges: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseRefinedReport {
    pub solution: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub backward_error: f64,
    pub refinement_steps: usize,
    pub diagnostics: SparseFactorizationDiagnostics,
}

#[derive(Debug, Clone)]
pub struct SparseLuFactorization {
    analysis: SparseSymbolicAnalysis,
    options: SparseNumericOptions,
    factors: Vec<BTreeMap<usize, f64>>,
    row_permutation: Vec<usize>,
    matrix: CsrMatrix<f64>,
    diagnostics: SparseFactorizationDiagnostics,
}

impl SparseLuFactorization {
    pub fn factor(
        matrix: &CsrMatrix<f64>,
        ordering: SparseOrdering,
        options: SparseNumericOptions,
    ) -> Result<Self, SparseDirectError> {
        let analysis = SparseSymbolicAnalysis::analyze(matrix, ordering)?;
        Self::factor_with_analysis(matrix, analysis, options)
    }

    pub fn factor_with_analysis(
        matrix: &CsrMatrix<f64>,
        analysis: SparseSymbolicAnalysis,
        options: SparseNumericOptions,
    ) -> Result<Self, SparseDirectError> {
        options.validate()?;
        analysis.validate_pattern(matrix)?;
        let (factors, row_permutation, diagnostics) = factor_lu(matrix, &analysis, &options)?;
        Ok(Self {
            analysis,
            options,
            factors,
            row_permutation,
            matrix: matrix.clone(),
            diagnostics,
        })
    }

    pub fn symbolic_analysis(&self) -> &SparseSymbolicAnalysis {
        &self.analysis
    }
    pub fn diagnostics(&self) -> &SparseFactorizationDiagnostics {
        &self.diagnostics
    }

    pub fn refactor(&mut self, matrix: &CsrMatrix<f64>) -> Result<(), SparseDirectError> {
        self.analysis.validate_pattern(matrix)?;
        let (factors, row_permutation, diagnostics) =
            factor_lu(matrix, &self.analysis, &self.options)?;
        self.factors = factors;
        self.row_permutation = row_permutation;
        self.matrix = matrix.clone();
        self.diagnostics = diagnostics;
        Ok(())
    }

    /// Update values in the original CSR storage order.
    pub fn refactor_values(&mut self, values: Vec<f64>) -> Result<(), SparseDirectError> {
        self.refactor(&self.matrix.with_values(values)?)
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.analysis.size)?;
        let mut ordered: Vec<_> = self.analysis.permutation.iter().map(|&i| rhs[i]).collect();
        ordered = self.row_permutation.iter().map(|&i| ordered[i]).collect();
        let n = ordered.len();
        for row in 0..n {
            let contribution: f64 = self.factors[row]
                .range(..row)
                .map(|(&c, &v)| v * ordered[c])
                .sum();
            ordered[row] -= contribution;
        }
        for row in (0..n).rev() {
            let diagonal = self.factors[row].get(&row).copied().unwrap_or(0.0);
            if diagonal.abs() <= self.options.pivot_tolerance {
                return Err(SparseDirectError::Singular { pivot: row });
            }
            let contribution: f64 = self.factors[row]
                .range((row + 1)..)
                .map(|(&c, &v)| v * ordered[c])
                .sum();
            ordered[row] = (ordered[row] - contribution) / diagonal;
        }
        let mut solution = vec![0.0; n];
        for (ordered_index, &original_index) in self.analysis.permutation.iter().enumerate() {
            solution[original_index] = ordered[ordered_index];
        }
        finite_solution(solution)
    }

    pub fn solve_multiple(
        &self,
        right_hand_sides: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, SparseDirectError> {
        right_hand_sides.iter().map(|rhs| self.solve(rhs)).collect()
    }

    /// Solve Aᵀx = b using the stored row pivots and symbolic permutation.
    pub fn solve_transpose(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.analysis.size)?;
        let mut x: Vec<_> = self.analysis.permutation.iter().map(|&i| rhs[i]).collect();
        // Uᵀ followed by Lᵀ. Scatter updates traverse only stored entries.
        for (row, factor) in self.factors.iter().enumerate() {
            x[row] /= factor[&row];
            for (&column, &value) in factor.range((row + 1)..) {
                x[column] -= value * x[row];
            }
        }
        for (row, factor) in self.factors.iter().enumerate().rev() {
            for (&column, &value) in factor.range(..row) {
                x[column] -= value * x[row];
            }
        }
        let mut solution = vec![0.0; x.len()];
        for (row, &pivoted) in self.row_permutation.iter().enumerate() {
            solution[self.analysis.permutation[pivoted]] = x[row];
        }
        finite_solution(solution)
    }

    pub fn solve_transpose_refined(
        &self,
        rhs: &[f64],
    ) -> Result<SparseRefinedReport, SparseDirectError> {
        refine(
            &self.matrix,
            rhs,
            &self.options,
            &self.diagnostics,
            true,
            |r| self.solve_transpose(r),
        )
    }

    pub fn solve_refined(&self, rhs: &[f64]) -> Result<SparseRefinedReport, SparseDirectError> {
        refine(
            &self.matrix,
            rhs,
            &self.options,
            &self.diagnostics,
            false,
            |r| self.solve(r),
        )
    }
}

fn factor_lu(
    matrix: &CsrMatrix<f64>,
    analysis: &SparseSymbolicAnalysis,
    options: &SparseNumericOptions,
) -> Result<
    (
        Vec<BTreeMap<usize, f64>>,
        Vec<usize>,
        SparseFactorizationDiagnostics,
    ),
    SparseDirectError,
> {
    let source = matrix_rows(matrix)?;
    let n = source.len();
    let mut rows = vec![BTreeMap::new(); n];
    let mut maximum_input: f64 = 0.0;
    for original_row in 0..n {
        let row = analysis.inverse_permutation[original_row];
        for (&original_column, &value) in &source[original_row] {
            rows[row].insert(analysis.inverse_permutation[original_column], value);
            maximum_input = maximum_input.max(value.abs());
        }
    }
    let mut row_permutation: Vec<_> = (0..n).collect();
    let mut interchanges = 0;
    let mut minimum_pivot = f64::INFINITY;
    for pivot in 0..n {
        let selected = (pivot..n)
            .max_by(|&a, &b| {
                rows[a]
                    .get(&pivot)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    .total_cmp(&rows[b].get(&pivot).copied().unwrap_or(0.0).abs())
            })
            .unwrap();
        let pivot_value = rows[selected].get(&pivot).copied().unwrap_or(0.0);
        if pivot_value.abs() <= options.pivot_tolerance * maximum_input.max(1.0) {
            return Err(SparseDirectError::Singular { pivot });
        }
        if selected != pivot {
            rows.swap(selected, pivot);
            row_permutation.swap(selected, pivot);
            interchanges += 1;
        }
        let diagonal = rows[pivot][&pivot];
        minimum_pivot = minimum_pivot.min(diagonal.abs());
        let upper: Vec<_> = rows[pivot]
            .range((pivot + 1)..)
            .map(|(&c, &v)| (c, v))
            .collect();
        for row in pivot + 1..n {
            let entry = rows[row].get(&pivot).copied().unwrap_or(0.0);
            if entry == 0.0 {
                continue;
            }
            let multiplier = entry / diagonal;
            rows[row].insert(pivot, multiplier);
            for &(column, upper_value) in &upper {
                let value =
                    rows[row].get(&column).copied().unwrap_or(0.0) - multiplier * upper_value;
                if !value.is_finite() {
                    return Err(SparseDirectError::InvalidNumericInput);
                }
                if value.abs() <= options.drop_tolerance {
                    rows[row].remove(&column);
                } else {
                    rows[row].insert(column, value);
                }
            }
        }
    }
    let factor_nonzeros = rows.iter().map(BTreeMap::len).sum();
    let maximum_factor = rows
        .iter()
        .flat_map(|r| r.values())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let diagnostics = diagnostics(
        analysis,
        factor_nonzeros,
        minimum_pivot,
        maximum_factor / maximum_input.max(f64::MIN_POSITIVE),
        interchanges,
    );
    Ok((rows, row_permutation, diagnostics))
}

/// Reusable sparse LLᵀ factorization for symmetric positive definite systems.
/// Numeric elimination updates only the active neighbors of a pivot.
#[derive(Debug, Clone)]
pub struct SparseCholeskyFactorization {
    analysis: SparseSymbolicAnalysis,
    options: SparseNumericOptions,
    factor: CholeskyColumns,
    matrix: CsrMatrix<f64>,
    diagnostics: SparseFactorizationDiagnostics,
}

#[derive(Debug, Clone)]
struct CholeskyColumns {
    diagonal: Vec<f64>,
    lower: Vec<Vec<(usize, f64)>>,
}

impl SparseCholeskyFactorization {
    pub fn factor(
        matrix: &CsrMatrix<f64>,
        ordering: SparseOrdering,
        options: SparseNumericOptions,
    ) -> Result<Self, SparseDirectError> {
        let analysis = SparseSymbolicAnalysis::analyze(matrix, ordering)?;
        Self::factor_with_analysis(matrix, analysis, options)
    }

    pub fn factor_with_analysis(
        matrix: &CsrMatrix<f64>,
        analysis: SparseSymbolicAnalysis,
        options: SparseNumericOptions,
    ) -> Result<Self, SparseDirectError> {
        options.validate()?;
        analysis.validate_pattern(matrix)?;
        let (factor, diagnostics) = cholesky_columns(matrix, &analysis, &options)?;
        Ok(Self {
            analysis,
            options,
            factor,
            matrix: matrix.clone(),
            diagnostics,
        })
    }

    pub fn symbolic_analysis(&self) -> &SparseSymbolicAnalysis {
        &self.analysis
    }

    pub fn diagnostics(&self) -> &SparseFactorizationDiagnostics {
        &self.diagnostics
    }

    pub fn refactor(&mut self, matrix: &CsrMatrix<f64>) -> Result<(), SparseDirectError> {
        self.analysis.validate_pattern(matrix)?;
        let (factor, diagnostics) = cholesky_columns(matrix, &self.analysis, &self.options)?;
        self.factor = factor;
        self.matrix = matrix.clone();
        self.diagnostics = diagnostics;
        Ok(())
    }

    /// Update values in the original CSR storage order.
    pub fn refactor_values(&mut self, values: Vec<f64>) -> Result<(), SparseDirectError> {
        self.refactor(&self.matrix.with_values(values)?)
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.analysis.size)?;
        let mut x: Vec<_> = self.analysis.permutation.iter().map(|&i| rhs[i]).collect();
        for (j, column) in self.factor.lower.iter().enumerate() {
            x[j] /= self.factor.diagonal[j];
            for &(i, v) in column {
                x[i] -= v * x[j];
            }
        }
        for (j, column) in self.factor.lower.iter().enumerate().rev() {
            let known: f64 = column.iter().map(|&(i, v)| v * x[i]).sum();
            x[j] = (x[j] - known) / self.factor.diagonal[j];
        }
        let mut solution = vec![0.0; x.len()];
        for (j, &original) in self.analysis.permutation.iter().enumerate() {
            solution[original] = x[j];
        }
        finite_solution(solution)
    }

    pub fn solve_multiple(
        &self,
        right_hand_sides: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, SparseDirectError> {
        right_hand_sides.iter().map(|rhs| self.solve(rhs)).collect()
    }

    pub fn solve_refined(&self, rhs: &[f64]) -> Result<SparseRefinedReport, SparseDirectError> {
        refine(
            &self.matrix,
            rhs,
            &self.options,
            &self.diagnostics,
            false,
            |r| self.solve(r),
        )
    }

    pub fn solve_transpose_refined(
        &self,
        rhs: &[f64],
    ) -> Result<SparseRefinedReport, SparseDirectError> {
        refine(
            &self.matrix,
            rhs,
            &self.options,
            &self.diagnostics,
            true,
            |r| self.solve(r),
        )
    }
}

fn cholesky_columns(
    matrix: &CsrMatrix<f64>,
    analysis: &SparseSymbolicAnalysis,
    options: &SparseNumericOptions,
) -> Result<(CholeskyColumns, SparseFactorizationDiagnostics), SparseDirectError> {
    let source = matrix_rows(matrix)?;
    check_symmetric(&source)?;
    let n = source.len();
    let mut rows = vec![BTreeMap::new(); n];
    let mut maximum_input = 0.0_f64;
    for (i, row) in source.iter().enumerate() {
        let a = analysis.inverse_permutation[i];
        for (&j, &v) in row {
            let b = analysis.inverse_permutation[j];
            let mean = v.midpoint(source[j].get(&i).copied().unwrap_or(0.0));
            rows[a.min(b)].insert(a.max(b), mean);
            maximum_input = maximum_input.max(v.abs());
        }
    }
    let threshold = maximum_input * options.pivot_tolerance;
    let mut factor = CholeskyColumns {
        diagonal: Vec::with_capacity(n),
        lower: Vec::with_capacity(n),
    };
    let mut minimum_pivot = f64::INFINITY;
    let mut maximum_active = maximum_input;
    for k in 0..n {
        let mut active = std::mem::take(&mut rows[k]);
        let pivot = active.remove(&k).unwrap_or(0.0);
        if pivot <= threshold {
            return Err(SparseDirectError::NotPositiveDefinite { pivot: k });
        }
        minimum_pivot = minimum_pivot.min(pivot);
        let diagonal = pivot.sqrt();
        let column: Vec<_> = active
            .into_iter()
            .filter(|(_, v)| v.abs() > options.drop_tolerance)
            .map(|(i, v)| (i, v / diagonal))
            .collect();
        for (position, &(i, li)) in column.iter().enumerate() {
            for &(j, lj) in &column[position..] {
                let next = rows[i].get(&j).copied().unwrap_or(0.0) - li * lj;
                if !next.is_finite() {
                    return Err(SparseDirectError::InvalidNumericInput);
                }
                maximum_active = maximum_active.max(next.abs());
                if i != j && next.abs() <= options.drop_tolerance {
                    rows[i].remove(&j);
                } else {
                    rows[i].insert(j, next);
                }
            }
        }
        factor.diagonal.push(diagonal);
        factor.lower.push(column);
    }
    let nnz = n + factor.lower.iter().map(Vec::len).sum::<usize>();
    let diagnostics = diagnostics(
        analysis,
        nnz,
        minimum_pivot,
        maximum_active / maximum_input.max(f64::MIN_POSITIVE),
        0,
    );
    Ok((factor, diagnostics))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseInertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

#[derive(Debug, Clone)]
enum DiagonalBlock {
    One { index: usize, value: f64 },
    Two { index: usize, pivot: SymmetricPivot },
}

/// A symmetric 2×2 pivot normalized before forming its determinant.
#[derive(Debug, Clone, Copy)]
struct SymmetricPivot {
    scale: f64,
    a: f64,
    b: f64,
    c: f64,
    determinant: f64,
}

impl SymmetricPivot {
    fn new(a: f64, b: f64, c: f64) -> Self {
        let scale = a.abs().max(b.abs()).max(c.abs());
        let (a, b, c) = (a / scale, b / scale, c / scale);
        Self {
            scale,
            a,
            b,
            c,
            determinant: a.mul_add(c, -(b * b)),
        }
    }

    fn solve(self, first: f64, second: f64) -> (f64, f64) {
        let x = first / self.scale;
        let y = second / self.scale;
        (
            (self.c * x - self.b * y) / self.determinant,
            (self.a * y - self.b * x) / self.determinant,
        )
    }
}

#[derive(Debug, Clone)]
pub struct SparseLdlt {
    analysis: SparseSymbolicAnalysis,
    options: SparseNumericOptions,
    lower: Vec<BTreeMap<usize, f64>>,
    blocks: Vec<DiagonalBlock>,
    numeric_permutation: Vec<usize>,
    matrix: CsrMatrix<f64>,
    diagnostics: SparseFactorizationDiagnostics,
    inertia: SparseInertia,
}

impl SparseLdlt {
    pub fn factor(
        matrix: &CsrMatrix<f64>,
        ordering: SparseOrdering,
        options: SparseNumericOptions,
    ) -> Result<Self, SparseDirectError> {
        let analysis = SparseSymbolicAnalysis::analyze(matrix, ordering)?;
        Self::factor_with_analysis(matrix, analysis, options)
    }
    pub fn factor_with_analysis(
        matrix: &CsrMatrix<f64>,
        analysis: SparseSymbolicAnalysis,
        options: SparseNumericOptions,
    ) -> Result<Self, SparseDirectError> {
        options.validate()?;
        analysis.validate_pattern(matrix)?;
        let (lower, blocks, numeric_permutation, diagnostics, inertia) =
            factor_ldlt(matrix, &analysis, &options)?;
        Ok(Self {
            analysis,
            options,
            lower,
            blocks,
            numeric_permutation,
            matrix: matrix.clone(),
            diagnostics,
            inertia,
        })
    }
    pub fn symbolic_analysis(&self) -> &SparseSymbolicAnalysis {
        &self.analysis
    }
    pub fn diagnostics(&self) -> &SparseFactorizationDiagnostics {
        &self.diagnostics
    }
    pub fn inertia(&self) -> SparseInertia {
        self.inertia
    }
    pub fn refactor(&mut self, matrix: &CsrMatrix<f64>) -> Result<(), SparseDirectError> {
        self.analysis.validate_pattern(matrix)?;
        let (lower, blocks, numeric_permutation, diagnostics, inertia) =
            factor_ldlt(matrix, &self.analysis, &self.options)?;
        self.lower = lower;
        self.blocks = blocks;
        self.numeric_permutation = numeric_permutation;
        self.matrix = matrix.clone();
        self.diagnostics = diagnostics;
        self.inertia = inertia;
        Ok(())
    }

    /// Update values in the original CSR storage order.
    pub fn refactor_values(&mut self, values: Vec<f64>) -> Result<(), SparseDirectError> {
        self.refactor(&self.matrix.with_values(values)?)
    }
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseDirectError> {
        validate_rhs(rhs, self.analysis.size)?;
        let symbolically_ordered: Vec<_> =
            self.analysis.permutation.iter().map(|&i| rhs[i]).collect();
        let mut x: Vec<_> = self
            .numeric_permutation
            .iter()
            .map(|&i| symbolically_ordered[i])
            .collect();
        let n = x.len();
        for row in 0..n {
            x[row] -= self.lower[row]
                .range(..row)
                .map(|(&c, &v)| v * x[c])
                .sum::<f64>();
        }
        for block in &self.blocks {
            match *block {
                DiagonalBlock::One { index, value } => x[index] /= value,
                DiagonalBlock::Two { index, pivot } => {
                    (x[index], x[index + 1]) = pivot.solve(x[index], x[index + 1]);
                }
            }
        }
        for (row, factor) in self.lower.iter().enumerate().rev() {
            for (&column, &value) in factor.range(..row) {
                x[column] -= value * x[row];
            }
        }
        let mut after_numeric = vec![0.0; n];
        for (i, &prior) in self.numeric_permutation.iter().enumerate() {
            after_numeric[prior] = x[i];
        }
        let mut solution = vec![0.0; n];
        for (i, &original) in self.analysis.permutation.iter().enumerate() {
            solution[original] = after_numeric[i];
        }
        finite_solution(solution)
    }
    pub fn solve_multiple(
        &self,
        right_hand_sides: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, SparseDirectError> {
        right_hand_sides.iter().map(|rhs| self.solve(rhs)).collect()
    }
    pub fn solve_refined(&self, rhs: &[f64]) -> Result<SparseRefinedReport, SparseDirectError> {
        refine(
            &self.matrix,
            rhs,
            &self.options,
            &self.diagnostics,
            false,
            |r| self.solve(r),
        )
    }

    pub fn solve_transpose_refined(
        &self,
        rhs: &[f64],
    ) -> Result<SparseRefinedReport, SparseDirectError> {
        refine(
            &self.matrix,
            rhs,
            &self.options,
            &self.diagnostics,
            true,
            |r| self.solve(r),
        )
    }
}

type LdltParts = (
    Vec<BTreeMap<usize, f64>>,
    Vec<DiagonalBlock>,
    Vec<usize>,
    SparseFactorizationDiagnostics,
    SparseInertia,
);
fn factor_ldlt(
    matrix: &CsrMatrix<f64>,
    analysis: &SparseSymbolicAnalysis,
    options: &SparseNumericOptions,
) -> Result<LdltParts, SparseDirectError> {
    let source = matrix_rows(matrix)?;
    check_symmetric(&source)?;
    let n = source.len();
    let mut active = vec![BTreeMap::new(); n];
    let mut maximum_input = 0.0_f64;
    for original_row in 0..n {
        let row = analysis.inverse_permutation[original_row];
        for (&original_column, &value) in &source[original_row] {
            let column = analysis.inverse_permutation[original_column];
            let transpose = source[original_column]
                .get(&original_row)
                .copied()
                .unwrap_or(0.0);
            let mean = value.midpoint(transpose);
            active[row].insert(column, mean);
            active[column].insert(row, mean);
            maximum_input = maximum_input.max(value.abs());
        }
    }
    let mut lower = vec![BTreeMap::new(); n];
    for i in 0..n {
        lower[i].insert(i, 1.0);
    }
    let mut permutation: Vec<_> = (0..n).collect();
    let mut blocks = Vec::new();
    let mut k = 0;
    let mut swaps = 0;
    let mut minimum_pivot = f64::INFINITY;
    let threshold = options.pivot_tolerance * maximum_input.max(1.0);
    while k < n {
        let mut best_diag = (k, 0.0_f64);
        let mut best_off = (k, k, 0.0_f64);
        for i in k..n {
            let d = active[i].get(&i).copied().unwrap_or(0.0).abs();
            if d > best_diag.1 {
                best_diag = (i, d);
            }
            for (&j, &value) in active[i].range((i + 1)..) {
                let v = value.abs();
                if v > best_off.2 {
                    best_off = (i, j, v);
                }
            }
        }
        if best_diag.1 <= threshold && best_off.2 <= threshold {
            return Err(SparseDirectError::Singular { pivot: k });
        }
        let use_one = k + 1 == n || best_diag.1 >= best_off.2;
        if use_one {
            if best_diag.0 != k {
                symmetric_swap(&mut active, &mut lower, &mut permutation, k, best_diag.0, k);
                swaps += 1;
            }
            let d = active[k].get(&k).copied().unwrap_or(0.0);
            minimum_pivot = minimum_pivot.min(d.abs());
            let mut neighbors = Vec::new();
            for (&i, &v) in active[k].range((k + 1)..) {
                let l = v / d;
                if !l.is_finite() {
                    return Err(SparseDirectError::InvalidNumericInput);
                }
                if l.abs() > options.drop_tolerance {
                    lower[i].insert(k, l);
                    neighbors.push((i, l));
                }
            }
            for (position, &(i, li)) in neighbors.iter().enumerate() {
                for &(j, lj) in &neighbors[position..] {
                    update_symmetric(&mut active, i, j, -li * d * lj, options.drop_tolerance)?;
                }
            }
            blocks.push(DiagonalBlock::One { index: k, value: d });
            k += 1;
        } else {
            let (first, second, _) = best_off;
            if first != k {
                symmetric_swap(&mut active, &mut lower, &mut permutation, k, first, k);
                swaps += 1;
            }
            let second_now = if second == k {
                first
            } else if second == first {
                k
            } else {
                second
            };
            if second_now != k + 1 {
                symmetric_swap(
                    &mut active,
                    &mut lower,
                    &mut permutation,
                    k + 1,
                    second_now,
                    k,
                );
                swaps += 1;
            }
            let a = active[k].get(&k).copied().unwrap_or(0.0);
            let b = active[k].get(&(k + 1)).copied().unwrap_or(0.0);
            let c = active[k + 1].get(&(k + 1)).copied().unwrap_or(0.0);
            let pivot = SymmetricPivot::new(a, b, c);
            if pivot.determinant.abs() <= (threshold / pivot.scale).powi(2) {
                return Err(SparseDirectError::Singular { pivot: k });
            }
            minimum_pivot = minimum_pivot.min(pivot.scale * pivot.determinant.abs().sqrt());
            let neighbors: BTreeSet<_> = active[k]
                .range((k + 2)..)
                .chain(active[k + 1].range((k + 2)..))
                .map(|(&i, _)| i)
                .collect();
            let mut multipliers = Vec::new();
            for i in neighbors {
                let r1 = active[i].get(&k).copied().unwrap_or(0.0);
                let r2 = active[i].get(&(k + 1)).copied().unwrap_or(0.0);
                let (l1, l2) = pivot.solve(r1, r2);
                if !l1.is_finite() || !l2.is_finite() {
                    return Err(SparseDirectError::InvalidNumericInput);
                }
                if l1.abs() > options.drop_tolerance {
                    lower[i].insert(k, l1);
                }
                if l2.abs() > options.drop_tolerance {
                    lower[i].insert(k + 1, l2);
                }
                let l1 = lower[i].get(&k).copied().unwrap_or(0.0);
                let l2 = lower[i].get(&(k + 1)).copied().unwrap_or(0.0);
                if l1 != 0.0 || l2 != 0.0 {
                    multipliers.push((i, l1, l2));
                }
            }
            for (position, &(i, li1, li2)) in multipliers.iter().enumerate() {
                for &(j, lj1, lj2) in &multipliers[position..] {
                    let delta = -(li1 * (a * lj1 + b * lj2) + li2 * (b * lj1 + c * lj2));
                    update_symmetric(&mut active, i, j, delta, options.drop_tolerance)?;
                }
            }
            blocks.push(DiagonalBlock::Two { index: k, pivot });
            k += 2;
        }
    }
    let inertia = block_inertia(&blocks, threshold);
    let factor_nonzeros = lower.iter().map(BTreeMap::len).sum::<usize>()
        + blocks
            .iter()
            .map(|b| {
                if matches!(b, DiagonalBlock::One { .. }) {
                    1
                } else {
                    3
                }
            })
            .sum::<usize>();
    let maximum_lower = lower
        .iter()
        .flat_map(|r| r.values())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let maximum_diagonal = blocks.iter().fold(0.0_f64, |maximum, block| match *block {
        DiagonalBlock::One { value, .. } => maximum.max(value.abs()),
        DiagonalBlock::Two { pivot, .. } => maximum.max(pivot.scale),
    });
    let maximum_factor = maximum_lower.max(maximum_diagonal);
    let diagnostics = diagnostics(
        analysis,
        factor_nonzeros,
        minimum_pivot,
        maximum_factor / maximum_input.max(f64::MIN_POSITIVE),
        swaps,
    );
    Ok((lower, blocks, permutation, diagnostics, inertia))
}

fn symmetric_swap(
    matrix: &mut Vec<BTreeMap<usize, f64>>,
    lower: &mut Vec<BTreeMap<usize, f64>>,
    permutation: &mut Vec<usize>,
    a: usize,
    b: usize,
    completed: usize,
) {
    if a == b {
        return;
    }
    let n = matrix.len();
    let map = |i| {
        if i == a {
            b
        } else if i == b {
            a
        } else {
            i
        }
    };
    let old = std::mem::replace(matrix, vec![BTreeMap::new(); n]);
    for (r, row) in old.into_iter().enumerate() {
        for (c, value) in row {
            matrix[map(r)].insert(map(c), value);
        }
    }
    for column in 0..completed {
        let av = lower[a].remove(&column);
        let bv = lower[b].remove(&column);
        if let Some(v) = bv {
            lower[a].insert(column, v);
        }
        if let Some(v) = av {
            lower[b].insert(column, v);
        }
    }
    permutation.swap(a, b);
}

fn update_symmetric(
    matrix: &mut [BTreeMap<usize, f64>],
    row: usize,
    column: usize,
    delta: f64,
    drop: f64,
) -> Result<(), SparseDirectError> {
    let value = matrix[row].get(&column).copied().unwrap_or(0.0) + delta;
    if !value.is_finite() {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    if value.abs() <= drop {
        matrix[row].remove(&column);
        matrix[column].remove(&row);
    } else {
        matrix[row].insert(column, value);
        matrix[column].insert(row, value);
    }
    Ok(())
}

fn block_inertia(blocks: &[DiagonalBlock], tolerance: f64) -> SparseInertia {
    let mut result = SparseInertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut count = |value: f64, tolerance: f64| {
        if value > tolerance {
            result.positive += 1
        } else if value < -tolerance {
            result.negative += 1
        } else {
            result.zero += 1
        }
    };
    for block in blocks {
        match *block {
            DiagonalBlock::One { value, .. } => count(value, tolerance),
            DiagonalBlock::Two { pivot, .. } => {
                let SymmetricPivot {
                    a,
                    b,
                    c,
                    scale,
                    determinant,
                } = pivot;
                let middle = 0.5 * (a + c);
                let radius = (0.5 * (a - c)).hypot(b);
                let large = middle + radius.copysign(middle);
                count(large, tolerance / scale);
                count(determinant / large, tolerance / scale);
            }
        }
    }
    result
}

fn diagnostics(
    analysis: &SparseSymbolicAnalysis,
    factor_nonzeros: usize,
    minimum_abs_pivot: f64,
    pivot_growth: f64,
    row_interchanges: usize,
) -> SparseFactorizationDiagnostics {
    SparseFactorizationDiagnostics {
        ordering: analysis.ordering,
        original_nonzeros: analysis.original_nonzeros,
        factor_nonzeros,
        estimated_factor_nonzeros: analysis.estimated_factor_nonzeros,
        fill_ratio: factor_nonzeros as f64 / analysis.original_nonzeros.max(1) as f64,
        minimum_abs_pivot,
        pivot_growth,
        row_interchanges,
    }
}

fn matrix_rows(matrix: &CsrMatrix<f64>) -> Result<Vec<BTreeMap<usize, f64>>, SparseDirectError> {
    if matrix.rows() != matrix.cols() {
        return Err(SparseDirectError::NonSquare {
            rows: matrix.rows(),
            cols: matrix.cols(),
        });
    }
    let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); matrix.rows()];
    for (row, column, value) in matrix.iter_entries() {
        if !value.is_finite() {
            return Err(SparseDirectError::InvalidNumericInput);
        }
        *rows[row].entry(column).or_default() += value;
    }
    if rows.iter().any(|r| r.values().any(|v| !v.is_finite())) {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    Ok(rows)
}
fn check_symmetric(rows: &[BTreeMap<usize, f64>]) -> Result<(), SparseDirectError> {
    for (r, row) in rows.iter().enumerate() {
        for (&c, &v) in row {
            let reverse = rows[c].get(&r).copied().unwrap_or(0.0);
            if (v - reverse).abs() > 1e-12 * v.abs().max(reverse.abs()).max(1.0) {
                return Err(SparseDirectError::NotSymmetric { row: r, column: c });
            }
        }
    }
    Ok(())
}
fn validate_rhs(rhs: &[f64], size: usize) -> Result<(), SparseDirectError> {
    if rhs.len() != size {
        return Err(SparseDirectError::RightHandSideLength {
            actual: rhs.len(),
            expected: size,
        });
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    Ok(())
}
fn residual(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    solution: &[f64],
    transpose: bool,
) -> Result<Vec<f64>, SparseDirectError> {
    validate_rhs(rhs, matrix.rows())?;
    let product = if transpose {
        matrix.transpose_spmv(solution)
    } else {
        matrix.spmv(solution)
    }
    .map_err(|_| SparseDirectError::InvalidNumericInput)?;
    finite_solution(rhs.iter().zip(product).map(|(b, ax)| b - ax).collect())
}
fn operator_norm(matrix: &CsrMatrix<f64>, transpose: bool) -> f64 {
    let mut sums = vec![
        0.0;
        if transpose {
            matrix.cols()
        } else {
            matrix.rows()
        }
    ];
    for (r, c, v) in matrix.iter_entries() {
        sums[if transpose { c } else { r }] += v.abs();
    }
    sums.into_iter().fold(0.0, f64::max)
}
fn norm_inf(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}
fn backward_error(a: f64, b: f64, x: &[f64], residual: &[f64]) -> f64 {
    let x = norm_inf(x);
    let r = norm_inf(residual);
    let product = a * x;
    if product.is_infinite() {
        return ((r / a) / x) / (1.0 + (b / a) / x);
    }
    let scale = product.max(b);
    if scale == 0.0 {
        return r;
    }
    (r / scale) / (product / scale + b / scale)
}
fn finite_solution(solution: Vec<f64>) -> Result<Vec<f64>, SparseDirectError> {
    if solution.iter().all(|x| x.is_finite()) {
        Ok(solution)
    } else {
        Err(SparseDirectError::InvalidNumericInput)
    }
}

fn refine(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    options: &SparseNumericOptions,
    diagnostics: &SparseFactorizationDiagnostics,
    transpose: bool,
    solve: impl Fn(&[f64]) -> Result<Vec<f64>, SparseDirectError>,
) -> Result<SparseRefinedReport, SparseDirectError> {
    let mut solution = solve(rhs)?;
    let a_norm = operator_norm(matrix, transpose);
    if !a_norm.is_finite() {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    let b_norm = norm_inf(rhs);
    let mut r = residual(matrix, rhs, &solution, transpose)?;
    let mut error = backward_error(a_norm, b_norm, &solution, &r);
    let mut steps = 0;
    while steps < options.refinement_steps && error > options.refinement_tolerance {
        let correction = solve(&r)?;
        let next: Vec<_> = solution
            .iter()
            .zip(correction)
            .map(|(x, d)| x + d)
            .collect();
        if next.iter().any(|x| !x.is_finite()) {
            break;
        }
        let next_residual = residual(matrix, rhs, &next, transpose)?;
        let next_error = backward_error(a_norm, b_norm, &next, &next_residual);
        if next_error >= error {
            break;
        }
        solution = next;
        r = next_residual;
        error = next_error;
        steps += 1;
    }
    let norm = |values: &[f64]| values.iter().fold(0.0_f64, |n, v| n.hypot(*v));
    let residual_norm = norm(&r);
    let rhs_norm = norm(rhs);
    let relative_residual = if rhs_norm > 0.0 {
        residual_norm / rhs_norm
    } else {
        residual_norm
    };
    if !residual_norm.is_finite()
        || !rhs_norm.is_finite()
        || !relative_residual.is_finite()
        || !error.is_finite()
    {
        return Err(SparseDirectError::InvalidNumericInput);
    }
    Ok(SparseRefinedReport {
        solution,
        residual_norm,
        relative_residual,
        backward_error: error,
        refinement_steps: steps,
        diagnostics: diagnostics.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn csr(n: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut rows = vec![BTreeMap::new(); n];
        for &(r, c, v) in entries {
            rows[r].insert(c, v);
        }
        let mut rp = vec![0];
        let mut ci = vec![];
        let mut vs = vec![];
        for row in rows {
            for (c, v) in row {
                ci.push(c);
                vs.push(v);
            }
            rp.push(vs.len());
        }
        CsrMatrix::new(n, n, rp, ci, vs).unwrap()
    }
    #[test]
    fn lu_reuses_analysis_and_multiple_rhs() {
        let a = csr(
            3,
            &[
                (0, 0, 4.),
                (0, 1, 1.),
                (1, 0, 2.),
                (1, 1, 5.),
                (1, 2, 1.),
                (2, 1, 1.),
                (2, 2, 3.),
            ],
        );
        let mut f =
            SparseLuFactorization::factor(&a, SparseOrdering::MinimumDegree, Default::default())
                .unwrap();
        let solved = f
            .solve_multiple(&[vec![6., 15., 11.], vec![1., 2., 3.]])
            .unwrap();
        assert!(solved[0]
            .iter()
            .zip([1., 2., 3.])
            .all(|(a, b)| (a - b).abs() < 1e-10));
        let b = csr(
            3,
            &[
                (0, 0, 5.),
                (0, 1, 1.),
                (1, 0, 2.),
                (1, 1, 6.),
                (1, 2, 1.),
                (2, 1, 1.),
                (2, 2, 4.),
            ],
        );
        f.refactor(&b).unwrap();
        assert!(f
            .solve(&[7., 17., 14.])
            .unwrap()
            .iter()
            .zip([1., 2., 3.])
            .all(|(a, b)| (a - b).abs() < 1e-10));
    }
    #[test]
    fn refactor_rejects_changed_pattern() {
        let a = csr(2, &[(0, 0, 2.), (1, 1, 3.)]);
        let mut f =
            SparseLuFactorization::factor(&a, SparseOrdering::Natural, Default::default()).unwrap();
        let b = csr(2, &[(0, 0, 2.), (0, 1, 1.), (1, 1, 3.)]);
        assert_eq!(f.refactor(&b), Err(SparseDirectError::PatternMismatch));
    }
    #[test]
    fn ldlt_handles_two_by_two_indefinite_pivot() {
        let a = csr(2, &[(0, 1, 1.), (1, 0, 1.)]);
        let f = SparseLdlt::factor(&a, SparseOrdering::Natural, Default::default()).unwrap();
        assert_eq!(
            f.inertia(),
            SparseInertia {
                positive: 1,
                negative: 1,
                zero: 0
            }
        );
        assert_eq!(f.solve(&[2., 3.]).unwrap(), vec![3., 2.]);
    }
    #[test]
    fn refinement_reports_diagnostics() {
        let a = csr(2, &[(0, 0, 4.), (0, 1, 1.), (1, 0, 1.), (1, 1, 3.)]);
        let f = SparseLdlt::factor(&a, SparseOrdering::MinimumDegree, Default::default()).unwrap();
        let report = f.solve_refined(&[6., 7.]).unwrap();
        assert!(report.residual_norm < 1e-12);
        assert!(report.diagnostics.factor_nonzeros >= 2);
    }
}
