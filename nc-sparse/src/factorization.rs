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
        let pattern: Vec<_> = matrix.iter_entries().map(|(r, c, _)| (r, c)).collect();
        if pattern != self.pattern {
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
    let mut active = vec![true; n];
    let mut order = Vec::with_capacity(n);
    let mut estimate = n;
    for step in 0..n {
        let vertex = prescribed.as_ref().map(|p| p[step]).unwrap_or_else(|| {
            (0..n)
                .filter(|&v| active[v])
                .min_by_key(|&v| (graph[v].iter().filter(|&&u| active[u]).count(), v))
                .unwrap()
        });
        let neighbors: Vec<_> = graph[vertex]
            .iter()
            .copied()
            .filter(|&u| active[u])
            .collect();
        estimate += 2 * neighbors.len();
        for (position, &left) in neighbors.iter().enumerate() {
            for &right in &neighbors[position + 1..] {
                graph[left].insert(right);
                graph[right].insert(left);
            }
            graph[left].remove(&vertex);
        }
        active[vertex] = false;
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
        Ok(solution)
    }

    pub fn solve_multiple(
        &self,
        right_hand_sides: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, SparseDirectError> {
        right_hand_sides.iter().map(|rhs| self.solve(rhs)).collect()
    }

    pub fn solve_refined(&self, rhs: &[f64]) -> Result<SparseRefinedReport, SparseDirectError> {
        let mut solution = self.solve(rhs)?;
        let mut steps = 0;
        for _ in 0..self.options.refinement_steps {
            let residual = residual(&self.matrix, rhs, &solution)?;
            let error = backward_error(&self.matrix, rhs, &solution, &residual);
            if error <= self.options.refinement_tolerance {
                break;
            }
            let correction = self.solve(&residual)?;
            for (x, delta) in solution.iter_mut().zip(correction) {
                *x += delta;
            }
            steps += 1;
        }
        make_report(&self.matrix, rhs, solution, steps, self.diagnostics.clone())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseInertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

#[derive(Debug, Clone)]
enum DiagonalBlock {
    One {
        index: usize,
        value: f64,
    },
    Two {
        index: usize,
        a: f64,
        b: f64,
        c: f64,
    },
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
                DiagonalBlock::Two { index, a, b, c } => {
                    let determinant = a * c - b * b;
                    let first = x[index];
                    let second = x[index + 1];
                    x[index] = (c * first - b * second) / determinant;
                    x[index + 1] = (-b * first + a * second) / determinant;
                }
            }
        }
        for row in (0..n).rev() {
            let contribution: f64 = ((row + 1)..n)
                .filter_map(|r| self.lower[r].get(&row).map(|v| v * x[r]))
                .sum();
            x[row] -= contribution;
        }
        let mut after_numeric = vec![0.0; n];
        for (i, &prior) in self.numeric_permutation.iter().enumerate() {
            after_numeric[prior] = x[i];
        }
        let mut solution = vec![0.0; n];
        for (i, &original) in self.analysis.permutation.iter().enumerate() {
            solution[original] = after_numeric[i];
        }
        Ok(solution)
    }
    pub fn solve_multiple(
        &self,
        right_hand_sides: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, SparseDirectError> {
        right_hand_sides.iter().map(|rhs| self.solve(rhs)).collect()
    }
    pub fn solve_refined(&self, rhs: &[f64]) -> Result<SparseRefinedReport, SparseDirectError> {
        let mut solution = self.solve(rhs)?;
        let mut steps = 0;
        for _ in 0..self.options.refinement_steps {
            let r = residual(&self.matrix, rhs, &solution)?;
            if backward_error(&self.matrix, rhs, &solution, &r) <= self.options.refinement_tolerance
            {
                break;
            }
            let correction = self.solve(&r)?;
            for (x, d) in solution.iter_mut().zip(correction) {
                *x += d;
            }
            steps += 1;
        }
        make_report(&self.matrix, rhs, solution, steps, self.diagnostics.clone())
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
            active[row].insert(analysis.inverse_permutation[original_column], value);
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
            for j in i + 1..n {
                let v = active[i].get(&j).copied().unwrap_or(0.0).abs();
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
            for i in k + 1..n {
                let l = active[i].get(&k).copied().unwrap_or(0.0) / d;
                if l.abs() > options.drop_tolerance {
                    lower[i].insert(k, l);
                }
            }
            for i in k + 1..n {
                let li = lower[i].get(&k).copied().unwrap_or(0.0);
                for j in i..n {
                    let lj = lower[j].get(&k).copied().unwrap_or(0.0);
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
            let determinant = a * c - b * b;
            if determinant.abs() <= threshold * threshold {
                return Err(SparseDirectError::Singular { pivot: k });
            }
            minimum_pivot = minimum_pivot.min(determinant.abs().sqrt());
            for i in k + 2..n {
                let r1 = active[i].get(&k).copied().unwrap_or(0.0);
                let r2 = active[i].get(&(k + 1)).copied().unwrap_or(0.0);
                let l1 = (c * r1 - b * r2) / determinant;
                let l2 = (-b * r1 + a * r2) / determinant;
                if l1.abs() > options.drop_tolerance {
                    lower[i].insert(k, l1);
                }
                if l2.abs() > options.drop_tolerance {
                    lower[i].insert(k + 1, l2);
                }
            }
            for i in k + 2..n {
                let li1 = lower[i].get(&k).copied().unwrap_or(0.0);
                let li2 = lower[i].get(&(k + 1)).copied().unwrap_or(0.0);
                for j in i..n {
                    let lj1 = lower[j].get(&k).copied().unwrap_or(0.0);
                    let lj2 = lower[j].get(&(k + 1)).copied().unwrap_or(0.0);
                    let delta = -(li1 * (a * lj1 + b * lj2) + li2 * (b * lj1 + c * lj2));
                    update_symmetric(&mut active, i, j, delta, options.drop_tolerance)?;
                }
            }
            blocks.push(DiagonalBlock::Two { index: k, a, b, c });
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
        DiagonalBlock::Two { a, b, c, .. } => maximum.max(a.abs()).max(b.abs()).max(c.abs()),
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
    let mut count = |value: f64| {
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
            DiagonalBlock::One { value, .. } => count(value),
            DiagonalBlock::Two { a, b, c, .. } => {
                let middle = 0.5 * (a + c);
                let radius = (0.25 * (a - c).powi(2) + b * b).sqrt();
                count(middle + radius);
                count(middle - radius);
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
) -> Result<Vec<f64>, SparseDirectError> {
    validate_rhs(rhs, matrix.rows())?;
    let product = matrix
        .spmv(solution)
        .map_err(|_| SparseDirectError::InvalidNumericInput)?;
    Ok(rhs.iter().zip(product).map(|(b, ax)| b - ax).collect())
}
fn backward_error(matrix: &CsrMatrix<f64>, rhs: &[f64], solution: &[f64], residual: &[f64]) -> f64 {
    let mut row_sums = vec![0.0; matrix.rows()];
    for (r, _, v) in matrix.iter_entries() {
        row_sums[r] += v.abs();
    }
    let a_norm = row_sums.into_iter().fold(0.0_f64, f64::max);
    let x_norm = solution.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let b_norm = rhs.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let r_norm = residual.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    r_norm / (a_norm * x_norm + b_norm).max(f64::MIN_POSITIVE)
}
fn make_report(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    solution: Vec<f64>,
    refinement_steps: usize,
    diagnostics: SparseFactorizationDiagnostics,
) -> Result<SparseRefinedReport, SparseDirectError> {
    let r = residual(matrix, rhs, &solution)?;
    let residual_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let backward_error = backward_error(matrix, rhs, &solution, &r);
    Ok(SparseRefinedReport {
        solution,
        residual_norm,
        relative_residual: residual_norm / rhs_norm.max(f64::MIN_POSITIVE),
        backward_error,
        refinement_steps,
        diagnostics,
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
