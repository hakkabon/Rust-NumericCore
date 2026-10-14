//! Linear finite-element assembly for thermal triangles and planar trusses.

use crate::OptimizeError;
use nc_sparse::{CooMatrix, CsrMatrix, SparseCholesky};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemNode2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemConstraint {
    pub dof: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemNodalLoad {
    pub dof: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalMaterial {
    pub conductivity: f64,
    pub thickness: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalTriangle3 {
    pub nodes: [usize; 3],
    pub material: usize,
    pub heat_source: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalFemModel {
    pub nodes: Vec<FemNode2D>,
    pub materials: Vec<ThermalMaterial>,
    pub elements: Vec<ThermalTriangle3>,
    pub constraints: Vec<FemConstraint>,
    pub loads: Vec<FemNodalLoad>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrussMaterial {
    pub young_modulus: f64,
    pub area: f64,
    /// Mass density per unit volume. Static analysis permits zero; dynamics requires positive density.
    pub density: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrussElement2D {
    pub nodes: [usize; 2],
    pub material: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrussFemModel {
    pub nodes: Vec<FemNode2D>,
    pub materials: Vec<TrussMaterial>,
    pub elements: Vec<TrussElement2D>,
    pub constraints: Vec<FemConstraint>,
    pub loads: Vec<FemNodalLoad>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrussElementResult {
    pub strain: f64,
    pub stress: f64,
    pub axial_force: f64,
}

#[derive(Debug, Clone)]
pub struct FemStaticResult {
    pub values: Vec<f64>,
    pub reactions: Vec<f64>,
    pub strain_energy: f64,
    pub maximum_free_residual: f64,
    pub reduced_relative_residual: f64,
    pub stiffness: CsrMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct TrussStaticResult {
    pub static_result: FemStaticResult,
    pub elements: Vec<TrussElementResult>,
}

impl ThermalFemModel {
    pub fn solve(&self) -> Result<FemStaticResult, OptimizeError> {
        self.validate()?;
        let n = self.nodes.len();
        let mut coo = CooMatrix::new(n, n);
        let mut rhs = vec![0.; n];
        add_loads(&mut rhs, &self.loads)?;
        for element in &self.elements {
            let coordinates = element.nodes.map(|node| self.nodes[node]);
            let (area, b, c) = triangle_geometry(coordinates)?;
            let material = self.materials[element.material];
            let factor = material.conductivity * material.thickness / (4. * area);
            for local_row in 0..3 {
                rhs[element.nodes[local_row]] +=
                    element.heat_source * material.thickness * area / 3.;
                for local_column in 0..3 {
                    coo.push(
                        element.nodes[local_row],
                        element.nodes[local_column],
                        factor * (b[local_row] * b[local_column] + c[local_row] * c[local_column]),
                    )?;
                }
            }
        }
        solve_constrained(coo.to_csr()?, rhs, &self.constraints)
    }

    fn validate(&self) -> Result<(), OptimizeError> {
        validate_common(self.nodes.len(), &self.constraints, &self.loads)?;
        if self.nodes.len() < 3
            || self.materials.is_empty()
            || self.elements.is_empty()
            || self
                .nodes
                .iter()
                .any(|n| !n.x.is_finite() || !n.y.is_finite())
            || self.materials.iter().any(|m| {
                !m.conductivity.is_finite()
                    || m.conductivity <= 0.
                    || !m.thickness.is_finite()
                    || m.thickness <= 0.
            })
            || self.elements.iter().any(|e| {
                e.material >= self.materials.len()
                    || e.nodes.iter().any(|&n| n >= self.nodes.len())
                    || !e.heat_source.is_finite()
            })
        {
            return Err(invalid("invalid thermal FEM model"));
        }
        let mut used = BTreeSet::new();
        for element in &self.elements {
            let mut key = element.nodes;
            key.sort();
            if element.nodes.iter().collect::<BTreeSet<_>>().len() != 3 || !used.insert(key) {
                return Err(invalid(
                    "thermal elements must be distinct non-repeated triangles",
                ));
            }
            triangle_geometry(element.nodes.map(|node| self.nodes[node]))?;
        }
        Ok(())
    }
}

impl TrussFemModel {
    pub fn solve(&self) -> Result<TrussStaticResult, OptimizeError> {
        self.validate()?;
        let n = 2 * self.nodes.len();
        let (stiffness, _) = self.assemble_structural_matrices(false)?;
        let mut rhs = vec![0.; n];
        add_loads(&mut rhs, &self.loads)?;
        let static_result = solve_constrained(stiffness, rhs, &self.constraints)?;
        let elements = self
            .elements
            .iter()
            .map(|element| {
                let first = self.nodes[element.nodes[0]];
                let second = self.nodes[element.nodes[1]];
                let (length, cosine, sine) = truss_geometry(first, second)?;
                let dofs = [
                    2 * element.nodes[0],
                    2 * element.nodes[0] + 1,
                    2 * element.nodes[1],
                    2 * element.nodes[1] + 1,
                ];
                let direction = [-cosine, -sine, cosine, sine];
                let extension = (0..4)
                    .map(|i| direction[i] * static_result.values[dofs[i]])
                    .sum::<f64>();
                let strain = extension / length;
                let material = self.materials[element.material];
                let stress = material.young_modulus * strain;
                Ok(TrussElementResult {
                    strain,
                    stress,
                    axial_force: material.area * stress,
                })
            })
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        Ok(TrussStaticResult {
            static_result,
            elements,
        })
    }

    fn validate(&self) -> Result<(), OptimizeError> {
        validate_common(2 * self.nodes.len(), &self.constraints, &self.loads)?;
        if self.nodes.len() < 2
            || self.materials.is_empty()
            || self.elements.is_empty()
            || self
                .nodes
                .iter()
                .any(|n| !n.x.is_finite() || !n.y.is_finite())
            || self.materials.iter().any(|m| {
                !m.young_modulus.is_finite()
                    || m.young_modulus <= 0.
                    || !m.area.is_finite()
                    || m.area <= 0.
                    || !m.density.is_finite()
                    || m.density < 0.
            })
            || self.elements.iter().any(|e| {
                e.material >= self.materials.len()
                    || e.nodes.iter().any(|&n| n >= self.nodes.len())
                    || e.nodes[0] == e.nodes[1]
            })
        {
            return Err(invalid("invalid truss FEM model"));
        }
        let mut used = BTreeSet::new();
        for element in &self.elements {
            let key = [
                element.nodes[0].min(element.nodes[1]),
                element.nodes[0].max(element.nodes[1]),
            ];
            if !used.insert(key) {
                return Err(invalid("duplicate truss element"));
            }
            truss_geometry(self.nodes[element.nodes[0]], self.nodes[element.nodes[1]])?;
        }
        Ok(())
    }

    fn assemble_structural_matrices(
        &self,
        require_mass: bool,
    ) -> Result<(CsrMatrix<f64>, CsrMatrix<f64>), OptimizeError> {
        self.validate()?;
        if require_mass && self.materials.iter().any(|m| m.density <= 0.) {
            return Err(invalid(
                "modal and dynamic analysis require positive material density",
            ));
        }
        let n = 2 * self.nodes.len();
        let mut stiffness = CooMatrix::new(n, n);
        let mut mass = CooMatrix::new(n, n);
        for element in &self.elements {
            let (length, cosine, sine) =
                truss_geometry(self.nodes[element.nodes[0]], self.nodes[element.nodes[1]])?;
            let material = self.materials[element.material];
            let direction = [-cosine, -sine, cosine, sine];
            let dofs = [
                2 * element.nodes[0],
                2 * element.nodes[0] + 1,
                2 * element.nodes[1],
                2 * element.nodes[1] + 1,
            ];
            let stiffness_factor = material.young_modulus * material.area / length;
            let mass_factor = material.density * material.area * length / 6.;
            for row in 0..4 {
                for column in 0..4 {
                    stiffness.push(
                        dofs[row],
                        dofs[column],
                        stiffness_factor * direction[row] * direction[column],
                    )?;
                    let node_factor = if row / 2 == column / 2 { 2. } else { 1. };
                    let component = if row % 2 == column % 2 { 1. } else { 0. };
                    mass.push(
                        dofs[row],
                        dofs[column],
                        mass_factor * node_factor * component,
                    )?;
                }
            }
        }
        Ok((stiffness.to_csr()?, mass.to_csr()?))
    }
}

#[derive(Debug, Clone)]
pub struct FemModalResult {
    pub angular_frequencies: Vec<f64>,
    pub frequencies_hz: Vec<f64>,
    /// Full-DOF, mass-normalized mode shapes. Constrained entries are zero.
    pub mode_shapes: Vec<Vec<f64>>,
    pub maximum_relative_residual: f64,
    pub maximum_mass_orthogonality_error: f64,
    pub mass: CsrMatrix<f64>,
    pub stiffness: CsrMatrix<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FemDynamicOptions {
    pub time_step: f64,
    pub steps: usize,
    /// Rayleigh coefficient in `C = alpha M + beta K`.
    pub mass_damping: f64,
    /// Rayleigh coefficient in `C = alpha M + beta K`.
    pub stiffness_damping: f64,
}

impl Default for FemDynamicOptions {
    fn default() -> Self {
        Self {
            time_step: 0.01,
            steps: 1,
            mass_damping: 0.,
            stiffness_damping: 0.,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FemDynamicState {
    pub time: f64,
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
    pub acceleration: Vec<f64>,
    pub kinetic_energy: f64,
    pub strain_energy: f64,
}

#[derive(Debug, Clone)]
pub struct FemDynamicResult {
    pub states: Vec<FemDynamicState>,
    pub maximum_relative_equilibrium_residual: f64,
    pub mass: CsrMatrix<f64>,
    pub stiffness: CsrMatrix<f64>,
}

impl TrussFemModel {
    /// Computes the lowest flexible modes of the constrained generalized problem `K phi = lambda M phi`.
    pub fn modal_analysis(&self, mode_count: usize) -> Result<FemModalResult, OptimizeError> {
        let (stiffness, mass) = self.assemble_structural_matrices(true)?;
        let free = self.dynamic_free_dofs()?;
        if mode_count == 0 || mode_count > free.len() {
            return Err(invalid(
                "mode count must be between one and the number of free DOFs",
            ));
        }
        let k = csr_submatrix_dense(&stiffness, &free);
        let m = csr_submatrix_dense(&mass, &free);
        let l = dense_cholesky(&m)?;
        // A = L^-1 K L^-T converts the generalized symmetric problem to a standard one.
        let mut a = vec![vec![0.; free.len()]; free.len()];
        for column in 0..free.len() {
            let mut unit = vec![0.; free.len()];
            unit[column] = 1.;
            let right = solve_upper_from_lower(&l, &unit);
            let product = dense_mv(&k, &right);
            let transformed = solve_lower(&l, &product);
            for row in 0..free.len() {
                a[row][column] = transformed[row];
            }
        }
        symmetrize(&mut a);
        let (eigenvalues, eigenvectors) = jacobi_eigen(a)?;
        let mut order = (0..eigenvalues.len())
            .filter(|&i| eigenvalues[i] > 1e-12)
            .collect::<Vec<_>>();
        order.sort_by(|&i, &j| eigenvalues[i].total_cmp(&eigenvalues[j]));
        if order.len() < mode_count {
            return Err(invalid(
                "constrained structure contains a rigid-body mechanism",
            ));
        }
        let mut shapes = Vec::with_capacity(mode_count);
        let mut omegas = Vec::with_capacity(mode_count);
        let mut max_residual = 0_f64;
        for &index in order.iter().take(mode_count) {
            let y = (0..free.len())
                .map(|row| eigenvectors[row][index])
                .collect::<Vec<_>>();
            let reduced_shape = solve_upper_from_lower(&l, &y);
            let mut shape = vec![0.; 2 * self.nodes.len()];
            for (local, &dof) in free.iter().enumerate() {
                shape[dof] = reduced_shape[local];
            }
            let kp = stiffness.spmv(&shape)?;
            let mp = mass.spmv(&shape)?;
            let lambda = eigenvalues[index];
            let numerator = free
                .iter()
                .map(|&d| (kp[d] - lambda * mp[d]).powi(2))
                .sum::<f64>()
                .sqrt();
            let denominator = free
                .iter()
                .map(|&d| kp[d].powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1.);
            max_residual = max_residual.max(numerator / denominator);
            omegas.push(lambda.sqrt());
            shapes.push(shape);
        }
        let mut orthogonality = 0_f64;
        for i in 0..shapes.len() {
            for j in 0..shapes.len() {
                let mj = mass.spmv(&shapes[j])?;
                let value = shapes[i].iter().zip(mj).map(|(x, y)| x * y).sum::<f64>();
                orthogonality = orthogonality.max((value - if i == j { 1. } else { 0. }).abs());
            }
        }
        Ok(FemModalResult {
            frequencies_hz: omegas
                .iter()
                .map(|w| w / (2. * std::f64::consts::PI))
                .collect(),
            angular_frequencies: omegas,
            mode_shapes: shapes,
            maximum_relative_residual: max_residual,
            maximum_mass_orthogonality_error: orthogonality,
            mass,
            stiffness,
        })
    }

    /// Integrates `M u'' + C u' + K u = f(t)` with average-acceleration Newmark (`beta=1/4`, `gamma=1/2`).
    /// `load_history` contains the nodal loads at times 0 through `steps`.
    pub fn dynamic_response(
        &self,
        options: FemDynamicOptions,
        initial_displacement: &[f64],
        initial_velocity: &[f64],
        load_history: &[Vec<FemNodalLoad>],
    ) -> Result<FemDynamicResult, OptimizeError> {
        let (stiffness, mass) = self.assemble_structural_matrices(true)?;
        let n = 2 * self.nodes.len();
        if !options.time_step.is_finite()
            || options.time_step <= 0.
            || options.steps == 0
            || !options.mass_damping.is_finite()
            || options.mass_damping < 0.
            || !options.stiffness_damping.is_finite()
            || options.stiffness_damping < 0.
            || initial_displacement.len() != n
            || initial_velocity.len() != n
            || initial_displacement
                .iter()
                .chain(initial_velocity)
                .any(|v| !v.is_finite())
            || load_history.len() != options.steps + 1
        {
            return Err(invalid("invalid FEM dynamic configuration"));
        }
        let free = self.dynamic_free_dofs()?;
        if self.constraints.iter().any(|c| {
            initial_displacement[c.dof].abs() > 1e-14 || initial_velocity[c.dof].abs() > 1e-14
        }) {
            return Err(invalid(
                "constrained initial displacement and velocity must be zero",
            ));
        }
        let k = csr_submatrix_dense(&stiffness, &free);
        let m = csr_submatrix_dense(&mass, &free);
        let c = dense_combine(&m, options.mass_damping, &k, options.stiffness_damping);
        let forces = load_history
            .iter()
            .map(|loads| {
                let mut full = vec![0.; n];
                add_loads(&mut full, loads)?;
                Ok(free.iter().map(|&d| full[d]).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        let mut u = free
            .iter()
            .map(|&d| initial_displacement[d])
            .collect::<Vec<_>>();
        let mut v = free
            .iter()
            .map(|&d| initial_velocity[d])
            .collect::<Vec<_>>();
        let mass_factor = dense_cholesky(&m)?;
        let mut rhs0 = forces[0].clone();
        sub_assign(&mut rhs0, &dense_mv(&c, &v));
        sub_assign(&mut rhs0, &dense_mv(&k, &u));
        let mut acceleration = dense_cholesky_solve(&mass_factor, &rhs0);
        let dt = options.time_step;
        let a0 = 4. / (dt * dt);
        let a1 = 2. / dt;
        let effective = dense_combine(&dense_combine(&k, 1., &m, a0), 1., &c, a1);
        let effective_factor = dense_cholesky(&effective)?;
        let mut states = Vec::with_capacity(options.steps + 1);
        let mut max_residual = 0_f64;
        states.push(dynamic_state(0., &u, &v, &acceleration, &free, n, &m, &k));
        for step in 1..=options.steps {
            let mut effective_rhs = forces[step].clone();
            add_assign(
                &mut effective_rhs,
                &dense_mv(
                    &m,
                    &u.iter()
                        .zip(&v)
                        .zip(&acceleration)
                        .map(|((&x, &y), &z)| a0 * x + 2. * a1 * y + z)
                        .collect::<Vec<_>>(),
                ),
            );
            add_assign(
                &mut effective_rhs,
                &dense_mv(
                    &c,
                    &u.iter()
                        .zip(&v)
                        .zip(&acceleration)
                        .map(|((&x, &y), &z)| a1 * x + y + 0.5 * dt * z)
                        .collect::<Vec<_>>(),
                ),
            );
            let next_u = dense_cholesky_solve(&effective_factor, &effective_rhs);
            let next_a = next_u
                .iter()
                .zip(&u)
                .zip(&v)
                .zip(&acceleration)
                .map(|(((&un, &uo), &vo), &ao)| a0 * (un - uo) - 2. * a1 * vo - ao)
                .collect::<Vec<_>>();
            let next_v = v
                .iter()
                .zip(&acceleration)
                .zip(&next_a)
                .map(|((&vo, &ao), &an)| vo + 0.5 * dt * (ao + an))
                .collect::<Vec<_>>();
            let mut equilibrium = dense_mv(&m, &next_a);
            add_assign(&mut equilibrium, &dense_mv(&c, &next_v));
            add_assign(&mut equilibrium, &dense_mv(&k, &next_u));
            sub_assign(&mut equilibrium, &forces[step]);
            let denom = forces[step]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(1.);
            max_residual =
                max_residual.max(equilibrium.iter().map(|x| x * x).sum::<f64>().sqrt() / denom);
            u = next_u;
            v = next_v;
            acceleration = next_a;
            states.push(dynamic_state(
                step as f64 * dt,
                &u,
                &v,
                &acceleration,
                &free,
                n,
                &m,
                &k,
            ));
        }
        Ok(FemDynamicResult {
            states,
            maximum_relative_equilibrium_residual: max_residual,
            mass,
            stiffness,
        })
    }

    fn dynamic_free_dofs(&self) -> Result<Vec<usize>, OptimizeError> {
        if self.constraints.iter().any(|c| c.value.abs() > 1e-14) {
            return Err(invalid(
                "modal and dynamic analysis require homogeneous constraints",
            ));
        }
        let fixed = self
            .constraints
            .iter()
            .map(|c| c.dof)
            .collect::<BTreeSet<_>>();
        let free = (0..2 * self.nodes.len())
            .filter(|d| !fixed.contains(d))
            .collect::<Vec<_>>();
        if free.is_empty() {
            Err(invalid("FEM model has no free degrees of freedom"))
        } else {
            Ok(free)
        }
    }
}

fn csr_submatrix_dense(matrix: &CsrMatrix<f64>, indices: &[usize]) -> Vec<Vec<f64>> {
    let map = indices
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i))
        .collect::<BTreeMap<_, _>>();
    let mut dense = vec![vec![0.; indices.len()]; indices.len()];
    for (r, c, v) in matrix.iter_entries() {
        if let (Some(&i), Some(&j)) = (map.get(&r), map.get(&c)) {
            dense[i][j] += v;
        }
    }
    dense
}
fn dense_mv(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}
fn dense_combine(a: &[Vec<f64>], alpha: f64, b: &[Vec<f64>], beta: f64) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| alpha * p + beta * q).collect())
        .collect()
}
fn add_assign(a: &mut [f64], b: &[f64]) {
    for (i, x) in b.iter().enumerate() {
        a[i] += x;
    }
}
fn sub_assign(a: &mut [f64], b: &[f64]) {
    for (i, x) in b.iter().enumerate() {
        a[i] -= x;
    }
}
fn dense_cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OptimizeError> {
    let n = a.len();
    let mut l = vec![vec![0.; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !s.is_finite() || s <= 1e-14 {
                    return Err(invalid(
                        "FEM mass or effective matrix is not positive definite",
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Ok(l)
}
fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.; b.len()];
    for i in 0..b.len() {
        x[i] = (b[i] - (0..i).map(|j| l[i][j] * x[j]).sum::<f64>()) / l[i][i];
    }
    x
}
fn solve_upper_from_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.; b.len()];
    for i in (0..b.len()).rev() {
        x[i] = (b[i] - (i + 1..b.len()).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
    }
    x
}
fn dense_cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    solve_upper_from_lower(l, &solve_lower(l, b))
}
fn symmetrize(a: &mut [Vec<f64>]) {
    for i in 0..a.len() {
        for j in 0..i {
            let v = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = v;
            a[j][i] = v;
        }
    }
}
fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> Result<(Vec<f64>, Vec<Vec<f64>>), OptimizeError> {
    let n = a.len();
    let mut v = vec![vec![0.; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.;
    }
    for _ in 0..100 * n * n {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.;
        for i in 0..n {
            for j in i + 1..n {
                if a[i][j].abs() > largest {
                    largest = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest < 1e-12 {
            return Ok(((0..n).map(|i| a[i][i]).collect(), v));
        }
        let theta = 0.5 * (2. * a[p][q]).atan2(a[q][q] - a[p][p]);
        let c = theta.cos();
        let s = theta.sin();
        for k in 0..n {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2. * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2. * s * c * apq + c * c * aqq;
        a[p][q] = 0.;
        a[q][p] = 0.;
        for row in &mut v {
            let vp = row[p];
            let vq = row[q];
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    Err(invalid("modal eigensolver did not converge"))
}
fn dynamic_state(
    time: f64,
    u: &[f64],
    v: &[f64],
    a: &[f64],
    free: &[usize],
    n: usize,
    m: &[Vec<f64>],
    k: &[Vec<f64>],
) -> FemDynamicState {
    let mut displacement = vec![0.; n];
    let mut velocity = vec![0.; n];
    let mut acceleration = vec![0.; n];
    for (i, &d) in free.iter().enumerate() {
        displacement[d] = u[i];
        velocity[d] = v[i];
        acceleration[d] = a[i];
    }
    FemDynamicState {
        time,
        displacement,
        velocity,
        acceleration,
        kinetic_energy: 0.5
            * v.iter()
                .zip(dense_mv(m, v))
                .map(|(x, y)| x * y)
                .sum::<f64>(),
        strain_energy: 0.5
            * u.iter()
                .zip(dense_mv(k, u))
                .map(|(x, y)| x * y)
                .sum::<f64>(),
    }
}

fn solve_constrained(
    stiffness: CsrMatrix<f64>,
    rhs: Vec<f64>,
    constraints: &[FemConstraint],
) -> Result<FemStaticResult, OptimizeError> {
    let n = rhs.len();
    let prescribed = constraints
        .iter()
        .map(|c| (c.dof, c.value))
        .collect::<BTreeMap<_, _>>();
    let free = (0..n)
        .filter(|dof| !prescribed.contains_key(dof))
        .collect::<Vec<_>>();
    if free.is_empty() {
        return Err(invalid("FEM model has no free degrees of freedom"));
    }
    let free_map = free
        .iter()
        .enumerate()
        .map(|(i, &dof)| (dof, i))
        .collect::<BTreeMap<_, _>>();
    let mut reduced = CooMatrix::new(free.len(), free.len());
    let mut reduced_rhs = free.iter().map(|&dof| rhs[dof]).collect::<Vec<_>>();
    for (row, column, value) in stiffness.iter_entries() {
        if let Some(&r) = free_map.get(&row) {
            if let Some(&c) = free_map.get(&column) {
                reduced.push(r, c, value)?;
            } else if let Some(prescribed_value) = prescribed.get(&column) {
                reduced_rhs[r] -= value * prescribed_value;
            }
        }
    }
    let reduced = reduced.to_csr()?;
    let factor = SparseCholesky::factor(&reduced, 0.)
        .map_err(|e| invalid(&format!("FEM factorization failed: {e}")))?;
    let report = factor
        .solve_report(&reduced, &reduced_rhs)
        .map_err(|e| invalid(&format!("FEM solve failed: {e}")))?;
    let mut values = vec![0.; n];
    for (&dof, &value) in &prescribed {
        values[dof] = value;
    }
    for (index, &dof) in free.iter().enumerate() {
        values[dof] = report.solution[index];
    }
    let product = stiffness.spmv(&values)?;
    let residual = product
        .iter()
        .zip(&rhs)
        .map(|(ku, f)| ku - f)
        .collect::<Vec<_>>();
    let maximum_free_residual = free
        .iter()
        .map(|&dof| residual[dof].abs())
        .fold(0_f64, f64::max);
    let reactions = residual
        .iter()
        .enumerate()
        .map(|(dof, value)| {
            if prescribed.contains_key(&dof) {
                *value
            } else {
                0.
            }
        })
        .collect();
    let strain_energy = 0.5
        * values
            .iter()
            .zip(product)
            .map(|(u, ku)| u * ku)
            .sum::<f64>();
    Ok(FemStaticResult {
        values,
        reactions,
        strain_energy,
        maximum_free_residual,
        reduced_relative_residual: report.relative_residual,
        stiffness,
    })
}

fn validate_common(
    dofs: usize,
    constraints: &[FemConstraint],
    loads: &[FemNodalLoad],
) -> Result<(), OptimizeError> {
    let mut fixed = BTreeSet::new();
    if constraints.is_empty()
        || constraints
            .iter()
            .any(|c| c.dof >= dofs || !c.value.is_finite() || !fixed.insert(c.dof))
        || loads
            .iter()
            .any(|load| load.dof >= dofs || !load.value.is_finite())
    {
        return Err(invalid("invalid FEM boundary conditions or loads"));
    }
    Ok(())
}

fn add_loads(rhs: &mut [f64], loads: &[FemNodalLoad]) -> Result<(), OptimizeError> {
    for load in loads {
        rhs[load.dof] += load.value;
    }
    if rhs.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(invalid("non-finite assembled FEM load"))
    }
}

fn triangle_geometry(nodes: [FemNode2D; 3]) -> Result<(f64, [f64; 3], [f64; 3]), OptimizeError> {
    let twice_area = (nodes[1].x - nodes[0].x) * (nodes[2].y - nodes[0].y)
        - (nodes[2].x - nodes[0].x) * (nodes[1].y - nodes[0].y);
    let area = 0.5 * twice_area.abs();
    if !area.is_finite() || area <= 1e-14 {
        return Err(invalid("degenerate thermal triangle"));
    }
    Ok((
        area,
        [
            nodes[1].y - nodes[2].y,
            nodes[2].y - nodes[0].y,
            nodes[0].y - nodes[1].y,
        ],
        [
            nodes[2].x - nodes[1].x,
            nodes[0].x - nodes[2].x,
            nodes[1].x - nodes[0].x,
        ],
    ))
}

fn truss_geometry(first: FemNode2D, second: FemNode2D) -> Result<(f64, f64, f64), OptimizeError> {
    let dx = second.x - first.x;
    let dy = second.y - first.y;
    let length = dx.hypot(dy);
    if !length.is_finite() || length <= 1e-14 {
        return Err(invalid("zero-length truss element"));
    }
    Ok((length, dx / length, dy / length))
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_triangle_recovers_linear_temperature_field() {
        let model = ThermalFemModel {
            nodes: vec![
                FemNode2D { x: 0., y: 0. },
                FemNode2D { x: 1., y: 0. },
                FemNode2D { x: 0., y: 1. },
            ],
            materials: vec![ThermalMaterial {
                conductivity: 2.,
                thickness: 1.,
            }],
            elements: vec![ThermalTriangle3 {
                nodes: [0, 1, 2],
                material: 0,
                heat_source: 0.,
            }],
            constraints: vec![
                FemConstraint { dof: 0, value: 0. },
                FemConstraint { dof: 1, value: 1. },
            ],
            loads: vec![FemNodalLoad { dof: 2, value: 0. }],
        };
        let result = model.solve().unwrap();
        assert!(result.values[2].abs() < 1e-12);
        assert!(result.maximum_free_residual < 1e-12);
        assert!((result.reactions[0] + result.reactions[1]).abs() < 1e-12);
    }

    #[test]
    fn axial_bar_returns_displacement_stress_and_reaction() {
        let model = TrussFemModel {
            nodes: vec![FemNode2D { x: 0., y: 0. }, FemNode2D { x: 2., y: 0. }],
            materials: vec![TrussMaterial {
                young_modulus: 200.,
                area: 3.,
                density: 0.,
            }],
            elements: vec![TrussElement2D {
                nodes: [0, 1],
                material: 0,
            }],
            constraints: vec![
                FemConstraint { dof: 0, value: 0. },
                FemConstraint { dof: 1, value: 0. },
                FemConstraint { dof: 3, value: 0. },
            ],
            loads: vec![FemNodalLoad { dof: 2, value: 30. }],
        };
        let result = model.solve().unwrap();
        assert!((result.static_result.values[2] - 0.1).abs() < 1e-12);
        assert!((result.elements[0].stress - 10.).abs() < 1e-12);
        assert!((result.elements[0].axial_force - 30.).abs() < 1e-12);
        assert!((result.static_result.reactions[0] + 30.).abs() < 1e-12);
    }

    fn dynamic_bar() -> TrussFemModel {
        TrussFemModel {
            nodes: vec![FemNode2D { x: 0., y: 0. }, FemNode2D { x: 2., y: 0. }],
            materials: vec![TrussMaterial {
                young_modulus: 200.,
                area: 3.,
                density: 5.,
            }],
            elements: vec![TrussElement2D {
                nodes: [0, 1],
                material: 0,
            }],
            constraints: vec![
                FemConstraint { dof: 0, value: 0. },
                FemConstraint { dof: 1, value: 0. },
                FemConstraint { dof: 3, value: 0. },
            ],
            loads: vec![],
        }
    }

    #[test]
    fn axial_bar_modal_frequency_is_mass_normalized() {
        let result = dynamic_bar().modal_analysis(1).unwrap();
        let expected = (3_f64 * 200. / (5. * 4.)).sqrt();
        assert!((result.angular_frequencies[0] - expected).abs() < 1e-10);
        assert!(result.maximum_relative_residual < 1e-12);
        assert!(result.maximum_mass_orthogonality_error < 1e-12);
        assert_eq!(result.mode_shapes[0][0], 0.);
    }

    #[test]
    fn newmark_free_vibration_conserves_energy() {
        let model = dynamic_bar();
        let options = FemDynamicOptions {
            time_step: 0.01,
            steps: 100,
            mass_damping: 0.,
            stiffness_damping: 0.,
        };
        let loads = vec![vec![]; options.steps + 1];
        let result = model
            .dynamic_response(options, &[0., 0., 0.1, 0.], &[0.; 4], &loads)
            .unwrap();
        let energies = result
            .states
            .iter()
            .map(|s| s.kinetic_energy + s.strain_energy)
            .collect::<Vec<_>>();
        assert!((energies.last().unwrap() - energies[0]).abs() < 1e-10);
        assert!(result.maximum_relative_equilibrium_residual < 1e-10);
    }
}
