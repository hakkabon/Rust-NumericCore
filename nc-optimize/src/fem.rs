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
        let mut coo = CooMatrix::new(n, n);
        let mut rhs = vec![0.; n];
        add_loads(&mut rhs, &self.loads)?;
        for element in &self.elements {
            let first = self.nodes[element.nodes[0]];
            let second = self.nodes[element.nodes[1]];
            let (length, cosine, sine) = truss_geometry(first, second)?;
            let material = self.materials[element.material];
            let direction = [-cosine, -sine, cosine, sine];
            let dofs = [
                2 * element.nodes[0],
                2 * element.nodes[0] + 1,
                2 * element.nodes[1],
                2 * element.nodes[1] + 1,
            ];
            let factor = material.young_modulus * material.area / length;
            for row in 0..4 {
                for column in 0..4 {
                    coo.push(
                        dofs[row],
                        dofs[column],
                        factor * direction[row] * direction[column],
                    )?;
                }
            }
        }
        let static_result = solve_constrained(coo.to_csr()?, rhs, &self.constraints)?;
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
}
