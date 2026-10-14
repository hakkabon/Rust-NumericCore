//! Small-signal AC analysis and response-driven circuit optimization.

use crate::{
    minimize_lbfgsb, Bound, CircuitAcTarget::Magnitude, CircuitDevice, CircuitModel, LbfgsOptions,
    NonlinearTermination, OptimizeError,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircuitComplex {
    pub real: f64,
    pub imaginary: f64,
}

impl CircuitComplex {
    pub fn magnitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }
    pub fn phase(self) -> f64 {
        self.imaginary.atan2(self.real)
    }
    fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            real: magnitude * phase.cos(),
            imaginary: magnitude * phase.sin(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitAcExcitation {
    pub device_name: String,
    pub magnitude: f64,
    pub phase: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitAcPoint {
    pub frequency: f64,
    pub unknowns: Vec<CircuitComplex>,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitAcSweep {
    pub points: Vec<CircuitAcPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitAcTarget {
    Magnitude {
        frequency: f64,
        unknown: usize,
        target: f64,
        weight: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitDesignVariable {
    pub name: String,
    pub initial: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone)]
pub struct CircuitOptimizationOptions {
    pub solver: LbfgsOptions,
    pub relative_difference_step: f64,
}

impl Default for CircuitOptimizationOptions {
    fn default() -> Self {
        Self {
            solver: LbfgsOptions::default(),
            relative_difference_step: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitOptimizationResult {
    pub parameters: Vec<f64>,
    pub objective: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: NonlinearTermination,
    pub sweep: CircuitAcSweep,
}

impl CircuitModel {
    pub fn ac_analysis(
        &self,
        operating_state: &[f64],
        frequencies: &[f64],
        excitations: &[CircuitAcExcitation],
    ) -> Result<CircuitAcSweep, OptimizeError> {
        if operating_state.len() != self.unknown_count()
            || frequencies.is_empty()
            || frequencies.iter().any(|f| !f.is_finite() || *f < 0.)
            || excitations.is_empty()
            || excitations.iter().any(|e| {
                e.device_name.is_empty() || !e.magnitude.is_finite() || !e.phase.is_finite()
            })
        {
            return Err(invalid("invalid AC analysis configuration"));
        }
        let zero = self.zero_derivative();
        let assembly = self.assemble(0., operating_state, &zero)?;
        let conductance = dense(&assembly.jacobian_state);
        let storage = dense(&assembly.jacobian_derivative);
        let n = self.unknown_count();
        let mut rhs = vec![
            CircuitComplex {
                real: 0.,
                imaginary: 0.
            };
            n
        ];
        for excitation in excitations {
            let phasor = CircuitComplex::from_polar(excitation.magnitude, excitation.phase);
            let Some((device_index, device)) = self
                .devices
                .iter()
                .enumerate()
                .find(|(_, device)| device.name() == excitation.device_name)
            else {
                return Err(invalid("AC excitation device was not found"));
            };
            match device {
                CircuitDevice::CurrentSource {
                    positive, negative, ..
                } => {
                    if let Some(index) = self.node_unknown_index(*positive)? {
                        rhs[index].real -= phasor.real;
                        rhs[index].imaginary -= phasor.imaginary;
                    }
                    if let Some(index) = self.node_unknown_index(*negative)? {
                        rhs[index].real += phasor.real;
                        rhs[index].imaginary += phasor.imaginary;
                    }
                }
                CircuitDevice::VoltageSource { .. } => {
                    let index = self
                        .branch_unknown_index(device_index)?
                        .ok_or_else(|| invalid("AC voltage source has no branch equation"))?;
                    rhs[index].real += phasor.real;
                    rhs[index].imaginary += phasor.imaginary;
                }
                _ => return Err(invalid("AC excitation must name an independent source")),
            }
        }
        let mut points = Vec::with_capacity(frequencies.len());
        for &frequency in frequencies {
            let omega = std::f64::consts::TAU * frequency;
            let mut matrix = vec![vec![0.; 2 * n]; 2 * n];
            let mut block_rhs = vec![0.; 2 * n];
            for row in 0..n {
                block_rhs[row] = rhs[row].real;
                block_rhs[row + n] = rhs[row].imaginary;
                for column in 0..n {
                    matrix[row][column] = conductance[row][column];
                    matrix[row][column + n] = -omega * storage[row][column];
                    matrix[row + n][column] = omega * storage[row][column];
                    matrix[row + n][column + n] = conductance[row][column];
                }
            }
            let solution = solve_dense(matrix, block_rhs)?;
            let unknowns = (0..n)
                .map(|index| CircuitComplex {
                    real: solution[index],
                    imaginary: solution[index + n],
                })
                .collect::<Vec<_>>();
            let mut residual_norm = 0_f64;
            for row in 0..n {
                let mut real = -rhs[row].real;
                let mut imaginary = -rhs[row].imaginary;
                for column in 0..n {
                    real += conductance[row][column] * unknowns[column].real
                        - omega * storage[row][column] * unknowns[column].imaginary;
                    imaginary += omega * storage[row][column] * unknowns[column].real
                        + conductance[row][column] * unknowns[column].imaginary;
                }
                residual_norm = residual_norm.max(real.abs()).max(imaginary.abs());
            }
            points.push(CircuitAcPoint {
                frequency,
                unknowns,
                residual_norm,
            });
        }
        Ok(CircuitAcSweep { points })
    }
}

pub fn optimize_circuit_ac<F>(
    variables: &[CircuitDesignVariable],
    excitations: &[CircuitAcExcitation],
    targets: &[CircuitAcTarget],
    options: CircuitOptimizationOptions,
    mut build: F,
) -> Result<CircuitOptimizationResult, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<CircuitModel, OptimizeError>,
{
    validate_optimization(variables, targets, &options)?;
    let initial = variables.iter().map(|v| v.initial).collect::<Vec<_>>();
    let bounds = variables
        .iter()
        .map(|v| Bound {
            lower: Some(v.lower),
            upper: Some(v.upper),
        })
        .collect::<Vec<_>>();
    let frequencies = target_frequencies(targets);
    let step = options.relative_difference_step;
    let result = minimize_lbfgsb(&initial, &bounds, options.solver, |parameters| {
        let value = circuit_objective(parameters, &frequencies, excitations, targets, &mut build)?;
        let mut gradient = vec![0.; parameters.len()];
        for index in 0..parameters.len() {
            let h = step * parameters[index].abs().max(1.);
            let lower = variables[index].lower;
            let upper = variables[index].upper;
            let mut plus = parameters.to_vec();
            let mut minus = parameters.to_vec();
            plus[index] = (parameters[index] + h).min(upper);
            minus[index] = (parameters[index] - h).max(lower);
            if plus[index] == minus[index] {
                continue;
            }
            let high = circuit_objective(&plus, &frequencies, excitations, targets, &mut build)?;
            let low = circuit_objective(&minus, &frequencies, excitations, targets, &mut build)?;
            gradient[index] = (high - low) / (plus[index] - minus[index]);
        }
        Ok((value, gradient))
    })?;
    let model = build(&result.point)?;
    let operating =
        model.dc_operating_point_continuation(None, Default::default(), Default::default())?;
    let sweep = model.ac_analysis(&operating.state, &frequencies, excitations)?;
    Ok(CircuitOptimizationResult {
        parameters: result.point,
        objective: result.objective,
        gradient: result.gradient,
        iterations: result.iterations,
        evaluations: result.evaluations,
        termination: result.termination,
        sweep,
    })
}

fn circuit_objective<F>(
    parameters: &[f64],
    frequencies: &[f64],
    excitations: &[CircuitAcExcitation],
    targets: &[CircuitAcTarget],
    build: &mut F,
) -> Result<f64, OptimizeError>
where
    F: FnMut(&[f64]) -> Result<CircuitModel, OptimizeError>,
{
    let model = build(parameters)?;
    let operating =
        model.dc_operating_point_continuation(None, Default::default(), Default::default())?;
    let sweep = model.ac_analysis(&operating.state, frequencies, excitations)?;
    let mut objective = 0.;
    for target in targets {
        match *target {
            Magnitude {
                frequency,
                unknown,
                target,
                weight,
            } => {
                let point = sweep
                    .points
                    .iter()
                    .find(|p| p.frequency == frequency)
                    .unwrap();
                let response = point
                    .unknowns
                    .get(unknown)
                    .ok_or_else(|| invalid("AC target unknown is out of range"))?
                    .magnitude();
                objective += 0.5 * weight * (response - target).powi(2);
            }
        }
    }
    Ok(objective)
}

fn target_frequencies(targets: &[CircuitAcTarget]) -> Vec<f64> {
    let mut values = targets
        .iter()
        .map(|target| match target {
            Magnitude { frequency, .. } => *frequency,
        })
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values.dedup();
    values
}

fn validate_optimization(
    variables: &[CircuitDesignVariable],
    targets: &[CircuitAcTarget],
    options: &CircuitOptimizationOptions,
) -> Result<(), OptimizeError> {
    let mut names = BTreeSet::new();
    if variables.is_empty()
        || variables.iter().any(|v| {
            v.name.is_empty()
                || !names.insert(v.name.as_str())
                || !v.initial.is_finite()
                || !v.lower.is_finite()
                || !v.upper.is_finite()
                || v.lower > v.initial
                || v.initial > v.upper
        })
        || targets.is_empty()
        || !options.relative_difference_step.is_finite()
        || options.relative_difference_step <= 0.
    {
        return Err(invalid("invalid circuit optimization configuration"));
    }
    if targets.iter().any(|target| match target {
        Magnitude {
            frequency,
            target,
            weight,
            ..
        } => {
            !frequency.is_finite()
                || *frequency < 0.
                || !target.is_finite()
                || *target < 0.
                || !weight.is_finite()
                || *weight <= 0.
        }
    }) {
        return Err(invalid("invalid AC optimization target"));
    }
    Ok(())
}

fn dense(matrix: &nc_sparse::CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.; matrix.cols()]; matrix.rows()];
    for (row, column, value) in matrix.iter_entries() {
        result[row][column] = value;
    }
    result
}

fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, OptimizeError> {
    let n = rhs.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        if matrix[pivot][column].abs() <= 1e-14 {
            return Err(invalid("singular AC system"));
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in column + 1..n {
            let factor = matrix[row][column] / matrix[column][column];
            matrix[row][column] = 0.;
            for j in column + 1..n {
                matrix[row][j] -= factor * matrix[column][j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.; n];
    for row in (0..n).rev() {
        let tail = (row + 1..n)
            .map(|j| matrix[row][j] * solution[j])
            .sum::<f64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(invalid("non-finite AC solution"));
    }
    Ok(solution)
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CircuitNode, CircuitSource};

    fn low_pass(resistance: f64) -> Result<CircuitModel, OptimizeError> {
        CircuitModel::new(
            vec![
                CircuitNode {
                    name: "0".into(),
                    initial_voltage: 0.,
                },
                CircuitNode {
                    name: "in".into(),
                    initial_voltage: 0.,
                },
                CircuitNode {
                    name: "out".into(),
                    initial_voltage: 0.,
                },
            ],
            0,
            vec![
                CircuitDevice::VoltageSource {
                    name: "V1".into(),
                    positive: 1,
                    negative: 0,
                    source: CircuitSource::dc(0.),
                    initial_current: 0.,
                },
                CircuitDevice::Resistor {
                    name: "R1".into(),
                    positive: 1,
                    negative: 2,
                    resistance,
                },
                CircuitDevice::Capacitor {
                    name: "C1".into(),
                    positive: 2,
                    negative: 0,
                    capacitance: 1.,
                },
            ],
        )
    }

    #[test]
    fn ac_low_pass_matches_closed_form_response() {
        let model = low_pass(1.).unwrap();
        let operating = model.dc_operating_point(None, Default::default()).unwrap();
        let sweep = model
            .ac_analysis(
                &operating.state,
                &[1. / std::f64::consts::TAU],
                &[CircuitAcExcitation {
                    device_name: "V1".into(),
                    magnitude: 1.,
                    phase: 0.,
                }],
            )
            .unwrap();
        let output = sweep.points[0].unknowns[1];
        assert!((output.magnitude() - 1. / 2f64.sqrt()).abs() < 1e-10);
        assert!((output.phase() + std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!(sweep.points[0].residual_norm < 1e-12);
    }

    #[test]
    fn response_optimization_recovers_filter_resistance() {
        let result = optimize_circuit_ac(
            &[CircuitDesignVariable {
                name: "R1".into(),
                initial: 2.,
                lower: 0.2,
                upper: 3.,
            }],
            &[CircuitAcExcitation {
                device_name: "V1".into(),
                magnitude: 1.,
                phase: 0.,
            }],
            &[CircuitAcTarget::Magnitude {
                frequency: 1. / std::f64::consts::TAU,
                unknown: 1,
                target: 1. / 2f64.sqrt(),
                weight: 1.,
            }],
            CircuitOptimizationOptions::default(),
            |parameters| low_pass(parameters[0]),
        )
        .unwrap();
        assert!((result.parameters[0] - 1.).abs() < 2e-4);
        assert!(result.objective < 1e-10);
    }
}
