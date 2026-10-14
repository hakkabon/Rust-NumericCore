//! Modified nodal analysis for foundational circuit devices.

use crate::{
    integrate_dae, solve_nonlinear_system, DaeEvaluation, EngineeringAssembly,
    NonlinearSystemEvaluation, NonlinearSystemOptions, NonlinearSystemTermination, OptimizeError,
    TimeIntegrationOptions, TimeIntegrationResult,
};
use nc_sparse::{CooMatrix, CsrMatrix};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitNode {
    pub name: String,
    pub initial_voltage: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitWaveform {
    Constant(f64),
    Sinusoidal {
        offset: f64,
        amplitude: f64,
        frequency: f64,
        phase: f64,
    },
    Pulse {
        low: f64,
        high: f64,
        delay: f64,
        rise: f64,
        fall: f64,
        width: f64,
        period: f64,
    },
}

impl CircuitWaveform {
    pub fn value(&self, time: f64) -> f64 {
        match *self {
            Self::Constant(value) => value,
            Self::Sinusoidal {
                offset,
                amplitude,
                frequency,
                phase,
            } => offset + amplitude * (std::f64::consts::TAU * frequency * time + phase).sin(),
            Self::Pulse {
                low,
                high,
                delay,
                rise,
                fall,
                width,
                period,
            } => {
                if time < delay {
                    return low;
                }
                let local = if period > 0.0 {
                    (time - delay) % period
                } else {
                    time - delay
                };
                if rise > 0.0 && local < rise {
                    low + (high - low) * local / rise
                } else if local < rise + width {
                    high
                } else if fall > 0.0 && local < rise + width + fall {
                    high + (low - high) * (local - rise - width) / fall
                } else {
                    low
                }
            }
        }
    }

    fn valid(&self) -> bool {
        match self {
            Self::Constant(value) => value.is_finite(),
            Self::Sinusoidal {
                offset,
                amplitude,
                frequency,
                phase,
            } => {
                [*offset, *amplitude, *frequency, *phase]
                    .iter()
                    .all(|v| v.is_finite())
                    && *frequency >= 0.0
            }
            Self::Pulse {
                low,
                high,
                delay,
                rise,
                fall,
                width,
                period,
            } => {
                [*low, *high, *delay, *rise, *fall, *width, *period]
                    .iter()
                    .all(|v| v.is_finite())
                    && *delay >= 0.0
                    && *rise >= 0.0
                    && *fall >= 0.0
                    && *width >= 0.0
                    && *period >= 0.0
                    && (*period == 0.0 || *period >= *rise + *width + *fall)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitSource {
    pub dc: f64,
    pub transient: Option<CircuitWaveform>,
}

impl CircuitSource {
    pub fn dc(value: f64) -> Self {
        Self {
            dc: value,
            transient: None,
        }
    }
    pub fn transient(dc: f64, waveform: CircuitWaveform) -> Self {
        Self {
            dc,
            transient: Some(waveform),
        }
    }
    fn value(&self, time: f64, transient: bool) -> f64 {
        if transient {
            self.transient
                .as_ref()
                .map_or(self.dc, |wave| wave.value(time))
        } else {
            self.dc
        }
    }
    fn valid(&self) -> bool {
        self.dc.is_finite() && self.transient.as_ref().map_or(true, CircuitWaveform::valid)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitDevice {
    Resistor {
        name: String,
        positive: usize,
        negative: usize,
        resistance: f64,
    },
    Capacitor {
        name: String,
        positive: usize,
        negative: usize,
        capacitance: f64,
    },
    Inductor {
        name: String,
        positive: usize,
        negative: usize,
        inductance: f64,
        initial_current: f64,
    },
    CurrentSource {
        name: String,
        positive: usize,
        negative: usize,
        source: CircuitSource,
    },
    VoltageSource {
        name: String,
        positive: usize,
        negative: usize,
        source: CircuitSource,
        initial_current: f64,
    },
    Diode {
        name: String,
        anode: usize,
        cathode: usize,
        saturation_current: f64,
        thermal_voltage: f64,
    },
}

impl CircuitDevice {
    fn name(&self) -> &str {
        match self {
            Self::Resistor { name, .. }
            | Self::Capacitor { name, .. }
            | Self::Inductor { name, .. }
            | Self::CurrentSource { name, .. }
            | Self::VoltageSource { name, .. }
            | Self::Diode { name, .. } => name,
        }
    }
    fn nodes(&self) -> (usize, usize) {
        match *self {
            Self::Resistor {
                positive, negative, ..
            }
            | Self::Capacitor {
                positive, negative, ..
            }
            | Self::Inductor {
                positive, negative, ..
            }
            | Self::CurrentSource {
                positive, negative, ..
            }
            | Self::VoltageSource {
                positive, negative, ..
            } => (positive, negative),
            Self::Diode { anode, cathode, .. } => (anode, cathode),
        }
    }
    fn has_branch(&self) -> bool {
        matches!(self, Self::Inductor { .. } | Self::VoltageSource { .. })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitModel {
    pub nodes: Vec<CircuitNode>,
    pub reference_node: usize,
    pub devices: Vec<CircuitDevice>,
    node_unknowns: Vec<Option<usize>>,
    branch_unknowns: Vec<Option<usize>>,
    unknown_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CircuitOperatingPoint {
    pub node_voltages: Vec<f64>,
    pub branch_currents: Vec<(String, f64)>,
    pub state: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
    pub termination: NonlinearSystemTermination,
    pub diagnostics: CircuitDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitDiagnostics {
    pub finite: bool,
    pub maximum_kcl_residual: f64,
    pub maximum_branch_residual: f64,
}

impl CircuitModel {
    pub fn new(
        nodes: Vec<CircuitNode>,
        reference_node: usize,
        devices: Vec<CircuitDevice>,
    ) -> Result<Self, OptimizeError> {
        if nodes.is_empty() || reference_node >= nodes.len() {
            return Err(invalid("invalid reference node"));
        }
        let mut names = BTreeSet::new();
        if nodes.iter().any(|node| {
            node.name.is_empty()
                || !node.initial_voltage.is_finite()
                || !names.insert(node.name.as_str())
        }) {
            return Err(invalid("node names must be unique and values finite"));
        }
        names.clear();
        if devices
            .iter()
            .any(|device| device.name().is_empty() || !names.insert(device.name()))
        {
            return Err(invalid("device names must be non-empty and unique"));
        }
        let mut node_unknowns = vec![None; nodes.len()];
        let mut unknown_names = vec![];
        for node in 0..nodes.len() {
            if node != reference_node {
                node_unknowns[node] = Some(unknown_names.len());
                unknown_names.push(format!("V({})", nodes[node].name));
            }
        }
        let mut branch_unknowns = vec![None; devices.len()];
        for (index, device) in devices.iter().enumerate() {
            let (a, b) = device.nodes();
            if a >= nodes.len() || b >= nodes.len() {
                return Err(invalid("device node is out of range"));
            }
            match device {
                CircuitDevice::Resistor { resistance, .. }
                    if !resistance.is_finite() || *resistance <= 0. =>
                {
                    return Err(invalid("resistance must be positive"))
                }
                CircuitDevice::Capacitor { capacitance, .. }
                    if !capacitance.is_finite() || *capacitance <= 0. =>
                {
                    return Err(invalid("capacitance must be positive"))
                }
                CircuitDevice::Inductor {
                    inductance,
                    initial_current,
                    ..
                } if !inductance.is_finite()
                    || *inductance <= 0.
                    || !initial_current.is_finite() =>
                {
                    return Err(invalid("invalid inductor"))
                }
                CircuitDevice::CurrentSource { source, .. }
                | CircuitDevice::VoltageSource { source, .. }
                    if !source.valid() =>
                {
                    return Err(invalid("invalid source"))
                }
                CircuitDevice::VoltageSource {
                    initial_current, ..
                } if !initial_current.is_finite() => return Err(invalid("invalid source current")),
                CircuitDevice::Diode {
                    saturation_current,
                    thermal_voltage,
                    ..
                } if !saturation_current.is_finite()
                    || *saturation_current <= 0.
                    || !thermal_voltage.is_finite()
                    || *thermal_voltage <= 0. =>
                {
                    return Err(invalid("invalid diode"))
                }
                _ => {}
            }
            if device.has_branch() {
                branch_unknowns[index] = Some(unknown_names.len());
                unknown_names.push(format!("I({})", device.name()));
            }
        }
        if unknown_names.is_empty() {
            return Err(invalid("circuit has no unknowns"));
        }
        Ok(Self {
            nodes,
            reference_node,
            devices,
            node_unknowns,
            branch_unknowns,
            unknown_names,
        })
    }

    pub fn unknown_count(&self) -> usize {
        self.unknown_names.len()
    }
    pub fn unknown_names(&self) -> &[String] {
        &self.unknown_names
    }
    pub fn initial_state(&self) -> Vec<f64> {
        let mut state = vec![0.; self.unknown_count()];
        for (node, index) in self.node_unknowns.iter().enumerate() {
            if let Some(i) = index {
                state[*i] = self.nodes[node].initial_voltage;
            }
        }
        for (device, index) in self.branch_unknowns.iter().enumerate() {
            if let Some(i) = index {
                state[*i] = match self.devices[device] {
                    CircuitDevice::Inductor {
                        initial_current, ..
                    }
                    | CircuitDevice::VoltageSource {
                        initial_current, ..
                    } => initial_current,
                    _ => 0.,
                };
            }
        }
        state
    }
    pub fn zero_derivative(&self) -> Vec<f64> {
        vec![0.; self.unknown_count()]
    }
    pub fn node_voltages(&self, state: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        if state.len() != self.unknown_count() {
            return Err(invalid("invalid circuit state"));
        }
        Ok(self
            .node_unknowns
            .iter()
            .map(|index| index.map_or(0., |i| state[i]))
            .collect())
    }

    pub fn assemble(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
    ) -> Result<EngineeringAssembly, OptimizeError> {
        self.assemble_mode(time, state, derivative, true)
    }
    pub fn dae_evaluation(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
    ) -> Result<DaeEvaluation, OptimizeError> {
        Ok(self.assemble(time, state, derivative)?.to_dae_evaluation())
    }

    pub fn transient(
        &self,
        start: f64,
        end: f64,
        initial_state: &[f64],
        initial_derivative: &[f64],
        options: TimeIntegrationOptions,
    ) -> Result<TimeIntegrationResult, OptimizeError> {
        integrate_dae(
            start,
            end,
            initial_state,
            initial_derivative,
            options,
            |time, state, derivative| self.dae_evaluation(time, state, derivative),
        )
    }

    pub fn dc_operating_point(
        &self,
        initial: Option<&[f64]>,
        options: NonlinearSystemOptions,
    ) -> Result<CircuitOperatingPoint, OptimizeError> {
        let start = initial.map_or_else(|| self.initial_state(), |value| value.to_vec());
        if start.len() != self.unknown_count() {
            return Err(invalid("invalid DC initial state"));
        }
        let zero = self.zero_derivative();
        let solved = solve_nonlinear_system(&start, options, |state| {
            let assembled = self.assemble_mode(0., state, &zero, false)?;
            Ok(NonlinearSystemEvaluation {
                residuals: assembled.residuals,
                jacobian: dense(&assembled.jacobian_state),
            })
        })?;
        let node_voltages = self.node_voltages(&solved.point)?;
        let branch_currents = self
            .branch_unknowns
            .iter()
            .enumerate()
            .filter_map(|(device, index)| {
                index.map(|i| (self.devices[device].name().to_string(), solved.point[i]))
            })
            .collect();
        let final_assembly = self.assemble_mode(0., &solved.point, &zero, false)?;
        let diagnostics = self.diagnostics(&final_assembly.residuals);
        Ok(CircuitOperatingPoint {
            node_voltages,
            branch_currents,
            state: solved.point,
            residual_norm: solved.residual_norm,
            iterations: solved.iterations,
            termination: solved.termination,
            diagnostics,
        })
    }

    fn diagnostics(&self, residuals: &[f64]) -> CircuitDiagnostics {
        let node_equations = self.nodes.len() - 1;
        let maximum = |values: &[f64]| {
            values
                .iter()
                .fold(0_f64, |value, item| value.max(item.abs()))
        };
        CircuitDiagnostics {
            finite: residuals.iter().all(|value| value.is_finite()),
            maximum_kcl_residual: maximum(&residuals[..node_equations]),
            maximum_branch_residual: maximum(&residuals[node_equations..]),
        }
    }

    fn assemble_mode(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        transient: bool,
    ) -> Result<EngineeringAssembly, OptimizeError> {
        let n = self.unknown_count();
        if !time.is_finite()
            || state.len() != n
            || derivative.len() != n
            || state.iter().chain(derivative).any(|v| !v.is_finite())
        {
            return Err(invalid("invalid circuit evaluation point"));
        }
        let mut residual = vec![0.; n];
        let mut jy = vec![vec![0.; n]; n];
        let mut jyd = vec![vec![0.; n]; n];
        for (device_index, device) in self.devices.iter().enumerate() {
            let (positive, negative) = device.nodes();
            let p = self.node_unknowns[positive];
            let m = self.node_unknowns[negative];
            let voltage = value(state, p) - value(state, m);
            let voltage_dot = value(derivative, p) - value(derivative, m);
            match device {
                CircuitDevice::Resistor { resistance, .. } => stamp_two_terminal(
                    p,
                    m,
                    voltage / resistance,
                    1. / resistance,
                    0.,
                    &mut residual,
                    &mut jy,
                    &mut jyd,
                ),
                CircuitDevice::Capacitor { capacitance, .. } => stamp_two_terminal(
                    p,
                    m,
                    if transient {
                        capacitance * voltage_dot
                    } else {
                        0.
                    },
                    0.,
                    if transient { *capacitance } else { 0. },
                    &mut residual,
                    &mut jy,
                    &mut jyd,
                ),
                CircuitDevice::CurrentSource { source, .. } => {
                    stamp_current(p, m, source.value(time, transient), &mut residual)
                }
                CircuitDevice::Diode {
                    saturation_current,
                    thermal_voltage,
                    ..
                } => {
                    let exponential = (voltage / thermal_voltage).exp();
                    if !exponential.is_finite() {
                        return Err(invalid("diode exponential overflow"));
                    }
                    stamp_two_terminal(
                        p,
                        m,
                        saturation_current * (exponential - 1.),
                        saturation_current * exponential / thermal_voltage,
                        0.,
                        &mut residual,
                        &mut jy,
                        &mut jyd,
                    );
                }
                CircuitDevice::VoltageSource { source, .. } => {
                    let branch = self.branch_unknowns[device_index].unwrap();
                    stamp_branch_kcl(p, m, branch, state[branch], &mut residual, &mut jy);
                    residual[branch] += voltage - source.value(time, transient);
                    stamp_voltage(p, m, branch, 1., &mut jy);
                }
                CircuitDevice::Inductor { inductance, .. } => {
                    let branch = self.branch_unknowns[device_index].unwrap();
                    stamp_branch_kcl(p, m, branch, state[branch], &mut residual, &mut jy);
                    residual[branch] += voltage
                        - if transient {
                            inductance * derivative[branch]
                        } else {
                            0.
                        };
                    stamp_voltage(p, m, branch, 1., &mut jy);
                    if transient {
                        jyd[branch][branch] -= inductance;
                    }
                }
            }
        }
        if residual
            .iter()
            .chain(jy.iter().flatten())
            .chain(jyd.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(invalid("circuit assembly produced non-finite values"));
        }
        Ok(EngineeringAssembly {
            residuals: residual,
            jacobian_state: csr(&jy)?,
            jacobian_derivative: csr(&jyd)?,
            parameter_jacobian: CsrMatrix::new(n, 0, vec![0; n + 1], vec![], vec![])?,
        })
    }
}

fn value(values: &[f64], index: Option<usize>) -> f64 {
    index.map_or(0., |i| values[i])
}
fn stamp_current(p: Option<usize>, m: Option<usize>, current: f64, r: &mut [f64]) {
    if let Some(i) = p {
        r[i] += current;
    }
    if let Some(i) = m {
        r[i] -= current;
    }
}
fn stamp_two_terminal(
    p: Option<usize>,
    m: Option<usize>,
    current: f64,
    g: f64,
    c: f64,
    r: &mut [f64],
    jy: &mut [Vec<f64>],
    jyd: &mut [Vec<f64>],
) {
    stamp_current(p, m, current, r);
    for (row, rs) in [(p, 1.), (m, -1.)] {
        if let Some(i) = row {
            if let Some(j) = p {
                jy[i][j] += rs * g;
                jyd[i][j] += rs * c;
            }
            if let Some(j) = m {
                jy[i][j] -= rs * g;
                jyd[i][j] -= rs * c;
            }
        }
    }
}
fn stamp_branch_kcl(
    p: Option<usize>,
    m: Option<usize>,
    branch: usize,
    current: f64,
    r: &mut [f64],
    jy: &mut [Vec<f64>],
) {
    stamp_current(p, m, current, r);
    if let Some(i) = p {
        jy[i][branch] += 1.;
    }
    if let Some(i) = m {
        jy[i][branch] -= 1.;
    }
}
fn stamp_voltage(p: Option<usize>, m: Option<usize>, row: usize, factor: f64, jy: &mut [Vec<f64>]) {
    if let Some(i) = p {
        jy[row][i] += factor;
    }
    if let Some(i) = m {
        jy[row][i] -= factor;
    }
}
fn csr(matrix: &[Vec<f64>]) -> Result<CsrMatrix<f64>, OptimizeError> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    let mut coo = CooMatrix::new(rows, cols);
    for (i, row) in matrix.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0. {
                coo.push(i, j, v)?;
            }
        }
    }
    Ok(coo.to_csr()?)
}
fn dense(matrix: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.; matrix.cols()]; matrix.rows()];
    for (r, c, v) in matrix.iter_entries() {
        result[r][c] = v;
    }
    result
}
fn invalid(message: &str) -> OptimizeError {
    OptimizeError::InvalidProblem(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn nodes() -> Vec<CircuitNode> {
        vec![
            CircuitNode {
                name: "0".into(),
                initial_voltage: 0.,
            },
            CircuitNode {
                name: "out".into(),
                initial_voltage: 0.,
            },
        ]
    }
    #[test]
    fn resistor_current_source_dc_operating_point() {
        let circuit = CircuitModel::new(
            nodes(),
            0,
            vec![
                CircuitDevice::Resistor {
                    name: "R1".into(),
                    positive: 1,
                    negative: 0,
                    resistance: 2.,
                },
                CircuitDevice::CurrentSource {
                    name: "I1".into(),
                    positive: 0,
                    negative: 1,
                    source: CircuitSource::dc(1.),
                },
            ],
        )
        .unwrap();
        let result = circuit
            .dc_operating_point(None, Default::default())
            .unwrap();
        assert_eq!(
            result.termination,
            NonlinearSystemTermination::ConvergedResidual
        );
        assert!((result.node_voltages[1] - 2.).abs() < 1e-12);
        assert!(result.diagnostics.finite);
        assert!(result.diagnostics.maximum_kcl_residual < 1e-12);
        assert_eq!(result.diagnostics.maximum_branch_residual, 0.);
    }
    #[test]
    fn voltage_source_adds_branch_unknown() {
        let circuit = CircuitModel::new(
            nodes(),
            0,
            vec![
                CircuitDevice::VoltageSource {
                    name: "V1".into(),
                    positive: 1,
                    negative: 0,
                    source: CircuitSource::dc(5.),
                    initial_current: 0.,
                },
                CircuitDevice::Resistor {
                    name: "R1".into(),
                    positive: 1,
                    negative: 0,
                    resistance: 10.,
                },
            ],
        )
        .unwrap();
        let result = circuit
            .dc_operating_point(None, Default::default())
            .unwrap();
        assert_eq!(result.node_voltages[1], 5.);
        assert!((result.branch_currents[0].1 + 0.5).abs() < 1e-12);
        assert!(result.diagnostics.maximum_branch_residual < 1e-12);
    }
    #[test]
    fn rc_transient_uses_dae_stamp() {
        let circuit = CircuitModel::new(
            nodes(),
            0,
            vec![
                CircuitDevice::Resistor {
                    name: "R1".into(),
                    positive: 1,
                    negative: 0,
                    resistance: 2.,
                },
                CircuitDevice::Capacitor {
                    name: "C1".into(),
                    positive: 1,
                    negative: 0,
                    capacitance: 0.5,
                },
                CircuitDevice::CurrentSource {
                    name: "I1".into(),
                    positive: 0,
                    negative: 1,
                    source: CircuitSource::dc(1.),
                },
            ],
        )
        .unwrap();
        let options = crate::TimeIntegrationOptions {
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            initial_step: 0.01,
            maximum_step: 0.05,
            ..Default::default()
        };
        let result = circuit.transient(0., 1., &[0.], &[2.], options).unwrap();
        assert!((result.points.last().unwrap().state[0] - 2. * (1. - (-1f64).exp())).abs() < 4e-3);
    }
    #[test]
    fn diode_and_waveforms_are_finite() {
        let sine = CircuitWaveform::Sinusoidal {
            offset: 1.,
            amplitude: 2.,
            frequency: 0.5,
            phase: 0.,
        };
        assert!((sine.value(0.5) - 3.).abs() < 1e-12);
        let circuit = CircuitModel::new(
            nodes(),
            0,
            vec![
                CircuitDevice::Diode {
                    name: "D1".into(),
                    anode: 1,
                    cathode: 0,
                    saturation_current: 1e-12,
                    thermal_voltage: 0.026,
                },
                CircuitDevice::CurrentSource {
                    name: "I1".into(),
                    positive: 0,
                    negative: 1,
                    source: CircuitSource::dc(1e-3),
                },
            ],
        )
        .unwrap();
        let result = circuit
            .dc_operating_point(Some(&[0.5]), Default::default())
            .unwrap();
        assert!(result.node_voltages[1] > 0. && result.node_voltages[1] < 1.);
    }
}
