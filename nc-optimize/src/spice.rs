//! A deterministic SPICE front end for the native circuit-analysis kernel.

use crate::{
    CircuitAcExcitation, CircuitDevice, CircuitModel, CircuitNode, CircuitSource, CircuitWaveform,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct SpiceDiagnostic {
    pub line: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpiceNetlistError {
    pub diagnostics: Vec<SpiceDiagnostic>,
}

impl fmt::Display for SpiceNetlistError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(first) = self.diagnostics.first() else {
            return write!(formatter, "invalid SPICE netlist");
        };
        write!(formatter, "SPICE line {}: {}", first.line, first.message)
    }
}
impl Error for SpiceNetlistError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiceAcScale {
    Linear,
    Decade,
    Octave,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpiceAcAnalysis {
    pub scale: SpiceAcScale,
    pub points: usize,
    pub start: f64,
    pub stop: f64,
}
impl SpiceAcAnalysis {
    pub fn frequencies(&self) -> Vec<f64> {
        if self.points == 0 {
            return vec![];
        }
        if self.start == self.stop {
            return vec![self.start];
        }
        match self.scale {
            SpiceAcScale::Linear => {
                if self.points == 1 {
                    return vec![self.start];
                }
                (0..self.points)
                    .map(|i| {
                        self.start + (self.stop - self.start) * i as f64 / (self.points - 1) as f64
                    })
                    .collect()
            }
            SpiceAcScale::Decade | SpiceAcScale::Octave => {
                let base: f64 = if self.scale == SpiceAcScale::Decade {
                    10.0
                } else {
                    2.0
                };
                let intervals = ((self.stop / self.start).ln() / base.ln() * self.points as f64)
                    .ceil() as usize;
                (0..=intervals)
                    .map(|i| (self.start * base.powf(i as f64 / self.points as f64)).min(self.stop))
                    .collect()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpiceAnalysis {
    OperatingPoint,
    Transient { step: f64, stop: f64, start: f64 },
    Ac(SpiceAcAnalysis),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSpiceNetlist {
    pub title: Option<String>,
    pub circuit: CircuitModel,
    pub analyses: Vec<SpiceAnalysis>,
    pub ac_excitations: Vec<CircuitAcExcitation>,
    pub warnings: Vec<SpiceDiagnostic>,
}

#[derive(Debug, Clone)]
enum Model {
    Diode {
        saturation: f64,
        ideality: f64,
    },
    Nmos {
        beta: f64,
        threshold: f64,
        lambda: f64,
    },
}

#[derive(Debug, Clone)]
struct SourceSpec {
    source: CircuitSource,
    ac: Option<(f64, f64)>,
}

pub fn parse_spice_netlist(source: &str) -> Result<ParsedSpiceNetlist, SpiceNetlistError> {
    let lines = logical_lines(source);
    if lines.is_empty() {
        return Err(one_error(0, "netlist is empty"));
    }
    let mut parameters = BTreeMap::new();
    let mut models = BTreeMap::new();
    for (line, text) in &lines {
        let tokens = tokens(text);
        if tokens.is_empty() {
            continue;
        }
        if tokens[0].eq_ignore_ascii_case(".param") {
            parse_assignments(&tokens[1..], *line, &parameters)?
                .into_iter()
                .for_each(|(key, value)| {
                    parameters.insert(key, value);
                });
        }
    }
    for (line, text) in &lines {
        let values = tokens(text);
        if values
            .first()
            .is_some_and(|v| v.eq_ignore_ascii_case(".model"))
        {
            let (name, model) = parse_model(&values, *line, &parameters)?;
            if models.insert(name, model).is_some() {
                return Err(one_error(*line, "duplicate model name"));
            }
        }
    }

    let mut nodes = vec![CircuitNode {
        name: "0".into(),
        initial_voltage: 0.0,
    }];
    let mut node_map = BTreeMap::from([("0".into(), 0usize), ("gnd".into(), 0usize)]);
    let mut devices = vec![];
    let mut device_names = BTreeSet::new();
    let mut analyses = vec![];
    let mut excitations = vec![];
    let mut initial_nodes = BTreeMap::new();
    let mut warnings = vec![];
    let mut title = None;
    let mut saw_content = false;

    for (line, text) in &lines {
        let values = tokens(text);
        if values.is_empty() {
            continue;
        }
        let head = values[0].to_ascii_lowercase();
        // Classical SPICE reserves the first non-comment line as the title.
        if !saw_content && !head.starts_with('.') {
            title = Some(text.trim().to_string());
            saw_content = true;
            continue;
        }
        saw_content = true;
        if head.starts_with('.') {
            match head.as_str() {
                ".param" | ".model" | ".end" => {}
                ".op" => analyses.push(SpiceAnalysis::OperatingPoint),
                ".tran" => analyses.push(parse_tran(&values, *line, &parameters)?),
                ".ac" => analyses.push(SpiceAnalysis::Ac(parse_ac(&values, *line, &parameters)?)),
                ".ic" => {
                    parse_initial_conditions(&values[1..], *line, &parameters, &mut initial_nodes)?
                }
                ".include" | ".lib" | ".subckt" | ".ends" => {
                    return Err(one_error(
                        *line,
                        "unsupported structural directive cannot be ignored safely",
                    ))
                }
                _ => warnings.push(SpiceDiagnostic {
                    line: *line,
                    message: format!("ignored unsupported directive {}", values[0]),
                }),
            }
            continue;
        }
        if values[0].is_empty() {
            continue;
        }
        let name_key = values[0].to_ascii_lowercase();
        if !device_names.insert(name_key) {
            return Err(one_error(*line, "duplicate device name"));
        }
        let node = |name: &str,
                    nodes: &mut Vec<CircuitNode>,
                    map: &mut BTreeMap<String, usize>|
         -> usize {
            let key = name.to_ascii_lowercase();
            if key == "0" || key == "gnd" {
                return 0;
            }
            if let Some(index) = map.get(&key) {
                return *index;
            }
            let index = nodes.len();
            map.insert(key, index);
            nodes.push(CircuitNode {
                name: name.into(),
                initial_voltage: 0.0,
            });
            index
        };
        let need = |count: usize| {
            if values.len() < count {
                Err(one_error(*line, "too few element fields"))
            } else {
                Ok(())
            }
        };
        let designator = values[0].chars().next().unwrap().to_ascii_uppercase();
        match designator {
            'R' | 'C' | 'L' => {
                need(4)?;
                let p = node(&values[1], &mut nodes, &mut node_map);
                let n = node(&values[2], &mut nodes, &mut node_map);
                let value = parse_value(&values[3], *line, &parameters)?;
                let initial = option_value(&values[4..], "ic", *line, &parameters)?.unwrap_or(0.0);
                let device = match designator {
                    'R' => CircuitDevice::Resistor {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        resistance: value,
                    },
                    'C' => CircuitDevice::Capacitor {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        capacitance: value,
                    },
                    _ => CircuitDevice::Inductor {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        inductance: value,
                        initial_current: initial,
                    },
                };
                devices.push(device);
            }
            'V' | 'I' => {
                need(4)?;
                let p = node(&values[1], &mut nodes, &mut node_map);
                let n = node(&values[2], &mut nodes, &mut node_map);
                let specification = parse_source(&values[3..], *line, &parameters)?;
                if let Some((magnitude, phase)) = specification.ac {
                    excitations.push(CircuitAcExcitation {
                        device_name: values[0].clone(),
                        magnitude,
                        phase,
                    });
                }
                devices.push(if designator == 'V' {
                    CircuitDevice::VoltageSource {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        source: specification.source,
                        initial_current: 0.0,
                    }
                } else {
                    CircuitDevice::CurrentSource {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        source: specification.source,
                    }
                });
            }
            'D' => {
                need(4)?;
                let a = node(&values[1], &mut nodes, &mut node_map);
                let c = node(&values[2], &mut nodes, &mut node_map);
                let Some(Model::Diode {
                    saturation,
                    ideality,
                }) = models.get(&values[3].to_ascii_lowercase())
                else {
                    return Err(one_error(*line, "unknown or non-diode model"));
                };
                devices.push(CircuitDevice::Diode {
                    name: values[0].clone(),
                    anode: a,
                    cathode: c,
                    saturation_current: *saturation,
                    thermal_voltage: 0.025_85 * *ideality,
                });
            }
            'G' | 'E' => {
                need(6)?;
                let p = node(&values[1], &mut nodes, &mut node_map);
                let n = node(&values[2], &mut nodes, &mut node_map);
                let cp = node(&values[3], &mut nodes, &mut node_map);
                let cn = node(&values[4], &mut nodes, &mut node_map);
                let gain = parse_value(&values[5], *line, &parameters)?;
                devices.push(if designator == 'G' {
                    CircuitDevice::VoltageControlledCurrentSource {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        control_positive: cp,
                        control_negative: cn,
                        transconductance: gain,
                    }
                } else {
                    CircuitDevice::VoltageControlledVoltageSource {
                        name: values[0].clone(),
                        positive: p,
                        negative: n,
                        control_positive: cp,
                        control_negative: cn,
                        gain,
                        initial_current: 0.0,
                    }
                });
            }
            'M' => {
                need(6)?;
                let d = node(&values[1], &mut nodes, &mut node_map);
                let g = node(&values[2], &mut nodes, &mut node_map);
                let s = node(&values[3], &mut nodes, &mut node_map);
                let b = node(&values[4], &mut nodes, &mut node_map);
                if b != s {
                    return Err(one_error(*line, "level-1 MOS body must be tied to source"));
                }
                let Some(Model::Nmos {
                    beta,
                    threshold,
                    lambda,
                }) = models.get(&values[5].to_ascii_lowercase())
                else {
                    return Err(one_error(*line, "unknown or non-NMOS model"));
                };
                devices.push(CircuitDevice::NmosLevel1 {
                    name: values[0].clone(),
                    drain: d,
                    gate: g,
                    source: s,
                    transconductance: *beta,
                    threshold: *threshold,
                    channel_length_modulation: *lambda,
                });
            }
            _ => return Err(one_error(*line, "unsupported element designator")),
        }
    }
    for (name, voltage) in initial_nodes {
        let Some(&index) = node_map.get(&name) else {
            return Err(one_error(0, &format!(".IC references unknown node {name}")));
        };
        nodes[index].initial_voltage = voltage;
    }
    let circuit =
        CircuitModel::new(nodes, 0, devices).map_err(|error| one_error(0, &error.to_string()))?;
    Ok(ParsedSpiceNetlist {
        title,
        circuit,
        analyses,
        ac_excitations: excitations,
        warnings,
    })
}

fn logical_lines(source: &str) -> Vec<(usize, String)> {
    let mut result: Vec<(usize, String)> = vec![];
    for (offset, raw) in source.lines().enumerate() {
        let line = offset + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') {
            continue;
        }
        let text = trimmed.split('$').next().unwrap_or("").trim();
        if let Some(rest) = text.strip_prefix('+') {
            if let Some((_, prior)) = result.last_mut() {
                prior.push(' ');
                prior.push_str(rest.trim());
            }
        } else if !text.is_empty() {
            result.push((line, text.into()));
            if text.eq_ignore_ascii_case(".end") {
                break;
            }
        }
    }
    result
}

fn tokens(text: &str) -> Vec<String> {
    let mut result = vec![];
    let mut current = String::new();
    for character in text.chars() {
        if character.is_whitespace() || matches!(character, '(' | ')' | ',' | '=') {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

fn parse_value(
    token: &str,
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<f64, SpiceNetlistError> {
    let stripped = token.trim_matches(|c| c == '{' || c == '}');
    if let Some(value) = parameters.get(&stripped.to_ascii_lowercase()) {
        return Ok(*value);
    }
    let bytes = stripped.as_bytes();
    let mut end = 0;
    let mut exponent = false;
    for (index, &byte) in bytes.iter().enumerate() {
        let character = byte as char;
        let accepted = character.is_ascii_digit()
            || character == '.'
            || ((character == '+' || character == '-') && (index == 0 || exponent))
            || ((character == 'e' || character == 'E') && !exponent);
        if !accepted {
            break;
        }
        end = index + 1;
        exponent = character == 'e' || character == 'E';
        if character.is_ascii_digit() || character == '.' {
            exponent = false;
        }
    }
    let base = stripped[..end]
        .parse::<f64>()
        .map_err(|_| one_error(line, &format!("invalid numeric value {token}")))?;
    let suffix = stripped[end..].to_ascii_lowercase();
    let multiplier = if suffix.starts_with("meg") {
        1e6
    } else {
        match suffix.chars().next() {
            None => 1.0,
            Some('t') => 1e12,
            Some('g') => 1e9,
            Some('k') => 1e3,
            Some('m') => 1e-3,
            Some('u') => 1e-6,
            Some('n') => 1e-9,
            Some('p') => 1e-12,
            Some('f') => 1e-15,
            _ => {
                return Err(one_error(
                    line,
                    &format!("unknown numeric suffix in {token}"),
                ))
            }
        }
    };
    let value = base * multiplier;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(one_error(line, "numeric value is not finite"))
    }
}

fn parse_assignments(
    tokens: &[String],
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<Vec<(String, f64)>, SpiceNetlistError> {
    if tokens.len() % 2 != 0 {
        return Err(one_error(line, "expected name=value assignments"));
    }
    tokens
        .chunks(2)
        .map(|pair| {
            Ok((
                pair[0].to_ascii_lowercase(),
                parse_value(&pair[1], line, parameters)?,
            ))
        })
        .collect()
}
fn option_value(
    tokens: &[String],
    name: &str,
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<Option<f64>, SpiceNetlistError> {
    for pair in tokens.windows(2) {
        if pair[0].eq_ignore_ascii_case(name) {
            return parse_value(&pair[1], line, parameters).map(Some);
        }
    }
    Ok(None)
}
fn parse_model(
    tokens: &[String],
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<(String, Model), SpiceNetlistError> {
    if tokens.len() < 3 {
        return Err(one_error(line, "incomplete .MODEL directive"));
    }
    let options = parse_assignments(&tokens[3..], line, parameters)?;
    let map: BTreeMap<_, _> = options.into_iter().collect();
    let model = match tokens[2].to_ascii_lowercase().as_str() {
        "d" | "diode" => Model::Diode {
            saturation: *map.get("is").unwrap_or(&1e-14),
            ideality: *map.get("n").unwrap_or(&1.0),
        },
        "nmos" => Model::Nmos {
            beta: *map.get("kp").unwrap_or(&1e-3),
            threshold: *map.get("vto").unwrap_or(&0.7),
            lambda: *map.get("lambda").unwrap_or(&0.0),
        },
        _ => {
            return Err(one_error(
                line,
                "only diode and NMOS level-1 models are supported",
            ))
        }
    };
    Ok((tokens[1].to_ascii_lowercase(), model))
}
fn parse_source(
    tokens: &[String],
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<SourceSpec, SpiceNetlistError> {
    let mut dc = 0.0;
    let mut transient = None;
    let mut ac = None;
    let mut index = 0;
    if !tokens.is_empty()
        && !matches!(
            tokens[0].to_ascii_lowercase().as_str(),
            "dc" | "ac" | "sin" | "pulse"
        )
    {
        dc = parse_value(&tokens[0], line, parameters)?;
        index = 1;
    }
    while index < tokens.len() {
        match tokens[index].to_ascii_lowercase().as_str() {
            "dc" => {
                index += 1;
                dc = parse_value(
                    tokens
                        .get(index)
                        .ok_or_else(|| one_error(line, "DC source value is missing"))?,
                    line,
                    parameters,
                )?;
                index += 1;
            }
            "ac" => {
                index += 1;
                let magnitude = parse_value(
                    tokens
                        .get(index)
                        .ok_or_else(|| one_error(line, "AC magnitude is missing"))?,
                    line,
                    parameters,
                )?;
                index += 1;
                let phase = if index < tokens.len()
                    && !matches!(
                        tokens[index].to_ascii_lowercase().as_str(),
                        "dc" | "sin" | "pulse"
                    ) {
                    let value = parse_value(&tokens[index], line, parameters)?.to_radians();
                    index += 1;
                    value
                } else {
                    0.0
                };
                ac = Some((magnitude, phase));
            }
            "sin" => {
                if index + 4 > tokens.len() {
                    return Err(one_error(
                        line,
                        "SIN requires offset, amplitude, and frequency",
                    ));
                }
                let offset = parse_value(&tokens[index + 1], line, parameters)?;
                let amplitude = parse_value(&tokens[index + 2], line, parameters)?;
                let frequency = parse_value(&tokens[index + 3], line, parameters)?;
                let phase = if index + 4 < tokens.len() {
                    parse_value(&tokens[index + 4], line, parameters)?.to_radians()
                } else {
                    0.0
                };
                transient = Some(CircuitWaveform::Sinusoidal {
                    offset,
                    amplitude,
                    frequency,
                    phase,
                });
                index = tokens.len();
            }
            "pulse" => {
                if index + 8 > tokens.len() {
                    return Err(one_error(line, "PULSE requires seven values"));
                }
                let value = |at: usize| parse_value(&tokens[index + at], line, parameters);
                transient = Some(CircuitWaveform::Pulse {
                    low: value(1)?,
                    high: value(2)?,
                    delay: value(3)?,
                    rise: value(4)?,
                    fall: value(5)?,
                    width: value(6)?,
                    period: value(7)?,
                });
                index += 8;
            }
            _ => return Err(one_error(line, "unsupported source specification")),
        }
    }
    Ok(SourceSpec {
        source: CircuitSource { dc, transient },
        ac,
    })
}
fn parse_tran(
    tokens: &[String],
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<SpiceAnalysis, SpiceNetlistError> {
    if tokens.len() < 3 {
        return Err(one_error(line, ".TRAN requires step and stop"));
    }
    let step = parse_value(&tokens[1], line, parameters)?;
    let stop = parse_value(&tokens[2], line, parameters)?;
    let start = if tokens.len() > 3 {
        parse_value(&tokens[3], line, parameters)?
    } else {
        0.0
    };
    if step <= 0.0 || stop <= start {
        return Err(one_error(line, "invalid .TRAN interval"));
    }
    Ok(SpiceAnalysis::Transient { step, stop, start })
}
fn parse_ac(
    tokens: &[String],
    line: usize,
    parameters: &BTreeMap<String, f64>,
) -> Result<SpiceAcAnalysis, SpiceNetlistError> {
    if tokens.len() < 5 {
        return Err(one_error(
            line,
            ".AC requires scale, points, start, and stop",
        ));
    }
    let scale = match tokens[1].to_ascii_lowercase().as_str() {
        "lin" => SpiceAcScale::Linear,
        "dec" => SpiceAcScale::Decade,
        "oct" => SpiceAcScale::Octave,
        _ => return Err(one_error(line, "unknown .AC scale")),
    };
    let points_value = parse_value(&tokens[2], line, parameters)?;
    let points = points_value as usize;
    let start = parse_value(&tokens[3], line, parameters)?;
    let stop = parse_value(&tokens[4], line, parameters)?;
    if points == 0 || points_value != points as f64 || start <= 0.0 || stop < start {
        return Err(one_error(line, "invalid .AC sweep"));
    }
    Ok(SpiceAcAnalysis {
        scale,
        points,
        start,
        stop,
    })
}
fn parse_initial_conditions(
    tokens: &[String],
    line: usize,
    parameters: &BTreeMap<String, f64>,
    result: &mut BTreeMap<String, f64>,
) -> Result<(), SpiceNetlistError> {
    let mut index = 0;
    while index < tokens.len() {
        if !tokens[index].eq_ignore_ascii_case("v") || index + 2 >= tokens.len() {
            return Err(one_error(line, "expected .IC V(node)=value"));
        }
        result.insert(
            tokens[index + 1].to_ascii_lowercase(),
            parse_value(&tokens[index + 2], line, parameters)?,
        );
        index += 3;
    }
    Ok(())
}
fn one_error(line: usize, message: &str) -> SpiceNetlistError {
    SpiceNetlistError {
        diagnostics: vec![SpiceDiagnostic {
            line,
            message: message.into(),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_solves_dc_and_ac_low_pass() {
        let parsed = parse_spice_netlist("RC low pass\nV1 in 0 DC 1 AC 1\nR1 in out 1k\nC1 out 0 1u\n.OP\n.AC DEC 10 10 10k\n.END").unwrap();
        assert_eq!(parsed.title.as_deref(), Some("RC low pass"));
        assert_eq!(parsed.circuit.devices.len(), 3);
        let operating = parsed
            .circuit
            .dc_operating_point(None, Default::default())
            .unwrap();
        let ac = parsed
            .analyses
            .iter()
            .find_map(|analysis| {
                if let SpiceAnalysis::Ac(value) = analysis {
                    Some(value)
                } else {
                    None
                }
            })
            .unwrap();
        let sweep = parsed
            .circuit
            .ac_analysis(&operating.state, &ac.frequencies(), &parsed.ac_excitations)
            .unwrap();
        assert_eq!(sweep.points.len(), 31);
        assert!(sweep.points.iter().all(|point| point.residual_norm < 1e-9));
    }

    #[test]
    fn parses_models_parameters_initial_conditions_and_transient_sources() {
        let parsed = parse_spice_netlist("nonlinear\n.PARAM RVAL=2k\n.MODEL DMOD D(IS=1n N=2)\nV1 in 0 PULSE(0 1 1u 1n 1n 2u 5u)\nR1 in out {RVAL}\nD1 out 0 DMOD\n.IC V(out)=0.2\n.TRAN 10n 10u\n.END").unwrap();
        assert_eq!(parsed.circuit.nodes[2].initial_voltage, 0.2);
        assert!(matches!(
            parsed.analyses[0],
            SpiceAnalysis::Transient { .. }
        ));
        assert_eq!(
            parsed
                .circuit
                .transient_breakpoints(0.0, 10e-6)
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn reports_source_line_for_unsupported_semantics() {
        let error = parse_spice_netlist("bad\nQ1 c b e QMOD\n.END").unwrap_err();
        assert_eq!(error.diagnostics[0].line, 2);
        assert!(error.to_string().contains("unsupported"));
    }
}
