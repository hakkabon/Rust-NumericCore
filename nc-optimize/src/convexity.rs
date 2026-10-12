//! Conservative convexity analysis and graph-based global relaxations.
//!
//! The analysis deliberately returns `Unknown` whenever a composition rule is
//! not provably valid on the supplied box.  Relaxations use interval bounds,
//! exact affine graph rows, McCormick envelopes for products, and
//! tangent/secant envelopes for supported univariate atoms.

use crate::{
    Bound, NonlinearConstraint, NonlinearExpression, NonlinearModel, NonlinearNode, OptimizeError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpressionConvexity {
    Constant,
    Affine,
    Convex,
    Concave,
    Unknown,
}

impl ExpressionConvexity {
    pub fn is_convex(self) -> bool {
        matches!(self, Self::Constant | Self::Affine | Self::Convex)
    }
    pub fn is_concave(self) -> bool {
        matches!(self, Self::Constant | Self::Affine | Self::Concave)
    }
    fn negated(self) -> Self {
        match self {
            Self::Convex => Self::Concave,
            Self::Concave => Self::Convex,
            other => other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub fn new(lower: f64, upper: f64) -> Option<Self> {
        (!lower.is_nan() && !upper.is_nan() && lower <= upper).then_some(Self { lower, upper })
    }
    pub fn whole() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }
    pub fn is_finite(self) -> bool {
        self.lower.is_finite() && self.upper.is_finite()
    }
    pub fn is_point(self) -> bool {
        self.lower == self.upper
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeAnalysis {
    pub interval: Interval,
    pub convexity: ExpressionConvexity,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionAnalysis {
    pub nodes: Vec<NodeAnalysis>,
    pub output: usize,
}

impl ExpressionAnalysis {
    pub fn output_analysis(&self) -> &NodeAnalysis {
        &self.nodes[self.output]
    }
}

/// An affine row over graph-node values: `lower <= coefficients * nodes <= upper`.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineRelaxationRow {
    pub coefficients: Vec<f64>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlobalExpressionRelaxation {
    pub analysis: ExpressionAnalysis,
    pub rows: Vec<AffineRelaxationRow>,
    /// True when every graph atom is represented exactly or by a recognized
    /// global envelope.  False still denotes a valid, but weaker, relaxation.
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearConvexityCertificate {
    pub objective: ExpressionAnalysis,
    pub constraints: Vec<ExpressionAnalysis>,
    pub globally_convex: bool,
    pub finite_parameter_box: bool,
    pub relaxation_complete: bool,
}

pub fn analyze_expression(
    expression: &NonlinearExpression,
    bounds: &[Bound],
) -> Result<ExpressionAnalysis, OptimizeError> {
    expression.validate(bounds.len())?;
    let mut nodes: Vec<NodeAnalysis> = Vec::with_capacity(expression.nodes.len());
    for node in &expression.nodes {
        let item = match *node {
            NonlinearNode::Constant(value) => NodeAnalysis {
                interval: Interval {
                    lower: value,
                    upper: value,
                },
                convexity: ExpressionConvexity::Constant,
            },
            NonlinearNode::Parameter(index) => NodeAnalysis {
                interval: Interval {
                    lower: bounds[index].lower.unwrap_or(f64::NEG_INFINITY),
                    upper: bounds[index].upper.unwrap_or(f64::INFINITY),
                },
                convexity: ExpressionConvexity::Affine,
            },
            NonlinearNode::Add(a, b) => combine_add(nodes[a].clone(), nodes[b].clone(), false),
            NonlinearNode::Subtract(a, b) => combine_add(nodes[a].clone(), nodes[b].clone(), true),
            NonlinearNode::Negate(a) => NodeAnalysis {
                interval: neg(nodes[a].interval),
                convexity: nodes[a].convexity.negated(),
            },
            NonlinearNode::Multiply(a, b) => multiply_analysis(&nodes[a], &nodes[b], a == b),
            NonlinearNode::Divide(a, b) => divide_analysis(&nodes[a], &nodes[b]),
            NonlinearNode::Exp(a) => unary_analysis(&nodes[a], Unary::Exp),
            NonlinearNode::Log(a) => unary_analysis(&nodes[a], Unary::Log),
            NonlinearNode::Sqrt(a) => unary_analysis(&nodes[a], Unary::Sqrt),
            NonlinearNode::Sin(a) => unary_analysis(&nodes[a], Unary::Sin),
            NonlinearNode::Cos(a) => unary_analysis(&nodes[a], Unary::Cos),
            NonlinearNode::Powf(a, exponent) => pow_analysis(&nodes[a], exponent),
        };
        nodes.push(item);
    }
    Ok(ExpressionAnalysis {
        nodes,
        output: expression.output,
    })
}

pub fn relax_expression(
    expression: &NonlinearExpression,
    bounds: &[Bound],
) -> Result<GlobalExpressionRelaxation, OptimizeError> {
    let analysis = analyze_expression(expression, bounds)?;
    let count = expression.nodes.len();
    let mut rows = Vec::new();
    let mut complete = true;
    for (index, node) in expression.nodes.iter().enumerate() {
        let row = || vec![0.0; count];
        match *node {
            NonlinearNode::Constant(value) => {
                let mut c = row();
                c[index] = 1.0;
                rows.push(eq(c, value));
            }
            NonlinearNode::Parameter(_) => {}
            NonlinearNode::Add(a, b) => {
                let mut c = row();
                c[index] = 1.0;
                c[a] = -1.0;
                c[b] -= 1.0;
                rows.push(eq(c, 0.0));
            }
            NonlinearNode::Subtract(a, b) => {
                let mut c = row();
                c[index] = 1.0;
                c[a] = -1.0;
                c[b] += 1.0;
                rows.push(eq(c, 0.0));
            }
            NonlinearNode::Negate(a) => {
                let mut c = row();
                c[index] = 1.0;
                c[a] = 1.0;
                rows.push(eq(c, 0.0));
            }
            NonlinearNode::Multiply(a, b) => {
                let x = analysis.nodes[a].interval;
                let y = analysis.nodes[b].interval;
                if x.is_point() {
                    let mut c = row();
                    c[index] = 1.0;
                    c[b] = -x.lower;
                    rows.push(eq(c, 0.0));
                } else if y.is_point() {
                    let mut c = row();
                    c[index] = 1.0;
                    c[a] = -y.lower;
                    rows.push(eq(c, 0.0));
                } else if x.is_finite() && y.is_finite() {
                    let mut c = row();
                    c[index] = 1.0;
                    c[b] = -x.lower;
                    c[a] -= y.lower;
                    rows.push(lower(c, -x.lower * y.lower));
                    let mut c = row();
                    c[index] = 1.0;
                    c[b] = -x.upper;
                    c[a] -= y.upper;
                    rows.push(lower(c, -x.upper * y.upper));
                    let mut c = row();
                    c[index] = 1.0;
                    c[b] = -x.upper;
                    c[a] -= y.lower;
                    rows.push(upper(c, -x.upper * y.lower));
                    let mut c = row();
                    c[index] = 1.0;
                    c[b] = -x.lower;
                    c[a] -= y.upper;
                    rows.push(upper(c, -x.lower * y.upper));
                } else {
                    complete = false;
                }
            }
            NonlinearNode::Divide(a, b)
                if analysis.nodes[b].interval.is_point()
                    && analysis.nodes[b].interval.lower != 0.0 =>
            {
                let mut c = row();
                c[index] = 1.0;
                c[a] = -1.0 / analysis.nodes[b].interval.lower;
                rows.push(eq(c, 0.0));
            }
            NonlinearNode::Exp(a) => {
                complete &= add_unary_envelope(
                    &mut rows,
                    count,
                    index,
                    a,
                    analysis.nodes[a].interval,
                    Unary::Exp,
                    true,
                )
            }
            NonlinearNode::Log(a) => {
                complete &= add_unary_envelope(
                    &mut rows,
                    count,
                    index,
                    a,
                    analysis.nodes[a].interval,
                    Unary::Log,
                    false,
                )
            }
            NonlinearNode::Sqrt(a) => {
                complete &= add_unary_envelope(
                    &mut rows,
                    count,
                    index,
                    a,
                    analysis.nodes[a].interval,
                    Unary::Sqrt,
                    false,
                )
            }
            NonlinearNode::Powf(a, p) => {
                let convex = analysis.nodes[index].convexity.is_convex()
                    && !analysis.nodes[index].convexity.is_concave();
                let concave = analysis.nodes[index].convexity.is_concave()
                    && !analysis.nodes[index].convexity.is_convex();
                if p == 1.0 {
                    let mut c = row();
                    c[index] = 1.0;
                    c[a] = -1.0;
                    rows.push(eq(c, 0.0));
                } else if p == 0.0 {
                    let mut c = row();
                    c[index] = 1.0;
                    rows.push(eq(c, 1.0));
                } else if convex || concave {
                    complete &= add_pow_envelope(
                        &mut rows,
                        count,
                        index,
                        a,
                        analysis.nodes[a].interval,
                        p,
                        convex,
                    );
                } else {
                    complete = false;
                }
            }
            _ => {
                if !analysis.nodes[index].interval.is_point() {
                    complete = false;
                }
            }
        }
    }
    Ok(GlobalExpressionRelaxation {
        analysis,
        rows,
        complete,
    })
}

pub fn certify_convex_model(
    model: &NonlinearModel,
    constraints: &[NonlinearConstraint],
) -> Result<NonlinearConvexityCertificate, OptimizeError> {
    model.validate()?;
    let objective = analyze_expression(
        model.objective.as_ref().ok_or_else(|| {
            OptimizeError::InvalidProblem(
                "convexity certification requires a scalar objective".into(),
            )
        })?,
        &model.bounds,
    )?;
    let constraint_analyses = constraints
        .iter()
        .map(|c| analyze_expression(&c.expression, &model.bounds))
        .collect::<Result<Vec<_>, _>>()?;
    let objective_ok = objective.output_analysis().convexity.is_convex();
    let constraints_ok =
        constraints
            .iter()
            .zip(&constraint_analyses)
            .all(|(constraint, analysis)| {
                let curvature = analysis.output_analysis().convexity;
                let lower_ok = constraint.bound.lower.is_none() || curvature.is_concave();
                let upper_ok = constraint.bound.upper.is_none() || curvature.is_convex();
                lower_ok && upper_ok
            });
    let finite_parameter_box = model
        .bounds
        .iter()
        .all(|b| b.lower.is_some() && b.upper.is_some());
    let mut relaxation_complete =
        relax_expression(model.objective.as_ref().unwrap(), &model.bounds)?.complete;
    for constraint in constraints {
        relaxation_complete &= relax_expression(&constraint.expression, &model.bounds)?.complete;
    }
    Ok(NonlinearConvexityCertificate {
        objective,
        constraints: constraint_analyses,
        globally_convex: objective_ok && constraints_ok,
        finite_parameter_box,
        relaxation_complete,
    })
}

#[derive(Clone, Copy)]
enum Unary {
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
}
fn unary_value(kind: Unary, x: f64) -> f64 {
    match kind {
        Unary::Exp => x.exp(),
        Unary::Log => x.ln(),
        Unary::Sqrt => x.sqrt(),
        Unary::Sin => x.sin(),
        Unary::Cos => x.cos(),
    }
}
fn unary_derivative(kind: Unary, x: f64) -> f64 {
    match kind {
        Unary::Exp => x.exp(),
        Unary::Log => 1.0 / x,
        Unary::Sqrt => 0.5 / x.sqrt(),
        Unary::Sin => x.cos(),
        Unary::Cos => -x.sin(),
    }
}
fn unary_analysis(a: &NodeAnalysis, kind: Unary) -> NodeAnalysis {
    let i = a.interval;
    if i.is_point() {
        let value = unary_value(kind, i.lower);
        return if value.is_finite() {
            NodeAnalysis {
                interval: Interval {
                    lower: value,
                    upper: value,
                },
                convexity: ExpressionConvexity::Constant,
            }
        } else {
            NodeAnalysis {
                interval: Interval::whole(),
                convexity: ExpressionConvexity::Unknown,
            }
        };
    }
    match kind {
        Unary::Exp => NodeAnalysis {
            interval: Interval {
                lower: i.lower.exp(),
                upper: i.upper.exp(),
            },
            convexity: if a.convexity.is_convex() {
                ExpressionConvexity::Convex
            } else {
                ExpressionConvexity::Unknown
            },
        },
        Unary::Log if i.lower > 0.0 => NodeAnalysis {
            interval: Interval {
                lower: i.lower.ln(),
                upper: i.upper.ln(),
            },
            convexity: if a.convexity.is_concave() {
                ExpressionConvexity::Concave
            } else {
                ExpressionConvexity::Unknown
            },
        },
        Unary::Sqrt if i.lower >= 0.0 => NodeAnalysis {
            interval: Interval {
                lower: i.lower.sqrt(),
                upper: i.upper.sqrt(),
            },
            convexity: if a.convexity.is_concave() {
                ExpressionConvexity::Concave
            } else {
                ExpressionConvexity::Unknown
            },
        },
        Unary::Sin | Unary::Cos => NodeAnalysis {
            interval: Interval {
                lower: -1.0,
                upper: 1.0,
            },
            convexity: if i.is_point() {
                ExpressionConvexity::Constant
            } else {
                ExpressionConvexity::Unknown
            },
        },
        _ => NodeAnalysis {
            interval: Interval::whole(),
            convexity: ExpressionConvexity::Unknown,
        },
    }
}
fn pow_analysis(a: &NodeAnalysis, p: f64) -> NodeAnalysis {
    let i = a.interval;
    if p == 0.0 {
        return NodeAnalysis {
            interval: Interval {
                lower: 1.0,
                upper: 1.0,
            },
            convexity: ExpressionConvexity::Constant,
        };
    }
    if p == 1.0 {
        return a.clone();
    }
    if i.is_point() {
        let value = i.lower.powf(p);
        return if value.is_finite() {
            NodeAnalysis {
                interval: Interval {
                    lower: value,
                    upper: value,
                },
                convexity: ExpressionConvexity::Constant,
            }
        } else {
            NodeAnalysis {
                interval: Interval::whole(),
                convexity: ExpressionConvexity::Unknown,
            }
        };
    }
    let integer = p.fract() == 0.0;
    let interval = if integer && p > 0.0 {
        let lo = i.lower.powf(p);
        let hi = i.upper.powf(p);
        if (p as i64) % 2 == 0 && i.lower <= 0.0 && i.upper >= 0.0 {
            Interval {
                lower: 0.0,
                upper: lo.max(hi),
            }
        } else {
            Interval {
                lower: lo.min(hi),
                upper: lo.max(hi),
            }
        }
    } else if i.lower >= 0.0 && p > 0.0 {
        Interval {
            lower: i.lower.powf(p),
            upper: i.upper.powf(p),
        }
    } else {
        Interval::whole()
    };
    let even_affine = (p == 2.0 || (integer && p >= 2.0 && (p as i64) % 2 == 0))
        && a.convexity == ExpressionConvexity::Affine;
    let positive_power = p > 1.0 && i.lower >= 0.0 && a.convexity.is_convex();
    let convexity = if even_affine || positive_power {
        ExpressionConvexity::Convex
    } else if p > 0.0 && p < 1.0 && i.lower >= 0.0 && a.convexity.is_concave() {
        ExpressionConvexity::Concave
    } else {
        ExpressionConvexity::Unknown
    };
    NodeAnalysis {
        interval,
        convexity,
    }
}
fn combine_add(a: NodeAnalysis, b: NodeAnalysis, subtract: bool) -> NodeAnalysis {
    let bcur = if subtract {
        b.convexity.negated()
    } else {
        b.convexity
    };
    let convexity = combine_curvature(a.convexity, bcur);
    let interval = if subtract {
        Interval {
            lower: a.interval.lower - b.interval.upper,
            upper: a.interval.upper - b.interval.lower,
        }
    } else {
        Interval {
            lower: a.interval.lower + b.interval.lower,
            upper: a.interval.upper + b.interval.upper,
        }
    };
    NodeAnalysis {
        interval,
        convexity,
    }
}
fn combine_curvature(a: ExpressionConvexity, b: ExpressionConvexity) -> ExpressionConvexity {
    use ExpressionConvexity::*;
    if a == Constant {
        return b;
    }
    if b == Constant {
        return a;
    }
    match (a, b) {
        (Affine, Affine) => Affine,
        (Affine, Convex) | (Convex, Affine) | (Convex, Convex) => Convex,
        (Affine, Concave) | (Concave, Affine) | (Concave, Concave) => Concave,
        _ => Unknown,
    }
}
fn multiply_analysis(a: &NodeAnalysis, b: &NodeAnalysis, same: bool) -> NodeAnalysis {
    let interval = mul_interval(a.interval, b.interval);
    if a.interval.is_point() {
        return NodeAnalysis {
            interval,
            convexity: scale_curvature(b.convexity, a.interval.lower),
        };
    }
    if b.interval.is_point() {
        return NodeAnalysis {
            interval,
            convexity: scale_curvature(a.convexity, b.interval.lower),
        };
    }
    let convexity = if same && a.convexity == ExpressionConvexity::Affine {
        ExpressionConvexity::Convex
    } else {
        ExpressionConvexity::Unknown
    };
    NodeAnalysis {
        interval,
        convexity,
    }
}
fn divide_analysis(a: &NodeAnalysis, b: &NodeAnalysis) -> NodeAnalysis {
    if b.interval.is_point() && b.interval.lower != 0.0 {
        return NodeAnalysis {
            interval: mul_interval(
                a.interval,
                Interval {
                    lower: 1.0 / b.interval.lower,
                    upper: 1.0 / b.interval.lower,
                },
            ),
            convexity: scale_curvature(a.convexity, 1.0 / b.interval.lower),
        };
    }
    if b.interval.lower <= 0.0 && b.interval.upper >= 0.0 {
        return NodeAnalysis {
            interval: Interval::whole(),
            convexity: ExpressionConvexity::Unknown,
        };
    }
    let reciprocal = Interval {
        lower: (1.0 / b.interval.lower).min(1.0 / b.interval.upper),
        upper: (1.0 / b.interval.lower).max(1.0 / b.interval.upper),
    };
    NodeAnalysis {
        interval: mul_interval(a.interval, reciprocal),
        convexity: ExpressionConvexity::Unknown,
    }
}
fn scale_curvature(c: ExpressionConvexity, k: f64) -> ExpressionConvexity {
    if k == 0.0 {
        ExpressionConvexity::Constant
    } else if k < 0.0 {
        c.negated()
    } else {
        c
    }
}
fn neg(i: Interval) -> Interval {
    Interval {
        lower: -i.upper,
        upper: -i.lower,
    }
}
fn mul_interval(a: Interval, b: Interval) -> Interval {
    let values = [
        a.lower * b.lower,
        a.lower * b.upper,
        a.upper * b.lower,
        a.upper * b.upper,
    ];
    if values.iter().any(|v| v.is_nan()) {
        Interval::whole()
    } else {
        Interval {
            lower: values.iter().copied().fold(f64::INFINITY, f64::min),
            upper: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}
fn eq(coefficients: Vec<f64>, value: f64) -> AffineRelaxationRow {
    AffineRelaxationRow {
        coefficients,
        lower: Some(value),
        upper: Some(value),
    }
}
fn lower(coefficients: Vec<f64>, value: f64) -> AffineRelaxationRow {
    AffineRelaxationRow {
        coefficients,
        lower: Some(value),
        upper: None,
    }
}
fn upper(coefficients: Vec<f64>, value: f64) -> AffineRelaxationRow {
    AffineRelaxationRow {
        coefficients,
        lower: None,
        upper: Some(value),
    }
}
fn add_unary_envelope(
    rows: &mut Vec<AffineRelaxationRow>,
    count: usize,
    z: usize,
    x: usize,
    i: Interval,
    kind: Unary,
    convex: bool,
) -> bool {
    if i.is_point() {
        return true;
    }
    if !i.is_finite() {
        return false;
    }
    let fl = unary_value(kind, i.lower);
    let fu = unary_value(kind, i.upper);
    let mid = (i.lower + i.upper) / 2.0;
    let fm = unary_value(kind, mid);
    let d = unary_derivative(kind, mid);
    if ![fl, fu, fm, d].iter().all(|v| v.is_finite()) {
        return false;
    }
    let slope = (fu - fl) / (i.upper - i.lower);
    let intercept = fl - slope * i.lower;
    let tangent = fm - d * mid;
    let mut sec = vec![0.0; count];
    sec[z] = 1.0;
    sec[x] = -slope;
    let mut tan = vec![0.0; count];
    tan[z] = 1.0;
    tan[x] = -d;
    if convex {
        rows.push(lower(tan, tangent));
        rows.push(upper(sec, intercept));
    } else {
        rows.push(lower(sec, intercept));
        rows.push(upper(tan, tangent));
    }
    true
}
fn add_pow_envelope(
    rows: &mut Vec<AffineRelaxationRow>,
    count: usize,
    z: usize,
    x: usize,
    i: Interval,
    p: f64,
    convex: bool,
) -> bool {
    if i.is_point() {
        return true;
    }
    if !i.is_finite() {
        return false;
    }
    let fl = i.lower.powf(p);
    let fu = i.upper.powf(p);
    let mid = (i.lower + i.upper) / 2.0;
    let fm = mid.powf(p);
    let d = p * mid.powf(p - 1.0);
    if ![fl, fu, fm, d].iter().all(|v| v.is_finite()) {
        return false;
    }
    let slope = (fu - fl) / (i.upper - i.lower);
    let intercept = fl - slope * i.lower;
    let tangent = fm - d * mid;
    let mut sec = vec![0.0; count];
    sec[z] = 1.0;
    sec[x] = -slope;
    let mut tan = vec![0.0; count];
    tan[z] = 1.0;
    tan[x] = -d;
    if convex {
        rows.push(lower(tan, tangent));
        rows.push(upper(sec, intercept));
    } else {
        rows.push(lower(sec, intercept));
        rows.push(upper(tan, tangent));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifted_square_is_convex_and_relaxed() {
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Constant(2.4),
                NonlinearNode::Subtract(0, 1),
                NonlinearNode::Powf(2, 2.0),
            ],
            3,
        );
        let bounds = [Bound {
            lower: Some(0.0),
            upper: Some(4.0),
        }];
        let analysis = analyze_expression(&expression, &bounds).unwrap();
        assert_eq!(
            analysis.output_analysis().convexity,
            ExpressionConvexity::Convex
        );
        assert_eq!(
            analysis.output_analysis().interval,
            Interval {
                lower: 0.0,
                upper: 5.76
            }
        );
        let relaxation = relax_expression(&expression, &bounds).unwrap();
        assert!(relaxation.complete);
        assert!(relaxation.rows.len() >= 4);
    }
    #[test]
    fn bilinear_product_gets_mccormick_rows() {
        let expression = NonlinearExpression::new(
            vec![
                NonlinearNode::Parameter(0),
                NonlinearNode::Parameter(1),
                NonlinearNode::Multiply(0, 1),
            ],
            2,
        );
        let bounds = [
            Bound {
                lower: Some(-1.0),
                upper: Some(2.0),
            },
            Bound {
                lower: Some(3.0),
                upper: Some(5.0),
            },
        ];
        let relaxation = relax_expression(&expression, &bounds).unwrap();
        assert!(relaxation.complete);
        assert_eq!(relaxation.rows.len(), 4);
        assert_eq!(
            relaxation.analysis.output_analysis().interval,
            Interval {
                lower: -5.0,
                upper: 10.0
            }
        );
    }
    #[test]
    fn model_certificate_rejects_nonconvex_constraint_direction() {
        let square = NonlinearExpression::new(
            vec![NonlinearNode::Parameter(0), NonlinearNode::Powf(0, 2.0)],
            1,
        );
        let model = NonlinearModel {
            parameter_count: 1,
            bounds: vec![Bound {
                lower: Some(-2.0),
                upper: Some(2.0),
            }],
            objective: Some(square.clone()),
            residuals: vec![],
        };
        let convex = certify_convex_model(
            &model,
            &[NonlinearConstraint {
                expression: square.clone(),
                bound: Bound {
                    lower: None,
                    upper: Some(1.0),
                },
            }],
        )
        .unwrap();
        assert!(convex.globally_convex);
        let nonconvex = certify_convex_model(
            &model,
            &[NonlinearConstraint {
                expression: square,
                bound: Bound {
                    lower: Some(1.0),
                    upper: None,
                },
            }],
        )
        .unwrap();
        assert!(!nonconvex.globally_convex);
    }
}
