# Phase 8: bound constraints and robust fitting

`nc-optimize` now exposes box-constrained limited-memory optimization through
`minimize_lbfgsb` and weighted robust nonlinear least squares through
`nonlinear_least_squares_configured`.

The bounded optimizer projects its initial point and trial points, suppresses
directions that leave an active bound, and tests convergence with the projected
gradient. Its projected Armijo search and curvature updates use the actual
feasible displacement.

Configured nonlinear least squares accepts one bound per parameter, optional
non-negative observation weights, and squared, Huber, or Cauchy loss. The
Levenberg–Marquardt system uses IRLS weights; candidate acceptance and stopping
use the true selected robust objective. An empty weights slice means unit
weights, and a zero weight excludes an observation without changing dimensions.

The Swift implementation intentionally mirrors these semantics through
`LBFGSB`, `ParameterBound`, `RobustLoss`, and the configured
`NonlinearLeastSquares.solve` overload.
