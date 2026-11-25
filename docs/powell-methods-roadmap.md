# Powell methods roadmap

NumericCore already provides Powell's direction-set method and a
Powell–Hestenes–Rockafellar augmented Lagrangian. Its model-based bounded
and constrained derivative-free solvers provide starting points for the
later Powell methods. This iteration adds Powell damping to the Rust and
Swift BFGS-SQP updates. Damping blends the Lagrangian gradient difference
with the current Hessian action when curvature is weak. The regression
case checks that a negative curvature observation no longer discards
curvature in an independent direction.

## 1. BOBYQA

NumericCore exposes BOBYQA through `minimize_bobyqa` in Rust and
`BobyqaSolver` in Swift. Both use version 0.3.2 of the pure-Rust PRIMA port.
The port maintains a quadratic interpolation set with minimum-Frobenius
Hessian-change updates, geometry steps, and RESCUE for loss of model
accuracy. Its default configuration follows PRIMA's trajectory.

NumericCore removes fixed coordinates before interpolation, projects the
start into the box, and caps the initial radius at half the narrowest
nonfixed interval. The evaluation count measures calls to the objective,
including model initialization. The caller's budget applies even before
enough points have been sampled to construct the first quadratic. The
objective must be finite with magnitude below `1e30`.

`RadiusReached` reports trust-region resolution. `EvaluationLimit` and
`ModelDegenerate` return the best evaluated point with their respective
stop reasons. An all-fixed box is evaluated once. The manifest lists BOBYQA
as provisional, with a synchronous value callback on the UniFFI path.

The contract suite checks a rotated quadratic with an active bound, a
one-dimensional fit after fixed-coordinate elimination, and a synthetic
exponential-response calibration. Under a shared budget of 160 calls on the
calibration, BOBYQA reached a residual sum of squares of `1.69e-13` in
74 calls. The diagonal-model solver reached `2.77e-2` in 155 calls on
the same fixture.

[Powell's BOBYQA report](https://optimization-online.org/wp-content/uploads/2010/05/2616.pdf) ·
[PRIMA](https://github.com/libprima/prima) ·
[Pinned Rust port](https://crates.io/crates/bobyqa/0.3.2) ·
[Third-party notice](../third-party/bobyqa/PRIMA-NOTICE)

## 2. COBYLA and LINCOA

Implement COBYLA's simplex interpolation and constraint-aware trust-region
subproblem for black-box inequalities. Compare behavior from infeasible
starts and at active constraints. Evaluate equality handling separately.
Implement LINCOA for a black-box objective with explicit linear inequalities.
Compare feasibility and evaluation counts on active-set changes, dependent
rows, and poorly scaled constraints.

[Powell's COBYLA chapter](https://link.springer.com/chapter/10.1007/978-94-015-8330-5_4) ·
[Powell's LINCOA subproblem paper](https://link.springer.com/article/10.1007/s12532-015-0084-4)

## 3. NEWUOA and UOBYQA

Reuse validated interpolation components for NEWUOA's unconstrained
least-change quadratic models. Add UOBYQA as a full-quadratic reference
for small problems. Compare the methods under equal evaluation budgets
on coupled quadratics and nonquadratic functions.

[Powell's NEWUOA report](https://optimization-online.org/2005/01/1045/) ·
[Powell's UOBYQA paper](https://doi.org/10.1007/s101070100290)

## 4. Powell hybrid method

Add a Jacobian-based trust-region dogleg method for square root systems.
Use rank-one Jacobian updates between refreshes and distinguish residual
convergence from a small step with a nonzero residual. Validate against
the existing Newton solver and MINPACK reference cases before deciding
whether to add a finite-difference entry point.

[Argonne MINPACK guide](https://www.mcs.anl.gov/~more/ANL8074b.pdf)

For each solver, record reference provenance, agree on termination and
evaluation-budget semantics, and run conformance cases in both codebases.
Track objective and feasibility tolerances and evaluation counts separately
from wall time. Use the established release workflow for any new FFI
surface.
