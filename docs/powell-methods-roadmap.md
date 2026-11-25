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

`minimize_cobyla` in Rust and `CobylaSolver` in Swift share Basin 3.6.0's
PRIMA-based implementation. It uses simplex interpolation and a
constraint-aware trust-region step. One callback returns the objective
and all residuals for a point. Inequality residuals use `c(x) <= 0`, and
equality residuals become two inequalities of opposite sign.

NumericCore eliminates fixed coordinates and restores them for each callback.
COBYLA can evaluate points outside the variable bounds and nonlinear
constraints. The evaluation cap includes simplex initialization, and the
result reports the best feasible evaluation under the configured tolerance
or the least-violating evaluation when the run found no feasible point.
The original residuals, maximum violation, and feasibility flag accompany
the termination reason. Reaching the final radius describes resolution.
Model values must be finite with magnitude at most `2^100`.

The contract suite checks an infeasible start with a curved active
constraint, equality restoration with a fixed coordinate, a budget too
small to initialize the simplex, and malformed or throwing callbacks.
On the curved-constraint fixture, both methods reached an objective near
one within the same 220-call budget and feasibility tolerance. COBYLA
used 71 calls and the existing constrained derivative-free method used 219.

`minimize_lincoa` in Rust and `LincoaSolver` in Swift use Basin 3.6.0's
LINCOA for a value-only objective over continuous linear inequalities.
Both use the existing sparse linear model for the feasible region. The
callback supplies the objective, so the model's linear cost is unused.
NumericCore eliminates fixed coordinates, translates the constraints about
the feasible starting point, scales the rows, and passes a dense constraint
block to Basin. Nonconstant equality rows and integer variables are unsupported.

The objective may be evaluated outside the constraints and bounds during
interpolation. NumericCore recomputes feasibility from the original rows
and bounds using the requested absolute tolerance and returns the best
eligible evaluation. The evaluation budget includes initialization. Reaching
the final radius describes resolution. The contract suite checks active-set
changes with a dependent row scaled from `1e-160` to `1e160` and compares
LINCOA and COBYLA on the same linear model under a shared 100-call budget
and feasibility tolerance. Both reached the analytical objective of 2.125
within `1e-6`. LINCOA used 29 calls and COBYLA used 13 on that fixture.

[Powell's COBYLA chapter](https://link.springer.com/chapter/10.1007/978-94-015-8330-5_4) ·
[Basin COBYLA](https://docs.rs/basin/3.6.0/basin/struct.Cobyla.html) ·
[Third-party notice](../third-party/basin/COPYRIGHT) ·
[Powell's LINCOA subproblem paper](https://link.springer.com/article/10.1007/s12532-015-0084-4) ·
[Basin LINCOA](https://docs.rs/basin/3.6.0/basin/struct.Lincoa.html)

## 3. NEWUOA and UOBYQA

`minimize_newuoa` and `NewuoaSolver` run Basin 3.6.0's NEWUOA with
`2n + 1` interpolation points and least-Frobenius-norm Hessian updates.

`minimize_uobyqa` and `UobyqaSolver` run NumericCore's dense UOBYQA reference
variant on up to 16 variables. It uses `(n + 1)(n + 2)/2` points to determine
a full quadratic and selects replacement points using quadratic Lagrange
polynomials. It refactors the scaled interpolation system directly and
uses dense eigensolves for the objective and geometry subproblems. Its
radius and geometry rules are specific to NumericCore.

Both methods count the initial interpolation samples against the callback
budget. They return the best evaluated point and distinguish radius
resolution, iteration limits, evaluation limits, and numerical failure.
On the contract fixtures, both methods reached objective values below
`1e-10` with a 500-call budget. NEWUOA and UOBYQA used 62 and 86 calls
on the three-variable coupled quadratic, and 149 and 122 on the
two-variable Rosenbrock problem.

[Powell's NEWUOA report](https://optimization-online.org/2005/01/1045/) ·
[Basin NEWUOA](https://docs.rs/basin/3.6.0/basin/struct.Newuoa.html) ·
[Powell's UOBYQA paper](https://doi.org/10.1007/s101070100290)

## 4. Powell hybrid method

`solve_powell_hybrid` in Rust and `NonlinearSystemSolver.solvePowellHybrid`
in Swift provide a dense dogleg method for square systems with analytic
Jacobians. The step combines the Newton and scaled steepest-descent
directions inside a trust region. The solver applies good Broyden updates
between analytic refreshes and refreshes after repeated poor reduction.
It refactors the current dense Jacobian by QR when computing the Newton
endpoint. The equation-graph entry point evaluates values at trial points
and differentiates the graph at Jacobian refreshes.

The result records the weighted residual norm, trial iterations, residual
calls, analytic Jacobian calls, and rank-one updates. Only reaching the
residual tolerance reports convergence. A small step or radius with a
remaining residual reports a stall. The contract suite includes the
MINPACK tridiagonal example, Rosenbrock equations, a scaled rank-deficient
system, and a stationary point with no root. On the nine-variable MINPACK
example, hybrid used 12 residual calls and one analytic Jacobian call.
Damped Newton reached the same root using six combined residual-and-Jacobian
calls from the same starting point. A finite-difference Jacobian entry point
remains a separate extension.

[Argonne MINPACK guide](https://www.mcs.anl.gov/~more/ANL8074b.pdf)

For each solver, record reference provenance, agree on termination and
evaluation-budget semantics, and run conformance cases in both codebases.
Track objective and feasibility tolerances and evaluation counts separately
from wall time. Use the established release workflow for any new FFI
surface.
