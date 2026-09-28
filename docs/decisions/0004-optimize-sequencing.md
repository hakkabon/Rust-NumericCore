# 0004 — AMPL/optimization layer: LP → MILP → NLP sequencing

## Status
Accepted.

## Context
An AMPL-style system decomposes into four subsystems: modeling language,
presolve, solver interface, and solvers. The solver piece itself splits
further into LP (tractable, well-understood — simplex or interior-point),
MILP (branch-and-bound/cut — substantial, and competitive
general-purpose MILP is a research area commercial vendors spend decades
on), and NLP (needs automatic differentiation, has deep numerical
stability subtleties — flagged during scoping as the item most likely to
produce open-ended surprises).

Estimated effort discussed during scoping: LP-complete system ~2–3 years
at solo/part-time pace; adding MILP ~3.5–5 years; full NLP support
pushes further still. Building all three simultaneously risks a
long stretch with nothing working end-to-end.

## Decision
Sequence strictly: **modeling language → presolve → `Problem` format →
LP solver**, get that whole pipeline working end-to-end, and only then
decide whether MILP or NLP is actually needed based on real usage.

Concretely in this codebase:
- `nc-optimize::Problem`/`Solution`/`Solver` (the ASL-equivalent
  interface) is built now, deliberately solver-agnostic, so it doesn't
  need to change shape when a real solver replaces `StubSolver`.
- `StubSolver` exists purely to prove the pipeline (parse → presolve →
  `Problem` → `Solver` → `Solution`) can be wired and tested end-to-end
  before any real numerical solving exists.
- No MILP- or NLP-specific types exist yet anywhere in `nc-optimize`.
  `Bound` (used for both variable and constraint bounds) is deliberately
  general enough to support range constraints and fixed variables, which
  LP already needs — it wasn't designed with MILP integrality
  constraints in mind and will need extension (e.g. an `is_integer: Vec<bool>`
  field on `Problem`) if/when MILP work starts.

## Consequences
- The project can demonstrate real value (declaring a model, solving an
  LP) well before MILP/NLP exist, rather than nothing working until
  everything works.
- `Problem`'s current shape may need a breaking addition for MILP
  (integrality flags) and a larger one for NLP (nonlinear expression
  trees can't be represented as a flat objective vector + linear
  constraint matrix — NLP likely needs a materially different `Problem`
  variant, not an extension of this one).
- Whether MILP or NLP is built at all is deferred to a real decision
  point informed by actual usage, not decided now.

## Alternatives considered
- **Design `Problem` to accommodate MILP/NLP from day one** (e.g.
  including integrality flags and nonlinear expression support now).
  Rejected: adds real complexity to the LP path for capabilities that
  may never be exercised, and risks guessing wrong about what NLP's
  actual data model needs before any NLP work has started.

## Update (LP: revised simplex implemented)
`nc-optimize::simplex::RevisedSimplexSolver` — a real `Solver`, not
`StubSolver` — is implemented and tested (10/10 tests passing,
including a hand-verified-by-geometry solve of the AMPL grammar doc's
own example, a case that genuinely exercises Phase 1, equality
constraints, infeasibility/unboundedness detection, and free
variables). See `simplex.rs`'s module docs for the full formulation
(bounded equality form via one slack per row, a composite Phase 1
needing no artificial variables or Big-M constant since the slacks
already play that role) and its deliberate scope (dense, not
sparse-with-incremental-factorization; Dantzig's rule, not Bland's —
both fine for the "smallish problems" this path targets, real future
work if a large-problem need shows up).

**Solver selection is explicit, not policy-based** — a caller picks
`RevisedSimplexSolver` vs (future) `InteriorPointSolver` by name/type,
mirroring the discussion that led here: auto-selection (a
`DispatchPolicy`-style mechanism choosing a solver by problem
size/density) is exactly the kind of thing this project's principles
say not to build until a concrete case shows the wrong default
actually hurting someone.

Interior-point is next, expected to lean on `solveSPD` (Swift side) or
a Rust-native Cholesky, since primal-dual path-following's Newton
system reduces to a symmetric positive-definite normal-equations solve
each iteration — a nice, unplanned callback to the SPD work that
happened earlier for `Swift-DataLens`'s benefit.

## Update (interior-point implemented)
`nc-optimize::interior_point::InteriorPointSolver` — a primal-dual
path-following method, implemented and tested (17/17 tests passing in
`nc-optimize`, full workspace green). The SPD prediction above was
correct: eliminating the barrier method's per-iteration Newton system
down to `(A D⁻¹ Aᵀ) dy = rhs` gives a genuinely SPD system, solved via
`nc_decomp::cholesky_solve` — a small, dense, pure-Rust Cholesky added
to `nc-decomp` (previously an empty scaffold) specifically for this,
since `nc-optimize` has no Accelerate available the way the Swift side
does.

Two real scope boundaries this solver has that `RevisedSimplexSolver`
doesn't, both documented in `interior_point.rs`'s module docs rather
than silently handled: it **rejects equality constraints and fixed
variables** (a barrier method has no interior to work in across a
zero-width bound — real barrier-method implementations handle this via
a separate presolve/elimination step this solver doesn't have yet, not
inside the barrier iteration), and its unboundedness detection is
**heuristic, not certificate-based** (a proper answer needs a
self-dual embedding; this solver instead substitutes a large finite
bound for a missing one and flags a final solution that ends up
suspiciously close to that substitute).

**Correctness confidence for this one is higher than usual**: two of
the test cases solve the exact same problems already hand-verified for
`RevisedSimplexSolver` (the AMPL grammar doc's own example, and a
3-variable greedy-knapsack-style LP) and check that interior-point's
independently-derived iteration converges to the same optimum simplex
found by an entirely different method — a real cross-validation, not
just two solvers separately trusting their own hand-computed answers.

## Update (MILP: branch-and-bound implemented)
`nc-optimize::branch_and_bound::BranchAndBoundSolver` — LP-relaxation
branch-and-bound, wrapping `RevisedSimplexSolver` (pluggable, but
documented as the required default — see that module's docs on why
`InteriorPointSolver`'s heuristic unboundedness detection and its
equality/fixed-variable rejection make it a poor fit as the relaxation
solver here, since a branch's floor/ceil bound can easily pin a
variable to a single value). `Problem` gained the `is_integer: Vec<bool>`
field this ADR's original text anticipated MILP would need — every
existing `Solver` (`RevisedSimplexSolver`, `InteriorPointSolver`)
ignores it entirely; only `BranchAndBoundSolver` reads it.

Depth-first search, most-fractional-variable branching, a single
pruning rule (relaxation bound no better than the current incumbent),
and a node-count cap rather than a proven-optimal stopping rule —
matches this whole LP/MILP path's established "smallish problems,
document the honest simplification rather than attempt full production
sophistication" stance. 7/7 new tests pass, including a classic
2-variable MILP whose LP relaxation lands on a genuinely fractional
vertex (verified by hand: relaxation optimum (3, 1.5) objective 21;
integer optimum (4, 0) objective 20, confirmed by enumerating every
nearby feasible integer point) and a 0/1 knapsack. Full workspace green
(`nc-ffi`'s 28 tests included, confirming the `Problem.is_integer`
field addition didn't disturb the existing LP-only FFI path, which
always passes `is_integer: vec![false; n]` today).

## Update (MILP: wired through FFI)
`FfiProblem` gained `is_integer: Vec<bool>` and `nc-ffi` now exports
`solve_milp_branch_and_bound`, mirroring the pattern already
established for the two LP solvers. An empty `is_integer` list is
treated as "all continuous" (backward-compatible with callers built
against the pre-integrality `FfiProblem` shape, and with a caller that
genuinely has no integer variables) rather than a length-mismatch
error — a real length mismatch (non-empty but wrong length) is still
caught by `BranchAndBoundSolver` itself. 34/34 tests pass in `nc-ffi`
(31 previous + 3 new), including the same classic 2-variable MILP
already hand-verified on the Rust side crossing the FFI boundary
correctly, an empty-`is_integer` call matching `solve_lp_simplex`
exactly on the same problem, and integrality-caused infeasibility
detection. Full workspace green. Wiring `NumericCoreAMPL`'s Swift side
(a `var x integer;` grammar keyword, `Model`/`CompiledProblem` carrying
per-variable integrality, `Solve.swift` gaining a `.branchAndBound`
case) is the follow-up on the Swift-NumericCore side.

## Update (Phase 3: complete LP/MILP integration)

All solver-independent problem dimensions, coefficients, and bounds are now
validated before an algorithm indexes parallel storage. Configurable UniFFI
entry points expose simplex, interior-point, and branch-and-bound limits and
tolerances while retaining the original default-configured functions.

Branch-and-bound additionally returns a search report: nodes explored, global
best bound, and absolute/relative optimality gaps when an incumbent exists.
A node-limit or incomplete LP relaxation is reported as `IterationLimit`,
never collapsed into `Infeasible`. The legacy `Solver::solve` contract remains
available and projects the report down to its `Solution`.

## Update (Phase 5: deepen the native solver, retain an honest boundary)

The production decision is to continue the native LP/MILP path as an
embedded, inspectable solver for small and medium structured models, while
not presenting it as a replacement for mature general-purpose engines.
That keeps NumericCore self-contained for its intended workloads and makes
the solver a useful numerical foundation without turning feature parity
with commercial MILP systems into an implicit promise.

The first deepening step changes branch-and-bound's default traversal from
depth-first to best-bound, retains depth-first as an explicit policy, adds
absolute and relative gap stopping, and records the termination reason,
infeasibility/bound pruning counts, and maximum tree depth. Gap termination
is deliberately distinguishable from exact tree exhaustion in the native
report. Revised simplex now detects a repeated basis and switches from
Dantzig pricing to Bland-style deterministic entering/leaving choices to
prevent cycling on degenerate relaxations.

The next release boundary should extend the UniFFI records with the new
search policy, gap tolerances, termination reason, and pruning telemetry.
The existing FFI shape remains binary-compatible in this change: it adopts
best-bound internally and continues to expose its Phase 3 node/bound/gap
certificate. The Swift layer now exposes all configuration and report data
already present in that v0.8 boundary, so callers no longer need to reach
into generated bindings.

Before broadening the production claim, the native implementation still
needs sparse basis factorization with stable updates/refactorization,
presolve and scaling, warm-started child relaxations, stronger branching,
and cuts. Until then, large or operationally critical MILPs should use an
external production solver behind the same compiled-problem boundary.

## Update (Phase 6: nonlinear least squares and L-BFGS)

The first nonlinear layer is intentionally programmatic rather than an AMPL
grammar extension. `minimize_lbfgs` accepts a smooth objective and analytic
gradient, uses the limited-memory two-loop recursion, and globalizes each
iteration with Armijo backtracking. `nonlinear_least_squares` accepts residuals
and an analytic Jacobian and uses damped Gauss-Newton/Levenberg-Marquardt
steps. Both validate callback dimensions and finite arithmetic and return
iteration/evaluation counts plus explicit convergence or failure reasons.

These callback APIs do not cross UniFFI: arbitrary Swift closures are not a
sound release-binary interface. Swift-NumericCore therefore provides the same
contract natively in `NumericCoreOptimization`. A later expression graph,
automatic-differentiation representation, or reverse-communication protocol
can become the shared FFI boundary when constrained nonlinear optimization
and nonlinear AMPL syntax are designed. Convex quadratic programming remains
deferred as requested rather than being introduced indirectly here.
