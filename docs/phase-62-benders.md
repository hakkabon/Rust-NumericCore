# Phase 62 — Classical Benders and L-shaped decomposition

`solve_benders` minimizes first-stage cost plus the weighted recourse cost
of continuous LP scenarios. Construct a single deterministic recourse
block with `BendersProblem::classical`. For an L-shaped solve, pass multiple
`BendersScenario` values with distinct IDs and positive probabilities that
sum to one. The first-stage LP holds the constraints on x. Each scenario holds
W, T, the recourse objective and variable bounds, and the unshifted row
bounds in the convention `l ≤ W y + T x ≤ u`.

## Bounds and cuts

The master uses a lower bound for each recourse epigraph. The solver obtains
it from the recourse variable box or, when a contributing side is infinite,
an auxiliary LP over the first stage and that scenario. An auxiliary solve
with no verified finite floor ends with an explicit diagnostic. In particular,
some models with a bounded total objective can lack an individual finite
recourse floor.

At a trial x, each retained simplex workspace solves the LP with row bounds
shifted by `-T x`. A verified dual supplies the affine recourse minorant with
slope `-Tᵀπ`, where π is the row multiplier. The intercept uses the original
row bounds and the variable reduced costs. A Farkas certificate uses the same
construction with its separating variable coefficients. Its affine
contradiction must be nonpositive at any feasible first-stage point.

Multi-cut mode keeps one epigraph per scenario. Single-cut mode aggregates
the optimality planes with the scenario probabilities once all recourse
blocks are feasible. Feasibility cuts always retain their scenario identity.

## Repeated solves and reporting

The master and recourse LPs retain simplex workspaces. Parallel recourse
evaluation uses separate workspaces and gathers results in input order.
An uncertified warm solve is retried cold unless it hit its LP iteration
limit. Optional trust-region trials restrict first-stage bounds while the
unrestricted master supplies the lower bound. Active cuts are retained when
aging is enabled, and the best proven lower bound survives cut pruning.

`BendersResult` reports an incumbent only after every scenario has supplied
verified recourse evidence at the same first-stage point. Bounds, gaps,
cut counts, warm-start counts, and trust-region use are available in the
result and its iteration history. Convergence is numerical and uses the
requested absolute or relative gap. Infeasibility requires a verified
Farkas certificate on the master or auxiliary relaxation. A master
relaxation can report unboundedness without proving the original model
unbounded. The continuous implementation uses the current dense-basis
simplex backend.
