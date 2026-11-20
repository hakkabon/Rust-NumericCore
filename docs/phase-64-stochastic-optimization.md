# Phase 64 — Stochastic and scenario optimization

`ScenarioTree` accepts a sparse linear block at each information node. Parents
precede children. A node can add columns from any strict ancestor to its row
activities. Every scenario descending from a node uses the same decision block,
so nonanticipativity is built into the column layout. Sibling probabilities
must be positive and sum to one. Node costs are incremental and accumulate
along each path.

`solve_scenario_tree` solves the sparse extensive form for expected cost or a
convex combination of expected cost and upper-tail CVaR of total path cost.
This is a static root risk measure. The solver supports continuous and
finite-bounded integer decisions, returns an incumbent policy when available,
and uses the audited LP tree to report lower bounds and limit statuses. The
integer search can be expensive on large trees. A reported unbounded relaxation
describes the relaxation used in the search.

For a finite two-stage distribution, `draw_scenarios` provides seeded IID
sampling with replacement. `sample_average_problem` merges duplicate draws into
empirical weights while preserving source order. Evaluate a fixed first-stage
decision with `evaluate_scenario_sample` on a separate draw stream. Its normal
95% interval estimates the expected realized cost of that fixed policy under
the input distribution. Small samples, rare events, or skewed costs can lead
to poor coverage. Scenario reduction uses greedy forward selection in caller-
scaled Euclidean feature space. Its weighted distance measures redistribution
in feature space.

`solve_progressive_hedging` solves continuous, expected-cost, two-stage linear
models using penalized QP subproblems. It reuses per-scenario workspaces and
can run them in bounded local worker threads with deterministic result order.
Versioned checkpoints include consensus, multipliers, and the QP warm starts.
Resume with the same model, scenario order, penalty, and tolerance. Iteration
and worker limits can change. The returned consensus and residuals support
iterative progress reporting. Solve the fixed-policy recourse problem to
check feasibility and evaluate cost. Cross-machine scheduling, nested risk
measures, and mixed-integer hedging are future extensions.

The Rust integration tests check analytical multistage and CVaR optima,
bounded-integer decisions, sampling, reduction, and checkpoint replay. The
UniFFI surface and capability manifest are versioned with the Rust artifact.
