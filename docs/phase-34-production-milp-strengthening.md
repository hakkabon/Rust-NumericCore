# Phase 34: production MILP strengthening

The branch-and-bound solver now performs three complementary reductions:

- deterministic minimal-cover cuts for eligible nonnegative binary knapsacks;
- iterative implied-bound propagation over all linear rows, including inward
  integer rounding and contradiction detection; and
- rounding-and-repair, which fixes rounded integer values and re-solves the
  continuous LP before accepting a candidate incumbent.

Cover inequalities are globally valid, so they strengthen every node without
changing the feasible integer set. Propagation can prune a node before an LP
solve. Repaired candidates pass the same independent row, bound, and
integrality diagnostics as warm starts and ordinary incumbents.

The public and FFI option shapes remain compatible. Existing search counters
record tightened bounds, extra repair relaxations, heuristic attempts, and
successful incumbents. Tests separately cover multi-variable propagation,
continuous repair, and minimal-cover construction.
