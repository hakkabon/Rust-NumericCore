# Phase 30: global MINLP strengthening

Phase 30 adds a distinct global MINLP solver while preserving the existing
local MINLP solver and its semantics. Every search node runs nonlinear
propagation, normalizes integer domains, and solves the graph LP relaxation to
obtain a rigorous lower bound.

Fractional integer variables are branched first. Once the LP point is integer,
the search continues with spatial subdivision wherever graph-envelope gaps
remain, including integer-domain partitioning when necessary. Rounded integer
assignments are fixed before L-BFGS-B or SQP polishing, so every incumbent is
integral by construction. Objective bounds, infeasibility, and propagation all
prune nodes.

Results separately report integer and spatial branches, propagation work,
heuristic activity, LP relaxations, global bounds, gaps, and certification.
The UniFFI endpoint transports the complete problem, controls, result metrics,
and termination state. Node limits, unresolved spatial resolution, relaxation
failures, and cancellation never claim global optimality.

The method currently requires finite parameter boxes and complete factorable
graph relaxations. Strong branching, pseudocosts, cut generation, and adaptive
piecewise envelopes remain future strengthening opportunities.
