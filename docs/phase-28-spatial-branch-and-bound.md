# Phase 28: spatial branch-and-bound

Phase 28 turns the Phase 27 graph relaxations into operational global bounds
for bounded factorable nonlinear programs.

Each search node recompiles interval bounds and affine graph envelopes into the
existing solver-independent LP representation. Shared parameter variables are
linked to each expression graph, objective and constraint output nodes are
connected to their model bounds, and revised simplex supplies a certified node
lower bound. A bounded L-BFGS-B or SQP solve started from the LP point supplies
a feasible incumbent. Best-bound node selection, objective-bound pruning, and
widest-domain bisection then close an absolute or relative global gap.

The result separates relaxation bounds, local incumbents, global gaps, pruning
counts, spatial resolution, node limits, cancellation, and certification.
Unsupported atoms, infinite boxes, non-finite node intervals, and incomplete
relaxations fail closed before search. The UniFFI surface transports the same
problem, option, result, and termination concepts for the next generated Swift
binding update.

The current phase is continuous NLP spatial search. Joint integer/spatial
branching, contractors, optimization-based bound tightening, and piecewise
envelope refinement remain subsequent work.
