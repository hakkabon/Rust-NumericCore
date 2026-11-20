# Phase 63 — Integer and logic-based Benders

`solve_integer_benders` accepts the Phase 62 matrix representation with a
mixed-integer first stage and continuous or mixed-integer recourse. Master
and recourse integer variables need finite bounds within the exactly
representable integer range. LP recourse can link to any master variable.
Integer recourse links through binary master columns so its value and
feasibility can be certified for a discrete assignment.

The master uses one branch-and-bound search. LP duals and Farkas
certificates audit node bounds and infeasibility. Only integer candidates
trigger lazy recourse evaluation. Globally valid cuts are appended to the
same master workspace and apply to all remaining nodes. An incumbent must
have a feasible plan for every scenario at the same first-stage point.

Continuous recourse supplies classical dual or feasibility cuts. For
integer recourse, a separate audited search establishes the assignment
lower bound and feasible cost used for the cut and incumbent respectively.
Binary no-good cuts reject infeasible assignments. For an assignment with
proven value bound q and global scenario floor L, the optimality cut uses
`theta ≥ q - (q - L) d`, where d is the Hamming distance to the
assignment. Exact assignments are cached for each discrete scenario.

## Oracle contract

`solve_logic_benders` allows model and logic scenarios in the same
problem. Logic scenarios declare an ordered binary projection and a
caller-proven lower bound. A synchronous oracle receives the scenario ID
and that bit vector. It may return an attained optimum and a validity
scope, an infeasibility conflict, or `Unresolved`. Scope positions refer
to the supplied vector. `None` means the entire vector. An empty scope
asserts the outcome's bound or infeasibility across all assignments.

For an optimal response, the reported cost must be attained at the
evaluated assignment and lower-bound the recourse value for every
assignment matching the validity scope. For a conflict, all assignments
matching the conflict scope must be infeasible. NumericCore checks input
shape and numerical consistency. The result marks oracle-based runs
because their guarantees rely on these application-supplied assertions.

Node and candidate limits, unresolved oracles, and absent numerical
evidence preserve available bounds without asserting convergence.
Unbounded master relaxations have a separate status. The node budget
counts relaxation visits, including revisits after lazy cut insertion.
The Phase 63 bridge ships with the v0.38.0 Rust artifacts.
