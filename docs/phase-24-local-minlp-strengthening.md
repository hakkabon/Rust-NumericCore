# Phase 24: local MINLP strengthening

Phase 24 improves the existing nonconvex local MINLP branch-and-bound without
turning local NLP relaxation values into global certificates.

Fractional relaxations can now trigger a rounding-and-polishing heuristic. The
integer variables are rounded and fixed, then the continuous variables are
reoptimized with the configured SQP, augmented-Lagrangian, or bounded L-BFGS
relaxation. A validated integer-feasible warm incumbent may also seed search.

Node processing is selectable between depth-first search and best local bound.
The latter is a search-order heuristic only: nonlinear relaxation objectives
are not used for certified pruning. Results report heuristic attempts,
heuristic successes, and whether the supplied warm incumbent was accepted.
`global_optimality_certified` therefore remains false.
