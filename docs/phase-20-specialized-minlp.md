# Phase 20: specialized and mixed-integer nonlinear methods

Phase 20 adds mixed-integer nonlinear search over the shared expression graph.
Each search node tightens integer-variable bounds and solves a specialized
continuous relaxation using either SQP or the augmented-Lagrangian method.
Unconstrained nodes use bounded L-BFGS directly. Fractional variables are
selected by maximum fractionality, integer incumbents are snapped and
re-evaluated, and infeasible snapped candidates are rejected.

The result records node and relaxation counts, infeasibility pruning, maximum
depth, incumbent updates, the best observed local-relaxation objective, local
absolute and relative gaps, feasibility, stationarity, and multipliers. Node,
local-gap, cancellation, relaxation-failure, infeasibility, and exhausted-tree
terminations are distinct.

This is deliberately a local MINLP method. The current nonlinear graph does
not certify convexity, and SQP or augmented-Lagrangian relaxations can converge
to local solutions. Therefore `global_optimality_certified` is always false,
and reported gaps are explicitly local-relaxation diagnostics. Future global
certificates require convexity propagation or spatial branch-and-bound with
valid relaxations.
