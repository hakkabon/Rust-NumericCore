# Phase 58 — MILP/global-NLP selective strengthening

Phase 58 adds a selective integer-hull cut family to factorable global MINLP.
For a direct square atom `z = x^2` whose input is an integer parameter, each
adjacent integer pair in the current node domain contributes the valid secant

`z >= (2k + 1)x - k(k + 1)`.

These inequalities dominate the continuous square epigraph between integers
and can close the root relaxation gap without branching. They are added only
after propagation and integer-domain normalization, preserve the global lower
bound, and therefore preserve the solver's optimality certificate.

Generation is enabled by default, can be disabled, and is capped per node.
`GlobalMinlpResult` reports the cumulative number generated. General
Gomory/MIR cuts, RLT products, affine square operands, higher integer powers,
cut aging, and persistent cut pools remain deliberately outside this phase.
