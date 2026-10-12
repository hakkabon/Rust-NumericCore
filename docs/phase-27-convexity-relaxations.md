# Phase 27: convexity analysis and global nonlinear relaxations

Phase 27 adds conservative structural reasoning to the shared nonlinear
expression graph. `analyze_expression` propagates interval bounds and
curvature (`constant`, `affine`, `convex`, `concave`, or `unknown`) over a
supplied parameter box. Unsupported composition rules fail closed as
`unknown`; they are never guessed from point samples.

`relax_expression` produces a solver-independent affine graph relaxation.
It contains exact rows for affine atoms, McCormick envelopes for bounded
products, and tangent/secant envelopes for supported convex or concave unary
atoms. Every node also carries an interval. `complete == false` means the
artifact is still a valid relaxation, but one or more nodes are represented
only by intervals.

`certify_convex_model` checks minimization objectives and constraint
directions. Upper-bounded expressions must be convex, lower-bounded
expressions concave, and ranged/equality expressions therefore affine. MINLP
search now reports a global certificate only after exhaustive search, with an
incumbent, no failed continuous relaxation, and a successful convexity
certificate. Gap-limit and node-limit exits remain explicitly local.

This phase does not yet solve the affine graph relaxation as a standalone LP
at every spatial node. That is the next step toward a full spatial
branch-and-bound solver for nonconvex continuous variables.
