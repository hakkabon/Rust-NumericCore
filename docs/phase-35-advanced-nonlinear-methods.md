# Phase 35: advanced nonlinear methods

`nc-optimize` now provides two complementary unconstrained methods:

- `minimize_newton_cg` uses gradient and Hessian-vector callbacks. Truncated
  Steihaug CG handles negative curvature and trust-region boundary crossings;
  actual/predicted reduction controls acceptance and radius updates.
- `minimize_powell` uses objective values only. Bracketed golden-section line
  searches update a conjugate direction set using the displacement of each
  completed cycle.

Both return structured iteration, evaluation, termination, and method-specific
work counts and validate every callback result. The APIs mirror the native
Swift implementations without expanding UniFFI.

The derivative-free implementation follows M. J. D. Powell's classical
direction-set method. It does not claim compatibility with his later NEWUOA,
BOBYQA, or COBYLA interpolation methods; those require distinct model geometry,
constraint handling, and trust-region maintenance.
