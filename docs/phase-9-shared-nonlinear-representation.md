# Phase 9: shared nonlinear representation

Phase 9 introduces a solver-independent representation for smooth nonlinear
objectives and residual models in `nc-optimize`.

`NonlinearExpression` is a flat, topologically ordered graph of constants,
parameters, arithmetic operations, elementary functions, and constant powers.
Evaluation produces a value and exact first derivative. A vector of expressions
therefore produces residuals and a row-major Jacobian without requiring a
second derivative callback contract.

`NonlinearModel` combines the parameter count, parameter bounds, and exactly
one model kind: a scalar objective or a residual vector. It validates graph
references, parameter indices, bounds, dimensions, constants, and exponents
before evaluation. Domain failures such as invalid logarithms or division by
zero fail closed as non-finite evaluations.

The representation feeds the existing solvers through:

- `minimize_model_lbfgs` for free objective models;
- `minimize_model_lbfgsb` for bounded objective models; and
- `solve_model_least_squares` for weighted and robust residual models.

Closure-based APIs remain available for models implemented directly in code.
The graph is the portable seam intended for future modeling-language, FFI, and
automatic-differentiation work. Swift mirrors the same node vocabulary, graph
invariants, derivative semantics, model distinction, and solver adapters.
