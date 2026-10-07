# Phase 11: convex quadratic programming

Phase 11 adds the solver-independent convex QP form

```text
minimize    0.5 x' Q x + c' x + constant
subject to  row_lower <= A x <= row_upper
            var_lower <= x   <= var_upper
```

`Q` is dense and validated as finite, symmetric, and positive semidefinite;
`A` uses the existing canonical CSR representation. Equality, inequality,
range, fixed-variable, and free bounds all use the established `Bound` type.

The initial solver is an operator-splitting ADMM method. Linear and variable
bounds are combined into one projection, while the constant positive-definite
system `Q + rho C'C` is factored once and reused. The solver accepts primal and
dual warm starts and supports cooperative cancellation.

Results report the verified original objective, row activities, row and
variable dual estimates, primal and dual ADMM residuals, KKT stationarity, and
independently recomputed row and variable-bound violations. Termination
distinguishes convergence, iteration limits, numerical failure, and
cancellation.

This is the reusable continuous-QP foundation for constrained regression,
control, and future SQP subproblems. It does not yet include adaptive `rho`,
scaling, polishing, sparse factorization, or primal/dual infeasibility
certificates. Those should precede claims of large-scale production QP parity
with specialized external solvers.
