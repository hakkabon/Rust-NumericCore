# Phase 15: nonlinear AMPL integration

The Swift AMPL front end now lowers nonlinear scalar syntax directly into
Rust-NumericCore's shared nonlinear graph. No parser or separate nonlinear AST
is added on the Rust side: Phase 12's value-only UniFFI transport remains the
language boundary, and Phase 14's SQP plus the bounded L-BFGS and
augmented-Lagrangian solvers execute the compiled model unchanged.

The supported source operations map one-to-one to `NonlinearNode`: arithmetic,
constant powers, `exp`, `log`, `sqrt`, `sin`, and `cos`. Constrained source
models become `ConstrainedNonlinearProblem` values and can execute through the
Rust SQP export. Unconstrained models use the existing bounded objective solve.

This phase intentionally requires no new Rust ABI. Its cross-codebase value is
the completed vertical path from AMPL text through the shared graph and UniFFI
to the Rust nonlinear solvers. Certified global MINLP, indexed modeling constructs, nonsmooth
operators, and user-defined functions remain future work.
