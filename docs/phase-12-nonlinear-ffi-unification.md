# Phase 12: nonlinear FFI unification

Phase 12 promotes the shared nonlinear expression graph to the stable UniFFI
boundary. Swift sends a value-only `FfiNonlinearModel`; callbacks never cross
the language boundary, so all graph evaluation and differentiation stays in
Rust during a solve.

The transport supports every shared node operation, parameter bounds, bounded
L-BFGS, weighted and robust nonlinear least squares, and nonlinear constraints.
Dedicated option, result, termination, multiplier, and diagnostic records keep
the public contract explicit and prevent convergence failures from being
collapsed into transport errors.

Malformed graphs, invalid dimensions, invalid bounds, and invalid solver
configuration are validated by `nc-optimize` and surface as `FfiError` rather
than panics. Integer iteration fields are checked when converting from the
portable `u64` ABI to the host's `usize`.

This is a one-shot solve interface. Observer callbacks and cooperative
cancellation are intentionally not part of the FFI contract; adding them later
should use an opaque solve/session object rather than synchronous foreign
callbacks in numerical hot loops.
