# Phase 46: model-based derivative-free optimization

`minimize_model_based` and its observer variant build a local diagonal
quadratic interpolation model using objective values only. Unequal centered
stencils support interior points, while two-sample one-sided stencils preserve
model geometry at active box bounds.

A bound-feasible model step is restricted by a Euclidean trust region. The
ratio of actual to predicted decrease controls acceptance, contraction, and
expansion. Results expose evaluation and model counts, accepted steps,
one-sided stencil use, final radius, projected-gradient norm, reduction ratio,
and explicit convergence, budget, iteration, or cancellation termination.

The implementation is BOBYQA-inspired but is not represented as BOBYQA. It
does not yet maintain a full quadratic interpolation set or use
minimum-Frobenius-change Hessian updates. Those are natural later extensions
once the present model geometry and trust-region contracts have accumulated
broader validation evidence.
