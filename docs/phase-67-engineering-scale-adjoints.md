# Phase 67 — Engineering-scale adjoints

`EngineeringModel::adjoint_sensitivities` computes steady or single-step
response gradients with one reusable sparse factor and one transpose solve
per response. It contracts the sparse parameter Jacobian with the adjoint,
so the cost in linear solves scales with the number of responses. Direct
parameter dependence is included. The forward-and-adjoint API remains
available for conformance checks. Multiphysics design uses the adjoint-only
path for its objective gradient.

Element assembly uses reverse AD and reads only inputs referenced by each
expression graph. The sparse LU kernel tracks active rows by column during
elimination so pivot selection and elimination visit rows with a stored
entry in the current pivot column. Partial pivoting and permutation
semantics are preserved.

`EngineeringTrajectory::integrate` uses sparse backward Euler on a caller-
supplied, strictly increasing time grid. At step `k` it solves
`F(t_k, y_k, (y_k - y_{k-1}) / h_k, p) = 0`. One symbolic analysis is reused
while the numerical tangent pattern is unchanged. A retained trajectory
can evaluate several objectives in a single reverse pass. A response is a
terminal expression, a right-endpoint running quadrature, or both. Graph
inputs are time, all states, all derivatives, then all model parameters.

The reverse pass solves the transpose of the same discrete tangent and
propagates the previous-state contribution from the derivative Jacobian.
Response derivatives with respect to state derivatives contribute at both
ends of the step. The parameter gradient holds the initial state fixed.
For parameterized initialization, contract the separate initial state
pullback with the derivative of the initialization map and add it to the
parameter gradient. The solver does not differentiate an initialization solve.

The tape stores the initial and final states plus every `checkpoint_stride`-th
state. The reverse sweep replays one segment at a time. Stride one stores
every state and avoids replay. Larger strides shrink the retained tape
and increase replay and the maximum temporary segment size. Diagnostics
include the maximum primal residual, adjoint backward errors, replayed
steps, and transpose solves. Singular tangents and steps that miss the
configured residual or backward-error tolerance return errors.

These derivatives apply to the specified backward-Euler discretization.
Adaptive step selection, event time, discontinuous model switches, and
mesh-changing sensitivities are outside this contract. The validation tests
compare against an analytical recurrence and central differences for an
index-1 DAE, and check a sparse steady adjoint with 1,024 design parameters.
