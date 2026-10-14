# Phase 45: FEM dynamics and modal analysis

Phase 45 extends planar trusses with material density and the consistent
translational element mass matrix. Both stiffness and mass retain canonical CSR
storage and use the same validated topology and global degree-of-freedom map as
the static solver.

`TrussFemModel::modal_analysis` eliminates homogeneous constrained DOFs and
solves `K phi = lambda M phi`. A dense Cholesky congruence is currently used for
the reduced foundation-scale eigenproblem. Flexible modes are frequency sorted,
mass normalized, expanded to the full DOF space, and certified with relative
eigen-residual and mass-orthogonality diagnostics.

`TrussFemModel::dynamic_response` solves `M u'' + C u' + K u = f(t)` with the
unconditionally stable average-acceleration Newmark scheme. Rayleigh damping is
`C = alpha M + beta K`; the effective matrix is factored once for a fixed time
step. Results retain all time states, full displacement/velocity/acceleration,
kinetic and strain energy, assembled matrices, and a maximum reduced-equilibrium
residual.

The scope is linear planar-truss dynamics with stationary homogeneous supports.
Sparse Lanczos/subspace iteration, reusable sparse effective factorizations,
base excitation, continuum mass matrices, thermal transients, nonlinear
structural dynamics, and harmonic response are deferred.
