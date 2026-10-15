# Phase 56 — Continuum FEM

Phase 56 extends FEM from trusses and scalar diffusion to two-dimensional
small-strain linear elasticity. `ContinuumFemModel` provides three-node
constant-strain triangles with two displacement DOFs per node and reuses sparse
assembly, exact prescribed-DOF elimination, reaction recovery, and Cholesky.

`LinearElasticContinuumMaterial` supports validated isotropic plane-stress and
plane-strain laws with thickness. Elements assemble consistent constant body
forces. Signed element orientation is retained in the strain-displacement
matrix while integration uses positive area, so clockwise and counter-clockwise
connectivity recover the same physical strain.

Element recovery reports engineering strain, in-plane and out-of-plane stress,
in-plane principal stresses, three-dimensional von Mises stress, area, and
strain energy. Recovered element energy is checked against global assembled
energy.

An analytic uniaxial plane-stress patch test verifies displacement, Poisson
contraction, stress, equilibrium, and energy. A plane-strain body-force case
exercises out-of-plane stress recovery. The patch test also runs in the
standard reproducibility corpus in Rust and Swift.

This phase remains linear and triangular. Quadrilateral and higher-order
elements, quadrature families, 3D solids, continuum dynamics, thermal strain,
nonlinear materials, geometric nonlinearity, contact, mesh import, error
estimators, and adaptive refinement remain later work.
