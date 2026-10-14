# Phase 44: FEM foundation

Phase 44 introduces a finite-element vertical slice above NumericCore's sparse
assembly and factorization layers. The common representation includes planar
mesh nodes, degree-of-freedom constraints, nodal loads, material tables, and
element connectivity. Model validation rejects out-of-range connectivity,
duplicate or degenerate elements, invalid properties, duplicate constraints,
and non-finite data before assembly.

The first two formulations deliberately exercise different engineering fields:

- three-node constant-gradient triangles solve steady scalar diffusion/heat
  conduction with conductivity, thickness, nodal loads, and uniform element
  heat sources;
- two-node planar trusses solve small-displacement linear elasticity with
  Young's modulus and area, then recover element strain, stress, and axial force.

Local matrices are scattered into canonical CSR stiffness matrices. Prescribed
values are eliminated exactly from the reduced right-hand side instead of being
approximated with penalty stiffness. Sparse Cholesky solves the reduced
positive-definite system, after which the original system recovers reactions,
strain energy, maximum free-DOF imbalance, and the reduced relative residual.

This phase is intentionally static and linear. Mesh import/generation, sparse
assembly pattern reuse, quadrature abstractions, continuum elasticity, mass and
damping matrices, transient structural/thermal analysis, nonlinear materials,
contact, adaptivity, and FEM design optimization remain later work.
