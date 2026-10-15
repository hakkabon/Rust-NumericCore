# Phase 50 — Expanded external validation

The production validation report now includes offline fixtures from two
independent numerical corpora:

- NIST StRD Misra1a, checked against the certified regression parameters and
  residual sum of squares published by NIST.
- CUTEst HS71, checked against the official constrained problem definition,
  reported solution, objective, and feasibility.

Each external `ValidationCase` retains a `ValidationProvenance` record with its
collection, problem identifier, citation, and source URL. The fixtures are
embedded so routine tests are deterministic and do not depend on network
availability. `run_reference_validation` now combines seven internal cases
with the two external cases, while `run_external_reference_validation` permits
running just the externally certified subset.

This phase establishes an extensible provenance and release-gating contract;
it is not a claim of full Netlib, Maros–Mészáros, MIPLIB, CUTEst, or NIST StRD
coverage.
