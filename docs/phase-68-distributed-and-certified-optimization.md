# Phase 68 — Distributed and certified optimization

The `distributed_benders` module implements multicut Benders for continuous,
two-stage linear problems. The master runs in the coordinator. Each scenario
worker receives its shifted recourse LP and returns a primal point with dual
row multipliers, a Farkas row multiplier vector, or an unresolved reply.
Recourse objectives need a finite lower bound computable from the local
variable bounds. Weights use the existing Benders probability convention.

`DistributedBendersCoordinator::pending_jobs` returns versioned jobs in
scenario order. Jobs are independent within a round and can be routed over
any transport. Responses can arrive out of order. The coordinator verifies
them against its own model and applies a complete batch in scenario order.
An identical retry is idempotent. A conflicting or foreign reply is rejected.
An unresolved reply leaves its job pending.

Build a stateless worker once per target host.

```sh
cargo build --release -p nc-optimize --bin nc-benders-worker
target/release/nc-benders-worker < jobs.jsonl > replies.jsonl
```

The worker reads one job object per line and emits one reply per line. A
malformed job produces a JSON error object. Transport, authentication,
process supervision, and retry policy belong to the caller.

Protocol version 1 jobs carry a `key` containing `run_id`, `round` and
`scenario`, along with the LP iteration limit, evidence tolerance, and the
shifted LP in CSR form. Round numbers start at one and scenario indices at
zero. The LP carries its objective and row and variable bounds. Bounds are
pairs `[lower, upper]` with `null` for a missing endpoint. A reply echoes
`version` and `key`. Its `evidence.kind` is `optimal` with `point` and
`row_multipliers`, `infeasible` with `row_multipliers`, or `unresolved`.
Positive row multipliers select lower bounds and negative multipliers select
upper bounds in NumericCore's minimization convention.

Export a checkpoint to persist completed evidence and any accepted replies
for the open round. Resume with the original model and the intended audit
tolerances. A completed-round limit can be raised on resume. The coordinator
rebuilds its master and rechecks the checkpoint before reissuing missing jobs.

Export a proof to audit the numerical result separately. `audit_benders_proof`
reconstructs the master from the original model, independently verifies every
LP witness, projects the cuts, and recomputes costs and bounds. It uses no LP
optimization. The verifier supplies tolerances, so the proof cannot declare
its own acceptable gap. The resulting claim is optimality within tolerance,
infeasibility from verified Farkas evidence, or the available bounds. The
certificate uses floating-point arithmetic and the existing numerical LP
verification contract.

The integration tests exercise real worker subprocesses, out-of-order
submissions, partial checkpoints, tampered evidence, round-limit recovery,
and infeasibility replay.
