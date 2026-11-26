# Realistic LP and MILP benchmark panel

The panel tracks changes in what the native NumericCore solvers can solve on
published models. It runs independently of the offline numerical validation
suite. The GitHub Actions workflow **Realistic LP and MILP benchmarks** can
be run on demand and uploads a JSON Lines report.

## Run locally

Use Python 3.10 or later, Cargo, curl, and a C compiler named `cc`. Install the
pinned MPS reader in a virtual environment, then run the driver from the
repository root.

```sh
python3 -m venv target/realistic-panel/venv
source target/realistic-panel/venv/bin/activate
python -m pip install -r scripts/realistic-panel/requirements.txt
python scripts/realistic-panel/run.py
```

The first run downloads the selected inputs and caches their original bytes
under `target/realistic-panel/inputs`. The script checks the source and
decoded MPS SHA-256 hashes and validates dimensions before solving. HiGHS is
used only for MPS import. Netlib's checksum-pinned `emps.c` decodes the four
Netlib LPs. After a complete first run, use `--offline` to use only cached
inputs and Cargo crates. Use repeated `--case` options to select instances,
for example `--case afiro --case p0033`. Each selected case retains its
manifest budgets.

The default output is `target/realistic-panel/results.jsonl`. Use
`--output PATH` to keep a named result for later comparison and `--cache DIR`
to reuse the corpus across checkouts.

## Fixed inputs and budgets

[Panel v1](../scripts/realistic-panel/panel.json) defines nine inputs and
18 runs. It records source URLs, hashes, dimensions, available published
objectives and their absolute comparison tolerances.

| Runs | Cases | Limits |
|---|---|---|
| Revised simplex | afiro, boeing2, ship04l, ship12l, qap15 | 20 s each |
| LP with production scaling and presolve | afiro, boeing2, ship04l, ship12l | 20 s each |
| Revised simplex follow-up | ship12l | 90 s |
| Branch-and-bound | p0033, markshare_4_0, neos17, air05 | 20 s and 300 nodes each |
| Larger MILP node budget | p0033, markshare_4_0 | 20 s and 10,000 nodes each |
| MILP root probe | neos17, air05 | 20 s and one node each |

Reference LP objectives come from [Netlib](https://netlib.org/lp/data/readme).
The two [ASU LP feasibility](https://plato.asu.edu/ftp/lpfeas.html) and
[optimal basic solution](https://plato.asu.edu/ftp/lpopt.html) lists both
include `qap15`. The [MIPLIB 3.0 catalog](https://raw.githubusercontent.com/coin-or-tools/Data-miplib3/master/miplib3.cat)
supplies the published optimum for `p0033`. MIPLIB 2017 supplies the reference
values for [markshare_4_0](https://miplib.zib.de/instance_details_markshare_4_0.html),
[neos17](https://miplib.zib.de/instance_details_neos17.html) and
[air05](https://miplib.zib.de/instance_details_air05.html). These last three
are listed in the [ASU MILP results](https://plato.asu.edu/ftp/milp.html).
The panel uses the original MIPLIB inputs. ASU runs use their own
preprocessing, hardware and budget.

## Compare reports

Compare runs by the manifest hash and run ID. The header records the Rust
revision, dirty working-tree flag, toolchain, dependency-lock hash and
platform. For timing comparisons use repeated runs on the same machine.
The wall cap covers the whole solver process, including reading the prepared
input and auditing. `solver_seconds` measures the solver path through its
return.

- `outcome` distinguishes a returned solver report, a wall limit and a
  runner error. A wall limit has no final solver report and implies no
  feasibility or optimality result. Execution and input failures cause a
  nonzero exit. A time or node limit is a benchmark observation.
- `result.primal` reports original-model objective and violations for a
  returned optimum or MILP incumbent, checked at absolute tolerance
  `1e-7`. `reference_matches` measures objective agreement independently
  of the feasibility check.
- `dual_bound` and `dual_gap` on the revised-simplex path mean the
  original-model witness passed the dual verifier. Null means no witness
  passed. The scaled LP path reports primal evidence. Both LP paths solve
  for an optimum, with no separate ASU feasibility-only run.
- MILP records include node and relaxation counts, incumbent and heuristic
  counts, and the solver-reported bound and gap. These bounds have no
  independent search replay in this panel. A one-node run includes work
  before and within the first node, including any root heuristics.

Start with original-model feasibility, dual evidence, reference agreement,
and the status at each fixed budget. Compare times and search counts after
checking that the reported outcomes have comparable quality. The panel
exercises the common Rust backend through its native interface. Swift and
NumericCoreAMPL are outside the measured path.

This command extracts the numerical outcomes first, leaving the rest of the
report available for diagnosis.

```sh
jq -c 'select(.type == "run") |
  {id, outcome, status: .result.status, objective: .result.primal.objective,
   primal_verified: .result.primal.verified, reference_matches,
   dual_gap: .result.dual_gap, milp_gap: .result.reported_gap}' \
  target/realistic-panel/results.jsonl
```
