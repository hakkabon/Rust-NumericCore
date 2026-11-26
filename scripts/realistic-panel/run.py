#!/usr/bin/env python3
"""Run the versioned LP/MILP panel against the current NumericCore checkout."""

import argparse
import bz2
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("panel.json")
EXAMPLE = "realistic_benchmark"
READER_VERSION = "1.16.0"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_digest(data, expected, label):
    actual = sha256(data)
    if actual != expected:
        raise ValueError(f"{label}: expected SHA-256 {expected}, got {actual}")


def source(url, digest, cache, offline):
    path = cache / digest
    if path.exists():
        data = path.read_bytes()
        verify_digest(data, digest, path)
        return data
    if offline:
        raise FileNotFoundError(f"{url}: missing from the offline cache {cache}")
    data = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                                    "--max-time", "60", url])
    verify_digest(data, digest, url)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
        temporary.write(data)
    os.replace(temporary.name, path)
    return data


def read_mps(reader_module, path, case):
    h = reader_module.Highs()
    h.silent()
    if h.readModel(str(path)) != reader_module.HighsStatus.kOk:
        raise ValueError(f"{path}: the MPS reader did not report success")
    if h.getModel().hessian_.dim_:
        raise ValueError(f"{path}: expected a linear objective")
    lp = h.getLp()
    matrix = lp.a_matrix_
    if matrix.format_ != reader_module.MatrixFormat.kColwise:
        raise ValueError(f"{path}: expected column storage")
    continuous = reader_module.HighsVarType.kContinuous
    integer = reader_module.HighsVarType.kInteger
    if any(kind not in (continuous, integer) for kind in lp.integrality_):
        raise ValueError(f"{path}: unsupported variable type")
    integers = [kind == integer for kind in lp.integrality_] or [False] * lp.num_col_
    for field, actual in {
        "rows": lp.num_row_,
        "columns": lp.num_col_,
        "nonzeros": len(matrix.value_),
        "integer_variables": sum(integers),
    }.items():
        if actual != case[field]:
            raise ValueError(f"{path}: expected {case[field]} {field}, got {actual}")

    rows = [[] for _ in range(lp.num_row_)]
    starts, indices, values = matrix.start_, matrix.index_, matrix.value_
    for column in range(lp.num_col_):
        for position in range(starts[column], starts[column + 1]):
            rows[indices[position]].append((column, float(values[position])))
    row_start, columns, coefficients = [0], [], []
    for row in rows:
        for column, coefficient in row:
            columns.append(column)
            coefficients.append(coefficient)
        row_start.append(len(columns))

    def bounds(lower, upper):
        return [[None if lo == -float("inf") else float(lo),
                 None if hi == float("inf") else float(hi)]
                for lo, hi in zip(lower, upper, strict=True)]

    sense = 1 if lp.sense_ == reader_module.ObjSense.kMinimize else -1
    return {
        "cost": [sense * float(cost) for cost in lp.col_cost_],
        "row_start": row_start,
        "columns": columns,
        "coefficients": coefficients,
        "rows": bounds(lp.row_lower_, lp.row_upper_),
        "bounds": bounds(lp.col_lower_, lp.col_upper_),
        "integer": integers,
        "offset": float(lp.offset_),
        "sense": sense,
    }


def prepare_models(cases, decoder, directory, cache, offline):
    import highspy

    if version("highspy") != READER_VERSION:
        raise ValueError(f"expected highspy=={READER_VERSION}")
    if any(case["compression"] == "netlib" for case in cases):
        decoder_source = directory / "emps.c"
        decoder_source.write_bytes(source(decoder["url"], decoder["sha256"], cache, offline))
        decoder_bin = directory / "emps"
        subprocess.run(["cc", "-O2", "-o", decoder_bin, decoder_source], check=True)

    for case in cases:
        data = source(case["url"], case["source_sha256"], cache, offline)
        compression = case["compression"]
        if compression == "netlib":
            data = subprocess.run([decoder_bin], input=data, stdout=subprocess.PIPE, check=True).stdout
        elif compression == "gzip":
            data = gzip.decompress(data)
        elif compression == "bzip2":
            data = bz2.decompress(data)
        else:
            raise ValueError(f"unsupported compression {compression}")
        mps = directory / f'{case["id"]}.mps'
        verify_digest(data, case["mps_sha256"], mps)
        mps.write_bytes(data)
        model = read_mps(highspy, mps, case)
        (directory / f'{case["id"]}.json').write_text(json.dumps(model, allow_nan=False))


def build_runner(offline):
    command = ["cargo", "build", "--release", "-p", "nc-optimize", "--example", EXAMPLE,
               "--message-format=json-render-diagnostics"]
    if offline:
        command.append("--offline")
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, text=True, check=True)
    for line in completed.stdout.splitlines():
        message = json.loads(line)
        if message["reason"] == "compiler-artifact" and message["target"]["name"] == EXAMPLE:
            return message["executable"]
    raise RuntimeError("Cargo did not report the benchmark executable")


def run_case(executable, model, case, run):
    record = {"type": "run", **run, "input": case}
    command = [executable, model, run["mode"]]
    if "node_limit" in run:
        command.append(str(run["node_limit"]))
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=run["wall_seconds"]
        )
        if completed.returncode:
            record.update(outcome="runner_error", return_code=completed.returncode,
                          error=completed.stderr.strip())
        else:
            record.update(json.loads(completed.stdout), outcome="returned")
            primal, reference = record["result"]["primal"], case["reference"]
            if primal and reference and primal["objective"] is not None:
                error = abs(primal["objective"] - reference["objective"])
                record.update(reference_absolute_error=error,
                              reference_matches=error <= reference["absolute_tolerance"])
    except subprocess.TimeoutExpired:
        record["outcome"] = "wall_limit"
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def command_output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="run every budget for this case (repeatable)")
    parser.add_argument("--offline", action="store_true", help="use only cached inputs and Cargo crates")
    parser.add_argument("--cache", type=Path, default=ROOT / "target/realistic-panel/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "target/realistic-panel/results.jsonl")
    args = parser.parse_args()

    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = {case["id"]: case for case in manifest["cases"]}
    selected = set(args.case or cases)
    unknown = selected - cases.keys()
    if unknown:
        parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "Cargo.lock"
    execution_error = False
    with tempfile.TemporaryDirectory(prefix="numericcore-panel-") as temporary, args.output.open("w") as output:
        directory = Path(temporary)

        def emit(record):
            print(json.dumps(record, allow_nan=False), file=output, flush=True)

        try:
            prepare_models([case for key, case in cases.items() if key in selected],
                           manifest["netlib_decoder"], directory, args.cache, args.offline)
            executable = build_runner(args.offline)
            emit({
                "type": "panel",
                "version": manifest["version"],
                "manifest_sha256": sha256(manifest_bytes),
                "revision": command_output("git", "rev-parse", "HEAD"),
                "dirty": bool(command_output("git", "status", "--porcelain")),
                "rustc": command_output("rustc", "--version", "--verbose"),
                "cargo_lock_sha256": sha256(lock.read_bytes()),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "mps_reader": f"highspy=={READER_VERSION}",
            })
            for run in manifest["runs"]:
                key = run["case"]
                if key in selected:
                    record = run_case(executable, directory / f"{key}.json", cases[key], run)
                    emit(record)
                    print(f'{run["id"]}: {record["outcome"]}', file=sys.stderr)
                    execution_error |= record["outcome"] == "runner_error"
        except (OSError, RuntimeError, ValueError, ImportError, subprocess.CalledProcessError) as error:
            emit({"type": "panel_error", "error": str(error)})
            print(error, file=sys.stderr)
            return 1
    print(f"Results written to {args.output}", file=sys.stderr)
    return int(execution_error)


if __name__ == "__main__":
    sys.exit(main())
