"""Stage 4 experiment batch runner (T-4) for ``_logical_reduce_kernel``.

Per experiment branch, in order:
  1. run the L0 precision regression (``python <branch> --level L0``);
  2. measure every requested tune workload with ``msprof op`` (kernel-only
     ``Task Duration(us)``, mean over ``LAUNCH_COUNT`` launches -- the single
     latency metric of Stage 4);
  3. append one line per measured workload to ``perf_records.jsonl``
     (append-only; field contract: _shared/standards/signal-registry.md #5);
  4. print the candidate-vs-current-best summary table.

The agent only edits ``EXPERIMENTS`` / CLI flags and reads the table; the
decision (winner / rollback / next candidate) stays with the agent.

Usage (inside perf_opt/):
  python run_experiments.py --round 1 --phase candidate --all
  python run_experiments.py --round 1 --phase candidate --only op2 --workloads sparsity-seq-float16
  python run_experiments.py --round 0 --phase baseline --all
  python run_experiments.py --round 3 --phase final --all
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import measure  # noqa: E402

KERNEL_ID = "_logical_reduce_kernel::main"
KERNEL_NAME = "main"
LAUNCH_COUNT = 30
WARM_UP = 5
TIMEOUT_S = 1800
CURRENT_BEST = "_logical_reduce_kernel_lab.py"  # round base (current best file)
CURRENT_BEST_ID = "baseline"  # candidate_id of the current merged best
PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"

# Round 1 branches: two structural levers, each isolated.
EXPERIMENTS = [
    {
        "file": "../_logical_reduce_kernel.py",
        "opt_id": "baseline",
        "parent_id": None,
        "knobs": {},
        "change": "Stage 3 baseline (read-only reference)",
    },
    {
        "file": "_logical_reduce_kernel_round1_op1_e8192_rprime0.py",
        "opt_id": "r1_op1_batchout",
        "parent_id": "baseline",
        "knobs": {"E_CAP": 8192, "RPRIME": 0},
        "change": "batched output DMA + per-core contiguous row chunk",
    },
    {
        "file": "_logical_reduce_kernel_round1_op2_e8192_rprime1.py",
        "opt_id": "r1_op2_rprime",
        "parent_id": "r1_op1_batchout",
        "knobs": {"E_CAP": 8192, "RPRIME": 1},
        "change": "single-tile resident R' for 8192 < N <= UB cap (DESIGN 1.6.4 A/B)",
    },
    {
        "file": "_logical_reduce_kernel_round1_op2b_rprime_refout.py",
        "opt_id": "r2_op2b_rprime_refout",
        "parent_id": "baseline",
        "knobs": {"E_CAP": 8192, "RPRIME": 1, "OUT_MODE": 0},
        "change": "single-tile resident R' with the Stage 3 output structure (isolates R' from the batched output)",
    },
    {
        "file": "_logical_reduce_kernel.py",
        "opt_id": "final_rprime",
        "parent_id": "r2_op2b_rprime_refout",
        "knobs": {"E_CAP": 8192, "RPRIME": 1, "OUT_MODE": 0},
        "change": "final: Stage 3 structure + R' single-tile resident dispatch",
    },
    {
        "file": "_logical_reduce_kernel_round2_op5_pipe2.py",
        "opt_id": "r2_op5_pipe2",
        "parent_id": "baseline",
        "knobs": {"E_CAP": 8192, "RPRIME": 0, "OUT_MODE": 0, "PIPE": 2},
        "change": "T.Pipelined(num_stages=2) on the resident task loop (explicit pipeline vs auto multi-buffer)",
    },
]


def safe_component(value):
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_l0(branch, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"L0_{safe_component(os.path.basename(branch))}.log")
    proc = subprocess.run(
        [sys.executable, branch, "--level", "L0"],
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    with open(log, "w") as f:
        f.write(proc.stdout + proc.stderr)
    return proc.returncode == 0, log


def tune_workloads():
    with open(WORKLOAD_INVENTORY) as handle:
        inv = json.load(handle)
    return [w["workload_id"] for w in inv["workloads"] if w.get("kind") == "tune"]


def append_record(record):
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def best_so_far(workload_id):
    """Latest duration for this workload from the current merged best id."""
    if not os.path.isfile(PERF_RECORDS):
        return None
    best = None
    with open(PERF_RECORDS) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("kernel_id") == KERNEL_ID
                and row.get("workload_id") == workload_id
                and row.get("candidate_id") == CURRENT_BEST_ID
            ):
                best = row["duration_us"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--phase", default="candidate",
                    choices=("baseline", "candidate", "merged", "final"))
    ap.add_argument("--only", help="single opt_id")
    ap.add_argument("--workloads", help="comma separated workload ids")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-l0", action="store_true")
    ap.add_argument("--launch-count", type=int, default=LAUNCH_COUNT)
    args = ap.parse_args()

    all_tune = tune_workloads()
    if args.all:
        selected = all_tune
    elif args.workloads:
        selected = [w for w in args.workloads.split(",") if w]
    else:
        ap.error("pass --all or --workloads")
    unknown = [w for w in selected if w not in all_tune]
    if unknown:
        ap.error(f"unknown workload ids: {unknown}")

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if not todo:
        ap.error("no matching experiments")

    rows = []
    for exp in todo:
        branch = exp["file"]
        print(f"=== {exp['opt_id']}: {branch} ===")
        if not os.path.isfile(branch):
            print("  MISSING FILE")
            rows.append({**exp, "status": "missing_file"})
            continue
        l0_pass = True
        if not args.skip_l0:
            l0_pass, l0_log = run_l0(branch, os.path.join("logs", f"round{args.round}"))
            print(f"  L0: {'PASS' if l0_pass else 'FAIL'} ({l0_log})")
        if not l0_pass:
            rows.append({**exp, "status": "l0_fail", "l0_pass": False})
            continue
        for wl in selected:
            out_dir = os.path.join(
                "profiles", args.phase, f"round{args.round}",
                safe_component(exp["opt_id"]), safe_component(wl),
            )
            log = os.path.join(
                "logs", f"round{args.round}",
                f"{safe_component(exp['opt_id'])}_{safe_component(wl)}.log",
            )
            res = measure(
                os.path.join(HERE, branch), wl, out_dir,
                kernel_name=KERNEL_NAME,
                launch_count=args.launch_count,
                warm_up=WARM_UP,
                log_path=log,
            )
            if not res.get("ok"):
                print(f"  {wl}: msprof FAILED")
                rows.append({**exp, "workload_id": wl, "status": "msprof_fail",
                             "l0_pass": True})
                continue
            op = res["ops"]["main"]
            append_record({
                "round": args.round,
                "candidate_id": exp["opt_id"],
                "parent_id": exp.get("parent_id"),
                "kernel_id": KERNEL_ID,
                "workload_id": wl,
                "phase": args.phase,
                "artifact_path": os.path.abspath(branch),
                "artifact_sha256": file_sha256(branch),
                "duration_us": round(op["mean_us"], 4),
                "l0_pass": True,
                "msprof_raw_path": os.path.abspath(out_dir),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            })
            rows.append({**exp, "workload_id": wl, "status": "ok", "l0_pass": True,
                         "duration_us": round(op["mean_us"], 4)})
            print(f"  {wl}: {op['mean_us']:.3f} us (median {op['median_us']:.3f}, "
                  f"min {op['min_us']:.3f})")

    print("\n| branch | workload_id | l0 | Task Duration(us) | vs current best | status |")
    print("|---|---|---|---:|---:|---|")
    for r in rows:
        dur = r.get("duration_us")
        base = best_so_far(r.get("workload_id")) if dur is not None else None
        rel = f"{(base - dur) / base * 100:+.1f}%" if (dur and base) else "-"
        print(f"| {r['opt_id']} | {r.get('workload_id', '-')} | {r.get('l0_pass', '-')} "
              f"| {dur if dur is not None else '-'} | {rel} | {r['status']} |")
    print(f"\n(current merged best candidate id: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
