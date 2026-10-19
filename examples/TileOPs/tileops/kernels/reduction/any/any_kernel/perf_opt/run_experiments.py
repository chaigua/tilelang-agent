"""Stage 4 experiment batch runner (T-4) for the AnyFwdOp ``_logical_reduce_kernel``.

Per experiment branch, in order:
  1. run the L0 precision regression (``python <branch> --level L0``);
  2. measure every requested tune workload with ``msprof op`` (kernel-only
     ``Task Duration(us)``, mean over ``LAUNCH_COUNT`` launches -- the single
     latency metric of Stage 4);
  3. append one line per measured workload to ``perf_records.jsonl``
     (append-only; field contract: _shared/standards/signal-registry.md #5);
  4. print the candidate-vs-current-best summary table.

Usage (inside perf_opt/):
  python run_experiments.py --round 0 --phase baseline \
      --opt-id baseline --impl ../_logical_reduce_kernel.py
  python run_experiments.py --round 1 --phase candidate \
      --opt-id r1_op1 --parent-id baseline --impl _logical_reduce_kernel_round1_op1.py --all
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import measure  # noqa: E402

KERNEL_ID = "_logical_reduce_kernel::main"
KERNEL_NAME = "main"
LAUNCH_COUNT = 30
WARM_UP = 5
TIMEOUT_S = 2400
CURRENT_BEST_ID = "baseline"  # candidate_id of the current merged best
PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"


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
    env = dict(os.environ)
    from run_bench import REPO_ROOT, TILEOPS_DIR

    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, TILEOPS_DIR, os.path.dirname(HERE), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, branch, "--level", "L0"],
        capture_output=True, text=True, timeout=TIMEOUT_S, env=env, cwd=HERE,
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
    ap.add_argument("--opt-id", required=True, help="candidate_id for the measured file")
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--impl", required=True, help="kernel file relative to perf_opt/ or absolute")
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

    branch = args.impl
    print(f"=== {args.opt_id}: {branch} ===", flush=True)
    if not os.path.isfile(branch):
        print("  MISSING FILE")
        return 1
    l0_pass = True
    if not args.skip_l0:
        l0_pass, l0_log = run_l0(branch, os.path.join("logs", f"round{args.round}"))
        print(f"  L0: {'PASS' if l0_pass else 'FAIL'} ({l0_log})", flush=True)
    if not l0_pass:
        print("  L0 failed -> no perf record for this branch")
        return 1

    rows = []
    for wl in selected:
        out_dir = os.path.join(
            "profiles", args.phase, f"round{args.round}",
            safe_component(args.opt_id), safe_component(wl),
        )
        log = os.path.join(
            "logs", f"round{args.round}",
            f"{safe_component(args.opt_id)}_{safe_component(wl)}.log",
        )
        res = measure(
            os.path.join(HERE, branch) if not os.path.isabs(branch) else branch,
            wl, out_dir,
            kernel_name=KERNEL_NAME,
            launch_count=args.launch_count,
            warm_up=WARM_UP,
            log_path=log,
        )
        if not res.get("ok"):
            print(f"  {wl}: msprof FAILED", flush=True)
            rows.append({"opt_id": args.opt_id, "workload_id": wl, "status": "msprof_fail"})
            continue
        op = res["ops"]["main"]
        append_record({
            "round": args.round,
            "candidate_id": args.opt_id,
            "parent_id": args.parent_id,
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
        pipe = res.get("pipe", {}).get("main", {})
        rows.append({"opt_id": args.opt_id, "workload_id": wl, "status": "ok",
                     "duration_us": round(op["mean_us"], 4), "pipe": pipe,
                     "n": op["n"], "median": op["median_us"], "min": op["min_us"]})
        print(f"  {wl}: {op['mean_us']:.3f} us (median {op['median_us']:.3f}, "
              f"min {op['min_us']:.3f}, n={op['n']}) "
              f"vec={pipe.get('aiv_vec_ratio')} mte2={pipe.get('aiv_mte2_ratio')} "
              f"mte3={pipe.get('aiv_mte3_ratio')}", flush=True)

    print("\n| branch | workload_id | l0 | Task Duration(us) | vs current best | status |")
    print("|---|---|---|---:|---:|---|")
    for r in rows:
        dur = r.get("duration_us")
        base = best_so_far(r.get("workload_id")) if dur is not None else None
        rel = f"{(base - dur) / base * 100:+.1f}%" if (dur and base) else "-"
        print(f"| {r['opt_id']} | {r.get('workload_id', '-')} | True "
              f"| {dur if dur is not None else '-'} | {rel} | {r['status']} |")
    print(f"\n(current merged best candidate id: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
