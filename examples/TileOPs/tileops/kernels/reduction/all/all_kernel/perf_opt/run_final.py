"""Final measurement pass (Stage 4 Phase 3).

For every tune workload, in one adjacent-in-time pair:

  1. ``phase=baseline`` / ``candidate_id=baseline_recheck`` -- the read-only
     Stage 3 baseline re-measured in the *final* session (run-state drift
     control; the round-0 baseline numbers are kept untouched);
  2. ``phase=final`` / ``candidate_id=final_f16ind`` -- the **delivered** file
     ``perf_opt/_logical_reduce_kernel.py`` (path + sha256 recorded per row).

Exactly one ``final`` row per tune workload is appended (append-only).

CLI: python run_final.py [--launch-count 30]
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import measure  # noqa: E402

KERNEL_ID = "_logical_reduce_kernel::main"
FINAL_IMPL = os.path.join(HERE, "_logical_reduce_kernel.py")
BASE_IMPL = os.path.abspath(os.path.join(HERE, "..", "_logical_reduce_kernel.py"))
PERF_RECORDS = os.path.join(HERE, "perf_records.jsonl")
WORKLOADS = [
    "mask-validation-4k-bool",
    "mask-validation-32k-bool",
    "3d-multidim-reduce-bool",
]


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def append(row):
    with open(PERF_RECORDS, "a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-count", type=int, default=30)
    args = ap.parse_args()

    for wl in WORKLOADS:
        for phase, cand, impl in (
            ("baseline", "baseline_recheck", BASE_IMPL),
            ("final", "final_f16ind", FINAL_IMPL),
        ):
            out_dir = os.path.join("profiles", phase, f"round4_{cand}", cand, wl)
            log = os.path.join("logs", "round4", f"{cand}_{wl}.log")
            res = measure(
                impl, wl, out_dir, kernel_name="main",
                launch_count=args.launch_count, warm_up=5, log_path=log,
                check=(phase == "final"),
            )
            if not res.get("ok"):
                print(f"FAILED {cand} {wl}: {res.get('stderr_tail', '')[-300:]}")
                return 1
            op = res["ops"]["main"]
            append({
                "round": 4,
                "candidate_id": cand,
                "parent_id": "baseline" if phase == "baseline" else "r3_op6_indf16",
                "kernel_id": KERNEL_ID,
                "workload_id": wl,
                "phase": phase,
                "artifact_path": os.path.abspath(impl),
                "artifact_sha256": sha256(impl),
                "duration_us": round(op["mean_us"], 4),
                "l0_pass": True,
                "msprof_raw_path": os.path.abspath(out_dir),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            })
            print(f"{cand:18s} {wl:26s} {op['mean_us']:.4f} us "
                  f"(median {op['median_us']:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
