"""Round-4 ``phase=final`` re-collection for the adopted artifact.

The accepted ``perf_opt/_logical_reduce_kernel.py`` changed after round 3
(bf16 resident tier, round-4 op2), so per the optimize skill Phase 3 note the
``phase=final`` records must be re-taken on the *new* file for every tune
workload (append new rows; the superseded round-3 final rows are preserved
verbatim in ``perf_records_full_history.jsonl``).

Calibre: msprof op, ``--launch-count=30 --warm-up=5``, kernel-only
``Task Duration(us)`` mean -- identical to every other row in the file.
"""

import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import measure  # noqa: E402

KERNEL_ID = "_logical_reduce_kernel::main"
IMPL = "_logical_reduce_kernel.py"
CANDIDATE_ID = "final_round4_bf16bm3"
PARENT_ID = "r4_op2_bf16_bm3"
ROUND = 4
LAUNCH_COUNT = 30
WORKLOADS = [
    "sparsity-hidden-float16",
    "sparsity-hidden-bfloat16",
    "sparsity-seq-float16",
    "3d-multidim-reduce-float16",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    sha = file_sha256(os.path.join(HERE, IMPL))
    print(f"artifact {IMPL} sha256={sha}", flush=True)
    rows = []
    for wl in WORKLOADS:
        out_dir = os.path.join(
            "profiles", "final", "round4", CANDIDATE_ID, wl
        )
        log = os.path.join("logs", "round4", f"final_{wl}.log")
        res = measure(
            os.path.join(HERE, IMPL), wl, out_dir,
            kernel_name="main", launch_count=LAUNCH_COUNT, warm_up=5,
            log_path=log, check=True,
        )
        if not res.get("ok"):
            print(f"{wl}: MSPROF FAILED: {res.get('stderr_tail')}")
            return 1
        op = res["ops"]["main"]
        rec = {
            "round": ROUND,
            "candidate_id": CANDIDATE_ID,
            "parent_id": PARENT_ID,
            "kernel_id": KERNEL_ID,
            "workload_id": wl,
            "phase": "final",
            "artifact_path": IMPL,
            "artifact_sha256": sha,
            "duration_us": round(op["mean_us"], 4),
            "l0_pass": True,
            "msprof_raw_path": os.path.abspath(os.path.join(HERE, out_dir)),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        rows.append(rec)
        print(f"{wl}: mean={op['mean_us']:.3f} median={op['median_us']:.3f} "
              f"min={op['min_us']:.3f} max={op['max_us']:.3f} n={op['n']}", flush=True)
    with open(os.path.join(HERE, "perf_records.jsonl"), "a") as handle:
        for rec in rows:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("appended", len(rows), "final rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
