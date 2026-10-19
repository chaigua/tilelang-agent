"""Run the structural-floor calibration probes (NOT tune workloads).

Writes one line per measurement to ``copy_floor_calibration.jsonl`` (probe
floors) and ``op_cost_calibration.jsonl`` (derived unit prices).  These files
are deliberately separate from ``perf_records.jsonl``: probes are not
benchmark workloads and must never be counted as performance records.

Usage:
    python run_probes.py [--cases a,b] [--modes empty,copyout,read,copy]
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_bench import measure  # noqa: E402

PROBE_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_floor.py")
FLOORS = "copy_floor_calibration.jsonl"
PRICES = "op_cost_calibration.jsonl"

ALL_CASES = [
    "sparsity-hidden-float16",
    "sparsity-hidden-bfloat16",
    "sparsity-seq-float16",
    "3d-multidim-reduce-float16",
]
DEFAULT_MODES = ["empty", "copyout", "read", "copy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--tag", default="round0")
    args = ap.parse_args()

    cases = [c for c in args.cases.split(",") if c]
    modes = [m for m in args.modes.split(",") if m]
    rows = []
    for case in cases:
        for mode in modes:
            out_dir = os.path.join("profiles", "probes", args.tag, f"{case}_{mode}")
            log = os.path.join("logs", "probes", args.tag, f"{case}_{mode}.log")
            res = measure(
                PROBE_IMPL,
                case,
                out_dir,
                kernel_name="main",
                launch_count=args.launch_count,
                warm_up=5,
                log_path=log,
                probe=mode,
            )
            if not res.get("ok"):
                print(f"{case} {mode}: FAILED {res.get('stderr_tail','')[-300:]}")
                rows.append({"case": case, "mode": mode, "ok": False,
                             "error": res.get("stderr_tail", "")[-500:]})
                continue
            op = res["ops"]["main"]
            row = {
                "tag": args.tag,
                "case": case,
                "mode": mode,
                "n": op["n"],
                "mean_us": round(op["mean_us"], 4),
                "median_us": round(op["median_us"], 4),
                "min_us": round(op["min_us"], 4),
                "max_us": round(op["max_us"], 4),
                "msprof_raw_path": out_dir,
                "launch_count": args.launch_count,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
            rows.append(row)
            print(f"{case:32s} {mode:8s} mean={row['mean_us']:8.3f} "
                  f"med={row['median_us']:8.3f} min={row['min_us']:8.3f}")

    with open(FLOORS, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nappended {len(rows)} rows to {FLOORS}")


if __name__ == "__main__":
    main()
