"""Round-4 interleaved A/B/A/B measurement for one workload (Stage 4 post-DONE).

A = accepted ``_logical_reduce_kernel.py`` (perf_opt final, ``final_rprime``),
B = round-4 candidate.  Runs are strictly interleaved (A,B,A,B,...) inside one
process so that both sides see the same run-state; each run is a separate
``msprof op`` invocation with the standard calibre
(``--launch-count=30 --warm-up=5``, kernel-only ``Task Duration(us)`` mean).

CLI:
  python ab_round4.py --a <impl> --b <impl> --case sparsity-hidden-bfloat16 \
      --pairs 3 [--launch-count 30] [--tag op1]

Writes profiles/ab_round4/<tag>/<arm>_run<k>/ and appends nothing (the perf
records are written by the agent from the printed JSON).
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--case", default="sparsity-hidden-bfloat16")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--launch-count", type=int, default=30)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    root = os.path.join(HERE, "profiles", "ab_round4", args.tag)
    runs = {"A": [], "B": []}
    for k in range(args.pairs):
        for arm, impl in (("A", args.a), ("B", args.b)):
            out_dir = os.path.join(root, f"{arm}_run{k + 1}")
            log = os.path.join(HERE, "logs", "round4", f"ab_{args.tag}_{arm}{k + 1}.log")
            res = measure(
                os.path.join(HERE, impl), args.case, out_dir,
                kernel_name="main", launch_count=args.launch_count, warm_up=5,
                log_path=log,
            )
            if not res.get("ok"):
                print(json.dumps({"ok": False, "arm": arm, "run": k + 1,
                                  "stderr": res.get("stderr_tail")}, ensure_ascii=False))
                return 1
            op = res["ops"]["main"]
            runs[arm].append({
                "run": k + 1, "mean_us": round(op["mean_us"], 4),
                "median_us": round(op["median_us"], 4),
                "min_us": round(op["min_us"], 4), "max_us": round(op["max_us"], 4),
                "n": op["n"], "out_dir": os.path.relpath(out_dir, HERE),
                "impl_sha256": res["impl_sha256"],
            })
            print(f"[{args.tag}] {arm} run{k + 1}: mean={op['mean_us']:.3f} "
                  f"median={op['median_us']:.3f} min={op['min_us']:.3f} "
                  f"max={op['max_us']:.3f}", flush=True)

    a = [r["mean_us"] for r in runs["A"]]
    b = [r["mean_us"] for r in runs["B"]]
    pairs = list(zip(a, b))
    deltas = [(y - x) / x * 100.0 for x, y in pairs]
    signs = [1 if y < x else (-1 if y > x else 0) for x, y in pairs]
    summary = {
        "tag": args.tag,
        "case": args.case,
        "launch_count": args.launch_count,
        "a_impl": args.a,
        "b_impl": args.b,
        "pairs": pairs,
        "a_runs": a,
        "b_runs": b,
        "a_median": round(statistics.median(a), 4),
        "b_median": round(statistics.median(b), 4),
        "a_range": round(max(a) - min(a), 4),
        "b_range": round(max(b) - min(b), 4),
        "b_rel_change_pct": round(statistics.median(b) / statistics.median(a) * 100 - 100, 2),
        "paired_deltas_pct": [round(d, 2) for d in deltas],
        "direction_consistent": all(s == signs[0] and s != 0 for s in signs),
        "b_median_le_29_99": statistics.median(b) <= 29.99,
        "runs": runs,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = os.path.join(root, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
