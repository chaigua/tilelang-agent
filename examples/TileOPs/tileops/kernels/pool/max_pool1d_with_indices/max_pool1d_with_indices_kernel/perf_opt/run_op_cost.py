"""Driver for the vector-op cost calibration probe (non-tune data).

Runs `msprof op` serially over the probe configs and appends one JSON line per
measurement to `op_cost_calibration.jsonl` (NOT perf_records.jsonl), with the
derived per-op cost.  Usage:

    python run_op_cost.py [--only vmax]
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

CORES = 40
MSPROF_LOCK = "/tmp/tilelang-msprof.lock"
OUT = "op_cost_calibration.jsonl"

CONFIGS = [
    # (op, m, ch, reps, dtype)
    ("vmax", 8, 64, 512, "float32"),      # 512-lane plane op
    ("vmax", 8, 128, 512, "float32"),     # 1024-lane plane op
    ("vcmp", 1, 64, 512, "float32"),
    ("vcmp", 8, 64, 512, "float32"),
    ("vselect", 1, 64, 512, "float32"),
    ("vselect", 8, 64, 512, "float32"),
    ("vshr", 8, 64, 512, "int32"),
    ("vxor", 8, 64, 512, "int32"),
    ("vmul_i", 8, 64, 512, "int32"),
]


def measure(op, m, ch, reps, dtype="float32"):
    tag = f"{op}_m{m}_ch{ch}_r{reps}"
    out_dir = os.path.join("profiles", "calibration", "ops", tag)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "flock", MSPROF_LOCK, "msprof", "op", "--kernel-name=main",
        f"--output={out_dir}", "--launch-count=20", "--warm-up=5", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory",
        sys.executable, "probe_ops.py", "--op", op, "--m", str(m), "--ch", str(ch),
        "--reps", str(reps), "--iters", "25", "--dtype", dtype,
    ]
    os.makedirs("logs", exist_ok=True)
    log = os.path.join("logs", f"opcost_{tag}.log")
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"tag": tag, "op": op, "m": m, "ch": ch, "reps": reps, "error": "msprof_rc"}
    durs, vec, scalar = [], [], []
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                for key, col in (("vec", "aiv_vec_time(us)"), ("scalar", "aiv_scalar_time(us)")):
                    v = row.get(col)
                    if v not in (None, "", "NA"):
                        (vec if key == "vec" else scalar).append(float(v))
    if not durs:
        return {"tag": tag, "op": op, "m": m, "ch": ch, "reps": reps, "error": "no_rows"}
    return {
        "tag": tag,
        "op": op,
        "m": m,
        "ch": ch,
        "reps": reps,
        "duration_us": round(statistics.median(durs), 3),
        "vec_us": round(statistics.median(vec), 3) if vec else None,
        "scalar_us": round(statistics.median(scalar), 3) if scalar else None,
        "lanes": m * ch,
        "msprof_raw_path": out_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    floors = {}
    for op, m, ch, reps, dtype in CONFIGS:
        if args.only and op != args.only:
            continue
        r = measure(op, m, ch, reps, dtype)
        r["lanes"] = m * ch
        if op == "floor":
            floors[(m, ch)] = r.get("duration_us")
        print(json.dumps(r))
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")

    # derive per-op cost from the stored rows
    rows = [json.loads(line) for line in open(OUT) if line.strip()]
    floor = {(r["m"], r["ch"]): r["duration_us"] for r in rows if r["op"] == "floor"}
    print("\n| op | m | ch | lanes | reps | Task(us) | per-op(us/core) | vec(us) | scalar(us) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["op"] == "floor" or "duration_us" not in r:
            continue
        f = floor.get((r["m"], r["ch"]))
        per = ((r["duration_us"] - f) / r["reps"]) if f else None
        print(f"| {r['op']} | {r['m']} | {r['ch']} | {r['lanes']} | {r['reps']} | "
              f"{r['duration_us']} | {per * 1000:.2f} ns | {r.get('vec_us')} | {r.get('scalar_us')} |")


if __name__ == "__main__":
    main()
