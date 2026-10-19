"""run_experiments.py -- Stage 4 experiment batch runner (T-4) for
`_fp8_quant_kernel`.

Adapted from .agents/skills/tilelang-op-optimize/references/run_experiments_template.py.

For each configured candidate branch, in order:
  1. runs the L0 precision regression (`python {branch} --level L0`);
  2. measures the kernel-only Task Duration via `msprof op` (the only latency
     metric) over the tune workloads;
  3. parses the GM/L2 roofline entry (`Ratio = Perf / computility`, the same
    口径 as the Stage 5 report) from the raw profile;
  4. appends one line per measurement to perf_records.jsonl (append-only);
  5. prints a candidate-vs-current-best summary table.

Usage (inside perf_opt/):
  python run_experiments.py --phase baseline --all
  python run_experiments.py --phase final --all
  python run_experiments.py --round 1 --workload kv-index-8k-d64-fp16 --only r1_vexp2
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys

# Make the tileops package importable (roofline parsing) regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TILEOPS_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 6)))
if _TILEOPS_ROOT not in sys.path:
    sys.path.insert(0, _TILEOPS_ROOT)

# --------------------------------------------------------------- CONFIG
OP = "_fp8_quant_kernel"
KERNEL_ID = (
    "tileops/kernels/quantization/fp8_quant/fp8_quant_kernel/"
    "_fp8_quant_kernel.py::_fp8_quant_kernel::_fp8_quant_fwd_main"
)
KERNEL_NAME = "_fp8_quant_fwd_main"
DIRECT_CMD = [
    sys.executable,
    "bench.py",
    "--impl", "{impl}",
    "--case", "{workload_id}",
    "--iters", "25",
]
LAUNCH_COUNT = 20
WARM_UP = 5
TIMEOUT_S = 2400
CURRENT_BEST = f"{OP}.py"
CURRENT_BEST_ID = "baseline"
MSPROF_METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,Roofline"
# --------------------------------------------------------------- /CONFIG

PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"

# ----------------------------------------------------------- EXPERIMENTS
EXPERIMENTS = [
    # round 0 baseline = the Stage 3 kernel (read-only reference)
    {"file": "../_fp8_quant_kernel.py", "opt_id": "baseline", "parent_id": None},
    # round 1/2 candidates (single primary change each, from the baseline)
    {"file": f"{OP}_r1_vexp2.py", "opt_id": "r1_vexp2", "parent_id": "baseline"},
    {"file": f"{OP}_r1_blk4096.py", "opt_id": "r1_blk4096", "parent_id": "baseline"},
    {"file": f"{OP}_r1b_recrip.py", "opt_id": "r1b_recrip", "parent_id": "r1_vexp2"},
    {"file": f"{OP}_r2_fused.py", "opt_id": "r2_fused", "parent_id": "r1_blk4096"},
    # merged current best (== the final drop-in, pending conductor wrapper flip)
    {"file": f"{OP}.py", "opt_id": "r2-merged", "parent_id": "r1_blk4096"},
]
# ------------------------------------------------------------------ /EXPERIMENTS


def build_bench_cmd(impl_path, workload_id):
    return [part.format(impl=impl_path, workload_id=workload_id) for part in DIRECT_CMD]


def safe_component(value):
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def run_l0(branch_file):
    proc = subprocess.run([sys.executable, branch_file, "--level", "L0"],
                          capture_output=True, text=True, timeout=TIMEOUT_S)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).splitlines()[-6:]
        print("    L0 stderr tail: " + " | ".join(tail))
    return proc.returncode == 0


def _roofline_ratio(out_dir):
    try:
        from tileops.benchmark.msprof_roofline import parse_bin_file
    except ImportError:
        return None
    bins = sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "visualize_data.bin")))
    if not bins:
        return None
    ratios = []
    for b in bins:
        try:
            info = parse_bin_file(b)
        except Exception:
            continue
        for e in info["roofline_entries"]:
            if e.get("title") == "GM/L2" and abs(float(e.get("bw", 0)) - 1.75) < 1e-6:
                ratios.append(float(e["ratio"]))
    return round(statistics.median(ratios), 5) if ratios else None


def run_msprof(branch_file, workload_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "msprof", "op",
        f"--kernel-name={KERNEL_NAME}",
        f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}",
        f"--warm-up={WARM_UP}",
        "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}",
        *build_bench_cmd(branch_file, workload_id),
    ]
    log = os.path.join("logs", safe_component(out_dir) + ".log")
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        f.write(f"# {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}
    durs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        return {"error": "no OpBasicInfo rows", "log": log}
    return {
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "roofline_ratio": _roofline_ratio(out_dir),
        "msprof_raw_path": out_dir,
    }


def append_record(record):
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tune_workload_ids():
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    return [item["workload_id"] for item in inventory["workloads"]
            if item.get("kernel_id") == KERNEL_ID and item.get("kind") == "tune"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--only", help="run a single opt_id")
    ap.add_argument("--workload", help="one non-smoke benchmark workload ID")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--phase", choices=("baseline", "candidate", "merged", "final"),
                    default="candidate")
    ap.add_argument("--best-id", default=None)
    args = ap.parse_args()

    global CURRENT_BEST_ID
    if args.best_id:
        CURRENT_BEST_ID = args.best_id
    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1
    if args.phase == "baseline" and args.round != 0:
        ap.error("baseline measurements must use round 0")

    available = tune_workload_ids()
    if args.phase == "candidate":
        if args.all:
            selected = list(available)
        elif args.workload:
            selected = [args.workload]
        else:
            ap.error("candidate measurements require --workload or --all")
    else:
        if not args.all or args.workload:
            ap.error("baseline/merged/final measurements require --all and no --workload")
        selected = list(available)
    unknown = [w for w in selected if w not in available]
    if unknown or not selected:
        ap.error(f"unknown or empty tune workload selection: {unknown or selected}")

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if args.phase in ("baseline", "merged", "final") and len(todo) != 1:
        ap.error("baseline/merged/final measurements must use exactly one candidate per kernel")

    results = []
    for exp in todo:
        branch = exp["file"]
        print(f"=== {exp['opt_id']}: {branch} ===")
        if not os.path.isfile(branch):
            results.append({**exp, "status": "missing_file"})
            continue
        if not run_l0(branch):
            results.append({**exp, "status": "l0_fail", "l0_pass": False})
            continue
        for workload_id in selected:
            prof = run_msprof(branch, workload_id, os.path.join(
                "profiles", args.phase, f"round{args.round}",
                safe_component(exp["opt_id"]), safe_component(workload_id)))
            if "error" in prof:
                results.append({**exp, "workload_id": workload_id, "status": "msprof_fail"})
                print(f"  {workload_id}: msprof FAILED: {prof['error']}")
                continue
            append_record({
                "round": args.round,
                "candidate_id": exp["opt_id"],
                "parent_id": exp.get("parent_id"),
                "kernel_id": KERNEL_ID,
                "workload_id": workload_id,
                "phase": args.phase,
                "artifact_path": branch,
                "artifact_sha256": file_sha256(branch),
                "duration_us": prof["duration_us"],
                "l0_pass": True,
                "roofline_ratio": prof["roofline_ratio"],
                "msprof_raw_path": prof["msprof_raw_path"],
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            results.append({**exp, "workload_id": workload_id, "status": "ok",
                            "l0_pass": True, "duration_us": prof["duration_us"],
                            "ratio": prof["roofline_ratio"]})
            print(f"  {workload_id}: L0 pass, median Task Duration = "
                  f"{prof['duration_us']} us, Ratio = {prof['roofline_ratio']}")

    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if r.get("kernel_id") == KERNEL_ID and r.get("candidate_id") == CURRENT_BEST_ID:
                bases[r["workload_id"]] = r["duration_us"]
    print("\n| branch | workload_id | l0 | Task Duration(us) | Ratio | vs current best | status |")
    print("|---|---|---|---:|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        rel = ""
        base = bases.get(r.get("workload_id"))
        if dur is not None and base:
            rel = f"{(base - dur) / base * 100:+.1f}%"
        print(f"| {r['opt_id']} | {r.get('workload_id', '-')} | {r.get('l0_pass', '-')} | "
              f"{dur if dur is not None else '-'} | {r.get('ratio', '-')} | {rel} | {r['status']} |")
    print(f"\n(current best candidate for the comparison column: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
