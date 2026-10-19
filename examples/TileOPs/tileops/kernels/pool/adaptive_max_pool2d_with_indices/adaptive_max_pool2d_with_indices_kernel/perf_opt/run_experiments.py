"""run_experiments.py -- Stage 4 experiment batch runner (T-4) for
`_adaptive_max_pool2d_with_indices_kernel`.

Adapted from .agents/skills/tilelang-op-optimize/references/run_experiments_template.py
(via the adaptive_max_pool2d Stage 4 sibling runner).

Per experiment branch this runner, in order:
  1. runs the L0 precision regression (branch file `--level L0`);
  2. wraps the msprof op measurement (kernel-only Task Duration, the only latency metric);
  3. appends one line per measured tune workload to perf_records.jsonl (append-only);
  4. prints a candidate-vs-current-best summary table.

Usage (inside perf_opt/):
  python run_experiments.py --phase baseline --all
  python run_experiments.py --round 1 --workload global-1x1-float16 --branch <file>
  python run_experiments.py --phase merged --all --best-id r1-merged --branch <file>
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

# --------------------------------------------------------------- CONFIG
OP = "_adaptive_max_pool2d_with_indices_kernel"
KERNEL_ID = (
    "tileops/kernels/pool/adaptive_max_pool2d_with_indices/"
    "adaptive_max_pool2d_with_indices_kernel/"
    "_adaptive_max_pool2d_with_indices_kernel.py::"
    "_adaptive_max_pool2d_with_indices_kernel::main"
)
KERNEL_NAME = "main"
DIRECT_CMD = [
    sys.executable,
    "bench.py",
    "--impl",
    "{impl}",
    "--case",
    "{workload_id}",
    "--iters",
    "25",
]
LAUNCH_COUNT = 20
WARM_UP = 5
TIMEOUT_S = 3600
CURRENT_BEST_ID = "baseline"
# --------------------------------------------------------------- /CONFIG

PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)

EXPERIMENTS = []


def build_bench_cmd(impl_path: str, workload_id: str) -> list[str]:
    return [part.format(impl=impl_path, workload_id=workload_id) for part in DIRECT_CMD]


def safe_component(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def run_l0(branch_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).splitlines()[-6:]
        print("    L0 stderr tail: " + " | ".join(tail))
    return proc.returncode == 0


def _parse_engines(out_dir: str) -> dict:
    """Median aiv_time / vec / mte2 / mte3 / scalar and GM<->UB bytes from the CSVs."""
    times = {"aiv_time": [], "vec": [], "mte2": [], "mte3": [], "scalar": []}
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                for key, col in (
                    ("aiv_time", "aiv_time(us)"),
                    ("vec", "aiv_vec_time(us)"),
                    ("mte2", "aiv_mte2_time(us)"),
                    ("mte3", "aiv_mte3_time(us)"),
                    ("scalar", "aiv_scalar_time(us)"),
                ):
                    v = row.get(col)
                    if v not in (None, "", "NA"):
                        times[key].append(float(v))
    out = {}
    for key, vals in times.items():
        out[key] = round(statistics.median(vals), 3) if vals else None
    gm = []
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "Memory_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                v = row.get("GM_to_UB_datas(KB)") or row.get("GM_to_UB(KB)")
                if v not in (None, "", "NA"):
                    gm.append(float(v))
    out["gm_to_ub_kb"] = round(statistics.median(gm), 1) if gm else None
    return out


def run_msprof(branch_file: str, workload_id: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    cmd = [
        "msprof",
        "op",
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
    blockdim = None
    for path in sorted(
        glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    ):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
                    blockdim = row.get("Block Dim", blockdim)
    if not durs:
        return {"error": "no OpBasicInfo rows", "log": log}
    return {
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "blockdim": blockdim,
        "engines": _parse_engines(out_dir),
        "msprof_raw_path": out_dir,
    }


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tune_workload_ids() -> list[str]:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("workloads"), list):
        raise ValueError("workload_inventory.json must contain a workloads array")
    return [
        item["workload_id"]
        for item in inventory["workloads"]
        if item.get("kind") == "tune"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--only", help="run a single opt_id")
    ap.add_argument("--branch", action="append", default=[],
                    help="branch file to measure (opt_id = file stem); repeatable")
    ap.add_argument("--workload", help="one non-smoke benchmark workload ID")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--phase", choices=("baseline", "candidate", "merged", "final"),
                    default="candidate")
    ap.add_argument("--best-id", default=None,
                    help="override CURRENT_BEST_ID for the comparison table")
    ap.add_argument("--record-path", default=None,
                    help="override artifact_path recorded in perf_records.jsonl")
    args = ap.parse_args()

    global CURRENT_BEST_ID, EXPERIMENTS
    if args.best_id:
        CURRENT_BEST_ID = args.best_id

    if args.branch:
        EXPERIMENTS = [
            {"file": b, "opt_id": os.path.splitext(os.path.basename(b))[0],
             "parent_id": args.best_id}
            for b in args.branch
        ]

    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1
    if args.phase == "baseline" and args.round != 0:
        ap.error("baseline measurements must use round 0")

    if args.phase == "candidate":
        if args.all or not args.workload:
            ap.error("candidate measurements require exactly one --workload")
    elif not args.all or args.workload:
        ap.error("baseline/merged/final measurements require --all and no --workload")
    available = tune_workload_ids()
    selected = list(available) if args.all else [args.workload]
    unknown = [w for w in selected if w not in available]
    if unknown or not selected:
        ap.error(f"unknown or empty tune workload selection: {unknown or selected}")

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if not todo:
        print("no experiments configured (edit EXPERIMENTS or pass --branch)")
        return 2
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
            prof = run_msprof(
                branch,
                workload_id,
                os.path.join(
                    "profiles",
                    args.phase,
                    f"round{args.round}",
                    safe_component(exp["opt_id"]),
                    safe_component(workload_id),
                ),
            )
            if "error" in prof:
                results.append({**exp, "workload_id": workload_id,
                                "status": "msprof_fail", "l0_pass": True})
                print(f"  {workload_id}: msprof FAILED: {prof['error']}")
                continue
            append_record(
                {
                    "round": args.round,
                    "candidate_id": exp["opt_id"],
                    "parent_id": exp.get("parent_id"),
                    "kernel_id": KERNEL_ID,
                    "workload_id": workload_id,
                    "phase": args.phase,
                    "artifact_path": args.record_path or branch,
                    "artifact_sha256": file_sha256(branch),
                    "duration_us": prof["duration_us"],
                    "l0_pass": True,
                    "msprof_raw_path": prof["msprof_raw_path"],
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
            results.append(
                {
                    **exp,
                    "workload_id": workload_id,
                    "status": "ok",
                    "l0_pass": True,
                    "duration_us": prof["duration_us"],
                    "blockdim": prof.get("blockdim"),
                    **prof.get("engines", {}),
                }
            )
            e = prof.get("engines", {})
            print(
                f"  {workload_id}: L0 pass, median Task Duration = {prof['duration_us']} us "
                f"| BlockDim={prof.get('blockdim')} aiv={e.get('aiv_time')} "
                f"vec={e.get('vec')} mte2={e.get('mte2')} mte3={e.get('mte3')} "
                f"scalar={e.get('scalar')} GM_to_UB={e.get('gm_to_ub_kb')}KB"
            )

    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if r.get("kernel_id") == KERNEL_ID and r.get("candidate_id") == CURRENT_BEST_ID:
                bases[r.get("workload_id")] = r["duration_us"]
    print("\n| branch | workload_id | l0 | Task Duration(us) | vs current best | status |")
    print("|---|---|---|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        rel = ""
        base = bases.get(r.get("workload_id"))
        if dur is not None and base:
            rel = f"{(base - dur) / base * 100:+.1f}%"
        print(
            f"| {r['opt_id']} | {r.get('workload_id', '-')} | {r.get('l0_pass', '-')} | "
            f"{dur if dur is not None else '-'} | {rel} | {r['status']} |"
        )
    print(f"\n(current best candidate for the comparison column: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
