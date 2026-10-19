"""Run the Step-0 copy-floor calibration through msprof op and persist results.

Writes structured rows to `copy_floor_calibration.jsonl` (NOT perf_records.jsonl,
which is reserved for kind=tune benchmark workloads) and prints the floor table.

    python calib_run.py
"""

import csv
import datetime
import glob
import json
import os
import statistics
import subprocess
import sys

from run_experiments import _parse_engines, safe_component

OUT = "copy_floor_calibration.jsonl"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5


def run_case(case: str, ch: int | None = None) -> dict:
    tag = f"{case}" + (f"_ch{ch}" if ch is not None else "")
    out_dir = os.path.join("profiles", "calibration", safe_component(tag))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ArithmeticUtilization",
        sys.executable, "calib_bench.py", "--case", case, "--iters", "25",
    ]
    if ch is not None:
        cmd += ["--ch", str(ch)]
    log = os.path.join("logs", safe_component("calib_" + tag) + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"msprof failed for {tag}: rc={proc.returncode}; see {log}")
    durs, blockdim = [], None
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
                    blockdim = row.get("Block Dim", blockdim)
    eng = _parse_engines(out_dir)
    return {
        "case": tag,
        "ch": ch,
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "blockdim": blockdim,
        "engines": eng,
        "msprof_raw_path": out_dir,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    rows = []
    for case, chs in (("global", (32, 64, 128)), ("general", (32, 64))):
        for ch in chs:
            rows.append(run_case(case, ch))
    for r in rows:
        e = r["engines"]
        print(
            f"[calib] {r['case']}: Task Duration = {r['duration_us']} us | "
            f"BlockDim={r['blockdim']} aiv={e.get('aiv_time')} vec={e.get('vec')} "
            f"mte2={e.get('mte2')} mte3={e.get('mte3')} GM_to_UB={e.get('gm_to_ub_kb')}KB"
        )
        print(f"        70% line = floor / 0.7 = {r['duration_us'] / 0.7:.3f} us")
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nappended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
