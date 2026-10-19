"""Write-side isolation probe for the global-1x1 calibration kernel.

The step-0 `dual` calibration measured a suspicious 96 KB/block UB->GM traffic
against a real write of ~4.5 KB/block (21x).  This probe measures the global
grid (ch=64) under several write forms to locate the amplification:

  elem   -- single-element fp16 keep-alive store only (max-pool style)
  value  -- full fp16 value output only (no index)
  index  -- full int64 index output only
  dual   -- fp16 value + int64 index (the step-0 form)
  dual1d -- same but written through 1-D contiguous regions

Writes structured rows to `write_probe.jsonl` (NOT perf_records.jsonl).

    python probe_write_run.py
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

OUT = "write_probe.jsonl"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5


def run_case(write: str, ch: int = 64) -> dict:
    tag = f"global_ch{ch}_{write}"
    out_dir = os.path.join("profiles", "write_probe", safe_component(tag))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ArithmeticUtilization",
        sys.executable, "calib_bench.py", "--case", "global", "--ch", str(ch),
        "--write", write, "--iters", "25",
    ]
    log = os.path.join("logs", safe_component("wp_" + tag) + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(f"[probe] {tag}: msprof rc={proc.returncode} (see {log})")
        return {"write": write, "error": "msprof_fail"}
    durs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        print(f"[probe] {tag}: no OpBasicInfo rows (see {log})")
        return {"write": write, "error": "invalid_profile"}
    eng = _parse_engines(out_dir)
    mem = {}
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "Memory_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                for c in ("read_main_memory_datas(KB)", "write_main_memory_datas(KB)",
                          "GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)"):
                    v = row.get(c)
                    if v not in (None, "", "NA"):
                        mem.setdefault(c, []).append(float(v))
    mem = {k: round(statistics.median(v), 1) for k, v in mem.items()}
    return {
        "write": write,
        "ch": ch,
        "latency_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "engines": eng,
        "mem": mem,
        "msprof_raw_path": out_dir,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    rows = []
    for write in ("elem", "dual", "stat3d", "stat1d"):
        r = run_case(write)
        rows.append(r)
        if "error" in r:
            print(f"[probe] {r['write']}: {r['error']}")
            continue
        e, m = r["engines"], r["mem"]
        print(
            f"[probe] {r['write']:7s}: Task={r['latency_us']:8.3f} us | aiv={e.get('aiv_time')} "
            f"vec={e.get('vec')} mte2={e.get('mte2')} mte3={e.get('mte3')} "
            f"scalar={e.get('scalar')} | GM_to_UB={m.get('GM_to_UB_datas(KB)')}KB "
            f"UB_to_GM={m.get('UB_to_GM_datas(KB)')}KB rdB={m.get('read_main_memory_datas(KB)')} "
            f"wrB={m.get('write_main_memory_datas(KB)')}"
        )
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nappended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
