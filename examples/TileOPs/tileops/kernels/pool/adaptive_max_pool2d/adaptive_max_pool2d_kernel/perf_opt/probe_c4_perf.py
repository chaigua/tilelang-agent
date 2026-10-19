"""msprof probe: isolate the C4 mte2 inflation (copy / +max / full).

    python probe_c4_perf.py --ch 64
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

from run_experiments import safe_component
import probe_ub2


def run(variant, ch):
    out_dir = os.path.join("profiles", "probe_c4perf", safe_component(f"{variant}_ch{ch}"))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", "--kernel-name=main", f"--output={out_dir}",
        "--launch-count=20", "--warm-up=5", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache",
        sys.executable, "probe_c4_bench.py", "--variant", variant, "--ch", str(ch),
    ]
    log = os.path.join("logs", safe_component(f"c4perf_{variant}_ch{ch}") + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"tag": f"{variant}_ch{ch}", "error": f"rc={proc.returncode}"}
    durs = []
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    eng = {}
    for key, col in (("aiv_time", "aiv_time(us)"), ("vec", "aiv_vec_time(us)"),
                     ("mte2", "aiv_mte2_time(us)"), ("mte3", "aiv_mte3_time(us)")):
        vals = []
        for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv")):
            with open(path) as f:
                for row in csv.DictReader(f):
                    v = row.get(col)
                    if v not in (None, "", "NA"):
                        vals.append(float(v))
        eng[key] = round(statistics.median(vals), 3) if vals else None
    return {"tag": f"{variant}_ch{ch}",
            "duration_us": round(statistics.median(durs), 3), **eng}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=64)
    args = ap.parse_args()
    rows = []
    for variant in ("maxonly", "lean", "square2r"):
        r = run(variant, args.ch)
        rows.append(r)
        if "error" in r:
            print(f"[c4perf] {r['tag']}: FAILED {r['error']}")
        else:
            print(f"[c4perf] {r['tag']}: Task={r['duration_us']}us aiv={r['aiv_time']} "
                  f"vec={r['vec']} mte2={r['mte2']} mte3={r['mte3']}")
    with open("probe_c4perf.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
