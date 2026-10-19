"""Ad-hoc probe: PL-1.13 dual-AIV split A/B on the global copy floor.

    python probe_split.py
"""

import csv
import glob
import json
import os
import statistics
import subprocess
import sys

from run_experiments import _parse_engines, safe_component


def run(case, ch, split):
    tag = f"{case}_ch{ch}" + ("_split" if split else "")
    out_dir = os.path.join("profiles", "probe_split", safe_component(tag))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", "--kernel-name=main", f"--output={out_dir}",
        "--launch-count=20", "--warm-up=5", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache",
        sys.executable, "calib_bench.py", "--case", case, "--iters", "25",
        "--ch", str(ch),
    ]
    if split:
        cmd.append("--split")
    log = os.path.join("logs", safe_component("probe_" + tag) + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"tag": tag, "error": f"rc={proc.returncode}",
                "tail": (proc.stdout + proc.stderr).splitlines()[-8:]}
    durs, bd = [], None
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
                    bd = row.get("Block Dim", bd)
    eng = _parse_engines(out_dir)
    return {"tag": tag, "duration_us": round(statistics.median(durs), 3),
            "blockdim": bd, "engines": eng, "msprof_raw_path": out_dir}


def main():
    rows = []
    for case, ch, split in (
        ("global", 32, False), ("global", 32, True),
        ("global", 64, False), ("global", 64, True),
        ("global", 128, False), ("global", 128, True),
    ):
        r = run(case, ch, split)
        rows.append(r)
        if "error" in r:
            print(f"[probe] {r['tag']}: FAILED {r['error']} :: {r.get('tail')}")
        else:
            e = r["engines"]
            print(f"[probe] {r['tag']}: Task={r['duration_us']} us BlockDim={r['blockdim']} "
                  f"aiv={e.get('aiv_time')} mte2={e.get('mte2')} GM_to_UB={e.get('gm_to_ub_kb')}KB")
    with open("probe_split.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
