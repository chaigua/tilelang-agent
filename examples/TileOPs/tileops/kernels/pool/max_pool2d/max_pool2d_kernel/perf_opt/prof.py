"""Parse an msprof op profile dir -> median Task Duration + pipe/memory metrics.

Reusable helper for the Stage 4 tuning loop (the same parsing logic as
run_copy_floor2.py, factored out so every round reports the identical schema).

  python3 prof.py <prof_dir> [<prof_dir> ...]
"""
import csv
import glob
import json
import os
import statistics
import sys

ENG = (("aiv_time", "aiv_time(us)"), ("vec", "aiv_vec_time(us)"),
       ("mte2", "aiv_mte2_time(us)"), ("mte3", "aiv_mte3_time(us)"),
       ("scalar", "aiv_scalar_time(us)"), ("fix", "aiv_fixpipe_ratio"),
       ("vec_ratio", "aiv_vec_ratio"), ("mte2_ratio", "aiv_mte2_ratio"),
       ("mte3_ratio", "aiv_mte3_ratio"), ("scalar_ratio", "aiv_scalar_ratio"))
MEM = (("gm_to_ub_kb", "GM_to_UB_datas(KB)"),
       ("ub_to_gm_kb", "UB_to_GM_datas(KB)"))


def parse(d):
    durs, blocks = [], None
    for p in sorted(glob.glob(os.path.join(d, "OPPROF_*", "*", "*",
                                           "OpBasicInfo_*.csv"))):
        for row in csv.DictReader(open(p)):
            if row.get("Task Duration(us)"):
                durs.append(float(row["Task Duration(us)"]))
                blocks = row.get("Block Dim", blocks)
    out = {"block_dim": blocks, "n": len(durs),
           "duration_us": round(statistics.median(durs), 3) if durs else None,
           "dur_min": round(min(durs), 3) if durs else None,
           "dur_max": round(max(durs), 3) if durs else None}
    buckets = {k: [] for k, _ in ENG + MEM}
    for p in glob.glob(os.path.join(d, "OPPROF_*", "*", "*",
                                    "PipeUtilization_*.csv")):
        for row in csv.DictReader(open(p)):
            for k, col in ENG:
                x = row.get(col)
                if x not in (None, "", "NA"):
                    buckets[k].append(float(x))
    for p in glob.glob(os.path.join(d, "OPPROF_*", "*", "*", "Memory_*.csv")):
        for row in csv.DictReader(open(p)):
            for k, col in MEM:
                x = row.get(col)
                if x not in (None, "", "NA"):
                    buckets[k].append(float(x))
    for k, vals in buckets.items():
        out[k] = round(statistics.median(vals), 3) if vals else None
    return out


def main():
    for d in sys.argv[1:]:
        r = parse(d)
        print(os.path.basename(d.rstrip("/")) + "\t" + json.dumps(r))


if __name__ == "__main__":
    main()
