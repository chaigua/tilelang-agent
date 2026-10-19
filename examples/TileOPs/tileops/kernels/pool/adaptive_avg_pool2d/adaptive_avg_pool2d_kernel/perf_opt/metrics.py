#!/usr/bin/env python3
"""Print the AICore pipe / memory metrics of the newest msprof op run under <dir>.

  python3 metrics.py <profile_out_dir>
"""
import csv
import glob
import os
import sys


def newest_csv(d, pattern):
    hits = sorted(glob.glob(os.path.join(d, "OPPROF_*", "*", "*", pattern)))
    return hits[-1] if hits else None


def main():
    d = sys.argv[1].rstrip("/")
    subs = sorted(glob.glob(os.path.join(d, "OPPROF_*", "*", "*")))
    # newest sub-block dir that has an OpBasicInfo file
    subs = [s for s in subs if glob.glob(os.path.join(s, "OpBasicInfo_*.csv"))]
    if not subs:
        print("no OpBasicInfo under", d)
        return
    sub = sorted(subs)[-1]
    print("dir:", sub)
    for pat, want in (
        ("OpBasicInfo_*.csv", ["Task Duration(us)", "Block Dim"]),
        ("PipeUtilization_*.csv", ["aiv_time(us)", "aiv_vec_time(us)", "aiv_mte2_time(us)",
                                   "aiv_mte3_time(us)", "aiv_scalar_wait_time(us)",
                                   "aiv_scalar_stall_by_ub_time(us)",
                                   "aiv_mte2_ratio", "aiv_vec_ratio"]),
        ("MemoryUB_*.csv", ["GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)",
                            "GM_to_UB_bandwidth(GB/s)", "UB_to_GM_bandwidth(GB/s)"]),
        ("ArithmeticUtilization_*.csv", ["aiv_vec_fops", "aiv_vec_total_instr_number"]),
    ):
        p = sorted(glob.glob(os.path.join(sub, pat)))
        if not p:
            continue
        with open(p[-1]) as f:
            rows = list(csv.DictReader(f))
        print("=", pat)
        for r in rows:
            cells = [f"{k}={r[k]}" for k in want if k in r and r[k] not in ("NA", "")]
            if cells:
                print("  " + "  ".join(cells))


if __name__ == "__main__":
    main()
