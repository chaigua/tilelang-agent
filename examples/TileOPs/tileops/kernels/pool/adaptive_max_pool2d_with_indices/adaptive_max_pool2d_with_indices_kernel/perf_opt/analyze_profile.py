"""Print msprof op engine/memory medians for a profile directory.

Usage:
    python analyze_profile.py <profile_dir> [<profile_dir> ...]

The directory is the one passed to `msprof op --output` (contains OPPROF_*/).
Prints median Task Duration, aiv engine split, GM traffic and BlockDim.
"""

import csv
import glob
import os
import statistics
import sys

BASIC = ["Task Duration(us)", "Block Dim"]
PIPE = [
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_mte2_time(us)",
    "aiv_mte3_time(us)",
    "aiv_scalar_time(us)",
    "aiv_scalar_wait_time(us)",
    "aiv_mte2_active_bw(GB/s)",
    "aiv_mte3_active_bw(GB/s)",
]
MEM = [
    "read_main_memory_datas(KB)",
    "write_main_memory_datas(KB)",
    "GM_to_UB_datas(KB)",
    "UB_to_GM_datas(KB)",
]


def collect(d, names, pattern):
    res = {c: [] for c in names}
    for p in glob.glob(os.path.join(d, "OPPROF_*", "*", "*", pattern)):
        with open(p) as f:
            for row in csv.DictReader(f):
                for c in names:
                    v = row.get(c)
                    if v not in (None, "", "NA"):
                        try:
                            res[c].append(float(v))
                        except ValueError:
                            pass
    return res


def main():
    for d in sys.argv[1:]:
        print(f"=== {d} ===")
        for names, pattern in ((BASIC, "OpBasicInfo_*.csv"), (PIPE, "PipeUtilization_*.csv"),
                               (MEM, "Memory_*.csv")):
            res = collect(d, names, pattern)
            for c in names:
                vals = res[c]
                if vals:
                    print(f"  {c}: med={statistics.median(vals):.3f} n={len(vals)} "
                          f"max={max(vals):.3f}")
                else:
                    print(f"  {c}: NA")


if __name__ == "__main__":
    main()
