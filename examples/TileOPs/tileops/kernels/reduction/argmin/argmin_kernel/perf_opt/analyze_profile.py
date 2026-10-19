"""Summarise the msprof PipeUtilization/Memory metrics of a captured kernel.

Usage: python analyze_profile.py <profile_dir> [op_name]
Prints the mean per-block pipe times/ratios for the named op (default: every
captured op, picking the one with the most rows).
"""

import csv
import glob
import os
import statistics
import sys

FIELDS = (
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte2_active_bw(GB/s)",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_scalar_dual_time(us)",
    "aiv_scalar_wait_time(us)",
    "aiv_gm_to_ub_bw(GB/s)",
    "aiv_main_mem_read_bw(GB/s)",
    "aiv_ub_read_bw_vector(GB/s)",
    "aiv_ub_write_bw_vector(GB/s)",
    "read_main_memory_datas(KB)",
    "write_main_memory_datas(KB)",
    "aiv_mte2_instructions",
    "aiv_vec_fops",
)


def summarise(profile_dir, op_name=None):
    if op_name:
        dirs = [d for d in glob.glob(os.path.join(profile_dir, "**", op_name), recursive=True)
                if os.path.isdir(d)]
    else:
        dirs = [d for d in glob.glob(os.path.join(profile_dir, "**", "*"))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "PipeUtilization_*.csv"))]
    for d in sorted(dirs):
        rows = []
        for f in glob.glob(os.path.join(d, "**", "PipeUtilization_*.csv"), recursive=True):
            rows += list(csv.DictReader(open(f)))
        if not rows:
            continue
        name = os.path.basename(d)
        nblocks = len(rows)
        print("== {} ({}) blocks={}".format(name, os.path.relpath(d, profile_dir), nblocks))
        for fld in FIELDS:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r.get(fld, "")))
                except (TypeError, ValueError):
                    pass
            if vals:
                print("   {:28s} mean={:10.4f} min={:10.4f} max={:10.4f}".format(
                    fld, statistics.mean(vals), min(vals), max(vals)))
        # basic info duration
        for f in glob.glob(os.path.join(d, "OpBasicInfo_*.csv")):
            for r in csv.DictReader(open(f)):
                print("   Task Duration(us)            {}".format(r.get("Task Duration(us)")))
                print("   Block Dim                    {}".format(r.get("Block Dim")))


if __name__ == "__main__":
    summarise(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
