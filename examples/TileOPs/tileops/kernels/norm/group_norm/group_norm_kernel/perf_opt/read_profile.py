#!/usr/bin/env python3
"""Parse an `msprof op` output directory into the Stage 4 comparison metrics.

Metrics (all `msprof op` derived -- the single latency metric is
`Task Duration(us)`):
  * task_duration_us: median across the recorded launches
  * per-pipe medians from PipeUtilization (aiv_vec / aiv_mte2 / aiv_mte3 /
    aiv_scalar, summed over blocks and reported per launch)
  * GM traffic from Memory (read_main_memory_datas / write_main_memory_datas)
  * achieved bandwidth = traffic / duration

Usage:
    python read_profile.py <OPPROF_dir_or_msprof_output_dir> [--json out.json]
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def find_launch_dirs(root):
    """Accept either the msprof --output dir or the inner OPPROF_* dir."""
    cands = glob.glob(os.path.join(root, "OPPROF_*")) or [root]
    dirs = []
    for c in cands:
        for op_dir in glob.glob(os.path.join(c, "*")):
            if os.path.isdir(op_dir):
                subs = [p for p in glob.glob(os.path.join(op_dir, "*"))
                        if os.path.isdir(p) and glob.glob(os.path.join(p, "OpBasicInfo_*.csv"))]
                dirs.extend(subs)
    return sorted(dirs, key=lambda p: (len(p), p))


def _read_csv_row(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def parse(root):
    launches = find_launch_dirs(root)
    if not launches:
        raise SystemExit(f"no OpBasicInfo CSV under {root}")

    durations = []
    op_names = set()
    pipe = {"aiv_time": [], "aiv_vec": [], "aiv_mte2": [], "aiv_mte3": [],
            "aiv_scalar": [], "mte2_bw": [], "mte3_bw": [], "blocks": []}
    mem = {"gm_read_kb": [], "gm_write_kb": []}

    for ld in launches:
        info = _read_csv_row(glob.glob(os.path.join(ld, "OpBasicInfo_*.csv"))[0])[0]
        durations.append(float(info["Task Duration(us)"]))
        op_names.add(info["Op Name"])
        pipe["blocks"].append(int(info["Block Dim"]))

        pu = glob.glob(os.path.join(ld, "PipeUtilization_*.csv"))
        if pu:
            rows = _read_csv_row(pu[0])
            for key, col in (("aiv_time", "aiv_time(us)"), ("aiv_vec", "aiv_vec_time(us)"),
                             ("aiv_mte2", "aiv_mte2_time(us)"), ("aiv_mte3", "aiv_mte3_time(us)"),
                             ("aiv_scalar", "aiv_scalar_time(us)")):
                vals = [float(r[col]) for r in rows if r.get(col) not in (None, "NA", "")]
                # per-launch value = max over blocks (the limiting block)
                pipe[key].append(max(vals) if vals else float("nan"))

        mm = glob.glob(os.path.join(ld, "Memory_*.csv"))
        if mm:
            rows = _read_csv_row(mm[0])
            for key, col in (("gm_read_kb", "read_main_memory_datas(KB)"),
                             ("gm_write_kb", "write_main_memory_datas(KB)")):
                vals = [float(r[col]) for r in rows if r.get(col) not in (None, "NA", "")]
                mem[key].append(sum(vals) if vals else float("nan"))

    out = {
        "launches": len(durations),
        "op_names": sorted(op_names),
        "block_dim": int(_med(pipe["blocks"])),
        "task_duration_us": round(_med(durations), 4),
        "task_duration_min_us": round(min(durations), 4),
        "task_duration_max_us": round(max(durations), 4),
    }
    for k in ("aiv_time", "aiv_vec", "aiv_mte2", "aiv_mte3", "aiv_scalar"):
        out[k + "_us"] = round(_med(pipe[k]), 4)
    r = _med(mem["gm_read_kb"])
    w = _med(mem["gm_write_kb"])
    out["gm_read_kb"] = round(r, 2)
    out["gm_write_kb"] = round(w, 2)
    traffic_bytes = (r + w) * 1024.0
    out["traffic_mb"] = round(traffic_bytes / 1e6, 4)
    out["achieved_gbps"] = round(traffic_bytes / (out["task_duration_us"] * 1e-6) / 1e9, 2) \
        if out["task_duration_us"] else None
    out["launch_dirs_sample"] = launches[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = parse(args.path)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
