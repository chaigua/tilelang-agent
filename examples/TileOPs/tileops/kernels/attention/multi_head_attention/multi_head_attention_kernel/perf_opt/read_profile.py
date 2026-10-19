#!/usr/bin/env python3
"""Parse an `msprof op` output directory into Stage 4 comparison metrics.

Single latency metric = `Task Duration(us)` (median across recorded launches).
Also reports the per-pipe medians (limiting block) used for bottleneck
diagnosis: aic_cube / aic_scalar, aiv_vec / aiv_mte2 / aiv_mte3 / aiv_scalar,
and the GM traffic from the Memory CSV.

Usage:
    python read_profile.py <msprof_output_dir> [--json out.json]
"""

import argparse
import csv
import glob
import json
import os
import statistics


def _med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def find_launch_dirs(root):
    cands = sorted(glob.glob(os.path.join(root, "OPPROF_*")))
    if not cands:
        cands = [root]
    dirs = []
    for c in cands:
        for op_dir in sorted(glob.glob(os.path.join(c, "*"))):
            if not os.path.isdir(op_dir):
                continue
            subs = [
                p for p in sorted(glob.glob(os.path.join(op_dir, "*")))
                if os.path.isdir(p)
                and glob.glob(os.path.join(p, "OpBasicInfo_*.csv"))
            ]
            dirs.extend(subs)
    return dirs


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _col_max(rows, col):
    vals = []
    for r in rows:
        v = r.get(col)
        if v not in (None, "NA", ""):
            vals.append(float(v))
    return max(vals) if vals else float("nan")


def _col_sum(rows, col):
    vals = []
    for r in rows:
        v = r.get(col)
        if v not in (None, "NA", ""):
            vals.append(float(v))
    return sum(vals) if vals else float("nan")


PIPES = (
    ("aic_cube", "aic_cube_time(us)"),
    ("aic_mte1", "aic_mte1_time(us)"),
    ("aic_mte2", "aic_mte2_time(us)"),
    ("aic_fixpipe", "aic_fixpipe_time(us)"),
    ("aic_scalar", "aic_scalar_time(us)"),
    ("aic_scalar_wait", "aic_scalar_wait_time(us)"),
    ("aiv_time", "aiv_time(us)"),
    ("aiv_vec", "aiv_vec_time(us)"),
    ("aiv_mte2", "aiv_mte2_time(us)"),
    ("aiv_mte3", "aiv_mte3_time(us)"),
    ("aiv_scalar", "aiv_scalar_time(us)"),
)


def parse(root):
    launches = find_launch_dirs(root)
    if not launches:
        raise SystemExit(f"no OpBasicInfo CSV under {root}")

    durations, op_names, blocks = [], set(), []
    pipe = {k: [] for k, _ in PIPES}
    mem = {"gm_read_kb": [], "gm_write_kb": []}

    for ld in launches:
        info = _rows(glob.glob(os.path.join(ld, "OpBasicInfo_*.csv"))[0])[0]
        durations.append(float(info["Task Duration(us)"]))
        op_names.add(info["Op Name"])
        blocks.append(int(info["Block Dim"]))

        pu = glob.glob(os.path.join(ld, "PipeUtilization_*.csv"))
        if pu:
            rows = _rows(pu[0])
            for key, col in PIPES:
                pipe[key].append(_col_max(rows, col))

        mm = glob.glob(os.path.join(ld, "Memory_*.csv"))
        if mm:
            rows = _rows(mm[0])
            mem["gm_read_kb"].append(_col_sum(rows, "read_main_memory_datas(KB)"))
            mem["gm_write_kb"].append(_col_sum(rows, "write_main_memory_datas(KB)"))

    out = {
        "launches": len(durations),
        "op_names": sorted(op_names),
        "block_dim": int(_med(blocks)),
        "task_duration_us": round(_med(durations), 4),
        "task_duration_min_us": round(min(durations), 4),
        "task_duration_max_us": round(max(durations), 4),
        "task_duration_all_us": [round(d, 4) for d in sorted(durations)],
    }
    for k, _ in PIPES:
        out[k + "_us"] = round(_med(pipe[k]), 4)
    r = _med(mem["gm_read_kb"])
    w = _med(mem["gm_write_kb"])
    out["gm_read_kb"] = round(r, 2)
    out["gm_write_kb"] = round(w, 2)
    traffic = (r + w) * 1024.0
    out["traffic_mb"] = round(traffic / 1e6, 4)
    out["achieved_gbps"] = (
        round(traffic / (out["task_duration_us"] * 1e-6) / 1e9, 2)
        if out["task_duration_us"] else None
    )
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
