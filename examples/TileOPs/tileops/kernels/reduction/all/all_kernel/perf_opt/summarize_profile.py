"""Summarise msprof PipeUtilization / Memory metrics for a profile directory.

Used by the Stage 4 log tables (candidate-vs-current-best: Task Duration,
vector / MTE2 / MTE3 pipe ratios, GM traffic, UB read/write bandwidth).

CLI: python summarize_profile.py <dir> [<dir> ...]   (one JSON line per dir)
"""

import csv
import glob
import json
import os
import statistics
import sys

PIPE_COLS = [
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_scalar_time(us)",
    "aiv_mte2_active_bw(GB/s)",
]
MEM_COLS = [
    "GM_to_UB_datas(KB)",
    "UB_to_GM_datas(KB)",
    "read_main_memory_datas(KB)",
    "write_main_memory_datas(KB)",
    "aiv_gm_to_ub_bw(GB/s)",
    "aiv_ub_to_gm_bw(GB/s)",
]


def _mean(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "NA", "")]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(d, launch=10):
    out = {"dir": d}
    info = sorted(glob.glob(os.path.join(d, "OPPROF_*", "main", str(launch),
                                         "OpBasicInfo_*.csv")))
    if not info:
        return {"dir": d, "error": "no capture"}
    rows = list(csv.DictReader(open(info[0])))
    out["task_duration_us"] = float(rows[0]["Task Duration(us)"])
    out["block_dim"] = rows[0]["Block Dim"]
    pipe = sorted(glob.glob(os.path.join(d, "OPPROF_*", "main", str(launch),
                                         "PipeUtilization_*.csv")))
    if pipe:
        prows = list(csv.DictReader(open(pipe[0])))
        for col in PIPE_COLS:
            out[col] = _mean(prows, col)
    mem = sorted(glob.glob(os.path.join(d, "OPPROF_*", "main", str(launch),
                                        "Memory_*.csv")))
    if mem:
        mrows = list(csv.DictReader(open(mem[0])))
        for col in MEM_COLS:
            out[col] = _mean(mrows, col)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d)))
