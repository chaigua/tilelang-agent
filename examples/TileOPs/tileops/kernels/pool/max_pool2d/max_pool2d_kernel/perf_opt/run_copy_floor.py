"""Runner for the step-0 copy-floor calibration (append-only jsonl).

For every variant of `copy_floor.py` it runs `msprof op` serially, parses the
kernel-only `Task Duration(us)` plus the AIV engine/memory counters, computes
the expected GM bytes of that variant and appends one line to
`copy_floor_calibration.jsonl` (a *separate* file from perf_records.jsonl:
calibration rows are not tune-workload measurements).
"""

import csv
import datetime
import glob
import json
import os
import statistics
import subprocess
import sys

from copy_floor import SHAPES, VARIANTS

OUT = "copy_floor_calibration.jsonl"
LAUNCH_COUNT = 20
WARM_UP = 5
METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
)
TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_bytes(variant):
    wkey, ch, rows, (in_per_out, in_rows), out_w, n_rows, kind = VARIANTS[variant]
    n, c, h, w = SHAPES[wkey]
    num_cblk = (c + ch - 1) // ch
    units = n * n_rows * num_cblk
    rd = 0 if kind == "write" else units * in_rows * ch * w * 2
    wr = units * ch * out_w * 2
    return rd, wr, units


def parse(out_dir):
    durs, blocks = [], None
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*",
                                             "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
                    blocks = row.get("Block Dim", blocks)
    eng = {"aiv_time": [], "vec": [], "mte2": [], "mte3": [], "scalar": []}
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*",
                                      "PipeUtilization_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                for key, col in (("aiv_time", "aiv_time(us)"),
                                 ("vec", "aiv_vec_time(us)"),
                                 ("mte2", "aiv_mte2_time(us)"),
                                 ("mte3", "aiv_mte3_time(us)"),
                                 ("scalar", "aiv_scalar_time(us)")):
                    v = row.get(col)
                    if v not in (None, "", "NA"):
                        eng[key].append(float(v))
    mem = {"gm_to_ub_kb": [], "ub_to_gm_kb": [], "mte2_bw": []}
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*",
                                      "Memory_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                for key, col in (("gm_to_ub_kb", "GM_to_UB_datas(KB)"),
                                 ("ub_to_gm_kb", "UB_to_GM_datas(KB)"),
                                 ("mte2_bw", "aiv_gm_to_ub_bw(GB/s)")):
                    v = row.get(col)
                    if v not in (None, "", "NA"):
                        mem[key].append(float(v))
    out = {"block_dim": blocks,
           "n": len(durs),
           "duration_us": round(statistics.median(durs), 3) if durs else None}
    for key, vals in list(eng.items()) + list(mem.items()):
        out[key] = round(statistics.median(vals), 3) if vals else None
    return out


def main():
    variants = sys.argv[1:] or sorted(VARIANTS)
    for variant in variants:
        out_dir = os.path.join("profiles", "copy_floor", variant)
        os.makedirs(out_dir, exist_ok=True)
        os.chmod(out_dir, 0o700)
        cmd = [
            "msprof", "op",
            "--kernel-name=main",
            f"--output={out_dir}",
            f"--launch-count={LAUNCH_COUNT}",
            f"--warm-up={WARM_UP}",
            "--dump=off",
            f"--aic-metrics={METRICS}",
            sys.executable, "copy_floor.py", "--case", variant,
            "--iters", str(WARM_UP + LAUNCH_COUNT),
        ]
        logdir = os.path.join("logs", "copy_floor")
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, variant + ".log"), "w") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            f.write(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            print(f"{variant}: msprof rc={proc.returncode} (see log)")
            continue
        res = parse(out_dir)
        rd, wr, units = expected_bytes(variant)
        dur = res["duration_us"]
        rec = {
            "variant": variant,
            "shape": SHAPES[VARIANTS[variant][0]],
            "dtype": "float16",
            "ch": VARIANTS[variant][1],
            "rows_per_unit": VARIANTS[variant][2],
            "in_rows_loaded": VARIANTS[variant][3][1],
            "out_w": VARIANTS[variant][4],
            "kind": VARIANTS[variant][6],
            "units": units,
            "expected_read_bytes": rd,
            "expected_write_bytes": wr,
            "duration_us": dur,
            "achieved_total_bw_gbs": round((rd + wr) / dur / 1e3, 1) if dur else None,
            "achieved_read_bw_gbs": round(rd / dur / 1e3, 1) if dur and rd else None,
            **res,
            "msprof_raw_path": out_dir,
            "timestamp": TS,
        }
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"{variant:10s} dur={dur}us read={rd/1e6:7.2f}MB "
            f"wr={wr/1e6:6.2f}MB tot_bw={rec['achieved_total_bw_gbs']}GB/s "
            f"rd_bw={rec['achieved_read_bw_gbs']} "
            f"mte2={res['mte2']} vec={res['vec']} scalar={res['scalar']} "
            f"GM_to_UB={res['gm_to_ub_kb']}KB blk={res['block_dim']}"
        )


if __name__ == "__main__":
    main()
