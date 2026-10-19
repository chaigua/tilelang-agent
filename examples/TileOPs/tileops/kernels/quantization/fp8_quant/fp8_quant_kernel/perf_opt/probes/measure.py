"""Measure one (impl, case) under msprof op and print Task Duration + pipe mix.

Usage: python probes/measure.py --impl <kernel.py> --case <workload_id> [--check]
"""

import argparse
import csv
import glob
import os
import shlex
import statistics
import subprocess
import sys

METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory"


def parse_profile(out_dir):
    dur = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    dur.append(float(row["Task Duration(us)"]))
    info = {"duration_us": round(statistics.median(dur), 3) if dur else None, "n": len(dur)}
    pu = sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv")))
    if pu:
        with open(pu[0]) as f:
            rows = list(csv.DictReader(f))
        if rows:
            r = rows[0]
            for src, dst in [
                ("aiv_time(us)", "aiv_time"), ("aiv_vec_time(us)", "vec_time"),
                ("aiv_vec_ratio", "vec_ratio"), ("aiv_mte2_time(us)", "mte2_time"),
                ("aiv_mte3_time(us)", "mte3_time"), ("aiv_scalar_time(us)", "scalar_time"),
                ("aiv_scalar_ratio", "scalar_ratio"),
            ]:
                try:
                    info[dst] = round(float(r.get(src, "NA")), 4)
                except (TypeError, ValueError):
                    info[dst] = None
    au = sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "ArithmeticUtilization_*.csv")))
    if au:
        with open(au[0]) as f:
            rows = list(csv.DictReader(f))
        if rows:
            try:
                info["vec_fops"] = int(float(rows[0].get("aiv_vec_fops", "0")))
            except (TypeError, ValueError):
                info["vec_fops"] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--launch-count", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kernel-name", default="_fp8_quant_fwd_main")
    args = ap.parse_args()

    if args.check:
        p = subprocess.run([sys.executable, "bench.py", "--impl", args.impl,
                            "--case", args.case, "--iters", "2", "--check"],
                           capture_output=True, text=True, timeout=2400)
        for line in (p.stdout + p.stderr).splitlines():
            if "[check]" in line:
                print("CHECK " + line.split("]", 1)[1].strip())
        if p.returncode != 0:
            print(f"CHECK FAILED rc={p.returncode}")

    out_dir = args.out or os.path.join("profiles", "probe",
                                       f"{os.path.basename(args.impl).replace('.py','')}_{args.case}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["msprof", "op", f"--kernel-name={args.kernel_name}", f"--output={out_dir}",
           f"--launch-count={args.launch_count}", "--warm-up=3", "--dump=off",
           f"--aic-metrics={METRICS}", "/workspace/venv/bin/python", "bench.py",
           "--impl", args.impl, "--case", args.case, "--iters", str(args.iters)]
    log = os.path.join("logs", f"measure_{os.path.basename(out_dir)}.log")
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        f.write("# " + shlex.join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        f.write(p.stdout + p.stderr)
    if p.returncode != 0:
        print(f"msprof FAILED rc={p.returncode} log={log}")
        return 1
    info = parse_profile(out_dir)
    print(f"{os.path.basename(args.impl)} | {args.case} | " +
          " ".join(f"{k}={v}" for k, v in info.items()) + f" | raw={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
