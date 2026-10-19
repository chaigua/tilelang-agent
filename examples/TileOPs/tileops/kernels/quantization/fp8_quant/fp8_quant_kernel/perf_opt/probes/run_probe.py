"""Probe runner: msprof op over varied PROBE_STAGE / PROBE_* variants.

Usage:
  python probes/run_probe.py --impl probes/probe_bisect.py \
      --stages s0_dma,s1_cast,... --case kv-index-8k-d64-fp16
"""

import argparse
import csv
import glob
import importlib.util
import os
import shlex
import statistics
import subprocess
import sys

METRICS = "BasicInfo,PipeUtilization,ArithmeticUtilization"


def load_kernel_module(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_profile(out_dir):
    dur, scalar_ratio, vec_ratio, aiv_time, vec_time, scalar_time = [], None, None, None, None, None
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    dur.append(float(row["Task Duration(us)"]))
    pu = sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "PipeUtilization_*.csv")))
    if pu:
        rows = []
        with open(pu[0]) as f:
            rows = list(csv.DictReader(f))
        if rows:
            r = rows[0]
            def g(k):
                v = r.get(k, "NA")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            aiv_time = g("aiv_time(us)")
            vec_time = g("aiv_vec_time(us)")
            scalar_time = g("aiv_scalar_time(us)")
            scalar_ratio = g("aiv_scalar_ratio")
            vec_ratio = g("aiv_vec_ratio")
    return {
        "duration_us": round(statistics.median(dur), 3) if dur else None,
        "scalar_ratio": scalar_ratio,
        "vec_ratio": vec_ratio,
        "aiv_time": aiv_time,
        "vec_time": vec_time,
        "scalar_time": scalar_time,
        "n": len(dur),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--stages", required=True, help="comma separated PROBE_STAGE values")
    ap.add_argument("--case", required=True)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--launch-count", type=int, default=8)
    ap.add_argument("--probe-env", default="", help="extra k=v;k=v env pairs")
    ap.add_argument("--out-root", default="profiles/probe")
    ap.add_argument("--kernel-name", default="_fp8_quant_fwd_main")
    args = ap.parse_args()

    extra_env = dict(kv.split("=", 1) for kv in args.probe_env.split(";") if kv)

    print(f"| stage | Task Duration(us) | aiv_time | vec_time | scalar_time | vec_ratio | scalar_ratio | n |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage in args.stages.split(","):
        stage = stage.strip()
        if not stage:
            continue
        out_dir = os.path.join(args.out_root, f"{stage}_{args.case}")
        os.makedirs(out_dir, exist_ok=True)
        env = dict(os.environ)
        env["PROBE_STAGE"] = stage
        env.update(extra_env)
        cmd = [
            "msprof", "op",
            f"--kernel-name={args.kernel_name}",
            f"--output={out_dir}",
            f"--launch-count={args.launch_count}",
            "--warm-up=3",
            "--dump=off",
            f"--aic-metrics={METRICS}",
            "/workspace/venv/bin/python", "bench.py",
            "--impl", args.impl,
            "--case", args.case,
            "--iters", str(args.iters),
        ]
        log = os.path.join("logs", f"probe_{stage}_{args.case}.log")
        os.makedirs("logs", exist_ok=True)
        with open(log, "w") as f:
            f.write(f"# {shlex.join(cmd)}\n# env PROBE_STAGE={stage} {extra_env}\n")
            f.flush()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=2400)
            f.write(p.stdout + p.stderr)
        if p.returncode != 0:
            print(f"| {stage} | FAIL rc={p.returncode} | - | - | - | - | - | - |")
            continue
        r = parse_profile(out_dir)
        print(f"| {stage} | {r['duration_us']} | {r['aiv_time']} | {r['vec_time']} | "
              f"{r['scalar_time']} | {r['vec_ratio']} | {r['scalar_ratio']} | {r['n']} |")


if __name__ == "__main__":
    sys.exit(main())
