"""Run the structural probes under ``msprof op`` and print a latency table."""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_bench import _parse_durations, _parse_pipe, REPO_ROOT, TILEOPS_DIR  # noqa: E402

AIC = "Roofline,BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,L2Cache,ResourceConflictRatio"
MSPROF_LOCK = "/tmp/tilelang-msprof.lock"
OUT_JSONL = os.path.join(HERE, "copy_floor_calibration.jsonl")

# (label, mode, M, N, tile, bm)
DEFAULT_PROBES = [
    ("3d_empty", "empty", 128, 16384, 8192, 1),
    ("3d_read", "read", 128, 16384, 8192, 1),
    ("3d_read_static", "read_static", 128, 16384, 8192, 1),
    ("3d_read_t16k", "read", 128, 16384, 16384, 1),
    ("32k_read_t16k", "read", 32, 32768, 16384, 1),
    ("3d_read_t4k", "read", 128, 16384, 4096, 1),
    ("32k_read_t4k", "read", 32, 32768, 4096, 1),
    ("3d_reduce", "reduce", 128, 16384, 8192, 1),
    ("3d_ind_red", "ind_red", 128, 16384, 8192, 1),
    ("3d_ind_vmin", "ind_vmin", 128, 16384, 8192, 1),
    ("3d_vmin_f16", "vmin_f16", 128, 16384, 8192, 1),
    ("32k_empty", "empty", 32, 32768, 8192, 1),
    ("32k_read", "read", 32, 32768, 8192, 1),
    ("32k_read_static", "read_static", 32, 32768, 8192, 1),
    ("32k_ind_red", "ind_red", 32, 32768, 8192, 1),
    ("32k_ind_vmin", "ind_vmin", 32, 32768, 8192, 1),
    ("4k_empty", "empty", 32, 4096, 4096, 2),
    ("4k_ind_red", "ind_red", 32, 4096, 4096, 2),
    ("4k_ind_vmin", "ind_vmin", 32, 4096, 4096, 2),
]


def run_probe(label, mode, M, N, tile, bm, launch_count, warm_up, tag):
    out_dir = os.path.join(HERE, "profiles", "probe", tag, label)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = ["flock", MSPROF_LOCK, "msprof", "op", "--kernel-name=main",
           "--output={}".format(out_dir),
           "--launch-count={}".format(launch_count),
           "--warm-up={}".format(warm_up),
           "--aic-metrics={}".format(AIC), "--dump=off",
           sys.executable, os.path.join(HERE, "probe_floor.py"),
           "--mode", mode, "--M", str(M), "--N", str(N), "--tile", str(tile),
           "--bm", str(bm), "--launches", str(warm_up + launch_count)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, TILEOPS_DIR, HERE, env.get("PYTHONPATH", "")])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    per_op = _parse_durations(out_dir)
    if "main" not in per_op:
        print("{}: FAILED rc={}\n{}\n{}".format(
            label, proc.returncode, proc.stdout[-1500:], proc.stderr[-1500:]))
        return None
    durs = sorted(per_op["main"])
    mean = sum(durs) / len(durs)
    pipe = _parse_pipe(out_dir).get("main", {})
    row = {"probe": label, "mode": mode, "M": M, "N": N, "tile": tile, "bm": bm,
           "mean_us": round(mean, 4), "median_us": round(durs[len(durs) // 2], 4),
           "min_us": round(durs[0], 4), "n": len(durs), "pipe": pipe,
           "out_dir": out_dir}
    print("{:<14s} {:>8.3f} us  median {:>7.3f}  vec={} mte2={} mte3={}".format(
        label, mean, row["median_us"], pipe.get("aiv_vec_ratio"),
        pipe.get("aiv_mte2_ratio"), pipe.get("aiv_mte3_ratio")), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma separated probe labels")
    ap.add_argument("--launch-count", type=int, default=30)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--tag", default="round1")
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else None
    rows = []
    for probe in DEFAULT_PROBES:
        if wanted and probe[0] not in wanted:
            continue
        row = run_probe(*probe, args.launch_count, args.warm_up, args.tag)
        if row:
            rows.append(row)
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n| probe | mean_us | median_us | vec | mte2 | mte3 |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        p = r["pipe"]
        print("| {} | {:.3f} | {:.3f} | {} | {} | {} |".format(
            r["probe"], r["mean_us"], r["median_us"], p.get("aiv_vec_ratio"),
            p.get("aiv_mte2_ratio"), p.get("aiv_mte3_ratio")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
