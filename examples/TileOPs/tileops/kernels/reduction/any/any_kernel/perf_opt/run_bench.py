"""msprof-op measurement helper for the AnyFwdOp Stage 4 perf_opt workspace.

Single latency metric: ``msprof op`` ``Task Duration(us)`` (kernel-only).
``launch-count`` / ``warm-up`` are explicit; the reported ``duration_us`` is the
**mean** of the captured launches, and the result also carries median / min /
max.

All msprof invocations are serialized with ``flock /tmp/tilelang-msprof.lock``
(msprof is exclusive on this box).

CLI:
    python run_bench.py --impl <file> --case <workload_id> \
        --out-dir profiles/<...> [--kernel-name main] [--probe read]
prints a JSON line with the measurement.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "bench_driver.py")
MSPROF_LOCK = "/tmp/tilelang-msprof.lock"
LAUNCH_COUNT = 30
WARM_UP = 5
AIC_METRICS = "Roofline,BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,L2Cache,ResourceConflictRatio"

# repo root: perf_opt -> all_kernel -> all -> reduction -> kernels -> tileops
# -> TileOPs -> examples -> <repo root>
REPO_ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 8))
TILEOPS_DIR = os.path.abspath(os.path.join(HERE, *[".."] * 7))


def _parse_durations(output_dir):
    """Return {op_name: [durations_us]} from OpBasicInfo*.csv under *output_dir*."""
    csv_files = sorted(
        glob.glob(os.path.join(output_dir, "**", "OpBasicInfo*.csv"), recursive=True)
    )
    per_op = {}
    for path in csv_files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                val = (row.get("Task Duration(us)") or "").strip()
                name = (row.get("Op Name") or "").strip() or "(unknown)"
                if not val:
                    continue
                try:
                    dur = float(val)
                except ValueError:
                    continue
                per_op.setdefault(name, []).append(dur)
    return per_op


def _parse_pipe(output_dir):
    """Best-effort per-op pipe utilization metrics (mean over block rows)."""
    out = {}
    for path in sorted(
        glob.glob(os.path.join(output_dir, "**", "PipeUtilization*.csv"), recursive=True)
    ):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            has_op_col = reader.fieldnames and "Op Name" in reader.fieldnames
            for row in reader:
                name = ((row.get("Op Name") or row.get("op_name") or "").strip()
                        if has_op_col else "main")
                if not name:
                    name = "main"
                keep = {}
                for key, val in row.items():
                    if key is None or key == "Op Name":
                        continue
                    try:
                        keep[key] = float(val)
                    except (TypeError, ValueError):
                        continue
                out.setdefault(name, []).append(keep)
    agg = {}
    for name, rows in out.items():
        keys = set()
        for r in rows:
            keys.update(r)
        agg[name] = {k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 4) for k in sorted(keys)}
    return agg


def measure(impl, case, out_dir, kernel_name=None, launch_count=LAUNCH_COUNT,
            warm_up=WARM_UP, extra_args=(), log_path=None, include_ops=("main",),
            check=False, probe=None, block_m=None):
    impl = os.path.abspath(impl)
    out_dir = os.path.abspath(out_dir)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)

    cmd = ["msprof", "op"]
    if kernel_name:
        cmd.append("--kernel-name={}".format(kernel_name))
    cmd += [
        "--output={}".format(out_dir),
        "--launch-count={}".format(launch_count),
        "--warm-up={}".format(warm_up),
        "--aic-metrics={}".format(AIC_METRICS),
        "--dump=off",
        sys.executable,
        DRIVER,
        "--impl",
        impl,
        "--case",
        case,
        "--launches",
        str(warm_up + launch_count),
    ]
    if check:
        cmd.append("--check")
    if probe:
        cmd += ["--probe", probe]
    if block_m is not None:
        cmd += ["--block-m", str(block_m)]
    cmd += list(extra_args)
    flock_cmd = ["flock", MSPROF_LOCK] + cmd

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, TILEOPS_DIR, os.path.dirname(HERE), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        flock_cmd, capture_output=True, text=True, timeout=1800, env=env
    )
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as handle:
            handle.write("CMD: {}\n\n--- stdout ---\n{}\n--- stderr ---\n{}\n".format(
                " ".join(flock_cmd), proc.stdout, proc.stderr))

    per_op = _parse_durations(out_dir)
    if not per_op:
        return {
            "ok": False,
            "case": case,
            "impl": impl,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    result = {
        "ok": True,
        "case": case,
        "probe": probe,
        "impl": impl,
        "impl_sha256": hashlib.sha256(open(impl, "rb").read()).hexdigest(),
        "kernel_name_filter": kernel_name,
        "launch_count": launch_count,
        "warm_up": warm_up,
        "out_dir": out_dir,
        "ops": {},
    }
    total_mean = 0.0
    for name, durs in sorted(per_op.items()):
        durs_sorted = sorted(durs)
        mean = sum(durs) / len(durs)
        result["ops"][name] = {
            "n": len(durs),
            "mean_us": mean,
            "min_us": durs_sorted[0],
            "median_us": durs_sorted[len(durs_sorted) // 2],
            "max_us": durs_sorted[-1],
        }
        if name in include_ops:
            total_mean += mean
    result["duration_us"] = total_mean
    result["total_mean_us"] = sum(o["mean_us"] for o in result["ops"].values())
    result["pipe"] = _parse_pipe(out_dir)
    result["stderr_tail"] = proc.stderr[-800:]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kernel-name", default=None)
    parser.add_argument("--launch-count", type=int, default=LAUNCH_COUNT)
    parser.add_argument("--warm-up", type=int, default=WARM_UP)
    parser.add_argument("--log", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--probe", default=None, choices=("empty", "copyout", "read", "read_f16", "copy"))
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--probe-shape", default=None)
    parser.add_argument(
        "--include-ops",
        default="main",
        help="comma-separated captured op names summed into duration_us",
    )
    args, _ = parser.parse_known_args()

    res = measure(
        args.impl,
        args.case,
        args.out_dir,
        kernel_name=args.kernel_name,
        launch_count=args.launch_count,
        warm_up=args.warm_up,
        log_path=args.log,
        include_ops=tuple(x for x in args.include_ops.split(",") if x),
        check=args.check,
        probe=args.probe,
        block_m=args.block_m,
        extra_args=("--probe-shape", args.probe_shape) if args.probe_shape else (),
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
