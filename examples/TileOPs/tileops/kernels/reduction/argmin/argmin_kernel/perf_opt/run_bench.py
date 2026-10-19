"""msprof-op measurement helper for the Stage 4 perf_opt workspace.

Single latency metric: ``msprof op`` ``Task Duration(us)`` (kernel-only).
``launch-count=20`` / ``warm-up=5``; the reported value is the mean of the
captured launches (also returns min/median/max and the per-op split, which
matters for the C6 dual-launch variant where the workload latency is the sum
of ``argreduce_partial`` + ``argreduce_merge``).

All msprof invocations are serialized with ``flock /tmp/tilelang-msprof.lock``
(msprof is exclusive on this box).

CLI:
    python run_bench.py --impl <file> --case <workload_id> \
        --out-dir profiles/<...> [--kernel-name main]
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
LAUNCH_COUNT = 20
WARM_UP = 5
AIC_METRICS = "Roofline"


def _parse_durations(output_dir):
    """Return {op_name: [durations_us]} from OpBasicInfo*.csv under *output_dir*."""
    csv_files = sorted(
        glob.glob(os.path.join(output_dir, "**", "OpBasicInfo*.csv"), recursive=True)
    )
    per_op = {}
    all_rows = []
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
                all_rows.append((name, dur))
    return per_op


def measure(impl, case, out_dir, kernel_name=None, launch_count=LAUNCH_COUNT,
            warm_up=WARM_UP, extra_args=(), log_path=None, include_ops=("main",),
            check=False):
    """Run one msprof-op measurement; returns a result dict.

    ``include_ops`` selects which captured op names contribute to the reported
    workload ``duration_us`` (the default ``main`` skips the Op-layer's
    host-side ``Cast_*`` prep kernel; the C6 dual-launch variant sums
    ``argreduce_partial`` + ``argreduce_merge``).
    """
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
    cmd += list(extra_args)
    flock_cmd = ["flock", MSPROF_LOCK] + cmd

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.getcwd(), os.path.dirname(HERE), env.get("PYTHONPATH", "")]
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
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
