"""Final measurement driver (Stage 4, session 8).

Measures the *frozen* `perf_opt/_max_pool2d_kernel.py` once per tune workload
and appends exactly one `phase="final"` row per workload to
`perf_records.jsonl` (append-only, restart-safe: an existing final row for the
same workload is skipped unless --force).

Per workload: `bench.py --check` (L0, bit-exact) -> `msprof op
--kernel-name=main --launch-count=20 --warm-up=5 --dump=off
--aic-metrics=PipeUtilization,Memory` -> median Task Duration(us).

  python3 run_final.py                  # all 9 tune workloads
  python3 run_final.py --check-smoke    # smoke cases, precision only
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prof  # noqa: E402
import run_sweep  # noqa: E402

IMPL = "./_max_pool2d_kernel.py"
REC = "perf_records.jsonl"
CANDIDATE = "merged_v3"
KERNEL_ID = "_max_pool2d_kernel"
LAUNCH = 20
WARM = 5
METRICS = "PipeUtilization,Memory"

TUNE = [
    "resnet-stem-float16", "resnet-stem-bfloat16", "resnet-stem-float32",
    "vgg-block-float16", "vgg-block-bfloat16", "vgg-block-float32",
    "alexnet-ceil-float16", "alexnet-ceil-bfloat16", "alexnet-ceil-float32",
]
SMOKE = [
    "smoke-1x8x16x16-k3s2p1-float16",
    "smoke-1x8x16x16-k3s2p1-bfloat16",
    "smoke-1x8x16x16-k3s2p1-float32",
]


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already(workload_id):
    if not os.path.exists(REC):
        return False
    for line in open(REC):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") == "final" and r.get("workload_id") == workload_id:
            return True
    return False


def check(case, iters="1"):
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "bench.py", "--impl", IMPL,
                        "--case", case, "--iters", iters, "--check"],
                       capture_output=True, text=True, env=env)
    exact = r.returncode == 0 and "exact=True" in r.stdout
    return exact, (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    dig = sha256(IMPL)

    if args.check_smoke:
        out = {}
        for case in SMOKE:
            exact, line, _ = check(case)
            out[case] = bool(exact)
            print(f"[smoke] {case}: exact={exact} | {line}", flush=True)
        with open("smoke_check.json", "w") as f:
            json.dump({"artifact_sha256": dig, "precision": out}, f, indent=1)
        return 0 if all(out.values()) else 1

    cases = [args.only] if args.only else TUNE
    for case in cases:
        if already(case) and not args.force:
            print(f"[skip] final row exists: {case}", flush=True)
            continue
        logdir = os.path.join("logs", "round4")
        os.makedirs(logdir, exist_ok=True)
        log_path = os.path.join(logdir, f"final__{case}.log")
        logf = open(log_path, "w")
        logf.write(f"# sha256={dig}\n")
        exact, line, _ = check(case)
        logf.write(f"[L0] exact={exact} | {line}\n")
        logf.flush()
        print(f"[L0] {case}: exact={exact} | {line}", flush=True)
        if not exact:
            logf.close()
            print(f"[FAIL] {case}: not bit-exact, not profiled", flush=True)
            return 1
        od = os.path.join("profiles", "final_v3", case)
        os.makedirs(od, exist_ok=True)
        os.chmod(od, 0o700)
        cmd = ["msprof", "op", "--kernel-name=main", f"--output={od}",
               f"--launch-count={LAUNCH}", f"--warm-up={WARM}", "--dump=off",
               f"--aic-metrics={METRICS}", sys.executable, "bench.py",
               "--impl", IMPL, "--case", case, "--iters", str(WARM + LAUNCH)]
        logf.write("\n===== " + " ".join(cmd) + " =====\n")
        logf.flush()
        t0 = time.time()
        pr = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ),
                            timeout=900)
        logf.write(pr.stdout + pr.stderr)
        logf.write(f"\n[elapsed] {time.time() - t0:.1f}s\n")
        logf.close()
        if pr.returncode != 0:
            print(f"[FAIL] {case}: msprof rc={pr.returncode} (see {log_path})",
                  flush=True)
            return 1
        r = prof.parse(od)
        if r["duration_us"] is None:
            print(f"[FAIL] {case}: no Task Duration rows (see {log_path})",
                  flush=True)
            return 1
        rec = {
            "round": 3,
            "candidate_id": CANDIDATE,
            "parent_id": "merged_v2",
            "kernel_id": KERNEL_ID,
            "workload_id": case,
            "phase": "final",
            "artifact_path": IMPL,
            "artifact_sha256": dig,
            "duration_us": float(r["duration_us"]),
            "l0_pass": True,
            "msprof_raw_path": od,
            "vec": r["vec"], "mte2": r["mte2"], "mte3": r["mte3"],
            "scalar": r["scalar"], "block_dim": r["block_dim"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rec.update(run_sweep.traffic(od))
        with open(REC, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print("[final] " + json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
