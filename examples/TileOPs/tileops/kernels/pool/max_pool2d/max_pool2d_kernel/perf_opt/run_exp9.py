"""Round-9 A/B runner (Stage 4, session 9): one workload, one variant tag.

Runs `bench.py --check` (bit-exact L0) and then `msprof op` on the *same*
artifact, printing Task Duration and pipe times.  Writes NO row to
perf_records.jsonl -- the final rows are produced by `run_final.py` only after a
variant is adopted.

  python3 run_exp9.py --case resnet-stem-float16 --tag pipe1 \
         [--env MAXPOOL2D_PHASE_PIPE=1] [--env MAXPOOL2D_PHASE_BH=4 ...]
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prof  # noqa: E402

IMPL = "./_max_pool2d_kernel.py"
LAUNCH = 20
WARM = 5
METRICS = "PipeUtilization,Memory"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--round", default="round4")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("TILELANG_ASCEND_MODE", "Developer")
    for item in args.env:
        k, _, v = item.partition("=")
        env[k] = v
    envdesc = " ".join(args.env) or "(defaults)"

    logdir = os.path.join("logs", args.round)
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, f"{args.case}__{args.tag}.log")
    logf = open(log_path, "w")
    logf.write(f"# env: {envdesc}\n")
    logf.flush()

    chk = subprocess.run([sys.executable, "bench.py", "--impl", IMPL,
                          "--case", args.case, "--iters", "1", "--check"],
                         capture_output=True, text=True, env=env)
    exact = chk.returncode == 0 and "exact=True" in chk.stdout
    logf.write(chk.stdout + chk.stderr)
    logf.flush()
    print(f"[L0] {args.case} {args.tag}: exact={exact} | "
          f"{chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else ''}",
          flush=True)
    if not exact:
        logf.close()
        return 1

    od = os.path.join("profiles", args.round, args.tag, args.case)
    os.makedirs(od, exist_ok=True)
    os.chmod(od, 0o700)
    cmd = ["msprof", "op", "--kernel-name=main", f"--output={od}",
           f"--launch-count={LAUNCH}", f"--warm-up={WARM}", "--dump=off",
           f"--aic-metrics={METRICS}", sys.executable, "bench.py",
           "--impl", IMPL, "--case", args.case, "--iters", str(WARM + LAUNCH)]
    logf.write("\n===== " + " ".join(cmd) + " =====\n")
    logf.flush()
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    logf.write(pr.stdout + pr.stderr + f"\n[elapsed] {time.time() - t0:.1f}s\n")
    logf.close()
    if pr.returncode != 0:
        print(f"[FAIL] {args.case} {args.tag}: msprof rc={pr.returncode}", flush=True)
        return 1
    r = prof.parse(od)
    print(f"[{args.tag}] {args.case}: dur={r['duration_us']} "
          f"(min {r['dur_min']} max {r['dur_max']} n={r['n']}) "
          f"vec={r['vec']} mte2={r['mte2']} mte3={r['mte3']} "
          f"scalar={r['scalar']} blk={r['block_dim']} "
          f"g2u={r['gm_to_ub_kb']}KB u2g={r['ub_to_gm_kb']}KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
