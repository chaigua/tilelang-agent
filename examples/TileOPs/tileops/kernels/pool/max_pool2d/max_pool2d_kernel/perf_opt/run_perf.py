"""One-workload-at-a-time perf measurement driver (Stage 4, session 6).

For a single workload id it does, in order:
  1. `bench.py --check`  (exact / max_diff==0 golden compare)  -> l0_pass
  2. `msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off`
     on the same workload -> median `Task Duration(us)` via prof.parse
  3. appends ONE row to perf_records.jsonl (append-only) immediately after the
     measurement, so a mid-batch kill loses nothing.

  python3 run_perf.py --case resnet-stem-float16 --phase baseline --round 1
  python3 run_perf.py --case resnet-stem-float16 --phase candidate --round 1

Existing (workload_id, phase, candidate_id) rows are skipped unless --force
(append-only file: never rewrite history).
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prof  # noqa: E402

LAUNCH = 20
WARM = 5
METRICS = "PipeUtilization,Memory"
IMPL = {"baseline": "../_max_pool2d_kernel.py",
        "candidate": "./_max_pool2d_kernel.py"}
ARTIFACT = {"baseline": "../_max_pool2d_kernel.py",
            "candidate": "./_max_pool2d_kernel.py"}
CAND_ID = {"baseline": "baseline", "candidate": "phase_v1"}
ROUND_DIR = "round2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already(rec_path, workload_id, phase, cand):
    if not os.path.exists(rec_path):
        return None
    with open(rec_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("workload_id") == workload_id and r.get("phase") == phase
                    and r.get("candidate_id") == cand):
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--phase", required=True, choices=["baseline", "candidate"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--slot", default="fp16")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rec_path = "perf_records.jsonl"
    cand = CAND_ID[args.phase]
    prev = already(rec_path, args.case, args.phase, cand)
    if prev and not args.force:
        print("[skip] already recorded: " + json.dumps(prev))
        return 0

    os.makedirs(os.path.join("logs", ROUND_DIR), exist_ok=True)
    log_path = os.path.join("logs", ROUND_DIR,
                            f"{args.case}__{args.phase}.log")
    logf = open(log_path, "w")
    logf.write("# " + " ".join(sys.argv) + "\n")
    logf.flush()

    # ---- 1. L0 (per-workload exactness) -----------------------------------
    chk = subprocess.run([sys.executable, "bench.py", "--impl", IMPL[args.phase],
                          "--case", args.case, "--iters", "1", "--check"],
                         capture_output=True, text=True)
    logf.write("\n===== bench.py --check (rc=%d) =====\n" % chk.returncode)
    logf.write(chk.stdout + chk.stderr)
    logf.flush()
    l0_pass = chk.returncode == 0 and "max_diff=0.000e+00" in chk.stdout
    print(f"[L0] {args.case} {args.phase}: rc={chk.returncode} "
          f"l0_pass={l0_pass} | {chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else ''}",
          flush=True)

    # ---- 2. msprof op -----------------------------------------------------
    od = os.path.join("profiles", args.phase, ROUND_DIR, args.case)
    os.makedirs(od, exist_ok=True)
    os.chmod(od, 0o700)
    cmd = ["msprof", "op", "--kernel-name=main", f"--output={od}",
           f"--launch-count={LAUNCH}", f"--warm-up={WARM}", "--dump=off",
           f"--aic-metrics={METRICS}", sys.executable, "bench.py",
           "--impl", IMPL[args.phase], "--case", args.case,
           "--iters", str(WARM + LAUNCH)]
    logf.write("\n===== " + " ".join(cmd) + " =====\n")
    logf.flush()
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    logf.write(pr.stdout + pr.stderr)
    logf.write(f"\n[elapsed] {time.time() - t0:.1f}s\n")
    logf.flush()
    logf.close()
    if pr.returncode != 0:
        print(f"[FAIL] {args.case} {args.phase}: msprof rc={pr.returncode} "
              f"(see {log_path})", flush=True)
        return 1
    r = prof.parse(od)
    if r["duration_us"] is None:
        print(f"[FAIL] {args.case} {args.phase}: no Task Duration rows", flush=True)
        return 1

    impl_abs = os.path.abspath(IMPL[args.phase])
    rec = {
        "round": args.round,
        "candidate_id": cand,
        "parent_id": None if args.phase == "baseline" else "baseline",
        "kernel_id": "_max_pool2d_kernel",
        "workload_id": args.case,
        "phase": args.phase,
        "artifact_path": ARTIFACT[args.phase],
        "artifact_sha256": sha256(impl_abs),
        "duration_us": float(r["duration_us"]),
        "l0_pass": bool(l0_pass),
        "msprof_raw_path": od,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(rec_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[rec] " + json.dumps(rec), flush=True)
    print(f"[prof] {args.case} {args.phase}: dur={r['duration_us']}us "
          f"(min {r['dur_min']} max {r['dur_max']} n={r['n']}) "
          f"mte2={r['mte2']} mte3={r['mte3']} vec={r['vec']} "
          f"blk={r['block_dim']} g2u={r['gm_to_ub_kb']}KB u2g={r['ub_to_gm_kb']}KB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
