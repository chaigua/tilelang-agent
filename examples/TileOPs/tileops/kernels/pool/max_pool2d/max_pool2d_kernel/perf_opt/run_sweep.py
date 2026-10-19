"""Single-config (BH, CH) sweep driver for the phase path (Stage 4 session 7).

For ONE (case, bh, ch) override it does:
  1. resolve the *actually chosen* plan in-process (env is read at module
     import, so we import after setting MAXPOOL2D_PHASE_BH/CH) -> chosen bh/ch
  2. `bench.py --check`  -> exact / max_diff == 0   (profile only if exact)
  3. `msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off`
     -> median Task Duration(us) + pipe counters + per-launch GM traffic sums
  4. appends ONE line to `tuning_sweep.jsonl` immediately (append-only,
     restart-safe: an existing (case, bh, ch) row is skipped unless --force).

  python3 run_sweep.py --case resnet-stem-float16 --bh 8 --ch 32

`bh=0/ch=0` means "leave the auto planner alone".
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prof  # noqa: E402

IMPL = "./_max_pool2d_kernel.py"
REC = "tuning_sweep.jsonl"
LAUNCH = 20
WARM = 5
METRICS = "PipeUtilization,Memory"


def load_mod():
    path = os.path.abspath(IMPL)
    name = "_sweep_kernel_mod_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan_of(mod, case):
    import bench
    n, c, h, w, k, s, p, d, ceil_mode, dt = bench.CASES[case]
    pl = mod._plan(n, c, h, w, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1],
                   ceil_mode, dt, vector_cores=40)
    ph = pl["phase"] or {}
    return {"path": pl["path"], "chosen_bh": ph.get("bh"),
            "chosen_ch": pl["ch"], "num_logical": pl["num_logical"],
            "num_local": pl["num_local"], "out_h": pl["out_h"],
            "out_w": pl["out_w"], "num_cblk": pl["num_cblk"]}


def traffic(raw_dir):
    """Sum GM traffic over the blocks of the first launch (KB)."""
    fs = sorted(glob.glob(os.path.join(raw_dir, "OPPROF_*", "main", "0",
                                       "Memory_*.csv")))
    if not fs:
        return {}
    rows = list(csv.DictReader(open(fs[0])))
    out = {}
    for key, col in (("g2u_sum_kb", "GM_to_UB_datas(KB)"),
                     ("u2g_sum_kb", "UB_to_GM_datas(KB)"),
                     ("read_dram_sum_kb", "read_main_memory_datas(KB)"),
                     ("write_dram_sum_kb", "write_main_memory_datas(KB)")):
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "NA")]
        out[key] = round(sum(vals), 3) if vals else None
    out["n_blocks"] = len(rows)
    return out


def already(bh, ch, case):
    if not os.path.exists(REC):
        return None
    for line in open(REC):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("case") == case and r.get("bh") == bh and r.get("ch") == ch:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--bh", type=int, default=0)
    ap.add_argument("--ch", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    prev = already(args.bh, args.ch, args.case)
    if prev and not args.force:
        print("[skip] already swept: " + json.dumps(prev), flush=True)
        return 0

    if args.bh:
        os.environ["MAXPOOL2D_PHASE_BH"] = str(args.bh)
    if args.ch:
        os.environ["MAXPOOL2D_PHASE_CH"] = str(args.ch)
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

    mod = load_mod()
    pl = plan_of(mod, args.case)
    tag = f"{args.case}__bh{args.bh}_ch{args.ch}"
    logdir = os.path.join("logs", "sweep")
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, tag + ".log")
    logf = open(log_path, "w")
    logf.write("# " + " ".join(sys.argv) + "\n")
    logf.write("[plan] " + json.dumps(pl) + "\n")
    logf.flush()
    print(f"[plan] {tag} -> {json.dumps(pl)}", flush=True)

    if pl["path"] != "phase":
        print(f"[FAIL] {tag}: override fell back to path={pl['path']} "
              f"(no phase plan) -- not profiling", flush=True)
        logf.write("[FAIL] fallback path, aborted\n")
        logf.close()
        return 2

    rec = {"case": args.case, "bh": args.bh, "ch": args.ch}
    rec.update(pl)

    env = dict(os.environ)
    # ---- 1. exactness (L0) ------------------------------------------------
    chk = subprocess.run([sys.executable, "bench.py", "--impl", IMPL,
                          "--case", args.case, "--iters", "1", "--check"],
                         capture_output=True, text=True, env=env)
    logf.write("\n===== bench.py --check (rc=%d) =====\n" % chk.returncode)
    logf.write(chk.stdout + chk.stderr)
    logf.flush()
    exact = chk.returncode == 0 and "exact=True" in chk.stdout
    rec["exact"] = bool(exact)
    print(f"[L0] {tag}: rc={chk.returncode} exact={exact} | "
          f"{chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else ''}",
          flush=True)
    if not exact:
        rec.update({"duration_us": None, "msprof_raw_path": None,
                    "log": log_path,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(REC, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        logf.close()
        print(f"[FAIL] {tag}: not bit-exact, no profile", flush=True)
        return 1

    # ---- 2. msprof op -----------------------------------------------------
    od = os.path.join("profiles", "sweep", tag)
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
    logf.write(pr.stdout + pr.stderr)
    logf.write(f"\n[elapsed] {time.time() - t0:.1f}s\n")
    logf.close()
    if pr.returncode != 0:
        print(f"[FAIL] {tag}: msprof rc={pr.returncode} (see {log_path})", flush=True)
        return 1
    r = prof.parse(od)
    if r["duration_us"] is None:
        print(f"[FAIL] {tag}: no Task Duration rows", flush=True)
        return 1

    rec.update({
        "duration_us": float(r["duration_us"]),
        "vec": r["vec"], "mte2": r["mte2"], "mte3": r["mte3"],
        "scalar": r["scalar"], "block_dim": r["block_dim"],
        "ub_to_gm_kb": r["ub_to_gm_kb"], "gm_to_ub_kb": r["gm_to_ub_kb"],
        "msprof_raw_path": od, "log": log_path,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    rec.update(traffic(od))
    with open(REC, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print("[rec] " + json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
