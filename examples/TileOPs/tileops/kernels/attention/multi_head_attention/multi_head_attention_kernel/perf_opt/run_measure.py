#!/usr/bin/env python3
"""Stage 4 batch measurement driver (kernel-only, `msprof op`).

For each requested benchmark case it runs, **serially** (exclusive device):

    msprof op --kernel-name=_gqa_bwd_main --launch-count N --warm-up W \
        --aic-metrics=BasicInfo,PipeUtilization,Memory \
        python bench.py --impl <impl> --case <label>/<dtype> [config knobs]

then parses the raw profile (read_profile.py) and appends one line to
`perf_records.jsonl` (append-only) unless `--no-record` is given.

Examples:
    # baseline sweep, all 8 tune cases
    python run_measure.py --impl ../_gqa_bwd_kernel.py --phase baseline \
        --candidate-id stage3-default --round 0

    # candidate branch, two cases only, extra module knobs
    python run_measure.py --impl _gqa_bwd_kernel_round1_causal_skip.py \
        --phase candidate --candidate-id r1-causal-skip --parent-id stage3-default \
        --round 1 --cases llama-3.1-8b-short/fp16 --mod-attr split_mode=1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench.py")
RECORDS = os.path.join(HERE, "perf_records.jsonl")
PY = "/workspace/venv/bin/python"
KERNEL_NAME = "_gqa_bwd_main"

sys.path.insert(0, HERE)
from read_profile import parse as parse_profile  # noqa: E402

TUNE_CASES = [
    "llama-3.1-8b-short/fp16",
    "llama-3.1-8b-short/bf16",
    "llama-3.1-8b-long/fp16",
    "llama-3.1-8b-long/bf16",
    "llama-3.1-70b-short/fp16",
    "llama-3.1-70b-short/bf16",
    "llama-3.1-70b-long/fp16",
    "llama-3.1-70b-long/bf16",
]
SMOKE_CASES = [
    "mha-bwd-smoke-s512-h8-d64/fp16",
    "mha-bwd-smoke-s512-h8-d64/bf16",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_case(args, case, impl_abs, impl_sha, round_no, tag, l0_pass=None):
    label, _, dtype = case.partition("/")
    out_dir = os.path.join(HERE, "profiles", tag, f"{label}_{dtype}")
    log_dir = os.path.join(HERE, "logs", tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    log_path = os.path.join(log_dir, f"{label}_{dtype}.log")

    cmd = [
        "msprof", "op",
        f"--kernel-name={KERNEL_NAME}",
        f"--output={out_dir}",
        f"--launch-count={args.launch_count}",
        f"--warm-up={args.warm_up}",
        "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory",
        PY, BENCH,
        "--impl", impl_abs,
        "--case", case,
        "--warm-up", str(args.warm_up),
        "--launch-count", str(args.launch_count),
        "--extra", str(args.extra),
        "--print-config",
    ]
    if args.bm:
        cmd += ["--bm", str(args.bm)]
    if args.bn:
        cmd += ["--bn", str(args.bn)]
    if args.ns:
        cmd += ["--ns", str(args.ns)]
    for kv in args.mod_attr:
        cmd += ["--mod-attr", kv]

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        "/workspace/tilelang-mlir-ascend:/workspace/tilelang-mlir-ascend/examples/TileOPs"
    )
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    row = {
        "round": round_no,
        "candidate_id": args.candidate_id,
        "parent_id": args.parent_id,
        "kernel_id": "_gqa_bwd_main",
        "workload_id": case,
        "phase": args.phase,
        "artifact_path": os.path.relpath(impl_abs, HERE),
        "artifact_sha256": impl_sha,
        "duration_us": None,
        "l0_pass": l0_pass,
        "msprof_raw_path": os.path.relpath(out_dir, HERE),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": args.config_desc,
        "wall_s": round(elapsed, 1),
        "status": "invalid",
    }
    if proc.returncode == 0:
        try:
            prof = parse_profile(out_dir)
            row["duration_us"] = prof["task_duration_us"]
            row["status"] = "valid"
            row["pipe"] = {
                k: prof[k + "_us"]
                for k in ("aic_cube", "aic_scalar", "aiv_vec", "aiv_mte2",
                          "aiv_mte3", "aiv_scalar")
                if k + "_us" in prof
            }
            row["traffic_mb"] = prof.get("traffic_mb")
            row["op_name"] = ",".join(prof.get("op_names", []))
        except SystemExit as exc:
            row["error"] = str(exc)
    else:
        row["error"] = f"msprof rc={proc.returncode} (see {os.path.relpath(log_path, HERE)})"

    if not args.no_record:
        with open(RECORDS, "a") as fh:
            fh.write(json.dumps(row, sort_keys=False) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "candidate", "merged", "final", "probe"])
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="explicit case list (default: the 8 tune cases)")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--bm", type=int, default=0)
    ap.add_argument("--bn", type=int, default=0)
    ap.add_argument("--ns", type=int, default=0)
    ap.add_argument("--mod-attr", action="append", default=[])
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--extra", type=int, default=5)
    ap.add_argument("--tag", default=None, help="profile/log subdirectory name")
    ap.add_argument("--no-record", action="store_true",
                    help="probe mode: do not append to perf_records.jsonl")
    ap.add_argument("--l0-pass", dest="l0_pass", action="store_true", default=None)
    ap.add_argument("--l0-fail", dest="l0_pass", action="store_false")
    args = ap.parse_args()

    impl_abs = os.path.abspath(args.impl)
    impl_sha = sha256(impl_abs)
    cases = args.cases or list(TUNE_CASES)
    if args.include_smoke:
        cases = cases + SMOKE_CASES
    cfg = {"block_m": args.bm or "default", "block_n": args.bn or "default",
           "num_stages": args.ns or "default"}
    args.config_desc = dict(cfg, mod_attr=args.mod_attr)
    tag = args.tag or f"{args.phase}_{args.candidate_id}"

    print(f"[measure] impl={os.path.basename(impl_abs)} sha={impl_sha[:12]} "
          f"phase={args.phase} cand={args.candidate_id} cases={len(cases)}")
    rows = []
    for case in cases:
        row = run_case(args, case, impl_abs, impl_sha, args.round, tag,
                       l0_pass=args.l0_pass)
        rows.append(row)
        status = row["status"]
        dur = row["duration_us"]
        print(f"  {case:28s} {status:8s} "
              f"{'' if dur is None else f'{dur:10.1f} us'}  {row.get('error', '')}")

    print("\n[label/dtype                  duration_us   aic_cube  aiv_vec  "
          "aiv_mte2  aiv_mte3  aiv_scalar  traffic_mb]")
    for row in rows:
        p = row.get("pipe", {})
        print(f"{row['workload_id']:28s} {str(row['duration_us']):>12s} "
              + " ".join(f"{p.get(k, float('nan')):8.1f}" for k in
                         ("aic_cube", "aiv_vec", "aiv_mte2", "aiv_mte3", "aiv_scalar"))
              + f" {str(row.get('traffic_mb')):>10s}")
    valid = [r for r in rows if r["status"] == "valid"]
    print(f"\n[measure] valid {len(valid)}/{len(rows)}; records -> "
          f"{'NOT written' if args.no_record else RECORDS}")


if __name__ == "__main__":
    main()
