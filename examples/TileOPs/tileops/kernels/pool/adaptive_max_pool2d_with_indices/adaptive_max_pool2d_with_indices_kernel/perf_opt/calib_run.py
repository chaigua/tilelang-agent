"""Run the Step-0 copy-floor calibration through msprof op and persist results.

Writes structured rows to `copy_floor_calibration.jsonl` (NOT perf_records.jsonl,
which is reserved for kind=tune benchmark workloads) and prints the floor table
plus the 70% line (floor / 0.7).

    python calib_run.py

Row schema: {workload, variant, bytes, dma_granularity, latency_us,
achieved_bw, engines, rounds, timestamp, toolchain}
"""

import csv
import datetime
import glob
import json
import os
import statistics
import subprocess
import sys

from run_experiments import _parse_engines, safe_component

OUT = "copy_floor_calibration.jsonl"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5
TOOLCHAIN = (
    "tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23 (67a0441) "
    "+ CANN 9.0.0 + Ascend910B3 (npu-smi 26.1.1, get_aicore_num=20 -> 40 Vector cores) "
    "+ torch 2.9.0+cpu / torch_npu 2.9.0.post8"
)

# global-1x1 raster and spp-6x6 raster (mirror of workload_inventory.json).
GLOBAL = dict(n=8, c_in=2048, h_in=7, w_in=7)
GENERAL = dict(n=2, c_in=128, h_in=56, w_in=56, out_h=6, out_w=6)


def _bytes(case, cfg):
    n, c_in = cfg["n"], cfg["c_in"]
    if case == "global":
        n_in = n * c_in * cfg["h_in"] * cfg["w_in"]
        n_out = n * c_in
        return n_in * 2 + n_out * 2 + n_out * 8, "whole-plane ch*plane fp16 (1 DMA/unit)"
    n_in = n * c_in * cfg["h_in"] * cfg["w_in"]
    n_out = n * c_in * cfg["out_h"] * cfg["out_w"]
    return (
        n_in * 2 + n_out * 2 + n_out * 8,
        "per-bin row ch*w_in fp16 (1 DMA/row)",
    )


def run_case(case: str, ch: int, n: int | None = None) -> dict:
    cfg = dict(GLOBAL if case == "global" else GENERAL)
    if n is not None:
        cfg["n"] = n
    tag = f"{case}_ch{ch}" + (f"_n{n}" if n is not None else "")
    out_dir = os.path.join("profiles", "calibration", safe_component(tag))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ArithmeticUtilization",
        sys.executable, "calib_bench.py", "--case", case, "--ch", str(ch),
        "--iters", "25",
    ]
    if n is not None:
        cmd += ["--n", str(n)]
    log = os.path.join("logs", safe_component("calib_" + tag) + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(f"[calib] {tag}: msprof rc={proc.returncode} (see {log})")
        return None
    durs, blockdim = [], None
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
                    blockdim = row.get("Block Dim", blockdim)
    if not durs:
        print(f"[calib] {tag}: no OpBasicInfo rows (profile invalid; see {log})")
        return None
    eng = _parse_engines(out_dir)
    nbytes, dma = _bytes(case, cfg)
    lat = statistics.median(durs)
    return {
        "workload": "global-1x1" if case == "global" else "spp-6x6",
        "variant": tag,
        "bytes": nbytes,
        "dma_granularity": dma,
        "latency_us": round(lat, 3),
        "achieved_bw_gbs": round(nbytes / lat / 1000.0, 2),
        "engines": eng,
        "blockdim": blockdim,
        "n_launch": len(durs),
        "msprof_raw_path": out_dir,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toolchain": TOOLCHAIN,
    }


def main():
    rows = []
    # global: channel block -> per-core serial rounds (13 / 7 / 5 at 40 blocks).
    for ch in (32, 64, 96):
        rows.append(run_case("global", ch))
    # general: channel block -> wave/round count (1/1/2/3 rounds).
    for ch in (64, 48, 32, 16):
        rows.append(run_case("general", ch))
    for r in rows:
        if r is None:
            continue
        e = r["engines"]
        print(
            f"[calib] {r['variant']}: Task={r['latency_us']} us | bytes={r['bytes']} "
            f"bw={r['achieved_bw_gbs']} GB/s | BlockDim={r['blockdim']} "
            f"aiv={e.get('aiv_time')} vec={e.get('vec')} mte2={e.get('mte2')} "
            f"mte3={e.get('mte3')} scalar={e.get('scalar')} GM_to_UB={e.get('gm_to_ub_kb')}KB"
        )
        print(f"        70% line = floor / 0.7 = {r['latency_us'] / 0.7:.3f} us")
    with open(OUT, "a") as f:
        for r in rows:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nappended {len([r for r in rows if r])} rows to {OUT}")


if __name__ == "__main__":
    main()
