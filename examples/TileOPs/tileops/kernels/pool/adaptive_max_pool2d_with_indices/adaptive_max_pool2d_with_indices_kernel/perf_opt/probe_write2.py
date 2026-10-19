"""Write-side isolation probe #2 (global-1x1 grid, ch=64).

Step-0 calibration + probe #1 showed a ~48 KB/block UB->GM term (~21x the real
4.5 KB dual-output write) on BOTH the fp16 value store and the int64 index
store, while the sibling single-output `_adaptive_max_pool2d_kernel` C4 path
stores the *same* `(real_c,1,1)` region in 0.786 us.  This probe isolates the
difference by removing candidate poisoners one at a time:

  p1  store source produced by a *contiguous* producer (T.vbrc -> T.vcast),
      no strided UB copy anywhere (max-pool-style store, minimal)
  p2  p1 + a strided UB copy `flat[:, :, 0] -> vstg2` (poisoner present but not
      on the store source)
  p3  store source produced BY the strided copy (step-0 value form)
  p4  p3 + the int64 index store (step-0 dual form)

All variants read the same whole-plane tile, so the read side is identical.

    python probe_write2.py
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
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401
from tilelang.utils.npu_utils import NPUUtils

OUT = "write_probe2.jsonl"
KERNEL_NAME = "main"
LAUNCH_COUNT = 20
WARM_UP = 5

N, C, H, W = 8, 2048, 7, 7
PLANE = H * W


def build(mode: str, ch: int = 64):
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, PLANE), "float16"),
            out: T.Tensor((N, C, 1, 1), "float16"),
            indices: T.Tensor((N, C, 1, 1), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), "float16")
                vstg32 = T.alloc_ub((ch, 1, 1), "float32")
                vstg = T.alloc_ub((ch, 1, 1), "float16")
                vstg2 = T.alloc_ub((ch, 1, 1), "float16")
                istg32 = T.alloc_ub((ch, 1, 1), "int32")
                istg = T.alloc_ub((ch, 1, 1), "int64")
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        real_c = T.min(ch, C - cb * ch)
                        # identical read side for every variant
                        T.copy(
                            x[nb, cb * ch : cb * ch + real_c, 0:1, 0:PLANE],
                            flat[0:real_c, 0:1, 0:PLANE],
                        )
                        if mode == "p1":
                            T.vbrc(T.float32(0.0), vstg32)
                            T.vcast(vstg32, vstg, round_mode="rint")
                        if mode == "p2":
                            T.vbrc(T.float32(0.0), vstg32)
                            T.vcast(vstg32, vstg, round_mode="rint")
                            # the strided UB gather, present but off the store
                            T.copy(
                                flat[0:real_c, 0:1, 0:1],
                                vstg2[0:real_c, 0:1, 0:1],
                            )
                        if mode in ("p3", "p4"):
                            T.copy(
                                flat[0:real_c, 0:1, 0:1],
                                vstg[0:real_c, 0:1, 0:1],
                            )
                        T.copy(
                            vstg[0:real_c, 0:1, 0:1],
                            out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                        )
                        if mode == "p4":
                            T.vcast(vstg, istg32, round_mode="rint")
                            T.vcast(istg32, istg, round_mode="rint")
                            T.copy(
                                istg[0:real_c, 0:1, 0:1],
                                indices[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                            )
                        if mode == "p2":
                            # keep the strided gather result live (no DCE)
                            T.copy(
                                vstg2[0:real_c, 0:1, 0:1],
                                out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                            )

        return main

    return _func()


def run_case(mode: str) -> dict:
    out_dir = os.path.join("profiles", "write_probe2", safe_component(f"p_{mode}"))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ArithmeticUtilization",
        sys.executable, "probe_write2.py", "--run", mode,
    ]
    log = os.path.join("logs", safe_component("wp2_" + mode) + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(f"[probe2] {mode}: msprof rc={proc.returncode}")
        return {"mode": mode, "error": "msprof_fail"}
    durs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        print(f"[probe2] {mode}: no OpBasicInfo rows")
        return {"mode": mode, "error": "invalid_profile"}
    eng = _parse_engines(out_dir)
    mem = {}
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "Memory_*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                for c in ("read_main_memory_datas(KB)", "write_main_memory_datas(KB)",
                          "GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)"):
                    v = row.get(c)
                    if v not in (None, "", "NA"):
                        mem.setdefault(c, []).append(float(v))
    mem = {k: round(statistics.median(v), 1) for k, v in mem.items()}
    return {
        "mode": mode,
        "latency_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "engines": eng,
        "mem": mem,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    if "--run" in sys.argv:
        mode = sys.argv[sys.argv.index("--run") + 1]
        k = build(mode)
        torch.manual_seed(0)
        x = torch.randn(N, C, 1, PLANE, dtype=torch.float16).npu()
        with torch.no_grad():
            for _ in range(25):
                out, idx = k(x)
        print(f"[probe2-run] mode={mode} out={tuple(out.shape)} idx={tuple(idx.shape)}")
        return

    rows = []
    for mode in ("p1", "p2", "p3", "p4"):
        r = run_case(mode)
        rows.append(r)
        if "error" in r:
            print(f"[probe2] {mode}: {r['error']}")
            continue
        e, m = r["engines"], r["mem"]
        print(
            f"[probe2] {mode}: Task={r['latency_us']:8.3f} us | aiv={e.get('aiv_time')} "
            f"vec={e.get('vec')} mte2={e.get('mte2')} mte3={e.get('mte3')} "
            f"scal={e.get('scalar')} | g2u={m.get('GM_to_UB_datas(KB)')} "
            f"u2g={m.get('UB_to_GM_datas(KB)')}"
        )
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nappended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
