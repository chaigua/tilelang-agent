"""Write-side isolation probe #3 (global-1x1 grid, ch=64): int64 store path.

probe_write2 `p1` proved the fp16 value store form is cheap (4.97 us, u2g
0.8 KB).  The dual-output step-0 form was ~105 us with u2g 96 KB/block.  This
probe adds exactly one thing at a time to p1:

  p1  fp16 value store only (control)
  p6  p1 + int64 index store from an uninitialised int64 buffer
  p7  p1 + int64 index store from vbrc(f32) -> vcast(int32) -> vcast(int64)
  p8  p1 + fp16 write to a *second* GM tensor (same shape) -- control for
      "two output tensors" without int64

    python probe_write3.py
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

OUT = "write_probe3.jsonl"
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
                vstg_b = T.alloc_ub((ch, 1, 1), "float16")
                istg32 = T.alloc_ub((ch, 1, 1), "int32")
                istg = T.alloc_ub((ch, 1, 1), "int64")
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(
                            x[nb, cb * ch : cb * ch + real_c, 0:1, 0:PLANE],
                            flat[0:real_c, 0:1, 0:PLANE],
                        )
                        T.vbrc(T.float32(0.0), vstg32)
                        T.vcast(vstg32, vstg, round_mode="rint")
                        if mode == "p8":
                            T.vcast(vstg32, vstg_b, round_mode="rint")
                        if mode == "p7":
                            T.vcast(vstg32, istg32, round_mode="rint")
                            T.vcast(istg32, istg, round_mode="rint")
                        # value store (identical in every variant)
                        T.copy(
                            vstg[0:real_c, 0:1, 0:1],
                            out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                        )
                        if mode in ("p6", "p7"):
                            T.copy(
                                istg[0:real_c, 0:1, 0:1],
                                indices[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                            )
                        if mode == "p8":
                            T.copy(
                                vstg_b[0:real_c, 0:1, 0:1],
                                indices[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                            )

        return main

    return _func()


def run_case(mode: str) -> dict:
    out_dir = os.path.join("profiles", "write_probe3", safe_component(f"q_{mode}"))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", f"--kernel-name=main", f"--output={out_dir}",
        f"--launch-count=20", f"--warm-up=5", "--dump=off",
        "--aic-metrics=BasicInfo,PipeUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ArithmeticUtilization",
        sys.executable, "probe_write3.py", "--run", mode,
    ]
    log = os.path.join("logs", safe_component("wp3_" + mode) + ".log")
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(f"[probe3] {mode}: msprof rc={proc.returncode}")
        return {"mode": mode, "error": "msprof_fail"}
    durs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        print(f"[probe3] {mode}: no OpBasicInfo rows")
        return {"mode": mode, "error": "invalid_profile"}
    eng = _parse_engines(out_dir)
    for key, col in (("mte3_instr", "aiv_mte3_instructions"), ("mte2_instr", "aiv_mte2_instructions")):
        vals = []
        for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "Memory_*.csv")):
            with open(path) as f:
                for row in csv.DictReader(f):
                    v = row.get(col)
                    if v not in (None, "", "NA"):
                        vals.append(float(v))
        if vals:
            eng[key] = round(statistics.median(vals), 1)
    return {
        "mode": mode,
        "latency_us": round(statistics.median(durs), 3),
        "engines": eng,
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
        print(f"[probe3-run] mode={mode}")
        return

    rows = []
    for mode in ("p1", "p6", "p7", "p8"):
        r = run_case(mode)
        rows.append(r)
        if "error" in r:
            print(f"[probe3] {mode}: {r['error']}")
            continue
        e = r["engines"]
        print(
            f"[probe3] {mode}: Task={r['latency_us']:8.3f} us | aiv={e.get('aiv_time')} "
            f"vec={e.get('vec')} mte2={e.get('mte2')} mte3={e.get('mte3')} "
            f"scalar={e.get('scalar')} | mte3_instr={e.get('mte3_instr')} "
            f"mte2_instr={e.get('mte2_instr')} g2u={e.get('gm_to_ub_kb')}"
        )
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nappended {len([r for r in rows if 'error' not in r])} rows to {OUT}")


if __name__ == "__main__":
    main()
