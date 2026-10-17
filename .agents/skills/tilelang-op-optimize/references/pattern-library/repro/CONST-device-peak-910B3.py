"""Device peak-bandwidth probe (910B4 vs the 1.75 TB/s fabric reference).

Measures the maximum sustainable GM read / write / mixed bandwidth on this
device with maximally contiguous DMA (60 KB chunks, 40 AIV cores, one chunk per
loop iteration).  Purpose: decide whether the user's `achieved_bw / 1.75 TB/s >=
70%` line (1.225 TB/s of algorithmic traffic) is above or below the hardware
ceiling of this part.

  python peak.py --mode read --mb 64 --span 30720 --iters 5
  msprof op --kernel-name=main ... python peak.py --mode read --mb 64 ...
Entry: pattern-library/constants.md CONST-device-peak-910B3 (AIV-side GM bandwidth triple)
First verified: 2026-10-17, task max_pool2d-_max_pool2d_kernel-20261017T043853Z (Stage 4
  optimize; tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025 + torch 2.9.0+cpu /
  torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3, aicore 20 -> AIV 40 / UB 196608 B)
Assertions (measurement-type: no inline assert; msprof Task Duration median, magnitude and
relative relations only, absolute values drift with driver/residency):
  (a) pure contiguous read ceiling < 1 TB/s   -- measured 753.3 GB/s (256 MiB) / 778.7 GB/s (64 MiB);
  (b) pure write > pure read                  -- measured 1078-1087 GB/s (256 MiB, 247.1-249.0 us);
  (c) mixed (read+write) aggregate < read ceiling -- measured 694-703 GB/s (256 MiB each way).
  => a target of `achieved_bw / 1.75 TB/s >= 70%` (= 1.225 TB/s) is *not reachable* on this part.
  Commands: msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off
              --aic-metrics=PipeUtilization,Memory python <this> --mode read1 --mb 256 --span 61440 --stages 3
            (write: --mode write --mb 256 --span 61440 --stages 1; mixed: --mode rw --mb 256 --span 30720 --stages 2)
Provenance (task workspace, may lapse): examples/TileOPs/tileops/kernels/pool/max_pool2d/
  max_pool2d_kernel/perf_opt/{peak.py, run_peak2.py, peak_recheck.jsonl, logs/peak_*.log}.
Registration (ED-C): byte-identical copy of perf_opt/peak.py (md5 652a9d385fb5da200574cb444cd37d70)
  + this registration block appended inside the module docstring; code and behaviour unchanged.
  Budget: 103 lines (19 of them this block) / 4918 B -- exceeds the 100-line ED-B budget, not
  trimmed (PL-1.10 precedent).
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils

_DT = "float16"


@tilelang.jit(out_idx=[1], target="npuir")
def _peak(core_elems, span, mode, stages):
    cores = NPUUtils.get().get_aicore_num() * 2
    n_chunks = core_elems // span

    @T.prim_func
    def main(
        x: T.Tensor((cores, core_elems), _DT),
        out: T.Tensor((cores, core_elems), _DT),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            buf = T.alloc_ub((1, span), _DT)
            obuf = T.alloc_ub((1, span), _DT)
            if mode == "write":
                T.clear(obuf)
            for t in (T.serial(n_chunks) if stages == 1
                      else T.Pipelined(n_chunks, num_stages=stages)):
                if mode in ("read", "rw", "read1"):
                    T.copy(x[cid, t * span:(t + 1) * span], buf[0:1, 0:span])
                if mode == "read":
                    T.copy(buf[0:1, 0:span], obuf[0:1, 0:span])
                if mode in ("write", "rw"):
                    T.copy(obuf[0:1, 0:span], out[cid, t * span:(t + 1) * span])
                if mode == "read1":
                    # keep the chunk DMA alive with a single 2-byte GM write
                    T.copy(buf[0:1, 0:1], out[cid, t * span:t * span + 1])
            if mode == "read":
                T.copy(obuf[0:1, 0:1], out[cid, 0:1])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="read", choices=["read", "read1", "write", "rw"])
    ap.add_argument("--mb", type=int, default=64, help="total tensor MB")
    ap.add_argument("--span", type=int, default=30720, help="elements per chunk")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--stages", type=int, default=1)
    args = ap.parse_args()

    cores = NPUUtils.get().get_aicore_num() * 2
    total = args.mb * 1024 * 1024 // 2
    core_elems = total // cores
    core_elems = (core_elems // args.span) * args.span
    kernel = _peak(core_elems, args.span, args.mode, args.stages)
    x = torch.randn(cores, core_elems, dtype=torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    moved = cores * core_elems * 2
    print(f"[peak] mode={args.mode} stages={args.stages} span={args.span * 2}B per-launch "
          f"{moved / 1e6:.1f} MB read"
          f"{' + same write' if args.mode != 'read' else ''}; out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
