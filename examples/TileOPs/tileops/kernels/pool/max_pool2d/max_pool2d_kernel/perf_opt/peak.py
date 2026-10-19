"""Device peak-bandwidth probe (910B4 vs the 1.75 TB/s fabric reference).

Measures the maximum sustainable GM read / write / mixed bandwidth on this
device with maximally contiguous DMA (60 KB chunks, 40 AIV cores, one chunk per
loop iteration).  Purpose: decide whether the user's `achieved_bw / 1.75 TB/s >=
70%` line (1.225 TB/s of algorithmic traffic) is above or below the hardware
ceiling of this part.

  python peak.py --mode read --mb 64 --span 30720 --iters 5
  msprof op --kernel-name=main ... python peak.py --mode read --mb 64 ...
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
