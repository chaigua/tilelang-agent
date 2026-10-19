"""Probe 9: widen the C4 global-path DMA (fewer, larger transfers) while keeping
the reduction width at the value that fits UB.

Measured: ch=32 (one 6272 B DMA + one 32-wide reduce per serial iteration) ->
16.7us; ch=64 overflows UB in the reduce lowering.
Idea: DMA k channel blocks into one f16 tile, then run one g-wide reduce per
sub-block.  Same reduction width, 1/k the serial iterations and DMA latency
chains.  (Static extents; resnet C=2048 is divisible by g*k.)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401
from tilelang.utils.npu_utils import NPUUtils

N, C, H, W = 8, 2048, 7, 7
HW = H * W
DTYPE = "float16"


def build(g: int = 32, k: int = 2, stages: int = 0):
    dma_ch = min(g * k, C)
    k = max(1, dma_ch // g)
    num_cblk = (C + dma_ch - 1) // dma_ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv_plane = 1.0 / float(HW)
    use_pipe = stages > 0

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                tile = T.alloc_ub((dma_ch, 1, HW), DTYPE)
                sub = T.alloc_ub((g, 1, HW), "float32")
                sred = T.alloc_ub((g, 1, 1), "float32")
                gout = T.alloc_ub((g, 1, 1), DTYPE)

                if use_pipe:
                    for t in T.Pipelined(num_local, num_stages=stages):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            T.copy(x[nb, cb * dma_ch : cb * dma_ch + dma_ch, 0:1, 0:HW], tile)
                            for kk in T.serial(k):
                                T.vcast(tile[kk * g : kk * g + g, 0:1, 0:HW], sub, round_mode="rint")
                                T.reduce_sum(sub, sred, dim=2)
                                T.vmul(sred, inv_plane, sred)
                                T.vcast(sred, gout, round_mode="rint")
                                T.copy(
                                    gout,
                                    out[nb, cb * dma_ch + kk * g : cb * dma_ch + kk * g + g, 0:1, 0:1],
                                )
                else:
                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            T.copy(x[nb, cb * dma_ch : cb * dma_ch + dma_ch, 0:1, 0:HW], tile)
                            for kk in T.serial(k):
                                T.vcast(tile[kk * g : kk * g + g, 0:1, 0:HW], sub, round_mode="rint")
                                T.reduce_sum(sub, sred, dim=2)
                                T.vmul(sred, inv_plane, sred)
                                T.vcast(sred, gout, round_mode="rint")
                                T.copy(
                                    gout,
                                    out[nb, cb * dma_ch + kk * g : cb * dma_ch + kk * g + g, 0:1, 0:1],
                                )

        return main

    return _func(dma_ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--stages", type=int, default=0)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    kernel = build(args.g, args.k, args.stages)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        print(f"[p9 g{args.g} k{args.k} s{args.stages}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p9 g{args.g} k{args.k}] ok")


if __name__ == "__main__":
    main()
