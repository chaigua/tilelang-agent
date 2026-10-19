"""Probe 8: pipelining the flattened C4 global path."""

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


def build(variant: str, ch: int = 32, stages: int = 2):
    ch = min(C, ch)
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv_plane = 1.0 / float(HW)
    use_pipe = variant == "pipe"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                tile = T.alloc_ub((ch, 1, HW), DTYPE)
                tile32 = T.alloc_ub((ch, 1, HW), "float32")
                sred = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                if use_pipe:
                    for t in T.Pipelined(num_local, num_stages=stages):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            real_c = T.min(ch, C - cb * ch)
                            T.copy(x[nb, cb * ch : cb * ch + real_c, 0:1, 0:HW], tile[0:real_c, 0:1, 0:HW])
                            T.vcast(tile, tile32, round_mode="rint")
                            T.reduce_sum(tile32, sred, dim=2)
                            T.vmul(sred, inv_plane, sred)
                            T.vcast(sred, gout, round_mode="rint")
                            T.copy(gout[0:real_c, 0:1, 0:1], out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1])
                else:
                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            real_c = T.min(ch, C - cb * ch)
                            T.copy(x[nb, cb * ch : cb * ch + real_c, 0:1, 0:HW], tile[0:real_c, 0:1, 0:HW])
                            T.vcast(tile, tile32, round_mode="rint")
                            T.reduce_sum(tile32, sred, dim=2)
                            T.vmul(sred, inv_plane, sred)
                            T.vcast(sred, gout, round_mode="rint")
                            T.copy(gout[0:real_c, 0:1, 0:1], out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1])

        return main

    return _func(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="serial")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    kernel = build(args.variant, args.ch, args.stages)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        print(f"[p8 {args.variant} ch{args.ch} s{args.stages}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p8 {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
