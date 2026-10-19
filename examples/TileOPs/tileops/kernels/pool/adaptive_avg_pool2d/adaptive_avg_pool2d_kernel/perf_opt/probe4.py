"""Probe 4: flattened spatial axis, no transpose (global pool).

The natural NCHW plane for one channel block is ch*H*W contiguous elements.
Declaring the GM tensor as (N, C, 1, H*W) keeps the copy contiguous AND gives a
wide innermost axis (49) so the reduce no longer runs over a 7-lane inner axis.

Variants:
  w1 : x(N,C,1,HW); f2 (ch,1,HW); copy; vcast; reduce dim=2 -> (ch,1,1); out
  w2 : same layout; serial vadd over HW into acc (ch,1,1)
  w3 : (ch,1,HW) transpose [0,2,1] -> (ch,HW,1); reduce dim=1 -> (ch,1,1)
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


def build(variant: str, ch: int = 64):
    ch = min(C, ch)
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv_plane = 1.0 / float(HW)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                f2 = T.alloc_ub((ch, 1, HW), DTYPE)
                f2_32 = T.alloc_ub((ch, 1, HW), "float32")
                f2t = T.alloc_ub((ch, HW, 1), "float32")
                m3 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], f2)
                        T.vcast(f2, f2_32, round_mode="rint")
                        if variant == "w1":
                            T.reduce_sum(f2_32, m3, dim=2)
                        elif variant == "w2":
                            T.clear(m3)
                            for s in T.serial(HW):
                                T.vadd(m3, f2_32[0:ch, 0:1, s : s + 1], m3)
                        else:  # w3
                            T.transpose(f2_32, f2t, permutation=[0, 2, 1])
                            T.reduce_sum(f2t, m3, dim=1)
                        T.vmul(m3, inv_plane, m3)
                        T.vcast(m3, gout, round_mode="rint")
                        T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])

        return main

    return _func(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    kernel = build(args.variant, args.ch)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        print(f"[p4 {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p4 {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
