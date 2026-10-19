"""Probe: C4 global-pool via explicit C-axis `T.vadd` accumulation.

`T.reduce_sum` over the innermost W=7 axis is scalarised on this toolchain
(bisect s1->s2: 3.6us -> 133us, ~19us/unit).  Instead: transpose the (ch,H,W)
tile so the C axis (64/128/256 wide) becomes innermost, then accumulate the
49 spatial positions with wide `T.vadd` (PL-1.2 C-axis accumulation).

Variants:
  t1      : single transpose [2,1,0] (ch,H,W)->(W,H,ch), correctness check only
  a49_1   : t1 + 49 vadds of width ch
  a14_1   : t1 + 7 vadds of width H*ch + 7 vadds of width ch
  a49_2   : 2-transpose chain to (H,W,ch) + 49 vadds of width ch
  a14_2   : 2-transpose chain to (H,W,ch) + 7 vadds of width W*ch + 7 of width ch
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
DTYPE = "float16"


def build(variant: str, ch: int = 64):
    ch = min(C, ch)
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv_plane = 1.0 / float(H * W)
    two_step = variant.endswith("_2")
    is49 = variant.startswith("a49")
    # a14: after the transpose the outer spatial axis is contiguous W*ch
    # (two_step: (H,W,ch) -> collapse H; one_step: (W,H,ch) -> collapse W),
    # then the remaining axis is collapsed with ch-wide vadds.
    outer_axis = H if two_step else W
    inner_axis = W if two_step else H

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, H, W), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, H, W), DTYPE)
                step = T.alloc_ub((H, ch, W), DTYPE)
                # single-step target (W,H,ch) or two-step target (H,W,ch)
                hwc = T.alloc_ub((W, H, ch) if not two_step else (H, W, ch), DTYPE)
                hwc32 = T.alloc_ub((W, H, ch) if not two_step else (H, W, ch), "float32")
                acc_h = T.alloc_ub((1, inner_axis, ch), "float32")
                acc_c = T.alloc_ub((1, 1, ch), "float32")
                gt = T.alloc_ub((1, 1, ch), DTYPE)
                o1 = T.alloc_ub((1, ch, 1), DTYPE)
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat)
                        if two_step:
                            T.transpose(flat, step, permutation=[1, 0, 2])
                            T.transpose(step, hwc, permutation=[0, 2, 1])
                        else:
                            T.transpose(flat, hwc, permutation=[2, 1, 0])
                        if variant == "t1":
                            T.vcast(hwc[0:1, 0:1, 0:ch], acc_c, round_mode="rint")
                            T.vmul(acc_c, inv_plane, acc_c)
                            T.vcast(acc_c, gt, round_mode="rint")
                            T.transpose(gt, o1, permutation=[0, 2, 1])
                            T.transpose(o1, gout, permutation=[1, 0, 2])
                            T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                        else:
                            T.vcast(hwc, hwc32, round_mode="rint")
                            if is49:
                                T.clear(acc_c)
                                for s in T.serial(H * W):
                                    T.vadd(
                                        acc_c,
                                        hwc32[s // W : s // W + 1, s % W : s % W + 1, 0:ch],
                                        acc_c,
                                    )
                            else:  # a14: collapse the outer spatial axis (contiguous
                                # W*ch / H*ch block) then the inner one
                                T.clear(acc_h)
                                for s in T.serial(outer_axis):
                                    T.vadd(
                                        acc_h,
                                        hwc32[s : s + 1, 0:inner_axis, 0:ch],
                                        acc_h,
                                    )
                                T.clear(acc_c)
                                for s in T.serial(inner_axis):
                                    T.vadd(
                                        acc_c,
                                        acc_h[0:1, s : s + 1, 0:ch],
                                        acc_c,
                                    )
                            T.vmul(acc_c, inv_plane, acc_c)
                            T.vcast(acc_c, gt, round_mode="rint")
                            T.transpose(gt, o1, permutation=[0, 2, 1])
                            T.transpose(o1, gout, permutation=[1, 0, 2])
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
        print(f"[accum {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[accum {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
