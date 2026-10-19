"""Probe 6: flattened spatial axis (H*W=49 contiguous per channel), no 3D W=7 inner axis.

The natural plane is ch*H*W contiguous. Declaring the GM tensor as (N, C, 1, H*W)
(or (N, C, H*W)) keeps the input DMA contiguous and makes the innermost axis 49
wide instead of 7, so reductions no longer run over a 7-lane inner axis.

Variants (auto-multi-buffer off to keep UB within budget):
  f1 : (ch,1,HW)  vcast ; reduce dim=2 -> (ch,1,1)
  f3 : (ch,1,HW)  vcast ; transpose [0,2,1] -> (ch,HW,1) ; reduce dim=1 -> (ch,1,1)
  f4 : (1,ch,HW)  vcast ; transpose [0,2,1] -> (1,HW,ch) ; 49 vadds -> (1,1,ch) ; 2 out transposes
  f5 : (1,ch,HW)  vcast ; transpose [0,2,1] -> (1,HW,ch) ; reduce dim=1 -> (1,1,ch) ; 2 out transposes
  f6 : (ch,1,HW)  vcast ; copy+1-elem store (DMA floor control)
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


def build(variant: str, ch: int = 64, mb: bool = False):
    ch = min(C, ch)
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv_plane = 1.0 / float(HW)
    pc = {"npuir.enable_auto_multi_buffer": mb}
    lead_ch = variant in ("f4", "f5")  # (1, ch, HW) layout

    @tilelang.jit(out_idx=[1], target="npuir", pass_configs=pc)
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                fa = T.alloc_ub((1, ch, HW), DTYPE)
                fa32 = T.alloc_ub((1, ch, HW), "float32")
                fb = T.alloc_ub((ch, 1, HW), DTYPE)
                fb32 = T.alloc_ub((ch, 1, HW), "float32")
                ft = T.alloc_ub((1, HW, ch), "float32")
                ot = T.alloc_ub((ch, HW, 1), "float32")
                acc = T.alloc_ub((1, 1, ch), "float32")
                m3 = T.alloc_ub((ch, 1, 1), "float32")
                gt = T.alloc_ub((1, 1, ch), DTYPE)
                o1 = T.alloc_ub((1, ch, 1), DTYPE)
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        if lead_ch:
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], fa)
                            T.vcast(fa, fa32, round_mode="rint")
                            if variant == "f4":
                                T.transpose(fa32, ft, permutation=[0, 2, 1])
                                T.clear(acc)
                                for s in T.serial(HW):
                                    T.vadd(acc, ft[0:1, s : s + 1, 0:ch], acc)
                                T.vmul(acc, inv_plane, acc)
                                T.vcast(acc, gt, round_mode="rint")
                                T.transpose(gt, o1, permutation=[0, 2, 1])
                                T.transpose(o1, gout, permutation=[1, 0, 2])
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                            else:  # f5
                                T.transpose(fa32, ft, permutation=[0, 2, 1])
                                T.reduce_sum(ft, acc, dim=1)
                                T.vmul(acc, inv_plane, acc)
                                T.vcast(acc, gt, round_mode="rint")
                                T.transpose(gt, o1, permutation=[0, 2, 1])
                                T.transpose(o1, gout, permutation=[1, 0, 2])
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                        else:
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], fb)
                            T.vcast(fb, fb32, round_mode="rint")
                            if variant == "f6":
                                T.copy(fb32[0:1, 0:1, 0:1], m3[0:1, 0:1, 0:1])
                                T.vcast(m3, gout, round_mode="rint")
                                T.copy(gout[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                            elif variant == "f1":
                                T.reduce_sum(fb32, m3, dim=2)
                                T.vmul(m3, inv_plane, m3)
                                T.vcast(m3, gout, round_mode="rint")
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                            else:  # f3
                                T.transpose(fb32, ot, permutation=[0, 2, 1])
                                T.reduce_sum(ot, m3, dim=1)
                                T.vmul(m3, inv_plane, m3)
                                T.vcast(m3, gout, round_mode="rint")
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])

        return main

    return _func(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--mb", type=int, default=0)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    kernel = build(args.variant, args.ch, bool(args.mb))
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        print(f"[p6 {args.variant} mb{args.mb} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p6 {args.variant} mb{args.mb} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
