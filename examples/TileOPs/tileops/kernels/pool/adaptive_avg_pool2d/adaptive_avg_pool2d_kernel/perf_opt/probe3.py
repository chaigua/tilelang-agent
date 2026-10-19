"""Probe 3: escape the transposed-DMA-layout penalty for the global pool.

Measured (resnet-global 8x2048x7x7->1x1, ch=64):
  copy + 1-elem store                      : 3.47us, GM_to_UB 36.75 KB/block
  copy + vcast + 1-elem store              : 3.56us, GM_to_UB 36.75 KB/block
  copy + transpose + vmul + 1-elem store   : 119.6us, GM_to_UB 336 KB/block
=> layout inference gives the UB buffer a transposed layout, which turns the
   GM->UB DMA into a 16 B/element strided gather (8x sector amplification).

Variants:
  v3a : declare x as (N,C,H*W); 2D copy; vcast; reduce dim=1 (inner, wide)
  v3b : declare x as (N,C,H*W); 2D copy; vcast; transpose [1,0]; reduce dim=0
  v3c : declare x as (N,C,H*W); 2D copy; vcast; transpose [1,0]; serial vadd
  v3e : 3D copy; vcast; UB->UB identity copy (layout firewall); transpose; vmul
  v3f : 3D copy; vcast; UB->UB identity copy; reduce dim=2 (firewall + reduce)
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
    flat2d = variant in ("v3a", "v3b", "v3c")

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, HW) if flat2d else (N, C, H, W), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                f2 = T.alloc_ub((ch, HW), DTYPE)
                f2_32 = T.alloc_ub((ch, HW), "float32")
                t2 = T.alloc_ub((HW, ch), "float32")
                m2 = T.alloc_ub((ch, 1), "float32")
                a2 = T.alloc_ub((1, ch), "float32")
                o2 = T.alloc_ub((1, ch), DTYPE)
                o1 = T.alloc_ub((1, ch, 1), DTYPE)
                gout = T.alloc_ub((ch, 1, 1), DTYPE)
                flat = T.alloc_ub((ch, H, W), DTYPE)
                flat32 = T.alloc_ub((ch, H, W), "float32")
                tmp = T.alloc_ub((ch, H, W), "float32")
                tmp3 = T.alloc_ub((ch, H, 1), "float32")
                t1 = T.alloc_ub((W, H, ch), DTYPE)
                t1_32 = T.alloc_ub((W, H, ch), "float32")

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        if flat2d:
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:HW], f2)
                            T.vcast(f2, f2_32, round_mode="rint")
                            if variant == "v3a":
                                T.reduce_sum(f2_32, m2, dim=1)
                                T.vcast(m2[0:1, 0:1], o2[0:1, 0:1], round_mode="rint")
                                T.transpose(o2, o1, permutation=[0, 2, 1])
                                T.transpose(o1, gout, permutation=[1, 0, 2])
                                T.vmul(gout, inv_plane, gout)
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                            else:
                                T.transpose(f2_32, t2, permutation=[1, 0])
                                if variant == "v3b":
                                    T.reduce_sum(t2, a2, dim=0)
                                else:
                                    T.clear(a2)
                                    for s in T.serial(HW):
                                        T.vadd(a2, t2[s : s + 1, 0:ch], a2)
                                T.vmul(a2, inv_plane, a2)
                                T.vcast(a2[0:1, 0:ch], o2, round_mode="rint")
                                T.transpose(o2, o1, permutation=[0, 2, 1])
                                T.transpose(o1, gout, permutation=[1, 0, 2])
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                        else:
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat)
                            T.vcast(flat, flat32, round_mode="rint")
                            if variant == "v3e":
                                T.copy(flat32, tmp)
                                T.transpose(tmp, t1_32, permutation=[2, 1, 0])
                                T.vmul(t1_32, inv_plane, t1_32)
                                T.vcast(t1_32, t1, round_mode="rint")
                                T.copy(t1[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                            else:  # v3f
                                T.copy(flat32, tmp)
                                T.reduce_sum(tmp, tmp3, dim=2)
                                T.vmul(tmp3, inv_plane, tmp3)
                                T.copy(tmp3[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])

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
        print(f"[p3 {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p3 {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
