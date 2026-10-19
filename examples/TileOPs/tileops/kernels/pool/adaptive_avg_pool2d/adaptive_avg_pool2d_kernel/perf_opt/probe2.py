"""Probe 2: isolate transpose vs reduce costs, and test small-CH (D-3 backup).

Findings so far (resnet-global 8x2048x7x7->1x1, ch=64, 40 blocks):
  copy only                      : 5.37us
  copy + vcast(whole tile)       : 5.61us
  copy + vcast + reduce innermost: 133us   <- reduce over inner axis scalarised
  transpose + vadd accumulation   : 148us  <- transpose cost

Variants:
  ct      : copy + 1 transpose + elementwise consume + 1-elem store
  ct2     : copy + 2-transpose chain + elementwise consume + 1-elem store
  red1    : copy + vcast + reduce dim=1 (non-innermost) + 1-elem store
  red2    : copy + vcast + reduce dim=2 (innermost)      + 1-elem store
  acc     : transpose + vcast + 49 vadds + out transposes
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

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, H, W), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, H, W), DTYPE)
                flat32 = T.alloc_ub((ch, H, W), "float32")
                t1 = T.alloc_ub((W, H, ch), DTYPE)
                t2 = T.alloc_ub((H, ch, W), DTYPE)
                t3 = T.alloc_ub((H, W, ch), DTYPE)
                t32 = T.alloc_ub((W, H, ch), "float32")
                mid2 = T.alloc_ub((ch, 1, W), "float32")
                mid = T.alloc_ub((ch, H, 1), "float32")
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
                        if variant == "ct":
                            T.transpose(flat, t1, permutation=[2, 1, 0])
                            T.vmul(t1, inv_plane, t1)
                            T.copy(t1[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                        elif variant == "ct2":
                            T.transpose(flat, t2, permutation=[1, 0, 2])
                            T.transpose(t2, t3, permutation=[0, 2, 1])
                            T.vmul(t3, inv_plane, t3)
                            T.copy(t3[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                        elif variant == "red1":
                            T.vcast(flat, flat32, round_mode="rint")
                            T.reduce_sum(flat32, mid2, dim=1)
                            T.vmul(mid2, inv_plane, mid2)
                            T.copy(mid2[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                        elif variant == "red2":
                            T.vcast(flat, flat32, round_mode="rint")
                            T.reduce_sum(flat32, mid, dim=2)
                            T.vmul(mid, inv_plane, mid)
                            T.copy(mid[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                        else:  # acc
                            T.transpose(flat, t1, permutation=[2, 1, 0])
                            T.vcast(t1, t32, round_mode="rint")
                            T.clear(acc_c)
                            for s in T.serial(H * W):
                                T.vadd(
                                    acc_c,
                                    t32[s // W : s // W + 1, s % W : s % W + 1, 0:ch],
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
        print(f"[p2 {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p2 {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
