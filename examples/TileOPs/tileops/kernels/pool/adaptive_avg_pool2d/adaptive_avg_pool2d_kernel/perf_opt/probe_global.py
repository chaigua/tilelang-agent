"""Probe: C4 global-pool reduce structures (resnet-global 8x2048x7x7 -> 1x1).

Baseline C4 path reduces (ch,H,W) first along dim=2 (W=7) then dim=1 (H) of a
(ch,H,1) tensor -> both reduce axes are narrow / non-vector -> ~160us.
DMA floor for the same traffic measured at 5.36us (probe_c4 copy variant).

Layout: transpose the (ch,H,W) tile to (H,W,ch) so the C axis (the only wide,
division-friendly axis) is innermost, then reduce over the outer H/W axes.

Variants:
  base    : current (vcast -> reduce dim2 -> reduce dim1)  [ch,H,W] layout
  caccred : transpose -> vcast -> reduce dim1 -> reduce dim0
  cacc    : transpose -> serial vadd over H*W on the C axis
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
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_kernels = min(num_logical, vector_cores)
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
                step1 = T.alloc_ub((H, ch, W), DTYPE)
                hwn = T.alloc_ub((H, W, ch), DTYPE)
                hwn32 = T.alloc_ub((H, W, ch), "float32")
                mid_t = T.alloc_ub((H, 1, ch), "float32")
                gacc_t = T.alloc_ub((1, 1, ch), "float32")
                gout_t = T.alloc_ub((1, 1, ch), DTYPE)
                ot1 = T.alloc_ub((1, ch, 1), "float32")
                ot2 = T.alloc_ub((ch, 1, 1), "float32")
                gout3 = T.alloc_ub((ch, 1, 1), DTYPE)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat)
                        if variant == "caccred_t2":
                            T.transpose(flat, step1, permutation=[1, 0, 2])
                            T.transpose(step1, hwn, permutation=[0, 2, 1])
                            T.vcast(hwn, hwn32, round_mode="rint")
                            T.reduce_sum(hwn32, mid_t, dim=1)
                            T.reduce_sum(mid_t, gacc_t, dim=0)
                            T.transpose(gacc_t, ot1, permutation=[0, 2, 1])
                            T.transpose(ot1, ot2, permutation=[1, 0, 2])
                            T.vmul(ot2, inv_plane, ot2)
                            T.vcast(ot2, gout3, round_mode="rint")
                            T.copy(gout3, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                        elif variant == "caccred_flat":
                            T.transpose(flat, step1, permutation=[1, 0, 2])
                            T.transpose(step1, hwn, permutation=[0, 2, 1])
                            T.vcast(hwn, hwn32, round_mode="rint")
                            T.reduce_sum(hwn32, mid_t, dim=1)
                            T.reduce_sum(mid_t, gacc_t, dim=0)
                            T.vmul(gacc_t, inv_plane, gacc_t)
                            T.vcast(gacc_t, gout_t, round_mode="rint")
                            T.copy(gout_t, out[nb, cb * ch : cb * ch + ch, 0, 0])
                        elif variant == "caccred":
                            T.transpose(flat, step1, permutation=[1, 0, 2])
                            T.transpose(step1, hwn, permutation=[0, 2, 1])
                            T.vcast(hwn, hwn32, round_mode="rint")
                            T.reduce_sum(hwn32, mid_t, dim=1)
                            T.reduce_sum(mid_t, gacc_t, dim=0)
                            T.vmul(gacc_t, inv_plane, gacc_t)
                            T.vcast(gacc_t, gout_t, round_mode="rint")
                            T.copy(gout_t, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                        else:  # cacc
                            T.transpose(flat, step1, permutation=[1, 0, 2])
                            T.transpose(step1, hwn, permutation=[0, 2, 1])
                            T.clear(gacc_t)
                            for h in T.serial(H):
                                for w in T.serial(W):
                                    T.vadd(
                                        gacc_t[0:1, 0:1, 0:ch],
                                        hwn[h : h + 1, w : w + 1, 0:ch],
                                        gacc_t[0:1, 0:1, 0:ch],
                                    )
                            T.vmul(gacc_t, inv_plane, gacc_t)
                            T.vcast(gacc_t, gout_t, round_mode="rint")
                            T.copy(gout_t, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])

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
        print(f"[probe {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[probe {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
