"""Bisection probe for the C4 global-pool path (resnet-global 8x2048x7x7 -> 1x1).

copy-only floor = 5.36us, full C4 chain = ~146-166us, yet the vector engine's
busy span is only 8-30us and the DMA floor is 5.4us -> the cost is in the
interaction (serialised per-unit dependency chain, MTE2 span 109-134us).

Stages (each adds exactly one op to the previous):
  s0 : DMA in only, 1-elem store
  s1 : + vcast(flat -> flat32)
  s2 : + reduce_sum(dim=2)
  s3 : + reduce_sum(dim=1) (+ vmul + vcast + full store)  == baseline chain
  s3p: s3 with T.Pipelined(num_stages=2) instead of T.serial
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


def build(stage: str, ch: int = 64):
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
                mid = T.alloc_ub((ch, H, 1), "float32")
                gacc = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                if stage == "s3p":
                    for t in T.Pipelined(num_local, num_stages=2):
                        uid = t * num_kernels + cid
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat)
                        T.vcast(flat, flat32, round_mode="rint")
                        T.reduce_sum(flat32, mid, dim=2)
                        T.reduce_sum(mid, gacc, dim=1)
                        T.vmul(gacc, inv_plane, gacc)
                        T.vcast(gacc, gout, round_mode="rint")
                        T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                else:
                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat)
                            if stage == "s0":
                                T.copy(flat[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                            elif stage == "s1":
                                T.vcast(flat, flat32, round_mode="rint")
                                T.copy(flat32[0:1, 0:1, 0:1], gacc[0:1, 0:1, 0:1])
                                T.vcast(gacc[0:1, 0:1, 0:1], gout[0:1, 0:1, 0:1], round_mode="rint")
                                T.copy(gout[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                            elif stage == "s2":
                                T.vcast(flat, flat32, round_mode="rint")
                                T.reduce_sum(flat32, mid, dim=2)
                                T.copy(mid[0:1, 0:1, 0:1], gacc[0:1, 0:1, 0:1])
                                T.vcast(gacc[0:1, 0:1, 0:1], gout[0:1, 0:1, 0:1], round_mode="rint")
                                T.copy(gout[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                            else:  # s3
                                T.vcast(flat, flat32, round_mode="rint")
                                T.reduce_sum(flat32, mid, dim=2)
                                T.reduce_sum(mid, gacc, dim=1)
                                T.vmul(gacc, inv_plane, gacc)
                                T.vcast(gacc, gout, round_mode="rint")
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])

        return main

    return _func(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()
    kernel = build(args.stage, args.ch)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[bisect {args.stage} ch{args.ch}] ok {tuple(out.shape)}")


if __name__ == "__main__":
    main()
