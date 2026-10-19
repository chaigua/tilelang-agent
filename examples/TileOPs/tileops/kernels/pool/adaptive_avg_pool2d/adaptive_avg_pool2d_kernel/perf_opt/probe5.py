"""Probe 5: auto-multi-buffer on/off (pass_configs["npuir.enable_auto_multi_buffer"]).

Hypothesis: the ~8x GM read amplification + 19us/unit stall appear when the
compiler multi-buffers the persistent loop with a layout-changing consumer.
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


def build(variant: str, ch: int = 64, mb: bool = True):
    ch = min(C, ch)
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv_plane = 1.0 / float(H * W)
    pc = {"npuir.enable_auto_multi_buffer": mb}

    @tilelang.jit(out_idx=[1], target="npuir", pass_configs=pc)
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
                mid = T.alloc_ub((ch, H, 1), "float32")
                m3 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat)
                        T.vcast(flat, flat32, round_mode="rint")
                        if variant == "dyn":
                            T.reduce_sum(flat32, mid, dim=2)
                            T.reduce_sum(mid, m3, dim=1)
                            T.vmul(m3, inv_plane, m3)
                            T.vcast(m3, gout, round_mode="rint")
                            T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                        else:  # ct
                            T.transpose(flat, t1, permutation=[2, 1, 0])
                            T.vmul(t1, inv_plane, t1)
                            T.copy(t1[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])

        return main

    return _func(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--mb", type=int, default=1)
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
        print(f"[p5 {args.variant} mb{args.mb} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p5 {args.variant} mb{args.mb} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
