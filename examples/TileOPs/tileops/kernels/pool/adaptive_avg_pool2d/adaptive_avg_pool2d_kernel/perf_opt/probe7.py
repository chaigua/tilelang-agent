"""Probe 7: transpose-free global pool on the flattened (ch, 1, H*W) layout.

Findings:
  * natural-layout (ch,H,W) reduce  -> 133-155us (inner extent 7, scalarised)
  * flattened (ch,1,HW) + transpose -> 68-81us  (transpose lowered via GM round-trip:
    UB_to_GM=48KB/block, mte3_time ~58us -- vs the (ch,H,W) case where layout
    propagation also destroyed the input DMA: GM_to_UB 336KB)
  * flattened layout alone keeps the input DMA contiguous: GM_to_UB 48KB, 3.5us

So: keep the flattened layout, and try to reduce WITHOUT a transpose.

Variants:
  r49   : (ch,1,HW) f16 -> vcast f32 -> reduce dim=2 (innermost 49) -> out
  r49f  : same but reduce directly in f16
  t16   : (1,ch,HW) f16 -> transpose [0,2,1] -> (1,HW,ch) f16 -> vmul -> 1-elem out
          (isolates the transpose cost on the flattened layout, f16)
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

    @tilelang.jit(out_idx=[1], target="npuir", pass_configs=pc)
    def _func(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                fb = T.alloc_ub((ch, 1, HW), DTYPE)
                fb32 = T.alloc_ub((ch, 1, HW), "float32")
                m3 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)
                fa16 = T.alloc_ub((1, ch, HW), DTYPE)
                ft16 = T.alloc_ub((1, HW, ch), DTYPE)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        if variant == "t16":
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], fa16)
                            T.transpose(fa16, ft16, permutation=[0, 2, 1])
                            T.vmul(ft16, inv_plane, ft16)
                            T.copy(ft16[0:1, 0:1, 0:1], out[nb, cb * ch : cb * ch + 1, 0:1, 0:1])
                        else:
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], fb)
                            if variant == "r49f":
                                T.reduce_sum(fb, m3)
                                T.vmul(m3, inv_plane, m3)
                                T.vcast(m3, gout, round_mode="rint")
                            else:  # r49
                                T.vcast(fb, fb32, round_mode="rint")
                                T.reduce_sum(fb32, m3, dim=2)
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
        print(f"[p7 {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[p7 {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
