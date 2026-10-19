"""Phase 0/1 probe: isolate the C4 (global-pool) slow path.

Baseline resnet-global (msprof, block dim 40): aiv_time ~142us with
aiv_mte2_time ~109us (77%) vs aiv_vec_time ~30us (22%); dynamic vs static copy
extent measured identical (166.7 vs 165.6us) -> not a dynamic-extent problem.

Variants (one change each):
  dyn        : baseline C4 path
  copy       : DMA only (no reduce/cast chain, minimal store) -- DMA floor
  reduce_f16 : reduce_sum directly on fp16 (skip the fp32 transit buffer)
  pipelined  : T.Pipelined over the persistent loop instead of T.serial
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
    is_pipe = variant == "pipelined"
    is_copy = variant == "copy"
    is_f16 = variant == "reduce_f16"

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
                mid16 = T.alloc_ub((ch, H, 1), DTYPE)
                gacc = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)

                if is_pipe:
                    for t in T.Pipelined(num_local, num_stages=2):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat[0:ch, 0:H, 0:W])
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
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:H, 0:W], flat[0:ch, 0:H, 0:W])
                            if is_copy:
                                T.copy(
                                    flat[0:1, 0:1, 0:1],
                                    out[nb, cb * ch : cb * ch + 1, 0:1, 0:1],
                                )
                            elif is_f16:
                                T.reduce_sum(flat, mid16, dim=2)
                                T.vcast(mid16, mid, round_mode="rint")
                                T.reduce_sum(mid, gacc, dim=1)
                                T.vmul(gacc, inv_plane, gacc)
                                T.vcast(gacc, gout, round_mode="rint")
                                T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])
                            else:
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
    if args.check and args.variant != "copy":
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        print(f"[probe {args.variant} ch{args.ch}] max_diff={(got-ref).abs().max().item():.3e}")
    else:
        print(f"[probe {args.variant} ch{args.ch}] ok")


if __name__ == "__main__":
    main()
