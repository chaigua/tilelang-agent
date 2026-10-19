"""Round-3 probe: does the UB tile *shape* of the GM->UB copy limit MTE2 bandwidth?

Real-kernel fact (f32 ch32, current best): aiv_mte2_time = 11.62 us with
aiv_mte2_active_bw = 4.27 GB/s -> the GM->UB transfer, not the round count,
is the wall.  The pure flat 1-D read floor at the same round count reaches
~8.3 GB/s/core.

Variants (same bytes, same 13 rounds, CH=32, resnet 8x2048x7x7 fp16):
  flat : x (N*C*HW,)  tile (CH*HW,)      one contiguous burst / round
  2d   : flat GM      tile (CH, HW)
  4d   : flat GM      tile (CH, 1, HW)
  gm4d : x (N,C,1,HW) tile (CH, 1, HW)   (= exact current-best declaration)

Each round keeps one element alive (guards against DCE) but does NOT reduce,
so only the DMA + keep-alive is measured.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

N, C, H, W = 8, 2048, 7, 7
HW = H * W
DTYPE = "float16"


def build(variant: str, ch: int, rounds: int, blocks: int = 40):
    nblk = (C + ch - 1) // ch
    nlogical = N * nblk
    nk = min(nlogical, blocks, NPUUtils.get().get_aicore_num() * 2)
    total = N * C * HW

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_c: int):
        @T.prim_func
        def main(x: T.Tensor((total,), DTYPE), out: T.Tensor((nk,), DTYPE)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                live = T.alloc_ub((1,), DTYPE)
                if variant == "flat":
                    buf = T.alloc_ub((ch * HW,), DTYPE)
                elif variant == "2d":
                    buf = T.alloc_ub((ch, HW), DTYPE)
                else:
                    buf = T.alloc_ub((ch, 1, HW), DTYPE)
                T.clear(live)
                for t in T.serial(rounds):
                    uid = t * nk + cid
                    if uid < nlogical:
                        nb = uid // nblk
                        cb = uid % nblk
                        base = (nb * C + cb * ch) * HW
                        T.copy(x[base : base + ch * HW], buf)
                        if variant == "flat":
                            T.vadd(live, buf[0:1], live)
                        elif variant == "2d":
                            T.vadd(live, buf[0, 0:1], live)
                        else:
                            T.vadd(live, buf[0, 0, 0:1], live)
                out[cid] = live[0]

        return main

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func_gm4d(_c: int):
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, HW), DTYPE), out: T.Tensor((nk,), DTYPE)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                live = T.alloc_ub((1,), DTYPE)
                buf = T.alloc_ub((ch, 1, HW), DTYPE)
                T.clear(live)
                for t in T.serial(rounds):
                    uid = t * nk + cid
                    if uid < nlogical:
                        nb = uid // nblk
                        cb = uid % nblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], buf)
                        T.vadd(live, buf[0, 0, 0:1], live)
                out[cid] = live[0]

        return main

    if variant == "gm4d":
        return _func_gm4d(ch), nk, rounds
    return _func(ch), nk, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("flat", "2d", "4d", "gm4d"))
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=13)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    kernel, nk, nl = build(args.variant, args.ch, args.rounds)
    total = N * C * HW
    torch.manual_seed(0)
    if args.variant == "gm4d":
        x = torch.randn(N, C, 1, HW, dtype=torch.float32).to(torch.float16).npu()
    else:
        x = torch.randn(total, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[dma {args.variant} ch{args.ch} r{args.rounds}] nk={nk} nl={nl} ok")


if __name__ == "__main__":
    main()
