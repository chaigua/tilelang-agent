"""Round-3 probe: bisect the C4 flattened-layout chain (resnet 8x2048x7x7 fp16).

Each stage adds exactly one op to the previous stage (same grid / same 13
rounds / CH=32):

  copy    T.copy(x -> tile)                                  (DMA only)
  vcast   + T.vcast(tile -> tile32)                          (f16->f32 plane)
  reduce  + T.reduce_sum(tile32 -> sred, dim=2)
  full    + T.vmul(sred, inv) + T.vcast(sred -> gout) + copy out

This isolates which op owns the ~8 us that sits between the DMA-only kernel
(8.75 us) and the full C4 kernel (16.78 us).
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


def build(stage: str, ch: int = 32, rounds: int = 13, blocks: int = 40):
    nblk = (C + ch - 1) // ch
    nlogical = N * nblk
    nk = min(nlogical, blocks, NPUUtils.get().get_aicore_num() * 2)
    inv = 1.0 / float(HW)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_c: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                tile = T.alloc_ub((ch, 1, HW), DTYPE)
                t32 = T.alloc_ub((ch, 1, HW), "float32")
                sred = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)
                for t in T.serial(rounds):
                    uid = t * nk + cid
                    if uid < nlogical:
                        nb = uid // nblk
                        cb = uid % nblk
                        T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], tile)
                        if stage != "copy":
                            T.vcast(tile, t32, round_mode="rint")
                            if stage != "vcast":
                                T.reduce_sum(t32, sred, dim=2)
                                if stage == "full":
                                    T.vmul(sred, inv, sred)
                                    T.vcast(sred, gout, round_mode="rint")
                                    T.copy(
                                        gout,
                                        out[nb, cb * ch : cb * ch + ch, 0:1, 0:1],
                                    )
                if stage == "copy":
                    T.copy(tile[0:1, 0:1, 0:1], out[0, 0:1, 0:1, 0:1])

        return main

    return _func(ch), nk, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("copy", "vcast", "reduce", "full"))
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=13)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    kernel, nk, nl = build(args.stage, args.ch, args.rounds)
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[bisect {args.stage} ch{args.ch} r{args.rounds}] nk={nk} ok")


if __name__ == "__main__":
    main()
