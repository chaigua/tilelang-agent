"""[repro] CONST-groupnorm-smallshape-memfloor-910B4 -- pattern-library/constants.md

Two floors that decide whether a small-shape roofline target is reachable:
  (a) empty-kernel fixed-cost floor: block count -> Task Duration
  (b) pure-DMA floor: same persistent structure, same traffic, zero compute

Measured on Ascend910B4 / CANN 9.0.0 / tilelang 0.1.2+2afbf3b (GroupNorm Stage 4,
task TileOPs-GroupNormFwdOp-20261018T021206Z), msprof op median of 20 launches:
  empty kernel:  1 block 1.13 | 2 blocks 1.42 | 29 blocks 2.32 |
                 32 blocks 2.46 | 40 blocks 2.89-2.91 us
  pure DMA (bm=3, D=4096 rows): 29 blocks, 4.10 MB -> 4.39 us (956 GB/s)
  pure DMA (bm=2, D=6272):      32 blocks, 3.14 MB -> 3.81 us (844 GB/s)
  pure DMA (bm=2, D=7200):      32 blocks, 1.81 MB -> 3.09 us (598 GB/s)
Both scale with block count and with transferred bytes; at 1.8-4.2 MB the
achieved DMA bandwidth (598-956 GB/s) is far below the same-family large-traffic
copies (1.06 TB/s @16 KB granularity, 1.23 TB/s @64 KB, see
CONST-copy-floor-dma-granularity), i.e. fixed cost dominates and the floor must
be measured per workload size -- never extrapolated from a big-shape constant.

Assertion (constant class, runnable part): both probes are numerically exact
(empty keep-alive write, DMA read+write round-trip bit-exact) and the empty
kernel duration is monotone in block count when measured (print only; the
timing step below is on demand and not executed by repro_runner).

Measurement (on demand):
  msprof op --kernel-name=main --output=<dir> --launch-count=20 --warm-up=5 \
      --dump=off python CONST-groupnorm-smallshape-memfloor-910B4.py --mode empty --blocks 40
  msprof op --kernel-name=main --output=<dir> --launch-count=20 --warm-up=5 \
      --dump=off python CONST-groupnorm-smallshape-memfloor-910B4.py --mode dma_rw --case image
Read `Task Duration(us)` from OpBasicInfo_*.csv (median).

First verified: 2026-10-18 (GroupNorm Stage 4 floor probes).
origin_task (provenance): TileOPs-GroupNormFwdOp-20261018T021206Z
Re-verification log (append-only):
  2026-10-18 PASS (runnable part: both probes exact, NPU executed)
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

CASES = {
    "image": dict(M=256, D=4096, bm=3, blocks=29, dtype="float16"),
    "wider": dict(M=128, D=6272, bm=2, blocks=32, dtype="float16"),
    "tail": dict(M=64, D=7200, bm=2, blocks=32, dtype="float16"),
}


def build_empty(blocks):
    @tilelang.jit(target="npuir")
    def _f():
        @T.prim_func
        def main(y: T.Tensor((blocks, 1), "float32")):
            with T.Kernel(blocks, is_npu=True) as (cid, _):
                s = T.alloc_shared((1,), "float32")
                T.vmul(s, 1.0, s)
                T.copy(s, y[cid : cid + 1, 0:1])

        return main

    return _f()


def build_dma(M, D, bm, dtype, blocks):
    num_logical = (M + bm - 1) // bm
    nb = min(num_logical, blocks)
    nl = (num_logical + nb - 1) // nb

    @tilelang.jit(target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, D), dtype), y: T.Tensor((M, D), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _):
                stage = T.alloc_shared((bm, D), dtype)
                for i in T.serial(nl):
                    block_id = i * nb + cid
                    if block_id < num_logical:
                        off = block_id * bm
                        rm = T.min(bm, M - off)
                        T.copy(x[off : off + rm, 0:D], stage[0:rm, 0:D])
                        T.copy(stage[0:rm, 0:D], y[off : off + rm, 0:D])

        return main

    return _f(), nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["empty", "dma_rw", "both"])
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--case", default="image", choices=sorted(CASES))
    ap.add_argument("--launches", type=int, default=3)
    args = ap.parse_args()

    if args.mode in ("empty", "both"):
        fn = build_empty(args.blocks)
        def _run(fill):
            y = torch.full((args.blocks, 1), fill, dtype=torch.float32,
                           device="npu")
            for _ in range(args.launches):
                fn(y)
            torch.npu.synchronize()
            return y.cpu()

        # The keep-alive store overwrites the sentinel with UB data (the value
        # is uninitialised, so only "sentinel is gone" is asserted).
        y1 = _run(1.0)
        assert not torch.equal(y1, torch.full_like(y1, 1.0)), (
            "keep-alive store must overwrite the GM sentinel"
        )
        print(f"empty blocks={args.blocks}: runnable PASS "
              f"(time it with msprof op, expected ~2.9 us at 40 blocks)")

    if args.mode in ("dma_rw", "both"):
        c = CASES[args.case]
        fn, nb = build_dma(c["M"], c["D"], c["bm"], c["dtype"], c["blocks"])
        x = torch.randn(c["M"], c["D"], dtype=torch.float16, device="npu")
        y = torch.empty_like(x)
        for _ in range(args.launches):
            fn(x, y)
        torch.npu.synchronize()
        assert torch.equal(x, y), "DMA probe must copy bit-exactly"
        mb = c["M"] * c["D"] * 2 / 1e6
        print(f"dma_rw case={args.case} blocks={nb} traffic={2 * mb:.2f} MB: "
              f"runnable PASS (time it with msprof op)")


if __name__ == "__main__":
    main()
