#!/usr/bin/env python3
"""Floor probes for the GroupNorm Stage 4 attribution (Round 0).

Every probe uses the *same* persistent structure as the tuned kernel family
(`T.Kernel(num_blocks, is_npu=True)` + optional in-core `T.serial`
grid-stride loop + UB staging) so that the measured `msprof op Task
Duration(us)` is comparable with the algorithm kernel. Probes are
**calibration only** -- their numbers go to `copy_floor_calibration.jsonl`,
never to `perf_records.jsonl`.

Modes
-----
empty      : per-block single-element keep-alive write (fixed/launch floor)
dma_read   : GM->UB load of the real tile, keep-alive = 1-element UB->GM copy
dma_rw     : GM->UB load + full UB->GM store (pure DMA floor)
chain      : load + 2 v-op keep-alive passes + full store (copy-floor method:
             CONST-copy-floor-method, adapted to this single-input kernel)

Usage (under msprof, one workload at a time):
    msprof op --kernel-name=main ... python probe_floor.py \
        --mode empty --blocks 40 --bm 1 --case image-g32/fp16
"""

import argparse
import importlib.util
import math
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench import CASES, spec_of  # noqa: E402


def build(mode, M, D, bm, dtype, blocks, multi_buffer):
    num_logical = (M + bm - 1) // bm
    num_blocks = min(num_logical, blocks) if mode != "empty" else blocks
    nl = (num_logical + num_blocks - 1) // num_blocks

    pass_cfg = None
    if not multi_buffer:
        pass_cfg = {"npuir.enable_auto_multi_buffer": False}

    # explicit output tensors (no out_idx): probes pass y in, so the
    # measurement loop does not pay a per-launch allocation
    kwargs = dict(target="npuir")
    if pass_cfg:
        kwargs["pass_configs"] = pass_cfg

    if mode == "empty":

        @tilelang.jit(**kwargs)
        def _f():
            @T.prim_func
            def main(y: T.Tensor[(blocks, 1), "float32"]):
                with T.Kernel(blocks, is_npu=True) as (cid, _):
                    s = T.alloc_shared((1,), "float32")
                    T.vmul(s, 1.0, s)
                    T.copy(s, y[cid : cid + 1, 0:1])

            return main

        return _f(), None

    if mode == "dma_read":

        @tilelang.jit(**kwargs)
        def _f():
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                y: T.Tensor[(num_blocks, 1), dtype],
            ):
                with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((bm, D), dtype)
                    keep = T.alloc_shared((1, 1), dtype)
                    for i in T.serial(nl):
                        block_id = i * num_blocks + cid
                        if block_id < num_logical:
                            off = block_id * bm
                            rm = T.min(bm, M - off)
                            T.copy(x[off : off + rm, 0:D], stage[0:rm, 0:D])
                            T.copy(stage[0:1, 0:1], keep)
                    T.copy(keep, y[cid : cid + 1, 0:1])

            return main

        return _f(), None

    if mode == "dma_rw":

        @tilelang.jit(**kwargs)
        def _f():
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                y: T.Tensor[(M, D), dtype],
            ):
                with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((bm, D), dtype)
                    for i in T.serial(nl):
                        block_id = i * num_blocks + cid
                        if block_id < num_logical:
                            off = block_id * bm
                            rm = T.min(bm, M - off)
                            T.copy(x[off : off + rm, 0:D], stage[0:rm, 0:D])
                            T.copy(stage[0:rm, 0:D], y[off : off + rm, 0:D])

            return main

        return _f(), None

    if mode == "chain":
        # 2 v-op keep-alive passes in the resident dtype, full store.
        @tilelang.jit(**kwargs)
        def _f():
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                y: T.Tensor[(M, D), dtype],
            ):
                with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((bm, D), dtype)
                    acc = T.alloc_shared((bm, D), dtype)
                    for i in T.serial(nl):
                        block_id = i * num_blocks + cid
                        if block_id < num_logical:
                            off = block_id * bm
                            rm = T.min(bm, M - off)
                            T.copy(x[off : off + rm, 0:D], stage[0:rm, 0:D])
                            T.vadd(stage, 1.0, acc)
                            T.vmul(acc, 1.0, acc)
                            T.copy(acc[0:rm, 0:D], y[off : off + rm, 0:D])

            return main

        return _f(), None

    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["empty", "dma_read", "dma_rw", "chain"])
    ap.add_argument("--case", default="image-g32/fp16")
    ap.add_argument("--bm", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--mb", type=int, default=1, help="1 = auto multi-buffer on")
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--extra", type=int, default=5)
    args = ap.parse_args()

    mode = args.mode
    if mode == "empty":
        fn, buffers = build(mode, 1, 1, 1, "float16", args.blocks, bool(args.mb))
        y = torch.empty((args.blocks, 1), dtype=torch.float32).npu()
        total = args.warm_up + args.launch_count + args.extra
        for _ in range(total):
            fn(y)
        return

    m, d, _g, dtype_str = spec_of(args.case)
    torch_dtype = getattr(torch, dtype_str)
    fn, _ = build(mode, m, d, args.bm, dtype_str, args.blocks, bool(args.mb))

    x = torch.randn(m, d, dtype=torch.float32).to(torch_dtype).npu()
    if mode == "dma_read":
        nb = min((m + args.bm - 1) // args.bm, args.blocks)
        y = torch.empty((nb, 1), dtype=torch_dtype).npu()
    else:
        y = torch.empty((m, d), dtype=torch_dtype).npu()

    # compile + one correctness-touch launch before the measured loop
    fn(x, y)

    total = args.warm_up + args.launch_count + args.extra
    for _ in range(total):
        fn(x, y)


if __name__ == "__main__":
    main()
