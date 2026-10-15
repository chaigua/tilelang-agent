"""ED-B repro/标定脚本 -- CONST-msprof-task-floor-910B3.

Calibrates, on Ascend910B3 / CANN 9.0.0 / AIV=40:
  * the msprof-op `Task Duration` fixed floor (empty kernel),
  * the minimum read+DMA-store floor for the 8k x 64 fp16 shape,
  * the per-elementwise-pass unit cost on a (block_m, dim) fp32 UB tile.

Measured values (this constant's body; Stage 4 of the fp8_quant task):
  empty kernel        : Task 3.37 us (aiv_time 1.01)  -> fixed overhead ~2.4 us
  read+cast+u8 store  : Task 4.25 us (aiv_time 1.85)  -> GM traffic floor ~4.25 us
  per elementwise pass: ~0.23-0.3 us  (ablation deltas: vlog2 -0.52, vexp2 -0.34,
                        reduce(dim=64 max) -1.06, 2D vdiv -0.22 us)
  fit                 : Task ~= aiv_time + ~2.4-2.6 us

This script only ASSERTS the models are built/run correctly; the durations are a
device measurement and must be read from the msprof output.  Run the two models
under:
  msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off \
            --aic-metrics=BasicInfo,PipeUtilization python <this-file> --mode {empty,rw}

First verified: 2026-10-15, tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96,
               CANN 9.0.0, Ascend910B3 (AIV=40)
origin_task: fp8_quant-_fp8_quant_kernel-optimize-20261015 (Stage 4)
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


def build(mode, rows=8192, dim=64, block_m=64):
    n_tiles = rows // block_m
    num_blocks = min(40, n_tiles)
    iters = (n_tiles + num_blocks - 1) // num_blocks

    @tilelang.jit(target="npuir")
    def _f(num_stages=0, block_m_hint=32):
        _ = (num_stages, block_m_hint)

        @T.prim_func
        def main(X: T.Tensor((rows, dim), "float16"),
                 Scale: T.Tensor((rows,), "float32"),
                 OutU8: T.Tensor((rows, dim), "uint8")):
            with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                x_in = T.alloc_ub((block_m, dim), "float16")
                u8 = T.alloc_ub((block_m, dim), "uint8")
                scale1 = T.alloc_ub((block_m,), "float32")
                for it in T.serial(iters):
                    tile = it * num_blocks + cid
                    if tile < n_tiles:
                        row0 = T.min(tile * block_m, rows - block_m)
                        if mode == "empty":
                            for i in T.Parallel(block_m):
                                scale1[i] = 1.0
                        else:
                            T.copy(X[row0, 0], x_in)
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(
                                    T.min(T.max(T.cast(x_in[i, j], "float32"), -128.0), 127.0),
                                    "uint8")
                            T.copy(u8, OutU8[row0, 0])
                            for i in T.Parallel(block_m):
                                scale1[i] = 1.0
                        T.copy(scale1, Scale[row0:row0 + block_m])

        return main

    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="empty", choices=["empty", "rw"])
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    k = build(args.mode)(0, 32)
    x = torch.randn(8192, 64, dtype=torch.float16, device="npu")
    s = torch.empty(8192, dtype=torch.float32, device="npu")
    o = torch.empty(8192, 64, dtype=torch.uint8, device="npu")
    with torch.no_grad():
        for _ in range(args.iters):
            k(x, s, o)
    print(f"[floor] mode={args.mode}: ran {args.iters} iterations (read Task Duration from msprof).")


if __name__ == "__main__":
    main()
