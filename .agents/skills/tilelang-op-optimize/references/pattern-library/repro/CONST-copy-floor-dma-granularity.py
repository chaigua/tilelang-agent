"""Round-1 bandwidth-ceiling probe: is ~1.10 TB/s the machine's GM read limit?

Pure-copy kernels on the same 16.7MB tensor at increasing DMA granularity:
  d1_bm8    (8, 4096) blocks  -> 64KB contiguous per DMA, 256 blocks
  d2_bm16   (16, 4096) blocks -> 128KB contiguous per DMA, 128 blocks
  d3_flat32 (32768,) flat blocks -> 64KB contiguous, 256 blocks
The consumer is a scalar read of one element per block (keeps the DMA live).

Usage: PROBE_LEVEL=<name> python bench_driver.py --impl <this> --case hidden-state-argmin-float16

条目: pattern-library/constants.md CONST-copy-floor-dma-granularity
      （ED-C 机械转正自 examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/
      perf_opt/_argreduce_kernel_round1_probe4.py，2026-10-15 蒸馏；代码与结构零改动，仅头部登记）
origin_task: argmin-_argreduce_kernel-20261015T023310Z（Stage 4，scenario=optimize）
First verified: 2026-10-15, tilelang 0.1.2（dev root build, commit 417dae9）
      + CANN 9.0.0 + Ascend910B3（npu-smi 26.1.1；aicore 20 → aivector 40）
断言说明: 同形纯拷贝地板的量级与相对关系（绝对值随环境漂移，仅打印对照）——
      在 (2048,4096) fp16 / 同 persistent 结构 / 同 16.7MB 流量下，64 KB 连续 DMA 档
      （d1_bm8 / d3_flat32）不慢于 16 KB 档（同结构对照 c0_base = 15.82us，1.06 TB/s）；
      任务内实测 d1_bm8 = 14.03us、d3_flat32 = 13.61us（1.23 TB/s）。
测量（人工按需，msprof op Task Duration(us)；repro_runner 只做语法/结构检查）:
      PROBE_LEVEL=d1_bm8   python bench_driver.py --impl <this> --case hidden-state-argmin-float16
      PROBE_LEVEL=d3_flat32 python bench_driver.py --impl <this> --case hidden-state-argmin-float16
形态说明: 本文件是测量脚本（三级 DMA 粒度 + 标量消费者保活，避免 DCE），无内联 assert；
      驱动入口为任务工作区的 bench_driver.py（过程文件，允许失效）——登记形态因此为
      「测量脚本 + 头部断言说明」，可运行部分仅覆盖 import / py_compile 级检查。
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M = 2048
N = 4096
VECTOR_CORES = 40
LEVELS = ("d1_bm8", "d2_bm16", "d3_flat32")
BM = {"d1_bm8": 8, "d2_bm16": 16, "d3_flat32": 1}


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    bm = BM[level]
    cols = N if level != "d3_flat32" else 32768
    rows = _ceildiv(M * N, cols)
    num_logical = _ceildiv(rows, bm)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        @T.prim_func
        def main(
            x: T.Tensor((rows, cols), dtype),
            out: T.Tensor((rows,), "int64"),
        ):
            with T.Kernel(grid, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, cols), dtype)
                out_ub = T.alloc_shared((bm,), "int64")

                for s in T.serial(local):
                    block_id = s * grid + cid
                    if block_id < num_logical:
                        off = block_id * bm
                        T.copy(x[off : off + bm, 0:cols], x_ub)
                        for i in T.Parallel(bm):
                            out_ub[i] = T.cast(x_ub[i, 0], "int64")
                        T.copy(out_ub, out[off : off + bm])

        return main

    return _func


def _select_config(M_, N_, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": 1, "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "d1_bm8"), dtype)
