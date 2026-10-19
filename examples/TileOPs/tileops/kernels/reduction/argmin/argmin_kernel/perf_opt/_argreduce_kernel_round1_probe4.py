"""Round-1 bandwidth-ceiling probe: is ~1.10 TB/s the machine's GM read limit?

Pure-copy kernels on the same 16.7MB tensor at increasing DMA granularity:
  d1_bm8    (8, 4096) blocks  -> 64KB contiguous per DMA, 256 blocks
  d2_bm16   (16, 4096) blocks -> 128KB contiguous per DMA, 128 blocks
  d3_flat32 (32768,) flat blocks -> 64KB contiguous, 256 blocks
The consumer is a scalar read of one element per block (keeps the DMA live).

Usage: PROBE_LEVEL=<name> python bench_driver.py --impl <this> --case hidden-state-argmin-float16
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
