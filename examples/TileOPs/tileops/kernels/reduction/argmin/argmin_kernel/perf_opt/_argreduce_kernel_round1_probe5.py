"""Round-1 reduce-shape probe: what makes T.reduce_min expensive?

Round-0/1 attribution: a single fp16 `reduce_min` over a (2, 4096) block costs
~3.9us Task Duration over a pure copy (~270 cycles/task), i.e. ~3.3x an
elementwise pass (83 cycles/task).  If the cost is driven by the tree shape
rather than the element count, a 2-level (or 3-D) reduce could be much cheaper.

All levels keep the same total data (8192 elements per task, 26 tasks/core).

  r1_2x4096     reduce (2,4096) -> (2,1)
  r2_2x32x128   reshape to (2,32,128), reduce dim=2
  r3_2x64x64    reshape to (2,64,64),  reduce dim=2
  r4_2x128x32   reshape to (2,128,32), reduce dim=2
  r5_2level     3-D reduce dim=2 then dim=1 on the (2,32,1) result
  r6_2x16x512   reshape to (2,16,512), reduce dim=2

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
BM = 2
VECTOR_CORES = 40
LEVELS = ("r1_2x4096", "r2_2x32x128", "r3_2x64x64", "r4_2x128x32", "r5_2level", "r6_2x16x512")
SPLIT = {
    "r2_2x32x128": (32, 128),
    "r3_2x64x64": (64, 64),
    "r4_2x128x32": (128, 32),
    "r5_2level": (32, 128),
    "r6_2x16x512": (16, 512),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        S, K = SPLIT.get(level, (1, N))

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(grid, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((BM, N), dtype)
                m = T.alloc_fragment((BM, 1), dtype)
                out_ub = T.alloc_shared((BM,), "int64")

                x3 = T.alloc_fragment((BM, S, K), dtype)
                m3 = T.alloc_fragment((BM, S, 1), dtype)
                m2 = T.alloc_fragment((BM, S), dtype)
                m1 = T.alloc_fragment((BM, 1), dtype)

                for s in T.serial(local):
                    block_id = s * grid + cid
                    if block_id < num_logical:
                        off_m = block_id * BM
                        T.copy(x[off_m : off_m + BM, 0:N], x_ub)

                        if level == "r1_2x4096":
                            T.reduce_min(x_ub, m, dim=1)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(m[i, 0], "int64")
                        elif level == "r5_2level":
                            T.reshape(x_ub, x3)
                            T.reduce_min(x3, m3, dim=2)
                            T.reshape(m3, m2)
                            T.reduce_min(m2, m1, dim=1)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(m1[i, 0], "int64")
                        else:
                            T.reshape(x_ub, x3)
                            T.reduce_min(x3, m3, dim=2)
                            T.reshape(m3, m2)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(m2[i, 0], "int64")

                        T.copy(out_ub, out[off_m : off_m + BM])

        return main

    return _func


def _select_config(M_, N_, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": BM, "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "r1_2x4096"), dtype)
