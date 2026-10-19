"""Round-1 cost calibration probe: per-op cost on the hidden-state block shape.

Same (M=2048, N=4096, bm=2, grid=40, 26 tasks/core) as the real workload, so
every level differs from the next by exactly one op class:

  q0_copy    copy + scalar consume
  q1_elem1   copy + 1 fp16 elementwise pass
  q2_elem2   copy + 2 chained fp16 elementwise passes
  q4_elem4   copy + 4 chained fp16 elementwise passes
  q5_red     copy + reduce_min fp16 over the full row
  q6_red32   copy + vcast fp32 + reduce_min fp32
  q7_red2    copy + 2 chained reduce_min fp16 (same source, independent dst)

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
LEVELS = ("q0_copy", "q1_elem1", "q2_elem2", "q4_elem4", "q5_red", "q6_red32", "q7_red2")


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(grid, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((BM, N), dtype)
                y1 = T.alloc_fragment((BM, N), dtype)
                y2 = T.alloc_fragment((BM, N), dtype)
                y3 = T.alloc_fragment((BM, N), dtype)
                y4 = T.alloc_fragment((BM, N), dtype)
                f32 = T.alloc_fragment((BM, N), "float32")
                m = T.alloc_fragment((BM, 1), dtype)
                m2 = T.alloc_fragment((BM, 1), dtype)
                m32 = T.alloc_fragment((BM, 1), "float32")
                out_ub = T.alloc_shared((BM,), "int64")

                for s in T.serial(local):
                    block_id = s * grid + cid
                    if block_id < num_logical:
                        off_m = block_id * BM
                        T.copy(x[off_m : off_m + BM, 0:N], x_ub)

                        if level == "q0_copy":
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(x_ub[i, 0], "int64")
                        elif level == "q1_elem1":
                            T.vmul(x_ub, x_ub, y1)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(y1[i, 0], "int64")
                        elif level == "q2_elem2":
                            T.vmul(x_ub, x_ub, y1)
                            T.vmul(y1, y1, y2)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(y2[i, 0], "int64")
                        elif level == "q4_elem4":
                            T.vmul(x_ub, x_ub, y1)
                            T.vmul(y1, y1, y2)
                            T.vmul(y2, y2, y3)
                            T.vmul(y3, y3, y4)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(y4[i, 0], "int64")
                        elif level == "q5_red":
                            T.reduce_min(x_ub, m, dim=1)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(m[i, 0], "int64")
                        elif level == "q6_red32":
                            T.vcast(x_ub, f32, round_mode="rint")
                            T.reduce_min(f32, m32, dim=1)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(m32[i, 0], "int64")
                        else:  # q7_red2
                            T.reduce_min(x_ub, m, dim=1)
                            T.reduce_min(x_ub, m2, dim=1)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(m[i, 0] + m2[i, 0], "int64")

                        T.copy(out_ub, out[off_m : off_m + BM])

        return main

    return _func


def _select_config(M, N, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": BM, "path": "probe", "tile_n": None}


def _argreduce_kernel(M, N, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "q0_copy"), dtype)
