"""Round-1 fused-candidate variants (hidden-state class).

Round-1 attribution: per-op cost on a (2,4096) fp16 block (26 tasks/core) is
~83 cycles for an elementwise op, ~215 for vcmp/vselect, ~270 for a reduce --
i.e. dominated by op class / dependency latency rather than bytes touched
(halving the candidate bytes via fp16 gave only 1.7%).

The explicit chain uses reduce + vbrc + vcmp + vselect + reduce.  The tiled path
proves that a single fused `T.Parallel + if_then_else` candidate loop vectorizes
at bm == 1 (and scalarizes at bm >= 2), which removes vcmp + vselect entirely.
This file builds the fused form (a) at bm=1 and (b) at bm=2 unrolled per row
(the argmax r3e per-row trick: each row is its own (1,N) Parallel loop).

Variants:
  f1_bm1_fused    bm=1, single fused candidate loop
  f2_bm2_perrow   bm=2, reduce/vbrc on (2,N), candidate loop unrolled per row
  f3_bm2_perrow_i16  f2 with the candidate built as int16 (no fp32 cand write)

Usage: PROBE_LEVEL=f2_bm2_perrow python bench_driver.py --impl <this> \
       --case hidden-state-argmin-float16 --check
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BIG = 2**30
M = 2048
N = 4096
VECTOR_CORES = 40
LEVELS = ("f1_bm1_fused", "f2_bm2_perrow", "f3_bm2_perrow_i16", "f4_bm2_perrow16")


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    BM = 1 if level == "f1_bm1_fused" else 2
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if level == "f1_bm1_fused":

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    extra = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    cand = T.alloc_fragment((BM, N), "float32")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.reduce_min(x_ub, extra, dim=1)
                            T.vbrc(extra, ext_brc)
                            for i, j in T.Parallel(BM, N):
                                cand[i, j] = T.if_then_else(
                                    x_ub[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + BM])

        elif level == "f4_bm2_perrow16":

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    extra = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    cand = T.alloc_fragment((BM, N), "int16")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.reduce_min(x_ub, extra, dim=1)
                            T.vbrc(extra, ext_brc)
                            for r in range(BM):
                                for i, j in T.Parallel(1, N):
                                    cand[r + i, j] = T.if_then_else(
                                        x_ub[r + i, j] == ext_brc[r + i, j],
                                        T.cast(j, "int16"),
                                        T.int16(32767),
                                    )
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + BM])

        else:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    extra = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    cand = T.alloc_fragment((BM, N), "float32")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.reduce_min(x_ub, extra, dim=1)
                            T.vbrc(extra, ext_brc)
                            for r in range(BM):
                                for i, j in T.Parallel(1, N):
                                    cand[r + i, j] = T.if_then_else(
                                        x_ub[r + i, j] == ext_brc[r + i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + BM])

        return main

    return _func


def _select_config(M_, N_, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    bm = 1 if os.environ.get("PROBE_LEVEL", "f2_bm2_perrow") == "f1_bm1_fused" else 2
    return {"block_m": bm, "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "f2_bm2_perrow"), dtype)
