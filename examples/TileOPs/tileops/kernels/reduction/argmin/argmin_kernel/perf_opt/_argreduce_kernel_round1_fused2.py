"""Round-1 winner exploration: per-row fused candidate loop (hidden-state class).

f2_bm2_perrow measured 15.25us vs the 31.35us baseline (2.06x) with the
structure: copy -> reduce_min(bm,N) -> vbrc -> [per-row unrolled fused
`T.Parallel(1,N) if_then_else` candidate] -> reduce_min -> epilogue.  The
fused per-row form removes vcmp + vselect entirely and the 2-D candidate loop
that scalarizes at bm>=2 is avoided by unrolling one row at a time.

This file parameterises {dtype} x {bm} x {pipelined stages} to find the best
configuration.  bf16 needs a fp32 working copy (T.reduce has no bf16 support).

Env: PROBE_LEVEL=<fp16|bf16>_bm<k>[_p<stages>]
Usage: PROBE_LEVEL=fp16_bm4 python bench_driver.py --impl <this> \
       --case hidden-state-argmin-float16 --check
"""

import os
import re

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BIG = 2**30
M = 2048
N = 4096
VECTOR_CORES = 40
_DT = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def _ceildiv(a, b):
    return (a + b - 1) // b


def parse(level):
    m = re.fullmatch(r"(fp16|bf16|fp32)_bm(\d+)(?:_p(\d+))?", level)
    if not m:
        raise ValueError(level)
    return m.group(1), int(m.group(2)), int(m.group(3) or 0)


def build(level, M=2048, N=4096, dt=None):
    tag, BM, stages = parse(level)
    dtype = dt or _DT[tag]
    N = ((N + 255) // 256) * 256   # pad contract: kernel declares N_padded
    is_bf16 = dtype == "bfloat16"
    use_pipe = bool(stages > 0)
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if is_bf16:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    x_work = T.alloc_fragment((BM, N), "float32")
                    extra = T.alloc_fragment((BM, 1), "float32")
                    ext_brc = T.alloc_fragment((BM, N), "float32")
                    cand = T.alloc_fragment((BM, N), "float32")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")

                    if use_pipe:
                        for s in T.Pipelined(local, num_stages=stages):
                            block_id = s * grid + cid
                            if block_id < num_logical:
                                off_m = block_id * BM
                                T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                                T.vcast(x_ub, x_work, round_mode="rint")
                                T.reduce_min(x_work, extra, dim=1)
                                T.vbrc(extra, ext_brc)
                                for r in range(BM):
                                    for i, j in T.Parallel(1, N):
                                        cand[r + i, j] = T.if_then_else(
                                            x_work[r + i, j] == ext_brc[r + i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                T.reduce_min(cand, first, dim=1)
                                T.reshape(first, first_flat)
                                T.vcast(first_flat, out_ub, round_mode="rint")
                                T.copy(out_ub, out[off_m : off_m + BM])
                    else:
                        for s in T.serial(local):
                            block_id = s * grid + cid
                            if block_id < num_logical:
                                off_m = block_id * BM
                                T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                                T.vcast(x_ub, x_work, round_mode="rint")
                                T.reduce_min(x_work, extra, dim=1)
                                T.vbrc(extra, ext_brc)
                                for r in range(BM):
                                    for i, j in T.Parallel(1, N):
                                        cand[r + i, j] = T.if_then_else(
                                            x_work[r + i, j] == ext_brc[r + i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
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

                    if use_pipe:
                        for s in T.Pipelined(local, num_stages=stages):
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
                    else:
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
    return {"block_m": parse(os.environ.get("PROBE_LEVEL", "fp16_bm2"))[1],
            "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "fp16_bm2"), M_, N_, dtype)
