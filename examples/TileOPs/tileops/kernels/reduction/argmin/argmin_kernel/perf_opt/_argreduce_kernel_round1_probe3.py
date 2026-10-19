"""Round-1 copy-floor probe: how fast can 16.7MB be read + 16KB written?

Findings that motivate this probe (hidden-state fp16, 2048x4096, bm=2, grid 40):
  - q0_copy (copy + per-task 16B output store): Task Duration 15.82us,
    aiv_time 13.10 = mte2 10.92 (10.3us floor) + scalar; mte3 7.99us for only
    26.6 store instructions => ~0.3us per tiny GM store, i.e. the per-task
    output write is a large *pipe* cost even though it moves 16 bytes.
  - Target for the operator is 13.7us (Ratio 70%), i.e. below the measured
    copy floor => the copy structure itself must be improved first.

Levels:
  c0_base       grid-stride mapping, per-task 16B output store (== q0_copy)
  c1_batchout   grid-stride mapping, outputs accumulated in UB, one store/core
  c2_contig     contiguous rows per core, one output store/core
  c3_pipe3      c0 with T.Pipelined(num_stages=3)
  c4_gm2frag    c0 but GM -> fragment copy (no shared staging)
  c5_contig_p2  c2 with T.Pipelined(num_stages=2)

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
LEVELS = ("c0_base", "c1_batchout", "c2_contig", "c3_pipe3", "c4_gm2frag", "c5_contig_p2")


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
                x_frag = T.alloc_fragment((BM, N), dtype)
                out_ub = T.alloc_shared((BM,), "int64")
                out_acc = T.alloc_shared((local * BM,), "int64")

                if level == "c3_pipe3":
                    for s in T.Pipelined(local, num_stages=3):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(x_ub[i, 0], "int64")
                            T.copy(out_ub, out[off_m : off_m + BM])
                elif level == "c5_contig_p2":
                    for s in T.Pipelined(local, num_stages=2):
                        block_id = cid * local + s
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            for i in T.Parallel(BM):
                                out_acc[s * BM + i] = T.cast(x_ub[i, 0], "int64")
                    start = cid * local * BM
                    if start < M:
                        valid = T.min(local * BM, M - start)
                        T.copy(out_acc[0:valid], out[start : start + valid])
                elif level == "c1_batchout":
                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(x_ub[i, 0], "int64")
                            T.copy(out_ub, out[off_m : off_m + BM])
                elif level == "c4_gm2frag":
                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_frag)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(x_frag[i, 0], "int64")
                            T.copy(out_ub, out[off_m : off_m + BM])
                else:  # c0_base
                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            for i in T.Parallel(BM):
                                out_ub[i] = T.cast(x_ub[i, 0], "int64")
                            T.copy(out_ub, out[off_m : off_m + BM])

        return main

    return _func


def _select_config(M, N, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": BM, "path": "probe", "tile_n": None}


def _argreduce_kernel(M, N, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "c0_base"), dtype)
