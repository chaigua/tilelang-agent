"""Round-1 structural variants for the argmin resident path (hidden-state class).

Diagnostic attribution (round 0/1 probes, hidden-state fp16 2048x4096 bm=2):
  time ~= max(mte2, vec) + scalar ; mte2 10.3us, vec 18.9us, scalar 9.7us
  -> compute (vec+scalar) is ~2.8x the DMA time, so compute is the bottleneck.

Variants of the *same* construction (2048, 4096) fp16:
  v1_nofrag     resident chain, no x_frag staging, static extents
  v2_pipe2      v1 + T.Pipelined(num_local_tasks, num_stages=2)
  v3_grid       T.Kernel(num_logical) (no persistent serial loop / guard)
  v4_bm4        v1 with block_m = 4
  v5_bm8        v1 with block_m = 8
  v6_bm2p2      v1 but num_kernels = 20 (2x tasks/core)  -- per-task overhead probe
  v7_fp16idx    v1 with a fp16 index ramp + chunked split (N<=2048 half) -- not impl yet

Usage: PROBE_LEVEL=<name> python bench_driver.py --impl <this> --case hidden-state-argmin-float16
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
LEVELS = ("v1_nofrag", "v2_pipe2", "v3_grid", "v4_bm4", "v5_bm8", "v6_bm2p2")


def _ceildiv(a, b):
    return (a + b - 1) // b


def _chain_body(T, x_ub, BM, N, idx_j, sent_v, m, ext_brc, cmp_eq, cand, first,
                first_flat, out_ub, out, off_m, dtype, use_pipe=False):
    raise NotImplementedError


def build(level, dtype="float16"):
    bm = {"v1_nofrag": 2, "v2_pipe2": 2, "v3_grid": 2, "v4_bm4": 4,
          "v5_bm8": 8, "v6_bm2p2": 2}[level]

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if level == "v3_grid":
            grid = _ceildiv(M, bm)
            local = 1
        elif level == "v6_bm2p2":
            grid = min(_ceildiv(M, bm), VECTOR_CORES // 2)
            local = _ceildiv(_ceildiv(M, bm), grid)
        else:
            grid = min(_ceildiv(M, bm), VECTOR_CORES)
            local = _ceildiv(_ceildiv(M, bm), grid)
        num_logical = _ceildiv(M, bm)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(grid, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, N), dtype)
                m = T.alloc_fragment((bm, 1), dtype)
                ext_brc = T.alloc_fragment((bm, N), dtype)
                cmp_eq = T.alloc_fragment((bm, N), "bool")
                idx_src = T.alloc_shared((N,), "float32")
                idx_row = T.alloc_shared((1, N), "float32")
                idx_j = T.alloc_shared((bm, N), "float32")
                sent_v = T.alloc_fragment((bm, N), "float32")
                cand = T.alloc_fragment((bm, N), "float32")
                first = T.alloc_fragment((bm, 1), "float32")
                first_flat = T.alloc_fragment((bm,), "float32")
                out_ub = T.alloc_shared((bm,), "int64")

                T.arange(idx_src, [1], 0)
                if bm >= 2:
                    T.reshape(idx_src, idx_row)
                    T.vbrc(idx_row, idx_j)
                else:
                    T.reshape(idx_src, idx_j)
                T.vbrc(T.float32(BIG), sent_v)

                if level == "v2_pipe2":
                    for s in T.Pipelined(local, num_stages=2):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * bm
                            T.copy(x[off_m : off_m + bm, 0:N], x_ub)
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + bm])
                else:
                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * bm
                            T.copy(x[off_m : off_m + bm, 0:N], x_ub)
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + bm])

        return main

    return _func


def _select_config(M, N, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": 2, "path": "probe", "tile_n": None}


def _argreduce_kernel(M, N, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "v1_nofrag"), dtype)
