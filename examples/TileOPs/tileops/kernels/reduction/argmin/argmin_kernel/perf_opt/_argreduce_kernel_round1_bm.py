"""Round-1 block-size / dtype sweep for the hidden-state shape (2048, 4096).

Baseline report: fp16 31.42us at bm=2, bf16 54.08us at bm=1 (_B_elem budget
puts bf16 at bm=1).  The real UB limit is 192KB, not the 64KB proxy budget, so
bm=2 for bf16 may well compile and halve the per-core task count (52 -> 26).

Levels: <dtype>_bm<k> for dtype in {fp16, bf16, fp32} and k in {1,2,4}.

Usage: PROBE_LEVEL=bf16_bm2 python bench_driver.py --impl <this> \
       --case hidden-state-argmin-bfloat16 --check
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
LEVELS = (
    "fp16_bm1", "fp16_bm2", "fp16_bm4",
    "bf16_bm1", "bf16_bm2", "bf16_bm4",
    "fp32_bm1", "fp32_bm2", "fp32_bm4",
)
_DT = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
_TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, _dtype=None):
    tag, bm_s = level.split("_bm")
    dtype = _DT[tag]
    BM = int(bm_s)
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)
    is_bf16 = dtype == "bfloat16"
    work = "float32" if is_bf16 else dtype

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
                    m = T.alloc_fragment((BM, 1), "float32")
                    ext_brc = T.alloc_fragment((BM, N), "float32")
                    cmp_eq = T.alloc_fragment((BM, N), "bool")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((BM, N), "float32")
                    sent_v = T.alloc_fragment((BM, N), "float32")
                    cand = T.alloc_fragment((BM, N), "float32")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")

                    T.arange(idx_src, [1], 0)
                    T.reshape(idx_src, idx_row)
                    T.vbrc(idx_row, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.vcast(x_ub, x_work, round_mode="rint")
                            T.reduce_min(x_work, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_work, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
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
                    m = T.alloc_fragment((BM, 1), work)
                    ext_brc = T.alloc_fragment((BM, N), work)
                    cmp_eq = T.alloc_fragment((BM, N), "bool")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((BM, N), "float32")
                    sent_v = T.alloc_fragment((BM, N), "float32")
                    cand = T.alloc_fragment((BM, N), "float32")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")

                    T.arange(idx_src, [1], 0)
                    T.reshape(idx_src, idx_row)
                    T.vbrc(idx_row, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + BM])

        return main

    return _func


def _level_bm():
    return int(os.environ.get("PROBE_LEVEL", "fp16_bm2").split("_bm")[1])


def _select_config(M_, N_, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": _level_bm(), "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "fp16_bm2"), dtype)
