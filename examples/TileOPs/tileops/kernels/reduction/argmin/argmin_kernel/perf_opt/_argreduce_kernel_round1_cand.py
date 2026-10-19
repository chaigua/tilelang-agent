"""Round-1 real-chain variants (argmin semantics, hidden-state class).

Round-0/1 attribution for the (2048,4096) fp16 resident chain (bm=2, grid 40):
  pure copy floor (16KB DMAs)       15.82 us
  + fp16 row reduce                  19.72 us   (+3.90)
  + fp32 candidate reduce (cast)     24.30 us   (+4.58 more)
  full chain                         31.35 us   (vbrc+vcmp+vselect+fp32 reduce = +12.66)
The candidate chain dominates; its biggest buffer traffic is the fp32
`idx_j`/`sent_v`/`cand` triple (96 KB per 16 KB block).

Variants (all real argmin; each level is its own prim_func -- buffers must not
be allocated inside a Python conditional block, C-3):
  v_base        baseline chain (fp32 idx/sent/cand)
  v_scalarsent  v_base but vselect's "not-matched" operand is a scalar BIG
  v_fp16cand    v_base but idx/sent/cand are fp16 (SPEED PROBE ONLY: fp16
                cannot represent every j < 4096 exactly -> not adoptable)

Usage: PROBE_LEVEL=<name> python bench_driver.py --impl <this> --case hidden-state-argmin-float16 --check
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
BM = 2
VECTOR_CORES = 40
LEVELS = ("v_base", "v_scalarsent", "v_fp16cand")


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if level == "v_fp16cand":

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    m = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    cmp_eq = T.alloc_fragment((BM, N), "bool")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")
                    idx_src = T.alloc_shared((N,), "float16")
                    idx_row = T.alloc_shared((1, N), "float16")
                    idx_j = T.alloc_shared((BM, N), "float16")
                    sent_v = T.alloc_fragment((BM, N), "float16")
                    cand = T.alloc_fragment((BM, N), "float16")
                    m16 = T.alloc_fragment((BM, 1), "float16")

                    T.arange(idx_src, [1], 0)
                    T.reshape(idx_src, idx_row)
                    T.vbrc(idx_row, idx_j)
                    T.vbrc(T.float16(0), sent_v)

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, m16, dim=1)
                            T.vcast(m16, first, round_mode="rint")
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + BM])

        elif level == "v_scalarsent":

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    m = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    cmp_eq = T.alloc_fragment((BM, N), "bool")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((BM, N), "float32")
                    cand = T.alloc_fragment((BM, N), "float32")

                    T.arange(idx_src, [1], 0)
                    T.reshape(idx_src, idx_row)
                    T.vbrc(idx_row, idx_j)

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, T.float32(BIG), cand)
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
                    m = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    cmp_eq = T.alloc_fragment((BM, N), "bool")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((BM, N), "float32")
                    sent_v = T.alloc_fragment((BM, N), "float32")
                    cand = T.alloc_fragment((BM, N), "float32")

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


def _select_config(M_, N_, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": BM, "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "v_base"), dtype)
