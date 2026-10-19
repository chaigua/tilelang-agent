"""Round-1 diagnostic probe: static extents + narrow (int16) candidate chain.

Round-0 attribution (hidden-state fp16, 2048x4096, bm=2, 40 cores, msprof):
  pa copy+1 reduce            aiv_time 16.54  vec 4.66  scalar 6.04  mte2 10.64
  pc full chain (dynamic)     aiv_time 29.20  vec 18.86 scalar 9.82  mte2 10.33
  baseline (== pd)            aiv_time 29.11  Task Duration 31.35
=> the scalar component (~6us on the minimal kernel) is suspected to come from
   the runtime-sized `real_m` subviews (`TMIN` + dynamic memref extents), and
   the fp32 candidate chain (idx_j/sent/cand fp32 = 4B) dominates vec_time.

Levels (all static-extent when M % BM == 0):
  s1_red           copy + reduce_min(x_ub)
  s2_chain_fp32    copy + reduce + vbrc + vcmp + vselect(fp32) + reduce(fp32)
  s3_chain_i16     copy + reduce + vbrc + vcmp + vselect(int16) + reduce(int16)
  s4_chain_fp32_frag  s2 + same-dtype x_frag staging
  s5_chain_i16_2red   s3 with the first index reduce replaced by 2-stage chunk reduce

Usage: PROBE_LEVEL=<name> python bench_driver.py --impl <this> --case hidden-state-argmin-float16
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BIG = 2**30
BIG_I16 = 32767

M = 2048
N = 4096
BM = 2
VECTOR_CORES = 40
LEVELS = (
    "s1_red",
    "s2_chain_fp32",
    "s3_chain_i16",
    "s4_chain_fp32_frag",
    "s5_chain_i16_i32",
)


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    num_logical = _ceildiv(M, BM)
    num_kernels = min(num_logical, VECTOR_CORES)
    num_local_tasks = _ceildiv(num_logical, num_kernels)
    static_m = (M % BM) == 0

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((BM, N), dtype)
                m = T.alloc_fragment((BM, 1), dtype)
                ext_brc = T.alloc_fragment((BM, N), dtype)
                cmp_eq = T.alloc_fragment((BM, N), "bool")
                out_ub = T.alloc_shared((BM,), "int64")

                x_frag = T.alloc_fragment((BM, N), dtype)

                idx_src_f = T.alloc_shared((N,), "float32")
                idx_row_f = T.alloc_shared((1, N), "float32")
                idx_j_f = T.alloc_shared((BM, N), "float32")
                sent_v_f = T.alloc_fragment((BM, N), "float32")
                cand_f = T.alloc_fragment((BM, N), "float32")
                first_f = T.alloc_fragment((BM, 1), "float32")
                first_flat_f = T.alloc_fragment((BM,), "float32")
                m_f = T.alloc_fragment((BM, 1), "float32")

                idx_src_s = T.alloc_shared((N,), "int16")
                idx_row_s = T.alloc_shared((1, N), "int16")
                idx_j_s = T.alloc_shared((BM, N), "int16")
                sent_v_s = T.alloc_fragment((BM, N), "int16")
                cand_s = T.alloc_fragment((BM, N), "int16")
                first_s = T.alloc_fragment((BM, 1), "int16")
                first_flat_s = T.alloc_fragment((BM,), "int16")

                T.arange(idx_src_f, [1], 0)
                T.reshape(idx_src_f, idx_row_f)
                T.vbrc(idx_row_f, idx_j_f)
                T.vbrc(T.float32(BIG), sent_v_f)

                T.arange(idx_src_s, [1], 0)
                T.reshape(idx_src_s, idx_row_s)
                T.vbrc(idx_row_s, idx_j_s)
                T.vbrc(T.int16(BIG_I16), sent_v_s)

                for s in T.serial(num_local_tasks):
                    block_id = s * num_kernels + cid
                    if block_id < num_logical:
                        off_m = block_id * BM
                        if static_m:
                            T.copy(x[off_m : off_m + BM, 0:N], x_ub)
                        else:
                            real_m = T.min(BM, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])

                        if level == "s1_red":
                            T.reduce_min(x_ub, m, dim=1)
                            T.vcast(m, m_f, round_mode="rint")
                            T.reshape(m_f, first_flat_f)
                        elif level == "s2_chain_fp32":
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j_f, sent_v_f, cand_f)
                            T.reduce_min(cand_f, first_f, dim=1)
                            T.reshape(first_f, first_flat_f)
                        elif level == "s4_chain_fp32_frag":
                            T.copy(x_ub, x_frag)
                            T.reduce_min(x_frag, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_frag, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j_f, sent_v_f, cand_f)
                            T.reduce_min(cand_f, first_f, dim=1)
                            T.reshape(first_f, first_flat_f)
                        elif level == "s3_chain_i16":
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j_s, sent_v_s, cand_s)
                            T.reduce_min(cand_s, first_s, dim=1)
                            T.vcast(first_s, m_f, round_mode="rint")
                            T.reshape(m_f, first_flat_f)
                        else:  # s5_chain_i16_i32
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.cast(cmp_eq, "int16", cand_s)
                            T.reduce_min(cand_s, first_s, dim=1)
                            T.vcast(first_s, m_f, round_mode="rint")
                            T.reshape(m_f, first_flat_f)

                        T.vcast(first_flat_f, out_ub, round_mode="rint")
                        if static_m:
                            T.copy(out_ub, out[off_m : off_m + BM])
                        else:
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


def _select_config(M, N, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": BM, "path": "probe", "tile_n": None}


def _argreduce_kernel(M, N, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "s2_chain_fp32"), dtype)
