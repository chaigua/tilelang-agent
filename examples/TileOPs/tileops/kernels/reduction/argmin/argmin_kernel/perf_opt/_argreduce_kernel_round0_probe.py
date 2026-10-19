"""Diagnostic probe: marginal vector cost of each op in the resident chain.

Same (M, N, bm, grid) as the hidden-state fp16 workload (2048, 4096) fp16
bm=2, so per-level msprof `Task Duration` differences attribute cost to
individual ops.  Level definitions (all consume into `out` to defeat DCE):

  pa_copy_reduce     copy + reduce_min(x_ub)                    -- 1 reduce
  pb_copy_xfrag      copy + same-dtype x_frag copy + reduce     -- +1 full copy
  pc_chain           copy + reduce + vbrc + vcmp + vselect + reduce(cand)
                                                                -- full chain, no x_frag
  pd_baseline        copy + x_frag copy + full chain            -- == baseline
  pe_two_reduces     copy + vcast fp32 + reduce + reduce        -- 2 reduces, no cmp/select
  pf_no_secondred    copy + reduce + vbrc + vcmp + vselect      -- no 2nd reduce

Usage: python _argreduce_kernel_round0_probe.py --level <name>
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BIG = 2**30

M = 2048
N = 4096
BM = 2
VECTOR_CORES = 40
LEVELS = (
    "pa_copy_reduce",
    "pb_copy_xfrag",
    "pc_chain",
    "pd_baseline",
    "pe_two_reduces",
    "pf_no_secondred",
)


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    num_logical = _ceildiv(M, BM)
    num_kernels = min(num_logical, VECTOR_CORES)
    num_local_tasks = _ceildiv(num_logical, num_kernels)

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
                idx_src = T.alloc_shared((N,), "float32")
                idx_row = T.alloc_shared((1, N), "float32")
                idx_j = T.alloc_shared((BM, N), "float32")
                sent_v = T.alloc_fragment((BM, N), "float32")
                cand = T.alloc_fragment((BM, N), "float32")
                first = T.alloc_fragment((BM, 1), "float32")
                m32 = T.alloc_fragment((BM, 1), "float32")
                first_flat = T.alloc_fragment((BM,), "float32")
                out_ub = T.alloc_shared((BM,), "int64")

                x_frag = T.alloc_fragment((BM, N), dtype)
                x_f32 = T.alloc_fragment((BM, N), "float32")

                T.arange(idx_src, [1], 0)
                T.reshape(idx_src, idx_row)
                T.vbrc(idx_row, idx_j)
                T.vbrc(T.float32(BIG), sent_v)

                for s in T.serial(num_local_tasks):
                    block_id = s * num_kernels + cid
                    if block_id < num_logical:
                        off_m = block_id * BM
                        real_m = T.min(BM, M - off_m)
                        T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])

                        if level == "pa_copy_reduce":
                            T.reduce_min(x_ub, m, dim=1)
                            T.vcast(m, m32, round_mode="rint")
                            T.reshape(m32, first)
                        elif level == "pb_copy_xfrag":
                            T.copy(x_ub, x_frag)
                            T.reduce_min(x_frag, m, dim=1)
                            T.vcast(m, m32, round_mode="rint")
                            T.reshape(m32, first)
                        elif level == "pc_chain":
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                        elif level == "pd_baseline":
                            T.copy(x_ub, x_frag)
                            T.reduce_min(x_frag, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_frag, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                        elif level == "pe_two_reduces":
                            T.vcast(x_ub, x_f32, round_mode="rint")
                            T.reduce_min(x_f32, m, dim=1)
                            T.reduce_min(x_f32, first, dim=1)
                        else:  # pf_no_secondred
                            T.reduce_min(x_ub, m, dim=1)
                            T.vbrc(m, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, m, dim=1)
                            T.vcast(m, m32, round_mode="rint")
                            T.reshape(m32, first)

                        T.reshape(first, first_flat)
                        T.vcast(first_flat, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


def _select_config(M, N, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": BM, "path": "probe", "tile_n": None}


def _argreduce_kernel(M, N, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "pd_baseline"), dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--launches", type=int, default=25)
    args, _ = parser.parse_known_args()

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    kernel = build(args.level, args.dtype)(BM)
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    for _ in range(3):
        y = kernel(x)
    torch.npu.synchronize()
    for _ in range(args.launches):
        y = kernel(x)
        torch.npu.synchronize()
    print("PROBE_OK level={} out0={}".format(args.level, int(y[0])), flush=True)


if __name__ == "__main__":
    main()
