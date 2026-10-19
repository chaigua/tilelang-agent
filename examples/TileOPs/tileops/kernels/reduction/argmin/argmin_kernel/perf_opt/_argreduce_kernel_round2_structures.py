"""Find a *correct* fused-candidate structure at block_m >= 2 (2048, 4096).

The per-row unrolled write `cand[r + i, j]` (i from `T.Parallel(1, N)`) measured
fast but produced wrong indices, so this file A/Bs four candidate constructions
for both correctness (torch.equal vs CPU argmin) and msprof latency:

  c1_2d        T.Parallel(bm, N) fused ite                 (known to scalarize)
  c2_perrow    per-row unrolled write cand[r + i, j]       (fast, suspect)
  c3_rowbuf    per-row separate (1,N) cand buffers + per-row reduce
  c4_chain     explicit chain (reference, correct)

Usage: PROBE_LEVEL=c3_rowbuf python bench_driver.py --impl <this> --case hidden-state-argmin-float16
       (also runnable standalone: python this_file.py --level c3_rowbuf --check)
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
LEVELS = ("c1_2d", "c2_perrow", "c3_rowbuf", "c4_chain")


def _ceildiv(a, b):
    return (a + b - 1) // b


def build(level, dtype="float16"):
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if level == "c1_2d":

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
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

        elif level == "c3_rowbuf":

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    extra = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    r0 = T.alloc_fragment((1, N), "float32")
                    r1 = T.alloc_fragment((1, N), "float32")
                    f0 = T.alloc_fragment((1, 1), "float32")
                    f1 = T.alloc_fragment((1, 1), "float32")
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
                            for i, j in T.Parallel(1, N):
                                r0[i, j] = T.if_then_else(
                                    x_ub[0, j] == ext_brc[0, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            for i, j in T.Parallel(1, N):
                                r1[i, j] = T.if_then_else(
                                    x_ub[1, j] == ext_brc[1, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(r0, f0, dim=1)
                            T.reduce_min(r1, f1, dim=1)
                            first[0, 0] = f0[0, 0]
                            first[1, 0] = f1[0, 0]
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[off_m : off_m + BM])

        elif level == "c2_perrow":

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
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

        else:  # c4_chain

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    extra = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
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
                            T.reduce_min(x_ub, extra, dim=1)
                            T.vbrc(extra, ext_brc)
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
    return build(os.environ.get("PROBE_LEVEL", "c3_rowbuf"), dtype)


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--check", action="store_true")
    args, _ = parser.parse_known_args()
    k = build(args.level)(BM)
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    y = k(x)
    torch.npu.synchronize()
    ref = torch.argmin(x.cpu(), dim=-1)
    print(f"[{args.level}] {'OK' if torch.equal(y.cpu(), ref) else 'MISMATCH'} "
          f"y0={int(y[0])} ref0={int(ref[0])}", flush=True)


if __name__ == "__main__":
    _main()
