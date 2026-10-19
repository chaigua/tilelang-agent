"""Round-2 correct fused structure: per-row candidate buffers + per-row reduce.

Root cause of the round-1 "fast" per-row form: writing `cand[r + i, j]` from an
unrolled per-row `T.Parallel(1, N)` loop into a shared (bm, N) fragment
miscompiles (measured: y0 = -3 for (2048, 4096) fp16).  The 2-D
`T.Parallel(bm, N)` form is correct but scalarizes at bm >= 2.

This file keeps the fused `if_then_else` candidate (no vcmp / vselect / fp32
index ramps) but gives every row its own (1, N) candidate buffer and its own
`reduce_min`, which is correct and still far leaner than the explicit chain.

Env: PROBE_LEVEL=<fp16|bf16>_rb<bm>[_p<stages>]
Usage: PROBE_LEVEL=fp16_rb4 python bench_driver.py --impl <this> \
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
VECTOR_CORES = 40
_DT = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def _ceildiv(a, b):
    return (a + b - 1) // b


def parse(level):
    m = re.fullmatch(r"(fp16|bf16|fp32)_rb(\d+)(?:_p(\d+))?", level)
    if not m:
        raise ValueError(level)
    return m.group(1), int(m.group(2)), int(m.group(3) or 0)



def _alloc_rows(BM, N, dtype):
    """Allocate one (1, N) candidate buffer per row (executed at trace time)."""
    return [T.alloc_fragment((1, N), dtype) for _ in range(BM)]


def _alloc_rowf(BM):
    return [T.alloc_fragment((1, 1), "float32") for _ in range(BM)]


def build(level, M=2048, N=4096, dt=None):
    tag, BM, stages = parse(level)
    dtype = dt or _DT[tag]
    N = ((N + 255) // 256) * 256
    is_bf16 = dtype == "bfloat16"
    use_pipe = bool(stages > 0)
    num_logical = _ceildiv(M, BM)
    grid = min(num_logical, VECTOR_CORES)
    local = _ceildiv(num_logical, grid)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        if is_bf16:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    work = T.alloc_fragment((BM, N), "float32")
                    extra = T.alloc_fragment((BM, 1), "float32")
                    ext_brc = T.alloc_fragment((BM, N), "float32")
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")
                    rows = _alloc_rows(BM, N, "float32")
                    rf = _alloc_rowf(BM)

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            real_m = T.min(BM, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.vcast(x_ub, work, round_mode="rint")
                            T.reduce_min(work, extra, dim=1)
                            T.vbrc(extra, ext_brc)
                            for r in range(BM):
                                for i, j in T.Parallel(1, N):
                                    rows[r][i, j] = T.if_then_else(
                                        work[r, j] == ext_brc[r, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(rows[r], rf[r], dim=1)
                                first[r, 0] = rf[r][0, 0]
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(grid, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((BM, N), dtype)
                    extra = T.alloc_fragment((BM, 1), dtype)
                    ext_brc = T.alloc_fragment((BM, N), dtype)
                    first = T.alloc_fragment((BM, 1), "float32")
                    first_flat = T.alloc_fragment((BM,), "float32")
                    out_ub = T.alloc_shared((BM,), "int64")
                    rows = _alloc_rows(BM, N, "float32")
                    rf = _alloc_rowf(BM)

                    for s in T.serial(local):
                        block_id = s * grid + cid
                        if block_id < num_logical:
                            off_m = block_id * BM
                            real_m = T.min(BM, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.reduce_min(x_ub, extra, dim=1)
                            T.vbrc(extra, ext_brc)
                            for r in range(BM):
                                for i, j in T.Parallel(1, N):
                                    rows[r][i, j] = T.if_then_else(
                                        x_ub[r, j] == ext_brc[r, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(rows[r], rf[r], dim=1)
                                first[r, 0] = rf[r][0, 0]
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


def _select_config(M_, N_, dtype):
    """Probe shim so the shared bench driver can drive this module."""
    return {"block_m": parse(os.environ.get("PROBE_LEVEL", "fp16_rb2"))[1],
            "path": "probe", "tile_n": None}


def _argreduce_kernel(M_, N_, op_kind, dtype):
    return build(os.environ.get("PROBE_LEVEL", "fp16_rb2"), M_, N_, dtype)


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True)
    parser.add_argument("--M", type=int, default=2048)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--dtype", default=None)
    args, _ = parser.parse_known_args()
    _, bm, _ = parse(args.level)
    torch_dtype = _DT[parse(args.level)[0]]
    N = ((args.N + 255) // 256) * 256
    k = build(args.level, args.M, args.N, args.dtype)(bm)
    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch_dtype, device="npu")
    if N != args.N:
        xp = torch.full((args.M, N), float("inf"), dtype=torch_dtype, device="npu")
        xp[:, 0 : args.N] = x
    else:
        xp = x
    y = k(xp)
    torch.npu.synchronize()
    ref = torch.argmin(x.cpu(), dim=-1)
    print(f"[{args.level}] {'OK' if torch.equal(y.cpu(), ref) else 'MISMATCH'} "
          f"y0={int(y[0])} ref0={int(ref[0])}", flush=True)


if __name__ == "__main__":
    _main()
