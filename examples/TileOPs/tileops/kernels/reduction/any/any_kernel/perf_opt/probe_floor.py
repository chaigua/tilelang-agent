"""Structural floor / op-cost probes for the AnyFwdOp Stage 4 workspace.

NOT a tune workload, never enters ``perf_records.jsonl``.  Results go to
``copy_floor_calibration.jsonl`` (floors) / ``op_cost_calibration.jsonl``.

One prim_func per mode (literal dtypes only -- this toolchain's parser rejects
dtype values carried in local Python variables), mirroring the baseline
path-C structure (persistent 40-kernel grid, ``T.serial`` task loop, column
tiles) with the chain cut progressively:

  empty      grid + per-task output store only            (launch/task floor)
  read       GM->UB tile copies + output store            (read floor)
  reduce     read + reduce_min(x) per tile                (reduce pass price)
  ind_red    read + vcmp + vcast + reduce_min per tile    (baseline chain)
  ind_vmin   read + vcmp + vcast + vmin into a plane      (deferred reduce)
             + one reduce_min at the end
  vmin_f16   same as ind_vmin with float16 indicator/acc plane

Usage:
  python probe_floor.py --mode ind_vmin --M 128 --N 16384 --tile 8192 --bm 1
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tvm import tir  # noqa: E402

CORES = 40


def _ceildiv(a, b):
    return (a + b - 1) // b


def _grid(M, bm):
    num_logical = _ceildiv(M, bm)
    num_kernels = min(num_logical, CORES)
    return num_kernels, _ceildiv(num_logical, num_kernels)


def build_empty(M, N, tile, bm, dtype="float32"):
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                out_ub = T.alloc_shared((bm,), "int8")
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    if start < M:
                        real_m = T.min(bm, M - start)
                        T.fill(out_ub, tir.const(1, "int8"))
                        T.copy(out_ub[0:real_m], out[start : start + real_m])
        return main

    return _func


def build_read(M, N, tile, bm, dtype="float32"):
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, tile), dtype)
                out_ub = T.alloc_shared((bm,), "int8")
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    if start < M:
                        real_m = T.min(bm, M - start)
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile : (t + 1) * tile],
                                x_ub[0:real_m, 0:tile],
                            )
                        T.fill(out_ub, tir.const(1, "int8"))
                        T.copy(out_ub[0:real_m], out[start : start + real_m])
        return main

    return _func


def build_read_static(M, N, tile, bm, dtype="float32"):
    """Read floor with a fully static copy extent (no ``T.min`` on the row count)."""
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)
    assert M % bm == 0

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, tile), dtype)
                out_ub = T.alloc_shared((bm,), "int8")
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    for t in T.serial(num_full):
                        T.copy(
                            x[start : start + bm, t * tile : (t + 1) * tile],
                            x_ub,
                        )
                    T.fill(out_ub, tir.const(1, "int8"))
                    T.copy(out_ub, out[start : start + bm])
        return main

    return _func


def build_reduce(M, N, tile, bm, dtype="float32"):
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, tile), dtype)
                mn = T.alloc_fragment((bm, 1), "float32")
                one_v = T.alloc_fragment((bm,), "float32")
                mn_flat = T.alloc_fragment((bm,), "float32")
                res = T.alloc_fragment((bm,), "float32")
                res_f16 = T.alloc_fragment((bm,), "float16")
                out_ub = T.alloc_shared((bm,), "int8")
                T.fill(one_v, tir.const(1, "float32"))
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    if start < M:
                        real_m = T.min(bm, M - start)
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile : (t + 1) * tile],
                                x_ub[0:real_m, 0:tile],
                            )
                            T.reduce_min(x_ub, mn, dim=1, clear=True)
                        T.reshape(mn, mn_flat)
                        T.vsub(one_v, mn_flat, res)
                        T.vcast(res, res_f16, round_mode="rint")
                        T.vcast(res_f16, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])
        return main

    return _func


def build_ind_red(M, N, tile, bm, dtype="float32"):
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, tile), dtype)
                zero_ub = T.alloc_shared((bm, tile), dtype)
                mask_eq = T.alloc_shared((bm, tile), "bool")
                ind = T.alloc_shared((bm, tile), "float32")
                mn = T.alloc_fragment((bm, 1), "float32")
                one_v = T.alloc_fragment((bm,), "float32")
                mn_flat = T.alloc_fragment((bm,), "float32")
                res = T.alloc_fragment((bm,), "float32")
                res_f16 = T.alloc_fragment((bm,), "float16")
                out_ub = T.alloc_shared((bm,), "int8")
                T.fill(one_v, tir.const(1, "float32"))
                T.fill(zero_ub, tir.const(0, dtype))
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    if start < M:
                        real_m = T.min(bm, M - start)
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile : (t + 1) * tile],
                                x_ub[0:real_m, 0:tile],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.reduce_min(ind, mn, dim=1, clear=True)
                        T.reshape(mn, mn_flat)
                        T.vsub(one_v, mn_flat, res)
                        T.vcast(res, res_f16, round_mode="rint")
                        T.vcast(res_f16, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])
        return main

    return _func


def build_ind_vmin(M, N, tile, bm, dtype="float32"):
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, tile), dtype)
                zero_ub = T.alloc_shared((bm, tile), dtype)
                mask_eq = T.alloc_shared((bm, tile), "bool")
                ind = T.alloc_shared((bm, tile), "float32")
                accp = T.alloc_shared((bm, tile), "float32")
                acc_new = T.alloc_shared((bm, tile), "float32")
                mn = T.alloc_fragment((bm, 1), "float32")
                one_v = T.alloc_fragment((bm,), "float32")
                mn_flat = T.alloc_fragment((bm,), "float32")
                res = T.alloc_fragment((bm,), "float32")
                res_f16 = T.alloc_fragment((bm,), "float16")
                out_ub = T.alloc_shared((bm,), "int8")
                T.fill(one_v, tir.const(1, "float32"))
                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(accp, tir.const(1, "float32"))
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    if start < M:
                        real_m = T.min(bm, M - start)
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile : (t + 1) * tile],
                                x_ub[0:real_m, 0:tile],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.vmin(accp, ind, acc_new)
                            T.copy(acc_new, accp)
                        T.reduce_min(accp, mn, dim=1, clear=True)
                        T.reshape(mn, mn_flat)
                        T.vsub(one_v, mn_flat, res)
                        T.vcast(res, res_f16, round_mode="rint")
                        T.vcast(res_f16, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])
        return main

    return _func


def build_vmin_f16(M, N, tile, bm, dtype="float32"):
    num_full = N // tile
    num_kernels, num_local = _grid(M, bm)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int8")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((bm, tile), dtype)
                zero_ub = T.alloc_shared((bm, tile), dtype)
                mask_eq = T.alloc_shared((bm, tile), "bool")
                ind = T.alloc_shared((bm, tile), "float16")
                accp = T.alloc_shared((bm, tile), "float16")
                acc_new = T.alloc_shared((bm, tile), "float16")
                mn = T.alloc_fragment((bm, 1), "float16")
                one_v = T.alloc_fragment((bm,), "float16")
                mn_flat = T.alloc_fragment((bm,), "float16")
                res = T.alloc_fragment((bm,), "float16")
                out_ub = T.alloc_shared((bm,), "int8")
                T.fill(one_v, tir.const(1, "float16"))
                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(accp, tir.const(1, "float16"))
                for s in T.serial(num_local):
                    start = (s * num_kernels + cid) * bm
                    if start < M:
                        real_m = T.min(bm, M - start)
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile : (t + 1) * tile],
                                x_ub[0:real_m, 0:tile],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.vmin(accp, ind, acc_new)
                            T.copy(acc_new, accp)
                        T.reduce_min(accp, mn, dim=1, clear=True)
                        T.reshape(mn, mn_flat)
                        T.vsub(one_v, mn_flat, res)
                        T.vcast(res, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])
        return main

    return _func


BUILDERS = {
    "empty": build_empty,
    "read": build_read,
    "read_static": build_read_static,
    "reduce": build_reduce,
    "ind_red": build_ind_red,
    "ind_vmin": build_ind_vmin,
    "vmin_f16": build_vmin_f16,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=tuple(BUILDERS))
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--tile", type=int, required=True)
    ap.add_argument("--bm", type=int, default=1)
    ap.add_argument("--launches", type=int, default=35)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    kernel = BUILDERS[args.mode](args.M, args.N, args.tile, args.bm)()
    print("PROBE: {} M={} N={} tile={} bm={}".format(
        args.mode, args.M, args.N, args.tile, args.bm), flush=True)
    x = torch.randn(args.M, args.N, dtype=torch.float32, device="npu")
    x[0] = 0.0
    for _ in range(3):
        y = kernel(x)
    torch.npu.synchronize()
    for _ in range(args.launches):
        y = kernel(x)
        torch.npu.synchronize()
    if args.check:
        ref = torch.any(x.cpu(), dim=-1)
        ok = torch.equal(y.cpu().bool(), ref)
        print("PROBE_CHECK:", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
