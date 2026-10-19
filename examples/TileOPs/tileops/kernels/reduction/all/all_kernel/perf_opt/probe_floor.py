"""Calibration probes for the AllFwdOp Stage 4 (NOT tune workloads).

These probes reuse the production path-R structure (block_m rows per task,
persistent grid) and ablate the vector chain and/or the GM traffic so the
structural floor of each benchmark shape can be attributed quantitatively:

  mode ``empty``    : per-task output store only (launch + task + tiny-DMA floor)
  mode ``copyout``  : GM->UB copy + per-task output store (adds MTE2)
  mode ``read``     : GM->UB copy + reduce_max(x_ub) + output store
                      (adds ONE f32 reduction plane; no vcmp/vcast chain)
  mode ``copy``     : GM->UB copy + UB->GM copy of the full tile (1:1 DMA floor)

The difference of the measured Task Durations gives the per-stage unit price:
  copyout - empty  -> MTE2 unit price (per tile)
  read - copyout   -> reduce pass price
  full - read      -> vcmp + vcast pass price

Each mode gets its OWN ``prim_func`` (compile-time buffer definitions inside a
prim_func are not parseable by this toolchain -- TRAP-primfunc-compile-time-if-
buffer-def).  Nothing here is a candidate kernel; results go to
``copy_floor_calibration.jsonl`` / ``op_cost_calibration.jsonl``, never into
``perf_records.jsonl``.

Output carrier matches the production kernel: ``(M,) int8`` (0/1), restored to
bool by the wrapper.  ``f32 -> i8`` has no ``vcast`` table entry, so the probe
uses the same ``f32 -> f16 -> i8`` round trip as the kernel.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tvm import tir  # noqa: E402

_DEFAULT_VECTOR_CORES = 40
E_CAP = 8192


def _ceildiv(a, b):
    return (a + b - 1) // b


def _vector_cores():
    try:
        from tilelang.utils.npu_arch import AscendArch

        return int(AscendArch().aivector_core_num)
    except Exception:  # noqa: BLE001
        return _DEFAULT_VECTOR_CORES


def _select_config(M, N, dtype, cap=E_CAP):
    """Same resident ladder as the production kernel (probe uses path R only)."""
    candidates = [p for p in (1, 2, 4, 8) if p * N <= cap]
    block_m = max(candidates) if candidates else 1
    if _ceildiv(M, block_m) > 80:
        extended = [p for p in (16, 32, 64, 128, 256, 512, 1024) if p * N <= cap and p <= M]
        block_m = max(extended + [block_m])
    block_m = min(block_m, M)
    return {"path": "R", "block_m": block_m, "tile_n": N, "num_full": 1, "tail": 0}


def _grid(M, block_m):
    num_logical = _ceildiv(M, block_m)
    num_kernels = min(num_logical, _vector_cores())
    return num_kernels, _ceildiv(num_logical, num_kernels)


# --- Op-layer emulation (the driver loads this file as the ``--impl`` module
# for probe runs, so the two helpers it needs must exist here) --------------
def _to_logical_float32(x):
    import torch

    if x.dtype in (torch.bool, torch.int32, torch.int64):
        return x.to(torch.float32)
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


def _normalize_dims(dim, ndim):
    if dim is None:
        return []
    if isinstance(dim, int):
        dim = [dim]
    return sorted({d % ndim for d in dim})


def _prepare_2d(x, dim):
    """Op-layer emulation: movedim reduced dims last, contiguous, reshape."""
    dims = _normalize_dims(dim, x.dim())
    if dim is None:
        xd = x.reshape(-1).contiguous()
        return xd.reshape(1, xd.numel()), ()
    keep = [d for d in range(x.dim()) if d not in dims]
    xd = x.permute(keep + dims).contiguous()
    M = 1
    for d in keep:
        M *= x.shape[d]
    N = xd.numel() // M
    return xd.reshape(M, N), tuple(x.shape[d] for d in keep)


def _build_empty(M, N, dtype, block_m):
    num_kernels, num_local_tasks = _grid(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                out_ub = T.alloc_shared((block_m,), "int8")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(out_ub, tir.const(0, "int8"))
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func()


def _build_copyout(M, N, dtype, block_m):
    num_kernels, num_local_tasks = _grid(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                out_ub = T.alloc_shared((block_m,), "int8")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.fill(out_ub, tir.const(0, "int8"))
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func()


def _build_read(M, N, dtype, block_m):
    num_kernels, num_local_tasks = _grid(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_flat = T.alloc_fragment((block_m,), "float32")
                out_f16 = T.alloc_fragment((block_m,), "float16")
                out_ub = T.alloc_shared((block_m,), "int8")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.reduce_max(x_ub, mx, dim=1, clear=True)
                        T.reshape(mx, mx_flat)
                        T.vcast(mx_flat, out_f16, round_mode="rint")
                        T.vcast(out_f16, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func()


def _build_copy(M, N, dtype, block_m):
    num_kernels, num_local_tasks = _grid(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                y_ub = T.alloc_shared((block_m, N), dtype)
                out_ub = T.alloc_shared((block_m,), "int8")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.copy(x_ub[0:real_m, 0:N], y_ub[0:real_m, 0:N])
                        T.fill(out_ub, tir.const(0, "int8"))
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func()


def _build_read_f16(M, N, dtype, block_m):
    """GM->UB copy + reduce_max on a **float16** plane of the same element count.

    Isolates the indicator-plane dtype price: the production chain reduces a
    f32 indicator plane; a f16 plane (exact for the 0/1 indicator) halves the
    reduce input bytes.
    """
    num_kernels, num_local_tasks = _grid(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                ind_ub = T.alloc_shared((block_m, N), "float16")
                mx = T.alloc_fragment((block_m, 1), "float16")
                mx_flat = T.alloc_fragment((block_m,), "float16")
                out_ub = T.alloc_shared((block_m,), "int8")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.vcast(x_ub, ind_ub, round_mode="rint")
                        T.reduce_max(ind_ub, mx, dim=1, clear=True)
                        T.reshape(mx, mx_flat)
                        T.vcast(mx_flat, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func()


_BUILDERS = {
    "empty": _build_empty,
    "copyout": _build_copyout,
    "read": _build_read,
    "copy": _build_copy,
    "read_f16": _build_read_f16,
}


def build_probe(mode, M, N, dtype, block_m=None, cap=E_CAP):
    if mode not in _BUILDERS:
        raise ValueError(f"unknown probe mode {mode!r}")
    if block_m is None:
        block_m = _select_config(M, N, dtype, cap=cap)["block_m"]
    return _BUILDERS[mode](M, N, dtype, block_m)
