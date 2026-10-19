"""Calibration probes for the count_nonzero Stage 4 (NOT tune workloads).

These probes reuse the production path-R structure (block_m rows per task,
persistent grid) and ablate the vector chain and/or the GM traffic so the
structural floor of each benchmark shape can be attributed quantitatively:

  mode ``empty``    : per-task output store only (launch + task + tiny-DMA floor)
  mode ``copyout``  : GM->UB copy + per-task output store (adds MTE2)
  mode ``read``     : GM->UB copy + reduce_sum(x_ub) + output store
                      (adds ONE f32 reduction plane; no vcmp/vcast chain)
  mode ``copy``     : GM->UB copy + UB->GM copy of the full tile (1:1 DMA floor)

The difference of the measured Task Durations gives the per-stage unit price:
  copyout - empty  -> MTE2 unit price (per tile)
  read - copyout   -> reduce_sum pass price
  full - read      -> vcmp + vcast pass price

Each mode gets its OWN ``prim_func`` (compile-time buffer definitions inside a
prim_func are not parseable by this toolchain -- TRAP-primfunc-compile-time-if-
buffer-def).  Nothing here is a candidate kernel; results go to
``copy_floor_calibration.jsonl`` / ``op_cost_calibration.jsonl``, never into
``perf_records.jsonl``.
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


def _prepare_2d(x, dim):
    """Op-layer emulation (same contract as the production kernel module)."""
    if dim is None:
        dims = []
    elif isinstance(dim, int):
        dims = [dim % x.dim()]
    else:
        dims = sorted({d % x.dim() for d in dim})
    keep = [d for d in range(x.dim()) if d not in dims]
    if dims:
        xd = x.permute(keep + dims).contiguous()
    else:
        xd = x.reshape(-1).contiguous()
    M = 1
    for d in keep:
        M *= x.shape[d]
    N = xd.numel() // M
    return xd.reshape(M, N), tuple(x.shape[d] for d in keep)


def _select_config(M, N, dtype):
    """Same resident ladder as the production kernel (probe uses path R only)."""
    candidates = [p for p in (1, 2, 4, 8) if p * N <= min(E_CAP, 24576)]
    block_m = max(candidates) if candidates else 1
    if _ceildiv(M, block_m) > 80:
        extended = [
            p for p in (16, 32, 64, 128, 256, 512, 1024)
            if p * N <= min(E_CAP, 24576) and p <= M
        ]
        block_m = max(extended + [block_m])
    block_m = min(block_m, M)
    return {"path": "R", "block_m": block_m, "tile_n": N, "num_full": 1, "tail": 0}


def _grid(M, block_m):
    num_logical = _ceildiv(M, block_m)
    num_kernels = min(num_logical, _vector_cores())
    return num_kernels, _ceildiv(num_logical, num_kernels)


def _build_empty(M, N, dtype, block_m):
    num_kernels, num_local_tasks = _grid(M, block_m)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                out_ub = T.alloc_shared((block_m,), "int64")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(out_ub, tir.const(0, "int64"))
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
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                out_ub = T.alloc_shared((block_m,), "int64")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.fill(out_ub, tir.const(0, "int64"))
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
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.reduce_sum(x_ub, zc, dim=1, clear=True)
                        T.reshape(zc, cnt)
                        T.vcast(cnt, out_ub, round_mode="rint")
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
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                y_ub = T.alloc_shared((block_m, N), dtype)
                out_ub = T.alloc_shared((block_m,), "int64")
                for s in T.serial(num_local_tasks):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.copy(x_ub[0:real_m, 0:N], y_ub[0:real_m, 0:N])
                        T.fill(out_ub, tir.const(0, "int64"))
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func()


_BUILDERS = {
    "empty": _build_empty,
    "copyout": _build_copyout,
    "read": _build_read,
    "copy": _build_copy,
}


def build_probe(mode, M, N, dtype, block_m=None):
    if mode not in _BUILDERS:
        raise ValueError(f"unknown probe mode {mode!r}")
    if block_m is None:
        block_m = _select_config(M, N, dtype)["block_m"]
    return _BUILDERS[mode](M, N, dtype, block_m)
