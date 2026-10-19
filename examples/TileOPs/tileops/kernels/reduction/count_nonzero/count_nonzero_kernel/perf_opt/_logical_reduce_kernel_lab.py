"""Count-nonzero Stage 4 experiment sandbox (per-candidate knob file).

The *behaviour* of this file is fully determined by the ``KNOBS`` dict below;
the experiment runner materialises one file per candidate by rewriting that
dict, so every candidate has its own artifact hash (perf_records traceability).

The baseline Stage 3 kernel module is imported (read-only) for its helpers
(``_prepare_2d`` / ``golden_count_nonzero`` / the L0/L1/L2/Boundary suites).

Structural change carried by every candidate in this sandbox (vs Stage 3):
**batched output write** -- the per-task ``(block_m,)`` int64 stores are staged
in a per-core ``out_all`` UB buffer and flushed with ONE ``T.copy`` at the end
of the core instead of one tiny 16-byte DMA per task.  This requires a
per-core *contiguous* row chunk (``start = (cid * L + s) * block_m``), so the
grid-stride mapping of Stage 3 is replaced by a contiguous mapping.

Candidate knobs
---------------
``E_CAP``      element budget per block for the resident ladder (default 8192)
``UB_ELEMS``   hard per-block element cap (24576 fp16/fp32, 16384 bf16)
``RPRIME``     1 = allow the single-tile resident path for E_CAP < N <= UB_ELEMS
"""

import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

KNOBS = {
    "E_CAP": 8192,
    "UB_ELEMS": 24576,
    "RPRIME": 1,
    "OUT_MODE": 1,
    "PIPE": 0,
}


def _knob(name):
    env = os.environ.get("CNZ_" + name)
    return type(KNOBS[name])(env) if env is not None else KNOBS[name]


def _load_base():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "_logical_reduce_kernel.py"
    )
    name = "cnz_stage3_base"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tvm import tir  # noqa: E402

# Re-exported unchanged helpers (harness / driver / tests)
_ceildiv = base._ceildiv
_vector_cores = base._vector_cores
_pick_tile_n = base._pick_tile_n
_prepare_2d = base._prepare_2d
_to_logical_float32 = base._to_logical_float32
golden_count_nonzero = base.golden_count_nonzero
MAX_EXACT_N = base.MAX_EXACT_N
DEFAULT_ALIGNMENT = base.DEFAULT_ALIGNMENT
LAUNCH_GATE = base.LAUNCH_GATE
_SUPPORTED_DTYPES = base._SUPPORTED_DTYPES
_DEFAULT_VECTOR_CORES = base._DEFAULT_VECTOR_CORES
ASCEND_MODE = base.ASCEND_MODE

__all__ = ["_logical_reduce_kernel", "_select_config", "golden_count_nonzero", "KNOBS"]

# bf16 vcmp operands are lifted to f32 -> 12 B/element (CONST-reduce-ub-per-element)
_UB_PER_ELEM = {"float16": 8, "float32": 8, "bfloat16": 12}


def _ub_elems(dtype):
    return min(_knob("UB_ELEMS"), 196608 // _UB_PER_ELEM[dtype])


def _select_config(M, N, dtype):
    """Element-budget ladder + path switch + (optional) single-tile resident."""
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")
    if N < 1:
        raise ValueError(f"reduction width N must be >= 1, got N={N}")
    if N > MAX_EXACT_N:
        raise ValueError(f"N={N} exceeds 2**24")

    e_cap = _knob("E_CAP")
    if N <= e_cap:
        path = "R"
        candidates = [p for p in (1, 2, 4, 8) if p * N <= e_cap]
        block_m = max(candidates) if candidates else 1
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [
                p for p in (16, 32, 64, 128, 256, 512, 1024)
                if p * N <= e_cap and p <= M
            ]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    elif _knob("RPRIME") and N <= _ub_elems(dtype):
        # Single-tile resident over-budget path (DESIGN 1.6.4 alternative R').
        path = "R"
        block_m = 1
        tile_n, num_full, tail = N, 1, 0
    else:
        path = "C"
        block_m = 1
        tile_n = _pick_tile_n(N, min(e_cap, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)
    return {
        "path": path,
        "block_m": block_m,
        "tile_n": tile_n,
        "num_full": num_full,
        "tail": tail,
    }


def _resolve_block_m(block_m, default_bm, tile_n, M):
    if block_m is None:
        return default_bm
    block_m = min(block_m, M)
    if block_m * tile_n > _knob("E_CAP"):
        return default_bm
    return block_m


# ---------------------------------------------------------------------------
# Resident builder (path R; bm=1 covers the R' single-tile over-budget variant)
# ---------------------------------------------------------------------------
def _build_resident(M, N, dtype, default_bm):
    """Whole row resident per tile; batched (single) output DMA per core."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, N, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                zero_ub = T.alloc_shared((block_m, N), dtype)
                mask_eq = T.alloc_shared((block_m, N), "bool")
                ind = T.alloc_shared((block_m, N), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_task = T.alloc_shared((block_m,), "int64")
                out_all = T.alloc_shared((num_local_tasks * block_m,), "int64")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (cid * num_local_tasks + s) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")
                        T.reduce_sum(ind, zc, dim=1, clear=True)
                        T.reshape(zc, cnt)
                        T.vsub(n_v, cnt, cnt)
                        T.vcast(cnt, out_task, round_mode="rint")
                        T.copy(out_task[0:real_m], out_all[s * block_m : s * block_m + real_m])

                core_start = cid * num_local_tasks * block_m
                if core_start < M:
                    core_rows = T.min(num_local_tasks * block_m, M - core_start)
                    T.copy(
                        out_all[0:core_rows],
                        out[core_start : core_start + core_rows],
                    )

        return main

    return _func


# ---------------------------------------------------------------------------
# Pipelined resident builder (path R; T.Pipelined(num_stages=2))
# ---------------------------------------------------------------------------
def _build_resident_pipe(M, N, dtype, default_bm, stages=2):
    """Same chain as ``_build_resident`` but with an explicit software pipeline.

    Stage 3 relies on the Developer auto-multi-buffer pass, whose measured
    MTE2/V overlap on this loop is only ~7% (aiv_time 22.56 vs vec 12.33 +
    mte2 11.61).  ``T.Pipelined`` states the same pipeline explicitly.
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, N, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                zero_ub = T.alloc_shared((block_m, N), dtype)
                mask_eq = T.alloc_shared((block_m, N), "bool")
                ind = T.alloc_shared((block_m, N), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.Pipelined(num_local_tasks, num_stages=stages):
                    start = (s * num_kernels + cid) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.copy(x[start : start + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")
                        T.reduce_sum(ind, zc, dim=1, clear=True)
                        T.reshape(zc, cnt)
                        T.vsub(n_v, cnt, cnt)
                        T.vcast(cnt, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Column-tiled builder (path C; N > UB element cap), batched output
# ---------------------------------------------------------------------------
def _build_tiled(M, N, tile_n, num_full, tail, dtype, default_bm):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, tile_n, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, tile_n), dtype)
                zero_ub = T.alloc_shared((block_m, tile_n), dtype)
                mask_eq = T.alloc_shared((block_m, tile_n), "bool")
                ind = T.alloc_shared((block_m, tile_n), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_task = T.alloc_shared((block_m,), "int64")
                out_all = T.alloc_shared((num_local_tasks * block_m,), "int64")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (cid * num_local_tasks + s) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(acc, tir.const(0, "float32"))
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile_n : (t + 1) * tile_n],
                                x_ub[0:real_m, 0:tile_n],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.reduce_sum(ind, zc, dim=1, clear=True)
                            T.vadd(acc, zc, new_acc)
                            T.copy(new_acc, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(n_v, acc_flat, cnt)
                        T.vcast(cnt, out_task, round_mode="rint")
                        T.copy(out_task[0:real_m], out_all[s * block_m : s * block_m + real_m])

                core_start = cid * num_local_tasks * block_m
                if core_start < M:
                    core_rows = T.min(num_local_tasks * block_m, M - core_start)
                    T.copy(
                        out_all[0:core_rows],
                        out[core_start : core_start + core_rows],
                    )

        return main

    return _func


def _build_tiled_tail(M, N, tile_n, num_full, tail, dtype, default_bm):
    """Column-tiled with static-width tail block (independent buffer group)."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None):
        block_m = _resolve_block_m(block_m, default_bm, tile_n, M)
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, _vector_cores())
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, tile_n), dtype)
                zero_ub = T.alloc_shared((block_m, tile_n), dtype)
                mask_eq = T.alloc_shared((block_m, tile_n), "bool")
                ind = T.alloc_shared((block_m, tile_n), "float32")
                x_tail = T.alloc_shared((block_m, tail), dtype)
                zero_tail = T.alloc_shared((block_m, tail), dtype)
                mask_tail = T.alloc_shared((block_m, tail), "bool")
                ind_tail = T.alloc_shared((block_m, tail), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                zc_tail = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc_t = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_task = T.alloc_shared((block_m,), "int64")
                out_all = T.alloc_shared((num_local_tasks * block_m,), "int64")

                T.fill(zero_ub, tir.const(0, dtype))
                T.fill(zero_tail, tir.const(0, dtype))
                T.fill(n_v, tir.const(N, "float32"))

                for s in T.serial(num_local_tasks):
                    start = (cid * num_local_tasks + s) * block_m
                    if start < M:
                        real_m = T.min(block_m, M - start)
                        T.fill(acc, tir.const(0, "float32"))
                        for t in T.serial(num_full):
                            T.copy(
                                x[start : start + real_m, t * tile_n : (t + 1) * tile_n],
                                x_ub[0:real_m, 0:tile_n],
                            )
                            T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                            T.vcast(mask_eq, ind, round_mode="rint")
                            T.reduce_sum(ind, zc, dim=1, clear=True)
                            T.vadd(acc, zc, new_acc)
                            T.copy(new_acc, acc)
                        T.copy(
                            x[
                                start : start + real_m,
                                num_full * tile_n : num_full * tile_n + tail,
                            ],
                            x_tail[0:real_m, 0:tail],
                        )
                        T.vcmp(x_tail, zero_tail, mask_tail, "eq")
                        T.vcast(mask_tail, ind_tail, round_mode="rint")
                        T.reduce_sum(ind_tail, zc_tail, dim=1, clear=True)
                        T.vadd(acc, zc_tail, new_acc_t)
                        T.copy(new_acc_t, acc)
                        T.reshape(acc, acc_flat)
                        T.vsub(n_v, acc_flat, cnt)
                        T.vcast(cnt, out_task, round_mode="rint")
                        T.copy(out_task[0:real_m], out_all[s * block_m : s * block_m + real_m])

                core_start = cid * num_local_tasks * block_m
                if core_start < M:
                    core_rows = T.min(num_local_tasks * block_m, M - core_start)
                    T.copy(
                        out_all[0:core_rows],
                        out[core_start : core_start + core_rows],
                    )

        return main

    return _func


def _logical_reduce_kernel(M, N, op_kind, dtype):
    if op_kind != "count_nonzero":
        raise ValueError(f"unsupported op_kind {op_kind!r}")
    cfg = _select_config(M, N, dtype)
    if _knob("PIPE") and cfg["path"] == "R":
        _f = _build_resident_pipe(M, N, dtype, cfg["block_m"], stages=_knob("PIPE"))
        _f.design_config = cfg
        _f.msprof_kernel_name = "count_nonzero_resident_pipe"
        return _f
    if _knob("OUT_MODE") == 0:
        # Reference output structure of Stage 3: grid-stride task mapping with
        # one tiny int64 DMA per task (isolates the R' effect from the batched
        # output change).
        if cfg["path"] == "R":
            _f = base._build_resident(M, N, dtype, cfg["block_m"])
        elif cfg["tail"] == 0:
            _f = base._build_tiled(
                M, N, cfg["tile_n"], cfg["num_full"], dtype, cfg["block_m"]
            )
        else:
            _f = base._build_tiled_tail(
                M, N, cfg["tile_n"], cfg["num_full"], cfg["tail"], dtype,
                cfg["block_m"],
            )
        _f.design_config = cfg
        _f.msprof_kernel_name = "count_nonzero_ref"
        return _f
    if cfg["path"] == "R":
        _f = _build_resident(M, N, dtype, cfg["block_m"])
        _f.msprof_kernel_name = "count_nonzero_resident"
    elif cfg["tail"] == 0:
        _f = _build_tiled(M, N, cfg["tile_n"], cfg["num_full"], 0, dtype, cfg["block_m"])
        _f.msprof_kernel_name = "count_nonzero_tiled"
    else:
        _f = _build_tiled_tail(
            M, N, cfg["tile_n"], cfg["num_full"], cfg["tail"], dtype, cfg["block_m"]
        )
        _f.msprof_kernel_name = "count_nonzero_tiled_tail"
    _f.design_config = cfg
    return _f


def build_probe(mode, M, N, dtype):
    raise NotImplementedError("use probe_floor.py for calibration probes")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    base._logical_reduce_kernel = _logical_reduce_kernel
    base._select_config = _select_config
    if _knob("RPRIME") or _knob("E_CAP") != 8192:
        # The Stage 3 suite asserts the *design-period* path/tile expectations
        # (L0-3/L0-4 expect path C for N=16384, ladder block_m values, ...).
        # Those are path-selection assertions, not precision gates; with the
        # R' / E_CAP candidates the path legitimately changes, so the
        # expectation is relaxed while the exact int64 golden comparison in
        # ``_run_tensor`` stays untouched.
        _orig_run_tensor = base._run_tensor

        def _run_tensor(x, dtype_str, tag, expect_path=None, expect_block_m=None,
                        expect_tile_n=None):
            keep_bm = None if _knob("E_CAP") != 8192 else expect_block_m
            return _orig_run_tensor(
                x, dtype_str, tag, expect_path=None,
                expect_block_m=keep_bm, expect_tile_n=None,
            )

        base._run_tensor = _run_tensor
    base.run_L0()
    if args.level == "all":
        base.run_L1()
        base.run_L2()
        base.run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
