"""AllFwdOp Stage 4 experiment sandbox (per-candidate knob file).

The *behaviour* of this file is fully determined by the ``KNOBS`` dict below;
the experiment runner materialises one file per candidate by rewriting that
dict, so every candidate has its own artifact hash (perf_records traceability).

The Stage 3 baseline module (``../_logical_reduce_kernel.py``) is imported
(read-only) for its builders (``_build_resident`` / ``_build_tiled`` /
``_build_tiled_tail``), golden and the L0/L1/L2/Boundary suites.  Only
``_select_config`` (the tiling single source of truth) is overridden here, so
each candidate changes exactly ONE dispatch structural point.

Candidate knobs
---------------
``E_CAP``      element budget per block for the resident ladder (default 8192)
``RPRIME``     1 = allow the single-tile resident path for
               ``E_CAP < N <= HARD_CAP_ELEMS[dtype]`` (DESIGN section 1.6.4 R')
``TILE_CAP``   column-tile width cap: ``"ecap"`` (= E_CAP, Stage 3) or
               ``"ub"`` (= the measured per-dtype UB hard cap)
``BM_POLICY``  row-block ladder: ``"ladder"`` (Stage 3: largest p with
               ``p*N <= E_CAP``) or ``"cores"`` (prefer the row block that fills
               the physical vector-core grid, i.e. never fewer than
               ``_vector_cores()`` logical blocks when a smaller block can do it)
"""

import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

KNOBS = {
    "E_CAP": 8192,
    "RPRIME": 1,
    "TILE_CAP": 'ecap',
    "BM_POLICY": 'ladder',
}


def _knob(name):
    env = os.environ.get("ALL_" + name)
    return type(KNOBS[name])(env) if env is not None else KNOBS[name]


def _load_base():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "_logical_reduce_kernel.py"
    )
    name = "all_stage3_base"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()

# Re-exported unchanged helpers (harness / driver / tests)
_ceildiv = base._ceildiv
_vector_cores = base._vector_cores
_pick_tile_n = base._pick_tile_n
_prepare_2d = base._prepare_2d
_to_logical_float32 = base._to_logical_float32
golden_all_logical_reduce = base.golden_all_logical_reduce
golden__logical_reduce_kernel = base.golden__logical_reduce_kernel
DEFAULT_ALIGNMENT = base.DEFAULT_ALIGNMENT
LAUNCH_GATE = base.LAUNCH_GATE
HARD_CAP_ELEMS = base.HARD_CAP_ELEMS
_SUPPORTED_DTYPES = base._SUPPORTED_DTYPES
_DEFAULT_VECTOR_CORES = base._DEFAULT_VECTOR_CORES
ASCEND_MODE = base.ASCEND_MODE

__all__ = ["_logical_reduce_kernel", "_select_config", "golden_all_logical_reduce", "KNOBS"]


def _tile_cap(dtype):
    if _knob("TILE_CAP") == "ub":
        return HARD_CAP_ELEMS[dtype]
    return _knob("E_CAP")


def _select_config(M, N, dtype):
    """Element-budget ladder + path switch (knob-driven Stage 4 candidate)."""
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}. "
            f"bool / int32 / int64 / complex inputs must be pre-converted to float32 "
            f"by the Op layer (to_logical_float32)."
        )
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")
    if N < 1:
        raise ValueError(f"reduction width N must be >= 1, got N={N}")

    e_cap = _knob("E_CAP")
    cap = _tile_cap(dtype)
    if N <= e_cap:
        path = "R"
        candidates = [p for p in (1, 2, 4, 8) if p * N <= e_cap]
        block_m = max(candidates) if candidates else 1
        if _knob("BM_POLICY") == "cores":
            cores = _vector_cores()
            filled = [p for p in sorted(candidates) if _ceildiv(M, p) >= cores]
            block_m = filled[0] if filled else min(candidates or [1])
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [
                p
                for p in (16, 32, 64, 128, 256, 512, 1024)
                if p * N <= e_cap and p <= M
            ]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    elif _knob("RPRIME") and N <= HARD_CAP_ELEMS[dtype]:
        # R' (DESIGN 1.6.4): single-tile resident, no tile loop / micro-merge.
        path = "R"
        block_m = 1
        tile_n, num_full, tail = N, 1, 0
    else:
        path = "C"
        block_m = 1
        tile_n = _pick_tile_n(N, min(cap, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)
    elems = block_m * (N if path == "R" else min(tile_n, N))
    if elems > HARD_CAP_ELEMS[dtype]:
        raise ValueError(
            f"({M},{N}) {dtype}: {elems} elements/block exceed the measured UB hard "
            f"limit {HARD_CAP_ELEMS[dtype]} (probe pass point)"
        )
    return {
        "path": path,
        "block_m": block_m,
        "tile_n": tile_n,
        "num_full": num_full,
        "tail": tail,
    }


def _logical_reduce_kernel(M, N, op_kind, dtype):
    if op_kind != "all":
        raise ValueError(
            f"unsupported op_kind {op_kind!r}: this migration unit implements "
            f"the 'all' kind only (any/count_nonzero are separate units)"
        )
    cfg = _select_config(M, N, dtype)
    if cfg["path"] == "R":
        _f = base._build_resident(M, N, dtype, cfg["block_m"])
        _f.msprof_kernel_name = "all_resident"
    elif cfg["tail"] == 0:
        _f = base._build_tiled(M, N, cfg["tile_n"], cfg["num_full"], dtype, cfg["block_m"])
        _f.msprof_kernel_name = "all_tiled"
    else:
        _f = base._build_tiled_tail(
            M, N, cfg["tile_n"], cfg["num_full"], cfg["tail"], dtype, cfg["block_m"]
        )
        _f.msprof_kernel_name = "all_tiled_tail"
    _f.design_config = cfg
    return _f


def build_probe(mode, M, N, dtype, block_m=None):
    raise NotImplementedError("use probe_floor.py for calibration probes")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    base._logical_reduce_kernel = _logical_reduce_kernel
    base._select_config = _select_config
    if _knob("RPRIME") or _knob("E_CAP") != 8192 or _knob("TILE_CAP") != "ecap" \
            or _knob("BM_POLICY") != "ladder":
        # The Stage 3 suite asserts *design-period* path / tile / block_m
        # expectations (L0-3 expects path C + tile_n 8192 for N=16384, the
        # ladder block_m values, ...).  Those are dispatch assertions, not
        # precision gates; with the knobs the dispatch legitimately changes, so
        # the expectation is relaxed while the exact golden comparison inside
        # ``_run_tensor`` / ``_exact_compare`` stays untouched.
        _orig_run_tensor = base._run_tensor

        def _run_tensor(x, dtype_str, tag, expect_path=None, expect_block_m=None,
                        expect_tile_n=None):
            return _orig_run_tensor(
                x, dtype_str, tag, expect_path=None,
                expect_block_m=None, expect_tile_n=None,
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
