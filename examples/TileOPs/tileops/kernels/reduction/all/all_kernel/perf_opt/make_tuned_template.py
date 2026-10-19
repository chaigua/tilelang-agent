"""Build the Stage 4 tuned template from the Stage 3 baseline (single point).

The template keeps the Stage 3 file structure (builders, golden, L0/L1/L2/
Boundary suites, ``main``) and changes exactly three things:

  1. the **indicator plane dtype** becomes ``float16`` (exactly 0.0/1.0, so the
     reduce and the ``1 - max`` complement stay bit-exact) -- this also buys
     the UB headroom that makes a 16384-element single tile compile;
  2. ``_select_config`` gains the Stage 4 dispatch knobs
     (``TUNED_RPRIME`` / ``TUNED_TILE_CAP_MODE``) as **module-level constants**,
     so the wrapper toggle block can pin them;
  3. ``__all__`` exports ``_select_config`` and the tuned constants.

Candidates are materialised from this template by flipping only the constants
(``make_variants.py``); the winner is delivered as-is.

Usage: python make_tuned_template.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "_logical_reduce_kernel.py"))
OUT = os.path.join(HERE, "_logical_reduce_kernel_tuned_tpl.py")

# ---------------------------------------------------------------------------
# 1. indicator plane: f32 -> f16 (builders)
# ---------------------------------------------------------------------------
IND_REPLACEMENTS = [
    (
        '''                ind = T.alloc_shared((block_m, N), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")''',
        '''                # Stage 4: the zero-hit indicator is exactly 0.0/1.0, so it is
                # materialised and reduced in float16 -- half the reduce input
                # bytes, one fewer epilogue cast, and 2 B/element of UB headroom
                # (which is what makes a 16384-element single tile fit UB).
                ind = T.alloc_shared((block_m, N), TUNED_IND_DTYPE)
                mx = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                mx_flat = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                one_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                all_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)''',
    ),
    (
        '''                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f32
                        T.reduce_max(ind, mx, dim=1, clear=True)  # 1.0 iff any zero
                        T.reshape(mx, mx_flat)  # (bm, 1) -> (bm,) bridge
                        T.vsub(one_v, mx_flat, all_v)  # all = 1 - any_zero
                        T.vcast(all_v, all_f16, round_mode="rint")  # f32 -> f16
                        T.vcast(all_f16, out_ub, round_mode="rint")  # f16 -> i8''',
        '''                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f16
                        T.reduce_max(ind, mx, dim=1, clear=True)  # 1.0 iff any zero
                        T.reshape(mx, mx_flat)  # (bm, 1) -> (bm,) bridge
                        T.vsub(one_v, mx_flat, all_v)  # all = 1 - any_zero
                        T.vcast(all_v, out_ub, round_mode="rint")  # f16 -> i8''',
    ),
    (
        '''                ind = T.alloc_shared((block_m, tile_n), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")''',
        '''                ind = T.alloc_shared((block_m, tile_n), TUNED_IND_DTYPE)
                mx = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                acc = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                new_acc = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                acc_flat = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                one_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                all_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)''',
    ),
    (
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)  # minuend is always 1
                        T.vcast(all_v, all_f16, round_mode="rint")
                        T.vcast(all_f16, out_ub, round_mode="rint")''',
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)  # minuend is always 1
                        T.vcast(all_v, out_ub, round_mode="rint")''',
    ),
    (
        '''                ind_tail = T.alloc_shared((block_m, tail), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_tail = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc_t = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")''',
        '''                ind_tail = T.alloc_shared((block_m, tail), TUNED_IND_DTYPE)
                mx = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                mx_tail = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                acc = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                new_acc = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                new_acc_t = T.alloc_fragment((block_m, 1), TUNED_IND_DTYPE)
                acc_flat = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                one_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)
                all_v = T.alloc_fragment((block_m,), TUNED_IND_DTYPE)''',
    ),
    (
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)
                        T.vcast(all_v, all_f16, round_mode="rint")
                        T.vcast(all_f16, out_ub, round_mode="rint")''',
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)
                        T.vcast(all_v, out_ub, round_mode="rint")''',
    ),
]

# ---------------------------------------------------------------------------
# 2. tuned constants + dispatch
# ---------------------------------------------------------------------------
OLD_CONSTANTS = '''DEFAULT_ALIGNMENT = 256
LAUNCH_GATE = 80  # > 2 waves of 40 vector cores -> engage the extended ladder'''

NEW_CONSTANTS = '''DEFAULT_ALIGNMENT = 256
LAUNCH_GATE = 80  # > 2 waves of 40 vector cores -> engage the extended ladder

# --- Stage 4 tuned defaults (module-level, pinned by the wrapper toggle) -----
# Measured on Ascend910B4 / 40 AIV / UB 196,608 B (msprof op Task Duration):
#   * the f32 indicator plane costs 12.14 B/element at a 16384-element block
#     (198,848 B > 196,608 B -> UB overflow), so a 16384-element tile is only
#     reachable with a narrower indicator plane;
#   * TUNED_IND_DTYPE = "float16" costs 2 B/element less and measured faster
#     (4k +4.0%, 3d +2.6%, 32k -1.1% within run noise);
#   * TUNED_RPRIME=True turns the (E_CAP, UB element cap] band into the
#     single-tile resident path (DESIGN section 1.6.4 R'), which needs the f16
#     indicator to fit;
#   * TUNED_TILE_CAP_MODE="ub" widens the column tile to the same UB element
#     cap (32k: 4 x 8192 -> 2 x 16384 tiles).
TUNED_IND_DTYPE = "float16"
TUNED_E_CAP = E_CAP
TUNED_RPRIME = True
TUNED_TILE_CAP_MODE = "ub"  # "ecap" (Stage 3) | "ub" (measured UB element cap)
TUNED_BM_POLICY = "ladder"  # "ladder" (Stage 3) | "cores" (grid-filling)


def _ub_elems(dtype):
    """Per-dtype single-tile element cap: min(measured pass point, UB budget).

    ``UNIT_PRICE`` is the measured compiler-side UB price of the Stage 3 chain;
    8 KB are kept as reserve, so the f16 indicator variant (2 B/element
    cheaper) is intentionally *not* extrapolated -- the cap is the Stage 3
    measured pass point, which the Stage 4 probes re-verified is a hard limit.
    """
    budget = (UB_BYTES - 8192) // UNIT_PRICE[dtype]
    return min(HARD_CAP_ELEMS[dtype], budget)'''

OLD_SELECT = '''def _select_config(M, N, dtype):
    """Element-budget ladder + path switch (DESIGN.md sections 3.3 / 5.2).

    Returns ``{"path", "block_m", "tile_n", "num_full", "tail"}`` with every
    value a factory-period Python int (static loop bounds).
    """
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

    if N <= E_CAP:
        # Path R: whole row resident in UB.
        path = "R"
        candidates = [p for p in (1, 2, 4, 8) if p * N <= E_CAP]
        block_m = max(candidates) if candidates else 1
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [
                p
                for p in (16, 32, 64, 128, 256, 512, 1024)
                if p * N <= E_CAP and p <= M
            ]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    else:
        # Path C: column tiles + no-alias running accumulator (+ optional tail).
        path = "C"
        block_m = 1
        tile_n = _pick_tile_n(N, min(E_CAP, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)  # never allocate more rows than exist

    # Capacity predicate (DESIGN.md section 4.5): the per-block element count
    # must sit inside both the race-safe budget and the measured hard limit.
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
    }'''

NEW_SELECT = '''def _select_config(M, N, dtype):
    """Element-budget ladder + three-path switch (DESIGN.md 3.3 / 5.2 / 1.6.4).

    Paths (every value is a factory-period Python int -> static loop bounds):

    * ``R``   -- ``N <= TUNED_E_CAP``: whole row resident, ladder ``block_m``.
    * ``R'``  -- ``TUNED_E_CAP < N <= _ub_elems(dtype)``: single-tile resident
      (``block_m = 1``, no tile loop, no micro-merge) -- DESIGN section 1.6.4's
      Stage 4 alternative, adopted here for the ``3d-multidim`` shape class.
    * ``C``   -- ``N > _ub_elems(dtype)``: column tiles + no-alias running
      ``max`` accumulator (+ optional static-width tail).  The tile width is
      ``TUNED_TILE_CAP_MODE`` (``"ecap"`` = Stage 3, ``"ub"`` = the measured UB
      element cap).

    Returns ``{"path", "block_m", "tile_n", "num_full", "tail"}``.
    """
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
    if TUNED_TILE_CAP_MODE not in ("ecap", "ub"):
        raise ValueError(f"unknown TUNED_TILE_CAP_MODE {TUNED_TILE_CAP_MODE!r}")

    ub_elems = _ub_elems(dtype)
    if N <= TUNED_E_CAP:
        # Path R: whole row resident in UB.
        path = "R"
        candidates = [p for p in (1, 2, 4, 8) if p * N <= TUNED_E_CAP]
        block_m = max(candidates) if candidates else 1
        if TUNED_BM_POLICY == "cores":
            cores = _vector_cores()
            filled = [p for p in sorted(candidates) if _ceildiv(M, p) >= cores]
            block_m = filled[0] if filled else min(candidates or [1])
        if _ceildiv(M, block_m) > LAUNCH_GATE:
            extended = [
                p
                for p in (16, 32, 64, 128, 256, 512, 1024)
                if p * N <= TUNED_E_CAP and p <= M
            ]
            block_m = max(extended + [block_m])
        tile_n, num_full, tail = N, 1, 0
    elif TUNED_RPRIME and N <= ub_elems:
        # Path R' (DESIGN 1.6.4): one tile per row, no tile loop / micro-merge.
        path = "R"
        block_m = 1
        tile_n, num_full, tail = N, 1, 0
    else:
        # Path C: column tiles + no-alias running accumulator (+ optional tail).
        path = "C"
        block_m = 1
        tile_cap = ub_elems if TUNED_TILE_CAP_MODE == "ub" else TUNED_E_CAP
        tile_n = _pick_tile_n(N, min(tile_cap, N))
        num_full = N // tile_n
        tail = N - num_full * tile_n

    block_m = min(block_m, M)  # never allocate more rows than exist

    # Capacity predicate (DESIGN.md section 4.5): the per-block element count
    # must sit inside the measured hard limit (the Stage 4 probes confirm it is
    # a hard compile limit, not a soft budget).
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
    }'''


def main():
    with open(BASE) as handle:
        src = handle.read()

    for old, new in IND_REPLACEMENTS:
        if src.count(old) != 1:
            raise SystemExit(f"indicator replacement not unique ({src.count(old)}): {old[:60]!r}")
        src = src.replace(old, new)
    for old, new in ((OLD_CONSTANTS, NEW_CONSTANTS), (OLD_SELECT, NEW_SELECT)):
        if src.count(old) != 1:
            raise SystemExit(f"block not unique ({src.count(old)}): {old[:60]!r}")
        src = src.replace(old, new)

    src = src.replace(
        '''__all__ = [
    "_logical_reduce_kernel",
    "golden_all_logical_reduce",
    "golden__logical_reduce_kernel",
    "ASCEND_MODE",
]''',
        '''__all__ = [
    "_logical_reduce_kernel",
    "_select_config",
    "golden_all_logical_reduce",
    "golden__logical_reduce_kernel",
    "TUNED_IND_DTYPE",
    "TUNED_E_CAP",
    "TUNED_RPRIME",
    "TUNED_TILE_CAP_MODE",
    "TUNED_BM_POLICY",
    "ASCEND_MODE",
]''',
    )
    if "__all__" not in src or "_select_config\"," not in src:
        raise SystemExit("__all__ patch failed")

    with open(OUT, "w") as handle:
        handle.write(src)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
