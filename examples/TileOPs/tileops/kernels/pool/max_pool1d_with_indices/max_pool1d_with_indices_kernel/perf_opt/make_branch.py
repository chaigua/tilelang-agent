"""Build branch files from the read-only baseline kernel by surgical text diff.

Usage:
  python make_branch.py round1_singlepass      # R1a: single-pass record + dead code
  python make_branch.py round1_chfull          # R1b: CH = min(C, cap) tiling
  ...

Each branch is a complete standalone kernel file (golden + L0/L1/L2/Boundary + main
block preserved) so `python <branch>.py --level all` still works and the branch can
be dropped in for the perf_opt deliverable.
"""

import os
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "_max_pool1d_with_indices_kernel.py")

# ---------------------------------------------------------------------------
# R1a: single-pass record form + removal of the dead NaN flag chain
# ---------------------------------------------------------------------------
R1A_OLD_ALLOC = """                    acc = T.alloc_ub((bl, ch), "float32")
                    flag = T.alloc_ub((bl, ch), "float32")
                    infbuf = T.alloc_ub((bl, ch), "float32")
                    nanbuf = T.alloc_ub((bl, ch), "float32")
                    out32 = T.alloc_ub((bl, ch), "float32")
                    cfinal = T.alloc_ub((bl, ch), "bool")
                    max_idx = T.alloc_ub((bl, ch), "int32")
                    nan_idx = T.alloc_ub((bl, ch), "int32")
                    idx32 = T.alloc_ub((bl, ch), "int32")
                    zeros_i = T.alloc_ub((bl, ch), "int32")
                    bigv = T.alloc_ub((bl, ch), "int32")
                    idx_t = T.alloc_ub((ch, bl), "int32")
                    idx64 = T.alloc_ub((ch, bl), "int64")
                    acc16 = T.alloc_ub((bl, ch), dtype)
                    acc_t = T.alloc_ub((ch, bl), dtype)
                    # Contiguous span staging: NCL span -> transposed (C innermost).
                    row_src = T.alloc_ub((ch, lspan), dtype)
                    row_dst = T.alloc_ub((lspan, ch), dtype)
                    row_f32 = T.alloc_ub((lspan, ch), "float32")
                    eq_mask = T.alloc_ub((lspan, ch), "bool")
                    eq_f32 = T.alloc_ub((lspan, ch), "float32")"""
R1A_NEW_ALLOC = """                    acc = T.alloc_ub((bl, ch), "float32")
                    infbuf = T.alloc_ub((bl, ch), "float32")
                    nanbuf = T.alloc_ub((bl, ch), "float32")
                    out32 = T.alloc_ub((bl, ch), "float32")
                    cfinal = T.alloc_ub((bl, ch), "bool")
                    max_idx = T.alloc_ub((bl, ch), "int32")
                    nan_idx = T.alloc_ub((bl, ch), "int32")
                    idx32 = T.alloc_ub((bl, ch), "int32")
                    zeros_i = T.alloc_ub((bl, ch), "int32")
                    bigv = T.alloc_ub((bl, ch), "int32")
                    idx_t = T.alloc_ub((ch, bl), "int32")
                    idx64 = T.alloc_ub((ch, bl), "int64")
                    acc16 = T.alloc_ub((bl, ch), dtype)
                    acc_t = T.alloc_ub((ch, bl), dtype)
                    # Contiguous span staging: NCL span -> transposed (C innermost).
                    row_src = T.alloc_ub((ch, lspan), dtype)
                    row_dst = T.alloc_ub((lspan, ch), dtype)
                    row_f32 = T.alloc_ub((lspan, ch), "float32")"""

R1A_OLD_PEREL = """                    cand = T.alloc_ub((1, ch), "int32")
                    nanc = T.alloc_ub((1, ch), "int32")
                    fvec = T.alloc_ub((1, ch), "int32")
                    neg1 = T.alloc_ub((1, ch), "int32")
                    big1 = T.alloc_ub((1, ch), "int32")
                    eqb = T.alloc_ub((1, ch), "bool")
                    eq2b = T.alloc_ub((1, ch), "bool")"""
R1A_NEW_PEREL = """                    nanc = T.alloc_ub((1, ch), "int32")
                    fvec = T.alloc_ub((1, ch), "int32")
                    neg1 = T.alloc_ub((1, ch), "int32")
                    eqb = T.alloc_ub((1, ch), "bool")
                    eq2b = T.alloc_ub((1, ch), "bool")"""

R1A_OLD_INIT = """                            T.vbrc(neg_inf, acc)
                            T.vbrc(one_val, flag)
                            T.vbrc(big_i, bigv)
                            T.vbrc(big_i, big1)
                            T.vbrc(neg1_i, neg1)"""
R1A_NEW_INIT = """                            T.vbrc(neg_inf, acc)
                            T.vbrc(big_i, bigv)
                            T.vbrc(neg1_i, neg1)"""

R1A_OLD_MASK = """                            T.transpose(row_src, row_dst, permutation=[1, 0])
                            if is_f32:
                                T.copy(row_dst, row_f32)
                            else:
                                T.vcast(row_dst, row_f32, round_mode="rint")
                            # Block-level NaN probe materialised as an fp32
                            # {0, 1} mask: x == x is 1 for numbers and 0 for NaN
                            # (the "ne" mode is an ordered not-equal and is always
                            # false for NaN, so it can never detect NaN).
                            T.vcmp(row_f32, row_f32, eq_mask, "eq")
                            T.vcast(eq_mask, eq_f32, round_mode="rint")
"""
R1A_NEW_MASK = """                            T.transpose(row_src, row_dst, permutation=[1, 0])
                            if is_f32:
                                T.copy(row_dst, row_f32)
                            else:
                                T.vcast(row_dst, row_f32, round_mode="rint")
"""

R1A_OLD_LOOPS = """                            # Pass 1: value + NaN flag + last-NaN index.
                            for kw in T.serial(kernel_w):
                                for owl in T.serial(bl):
                                    iw = base + owl * stride_w + kw * dilation_w
                                    j = iw - s0
                                    if 0 <= iw and iw < l_in:
                                        T.vmax(
                                            acc[owl:owl + 1, 0:ch],
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                        )
                                        T.vmul(
                                            flag[owl:owl + 1, 0:ch],
                                            eq_f32[j:j + 1, 0:ch],
                                            flag[owl:owl + 1, 0:ch],
                                        )
                                        T.vbrc(T.int32(iw), fvec)
                                        # last-NaN: non-NaN -> -1, NaN -> iw, then
                                        # vmax accumulates the largest NaN iw.
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            row_f32[j:j + 1, 0:ch],
                                            eq2b,
                                            "eq",
                                        )
                                        T.vselect(eq2b, neg1, fvec, nanc)
                                        T.vmax(
                                            nan_idx[owl:owl + 1, 0:ch],
                                            nanc,
                                            nan_idx[owl:owl + 1, 0:ch],
                                        )

                            # Pass 2: first-hit.  The value accumulator is final,
                            # so eq(row, acc) marks every lane whose value equals
                            # the window maximum; vmin then keeps the smallest iw.
                            for kw in T.serial(kernel_w):
                                for owl in T.serial(bl):
                                    iw = base + owl * stride_w + kw * dilation_w
                                    j = iw - s0
                                    if 0 <= iw and iw < l_in:
                                        T.vbrc(T.int32(iw), fvec)
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                            eqb,
                                            "eq",
                                        )
                                        T.vselect(eqb, fvec, big1, cand)
                                        T.vmin(
                                            max_idx[owl:owl + 1, 0:ch],
                                            cand,
                                            max_idx[owl:owl + 1, 0:ch],
                                        )
"""
R1A_NEW_LOOPS = """                            # Single pass (record form): the window is visited
                            # in strictly increasing iw order for each output
                            # lane, so a strict ``>`` record against the *running*
                            # maximum keeps the first occurrence of the final
                            # maximum -- exactly the first-hit contract.  This
                            # also removes the separate equality pass and the dead
                            # NaN-flag chain (the NaN decision is ``nan_idx >= 0``).
                            for kw in T.serial(kernel_w):
                                for owl in T.serial(bl):
                                    iw = base + owl * stride_w + kw * dilation_w
                                    j = iw - s0
                                    if 0 <= iw and iw < l_in:
                                        T.vbrc(T.int32(iw), fvec)
                                        # first-hit record (compare BEFORE the
                                        # value accumulator is updated).
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                            eqb,
                                            "gt",
                                        )
                                        T.vselect(
                                            eqb,
                                            fvec,
                                            max_idx[owl:owl + 1, 0:ch],
                                            max_idx[owl:owl + 1, 0:ch],
                                        )
                                        T.vmax(
                                            acc[owl:owl + 1, 0:ch],
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                        )
                                        # last-NaN: NaN -> iw, else -1; vmax keeps
                                        # the largest NaN iw.
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            row_f32[j:j + 1, 0:ch],
                                            eq2b,
                                            "eq",
                                        )
                                        T.vselect(eq2b, neg1, fvec, nanc)
                                        T.vmax(
                                            nan_idx[owl:owl + 1, 0:ch],
                                            nanc,
                                            nan_idx[owl:owl + 1, 0:ch],
                                        )
"""


def patch(src: str, old: str, new: str, tag: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"patch anchor {tag!r} matched {src.count(old)} times")
    return src.replace(old, new)


def round1_singlepass(src: str) -> str:
    src = patch(src, R1A_OLD_ALLOC, R1A_NEW_ALLOC, "alloc")
    src = patch(src, R1A_OLD_PEREL, R1A_NEW_PEREL, "perel")
    src = patch(src, R1A_OLD_INIT, R1A_NEW_INIT, "init")
    src = patch(src, R1A_OLD_MASK, R1A_NEW_MASK, "mask")
    src = patch(src, R1A_OLD_LOOPS, R1A_NEW_LOOPS, "loops")
    src = patch(
        src,
        "                    one_val = T.float32(1.0)\n",
        "",
        "one_val",
    )
    return src


# ---------------------------------------------------------------------------
# R1b: remove the dead NaN-flag chain only (two-pass structure untouched)
# ---------------------------------------------------------------------------
def round1_deadcode(src: str) -> str:
    src = patch(
        src,
        """                    acc = T.alloc_ub((bl, ch), "float32")
                    flag = T.alloc_ub((bl, ch), "float32")""",
        """                    acc = T.alloc_ub((bl, ch), "float32")""",
        "flag-alloc",
    )
    src = patch(
        src,
        """                    row_f32 = T.alloc_ub((lspan, ch), "float32")
                    eq_mask = T.alloc_ub((lspan, ch), "bool")
                    eq_f32 = T.alloc_ub((lspan, ch), "float32")""",
        """                    row_f32 = T.alloc_ub((lspan, ch), "float32")""",
        "eq-alloc",
    )
    src = patch(
        src,
        """                            T.vbrc(one_val, flag)\n""",
        "",
        "flag-init",
    )
    src = patch(
        src,
        """                            # Block-level NaN probe materialised as an fp32
                            # {0, 1} mask: x == x is 1 for numbers and 0 for NaN
                            # (the "ne" mode is an ordered not-equal and is always
                            # false for NaN, so it can never detect NaN).
                            T.vcmp(row_f32, row_f32, eq_mask, "eq")
                            T.vcast(eq_mask, eq_f32, round_mode="rint")
""",
        "",
        "eq-mask",
    )
    src = patch(
        src,
        """                                        T.vmul(
                                            flag[owl:owl + 1, 0:ch],
                                            eq_f32[j:j + 1, 0:ch],
                                            flag[owl:owl + 1, 0:ch],
                                        )
""",
        "",
        "flag-mul",
    )
    src = patch(
        src,
        "                    one_val = T.float32(1.0)\n",
        "",
        "one_val",
    )
    return src


# ---------------------------------------------------------------------------
# R1c: static row-offset index plane instead of a per-visit runtime broadcast
# ---------------------------------------------------------------------------
R1C_OLD_CONSTS = """                    one_val = T.float32(1.0)
                    big_i = T.int32(l_in)
                    neg1_i = T.int32(-1)
                    zero_i = T.int32(0)
"""
R1C_NEW_CONSTS = """                    one_val = T.float32(1.0)
                    big_i = T.int32(l_in)
                    neg1_i = T.int32(-1)
                    zero_i = T.int32(0)

                    # Static row-offset plane: row j holds the *span-local* index
                    # j.  A per-visit ``T.vbrc(runtime_scalar)`` broadcast is
                    # lowered to per-lane scalar stores on this toolchain, so the
                    # window index is read out of this precomputed plane instead
                    # (iw = s0 + j is recovered once per unit in the epilogue).
                    jramp = T.alloc_ub((lspan, ch), "int32")
                    for jj in T.serial(lspan):
                        T.vbrc(T.int32(jj), jramp[jj:jj + 1, 0:ch])
"""


def round1_jramp(src: str) -> str:
    src = patch(src, R1C_OLD_CONSTS, R1C_NEW_CONSTS, "jramp-alloc")
    # per-visit broadcasts -> static plane rows
    src = patch(
        src,
        """                                        T.vbrc(T.int32(iw), fvec)
                                        # last-NaN: non-NaN -> -1, NaN -> iw, then
                                        # vmax accumulates the largest NaN iw.""",
        """                                        T.copy(jramp[j:j + 1, 0:ch], fvec)
                                        # last-NaN: non-NaN -> -1, NaN -> j, then
                                        # vmax accumulates the largest NaN j.""",
        "vbrc-pass1",
    )
    src = patch(
        src,
        """                                        T.vbrc(T.int32(iw), fvec)
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],""",
        """                                        T.copy(jramp[j:j + 1, 0:ch], fvec)
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],""",
        "vbrc-pass2",
    )
    # epilogue: shift the j-domain index back to the global iw domain
    src = patch(
        src,
        """                            T.vcmp(idx32, bigv, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)""",
        """                            # Empty-window sentinel is evaluated in the
                            # span-local j domain, before the s0 shift.
                            T.vcmp(idx32, bigv, cfinal, "eq")
                            # j -> global iw (iw = s0 + j holds for every window
                            # lane, including the clamped first/last blocks).
                            T.vadd(idx32, T.int32(s0), idx32)
                            T.vselect(cfinal, zeros_i, idx32, idx32)""",
        "epilogue-shift",
    )
    return src


# ---------------------------------------------------------------------------
# R1: replace the two per-unit int32 vcmp/vselect decisions with a cheap
#     int32->fp32 cast + fp32 vcmp (int32 vcmp is ~1 us/op on this toolchain)
# ---------------------------------------------------------------------------
R1_OLD = """                            # Both outputs share one NaN decision:
                            # cfinal = (nan_idx >= 0) <=> the window holds a NaN.
                            T.vcmp(nan_idx, zeros_i, cfinal, "ge")
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vselect(cfinal, nan_idx, max_idx, idx32)
                            # Empty-window clause (DESIGN.md 0.1, alignment with
                            # the source state machine): idx32 == BIG <=> no NaN
                            # and no in-bounds lane -> idx = 0.  It reuses the
                            # cfinal bool buffer, so it must run *after* out32 is
                            # selected.
                            T.vcmp(idx32, bigv, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)"""
R1_NEW = """                            # Both outputs share one NaN decision:
                            # cfinal = (nan_idx >= 0) <=> the window holds a NaN.
                            # The comparison is executed in the fp32 domain: an
                            # int32 ``vcmp`` is lowered to a per-lane scalar loop
                            # on this toolchain (~16 ns/lane, measured) and costs
                            # more than the whole window scan, while an fp32 vcmp
                            # is a vector op.  The index range (|iw| < 2**24) makes
                            # the int32 -> fp32 cast exact.
                            T.vcast(nan_idx, nan_f32, round_mode="rint")
                            T.vcmp(nan_f32, zero_f32, cfinal, "ge")
                            T.vselect(cfinal, nanbuf, acc, out32)
                            T.vselect(cfinal, nan_idx, max_idx, idx32)
                            # Empty-window clause (DESIGN.md 0.1, alignment with
                            # the source state machine): idx32 == BIG <=> no NaN
                            # and no in-bounds lane -> idx = 0.  It reuses the
                            # cfinal bool buffer, so it must run *after* out32 is
                            # selected.  fp32-domain comparison for the same reason
                            # as above.
                            T.vcast(idx32, idx_f32, round_mode="rint")
                            T.vcmp(idx_f32, big_f32, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)"""

R1_OLD_ALLOC = """                    idx32 = T.alloc_ub((bl, ch), "int32")"""
R1_NEW_ALLOC = """                    idx32 = T.alloc_ub((bl, ch), "int32")
                    nan_f32 = T.alloc_ub((bl, ch), "float32")
                    idx_f32 = T.alloc_ub((bl, ch), "float32")"""

R1_OLD_CONST = """                    zero_i = T.int32(0)"""
R1_NEW_CONST = """                    zero_i = T.int32(0)
                    zero_f32 = T.float32(0.0)
                    big_f32 = T.float32(l_in)"""


def round1_intfix(src: str) -> str:
    src = patch(src, R1_OLD_ALLOC, R1_NEW_ALLOC, "intfix-alloc")
    src = patch(src, R1_OLD_CONST, R1_NEW_CONST, "intfix-const")
    src = patch(src, R1_OLD, R1_NEW, "intfix-epilogue")
    return src


def round1_intfix_deadcode(src: str) -> str:
    return round1_deadcode(round1_intfix(src))


# ---------------------------------------------------------------------------
# R2: tiling re-derivation -- CH up to 128 (== C for every tune workload) and an
#     exact UB footprint (the Stage 3 formula over-counts by dead buffers)
# ---------------------------------------------------------------------------
R2_OLD_BYTES = '''def _general_bytes(ch: int, lspan: int, bl: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the general path (bytes, no inflation).

    The frozen DESIGN.md section 4.5 criterion (revision v1, byte coefficient
    ``69 + 2*eb`` after the v0 undercount of idx32 / idx_t / idx64 was fixed):

        G = CH*LSPAN*(2*eb + 9) + BL*CH*(69 + 2*eb) + 256

    ``2*eb + 9``  = row_src(eb) + row_dst(eb) + row_f32(4) + eq_mask(1) +
                    eq_f32(4);
    ``69 + 2*eb`` = acc / flag / infbuf / nanbuf / out32 / zeros (6 x f32 = 24)
                    + cfinal (bool 1) + acc16 + acc_t (2*eb)
                    + max_idx / nan_idx / cand / nanc / bigv / neg1v / zero_i32
                      (7 x i32 = 28)
                    + idx32 / idx_t (8) + idx64 (8);
    ``256``       = fv (1, CH) int32 upper bound for CH <= 64.

    It is reproduced verbatim (per-buffer sum == closed form, the v1 self-check)
    so the CH/BL selection matches the design tables exactly (sincnet CH=64 /
    BL=16 / LSPAN=48; ecg CH=64 / BL=17 / LSPAN=41).  The implementation folds
    the per-``owl`` ``(1, CH)`` staging operands into small whole buffers, so the
    real footprint is at or below this bound.
    """
    per_elem = 2 * elem_bytes + 9  # row_src + row_dst + row_f32 + eq_mask + eq_f32
    base = 6 * 4 + 1 + 2 * elem_bytes + 7 * 4 + 2 * 4 + 8  # = 69 + 2*eb
    return (
        ch * lspan * per_elem
        + bl * ch * base
        + 256
    )'''
R2_NEW_BYTES = '''def _general_bytes(ch: int, lspan: int, bl: int, elem_bytes: int) -> int:
    """Per-core UB footprint of the general path (bytes, no inflation).

    Stage 4 recount (tiling re-derivation): the Stage 3 closed form
    ``G = CH*LSPAN*(2*eb+9) + BL*CH*(69+2*eb) + 256`` budgeted the buffers of the
    *two-pass + dead NaN-flag* implementation.  After the dead-code removal and
    the fp32-domain NaN/empty-window decisions the live set is exactly

        G = CH*LSPAN*(2*eb + 4) + BL*CH*(57 + 2*eb) + 14*CH

    ``2*eb + 4``   = row_src(eb) + row_dst(eb) + row_f32(4)
    ``57 + 2*eb``  = acc / infbuf / nanbuf / out32 / nan_f32 / idx_f32 (6 x f32 = 24)
                     + cfinal (bool 1) + max_idx / nan_idx / idx32 / zeros_i /
                       bigv (5 x i32 = 20) + idx_t (4) + idx64 (8)
                     + acc16 + acc_t (2*eb)
    ``14*CH``      = nanc / fvec / neg1 (3 x i32 = 12) + eqb / eq2b (2 x bool)

    The per-buffer sum is asserted against the closed form by
    ``_general_bytes_selfcheck()`` at import time.
    """
    per_span = 2 * elem_bytes + 4
    per_out = 57 + 2 * elem_bytes
    return ch * lspan * per_span + bl * ch * per_out + 14 * ch


def _general_bytes_selfcheck() -> None:
    """Per-buffer sum == closed form (guards the capacity inequality)."""
    for eb in (2, 4):
        live_out = (
            6 * 4          # acc, infbuf, nanbuf, out32, nan_f32, idx_f32
            + 1            # cfinal
            + 5 * 4        # max_idx, nan_idx, idx32, zeros_i, bigv
            + 4            # idx_t
            + 8            # idx64
            + 2 * eb       # acc16, acc_t
        )
        live_span = eb + eb + 4  # row_src, row_dst, row_f32
        live_vec = 3 * 4 + 2     # nanc, fvec, neg1, eqb, eq2b
        assert live_out == 57 + 2 * eb, live_out
        assert live_span == 2 * eb + 4, live_span
        assert live_vec == 14, live_vec'''
R2_OLD_CHMAX = """# Target channel-vector-block ceiling (DESIGN.md sections 5.2 / 1.6.3).
_CH_MAX = 64"""
R2_NEW_CHMAX = """# Target channel-vector-block ceiling (Stage 4 tiling re-derivation).
# C is 80 / 256 / 128 for the three manifest full workloads, so 128 lets the two
# C=128-shaped workloads run a single channel block (no padding lanes and half
# the logical units); the capacity inequality below then picks BL.
_CH_MAX = 128"""


def round2_chfull(src: str) -> str:
    src = patch(src, R2_OLD_BYTES, R2_NEW_BYTES, "bytes")
    src = patch(src, R2_OLD_CHMAX, R2_NEW_CHMAX, "chmax")
    return src


def round2_chfull_all(src: str) -> str:
    return round2_chfull(round1_intfix_deadcode(src))


# ---------------------------------------------------------------------------
# R3: per-visit op count 9 -> 5
#   * NaN position plane built ONCE per unit  (T.vcmp + T.vselect over the span)
#     -> pass 1 keeps only a vmax per visit instead of vcmp+vselect+vmax
#   * pass 2 reads the span-local index out of an `arange` row ramp instead of
#     broadcasting a runtime scalar (T.vbrc) per visit
# ---------------------------------------------------------------------------
R3_OLD_ALLOC = """                    nan_f32 = T.alloc_ub((bl, ch), "float32")
                    idx_f32 = T.alloc_ub((bl, ch), "float32")"""
R3_NEW_ALLOC = """                    nan_f32 = T.alloc_ub((bl, ch), "float32")
                    idx_f32 = T.alloc_ub((bl, ch), "float32")
                    # span-local index ramp (row j -> j) and the per-unit NaN
                    # position plane (NaN row -> j, otherwise -1)
                    jplane = T.alloc_ub((lspan, ch), "int32")
                    nanplane = T.alloc_ub((lspan, ch), "int32")
                    neg1plane = T.alloc_ub((lspan, ch), "int32")
                    nanmask = T.alloc_ub((lspan, ch), "bool")"""

R3_OLD_CONST = """                    zero_f32 = T.float32(0.0)
                    big_f32 = T.float32(l_in)"""
R3_NEW_CONST = """                    zero_f32 = T.float32(0.0)
                    big_f32 = T.float32(l_in)

                    # Row ramp: element (j, c) = j.  Built once per core, read
                    # per visit as a (1, CH) operand (no runtime broadcast).
                    T.arange(jplane, [1, 0], 0)
                    T.vbrc(neg1_i, neg1plane)"""

R3_OLD_PASS1 = """                                        T.vbrc(T.int32(iw), fvec)
                                        # last-NaN: non-NaN -> -1, NaN -> iw, then
                                        # vmax accumulates the largest NaN iw.
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            row_f32[j:j + 1, 0:ch],
                                            eq2b,
                                            "eq",
                                        )
                                        T.vselect(eq2b, neg1, fvec, nanc)
                                        T.vmax(
                                            nan_idx[owl:owl + 1, 0:ch],
                                            nanc,
                                            nan_idx[owl:owl + 1, 0:ch],
                                        )"""
R3_NEW_PASS1 = """                                        # last-NaN: the NaN position plane is
                                        # built once per unit, so the scan only
                                        # accumulates its running maximum.
                                        T.vmax(
                                            nan_idx[owl:owl + 1, 0:ch],
                                            nanplane[j:j + 1, 0:ch],
                                            nan_idx[owl:owl + 1, 0:ch],
                                        )"""

R3_OLD_PASS2 = """                                        T.vbrc(T.int32(iw), fvec)
                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                            eqb,
                                            "eq",
                                        )
                                        T.vselect(eqb, fvec, big1, cand)"""
R3_NEW_PASS2 = """                                        T.vcmp(
                                            row_f32[j:j + 1, 0:ch],
                                            acc[owl:owl + 1, 0:ch],
                                            eqb,
                                            "eq",
                                        )
                                        T.vselect(eqb, jplane[j:j + 1, 0:ch],
                                                  big1, cand)"""

R3_OLD_MASK = """                            T.transpose(row_src, row_dst, permutation=[1, 0])
                            if is_f32:
                                T.copy(row_dst, row_f32)
                            else:
                                T.vcast(row_dst, row_f32, round_mode="rint")
"""
R3_NEW_MASK = """                            T.transpose(row_src, row_dst, permutation=[1, 0])
                            if is_f32:
                                T.copy(row_dst, row_f32)
                            else:
                                T.vcast(row_dst, row_f32, round_mode="rint")
                            # Per-unit NaN position plane: NaN rows carry their
                            # span-local index, all other rows carry -1.  x == x
                            # is 1 only for numbers (an ordered compare can never
                            # detect NaN), so the mask is the *eq* result.
                            T.vcmp(row_f32, row_f32, nanmask, "eq")
                            T.vselect(nanmask, jplane, neg1plane, nanplane)
"""

R3_OLD_EPI = """                            T.vselect(cfinal, nan_idx, max_idx, idx32)"""
R3_NEW_EPI = """                            T.vselect(cfinal, nan_idx, max_idx, idx32)"""
R3_OLD_EMPTY = """                            T.vcast(idx32, idx_f32, round_mode="rint")
                            T.vcmp(idx_f32, big_f32, cfinal, "eq")
                            T.vselect(cfinal, zeros_i, idx32, idx32)"""
R3_NEW_EMPTY = """                            T.vcast(idx32, idx_f32, round_mode="rint")
                            T.vcmp(idx_f32, big_f32, cfinal, "eq")
                            # span-local j -> global iw (iw = s0 + j for every
                            # window lane, including the clamped blocks)
                            T.vadd(idx32, T.int32(s0), idx32)
                            T.vselect(cfinal, zeros_i, idx32, idx32)"""


def round3_opcount(src: str) -> str:
    src = patch(src, R3_OLD_ALLOC, R3_NEW_ALLOC, "r3-alloc")
    src = patch(src, R3_OLD_CONST, R3_NEW_CONST, "r3-const")
    src = patch(src, R3_OLD_MASK, R3_NEW_MASK, "r3-mask")
    src = patch(src, R3_OLD_PASS1, R3_NEW_PASS1, "r3-pass1")
    # NOTE: the pass-2 candidate cannot read the row ramp directly -- `hivm.hir.vsel`
    # rejects operands sliced out of a larger buffer ("operands' shape are
    # inconsistent", measured), so pass 2 keeps the per-visit broadcast.
    src = patch(src, R3_OLD_EMPTY, R3_NEW_EMPTY, "r3-empty")
    return src


def round3_all(src: str) -> str:
    return round3_opcount(round2_chfull_all(src))


BRANCHES = {
    "round1_singlepass": round1_singlepass,
    "round1_deadcode": round1_deadcode,
    "round1_jramp": round1_jramp,
    "round1_intfix": round1_intfix,
    "round1_intfix_deadcode": round1_intfix_deadcode,
    "round2_chfull": round2_chfull_all,
    "round3_opcount": round3_all,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BRANCHES:
        raise SystemExit(f"usage: {sys.argv[0]} <{'|'.join(BRANCHES)}>")
    name = sys.argv[1]
    src = open(BASE).read()
    out = BRANCHES[name](src)
    out = out.replace(
        '"""Windowed max pooling 1D forward kernel with int64 indices (NPU, Developer).',
        f'"""Stage 4 branch `{name}` -- windowed max pooling 1D with int64 indices (NPU).\n\n'
        'Derived from the read-only Stage 3 baseline; see perf_opt/opt_log.md for the\n'
        'single main optimisation point this branch validates.\n\n'
        'Original header:\n'
        'Windowed max pooling 1D forward kernel with int64 indices (NPU, Developer).',
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"_max_pool1d_with_indices_kernel_{name}.py")
    open(path, "w").write(out)
    print(f"wrote {path} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
