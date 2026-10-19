#!/usr/bin/env python
"""D-1 equivalence verification for ``_max_pool1d_with_indices_kernel`` (Stage 1).

Windowed 1D max pooling, dual output (values + int64 indices), PyTorch
``F.max_pool1d(..., return_indices=True)`` semantics.

SCOPE (sampled, fast -- per USER SUPPLEMENT S3)
-----------------------------------------------
The whole script must finish in a few minutes.  It does **not** enumerate the
C4/C4b whole-line specialisation; the ``(k, s, p, d, ceil_mode)`` space is
sampled with **one representative group per condition branch**:

  branch ``padding > 0``      -> cfg "p>0,s!=k" and "p>0,s!=k,Lnd"
  branch ``dilation > 1``     -> cfg "d>1,p>0"
  branch ``ceil_mode``        -> cfg "ceil,p>0,s!=k"
  branch ``stride != kernel`` -> cfg "p>0,s!=k", "ceil,p>0,s!=k", "out==1,s!=k"
  branch ``L non-divisible``  -> cfg "p>0,s!=k,Lnd" (L=63), "d>1,p>0" (L=63),
                                 "ceil,p>0,s!=k" (L=97), "ragged" (L=2049)
  branch ``empty window``     -> cfg "EW-min", "EW-oob", "EW-l2" (d_W>=2 and
                                 p_W>0 with the lane set skipping [0, L_in));
                                 asserts the SOURCE contract (idx=0), NOT torch
                                 equality (torch index is an out-of-range artifact)

Items (all contrast an *adopted* formula-level rewrite against its baseline):
  E1        semantic anchor (value half): "skip out-of-range lanes" == "x_pad =
            -inf with a -inf-initialised accumulator"; candidate = NPU baseline
            formula, baseline = ``F.max_pool1d`` (torch CPU).  Values only.
  E1-idx    the same anchor for the *index* half: literal source state machine
            vs ``F.max_pool1d`` indices.  Validates the §0.1 reading.
  N2N3      adopted index rewrite (two UB passes):
              value  : acc = fp32 vector max of in-bounds lanes, flag = prod(x==x)
              index  : first-hit  max_idx = min{iw : in-bounds and x[iw] == acc}
                       last-NaN  nan_idx = max{iw : in-bounds and isnan(x[iw])}
              output : has_nan ? (NaN, nan_idx) : (acc, max_idx)
            vs the source sequential state machine (values + indices).
  N2N3-neq  NaN-behaviour independence: the same rewrite with a NaN-**ignoring**
            value accumulator must give the *same* pair as with a NaN-propagating
            one on NaN windows -- i.e. the design does not depend on the
            undocumented ``T.vmax`` NaN behaviour.
  N2N3-ord  order guard: the wrong single-pass (winner test taken against the
            *prefix* running max -- the two passes fused) must be **detected**;
            the detection set must be non-empty.
  B2        multiplicative eq-mask NaN flag (``prod(x==x)``) vs sequential OR.
  C1        clamped contiguous span load (``base/span/LSPAN/s0`` + per-lane
            guard ``j = iw - s0``) vs element-wise in-bounds gather; asserts
            every in-bounds lane index lands inside ``[0, LSPAN)``.
  EW        empty-window contract: the design rewrite (``mech_b``) equals the
            source state machine (``idx = 0``) on windows with no in-bounds
            lane.  Torch is *not* used as an oracle there -- its index is an
            out-of-range loop artifact.

A separate ``specialisation_math_only()`` prints the **value half only** of the
C4/C4b whole-line / identity specialisations.  Those are recorded in DESIGN.md
§1.6.1 as a *math-equivalence argument* and are **NOT adopted** (see the S1
disposition paragraph); their index construction is deliberately not tested
because it is not implemented.

Run:  /workspace/venv/bin/python examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/verify_equiv.py
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import torch
import torch.nn.functional as F

F32 = np.float32

TARGET_DTYPES = {
    "float16": np.float16,
    "bfloat16": np.uint16,   # carrier only; rounding goes through torch
    "float32": np.float32,
}
TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# --------------------------------------------------------------------------
# output extent (verbatim copy of the source helper, incl. both ceil_mode fixes)
# --------------------------------------------------------------------------
def pool_output_dim(input_size, kernel_size, stride, padding, ceil_mode, dilation=1):
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1
    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1
    return max(out, 0)


# --------------------------------------------------------------------------
# reference implementations (fp32 comparison domain, matching accum_dtype)
# --------------------------------------------------------------------------
def golden(x_f32, k, s, p, d, ceil_mode):
    """torch CPU reference (the source-repo test benchmark)."""
    v, i = F.max_pool1d(
        torch.from_numpy(x_f32), kernel_size=k, stride=s, padding=p,
        dilation=d, ceil_mode=ceil_mode, return_indices=True,
    )
    return v.numpy(), i.numpy()


def source_sm(x, k, s, p, d, ceil_mode):
    """Literal transcription of the GPU source state machine (fp32 domain).

    ``max_idx``/``nan_idx`` start at 0, ``first_valid`` handling included, the
    visit order is ``kw`` ascending (== ``iw`` ascending).
    """
    n, c, l_in = x.shape
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    val = np.empty((n, c, out_l), F32)
    idx = np.empty((n, c, out_l), np.int64)
    for b in range(n):
        for ch in range(c):
            for ow in range(out_l):
                max_val = F32(-np.inf)
                has_nan = False
                max_idx = np.int64(0)
                nan_idx = np.int64(0)
                first_valid = True
                for kw in range(k):
                    iw = ow * s - p + kw * d
                    if 0 <= iw < l_in:
                        v = x[b, ch, iw]
                        if np.isnan(v):
                            nan_idx = np.int64(iw)
                            has_nan = True
                        elif first_valid:
                            max_val = v
                            max_idx = np.int64(iw)
                            first_valid = False
                        elif v > max_val:
                            max_val = v
                            max_idx = np.int64(iw)
                val[b, ch, ow] = F32(np.nan) if has_nan else max_val
                idx[b, ch, ow] = nan_idx if has_nan else max_idx
    return val, idx


def baseline_inf_pad(x, k, s, p, d, ceil_mode):
    """NPU baseline formula: acc starts at -inf, only in-bounds lanes folded."""
    n, c, l_in = x.shape
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    val = np.full((n, c, out_l), F32(-np.inf), F32)
    for kw in range(k):
        for ow in range(out_l):
            iw = ow * s - p + kw * d
            if 0 <= iw < l_in:
                val[:, :, ow] = np.fmax(val[:, :, ow], x[:, :, iw])
    return val


def _windows(x, k, s, p, d, ceil_mode):
    """Yield ``(ow, [in-bounds iw ...], [all iw ...])`` for every output position."""
    l_in = x.shape[-1]
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    for ow in range(out_l):
        iws = [ow * s - p + kw * d for kw in range(k)]
        yield ow, [i for i in iws if 0 <= i < l_in], iws


def mech_b(x, k, s, p, d, ceil_mode, nan_propagate=False):
    """Adopted index rewrite: two UB passes.

    pass 1 -- ``acc`` = fp32 vector max of the in-bounds lanes, ``flag`` = product
              of the eq masks (0 iff the window holds a NaN);
    pass 2 -- ``idx`` = min position attaining ``acc`` (NaN never equals anything
              -> non-NaN windows only), ``nan_idx`` = max NaN position;
    output -- ``has_nan ? NaN : acc`` and ``has_nan ? nan_idx : idx``.
    """
    n, c, l_in = x.shape
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    val = np.empty((n, c, out_l), F32)
    idx = np.empty((n, c, out_l), np.int64)
    for ow, ib, _ in _windows(x, k, s, p, d, ceil_mode):
        block = x[:, :, ib] if ib else np.zeros((n, c, 0), F32)
        # pass 1
        if nan_propagate:
            acc = np.max(block, axis=2) if ib else np.full((n, c), F32(-np.inf), F32)
        else:
            acc = np.full((n, c), F32(-np.inf), F32)
            for t in range(len(ib)):
                acc = np.fmax(acc, block[:, :, t])
        flag = np.ones((n, c), F32)
        for t in range(len(ib)):
            flag = flag * (block[:, :, t] == block[:, :, t]).astype(F32)
        # pass 2
        big = np.int64(l_in)
        best = np.full((n, c), big, np.int64)
        last_nan = np.full((n, c), np.int64(-1), np.int64)
        for t, iw in enumerate(ib):
            v = block[:, :, t]
            win = (v == acc)  # NaN never matches (IEEE)
            best = np.minimum(best, np.where(win, np.int64(iw), big))
            is_nan = np.isnan(v)
            last_nan = np.maximum(last_nan, np.where(is_nan, np.int64(iw), np.int64(-1)))
        has_nan = flag == 0
        # empty-window clause (v1): no in-bounds lane -> source emits idx = 0
        # (source ``max_idx`` initial value).  Non-empty windows always have
        # ``best < big`` because the window max is attained in-bounds, so this
        # rewrite only changes the empty-window regime.
        best_final = np.where(best == big, np.int64(0), best)
        val[:, :, ow] = np.where(has_nan, F32(np.nan), acc)
        idx[:, :, ow] = np.where(has_nan, last_nan, best_final)
    return val, idx


def mech_b_prefix(x, k, s, p, d, ceil_mode):
    """WRONG (fused) form: winner test taken against the *prefix* running max."""
    n, c, l_in = x.shape
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    val = np.empty((n, c, out_l), F32)
    idx = np.empty((n, c, out_l), np.int64)
    big = np.int64(l_in)
    for ow, ib, _ in _windows(x, k, s, p, d, ceil_mode):
        acc = np.full((n, c), F32(-np.inf), F32)
        best = np.full((n, c), big, np.int64)
        last_nan = np.full((n, c), np.int64(-1), np.int64)
        flag = np.ones((n, c), F32)
        for iw in ib:
            v = x[:, :, iw]
            acc = np.fmax(acc, v)                      # value update FIRST (defect)
            win = (v == acc)
            best = np.minimum(best, np.where(win, np.int64(iw), big))
            last_nan = np.maximum(last_nan, np.where(np.isnan(v), np.int64(iw), np.int64(-1)))
            flag = flag * (v == v).astype(F32)
        has_nan = flag == 0
        val[:, :, ow] = np.where(has_nan, F32(np.nan), acc)
        idx[:, :, ow] = np.where(has_nan, last_nan, best)
    return val, idx


def flag_multiplicative(x, k, s, p, d, ceil_mode):
    """B2 candidate: prod(x == x) over in-bounds lanes -> has_nan == (flag == 0)."""
    n, c = x.shape[0], x.shape[1]
    out_l = pool_output_dim(x.shape[-1], k, s, p, ceil_mode, d)
    flag = np.ones((n, c, out_l), F32)
    for ow, ib, _ in _windows(x, k, s, p, d, ceil_mode):
        for iw in ib:
            v = x[:, :, iw]
            flag[:, :, ow] = flag[:, :, ow] * (v == v).astype(F32)
    return flag == 0


def flag_sequential_or(x, k, s, p, d, ceil_mode):
    """B2 baseline: scalar sequential ``has_nan = has_nan | isnan(v)``."""
    n, c = x.shape[0], x.shape[1]
    out_l = pool_output_dim(x.shape[-1], k, s, p, ceil_mode, d)
    has = np.zeros((n, c, out_l), bool)
    for ow, ib, _ in _windows(x, k, s, p, d, ceil_mode):
        for iw in ib:
            has[:, :, ow] = has[:, :, ow] | np.isnan(x[:, :, iw])
    return has


def span_clamp(x, k, s, p, d, ceil_mode, bl):
    """C1 candidate: clamped contiguous span load + per-lane relative guard.

    ``(val, idx, lane_ok)``; ``lane_ok`` asserts every in-bounds lane index maps
    into ``[0, LSPAN)`` **and** that the loaded element is the original one.
    """
    n, c, l_in = x.shape
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    val = np.empty((n, c, out_l), F32)
    idx = np.empty((n, c, out_l), np.int64)
    big = np.int64(l_in)
    lane_ok = True
    for ow, ib, _ in _windows(x, k, s, p, d, ceil_mode):
        owl = ow % bl
        ow0 = ow - owl
        base = ow0 * s - p
        span = (bl - 1) * s + (k - 1) * d + 1
        lspan = min(span, l_in)
        s0 = max(0, min(base, l_in - lspan))
        if not (0 <= s0 and s0 + lspan <= l_in):
            lane_ok = False
        buf = x[:, :, s0:s0 + lspan]
        acc = np.full((n, c), F32(-np.inf), F32)
        flag = np.ones((n, c), F32)
        lanes = []
        for iw in ib:
            j = iw - s0
            if not (0 <= j < lspan) or s0 + j != iw:
                lane_ok = False
            lanes.append((iw, j))
            v = buf[:, :, j]
            acc = np.fmax(acc, v)
            flag = flag * (v == v).astype(F32)
        best = np.full((n, c), big, np.int64)
        last = np.full((n, c), np.int64(-1), np.int64)
        for iw, j in lanes:
            v = buf[:, :, j]
            best = np.minimum(best, np.where((v == acc), np.int64(iw), big))
            last = np.maximum(last, np.where(np.isnan(v), np.int64(iw), np.int64(-1)))
        has_nan = flag == 0
        val[:, :, ow] = np.where(has_nan, F32(np.nan), acc)
        idx[:, :, ow] = np.where(has_nan, last, best)
    return val, idx, lane_ok


# --------------------------------------------------------------------------
# data kinds
# --------------------------------------------------------------------------
def _finfo(dt_name):
    return {
        "float16": (6.1035e-05, 65504.0),
        "bfloat16": (1.1755e-38, 3.3895e38),
        "float32": (1.1755e-38, 3.4028e38),
    }[dt_name]


def make(kind, shape, dt, dt_name, rng):
    n, c, l = shape
    if kind == "rand":
        a = rng.uniform(-1.0, 1.0, size=shape)
    elif kind == "nan_two":
        a = rng.uniform(-1.0, 1.0, size=shape)
        a[..., 0] = np.nan
        a[..., (l - 1) % l] = np.nan
        a[..., l // 2] = np.nan
    elif kind == "nan_and_inf":
        a = rng.uniform(-1.0, 1.0, size=shape)
        a[..., 0] = np.nan
        a[..., 1 % l] = np.inf
        a[..., (l - 1) % l] = -np.inf
    elif kind == "ties_all_equal":
        a = np.ones(shape, F32) * 5.0
    elif kind == "tie_two_max":
        a = rng.uniform(-1.0, 1.0, size=shape)
        a[..., 0] = 9.0
        a[..., l // 2] = 9.0
        a[..., l - 1] = 9.0
    elif kind == "pm_inf":
        a = rng.uniform(-1.0, 1.0, size=shape)
        for t in range(l):
            a[..., t] = np.inf if t % 2 == 0 else -np.inf
    elif kind == "all_neg_inf":
        a = np.full(shape, -np.inf, F32)
    elif kind == "zeros":
        a = np.zeros(shape, F32)
    else:  # pragma: no cover
        raise ValueError(kind)
    # re-project through the target dtype (exactly representable for f16/bf16)
    if dt is np.float32:
        return a.astype(F32)
    t = torch.from_numpy(a.astype(F32))
    return t.to(TORCH_DTYPES[dt_name]).to(torch.float32).numpy()


KINDS = ["rand", "nan_two", "tie_two_max", "all_neg_inf", "pm_inf", "zeros"]

# (label, (n, c, l_in, k, s, p, d, ceil_mode)) -- one representative per branch.
CONFIGS = [
    ("p>0,s!=k",       (1, 8, 32, 3, 2, 1, 1, False)),      # padding>0, stride!=kernel
    ("p>0,s!=k,Lnd",   (1, 4, 63, 3, 2, 1, 1, False)),      # padding>0, stride!=kernel, L%k!=0
    ("d>1,p>0",        (1, 4, 63, 3, 3, 1, 2, False)),      # dilation>1, padding>0, s==k
    ("ceil,p>0,s!=k",  (1, 4, 97, 5, 3, 2, 1, True)),       # ceil_mode, padding>0, stride!=kernel
    ("out==1,s!=k",    (2, 4, 128, 128, 1, 0, 1, False)),   # manifest textcnn-global, L_out==1
    ("ragged",         (1, 2, 2049, 4, 7, 2, 5, False)),    # dilation>1, padding>0, s!=k, L non-div
    ("same_s,k,p0",    (1, 2, 8, 3, 3, 0, 1, False)),       # stride==kernel, no padding
    ("identity k=1",   (1, 2, 32, 1, 1, 0, 1, False)),      # k==1 identity parameters
]

# Empty-window (no in-bounds lane) configs -- SOURCE contract, NOT torch equality.
# Condition: d_W >= 2 and p_W > 0 and the arithmetic lane set skips [0, L_in)
# (minimal counterexample L_in=1,k=2,s=1,p=1,d=2 -> lanes {-1,1}).
EW_CONFIGS = [
    ("EW-min",  (1, 2, 1, 2, 1, 1, 2, False)),   # L_in=1,k=2,s=1,p=1,d=2 -> {-1,1}
    ("EW-oob",  (1, 2, 1, 2, 3, 1, 3, True)),    # L_in=1,k=2,s=3,p=1,d=3,ceil -> {-1,2}
    ("EW-l2",   (2, 3, 2, 2, 1, 1, 3, False)),   # L_in=2,k=2,s=1,p=1,d=3 -> {-1,2}
]


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def vals_equal(a, b):
    """NaN-vs-NaN counts equal; NaN-vs-finite always fails."""
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    if a_nan.any() or b_nan.any():
        if not np.array_equal(a_nan, b_nan):
            return False
    fin = ~a_nan
    return bool(np.array_equal(a[fin], b[fin]))


def mismatch(a, b):
    assert a.shape == b.shape and a.size > 0, "empty shield"
    return int(a.size) if not vals_equal(a, b) else 0


def mismatch_idx(a, b):
    assert a.shape == b.shape and a.size > 0, "empty shield"
    return 0 if np.array_equal(a, b) else int(a.size)


def round_to(x, dt_name):
    if dt_name is None or dt_name == "float32":
        return x
    t = torch.from_numpy(x.astype(F32))
    return t.to(TORCH_DTYPES[dt_name]).to(torch.float32).numpy()


# --------------------------------------------------------------------------
# item runners
# --------------------------------------------------------------------------
def run_item(name, fn, cases):
    checks = misses = 0
    worst = 0.0
    for cfg, dt_name, kind in cases:
        n, c, l, k, s, p, d, ceil = cfg
        dt = TARGET_DTYPES[dt_name]
        rng = np.random.default_rng(abs(hash((cfg, dt_name, kind))) % (2**32))
        x = make(kind, (n, c, l), dt, dt_name, rng)
        gv, gi = golden(x, k, s, p, d, ceil)
        gv = round_to(gv, dt_name)
        for candidate, baseline, is_idx in fn(x, k, s, p, d, ceil, gv, gi):
            checks += 1
            if candidate is None:
                continue
            if is_idx:
                miss = mismatch_idx(candidate, baseline)
            else:
                cr = round_to(candidate, dt_name)
                br = round_to(baseline, dt_name)
                miss = mismatch(cr, br)
                if miss:
                    fin = np.isfinite(cr) & np.isfinite(br)
                    if fin.any():
                        worst = max(worst, float(np.max(np.abs(cr[fin] - br[fin]))))
            misses += miss
    return checks, misses, worst


def item_e1(x, k, s, p, d, ceil, gv, gi):
    del gi
    acc = baseline_inf_pad(x, k, s, p, d, ceil)
    has = flag_sequential_or(x, k, s, p, d, ceil)
    return [(np.where(has, F32(np.nan), acc), gv, False)]


def item_e1_idx(x, k, s, p, d, ceil, gv, gi):
    sv, si = source_sm(x, k, s, p, d, ceil)
    del sv
    return [(si, gi, True)]


def item_n2n3(x, k, s, p, d, ceil, gv, gi):
    del gv, gi
    sv, si = source_sm(x, k, s, p, d, ceil)
    mv, mi = mech_b(x, k, s, p, d, ceil)
    return [(mv, sv, False), (mi, si, True)]


def item_n2n3_neq(x, k, s, p, d, ceil, gv, gi):
    del gv, gi
    if not np.isnan(x).any():
        return [(None, None, False)]
    av, ai = mech_b(x, k, s, p, d, ceil, nan_propagate=False)
    bv, bi = mech_b(x, k, s, p, d, ceil, nan_propagate=True)
    return [(av, bv, False), (ai, bi, True)]


def item_n2n3_ord(x, k, s, p, d, ceil, gv, gi):
    """Guard: the fused form must differ from the baseline somewhere (non-empty)."""
    del gv, gi
    sv, si = source_sm(x, k, s, p, d, ceil)
    fv, fi = mech_b_prefix(x, k, s, p, d, ceil)
    diff = mismatch(fv, sv) + mismatch_idx(fi, si)
    _STATS["ord_detected"] += 1 if diff > 0 else 0
    _STATS["ord_cases"] += 1
    return [(None, None, False)]


def item_b2(x, k, s, p, d, ceil, gv, gi):
    del gv, gi
    return [(flag_multiplicative(x, k, s, p, d, ceil).astype(np.int8),
             flag_sequential_or(x, k, s, p, d, ceil).astype(np.int8), True)]


def item_c1(x, k, s, p, d, ceil, gv, gi):
    del gv, gi
    sv, si = source_sm(x, k, s, p, d, ceil)
    res = []
    l_out = pool_output_dim(x.shape[-1], k, s, p, ceil, d)
    for bl in (2, 8):
        if bl > min(l_out, 32):
            continue
        cv, ci, ok = span_clamp(x, k, s, p, d, ceil, bl)
        assert ok, "span clamp lane escaped"
        _STATS["c1_lanes_checked"] += 1
        res.extend([(cv, sv, False), (ci, si, True)])
    return res if res else [(None, None, False)]


def item_ew(x, k, s, p, d, ceil, gv, gi):
    """Empty-window contract: design (mech_b) == source state machine (idx=0).

    NOT compared with torch (its index is an out-of-range loop artifact there).
    The assertion makes a mis-specified EW config fail loudly rather than pass.
    """
    del gv, gi
    sv, si = source_sm(x, k, s, p, d, ceil)
    mv, mi = mech_b(x, k, s, p, d, ceil)
    empties = [ow for ow, ib, _ in _windows(x, k, s, p, d, ceil) if not ib]
    assert empties, "EW config has no empty window"
    _STATS["ew_empty_windows"] += len(empties)
    return [(mv, sv, False), (mi, si, True)]


_STATS = {"ord_detected": 0, "ord_cases": 0, "c1_lanes_checked": 0, "ew_empty_windows": 0}

# only these configs exercise C1 (keep the span-clamp sweep small)
C1_CONFIGS = {"p>0,s!=k", "same_s,k,p0", "identity k=1"}
# B1b/order need NaN-bearing kinds
NAN_KINDS = {"nan_two", "nan_and_inf"}


# --------------------------------------------------------------------------
# §0.1 measured semantics evidence (torch CPU, exact commands printed)
# --------------------------------------------------------------------------
def semantics_evidence() -> int:
    print()
    print("=" * 100)
    print("§0.1 measured semantics evidence (torch 2.9.0 CPU, "
          "F.max_pool1d(..., return_indices=True))")
    print("=" * 100)
    fails = 0

    def show(label, row, k, s, p, d, ceil=False):
        nonlocal fails
        x = np.array([[row]], F32)
        v, i = golden(x, k, s, p, d, ceil)
        sv, si = source_sm(x, k, s, p, d, ceil)
        ok = np.array_equal(si, i) and vals_equal(sv, v)
        fails += 0 if ok else 1
        vv = v[0, 0].tolist()
        print(f"  {label:<28} x={[('nan' if np.isnan(t) else t) for t in row]} "
              f"k={k} s={s} p={p} d={d} -> val={vv} idx={i[0, 0].tolist()} "
              f"({'state-machine ok' if ok else 'STATE-MACHINE MISMATCH'})")

    # (a) index = input position iw along L (not a 2D coord, not a running counter)
    show("flat index = iw", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 2, 1, 0, 1)
    # (b) ties keep the first hit
    show("tie -> first hit", [5.0, 5.0, 5.0, 1.0], 3, 1, 0, 1)
    # (b2) two distinct maxima at iw=0 and iw=2 -> first
    show("tie two max -> first", [9.0, 1.0, 9.0, 1.0], 3, 1, 0, 1)
    # (c) NaN window: value NaN, index = LAST NaN visited
    show("two NaN -> last NaN", [np.nan, 5.0, np.nan, 2.0], 3, 1, 0, 1)
    show("NaN mid + tail", [1.0, np.nan, 3.0, np.nan], 3, 1, 0, 1)
    # (d) out-of-range lanes skipped (front/tail padding), all -inf
    show("pad skip, all -inf", [-np.inf] * 4, 3, 1, 1, 1)
    show("tail pad skip", [2.0, 7.0, -np.inf, -np.inf], 3, 1, 0, 1)
    # (e) {+inf,-inf} -> +inf (not NaN)
    show("{+inf,-inf} -> +inf", [np.inf, -np.inf, 1.0, 2.0], 2, 1, 0, 1)
    # (f) inf + NaN -> NaN, index = the NaN position
    show("inf + NaN -> NaN", [np.inf, np.nan, 1.0], 2, 1, 0, 1)
    # (g) dilation>1 index (window covers {0,2})
    show("dilation>1 index", [4.0, 1.0, 9.0, 1.0], 2, 2, 0, 2)
    # (h) stride != kernel, trailing partial window
    show("s!=k partial window", [1.0, 8.0, 2.0, 3.0, 7.0], 2, 3, 0, 1)

    print(f"  semantics evidence: {'all state-machine readings confirmed' if fails == 0 else f'{fails} MISMATCH'}")
    return fails


def hardcoded_guards() -> int:
    """Small, hard-coded counterexamples quoted in DESIGN.md §1.6.1."""
    fails = 0
    print()
    print("hard-coded guards (measured values quoted in DESIGN.md):")
    cases = [
        ("T1  ties keep first hit", [1.0, 3.0, 3.0, 2.0], 2, 1, 0, 1, [1, 1, 2]),
        ("T4  two NaN -> last NaN", [np.nan, 5.0, np.nan, 2.0], 3, 1, 0, 1, [2, 2]),
        ("T6  {+inf,-inf} -> +inf", [np.inf, -np.inf, 1.0, 2.0], 2, 1, 0, 1, [0, 2, 3]),
        ("P1  all -inf + front pad", [-np.inf] * 4, 3, 1, 1, 1, [0, 0, 1, 2]),
    ]
    for label, row, k, s, p, d, exp in cases:
        x = np.array([[row]], F32)
        gv, gi = golden(x, k, s, p, d, False)
        mv, mi = mech_b(x, k, s, p, d, False)
        ok = np.array_equal(gi[0, 0], np.array(exp, np.int64)) and np.array_equal(mi, gi) and vals_equal(mv, gv)
        fails += 0 if ok else 1
        print(f"  {label:<26} k={k} s={s} p={p} d={d} torch idx={gi[0, 0].tolist()} "
              f"mechanism-B={'ok' if np.array_equal(mi, gi) else 'MISMATCH'}")

    # fused / premature-accumulator counterexample: truth 1, fused form differs
    x = np.array([[[-np.inf, 5.0, -np.inf]]], F32)
    _, si = source_sm(x, 3, 1, 0, 1, False)
    _, fi = mech_b_prefix(x, 3, 1, 0, 1, False)
    detected = not np.array_equal(si, fi)
    print(f"  fused-order counterexample   x=[-inf,5,-inf] k3s1p0  truth idx="
          f"{si[0, 0].tolist()} fused idx={fi[0, 0].tolist()} "
          f"({'detected' if detected else 'NOT DETECTED'})")
    fails += 0 if detected else 1
    print(f"  hard-coded guards: {'all ok' if fails == 0 else f'{fails} FAILED'}")
    return fails


def empty_window_evidence() -> int:
    """Empty-window regime: source == design contract (idx=0); torch differs.

    Torch returns the last candidate lane's index (possibly outside [0, L_in)),
    so it is an out-of-range loop artifact and must not be used as an oracle.
    """
    print()
    print("=" * 100)
    print("empty-window contract (no in-bounds lane): source == design (idx=0); torch is NOT an oracle")
    print("=" * 100)
    fails = 0
    for label, (n, c, l, k, s, p, d, ceil) in EW_CONFIGS:
        x = np.full((n, c, l), 7.0, F32)
        sv, si = source_sm(x, k, s, p, d, ceil)
        mv, mi = mech_b(x, k, s, p, d, ceil)
        gv, gi = golden(x, k, s, p, d, ceil)
        empties = [ow for ow, ib, _ in _windows(x, k, s, p, d, ceil) if not ib]
        ok = (len(empties) > 0 and np.array_equal(si, mi)
              and np.array_equal(si, np.zeros_like(si)) and vals_equal(sv, mv))
        fails += 0 if ok else 1
        print(f"  {label:<8} k={k} s={s} p={p} d={d} ceil={ceil}  empty_ow={empties}  "
              f"source idx={si[0, 0].tolist()}  design idx={mi[0, 0].tolist()}  "
              f"torch idx={gi[0, 0].tolist()}  ({'contract ok' if ok else 'MISMATCH'})")
    print("  contract: source/design emit idx=0 for empty windows; the torch index is an "
          "out-of-range loop artifact -> not used as an oracle")
    return fails


def specialisation_math_only() -> int:
    """VALUE-ONLY math-equivalence argument for C4/C4b -- NOT adopted (see S1).

    C4 whole-line value (``L_out==1, p==0, d==1, (k-1)*d+1>=L_in``): the window
    is exactly ``[0, L_in)`` so ``max`` over it equals the whole-line max.
    C4b identity value (``k==1, d==1, p==0, s==1``): the window is a single
    element so ``out == x``.  The INDEX construction of both is deliberately
    not implemented/tested -- those paths are not adopted.
    """
    print()
    print("=" * 100)
    print("C4/C4b VALUE-ONLY math-equivalence argument (index construction NOT tested; "
          "paths NOT adopted per S1)")
    print("=" * 100)
    fails = 0
    cfgs = [
        ("C4-math",  (1, 4, 128, 128, 1, 0, 1, False)),  # out==1, p==0, d==1, k>=L
        ("C4b-math", (1, 4, 32, 1, 1, 0, 1, False)),     # identity
    ]
    for label, (n, c, l, k, s, p, d, ceil) in cfgs:
        for dt_name in TARGET_DTYPES:
            dt = TARGET_DTYPES[dt_name]
            rng = np.random.default_rng(abs(hash((label, dt_name))) % (2**32))
            bad = 0
            for kind in ("rand", "nan_two"):
                x = make(kind, (n, c, l), dt, dt_name, rng)
                gv, _ = golden(x, k, s, p, d, ceil)
                if label == "C4-math":
                    cand = np.max(x, axis=2)[:, :, None]
                else:
                    cand = x
                gvr = round_to(gv, dt_name)
                if mismatch(round_to(cand, dt_name), gvr):
                    bad += 1
            status = "EQUIV_PASS" if bad == 0 else "EQUIV_FAIL"
            fails += bad
            print(f"  {label:<10} {dt_name:<10} value-only  subsets_failed={bad}  {status}")
    print("  (value halves match; index construction is NOT adopted and NOT covered here)")
    return fails


def main() -> int:
    items = [
        ("E1", item_e1),
        ("E1-idx", item_e1_idx),
        ("N2N3", item_n2n3),
        ("N2N3-neq", item_n2n3_neq),
        ("N2N3-ord", item_n2n3_ord),
        ("B2", item_b2),
        ("C1", item_c1),
    ]
    print("check      dtype      config            kinds  worst|d|     viol    status")
    print("-" * 92)
    n_fail = 0
    total_checks = 0
    for name, fn in items:
        for label, cfg in CONFIGS:
            if name == "C1" and label not in C1_CONFIGS:
                continue
            kinds = list(KINDS)
            if name == "N2N3-neq":
                kinds = [k for k in KINDS if k in NAN_KINDS]
            if name == "N2N3-ord":
                kinds = [k for k in KINDS if k in ("rand", "tie_two_max", "zeros")]
            cases = [(cfg, dt_name, kind) for dt_name in TARGET_DTYPES for kind in kinds]
            checks, misses, worst = run_item(name, fn, cases)
            status = "EQUIV_PASS" if misses == 0 else "EQUIV_FAIL"
            n_fail += misses
            total_checks += checks
            print(f"{name:<10} {'all':<10} {label:<17} {len(kinds):<6} {worst:<11.4e} "
                  f"{misses:<7d} {status}  (checks={checks})")
    # empty-window item (own configs; source contract, NOT torch equality)
    for label, cfg in EW_CONFIGS:
        kinds = ["rand", "zeros", "all_neg_inf"]
        cases = [(cfg, dt_name, kind) for dt_name in TARGET_DTYPES for kind in kinds]
        checks, misses, worst = run_item("EW", item_ew, cases)
        status = "EQUIV_PASS" if misses == 0 else "EQUIV_FAIL"
        n_fail += misses
        total_checks += checks
        print(f"{'EW':<10} {'all':<10} {label:<17} {len(kinds):<6} {worst:<11.4e} "
              f"{misses:<7d} {status}  (checks={checks})")
    print("-" * 92)
    print("criterion: exact after rounding to the target dtype (0 ULP); NaN-vs-NaN counts equal, "
          "NaN-vs-finite fails; indices compared exactly (int64).")
    print(f"order guard: the fused/premature form differs from the baseline on "
          f"{_STATS['ord_detected']}/{_STATS['ord_cases']} cases (non-empty detection required)")
    print(f"C1 span-clamp: {_STATS['c1_lanes_checked']} (config, BL) structural checks; every "
          f"in-bounds lane landed inside [0, LSPAN)")
    print(f"empty-window contract: {_STATS['ew_empty_windows']} empty windows exercised; "
          f"source == design idx (0); torch differs (out-of-range), not used as an oracle")
    n_fail += semantics_evidence()
    n_fail += empty_window_evidence()
    n_fail += hardcoded_guards()
    n_fail += specialisation_math_only()
    print()
    print(f"[EQUIV] {'ALL PASS' if n_fail == 0 else 'FAIL'}  "
          f"(total item checks={total_checks}, failures={n_fail})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
