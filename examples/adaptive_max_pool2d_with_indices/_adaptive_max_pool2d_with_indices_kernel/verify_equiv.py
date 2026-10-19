#!/usr/bin/env python
"""Equivalence machine-verification (D-1) for the adaptive_max_pool2d_with_indices migration.

Verifies the math-equivalent reformulations adopted in DESIGN.md section 1.6.1 against the
**baseline** = the source GPU kernel's sequential scalar state machine (DESIGN.md section 0.1
semantics: row-major bin visit order, first-hit tie-break via strict ``>``, last-NaN index,
NaN wins the value, indices = flat ``ih*W_in+iw`` int64).

An fp64 reference (torch CPU) is used throughout; all comparisons round to the target dtype
(fp16 / bf16, the Op-layer dtype gate rejects fp32) for the value and compare the index as an
exact integer.  Adopted items:

  N1  has_nan as "multiplicative eq-mask product == 0"  (source: scalar OR of isnan)
  N2  last-NaN index as "max{ flat : isnan(v) }"        (source: sequential overwrite)
  N3  first-hit max index without the source ``first_valid`` branch, with ``max_idx``
      pre-seeded to the first visited element's flat index
  N3b same as N3 but with a NaN-*ignoring* ``vmax`` (tests independence from the undocumented
      ``T.vmax`` NaN behaviour: the result must be identical under both conventions)
  N3-ord  implementation-order guard: the strict-greater test MUST compare against the
      *pre-update* value accumulator (``gt = v > acc`` before ``acc = vmax(acc, v)``).  This
      item (a) re-checks the correct order against the baseline and (b) verifies the wrong
      order (value accumulator updated first, as in DESIGN.md v0 section 3.3/6.2) is *detected*
      on every output where the true first hit leaves the window's first visited element --
      plus a hardcoded counterexample ``arange(12) 3x4 -> (2,2)`` with ``bin(1,1)`` = 11
      (wrong order degenerates it to the seed 6).  See DESIGN.md section 1.6.1 N3 / section 9.2.
  C4  global-grid specialisation (out_H == out_W == 1): whole-plane max / argmax, last-NaN index
  C4b identity-grid specialisation (out_H == H_in and out_W == W_in): out = x, index = iota

Every item is checked over a case matrix of shapes x data kinds x dtypes.  All items must be
bit-exact (0 element mismatches).

Run:  /workspace/venv/bin/python verify_equiv.py
"""

import torch

NEG_INF = float("-inf")
NAN = float("nan")


# ---------------------------------------------------------------------------
# Baseline: the source sequential scalar state machine (DESIGN.md section 0.1)
# ---------------------------------------------------------------------------
def _bin(o: int, size_in: int, size_out: int):
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def baseline(x: torch.Tensor, out_h: int, out_w: int):
    """Source state machine: first-hit via strict ``>``, last-NaN index, NaN wins the value.

    ``x`` (N, C, H, W) already in the target dtype (fp64 copy).  Returns (val, idx).
    """
    n, c, h, w = x.shape
    val = torch.empty((n, c, out_h, out_w), dtype=torch.float64)
    idx = torch.empty((n, c, out_h, out_w), dtype=torch.int64)
    for oh in range(out_h):
        ih0, ih1 = _bin(oh, h, out_h)
        for ow in range(out_w):
            iw0, iw1 = _bin(ow, w, out_w)
            mv = torch.full((n, c), NEG_INF, dtype=torch.float64)
            mi = torch.zeros((n, c), dtype=torch.int64)
            ni = torch.zeros((n, c), dtype=torch.int64)   # source init: 0
            hn = torch.zeros((n, c), dtype=torch.bool)
            fv = torch.ones((n, c), dtype=torch.bool)     # first_valid
            for kh in range(ih1 - ih0):
                for kw in range(iw1 - iw0):
                    ih, iw = ih0 + kh, iw0 + kw
                    v = x[:, :, ih, iw]
                    f = ih * w + iw
                    isn = torch.isnan(v)
                    take_fv = (~isn) & fv
                    take_gt = (~isn) & (~fv) & (v > mv)
                    # NaN branch (source: nan_idx = flat; has_nan = True)
                    ni = torch.where(isn, torch.full_like(ni, f), ni)
                    hn = hn | isn
                    # first_valid branch
                    mv = torch.where(take_fv, v, mv)
                    mi = torch.where(take_fv, torch.full_like(mi, f), mi)
                    # strict-greater branch (first-hit tie-break)
                    mv = torch.where(take_gt, v, mv)
                    mi = torch.where(take_gt, torch.full_like(mi, f), mi)
                    fv = fv & isn
            val[:, :, oh, ow] = torch.where(hn, torch.tensor(NAN, dtype=torch.float64), mv)
            idx[:, :, oh, ow] = torch.where(hn, ni, mi)
    return val, idx


# ---------------------------------------------------------------------------
# Candidates (fp64 mirror of the vectorised NPU forms in DESIGN.md section 0.6 R3)
# ---------------------------------------------------------------------------
def _vector_core(x, out_h, out_w, vmax_is_nan_propagating: bool):
    """Running (acc, max_idx, nan_idx) with no ``first_valid``, plus N1/N2 repr.

    Returns (val, idx, has_nan_flag, first_idx, nan_index) so that N1/N2/N3 can be
    scored separately.
    """
    n, c, h, w = x.shape
    val = torch.empty((n, c, out_h, out_w), dtype=torch.float64)
    idx = torch.empty((n, c, out_h, out_w), dtype=torch.int64)
    flag = torch.empty((n, c, out_h, out_w), dtype=torch.bool)
    first = torch.empty((n, c, out_h, out_w), dtype=torch.int64)
    nanidx = torch.empty((n, c, out_h, out_w), dtype=torch.int64)

    def vmax(a, b):
        if vmax_is_nan_propagating:
            return torch.maximum(a, b)   # T.vmax semantics (NaN behaviour undocumented)
        return torch.fmax(a, b)          # NaN-ignoring convention

    for oh in range(out_h):
        ih0, ih1 = _bin(oh, h, out_h)
        for ow in range(out_w):
            iw0, iw1 = _bin(ow, w, out_w)
            acc = torch.full((n, c), NEG_INF, dtype=torch.float64)
            # max_idx pre-seeded to the FIRST visited element's flat index.
            midx = torch.full((n, c), ih0 * w + iw0, dtype=torch.int64)
            nidx = torch.full((n, c), -1, dtype=torch.int64)
            # N1: multiplicative eq-mask product (1 = NaN-free so far).
            hflg = torch.ones((n, c), dtype=torch.float64)
            for kh in range(ih1 - ih0):
                for kw in range(iw1 - iw0):
                    ih, iw = ih0 + kh, iw0 + kw
                    v = x[:, :, ih, iw]
                    f = ih * w + iw
                    isn = torch.isnan(v)
                    eq = (~isn).to(torch.float64)            # x == x
                    # N2: nan index = max flat among NaN positions.
                    nidx = torch.maximum(nidx, torch.where(isn, torch.full_like(nidx, f), torch.full_like(nidx, -1)))
                    # N1: flag product.
                    hflg = hflg * eq
                    # N3: first-hit index.  The source records the LAST record-breaker (strict
                    # ``>``); records occur at strictly increasing flat indices, so the max flat
                    # over record-breakers is exactly that last record.  ``max_idx`` is seeded to
                    # the first visited element's flat index (covers the all-(-inf) window, where
                    # no element is a record).
                    gt = v > acc
                    cand = torch.where(gt, torch.full_like(midx, f), torch.full_like(midx, -1))
                    midx = torch.maximum(midx, cand)
                    # value accumulator (vmax in-place).
                    acc = vmax(acc, v)
            has_nan = hflg == 0.0
            nan_i = nidx
            first_i = midx
            idx[:, :, oh, ow] = torch.where(has_nan, nan_i, first_i)
            val[:, :, oh, ow] = torch.where(
                has_nan, torch.tensor(NAN, dtype=torch.float64), acc
            )
            flag[:, :, oh, ow] = has_nan
            first[:, :, oh, ow] = first_i
            nanidx[:, :, oh, ow] = nan_i
    return val, idx, flag, first, nanidx


def _seed_index(shape, out_h, out_w):
    """Flat index of the FIRST visited element of each output window (the ``max_idx`` seed)."""
    n, c, h, w = shape
    seed = torch.empty((n, c, out_h, out_w), dtype=torch.int64)
    for oh in range(out_h):
        ih0, _ = _bin(oh, h, out_h)
        for ow in range(out_w):
            iw0, _ = _bin(ow, w, out_w)
            seed[:, :, oh, ow] = ih0 * w + iw0
    return seed


def _running_first_idx(x, out_h, out_w, value_first: bool):
    """Running (value, first-hit index, last-NaN index) state machine.

    ``value_first=False`` is the correct order (DESIGN.md section 3.3/6.2, frozen decision):
    ``gt = v > acc`` is taken *before* the in-place ``acc = vmax(acc, v)`` update, so the
    strict-greater test compares against the PRE-update accumulator -- exactly the source's
    ``val > max_val``.

    ``value_first=True`` reproduces the order regression of DESIGN.md v0 (value accumulator
    updated first): then ``gt = v > max(acc, v)`` is identically false for every finite ``v``,
    so ``max_idx`` stays at its seed and the index degenerates to the window's first visited
    element.  Used only as a negative control by the ``N3-ord`` guard.
    """
    n, c, h, w = x.shape
    val = torch.empty((n, c, out_h, out_w), dtype=torch.float64)
    idx = torch.empty((n, c, out_h, out_w), dtype=torch.int64)
    for oh in range(out_h):
        ih0, ih1 = _bin(oh, h, out_h)
        for ow in range(out_w):
            iw0, iw1 = _bin(ow, w, out_w)
            acc = torch.full((n, c), NEG_INF, dtype=torch.float64)
            midx = torch.full((n, c), ih0 * w + iw0, dtype=torch.int64)
            nidx = torch.full((n, c), -1, dtype=torch.int64)
            hflg = torch.ones((n, c), dtype=torch.float64)
            for kh in range(ih1 - ih0):
                for kw in range(iw1 - iw0):
                    ih, iw = ih0 + kh, iw0 + kw
                    v = x[:, :, ih, iw]
                    f = ih * w + iw
                    isn = torch.isnan(v)
                    nidx = torch.maximum(
                        nidx, torch.where(isn, torch.full_like(nidx, f), torch.full_like(nidx, -1))
                    )
                    hflg = hflg * (~isn).to(torch.float64)
                    if value_first:
                        acc = torch.maximum(acc, v)      # ORDER BUG: update before compare
                        gt = v > acc
                    else:
                        gt = v > acc                     # correct: compare the pre-update acc
                        acc = torch.maximum(acc, v)
                    cand = torch.where(gt, torch.full_like(midx, f), torch.full_like(midx, -1))
                    midx = torch.maximum(midx, cand)
            has_nan = hflg == 0.0
            idx[:, :, oh, ow] = torch.where(has_nan, nidx, midx)
            val[:, :, oh, ow] = torch.where(has_nan, torch.tensor(NAN, dtype=torch.float64), acc)
    return val, idx


def candidate_global(x):
    """C4: whole-plane max / argmax (first-hit) / last-NaN index, out = 1x1."""
    n, c, h, w = x.shape
    plane = x.reshape(n, c, h * w)
    acc = plane.amax(dim=2, keepdim=True)
    isn = torch.isnan(plane)
    has_nan = isn.any(dim=2, keepdim=True)
    k = torch.arange(h * w, dtype=torch.int64).reshape(1, 1, h * w)
    BIG = h * w
    eq = plane == acc
    first = torch.where(eq, k, torch.full_like(k, BIG)).amin(dim=2, keepdim=True)
    nan_i = torch.where(isn, k, torch.full_like(k, -1)).amax(dim=2, keepdim=True)
    idx = torch.where(has_nan, nan_i, first)
    val = torch.where(has_nan, torch.tensor(NAN, dtype=torch.float64), acc)
    return val.reshape(n, c, 1, 1), idx.reshape(n, c, 1, 1)


def candidate_identity(x):
    """C4b: out = x, index = flat iota (channel/batch independent)."""
    n, c, h, w = x.shape
    k = torch.arange(h * w, dtype=torch.int64).reshape(1, 1, h, w).expand(n, c, h, w)
    return x.clone(), k.clone()


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------
def _data(shape, kind):
    n, c, h, w = shape
    g = torch.Generator().manual_seed(1234 + h * 7 + w)
    if kind == "rand":
        return torch.randn(n, c, h, w, generator=g)
    if kind == "nan":
        x = torch.randn(n, c, h, w, generator=g)
        x[0, 0, 0, 0] = NAN
        x[0, min(1, c - 1), h - 1, w - 1] = NAN
        return x
    if kind == "nan_first":
        x = torch.randn(n, c, h, w, generator=g)
        x[0, 0, 0, 0] = NAN          # first visited element is NaN
        return x
    if kind == "inf":
        x = torch.randn(n, c, h, w, generator=g)
        x[0, 0, 0, 0] = float("inf")
        x[0, 0, 0, min(1, w - 1)] = float("-inf")
        return x
    if kind == "all_equal":
        return torch.full((n, c, h, w), 3.7)
    if kind == "all_neg_inf":
        return torch.full((n, c, h, w), NEG_INF)
    if kind == "tie_two_max":
        x = torch.zeros(n, c, h, w)
        x[0, 0, 0, 0] = 9.0
        x[0, 0, h - 1, w - 1] = 9.0
        return x
    if kind == "subnormal":
        return torch.randn(n, c, h, w, generator=g) * 1e-6
    if kind == "large":
        return torch.randn(n, c, h, w, generator=g) * 100.0
    if kind == "zeros":
        return torch.zeros(n, c, h, w)
    raise ValueError(kind)


KINDS = (
    "rand",
    "nan",
    "nan_first",
    "inf",
    "all_equal",
    "all_neg_inf",
    "tie_two_max",
    "subnormal",
    "large",
    "zeros",
)

# (shape, out_h, out_w) groups: general / global (C4) / identity (C4b) / expanding / non-div.
# Shapes are structural stand-ins (small, so the sequential baseline stays fast) that keep the
# semantics under test: non-divisible overlapping bins, expanding bins, partial-``None``,
# single output column, channel tail, global-grid and identity-grid.
GENERAL = [
    ((1, 8, 4, 4), 2, 2),
    ((1, 4, 11, 13), 6, 6),        # non-divisible overlap (11/6, 13/6)
    ((1, 4, 10, 12), 7, 7),        # non-divisible (10/7, 12/7)
    ((1, 12, 7, 9), 5, 3),         # asymmetric + channel tail
    ((1, 16, 11, 13), 5, 4),       # harness test shape
    ((1, 16, 11, 13), 5, 13),      # partial-None (W identity, H not)
    ((2, 8, 9, 11), 3, 1),         # single output column
    ((1, 8, 4, 4), 7, 7),          # expanding bins
    ((1, 1, 1, 1), 1, 1),          # minimal (global == identity)
]
GLOBAL = [
    ((1, 8, 4, 4), 1, 1),
    ((8, 8, 7, 7), 1, 1),
    ((2, 4, 13, 17), 1, 1),
    ((1, 1, 1, 1), 1, 1),
]
IDENTITY = [
    ((1, 8, 4, 4), 4, 4),
    ((1, 8, 9, 11), 9, 11),
    ((2, 4, 9, 11), 9, 11),
    ((1, 1, 1, 1), 1, 1),
]
DTYPES = (torch.float16, torch.bfloat16)


# ---------------------------------------------------------------------------
# Comparison helpers (NaN aware, exact)
# ---------------------------------------------------------------------------
def _val_mismatch(a, b):
    an, bn = torch.isnan(a), torch.isnan(b)
    eq = (a == b) | (an & bn)
    finite = ~an & ~bn
    mad = 0.0
    if finite.any():
        mad = float((a[finite] - b[finite]).abs().max().item())
    return int((~eq).sum().item()), int((an ^ bn).sum().item()), mad


def _idx_mismatch(a, b):
    return int((a != b).sum().item())


def _order_counterexample_ok() -> bool:
    """Hardcoded order-regression guard (DESIGN.md section 1.6.1 N3 / section 9.2).

    ``arange(12).reshape(1,1,3,4) -> (2,2)``: ``bin(1,1)`` covers flat 4..11, whose max is at
    flat 11, so the correct first-hit index is 11; with the v0 order (value accumulator updated
    first) the strict-greater test is always false and the index degenerates to the seed 6.
    """
    x = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    _, bidx = baseline(x, 2, 2)
    _, ok_idx = _running_first_idx(x, 2, 2, value_first=False)
    _, bad_idx = _running_first_idx(x, 2, 2, value_first=True)
    return (
        int(bidx[0, 0, 1, 1]) == 11
        and int(ok_idx[0, 0, 1, 1]) == 11
        and int(bad_idx[0, 0, 1, 1]) != 11
    )


def main():
    stats = {k: [0, 0, 0, 0.0] for k in
             ("N1", "N2", "N3", "N3b", "N3-ord", "C4", "C4b")}  # wrong, cases, viol, worst
    total_cases = 0

    # Hardcoded order-regression counterexample (2 checks: correct == 11, wrong != 11).
    sord = stats["N3-ord"]
    sord[1] += 2
    if not _order_counterexample_ok():
        sord[0] += 2
        sord[2] += 1

    for dtype in DTYPES:
        dt = str(dtype).replace("torch.", "")
        groups = [("general", GENERAL), ("global", GLOBAL), ("identity", IDENTITY)]
        for gname, group in groups:
            for shape, oh, ow in group:
                for kind in KINDS:
                    x = _data(shape, kind).to(dtype).to(torch.float64)
                    bval, bidx = baseline(x, oh, ow)
                    total_cases += 1

                    # N1 / N2 / N3 / N3b from the vectorised running core.
                    for item, prop in (("N3", True), ("N3b", False)):
                        v, i, flg, frst, nan_i = _vector_core(x, oh, ow, prop)
                        vmis, _, mad = _val_mismatch(v, bval)
                        imis = _idx_mismatch(i, bidx)
                        bad = vmis + imis
                        s = stats[item]
                        s[0] += bad
                        s[1] += 1
                        s[3] = max(s[3], mad)
                        if bad:
                            s[2] += 1
                    # N1: has_nan flag vs baseline (baseline has_nan == isnan(val))
                    v0, i0, flg0, frst0, nan_i0 = _vector_core(x, oh, ow, True)
                    b_has_nan = torch.isnan(bval)
                    m1 = int((flg0 != b_has_nan).sum().item())
                    # N2 is scored on the NaN-containing outputs only (nan_idx is otherwise unused).
                    m2 = int((nan_i0[b_has_nan] != bidx[b_has_nan]).sum().item())
                    s1 = stats["N1"]
                    s1[0] += m1
                    s1[1] += 1
                    if m1:
                        s1[2] += 1
                    s2 = stats["N2"]
                    s2[0] += m2
                    s2[1] += 1
                    if m2:
                        s2[2] += 1

                    # N3-ord: implementation-order guard.  (a) the correct order (gt before the
                    # in-place value update) must reproduce the baseline; (b) the value-first
                    # regression must be *detected* -- i.e. wherever the true first hit leaves the
                    # window's first visited element on a NaN-free window, the wrong order must
                    # disagree with the baseline (its index collapses to the seed f0).
                    okv, oki = _running_first_idx(x, oh, ow, value_first=False)
                    badv, badi = _running_first_idx(x, oh, ow, value_first=True)
                    seed = _seed_index(shape, oh, ow)
                    order_ok = (_val_mismatch(okv, bval)[0] + _idx_mismatch(oki, bidx)) == 0
                    detectable = (oki != seed) & (~torch.isnan(bval))
                    guard_ok = bool((~detectable | (badi != bidx)).all().item())
                    s = stats["N3-ord"]
                    s[1] += 1
                    if not (order_ok and guard_ok):
                        s[0] += 1
                        s[2] += 1

                    # C4 / C4b structural specialisations (compared to the baseline).
                    for item, applies, fn in (
                        ("C4", (oh == 1 and ow == 1), candidate_global),
                        ("C4b", (oh == shape[2] and ow == shape[3]), candidate_identity),
                    ):
                        if not applies:
                            continue
                        cv, ci = fn(x)
                        vmis, _, mad = _val_mismatch(cv, bval)
                        imis = _idx_mismatch(ci, bidx)
                        bad = vmis + imis
                        s = stats[item]
                        s[0] += bad
                        s[1] += 1
                        s[3] = max(s[3], mad)
                        if bad:
                            s[2] += 1

    print(f"{'item':6s} | {'checks':>6s} | {'worst ulp-ish diff':>18s} | {'miss':>5s} | {'viol_rate':>9s} | status")
    print("-" * 70)
    all_ok = True
    for item in ("N1", "N2", "N3", "N3b", "N3-ord", "C4", "C4b"):
        wrong, checks, viol, mad = stats[item]
        status = "EQUIV_PASS" if wrong == 0 else "EQUIV_FAIL"
        all_ok &= wrong == 0
        rate = (wrong / checks) if checks else 0.0
        print(f"{item:6s} | {checks:6d} | {mad:18.1e} | {wrong:5d} | {rate:8.1%} | {status}")
    print("-" * 70)
    print("criterion: exact (0 mismatches) after rounding to the target dtype (value diff = 0.0, i.e. no ULP deviation); "
          "NaN-vs-NaN counts equal, NaN-vs-finite fails; indices compared exactly.")
    print(f"total (shape, out, kind) cases swept: {total_cases} x {len(DTYPES)} dtypes")
    if all_ok:
        print("[EQUIV] ALL PASS")
    else:
        print("[EQUIV] FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
