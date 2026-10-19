"""D-1 equivalence verification for the max_pool1d NPU design (DESIGN.md 1.6.1).

Migrated op: ``_max_pool1d_kernel`` (TileOPs ``MaxPool1dFwdOp`` /
``torch.nn.functional.max_pool1d(return_indices=False)``).

Purpose
-------
Machine-verify, on torch CPU in **fp64**, that every mathematically-equivalent
optimisation adopted in DESIGN.md section 1.6.1 computes the same output as the
baseline windowed-max formula (bit-for-bit after rounding to the target dtype).

Checks
------
* ``E1``  -- the NPU baseline formulation (accumulator seeded to ``-inf``,
  out-of-length lanes *skipped* == ``-inf`` padding) against the golden
  ``F.max_pool1d``.  This is the core semantic claim of the design.  Includes a
  ``ceil_mode`` **step-back** case (``in=8, k=3, s=4, ceil`` -> ``out=2``).
* ``C1``  -- the clamped contiguous load (one ``(CH, LSPAN)`` GM->UB span per
  logical unit; the L window is read out of it) against the naive per-element
  guarded baseline.  The check also asserts the design's index identity
  ``lo <= iw < hi  <=>  0 <= iw < L_in`` (within the window range) with
  ``lo = max(0, base)``, ``hi = min(L_in, base + span)`` and verifies that every
  in-bounds read index ``iw - s0`` lands inside ``[0, LSPAN)``.
* ``C4``  -- whole-**line** reduction ("global" specialisation) against the
  baseline, restricted to configs where the **corrected** sufficient condition
  holds: ``L_out == 1 and p_W == 0 and d_W == 1 and (k_W-1)*d_W + 1 >= L_in``.
* ``C4x`` -- **dilation counterexamples** (``d_W > 1``): configs with ``out==1``
  that satisfy every clause *except* ``d_W==1``.  They must NOT be routed to the
  whole-line path; the general path is checked against ``F.max_pool1d``.  This
  is the 1D regression guard for the design-skill transferable lesson
  "整平面特化必须显式要求 d==1" (max_pool2d REVIEW problem 1) -- the window's
  coverage is ``{k_w * d_W} & [0, L_in)``, which equals the whole line only when
  ``d_W == 1``.
* ``C4b`` -- identity copy (only valid when the window is a single element and
  ``out == in``) against the baseline.

Comparison rule
---------------
Both sides are rounded to the target dtype and compared element-wise; a
mismatch is ``~(cand == base) & ~(isnan(cand) & isnan(base))`` (NaN-vs-NaN
counts equal, NaN-vs-finite always fails).  Every comparison asserts that both
sides are non-empty, so an "empty comparison" cannot pass vacuously.  Corner
data kinds cover ``rand / nan / inf(+/-) / subnormal / large / zeros``.

Run:  /workspace/venv/bin/python examples/max_pool1d/_max_pool1d_kernel/verify_equiv.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.manual_seed(0)

_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


# ---------------------------------------------------------------------------
# Helper: source output extent (pool_output_dim, verbatim from the source repo)
# ---------------------------------------------------------------------------
def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1
    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1
    return max(out, 0)


def out_len(l_in, kw, sw, pw, dw, ceil_mode):
    return pool_output_dim(l_in, kw, sw, pw, ceil_mode, dw)


def c4_condition(l_in, kw, sw, pw, dw, ceil_mode):
    """Corrected C4 sufficient condition (DESIGN.md 1.6.1, 1D form).

    ``d_W == 1`` is required: the window's coverage is ``{k_w * d_W} & [0, L_in)``
    which equals ``[0, L_in)`` only when ``d_W == 1``.  A conservative form is
    used (the ``L_in == 1`` escape hatch is NOT exploited).
    """
    return (
        out_len(l_in, kw, sw, pw, dw, ceil_mode) == 1
        and pw == 0
        and dw == 1
        and (kw - 1) * dw + 1 >= l_in
    )


# ---------------------------------------------------------------------------
# Baseline: general windowed max, -inf seed, out-of-bounds lanes skipped.
# NaN is tracked explicitly (as in the source kernel) so that "a NaN anywhere
# in the window wins" holds regardless of torch.maximum's propagation.
# ---------------------------------------------------------------------------
def baseline(x64, kw, sw, pw, dw, ceil_mode):
    n, c, l_in = x64.shape
    lo_n = out_len(l_in, kw, sw, pw, dw, ceil_mode)
    out = torch.full((n, c, lo_n), float("-inf"), dtype=torch.float64)
    nan_flag = torch.zeros((n, c, lo_n), dtype=torch.bool)
    for ow in range(lo_n):
        for kw_i in range(kw):
            iw = ow * sw - pw + kw_i * dw
            if not (0 <= iw < l_in):
                continue
            v = x64[:, :, iw]
            out[:, :, ow] = torch.maximum(out[:, :, ow], v)
            nan_flag[:, :, ow] |= torch.isnan(v)
    out = torch.where(nan_flag, torch.full_like(out, float("nan")), out)
    return out


def clamped_load(x64, kw, sw, pw, dw, ceil_mode, bl):
    """Emulate the NPU design's clamped contiguous load + buffer read (DESIGN 1.6.1 C1).

    For each output block of ``bl`` positions the design loads a contiguous input
    span ``[s0, s0 + LSPAN)`` (``LSPAN = min(span, L_in)``,
    ``s0 = clamp(base, 0, L_in - LSPAN)``, ``base = ow0*sw - pw``) and reads the
    window out of the buffer at index ``iw - s0``.  This function reproduces that
    arithmetic exactly (asserting the index lands inside the buffer and that the
    read guard matches ``0 <= iw < L_in``), then compares against the baseline.
    """
    n, c, l_in = x64.shape
    lo_n = out_len(l_in, kw, sw, pw, dw, ceil_mode)
    out = torch.full((n, c, lo_n), float("-inf"), dtype=torch.float64)
    nan_flag = torch.zeros((n, c, lo_n), dtype=torch.bool)
    num_oblk = (lo_n + bl - 1) // bl
    for oblk in range(num_oblk):
        ow0 = oblk * bl
        base = ow0 * sw - pw
        span = (bl - 1) * sw + (kw - 1) * dw + 1
        lspan = min(span, l_in)
        s0 = max(0, min(base, l_in - lspan))
        # emulate the clamped contiguous copy: buffer covers [s0, s0+lspan)
        buf = x64[:, :, s0:s0 + lspan]
        assert buf.shape[-1] == lspan and s0 + lspan <= l_in
        for owl in range(bl):
            ow = ow0 + owl
            if ow >= lo_n:
                break
            for kw_i in range(kw):
                iw = base + owl * sw + kw_i * dw
                in_bounds = 0 <= iw < l_in
                # index identity (DESIGN 1.6.1 C1): lo <= iw < hi  <=>  in_bounds
                assert in_bounds == (s0 <= iw < s0 + lspan), (iw, s0, lspan)
                if not in_bounds:
                    continue
                j = iw - s0
                assert 0 <= j < lspan, (j, lspan)
                v = buf[:, :, j]
                out[:, :, ow] = torch.maximum(out[:, :, ow], v)
                nan_flag[:, :, ow] |= torch.isnan(v)
    out = torch.where(nan_flag, torch.full_like(out, float("nan")), out)
    return out


def c4_whole_line(x64):
    n, c, l_in = x64.shape
    return x64.max(dim=-1).values.reshape(n, c, 1)


# ---------------------------------------------------------------------------
# Data generators (fp64 master copies; cast down per dtype at comparison time)
# ---------------------------------------------------------------------------
def make_data(kind, shape):
    if kind == "rand":
        return torch.randn(shape, dtype=torch.float64)
    if kind == "zeros":
        return torch.zeros(shape, dtype=torch.float64)
    if kind == "large":
        return torch.randn(shape, dtype=torch.float64) * 6.0e4
    if kind == "subnormal":
        return torch.randn(shape, dtype=torch.float64) * 5e-5
    if kind == "nan":
        x = torch.randn(shape, dtype=torch.float64)
        x.reshape(-1)[0] = float("nan")
        x.reshape(-1)[len(x.reshape(-1)) // 2] = float("nan")
        return x
    if kind == "inf":
        x = torch.randn(shape, dtype=torch.float64)
        flat = x.reshape(-1)
        flat[0] = float("inf")
        flat[1] = float("-inf")
        flat[len(flat) // 3] = float("inf")
        return x
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def mismatch(cand64, base64, dtype):
    assert cand64.numel() > 0 and base64.numel() > 0, "empty comparison is not allowed"
    assert cand64.shape == base64.shape, (
        f"shape mismatch {tuple(cand64.shape)} vs {tuple(base64.shape)}"
    )
    c = cand64.to(dtype)
    b = base64.to(dtype)
    eq = c == b
    both_nan = torch.isnan(c) & torch.isnan(b)
    bad = ~(eq | both_nan)
    n_bad = int(bad.sum().item())
    if n_bad == 0:
        worst = 0.0
    else:
        d = (c.float() - b.float()).abs()
        d = torch.where(bad, d, torch.zeros_like(d))
        worst = float(d.max().item())
    return n_bad, worst


_KINDS6 = ("rand", "nan", "inf", "subnormal", "large", "zeros")


def hdr():
    print(f"{'check':<5} {'dtype':<9} {'shape':<16} {'kW':<4} {'kinds':<6} "
          f"{'worst|d|':<11} {'viol':<6} status")
    print("-" * 92)


def emit(rows):
    for (name, dt, shape, kw, kinds, worst, viol, status) in rows:
        print(f"{name:<5} {dt:<9} {shape:<16} {kw:<4} {kinds:<6} "
              f"{worst:<11.4e} {viol:<6} {status}")


def main():
    all_rows = []

    # ---- E1: NPU baseline (guarded, -inf seed) vs golden F.max_pool1d ----
    e1_cfg = [
        ((1, 8, 32), 3, 2, 1, 1, False),    # smoke (k3 s2 p1)
        ((2, 8, 64), 3, 2, 1, 1, False),    # smoke fp16/bf16
        ((1, 4, 63), 3, 3, 1, 2, False),    # stride=None (==k) + dilation 2
        ((1, 4, 97), 5, 3, 2, 1, True),     # ceil + non-disjoint windows
        ((1, 8, 128), 128, 1, 0, 1, False),  # textcnn-global (C4 config, E1 sanity)
        ((16, 8, 2048), 5, 2, 2, 2, True),  # ecg dilated + ceil
        ((1, 8, 8), 3, 4, 0, 1, True),      # ceil step-back: out=2
    ]
    e1_rows = []
    for cfg in e1_cfg:
        shape, kw, sw, pw, dw, ceil_mode = cfg
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in _KINDS6:
                x64 = make_data(kind, shape)
                base = baseline(x64, kw, sw, pw, dw, ceil_mode)
                ref = F.max_pool1d(x64, kw, sw, pw, dw, ceil_mode)
                nb, wd = mismatch(base, ref, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            e1_rows.append(("E1", str(dtype).replace("torch.", ""), str(shape),
                            kw, 6, worst, viol, status))
    all_rows += e1_rows

    # ---- C1: clamped contiguous load vs naive guarded baseline ----
    c1_rows = []
    for cfg in e1_cfg:
        shape, kw, sw, pw, dw, ceil_mode = cfg
        for bl in (4, 8, 16):
            for dtype in _DTYPES:
                worst = 0.0
                viol = 0
                for kind in _KINDS6:
                    x64 = make_data(kind, shape)
                    base = baseline(x64, kw, sw, pw, dw, ceil_mode)
                    cand = clamped_load(x64, kw, sw, pw, dw, ceil_mode, bl)
                    nb, wd = mismatch(cand, base, dtype)
                    viol += nb
                    worst = max(worst, wd)
                status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
                c1_rows.append(("C1", str(dtype).replace("torch.", ""),
                                str(shape), kw, 6, worst, viol, status))
    all_rows += c1_rows

    # ---- C4: whole-line reduction on C4-valid configs (all d == 1) ----
    c4_cfg = [
        ((1, 8, 128), 128, 1, 0, 1, False),   # textcnn-global: k == in
        ((1, 8, 32), 32, 1, 0, 1, False),     # k == in
        ((2, 4, 17), 17, 1, 0, 1, False),     # non-power-of-two, k == in
        ((1, 16, 64), 64, 1, 0, 1, False),    # C=16
        ((3, 8, 33), 33, 1, 0, 1, False),     # k == in (odd length)
    ]
    c4_rows = []
    for cfg in c4_cfg:
        shape, kw, sw, pw, dw, ceil_mode = cfg
        l_in = shape[2]
        assert c4_condition(l_in, kw, sw, pw, dw, ceil_mode), (
            f"C4 config must satisfy the condition: {cfg}"
        )
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in _KINDS6:
                x64 = make_data(kind, shape)
                base = baseline(x64, kw, sw, pw, dw, ceil_mode)
                cand = c4_whole_line(x64)
                nb, wd = mismatch(cand, base, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            c4_rows.append(("C4", str(dtype).replace("torch.", ""), str(shape),
                            kw, 6, worst, viol, status))
    all_rows += c4_rows

    # ---- C4x: dilation counterexamples.  out==1 but d>1, so the window covers
    #      only {k*d} positions.  The corrected condition must reject them; the
    #      general path must match F.max_pool1d. ----
    c4x_cfg = [
        ((1, 8, 5), 2, 1, 0, 4, False),   # window = {0, 4}
        ((1, 8, 7), 3, 1, 0, 3, False),   # window = {0, 3, 6}
    ]
    c4x_rows = []
    for cfg in c4x_cfg:
        shape, kw, sw, pw, dw, ceil_mode = cfg
        l_in = shape[2]
        assert not c4_condition(l_in, kw, sw, pw, dw, ceil_mode), (
            "C4x config must NOT satisfy the C4 condition: " + str(cfg)
        )
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in _KINDS6:
                x64 = make_data(kind, shape)
                cand = clamped_load(x64, kw, sw, pw, dw, ceil_mode, 4)
                ref = F.max_pool1d(x64, kw, sw, pw, dw, ceil_mode)
                nb, wd = mismatch(cand, ref, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            c4x_rows.append(("C4x", str(dtype).replace("torch.", ""),
                             str(shape), kw, 6, worst, viol, status))
    all_rows += c4x_rows

    # ---- C4b: identity copy on C4b-valid configs (k=d=1, p=0, s=1) ----
    c4b_cfg = [
        ((1, 8, 32), 1, 1, 0, 1, False),
        ((2, 4, 17), 1, 1, 0, 1, False),
        ((1, 16, 64), 1, 1, 0, 1, False),
        ((3, 33, 25), 1, 1, 0, 1, False),
    ]
    c4b_rows = []
    for cfg in c4b_cfg:
        shape, kw, sw, pw, dw, ceil_mode = cfg
        assert (
            out_len(shape[2], kw, sw, pw, dw, ceil_mode) == shape[2]
            and c4_condition(shape[2], kw, sw, pw, dw,
                             ceil_mode) is False
        ), cfg
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in _KINDS6:
                x64 = make_data(kind, shape)
                base = baseline(x64, kw, sw, pw, dw, ceil_mode)
                cand = x64.clone()
                nb, wd = mismatch(cand, base, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            c4b_rows.append(("C4b", str(dtype).replace("torch.", ""),
                             str(shape), kw, 6, worst, viol, status))
    all_rows += c4b_rows

    hdr()
    emit(all_rows)

    # ---- Hazard evidence: the pre-fix (missing-dilation) condition would have
    #      routed this d>1 config to the whole-line path and produced 1.0 where
    #      the correct answer is 0.0.  The corrected guard rejects it. ----
    adv = torch.zeros((1, 1, 5), dtype=torch.float64)
    adv[0, 0, 2] = 1.0
    gen = float(clamped_load(adv, 2, 1, 0, 4, False, 4).item())
    ref = float(F.max_pool1d(adv, 2, 1, 0, 4, False).item())
    wrong = float(c4_whole_line(adv).item())
    assert c4_condition(5, 2, 1, 0, 4, False) is False
    assert gen == ref == 0.0 and wrong == 1.0, (gen, ref, wrong)
    print()
    print("[C4x-GUARD] dilation counterexample (L=5,k=2,s=1,p=0,d=4,x[2]=1):")
    print(f"[C4x-GUARD]   general path = {gen}  == golden {ref};  "
          f"whole-line = {wrong} (excluded by the d==1 guard)")
    print("[C4x-GUARD]   OK — corrected C4 condition rejects d>1")

    n_fail = sum(1 for r in all_rows if r[-1] == "EQUIV_FAIL")
    print()
    print("criterion: exact — 0 element mismatches after rounding to the target")
    print("dtype (NaN-vs-NaN counts equal; NaN-vs-finite always fails).")
    print(f"[EQUIV] {'ALL PASS' if n_fail == 0 else 'FAILURES'} — "
          f"{len(all_rows)} case-groups, {n_fail} failures")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
