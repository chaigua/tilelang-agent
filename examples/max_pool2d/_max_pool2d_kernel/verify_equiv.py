"""D-1 equivalence verification for the max_pool2d NPU design (DESIGN.md 1.6.1).

Migrated op: ``_max_pool2d_kernel`` (TileOPs ``MaxPool2dFwdOp`` /
``torch.nn.functional.max_pool2d(return_indices=False)``).

Purpose
-------
Machine-verify, on torch CPU in **fp64**, that every mathematically-equivalent
optimisation adopted in DESIGN.md section 1.6.1 computes the same output as the
baseline windowed-max formula (bit-for-bit after rounding to the target dtype).

Checks
------
* ``E1``  -- the NPU baseline formulation (accumulator seeded to ``-inf``,
  out-of-bounds lanes *skipped* == ``-inf`` padding) against the golden
  ``F.max_pool2d``.  This is the core semantic claim of the design.  Includes a
  ``ceil_mode`` **step-back** case (``in=8, k=3, s=4, ceil`` -> ``out=2``).
* ``C1``  -- guard-hoisted iteration (the H in-bounds test hoisted out of the
  ``ow``/``kw`` element loops) against the naive per-element guarded baseline.
* ``C4``  -- whole-plane reduction ("global" specialisation) against the
  baseline, restricted to configs where the **corrected** sufficient condition
  holds: ``out_H==out_W==1 and p_H==p_W==0 and d_H==d_W==1 and
  (k_H-1)*d_H+1 >= H_in and (k_W-1)*d_W+1 >= W_in``.
* ``C4x`` -- **dilation counterexamples** (``d>1``): configs with ``out==1x1``
  that satisfy every clause *except* ``d==1``.  They must NOT be routed to the
  whole-plane path; the general path is checked against ``F.max_pool2d``.
  This is the regression guard for REVIEW.md problem 1 (a ``d>1`` config would
  otherwise silently take the whole-plane reduction and return a wrong value,
  because the window covers only ``{k_h*d_H} x {k_w*d_W}``, not the full plane).
* ``C4b`` -- identity copy (only valid when the window is a single element and
  ``out == in``) against the baseline.

Comparison rule
---------------
Both sides are rounded to the target dtype and compared element-wise; a
mismatch is ``~(cand == base) & ~(isnan(cand) & isnan(base))`` (NaN-vs-NaN
counts equal, NaN-vs-finite always fails).  Every comparison asserts that both
sides are non-empty, so an "empty comparison" cannot pass vacuously.  Corner
data kinds cover ``rand / nan / inf(+/-) / subnormal / large / zeros``.

Run:  /workspace/venv/bin/python verify_equiv.py
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


def out_dims(h, w, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode):
    return (
        pool_output_dim(h, kH, sH, pH, ceil_mode, dH),
        pool_output_dim(w, kW, sW, pW, ceil_mode, dW),
    )


def c4_condition(h, w, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode):
    """Corrected C4 sufficient condition (DESIGN.md 1.6.1).

    ``d_H == d_W == 1`` is required: the window's H coverage is
    ``{k_h * d_H} & [0, H_in)`` which equals ``[0, H_in)`` only when ``d_H == 1``.
    A conservative form is used (the ``H_in == 1`` escape hatch is NOT exploited).
    """
    oh, ow = out_dims(h, w, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)
    return (
        oh == 1
        and ow == 1
        and pH == 0
        and pW == 0
        and dH == 1
        and dW == 1
        and (kH - 1) * dH + 1 >= h
        and (kW - 1) * dW + 1 >= w
    )


# ---------------------------------------------------------------------------
# Baseline: general windowed max, -inf seed, out-of-bounds lanes skipped.
# NaN is tracked explicitly (as in the source kernel) so that "a NaN anywhere
# in the window wins" holds regardless of torch.maximum's propagation.
# ---------------------------------------------------------------------------
def baseline(x64, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode, hoist=False):
    n, c, h, w = x64.shape
    oh_n, ow_n = out_dims(h, w, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)
    out = torch.full((n, c, oh_n, ow_n), float("-inf"), dtype=torch.float64)
    nan_flag = torch.zeros((n, c, oh_n, ow_n), dtype=torch.bool)
    for oh in range(oh_n):
        for ow in range(ow_n):
            for kh in range(kH):
                ih = oh * sH - pH + kh * dH
                # hoist: the H in-bounds test depends only on (oh, kh); pull it
                # out of the kw loop (DESIGN.md 1.6.1 C1).
                if hoist and not (0 <= ih < h):
                    continue
                for kw in range(kW):
                    iw = ow * sW - pW + kw * dW
                    if not hoist and not (0 <= ih < h):
                        continue
                    if not (0 <= iw < w):
                        continue
                    v = x64[:, :, ih, iw]
                    out[:, :, oh, ow] = torch.maximum(out[:, :, oh, ow], v)
                    nan_flag[:, :, oh, ow] |= torch.isnan(v)
    # NaN wins: any NaN in the window -> NaN output (source kernel semantics).
    out = torch.where(nan_flag, torch.full_like(out, float("nan")), out)
    return out


def c4_whole_plane(x64):
    n, c, h, w = x64.shape
    return x64.reshape(n, c, h * w).max(dim=-1).values.reshape(n, c, 1, 1)


# ---------------------------------------------------------------------------
# Data generators (fp64 master copies; cast down per dtype at comparison time)
# ---------------------------------------------------------------------------
def make_data(kind, shape):
    n, c, h, w = shape
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


def run_check(name, configs, kinds, make_candidate, require=None):
    """configs: list of (shape, kH,kW,sH,sW,pH,pW,dH,dW,ceil). Returns rows.

    ``require(cfg)`` (if given) must hold for every config; this is how the
    "must be routed to / away from the C4 path" contract is asserted.
    """
    rows = []
    for cfg in configs:
        if require is not None:
            assert require(cfg), f"{name}: config {cfg} violates the required predicate"
        shape, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode = cfg
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in kinds:
                x64 = make_data(kind, shape)
                base = baseline(x64, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode,
                                hoist=False)
                cand = make_candidate(x64, kH, kW, sH, sW, pH, pW, dH, dW,
                                      ceil_mode)
                nb, wd = mismatch(cand, base, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            rows.append((name, str(dtype).replace("torch.", ""), str(shape),
                         kH, kW, len(kinds), worst, viol, status))
    return rows


def hdr():
    print(f"{'check':<5} {'dtype':<9} {'shape':<16} {'kH,kW':<7} {'kinds':<6} "
          f"{'worst|d|':<11} {'viol':<6} status")
    print("-" * 92)


def emit(rows):
    for (name, dt, shape, kH, kW, kinds, worst, viol, status) in rows:
        print(f"{name:<5} {dt:<9} {shape:<16} {str((kH, kW)):<7} {kinds:<6} "
              f"{worst:<11.4e} {viol:<6} {status}")


def main():
    all_rows = []

    # ---- E1: NPU baseline (guarded, -inf seed) vs golden F.max_pool2d ----
    e1_cfg = [
        ((1, 8, 16, 16), 3, 3, 2, 2, 1, 1, 1, 1, False),   # smoke
        ((2, 4, 20, 20), 3, 3, 3, 3, 1, 1, 2, 1, False),   # default stride+dilation
        ((1, 4, 23, 27), 3, 5, 2, 3, 1, 2, 1, 1, True),    # non-square + ceil
        ((2, 4, 21, 21), 3, 3, 2, 2, 0, 0, 1, 1, True),    # ceil mode (no step-back)
        ((1, 8, 12, 14), 2, 2, 2, 2, 0, 0, 1, 1, False),   # vgg-style disjoint
        ((1, 8, 8, 8), 3, 3, 4, 4, 0, 0, 1, 1, True),      # ceil step-back: out=2
    ]
    e1_rows = []
    for cfg in e1_cfg:
        shape, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode = cfg
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in _KINDS6:
                x64 = make_data(kind, shape)
                base = baseline(x64, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode)
                ref = F.max_pool2d(
                    x64, (kH, kW), (sH, sW), (pH, pW), (dH, dW), ceil_mode
                )
                nb, wd = mismatch(base, ref, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            e1_rows.append(("E1", str(dtype).replace("torch.", ""), str(shape),
                            kH, kW, 6, worst, viol, status))
    all_rows += e1_rows

    # ---- C1: guard-hoisted vs naive guarded baseline ----
    c1_rows = run_check(
        "C1", e1_cfg, _KINDS6,
        lambda x, *a: baseline(x, *a, hoist=True),
    )
    all_rows += c1_rows

    # ---- C4: whole-plane reduction on C4-valid configs (all d == 1) ----
    c4_cfg = [
        ((1, 8, 4, 4), 4, 4, 1, 1, 0, 0, 1, 1, False),      # out=1x1, k==in
        ((2, 4, 7, 9), 7, 9, 1, 1, 0, 0, 1, 1, False),      # non-square, k==in
        ((1, 8, 16, 16), 16, 16, 1, 1, 0, 0, 1, 1, False),  # k==in
        ((2, 16, 7, 7), 7, 7, 1, 1, 0, 0, 1, 1, False),     # k==in, C=16
        ((1, 8, 8, 8), 8, 8, 1, 1, 0, 0, 1, 1, False),      # k==in (replaces the
    ]                                                        # old empty out==0 case)
    c4_rows = run_check(
        "C4", c4_cfg, _KINDS6,
        lambda x, *a: c4_whole_plane(x),
        require=lambda cfg: c4_condition(
            cfg[0][2], cfg[0][3], cfg[1], cfg[2], cfg[3], cfg[4],
            cfg[5], cfg[6], cfg[7], cfg[8], cfg[9]),
    )
    all_rows += c4_rows

    # ---- C4x: dilation counterexamples.  out==1x1 but d>1, so the window
    #      covers only {k*d} per axis.  The corrected condition must reject
    #      them; the general path must match F.max_pool2d. ----
    c4x_cfg = [
        ((1, 8, 5, 5), 2, 2, 1, 1, 0, 0, 4, 4, False),      # window = {0,4}^2
        ((1, 8, 7, 7), 3, 3, 1, 1, 0, 0, 3, 3, False),      # window = {0,3,6}^2
    ]
    c4x_rows = []
    for cfg in c4x_cfg:
        assert not c4_condition(cfg[0][2], cfg[0][3], cfg[1], cfg[2], cfg[3],
                                cfg[4], cfg[5], cfg[6], cfg[7], cfg[8], cfg[9]), (
            "C4x config must NOT satisfy the C4 condition: " + str(cfg)
        )
        shape, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode = cfg
        for dtype in _DTYPES:
            worst = 0.0
            viol = 0
            for kind in _KINDS6:
                x64 = make_data(kind, shape)
                cand = baseline(x64, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode,
                                hoist=True)
                ref = F.max_pool2d(
                    x64, (kH, kW), (sH, sW), (pH, pW), (dH, dW), ceil_mode
                )
                nb, wd = mismatch(cand, ref, dtype)
                viol += nb
                worst = max(worst, wd)
            status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
            c4x_rows.append(("C4x", str(dtype).replace("torch.", ""), str(shape),
                             kH, kW, 6, worst, viol, status))
    all_rows += c4x_rows

    # ---- C4b: identity copy on C4b-valid configs ----
    c4b_cfg = [
        ((1, 8, 4, 4), 1, 1, 1, 1, 0, 0, 1, 1, False),
        ((2, 4, 9, 11), 1, 1, 1, 1, 0, 0, 1, 1, False),
        ((1, 16, 5, 7), 1, 1, 1, 1, 0, 0, 1, 1, False),
        ((3, 33, 6, 8), 1, 1, 1, 1, 0, 0, 1, 1, False),
    ]
    c4b_rows = run_check(
        "C4b", c4b_cfg, _KINDS6,
        lambda x, *a: x.clone(),
    )
    all_rows += c4b_rows

    hdr()
    emit(all_rows)

    # ---- Hazard evidence: the pre-fix (missing-dilation) condition would have
    #      routed this d>1 config to the whole-plane path and produced 1.0 where
    #      the correct answer is 0.0.  The corrected guard rejects it. ----
    adv = torch.zeros((1, 1, 5, 5), dtype=torch.float64)
    adv[0, 0, 2, 2] = 1.0
    gen = float(baseline(adv, 2, 2, 1, 1, 0, 0, 4, 4, False).item())
    ref = float(F.max_pool2d(adv, (2, 2), (1, 1), (0, 0), (4, 4), False).item())
    wrong = float(c4_whole_plane(adv).item())
    assert c4_condition(5, 5, 2, 2, 1, 1, 0, 0, 4, 4, False) is False
    assert gen == ref == 0.0 and wrong == 1.0, (gen, ref, wrong)
    print()
    print("[C4x-GUARD] dilation counterexample (H=W=5,k=2,s=1,p=0,d=4,x[2,2]=1):")
    print(f"[C4x-GUARD]   general path = {gen}  == golden {ref};  "
          f"whole-plane = {wrong} (excluded by the d==1 guard)")
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
