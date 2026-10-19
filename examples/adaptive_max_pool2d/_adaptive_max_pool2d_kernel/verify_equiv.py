#!/usr/bin/env python
"""Equivalence machine-verification (D-1) for the adaptive_max_pool2d migration.

Verifies the math-equivalent optimisations adopted in DESIGN.md section 1.6.1
against the *baseline* adaptive-bin max formula (section 0.1 semantics) on a
**fp64 reference** (torch CPU).  Adopted items:

  * C4  -- global-grid specialisation: when ``out_H == out_W == 1`` the single
           bin ``bin(0,0) == [0,H_in) x [0,W_in)`` is the whole plane, so
           ``out = max(plane)`` with the source NaN-wins rule.  Expected
           **bit-exact** vs the baseline (max selects an existing element).
  * C4b -- identity-grid specialisation: when ``out_H == H_in and
           out_W == W_in`` every bin is a single element, so ``out = x``.
           Expected **bit-exact**.

Both candidates are compared element-wise, after rounding to the target dtype,
against the baseline generic bin loop.  Criteria follow DESIGN.md 1.6.1:
per-element mismatch == 0 (exact) for both items; NaN-vs-NaN counts as equal,
NaN-vs-finite always fails.

Run:  /workspace/venv/bin/python verify_equiv.py
"""

import torch

# ---------------------------------------------------------------------------
# Baseline: generic adaptive-bin max (DESIGN.md section 0.1 semantics).
# fp64 throughout; NaN anywhere in a window wins (verified against
# torch.nn.functional.adaptive_max_pool2d).
# ---------------------------------------------------------------------------
def _adaptive_bin(o: int, size_in: int, size_out: int):
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def baseline_bin_max(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """fp64 generic bin max of an NCHW tensor (no specialisation)."""
    n, c, h_in, w_in = x.shape
    x = x.to(torch.float64)
    out = torch.empty((n, c, out_h, out_w), dtype=torch.float64)
    for oh in range(out_h):
        ih0, ih1 = _adaptive_bin(oh, h_in, out_h)
        for ow in range(out_w):
            iw0, iw1 = _adaptive_bin(ow, w_in, out_w)
            win = x[:, :, ih0:ih1, iw0:iw1]
            m = win.amax(dim=(2, 3))
            nan = win.isnan().any(dim=(2, 3))
            out[:, :, oh, ow] = torch.where(nan, torch.tensor(float("nan")), m)
    return out


def candidate_c4(x: torch.Tensor) -> torch.Tensor:
    """Global-grid specialisation: whole-plane max with NaN-wins."""
    x = x.to(torch.float64)
    m = x.amax(dim=(2, 3), keepdim=True)
    nan = x.isnan().any(dim=(2, 3), keepdim=True)
    return torch.where(nan, torch.tensor(float("nan")), m)


def candidate_c4b(x: torch.Tensor) -> torch.Tensor:
    """Identity-grid specialisation: out == x."""
    return x.to(torch.float64)


# ---------------------------------------------------------------------------
# Comparison (NaN aware)
# ---------------------------------------------------------------------------
def compare(out, ref, dtype):
    o = out.to(dtype)
    r = ref.to(dtype)
    o_nan, r_nan = torch.isnan(o), torch.isnan(r)
    eq = (o == r) | (o_nan & r_nan)
    finite_mismatch = int((o_nan ^ r_nan).sum().item())
    mismatch = int((~eq).sum().item())
    # max |diff| over finite-vs-finite positions only
    finite = ~o_nan & ~r_nan
    mad = 0.0
    if finite.any():
        mad = float((o[finite].float() - r[finite].float()).abs().max().item())
    return {"max_abs_diff": mad, "finite_mismatch": finite_mismatch, "mismatch": mismatch}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def _special(shape, kind):
    n, c, h, w = shape
    x = torch.randn(n, c, h, w, dtype=torch.float32)
    if kind == "nan":
        x[0, 0, 0, 0] = float("nan")
    elif kind == "inf":
        x[0, 0, 0, 0] = float("inf")
        x[0, min(1, c - 1), h - 1, w - 1] = float("-inf")
    elif kind == "subnormal":
        x = x * 1e-6
    elif kind == "large":
        x = x * 100.0
    elif kind == "zeros":
        x = torch.zeros_like(x)
    return x


# Global-grid shapes (out == 1x1) -- bin(0,0) spans the whole plane.
GLOBAL_CASES = [
    (1, 8, 4, 4),
    (8, 2048, 7, 7),
    (2, 128, 56, 56),
    (2, 64, 55, 57),
    (1, 16, 10, 10),
    (1, 1, 1, 1),
]
# Identity-grid shapes (out == in) -- every bin is a single element.
IDENTITY_CASES = [
    (1, 8, 4, 4),
    (1, 8, 9, 11),
    (1, 16, 5, 7),
    (2, 4, 9, 11),
    (1, 1, 1, 1),
]
KINDS = ("rand", "nan", "inf", "subnormal", "large", "zeros")


def main():
    rows = []
    fails = 0
    for dtype in (torch.float16, torch.bfloat16):
        dt = str(dtype).split(".")[-1]
        for (n, c, h, w) in GLOBAL_CASES:
            for kind in KINDS:
                x = torch.randn(n, c, h, w) if kind == "rand" else _special((n, c, h, w), kind)
                ref = baseline_bin_max(x, 1, 1)
                r = compare(candidate_c4(x), ref, dtype)
                ok = r["mismatch"] == 0 and r["finite_mismatch"] == 0
                fails += 0 if ok else 1
                rows.append((f"C4/global/{kind}/{dt}/({n},{c},{h},{w})", "C4", dt,
                             r["max_abs_diff"], r["mismatch"], "EQUIV_PASS" if ok else "EQUIV_FAIL"))
        for (n, c, h, w) in IDENTITY_CASES:
            for kind in KINDS:
                x = torch.randn(n, c, h, w) if kind == "rand" else _special((n, c, h, w), kind)
                ref = baseline_bin_max(x, h, w)
                r = compare(candidate_c4b(x), ref, dtype)
                ok = r["mismatch"] == 0 and r["finite_mismatch"] == 0
                fails += 0 if ok else 1
                rows.append((f"C4b/identity/{kind}/{dt}/({n},{c},{h},{w})", "C4b", dt,
                             r["max_abs_diff"], r["mismatch"], "EQUIV_PASS" if ok else "EQUIV_FAIL"))

    hdr = f"{'case':46s} | {'kind':4s} | {'dtype':8s} | {'max_abs_diff':>12s} | {'viol':>4s} | status"
    print(hdr)
    print("-" * len(hdr))
    for case, kind, dtt, mad, mism, st in rows:
        print(f"{case:46s} | {kind:4s} | {dtt:8s} | {mad:12.3e} | {mism:4d} | {st}")
    print()
    print(f"criterion: C4/C4b exact (0 mismatch after rounding to target dtype); "
          f"'viol' = element mismatches (NaN-vs-finite always fails).")
    print()
    if fails:
        print(f"[EQUIV] {fails} FAILING case(s)")
    else:
        print(f"[EQUIV] ALL PASS — {len(rows)} cases, 0 failures")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
