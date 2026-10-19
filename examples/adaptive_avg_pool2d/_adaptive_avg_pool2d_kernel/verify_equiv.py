#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Design-time equivalence verification (D-1) for
``_adaptive_avg_pool2d_kernel``.

Verifies the mathematical-equivalence optimizations adopted in DESIGN.md
section 1.6.1 against the *baseline* formula (the general nested adaptive-bin
mean, computed here in float64 as the reference of record):

  C4  ``out=1x1``  : the general bin mean degenerates to the whole-plane mean
                     -> computed as a single flattened ``sum / (H*W)``.
  C4b identity     : when ``out_H == H_in`` and ``out_W == W_in`` every bin has
                     size 1 -> ``out == x`` (pure copy).
  C1  reciprocal   : ``sum / d``  ->  ``sum * (1/d)`` with the reciprocal
                     evaluated in fp32 (``inv = float32(1/d)``), i.e. the exact
                     form the kernel emits (compile-time scalar ``inv`` + one
                     ``T.vmul``).

C4 / C4b are exactly equivalent (not merely tolerance-equivalent) to the
baseline: the bin partition of output index ``o`` is
``[floor(o*in/out), ceil((o+1)*in/out))`` so for ``out == 1`` the single bin is
``[0, in)`` and for ``out == in`` every bin is a unit bin.  The script therefore
also acts as an index-arithmetic guard (off-by-one in the bin bounds would show
up as a mismatch on the non-divisible corner cases).

C1 is *tolerance*-equivalent: replacing the per-output division by a
compile-time fp32 reciprocal followed by an fp32 multiply perturbs the result by
at most 1 ulp of the target dtype (the fp32 reciprocal error is ~6e-8 relative,
far below fp16/bf16 resolution).  Its pass criterion is therefore "after
rounding to the target dtype, every differing element is within 1 ulp of the
baseline at that element's own magnitude" — recorded explicitly per case; a
bit-identical result (viol==0) is the common case.

The script is deliberately self-contained (torch CPU only, no NPU / no tilelang
import) so Stage 2 can re-run it anywhere.

Run:  python verify_equiv.py
Exit code 0 == all EQUIV_PASS, 1 == at least one EQUIV_FAIL.
"""

import math
import sys

import torch

# Tolerances: the optimized formulas must agree with the baseline *after*
# rounding both to the target dtype.
_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ATOL = {"float16": 1e-3, "bfloat16": 1.6e-2}
RTOL = {"float16": 1e-3, "bfloat16": 1.6e-2}

# C4 / C4b claim *exact* (bit-identical after rounding) equivalence;
# C1 claims *tolerance* equivalence bounded by 1 ulp of the target dtype.
_ULP_TOL = {"C4": 0.0, "C4b": 0.0, "C1": 1.0}


def adaptive_bin(o: int, size_in: int, size_out: int):
    """PyTorch adaptive-pool half-open bin ``[start, end)`` (source semantics)."""
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


# ---------------------------------------------------------------------------
# Baseline (source formula): general nested adaptive-bin mean, fp64
# ---------------------------------------------------------------------------
def baseline_general(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    n, c, h_in, w_in = x.shape
    x64 = x.double()
    out = torch.empty((n, c, out_h, out_w), dtype=torch.float64)
    for oh in range(out_h):
        ih0, ih1 = adaptive_bin(oh, h_in, out_h)
        for ow in range(out_w):
            iw0, iw1 = adaptive_bin(ow, w_in, out_w)
            blk = x64[:, :, ih0:ih1, iw0:iw1]
            out[:, :, oh, ow] = blk.mean(dim=(2, 3))
    return out


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def cand_flat_global(x: torch.Tensor) -> torch.Tensor:
    """C4: out = 1x1 -> single whole-plane mean (flattened reduce)."""
    n, c, h_in, w_in = x.shape
    return x.double().reshape(n, c, h_in * w_in).mean(dim=2, keepdim=False).reshape(n, c, 1, 1)


def cand_identity(x: torch.Tensor) -> torch.Tensor:
    """C4b: out == in -> unit bins -> out == x."""
    return x.double()


def cand_reciprocal_general(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """C1: general grid with ``sum * (1/d)`` (fp32 reciprocal), no division."""
    n, c, h_in, w_in = x.shape
    x64 = x.double()
    out = torch.empty((n, c, out_h, out_w), dtype=torch.float64)
    for oh in range(out_h):
        ih0, ih1 = adaptive_bin(oh, h_in, out_h)
        for ow in range(out_w):
            iw0, iw1 = adaptive_bin(ow, w_in, out_w)
            s = x64[:, :, ih0:ih1, iw0:iw1].sum(dim=(2, 3))
            d = (ih1 - ih0) * (iw1 - iw0)
            # exact fp32 reciprocal, matching the kernel's compile-time scalar
            inv = float(torch.tensor(1.0 / d, dtype=torch.float32))
            out[:, :, oh, ow] = s * inv
    return out


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------
def _random(shape, dtype, generator=None, scale=1.0):
    t = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return t.to(_DTYPE[dtype])


def _cases():
    """(name, x, out_h, out_w, kind) tuples.

    Includes the manifest / NPU-test shapes, plus the special-value corners
    (±inf / NaN / subnormal) required by the D-1 protocol.
    """
    cases = []
    dtype_shapes = [
        ("float16", (8, 2048, 7, 7)),   # resnet-global manifest workload
        ("float16", (2, 128, 56, 56)),  # spp-6x6 manifest workload
        ("bfloat16", (2, 64, 55, 57)),  # nondiv-7x7 manifest workload
        ("float16", (1, 8, 4, 4)),      # smoke manifest workload
        ("float16", (1, 8, 9, 11)),     # NPU test: all-None identity
        ("float16", (1, 16, 11, 13)),   # NPU test: partial-None
        ("bfloat16", (1, 16, 10, 10)),  # NPU test: scalar output_size 4
        ("float16", (3, 5, 7, 3)),      # odd tiny shapes
        ("bfloat16", (1, 1, 1, 1)),     # degenerate 1x1
    ]
    for dtype, shape in dtype_shapes:
        n, c, h_in, w_in = shape
        x = _random(shape, dtype)
        # global (out = 1x1) -> C4
        cases.append((f"C4/global/{dtype}/{shape}", x, 1, 1, "C4"))
        # identity (out == in) -> C4b
        cases.append((f"C4b/identity/{dtype}/{shape}", x, h_in, w_in, "C4b"))

    # C1 (reciprocal) lives on the *general* (non-degenerate) grid.
    c1_shapes = [
        ("float16", (2, 128, 56, 56), (6, 6)),  # spp: non-divisible W-bin
        ("bfloat16", (2, 64, 55, 57), (7, 7)),  # nondiv: both axes
        ("float16", (1, 8, 4, 4), (2, 2)),      # smoke
        ("float16", (3, 5, 7, 3), (2, 2)),      # odd tiny
        ("bfloat16", (1, 16, 10, 10), (4, 4)),  # scalar-4 output grid
    ]
    for dtype, shape, out in c1_shapes:
        x = _random(shape, dtype)
        cases.append((f"C1/reciprocal/{dtype}/{shape}/out{out}", x, out[0], out[1], "C1"))
    return cases


def _special_cases():
    """Special-value corners on shapes that exercise the candidates."""
    cases = []
    for dtype in ("float16", "bfloat16"):
        n, c, h, w = 2, 3, 5, 6
        base = _random((n, c, h, w), dtype)
        # subnormal-scale
        x_sub = (torch.randn((n, c, h, w), dtype=torch.float32) * 1e-5).to(_DTYPE[dtype])
        cases.append((f"C4/special/subnormal/{dtype}", x_sub, 1, 1, "C4"))
        cases.append((f"C4b/special/subnormal/{dtype}", x_sub, h, w, "C4b"))
        cases.append((f"C1/special/subnormal/{dtype}", x_sub, 3, 3, "C1"))
        # large magnitude
        x_large = (base.float() * 100.0).to(_DTYPE[dtype])
        cases.append((f"C4/special/large/{dtype}", x_large, 1, 1, "C4"))
        cases.append((f"C1/special/large/{dtype}", x_large, 3, 3, "C1"))
        # ±inf / NaN (mean propagates NaN/inf identically for both formulas)
        x_inf = base.clone()
        x_inf.view(-1)[0] = float("inf")
        x_inf.view(-1)[1] = float("-inf")
        x_inf.view(-1)[2] = float("nan")
        cases.append((f"C4/special/inf-nan/{dtype}", x_inf, 1, 1, "C4"))
        cases.append((f"C4b/special/inf-nan/{dtype}", x_inf, h, w, "C4b"))
        cases.append((f"C1/special/inf-nan/{dtype}", x_inf, 3, 3, "C1"))
        # zeros
        x0 = torch.zeros((n, c, h, w), dtype=_DTYPE[dtype])
        cases.append((f"C4/special/zeros/{dtype}", x0, 1, 1, "C4"))
        cases.append((f"C4b/special/zeros/{dtype}", x0, h, w, "C4b"))
        cases.append((f"C1/special/zeros/{dtype}", x0, 3, 3, "C1"))
    return cases


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _ulp_local(dtype: torch.dtype, ref: torch.Tensor) -> torch.Tensor:
    """Per-element one-step size of ``dtype`` at each ``ref`` magnitude (fp64)."""
    if dtype == torch.float16:
        eps = 2.0 ** -10
    else:  # bfloat16
        eps = 2.0 ** -8
    mag = ref.abs()
    tiny = float(torch.finfo(dtype).tiny)
    # exponent bucket of each element
    exp = torch.floor(torch.log2(torch.clamp(mag, min=tiny)))
    return eps * torch.pow(2.0, exp)


def _compare(cand64: torch.Tensor, base64: torch.Tensor, dtype: torch.dtype):
    """Round both fp64 results to ``dtype`` and compare.

    Returns (max_abs_diff_in_fp64, ulp_violations, max_ulp_diff, nan_mismatch).
    ``ulp_violations`` counts elements that differ by more than the kind's
    admissible ulp tolerance (handled by the caller via ``_ULP_TOL``).
    """
    ref = base64.to(dtype)          # baseline rounded to target dtype
    got = cand64.to(dtype)          # candidate rounded to target dtype
    ref_f = ref.double()
    got_f = got.double()

    nan_mask = torch.isnan(ref_f) | torch.isnan(got_f)
    both_nan = torch.isnan(ref_f) & torch.isnan(got_f)
    diff = (got_f - ref_f).abs()
    diff = torch.where(both_nan, torch.zeros_like(diff), diff)

    ulp = _ulp_local(dtype, ref_f)
    # never flag the both-NaN positions
    ratio = torch.where(both_nan, torch.zeros_like(diff), diff / ulp)
    max_ulp = float(ratio.max().item()) if ratio.numel() else 0.0
    max_diff = float(diff.max().item()) if diff.numel() else 0.0

    # NaN/finite mismatches are a hard failure regardless of kind
    nan_mismatch = int((nan_mask & ~both_nan).sum().item())
    return max_diff, ratio, max_ulp, nan_mismatch, ref_f, got_f


def main() -> int:
    torch.manual_seed(0)

    all_cases = _cases() + _special_cases()

    rows = []
    failures = []
    for name, x, out_h, out_w, kind in all_cases:
        base = baseline_general(x, out_h, out_w)
        if kind == "C4":
            cand = cand_flat_global(x)
        elif kind == "C4b":
            cand = cand_identity(x)
        else:  # C1
            cand = cand_reciprocal_general(x, out_h, out_w)
        max_diff, ratio, max_ulp, nan_mismatch, ref_f, got_f = _compare(cand, base, x.dtype)

        # C4/C4b: bit-identical after rounding (tol = 0 ulp).
        # C1: within 1 ulp of the target dtype (tolerance-equivalent).
        tol = _ULP_TOL[kind]
        excess = int((ratio > tol + 1e-6).sum().item())
        viol = excess + nan_mismatch  # admissible-tolerance violations
        bit_viol = int((got_f != ref_f).sum().item())  # informational
        status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
        if status == "EQUIV_FAIL":
            failures.append(name)
        rows.append(
            (name, kind, str(x.dtype).replace("torch.", ""), max_diff, viol, bit_viol, max_ulp, status)
        )

    w = max(len(r[0]) for r in rows)
    print(f"{'case'.ljust(w)} | kind | dtype    | max_abs_diff | viol | bitdiff | max_ulp | status")
    print("-" * (w + 70))
    for name, kind, dt, md, viol, bitdiff, ulp, status in rows:
        print(
            f"{name.ljust(w)} | {kind:<4} | {dt:<8} | {md:.3e}  | {viol:>3}  | "
            f"{bitdiff:>6}  | {ulp:>6.2f} | {status}"
        )
    print()
    print(
        "criterion: C4/C4b exact (0 ulp after rounding); C1 tolerance "
        "<= 1 ulp of target dtype; 'viol' counts admissible-tolerance "
        "violations (hard NaN/finite mismatches always fail); 'bitdiff' = "
        "bit-level differences (informational)."
    )
    print()
    if failures:
        print(f"[EQUIV] {len(failures)} FAILING case(s): " + ", ".join(failures))
        return 1
    print(f"[EQUIV] ALL PASS — {len(rows)} cases, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
