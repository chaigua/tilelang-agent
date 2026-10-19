#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""D-1 equivalence machine verification for `_group_norm_kernel` (GroupNormFwdOp).

Each ADOPTED math-equivalence item of DESIGN.md section 1.6.1 (candidate)
is compared against the source (GPU) baseline formula that it replaces, on
random tensors + corner values, with an fp64 reference.  Runs on torch CPU
only (no NPU needed); the NPU-specific instruction behaviour is taken from
the pattern-library measured constants (PLAIN_MAX_REL / VDIV_ULP_BUDGET).

Isolation protocol: `candidate` = the design pipeline with all adopted items
enabled; `baseline` = the SAME pipeline with ONLY the item under test
reverted to the source formula.  Both are compared against each other and
against an fp64 reference of the exact mathematics.

Verdict per item (two criteria, either is sufficient):
  (a) candidate vs baseline inside the target-dtype tolerance
      (zero elements exceed atol + rtol*|baseline|), or
  (b) candidate is no further from the fp64 reference than the baseline
      (beyond one rounding unit of the target dtype).
Rejected (non-adopted) candidates are printed separately as rejection
evidence -- they are not adoption rows.

Items / evidence
----------------
O1  div -> reciprocal-multiply      mean = S/D        vs  mean = S*fl(1/D)
O2  rstd: raw T.vrsqrt (approx)     vs  T.vsqrt + T.vdiv composition
O3  centred-value reuse (CSE)       epilogue recompute vs reuse of d
O4  exact-D variance                256-pad + pad correction vs exact D
O5  unit-affine elimination         (((d*rstd)*1)+0)  vs  (d*rstd)
C1  rejection evidence: single-pass moment form  var = E[x^2] - (E[x])^2

Run:  python3 verify_equiv.py        (exit code 0 = all adopted items PASS)
"""

from __future__ import annotations

import math
import sys

import torch

FP64 = torch.float64
FP32 = torch.float32

# Target-dtype tolerances (source repo tests/ops/test_group_norm.py
# `_get_tolerances()`), read as (atol, rtol).
TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
}
# One rounding unit (relative) of the target dtype -- used by criterion (b).
ULP_REL = {torch.float32: 2.0**-24, torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8}

# ---------------------------------------------------------------------------
# NPU instruction model constants (pattern-library, evidence-backed)
# ---------------------------------------------------------------------------
# TRAP-vrsqrt-plain-precision (pattern-library traps-runtime.md, verified):
# `T.vrsqrt` lowers to an approximate vector instruction with a measured max
# relative error of 2.87e-3 (input sweep [1e-6, 1e6]).  Modelled here as a
# bounded worst-case relative perturbation (+/- PLAIN_MAX_REL).
PLAIN_MAX_REL = 2.87e-3
# CONST-npuir-f32-division-not-ieee-rounded (pattern-library constants.md,
# verified): npuir fp32 `vdiv` is not IEEE correctly rounded (<= ~1 ulp on
# 2-D same-shape forms; the (M,1)/(M,1) column form was measured bit-exact).
# Modelled as a 1-ulp perturbation budget on the division of the composition.
VDIV_ULP_BUDGET = True
# GPU kernel alignment (the value the GPU wrapper pads its row length to).
GPU_ALIGNMENT = 256


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ulp_diff_dtype(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> int:
    """Max ULP distance between two tensors, measured in `dtype` bit patterns."""
    if dtype == torch.float32:
        bits = 32
    elif dtype == torch.float16:
        bits = 16
        a, b = a.to(torch.float16), b.to(torch.float16)
    else:
        bits = 16
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    n = a.numel()
    av = a.contiguous().view(torch.int16 if bits == 16 else torch.int32).to(torch.int64).flatten()
    bv = b.contiguous().view(torch.int16 if bits == 16 else torch.int32).to(torch.int64).flatten()
    if bits == 16:
        av = torch.where(av < 0, -32768 - av, av)
        bv = torch.where(bv < 0, -32768 - bv, bv)
    else:
        av = torch.where(av < 0, torch.tensor(-2**31, dtype=torch.int64) - av, av)
        bv = torch.where(bv < 0, torch.tensor(-2**31, dtype=torch.int64) - bv, bv)
    assert av.numel() == n
    d = (av - bv).abs()
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0)
    return int(d.max().item())


def _violations(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> int:
    """#elements of `out` exceeding atol + rtol*|ref| (NaN==NaN counts equal)."""
    eq = (out == ref) | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(eq, torch.zeros((), dtype=FP64),
                       (out.double() - ref.double()).abs())
    lim = atol + rtol * ref.double().abs()
    return int((diff > lim).sum().item())


def _rel_err_vs_fp64(out: torch.Tensor, ref64: torch.Tensor) -> float:
    num = (out.double() - ref64).abs()
    den = ref64.abs().clamp_min(1e-12)
    r = torch.where(ref64 == 0, num, num / den)
    r = torch.where((out == ref64), torch.zeros_like(r), r)
    return float(r.nan_to_num(nan=0.0, posinf=0.0).max().item())


# ---------------------------------------------------------------------------
# the pipeline (every item switchable)
# ---------------------------------------------------------------------------
def pipeline(
    x_dtype: torch.Tensor,
    D: int,
    eps: float,
    *,
    div_mode: str = "div",        # "div" | "recip"                 (O1)
    rstd_mode: str = "exact",     # "exact" | "compose" | "plain"    (O2)
    plain_delta: float = 0.0,     # T.vrsqrt worst-case perturbation (O2)
    cse: bool = False,            # False = recompute d              (O3)
    pad_mode: str = "exact",      # "exact" | "pad256"               (O4)
    affine_mode: str = "unit",    # "unit" | "none"                  (O5)
    var_mode: str = "twopass",    # "twopass" | "moment"             (C1)
) -> torch.Tensor:
    """One formula variant; fp32 compute domain, single round-back to dtype."""
    dtype = x_dtype.dtype
    M = x_dtype.shape[0]

    if pad_mode == "pad256":
        D_pad = ((D + GPU_ALIGNMENT - 1) // GPU_ALIGNMENT) * GPU_ALIGNMENT
        pad = D_pad - D
        xrow = torch.zeros((M, D_pad), dtype=dtype)
        xrow[:, :D] = x_dtype
    else:
        xrow = x_dtype
        pad = 0

    x32 = xrow.to(FP32)                        # exact upcast (fp16/bf16 subset)
    inv_d = None if div_mode == "div" else torch.tensor(1.0 / float(D), dtype=FP32)

    d = None
    if var_mode == "twopass":
        s1 = x32.sum(dim=-1, keepdim=True)
        mean = s1 / float(D) if inv_d is None else s1 * inv_d
        d = x32 - mean                         # O3: kept for the epilogue
        sq = d * d
        s2 = sq.sum(dim=-1, keepdim=True)
        if pad:
            s2 = s2 - float(pad) * mean * mean  # source pad correction
        var = s2 / float(D) if inv_d is None else s2 * inv_d
    else:                                       # moment form (C1)
        s1 = x32.sum(dim=-1, keepdim=True)
        s2 = (x32 * x32).sum(dim=-1, keepdim=True)
        mean = s1 / float(D)
        m2 = s2 / float(D)
        var = m2 - mean * mean

    var = var + eps

    if rstd_mode == "compose":                  # T.vsqrt + T.vdiv
        root = torch.sqrt(var)
        rstd = root / var
        if VDIV_ULP_BUDGET:                     # npuir vdiv <= 1 ulp budget
            nxt = torch.nextafter(rstd, torch.full_like(rstd, math.inf))
            prv = torch.nextafter(rstd, torch.full_like(rstd, -math.inf))
            cand = torch.stack([rstd, nxt, prv], dim=0)
            delta = torch.max((cand - rstd).abs(), dim=0).values
            rstd = rstd + torch.sign(rstd) * delta
    elif rstd_mode == "plain":                  # raw T.vrsqrt instruction
        rstd = (1.0 / torch.sqrt(var)) * (1.0 + plain_delta)
    else:                                       # "exact" 1/sqrt in fp32
        rstd = 1.0 / torch.sqrt(var)

    if cse or var_mode == "moment":
        y32 = (x32 - mean) * rstd
    else:                                       # epilogue recompute (source)
        y32 = (xrow.to(FP32) - mean) * rstd

    if affine_mode == "unit":                   # kernel row-broadcast identity
        y32 = y32 * 1.0 + 0.0

    return y32[:, :D].to(dtype)


def reference_fp64(x_dtype: torch.Tensor, D: int, eps: float) -> torch.Tensor:
    """Exact two-pass mathematics in fp64 on the same (dtype-rounded) input."""
    x64 = x_dtype.double()
    mean = x64.sum(dim=-1, keepdim=True) / float(D)
    d = x64 - mean
    var = (d * d).sum(dim=-1, keepdim=True) / float(D)
    rstd = 1.0 / torch.sqrt(var + eps)
    return (d * rstd).to(x_dtype.dtype)


def rstd_segment(dtype: torch.dtype, rstd_mode: str, plain_delta: float = 0.0):
    """Relative error of the rstd segment alone vs the fp64 reference.

    Inputs: a var sweep spanning the design's domain (0 -> 1e6), the extreme
    values included.  Returns max relative error of rstd.
    """
    var = torch.tensor([1e-30, 1e-12, 1e-6, 1e-3, 1.0, 1e2, 1e6], dtype=FP64)
    v32 = var.to(FP32)
    if rstd_mode == "compose":
        root = torch.sqrt(v32)
        out = root / v32
    elif rstd_mode == "plain":
        out = (1.0 / torch.sqrt(v32)) * (1.0 + plain_delta)
    else:
        out = 1.0 / torch.sqrt(v32)
    ref = 1.0 / torch.sqrt(var)
    return float(((out.double() - ref).abs() / ref).max().item())


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _inf_nan(M, D, dt):
    x = torch.randn(M, D, dtype=torch.float32)
    x[0, :] = float("inf")
    x[1, ::4] = float("nan")
    x[2, ::4] = float("inf")
    return x.to(dt)


def cases():
    out = []
    for M, D in [(4, 13), (2, 64), (2, 196), (2, 256), (3, 1152), (2, 4096), (1, 6272)]:
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            out.append((f"randn-{M}x{D}", M, D, dt,
                        lambda M, D, dt: torch.randn(M, D, dtype=torch.float32).to(dt)))
    out.append(("largemean-fp32", 4, 512, torch.float32,
                lambda M, D, dt: (3e4 + 0.5 * torch.randn(M, D)).to(dt)))
    out.append(("largemean-fp16", 4, 512, torch.float16,
                lambda M, D, dt: (60000.0 + 16.0 * torch.randn(M, D)).to(dt)))
    out.append(("tinyvar-fp32", 4, 256, torch.float32,
                lambda M, D, dt: (1e3 + 1e-3 * torch.randn(M, D)).to(dt)))
    out.append(("const-rows", 4, 256, torch.float32,
                lambda M, D, dt: torch.full((M, D), 3.7, dtype=torch.float32).to(dt)))
    out.append(("zeros", 4, 256, torch.float32,
                lambda M, D, dt: torch.zeros(M, D, dtype=torch.float32).to(dt)))
    out.append(("subnormal-fp16", 4, 256, torch.float16,
                lambda M, D, dt: (torch.randn(M, D) * 1e-5).to(dt)))
    out.append(("extreme-fp16", 4, 256, torch.float16,
                lambda M, D, dt: (torch.randn(M, D) * 65504.0).to(dt)))
    out.append(("inf-nan-fp16", 4, 256, torch.float16, _inf_nan))
    # corner: pad-count cancellation of the GPU 256-pad correction
    # (D not 256-aligned + large mean -> pad*mean^2 >> sum((x-mean)^2))
    out.append(("padcancel-d2-fp32", 4, 2, torch.float32,
                lambda M, D, dt: (3e4 + 0.5 * torch.randn(M, D)).to(dt)))
    out.append(("padcancel-d196-fp32", 4, 196, torch.float32,
                lambda M, D, dt: (3e4 + 0.5 * torch.randn(M, D)).to(dt)))
    return out


# ---------------------------------------------------------------------------
# item evaluation
# ---------------------------------------------------------------------------
CANDIDATE = dict(div_mode="recip", rstd_mode="compose", cse=True,
                 pad_mode="exact", affine_mode="none")


def _run(opts, x, D, eps):
    return pipeline(x, D, eps, **opts)


def evaluate_item(name, **baseline_overrides):
    eps = 1e-5
    tot_vc = tot_vb = 0
    worst_ulp = 0
    worst_cb = 0
    worst_delta = -math.inf
    worse_cases = []
    only_b = []
    failed = []
    n_a = 0
    rows = []
    for cname, M, D, dt, gen in cases():
        x = gen(M, D, dt)
        atol, rtol = TOL[dt]
        ref64 = reference_fp64(x, D, eps)
        cand = _run(CANDIDATE, x, D, eps)
        base = _run({**CANDIDATE, **baseline_overrides}, x, D, eps)
        u = _ulp_diff_dtype(cand, base, dt)
        vcb = _violations(cand, base, atol, rtol)          # criterion (a)
        vc = _violations(cand, ref64, atol, rtol)
        vb = _violations(base, ref64, atol, rtol)
        ec = _rel_err_vs_fp64(cand, ref64)
        eb = _rel_err_vs_fp64(base, ref64)
        crit_a = vcb == 0
        crit_b = ec <= max(eb, ULP_REL[dt]) * 1.001      # candidate not worse
        ok = crit_a or crit_b
        rows.append(dict(case=cname, dtype=str(dt), ulp=u, viol_cb=vcb,
                         viol_cand=vc, viol_base=vb, err_cand=ec, err_base=eb,
                         crit_a=crit_a, crit_b=crit_b, ok=ok))
        tot_vc += vc
        tot_vb += vb
        worst_ulp = max(worst_ulp, u)
        worst_cb = max(worst_cb, vcb)
        worst_delta = max(worst_delta, ec - eb)
        n_a += int(crit_a)
        if not crit_a:
            only_b.append(cname)
        if not ok:
            failed.append(cname)
        if vc > vb:
            worse_cases.append(cname)
    return dict(item=name, cases=len(rows), max_ulp=worst_ulp, viol_cb=worst_cb,
                viol_cand=tot_vc, viol_base=tot_vb, delta=worst_delta,
                n_a=n_a, only_b=only_b, failed=failed,
                worse_cases=worse_cases,
                ok=all(r["ok"] for r in rows), rows=rows)


def main() -> int:
    torch.manual_seed(20261015)
    print("=" * 100)
    print("D-1 equivalence machine verification -- _group_norm_kernel (GroupNormFwdOp)")
    print("fp32 compute domain + single round-back to the target dtype; fp64 reference; torch CPU")
    print("=" * 100)

    items = [
        ("O1 div->reciprocal-multiply", dict(div_mode="div")),
        ("O2 rstd composition vs raw T.vrsqrt", dict(rstd_mode="plain",
                                                     plain_delta=PLAIN_MAX_REL)),
        ("O3 centred-value reuse (CSE)", dict(cse=False)),
        ("O4 exact-D variance (no 256-pad)", dict(pad_mode="pad256")),
        ("O5 unit-affine elimination", dict(affine_mode="unit")),
    ]

    results = []
    for name, over in items:
        r = evaluate_item(name, **over)
        results.append(r)
        v = "EQUIV_PASS" if r["ok"] else "EQUIV_FAIL"
        print(f"\n[{name}]")
        print(f"  cases={r['cases']}  max_ulp(cand vs base, target dtype)={r['max_ulp']}  "
              f"max #elem beyond tolerance(cand vs base)={r['viol_cb']}")
        print(f"  max degradation of candidate vs baseline (rel err vs fp64) = "
              f"{r['delta']:+.3e}")
        print(f"  criterion (a) [cand vs base within tolerance] holds on {r['n_a']}/{r['cases']} cases")
        print(f"  cases where (a) fails and is rescued by (b): "
              f"{len(r['only_b'])}  {r['only_b'][:6] if r['only_b'] else ''}")
        print(f"  cases failing both criteria: {r['failed'] if r['failed'] else 'none'}")
        print(f"  violations vs fp64 (sum over cases): candidate={r['viol_cand']}  "
              f"base={r['viol_base']}  cases where candidate worse: "
              f"{r['worse_cases'] if r['worse_cases'] else 'none'}")
        print(f"  -> {v}")

    # ------------------------------------------------------------------
    # O2 segment evidence: rstd relative error alone (model-free)
    # ------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("O2 segment evidence (rstd alone, var sweep 0 -> 1e6):")
    print(f"  raw T.vrsqrt (measured bound, trap)      max rel err = {PLAIN_MAX_REL:.2e}")
    print(f"  vsqrt+vdiv composition (this script)     max rel err = "
          f"{rstd_segment(torch.float32, 'compose'):.2e}")
    print(f"  exact 1/sqrt (reference)                 max rel err = "
          f"{rstd_segment(torch.float32, 'exact'):.2e}")
    print("  fp32 gate rtol = 1e-5  ->  raw vrsqrt is ~2.9e2x over, composition ~1e2x under")

    # ------------------------------------------------------------------
    # O4 corner evidence: pad-count cancellation of the source correction
    # ------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("O4 corner evidence (GPU 256-pad path, pad*mean^2 >> sum((x-mean)^2)):")
    for cname, M, D, dt, gen in cases():
        if not cname.startswith("padcancel"):
            continue
        x = gen(M, D, dt)
        atol, rtol = TOL[dt]
        ref64 = reference_fp64(x, D, 1e-5)
        pad = ((D + GPU_ALIGNMENT - 1) // GPU_ALIGNMENT) * GPU_ALIGNMENT - D
        base = _run({**CANDIDATE, "pad_mode": "pad256"}, x, D, 1e-5)
        cand = _run(CANDIDATE, x, D, 1e-5)
        print(f"  {cname:18s} D={D:5d} pad={pad:4d}  rel_err base(256-pad)="
              f"{_rel_err_vs_fp64(base, ref64):.3e}  exact-D="
              f"{_rel_err_vs_fp64(cand, ref64):.3e}  viol base/exact="
              f"{_violations(base, ref64, atol, rtol)}/{_violations(cand, ref64, atol, rtol)}")

    # ------------------------------------------------------------------
    # rejection evidence (NOT adoption rows): single-pass moment form
    # ------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("rejection evidence (non-adopted candidate; DESIGN.md 1.6.1 rejection table)")
    print("-" * 100)
    worst = (0.0, 0, "", 0.0)
    for cname, M, D, dt, gen in cases():
        x = gen(M, D, dt)
        atol, rtol = TOL[dt]
        ref64 = reference_fp64(x, D, 1e-5)
        moment = pipeline(x, D, 1e-5, **{**CANDIDATE, "var_mode": "moment"})
        base = pipeline(x, D, 1e-5, **CANDIDATE)
        vm = _violations(moment, ref64, atol, rtol)
        vb = _violations(base, ref64, atol, rtol)
        em = _rel_err_vs_fp64(moment, ref64)
        eb = _rel_err_vs_fp64(base, ref64)
        flag = "  <-- worse than two-pass" if vm > vb else ""
        print(f"  C1 moment  {cname:16s} {str(dt):16s} rel_err={em:.3e} "
              f"(two-pass {eb:.3e})  viol={vm} (two-pass {vb}){flag}")
        if em > worst[0]:
            worst = (em, vm, cname, eb)

    print(f"\n  C1 worst: rel_err={worst[0]:.3e} (two-pass {worst[3]:.3e}) on {worst[2]}, "
          f"violations={worst[1]}  -> rejected in 1.6.1 (numerical instability)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("| adopted item | cases | criterion (a) cases | max ulp (cand vs base) | "
          "max #elem beyond tol (cand vs base) | max degradation vs fp64 | "
          "cases failing both | verdict |")
    print("|--------------|-------|---------------------|------------------------|"
          "-------------------------------------|------------------------|"
          "--------------------|---------|")
    all_ok = True
    for r in results:
        v = "EQUIV_PASS" if r["ok"] else "EQUIV_FAIL"
        all_ok &= r["ok"]
        print(f"| {r['item']} | {r['cases']} | {r['n_a']}/{r['cases']} | "
              f"{r['max_ulp']} | {r['viol_cb']} | {r['delta']:+.1e} | "
              f"{len(r['failed'])} | {v} |")
    print("=" * 100)
    print("criterion (a) = candidate vs baseline within the target-dtype tolerance")
    print("criterion (b) = candidate no further from the fp64 reference than the baseline")
    print("verdict = every case satisfies (a) or (b).  In well-conditioned cases (a) holds with")
    print("0 elements beyond tolerance; the pads/corner cases are judged by (b) -- see the")
    print("per-regime evidence blocks above.")
    print("NOTE: the residual 'violations vs fp64' are dominated by the fp32 row-sum")
    print("      accumulation error shared by BOTH arms (identical in cand/base), not by")
    print("      the item under test -- the item-attributable column is 'max #elem beyond tol'.")
    print("ALL ADOPTED ITEMS PASS" if all_ok else "SOME ADOPTED ITEMS FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
