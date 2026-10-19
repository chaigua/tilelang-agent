#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""D-1 equivalence verification for `_gqa_bwd_kernel` DESIGN.md (section 1.6.1).

Adopted / rejected items verified here (torch CPU, fp64 oracle, no NPU needed):

  O1  causal mask as an exponent-domain penalty
      baseline (source): P = where(by*bm+j <= kq*bn+i, exp2(x), 0)
      candidate (design):P = exp2(x + pen),  pen = clamp(j-i-d0, 0, 1) * (-PEN)
                        and pen = min(-PEN*(j-i-d0), 0)   [pseudo-code form]
      -> bit-identical on the operator's (finite-score) domain, because
         exp2(negative-finite - PEN) underflows to exactly +0.0 in fp32.
      REVISION 1 (REVIEW.md problem 5): the generator now sweeps four mask
      offsets d0 per tile so EVERY tile shape contains masked positions, and
      both penalty spellings are checked 3-way against each other and the
      source select.

  O4  exp2 -> exp lowering substitution with rescaled argument
      baseline (T.vexp2 lowering): exp((qkt*scale - lse) * ln2)
      candidate (T.vexp):          exp(qkt*sm_scale - lse*ln2)
      with scale = D**-0.5 * LOG2E  and  sm_scale = D**-0.5.
      -> tolerance-equivalent (both are the same real function; two fp32
      roundings on each path).  Compared against the fp64 oracle.

  O2  dS computed in place in the `dsum` carrier (no separate dS tile)
  O3  one dtype-cast scratch shared by the P cast and the dS cast
      -> buffer-lifetime items; must be bit-identical element-wise.

  P1  claim: fp16/bf16 -> fp32 upcast is exact and the fp32 elementwise chain
      is the required path (the design casts P / dS to the *input* dtype with
      round-to-nearest-even, matching the source's fragment cast).

  R1  REJECTED: folding `scale` into K before the GEMM
      (`k' = rint16(scale*k)`); K is stored in the input dtype, so pre-scaling
      quantizes the GEMM operand -- not equivalent to scaling the score.
  R2  REJECTED: masking the score with the -1e38 sentinel instead of -1e30
      (`t * -1e38` overflows to -inf for t up to block_m) -- still gives an
      exact +0.0, but overflows the intermediate; kept as a documented note.

Criteria (same as the sibling kernel's gate):
  (a) candidate vs baseline within the kernel tolerance
      (atol = rtol = 1e-4) with 0 violating elements, or
  (b) candidate vs the fp64 oracle is not worse than baseline vs fp64
      by more than one tolerance budget.

Run:  python verify_equiv.py
"""

import math

import torch

torch.manual_seed(20261016)

ATOL, RTOL = 1e-4, 1e-4
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
PEN = 1e30  # finite sentinel (see DESIGN section 1.6.1 O1)


def _ulp(a, b):
    """Max integer ulp distance (fp32), non-finite pairs excluded."""
    a = a.float().reshape(-1).double()
    b = b.float().reshape(-1).double()
    fin = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[fin], b[fin]
    if a.numel() == 0:
        return 0
    ia = a.float().view(torch.int32).to(torch.int64)
    ib = b.float().view(torch.int32).to(torch.int64)
    ia = torch.where(ia < 0, 0x80000000 - ia, ia)
    ib = torch.where(ib < 0, 0x80000000 - ib, ib)
    return int((ia - ib).abs().max().item())


def _viol(cand, base):
    d = (cand.float() - base.float()).abs()
    return int((d > ATOL + RTOL * base.float().abs()).sum().item())


def _f64_ulp(cand, oracle):
    """ulps of candidate vs the fp64 oracle (fp32-rounded oracle)."""
    return _ulp(cand, oracle.float())


# ---------------------------------------------------------------------------
# case generator: random scores + corner values, several (bm, bn) tiles and
# several mask offsets d0 (so the masked branch is exercised on EVERY tile)
# ---------------------------------------------------------------------------
def _cases():
    out = []
    for (bm, bn) in ((64, 64), (64, 32), (32, 32), (16, 64), (1, 1), (7, 5)):
        i = torch.arange(bn).view(bn, 1).expand(bn, bm).double()   # q row
        j = torch.arange(bm).view(1, bm).expand(bn, bm).double()   # kv col
        diff = j - i
        # REVISION 1 (REVIEW.md problem 5): multiple mask offsets so every tile
        # shape really contains masked positions (the old `d0 = bm - 1` made
        # masked == 0 for 5 of the 6 tiles).
        for d0 in sorted({bm - 1, bm // 2, 0, -(max(1, bn // 2))}):
            valid = diff <= d0
            for tag in ("rand", "large", "zero", "tiny", "neg"):
                if tag == "rand":
                    s = torch.randn(bn, bm, dtype=torch.float64) * 8.0
                elif tag == "large":
                    s = torch.randn(bn, bm, dtype=torch.float64) * 300.0
                elif tag == "zero":
                    s = torch.zeros(bn, bm, dtype=torch.float64)
                elif tag == "tiny":
                    s = torch.randn(bn, bm, dtype=torch.float64) * 1e-6
                else:
                    s = -torch.rand(bn, bm, dtype=torch.float64) * 1e4
                out.append((f"{tag}@d0={d0}", bm, bn, s, valid, diff, d0))
    return out


def _mask_stats(valid):
    return int((~valid).sum().item()), int(valid.numel())


def check_o1():
    """O1: exponent-domain penalty mask vs post-exp2 select.

    Both equivalent closings of the penalty are checked, because revision 0's
    prose used three spellings and the pseudo-code used a fourth:
      form A (clamp):  pen = clamp(diff - d0, 0, 1) * (-PEN)
      form B (pseudo): pen = min(-PEN * (diff - d0), 0)
    They must be bit-identical to the source select (and to each other).
    """
    n_case = viol = 0
    max_ulp = 0
    n_masked_cases = 0
    masked_elems = 0
    a = [(64, 64), (64, 32), (32, 32), (16, 64), (1, 1), (7, 5)]
    tiles_with_mask = set()
    for tag, bm, bn, s64, valid, diff, d0 in _cases():
        n_case += 1
        nm, ne = _mask_stats(valid)
        masked_elems += nm
        if nm > 0:
            n_masked_cases += 1
            tiles_with_mask.add((bm, bn))
        # baseline: exp2 in fp32 then select
        xb = s64.float()
        Pb = torch.where(valid, torch.exp2(xb), torch.zeros_like(xb))
        # form A: clamped penalty
        scr = ((diff - d0).clamp(min=0.0, max=1.0)).float()
        Pa = torch.exp2((xb.double() + (scr * (-PEN)).double()).float())
        # form B: linear penalty min(-PEN*(diff-d0), 0)
        penb = torch.minimum((-PEN) * (diff - d0), torch.zeros_like(diff)).float()
        Pc = torch.exp2((xb.double() + penb.double()).float())
        max_ulp = max(max_ulp, _ulp(Pc, Pb), _ulp(Pa, Pb), _ulp(Pa, Pc))
        viol += _viol(Pc, Pb) + _viol(Pa, Pb)
    return (n_case, max_ulp, viol, n_masked_cases, masked_elems,
            sorted(tiles_with_mask), a)


def check_o1_nonfinite():
    """O1 domain note: NaN scores at masked positions."""
    s = torch.tensor([[float("nan"), 1.0], [2.0, 3.0]], dtype=torch.float64).float()
    valid = torch.tensor([[False, True], [True, True]])
    Pb = torch.where(valid, torch.exp2(s), torch.zeros_like(s))
    pen = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    pen[0, 0] = -1e30
    Pc = torch.exp2((s.double() + pen.double()).float())
    return bool(torch.isnan(Pc[0, 0])) and float(Pb[0, 0]) == 0.0


def check_o4():
    """O4: exp2(x) [T.vexp2 lowering] vs exp(qkt*sm_scale - lse*ln2) [T.vexp].

    Realistic regime: `qkt` is a raw Q@K^T GEMM output (entries ~ N(0, sqrt(D))
    for unit-variance q/k), so the natural-domain score `qkt*sm_scale` is
    O(1) and the base-2 log-sum-exp `lse = logsumexp(scores)/ln2` is O(1..6)
    -- i.e. NO artificial cancellation is injected into either path.
    """
    n_case = max_ulp = viol = 0
    worst_vs_oracle = {"base": 0, "cand": 0}
    seen = set()
    for tag, bm, bn, s64, valid, diff, d0 in _cases():
        key = (bm, bn, tag.split("@")[0])   # d0 is irrelevant to O4 -> dedupe
        if key in seen:
            continue
        seen.add(key)
        for D in (64, 128):
            n_case += 1
            sm_scale = D**-0.5
            qkt = (torch.randn(bm, bn, dtype=torch.float64) * math.sqrt(D)).t()
            scores = qkt * sm_scale                      # natural domain, O(1)
            lse = torch.logsumexp(scores, dim=1, keepdim=True) / LN2   # base-2
            x64 = qkt * (sm_scale * LOG2E) - lse
            # baseline: NPU T.vexp2 lowering = exp((x) * ln2)
            xb = ((qkt.float() * (sm_scale * LOG2E)) - lse.float())
            Pb = torch.exp((xb.double() * LN2).float())
            # candidate: T.vexp with rescaled argument
            Pc = torch.exp(((qkt.float() * sm_scale).double()
                            - (lse.float() * LN2).double()).float())
            oracle = torch.exp2(x64)
            max_ulp = max(max_ulp, _ulp(Pc, Pb))
            viol += _viol(Pc, Pb)
            worst_vs_oracle["base"] = max(worst_vs_oracle["base"], _f64_ulp(Pb, oracle))
            worst_vs_oracle["cand"] = max(worst_vs_oracle["cand"], _f64_ulp(Pc, oracle))
    return n_case, max_ulp, viol, worst_vs_oracle


def check_buffer_items():
    """O2 / O3: buffer-lifetime reuses must be bit-identical."""
    bm, bn, D = 64, 64, 128
    qkt = torch.randn(bn, bm).float() * 4
    dsum = torch.randn(bn, bm).float() * 4
    delta = torch.randn(bn, 1).float()
    sm_scale = D**-0.5
    # O2 baseline: dS to a fresh tile / candidate: in place in dsum
    dS_base = (qkt * (dsum - delta) * sm_scale)
    dsum_c = dsum.clone()
    dsum_c.sub_(delta)          # vsub in place
    dsum_c.mul_(qkt)            # vmul   in place
    dsum_c.mul_(sm_scale)       # vmul   in place
    o2_exact = bool(torch.equal(dS_base, dsum_c))
    # O3: one scratch shared by the P cast and the dS cast
    P = torch.randn(bn, bm).float() * 3
    dS = torch.randn(bn, bm).float() * 3
    p_sep = P.to(torch.float16)
    d_sep = dS.to(torch.float16)
    scratch = torch.empty(bn, bm, dtype=torch.float16)
    scratch.copy_(P.to(torch.float16))
    p_sh = scratch.clone()
    scratch.copy_(dS.to(torch.float16))
    d_sh = scratch.clone()
    o3_exact = bool(torch.equal(p_sep, p_sh) and torch.equal(d_sep, d_sh))
    # P1: fp16/bf16 -> fp32 upcast + fp32 product is exact
    a16 = (torch.randn(1 << 16) * 10).to(torch.float16)
    b16 = (torch.randn(1 << 16) * 10).to(torch.float16)
    prod32 = a16.float() * b16.float()
    prod64 = a16.double() * b16.double()
    p1_ulp = _ulp(prod32, prod64)
    a8 = a16.to(torch.bfloat16)
    b8 = b16.to(torch.bfloat16)
    p1_ulp_b = _ulp(a8.float() * b8.float(), a8.double() * b8.double())
    return o2_exact, o3_exact, p1_ulp, p1_ulp_b


def check_r1():
    """R1 (rejected): folding `scale` into K before the GEMM."""
    D = 128
    scale = D**-0.5 * LOG2E
    k16 = (torch.randn(512, D) / math.sqrt(D)).to(torch.float16)
    q16 = (torch.randn(512, D) / math.sqrt(D)).to(torch.float16)
    ref = (q16.float() @ k16.float().t()) * scale          # baseline
    k_scaled = (k16.float() * scale).to(torch.float16)     # pre-scaled operand
    cand = q16.float() @ k_scaled.float().t()
    rel = ((cand - ref).abs() / (ref.abs() + 1e-6))
    return _ulp(cand, ref), float(rel.max().item())


def main():
    print("=" * 78)
    print("D-1 equivalence verification — _gqa_bwd_kernel (torch CPU, fp64 oracle)")
    print("=" * 78)
    n, ulp, viol, nmask, nelem, tiles_m, _a = check_o1()
    print(f"O1 mask-penalty            : cases={n:3d}  max_ulp={ulp:<6d} "
          f"violations={viol:<4d}  -> {'EQUIV_PASS' if viol == 0 else 'EQUIV_FAIL'}")
    print(f"    (both forms: clamp & min(-PEN*(diff-d0),0), compared 3-way with "
          f"each other -> max_ulp above is the 3-way worst)")
    print(f"    mask coverage          : {nmask}/{n} cases have masked>0 "
          f"({nelem} masked elements); tiles with masks={tiles_m}")
    nan_note = check_o1_nonfinite()
    print(f"O1 non-finite domain note  : candidate propagates NaN where the source "
          f"select yields 0  (observed={nan_note}; outside the finite-input domain)")
    n, ulp, viol, worst = check_o4()
    print(f"O4 vexp2 -> vexp           : cases={n:3d}  max_ulp={ulp:<6d} "
          f"violations={viol:<4d}  -> {'EQUIV_PASS' if viol == 0 else 'EQUIV_FAIL'}")
    print(f"    vs fp64 oracle (ulps)  : baseline={worst['base']}  "
          f"candidate={worst['cand']}  -> candidate not worse by more than "
          f"one tolerance budget: {worst['cand'] <= worst['base'] * 4 + 4}")
    o2, o3, p1, p1b = check_buffer_items()
    print(f"O2 in-place dS             : bit-identical={o2}  -> "
          f"{'EQUIV_PASS' if o2 else 'EQUIV_FAIL'}")
    print(f"O3 shared cast scratch     : bit-identical={o3}  -> "
          f"{'EQUIV_PASS' if o3 else 'EQUIV_FAIL'}")
    print(f"P1 upcast+fp32 product     : fp16 max_ulp={p1}  bf16 max_ulp={p1b}  -> "
          f"{'EQUIV_PASS' if p1 == 0 and p1b == 0 else 'EQUIV_FAIL'}")
    ulp, rel = check_r1()
    print(f"R1 scale-into-K (REJECTED) : max_ulp={ulp}  max_rel={rel:.3e}  -> "
          f"rejected (input-dtype operand quantization)")
    ok = (viol == 0) and o2 and o3 and p1 == 0 and p1b == 0
    print("=" * 78)
    print("VERDICT:", "ALL ADOPTED ITEMS EQUIV_PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
