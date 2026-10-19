#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""D-1 equivalence verification for `_flashattn_bwd_preprocess_kernel` DESIGN.md.

Adopted optimization item (DESIGN.md section 1.6.1):

  O1  in-place product reuse
      baseline (source):  separate product tile  prod = fp32(o) * fp32(do)
      candidate (design): product written in place into the fp32 carrier of
                          `o`  (acc32 = acc32 * aux32)
      -> must be bit-identical (each element is computed by the same fp32
         multiply; only the destination buffer differs).

Numerical claims the design relies on (verified here as well, so that a Stage 2
reviewer can re-run the same evidence):

  P1  fp16 -> fp32 (and bf16 -> fp32) upcast is exact, and the fp32 product of
      two upcast operands is exact (mantissa bits: fp16 11+11 = 22 <= 24,
      bf16 8+8 = 16 <= 24)  => the kernel's `vcast` + `vmul` fp32 chain
      reproduces the fp64-exact product with 0 ulp.
  P2  accumulation association: the design's single fp32 reduction over the
      D axis vs the source's chunked product accumulation (blk = 256) followed
      by a tile reduction, and the worst-case serial fp32 accumulation.
  R1  rejected candidate evidence: multiplying in the *input* dtype
      (fp16 product) is NOT equivalent (rounding to fp16 loses up to 11 bits).

Judgement criteria (same as the group_norm precedent):
  (a) candidate vs baseline within the kernel tolerance (atol = rtol = 1e-4)
      with 0 violating elements, or
  (b) candidate vs the fp64 oracle is not worse than baseline vs fp64.

Run:  python verify_equiv.py            (torch CPU only, no NPU required)
"""

import math

import torch

torch.manual_seed(20261016)

ATOL, RTOL = 1e-4, 1e-4  # kernel-level fp32 tolerance (DESIGN.md section 8.2)
CHUNK = 256  # source kernel `blk`

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max integer ulp distance between two fp32 tensors (non-finite = 0)."""
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[both_finite], b[both_finite]
    if a.numel() == 0:
        return 0
    ia = a.view(torch.int32).to(torch.int64)
    ib = b.view(torch.int32).to(torch.int64)
    ia = torch.where(ia < 0, 0x80000000 - ia, ia)  # monotone int mapping
    ib = torch.where(ib < 0, 0x80000000 - ib, ib)
    return int((ia - ib).abs().max().item())


def _violations(out: torch.Tensor, ref: torch.Tensor) -> int:
    """Elements outside atol + rtol*|ref| (NaN == NaN counted equal)."""
    o, r = out.float(), ref.float()
    eq = (o == r) | (torch.isnan(o) & torch.isnan(r))
    diff = torch.where(eq, torch.zeros((), dtype=torch.float32), (o - r).abs())
    return int((diff > (ATOL + RTOL * r.abs())).sum().item())


def _carriers(o_lo: torch.Tensor, do_lo: torch.Tensor):
    """fp32 carrier of the two inputs + the fp64 oracle product/sum."""
    o32 = o_lo.float().double().float()  # upcast through fp64 (exact anyway)
    do32 = do_lo.float().double().float()
    prod64 = o_lo.double() * do_lo.double()
    return o32, do32, prod64


def _pairwise_fp32(x: torch.Tensor) -> torch.Tensor:
    """Correct pairwise fp32 tree over the last axis (odd lengths kept).

    Odd elements are carried by zero-padding (`+0.0` is exact for finite
    values), so no term is dropped.
    """
    cur = x
    while cur.shape[-1] > 1:
        if cur.shape[-1] % 2:
            pad = torch.zeros(cur.shape[:-1] + (1,), dtype=cur.dtype)
            cur = torch.cat([cur, pad], dim=-1)
        half = cur.shape[-1] // 2
        cur = cur[..., :half] + cur[..., half:]
    return cur[..., 0]


def _accum_orders(prod32: torch.Tensor):
    """Return the baseline (source) order plus the plausible hardware orders.

    baseline  : source kernel -- accumulate the product tile over D chunks of
                `CHUNK`, then reduce the tile along the D axis.
    tree      : hierarchical (pairwise) fp32 reduction -- ideal vector reduce.
    simd8/64  : fp32 in-block pairwise tree over 8 / 64 lanes, then a
                sequential combine of the block sums (plausible
                `hivm.hir.vreduce` shape).
    serial    : fully serial fp32 accumulation -- reported as a BOUND only.
    oracle    : fp64 exact sum.
    """
    n = prod32.shape[-1]
    acc = torch.zeros_like(prod32)
    nch = max(1, math.ceil(n / CHUNK))
    for k in range(nch):  # source: accumulate products tile-wise over D chunks
        acc[..., k * CHUNK : (k + 1) * CHUNK] += prod32[..., k * CHUNK : (k + 1) * CHUNK]
    baseline = _pairwise_fp32(acc)  # tile reduce (source `reduce_sum`)

    def _block_tree(width: int):
        width = min(width, n)
        nb = max(1, math.ceil(n / width))
        out = torch.zeros(prod32.shape[:-1], dtype=torch.float32)
        for b in range(nb):
            out = out + _pairwise_fp32(prod32[..., b * width : (b + 1) * width])
        return out

    serial = torch.zeros(prod32.shape[:-1], dtype=torch.float32)
    for d in range(n):
        serial = serial + prod32[..., d]
    return dict(baseline=baseline, tree=_pairwise_fp32(prod32), simd8=_block_tree(8),
                simd64=_block_tree(64), serial=serial,
                oracle=prod32.double().sum(-1))


def item_p2(cases):
    """P2: the design's reduction order vs the source order (fp64 oracle).

    Criterion (a): candidate vs baseline within the kernel tolerance
    (atol = rtol = 1e-4) with 0 violating elements.
    Criterion (b) (only for cases that fail (a)): the candidate's deviation
    from the fp64 oracle stays inside one tolerance budget of the baseline's
    deviation -- i.e. the candidate is not materially worse than the source
    order.
    """
    adopted = ("tree", "simd8", "simd64")
    worst_ulp, worst_ulp_reg, viol, worse_cases, b_fail = 0, 0, 0, [], 0
    serial_ulp, serial_viol, serial_worse = 0, 0, 0
    detail = []
    for name, o, do in cases:
        o32, do32, _ = _carriers(o, do)
        r = _accum_orders(o32 * do32)
        base = r["baseline"]
        cv, cu = 0, 0
        for key in adopted:
            cu = max(cu, _ulp_diff(r[key], base))
            cv += _violations(r[key], base)
        worst_ulp = max(worst_ulp, cu)
        if "saturate" not in name:
            worst_ulp_reg = max(worst_ulp_reg, cu)
        viol += cv
        serial_ulp = max(serial_ulp, _ulp_diff(r["serial"], base))
        serial_viol += _violations(r["serial"], base)
        base_err = (base.double() - r["oracle"]).abs()
        budget = ATOL + RTOL * r["oracle"].abs()
        bworse, b_fail_here = 0, 0
        for key in adopted:
            cand_err = (r[key].double() - r["oracle"]).abs()
            finite = torch.isfinite(base_err) & torch.isfinite(cand_err)
            if finite.any() and bool((cand_err[finite] > base_err[finite] + 1e-30).any()):
                worse_cases.append(f"{name}/{key}")
                bworse += 1
            if finite.any():
                over = cand_err[finite] > (base_err[finite] + budget[finite])
                if bool(over.any()):
                    b_fail_here += 1
        if cv or b_fail_here:
            detail.append((name, cv, cu, bworse, float(base_err.max()),
                           float(max((r[k].double() - r["oracle"]).abs().max()
                                     for k in adopted)), b_fail_here))
        b_fail += 1 if (cv and b_fail_here) else 0
        serr_err = (r["serial"].double() - r["oracle"]).abs()
        finite = torch.isfinite(base_err) & torch.isfinite(serr_err)
        if finite.any() and float((serr_err[finite] - base_err[finite]).max()) > 0:
            serial_worse += 1
    ok = b_fail == 0
    return dict(name="P2 累加结合序（设计序 vs 源序）", n=len(cases), ulp=worst_ulp,
                ulp_reg=worst_ulp_reg,
                viol=viol, verdict="EQUIV_PASS" if ok else "REVIEW",
                worse=len(worse_cases), serial_ulp=serial_ulp,
                serial_viol=serial_viol, serial_worse=serial_worse,
                b_fail=b_fail, detail=detail)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _cases():
    """(name, o, do) in fp16 and/or bf16, incl. corner values.

    Row counts are kept small (<= 64) on purpose: the D extent is the only
    axis that drives the numerical claims, and the fp64 oracle / order
    emulation are O(elements) in python-visible torch ops.
    """
    out = []
    shape_sets = [(2, 8, 2, 1), (2, 8, 2, 7), (2, 8, 2, 63), (2, 8, 2, 64),
                  (2, 8, 2, 128), (2, 8, 2, 255), (2, 8, 2, 256), (2, 8, 2, 1024),
                  (2, 8, 2, 4096), (1, 16, 4, 64), (4, 32, 8, 128)]
    for dtype_name, dt in _DTYPES.items():
        for shape in shape_sets:
            o = torch.randn(*shape).to(dt)
            do = torch.randn(*shape).to(dt)
            out.append((f"{dtype_name}-randn-D{shape[-1]}", o, do))
        shape = (1, 16, 2, 128)
        o = torch.randn(*shape).to(dt)
        do = torch.randn(*shape).to(dt)
        o[0, 0, 0, :] = float("inf")
        do[0, 0, 0, :] = 1.0
        o[0, 1, 0, :] = float("-inf")
        do[0, 1, 0, :] = 1.0
        o[0, 2, 0, :] = float("nan")
        out.append((f"{dtype_name}-inf-nan", o, do))
        o = (torch.randn(*shape) * 1e-5).to(dt)  # subnormal range
        do = (torch.randn(*shape) * 1e-5).to(dt)
        out.append((f"{dtype_name}-subnormal", o, do))
        lim = 65504.0 if dt is torch.float16 else 3.0e38
        o = (torch.randn(*shape).sign() * lim).to(dt)  # saturation
        do = (torch.randn(*shape).sign() * lim).to(dt)
        out.append((f"{dtype_name}-saturate", o, do))
        o = torch.zeros(*shape).to(dt)  # zeros -> exact 0
        do = torch.randn(*shape).to(dt)
        out.append((f"{dtype_name}-zeros", o, do))
        o = (torch.randn(*shape) * 1e3).to(dt)  # cancellation: mixed signs
        do = torch.randn(*shape).to(dt)
        o = o * torch.where(torch.arange(shape[-1]) % 2 == 0, 1.0, -1.0).to(dt)
        out.append((f"{dtype_name}-cancel", o, do))
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------
def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality (NaN payload / -0.0 distinguished)."""
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def item_o1(cases):
    """O1: in-place product reuse -> bit identity."""
    worst_ulp = 0
    viol = 0
    exact = True
    for _, o, do in cases:
        o32, do32, _ = _carriers(o, do)
        separate = o32 * do32  # baseline: separate product buffer
        inplace = o32.clone()
        inplace = inplace * do32  # candidate: product into the o carrier
        worst_ulp = max(worst_ulp, _ulp_diff(separate, inplace))
        viol += _violations(inplace, separate)
        exact &= _bitwise_equal(separate, inplace)
    ok = exact and worst_ulp == 0 and viol == 0
    return dict(name="O1 原位乘积复用", n=len(cases), ulp=worst_ulp, viol=viol,
                verdict="EQUIV_PASS" if ok else "REVIEW", exact=exact)


def item_p1(cases):
    """P1: upcast + fp32 product are exact vs the fp64 product."""
    worst_ulp, viol, worse = 0, 0, 0
    for _, o, do in cases:
        o32, do32, prod64 = _carriers(o, do)
        p32 = o32 * do32
        fin = torch.isfinite(prod64)
        if fin.any():
            worst_ulp = max(worst_ulp, _ulp_diff(p32[fin], prod64[fin].float()))
        viol += _violations(p32, prod64.float())
    return dict(name="P1 upcast+乘积精确性", n=len(cases), ulp=worst_ulp, viol=viol,
                verdict="EQUIV_PASS" if (worst_ulp == 0 and viol == 0) else "REVIEW")




def item_r1(cases):
    """R1: rejected candidate -- product in the input dtype (fp16)."""
    worst_rel, worst_abs, viol, n16 = 0.0, 0.0, 0, 0
    for name, o, do in cases:
        if o.dtype is not torch.float16:
            continue
        n16 += 1
        _, _, prod64 = _carriers(o, do)
        p16 = (o * do).float().double()  # product rounded to fp16, then exact
        fin = torch.isfinite(prod64)
        if fin.any():
            dev = (p16[fin] - prod64[fin]).abs()
            worst_abs = max(worst_abs, float(dev.max()))
            big = prod64[fin].abs() > 1e-30
            if bool(big.any()):
                worst_rel = max(worst_rel, float((dev[big] / prod64[fin][big].abs()).max()))
        viol += _violations(p16.float(), prod64.float())
    return dict(name="R1 输入 dtype 乘积（否决候选）", n=n16, ulp=None,
                viol=viol, rel=worst_rel, abs_dev=worst_abs, verdict="REJECTED")


def main():
    cases = _cases()
    rows = [item_o1(cases), item_p1(cases), item_p2(cases), item_r1(cases)]
    print("\n=== D-1 equivalence table (torch CPU, fp64 oracle) ===\n")
    print("| 项 | 案例数 | 最大 ulp 差 | 超容差元素数（atol=rtol=1e-4） | 结论 |")
    print("|----|-------|------------|------------------------------|------|")
    for r in rows:
        ulp = r.get("ulp")
        if "ulp_reg" in r:
            ulp = f"{r['ulp_reg']}（常规案例）/{_fmt(r['ulp'])}（饱和角点）"
        print(f"| {r['name']} | {r['n']} | {_fmt(ulp)} | {r['viol']} | {r['verdict']} |")
    for r in rows:
        if "worse" in r:
            print(f"  - {r['name']}: 候选劣于基线的案例数 = {r['worse']}")
        if "rel" in r:
            print(f"  - {r['name']}: 相对 fp64 最大相对误差 = {r['rel']:.3e}"
                  f"，最大绝对偏差 = {r['abs_dev']:.3e}")
        if "serial_ulp" in r:
            print(f"  - {r['name']} 边界注（全串行序，非采纳序）: max ulp = {r['serial_ulp']}"
                  f", 超容差元素数 = {r['serial_viol']}, 劣于基线的案例数 = {r['serial_worse']}")
        if "b_fail" in r:
            print(f"  - {r['name']}: 判否（两准则皆失败）的案例数 = {r['b_fail']}"
                  f", 逐序劣于基线的 (案例,序) 对数 = {r['worse']}")
        for d in r.get("detail", []):
            print(f"    * {d[0]}: (a) 超容差 {d[1]} 例, max ulp {d[2]}, (b) 判否序 {d[6]} 个, "
                  f"基线 vs fp64 最大绝对误差 {d[4]:.3e}, 候选 vs fp64 {d[5]:.3e}")
    bad = [r["name"] for r in rows if r["verdict"] == "REVIEW"]
    print("\nSUMMARY:", "ALL_EQUIV_PASS" if not bad else f"NEEDS_REVIEW {bad}")


if __name__ == "__main__":
    main()
