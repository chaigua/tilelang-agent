#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""D-1 machine equivalence verification for ``_flashattn_bwd_postprocess_kernel``.

Contract under test
-------------------
Adopted equivalence item **O1** (DESIGN.md section 1.6.1):

    source (baseline)   : implicit cross-dtype ``T.copy`` cast  -> IEEE round-to-nearest-even
    adopted  (candidate): explicit ``T.vcast(round_mode="rint")``

Both are claimed to be IEEE ``round-to-nearest-even`` (RNE).  This script
verifies that claim by comparing two *independent* implementations of RNE on
the same fp32 inputs:

    baseline  = ``torch.Tensor.to(storage_dtype)``  (torch CPU)
    candidate = exact-rational round-to-nearest-even built here from the
                mathematical definition with ``fractions.Fraction`` -- it does
                not call torch or numpy for the rounding step

and, as negative controls, the rounding modes **rejected** in DESIGN.md
section 1.6.1 (``round`` = half-away-from-zero, ``trunc`` = toward zero).  They
must differ from the baseline, otherwise "rint" would not be a distinguishing
equivalence claim.

Coverage: random tensors + corner values (``+/-inf``, ``NaN``, ``+/-0``, fp16
subnormals, fp16 overflow boundary, fp32 extremes, exact halfway "tie" values)
for both storage dtypes.

Run (no NPU, torch CPU only):
    python verify_equiv.py
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr

import torch

__all__ = ["run", "main"]

# fp16 : 1 sign + 5 exponent + 10 mantissa bits
# bf16 : 1 sign + 8 exponent + 7  mantissa bits
_FMT = {
    "float16": dict(p=11, emin=-14, emax=15, torch_dtype=torch.float16),
    "bfloat16": dict(p=8, emin=-126, emax=127, torch_dtype=torch.bfloat16),
}

_SIGN_MASK = 0x8000_0000
_EXP_MASK = 0x7F80_0000
_MANT_MASK = 0x007F_FFFF


# --------------------------------------------------------------------------
# Exact rational rounding (the candidate implementation)
# --------------------------------------------------------------------------
def _round_int(num: int, den: int, mode: str) -> int:
    """Round the exact non-negative fraction ``num/den`` to an integer."""
    q, r = divmod(num, den)
    if mode == "trunc":
        return q
    twice = 2 * r
    if mode == "rint":  # round half to even
        return q + 1 if (twice > den or (twice == den and (q & 1))) else q
    if mode == "round":  # round half away from zero
        return q + 1 if twice >= den else q
    raise ValueError(mode)


def _round_value_to_format(bits: int, dtype: str, mode: str) -> float:
    """Round one fp32 bit pattern to ``dtype`` exactly; return a Python float."""
    spec = _FMT[dtype]
    p, emin, emax = spec["p"], spec["emin"], spec["emax"]

    sign = -1.0 if (bits & _SIGN_MASK) else 1.0
    exp = (bits & _EXP_MASK) >> 23
    mant = bits & _MANT_MASK
    if exp == 0xFF:  # inf / NaN pass through both float formats
        return math.nan if mant else sign * math.inf

    if exp == 0:
        a = Fr(mant, 1 << 149)             # subnormal fp32
    else:
        a = Fr((1 << 23) | mant, 1 << 23) * (Fr(2) ** (exp - 127))
    if a == 0:
        return sign * 0.0

    # binary exponent e with 2**e <= a < 2**(e+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fr(2) ** e > a:
        e -= 1
    while Fr(2) ** (e + 1) <= a:
        e += 1

    if e < emin:
        scale = emin - (p - 1)             # subnormal quantisation step
    else:
        scale = e - (p - 1)
    if scale >= 0:
        num, den = a.numerator, a.denominator << scale
    else:
        num, den = a.numerator << (-scale), a.denominator

    m_int = _round_int(num, den, mode)
    if e >= emin and m_int >= (1 << p):    # mantissa overflow -> next binade
        e += 1
        m_int >>= 1
        scale += 1
    if e > emax:
        return sign * math.inf
    return sign * float(Fr(m_int) * (Fr(2) ** scale))


def _reference_cast(x32: torch.Tensor, dtype: str, mode: str) -> torch.Tensor:
    """Independent exact-rational cast of an fp32 tensor (returns fp32 tensor).

    The returned fp32 tensor holds *exactly* the value the target storage
    format would hold (all such values are exactly representable in fp32), so
    a subsequent ``.to(storage_dtype)`` is lossless and reveals the bit
    pattern the candidate path produces.
    """
    bits = (x32.contiguous().view(torch.int32).to(torch.int64) & 0xFFFF_FFFF).tolist()
    vals = [_round_value_to_format(int(b), dtype, mode) for b in bits]
    return torch.tensor(vals, dtype=torch.float32)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Raw fp16/bf16 bit pattern as int64 (sign-magnitude, for exact compare)."""
    return t.view(torch.int16).to(torch.int64)


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------
def _cases(dtype: str) -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(20261016)
    out: list[tuple[str, torch.Tensor]] = []

    out.append(("random[-4,4]", torch.rand(4096, generator=g) * 8 - 4))
    out.append(("random[-7e4,7e4]", torch.rand(4096, generator=g) * 1.4e5 - 7e4))
    out.append(("random[-1e-4,1e-4]", torch.rand(4096, generator=g) * 2e-4 - 1e-4))

    # exact halfway ties at the storage-format mantissa width
    mant_bits = _FMT[dtype]["p"] - 1
    ties: list[float] = []
    for e in (-10, -3, 0, 1, 5):
        for k in range(8):
            m = Fr(1) + Fr(k, 1 << mant_bits)
            for extra in (Fr(1, 1 << (mant_bits + 1)), Fr(3, 1 << (mant_bits + 1))):
                v = float((m + extra) * (Fr(2) ** e))
                ties.extend([v, -v])
    out.append((f"exact-ties(2^-{mant_bits + 1})", torch.tensor(ties, dtype=torch.float32)))

    out.append(("corners", torch.tensor([
        0.0, -0.0,
        math.inf, -math.inf, math.nan,
        1.0, -1.0,
        65504.0, -65504.0,                 # fp16 max
        65519.99609375,                    # fp16 max + 1/2 ulp -> ties to inf
        65520.0,                           # fp16 first overflow
        6.103515625e-05,                   # fp16 smallest normal
        5.960464477539063e-08,             # fp16 smallest subnormal
        2.9802322387695312e-08,            # 1/2 of the smallest subnormal
        1.0009765625,                      # 1 + 2^-10 (exact in fp16)
        1.00048828125,                     # 1 + 2^-11 (tie -> rounds to even)
        1.5e-5, 3.0e-5,
        1.1754944e-38, 3.4028235e38, -3.4028235e38,
        1e-40, -1e-40,
    ], dtype=torch.float32)))
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _run_dtype(dtype: str) -> dict:
    tdt = _FMT[dtype]["torch_dtype"]
    rows, total_n, total_viol = [], 0, 0
    for name, x in _cases(dtype):
        x = x.contiguous().to(torch.float32)
        base = x.to(tdt)                                   # source semantics
        cand = _reference_cast(x, dtype, "rint").to(tdt)   # exact-rational RNE
        nan_mask = torch.isnan(base) | torch.isnan(cand)
        differ = (_bits(base) != _bits(cand)) & ~nan_mask
        viol = int(differ.sum().item())
        finite = torch.isfinite(base) & torch.isfinite(cand) & ~nan_mask
        ulp = int((_bits(cand)[finite] - _bits(base)[finite]).abs().max().item()) \
            if bool(finite.any()) else 0
        total_n += int(x.numel())
        total_viol += viol
        rows.append({"case": name, "n": int(x.numel()), "ulp": ulp, "viol": viol,
                     "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"})

    xt = torch.cat([c[1] for c in _cases(dtype)]).contiguous().to(torch.float32)
    base = xt.to(tdt)
    rejects = {}
    for mode in ("round", "trunc"):
        cand = _reference_cast(xt, dtype, mode).to(tdt)
        nan_mask = torch.isnan(base) | torch.isnan(cand)
        rejects[mode] = int(((_bits(base) != _bits(cand)) & ~nan_mask).sum().item())

    return {"rows": rows, "total_n": total_n, "total_viol": total_viol,
            "rejects": rejects,
            "verdict": "EQUIV_PASS" if total_viol == 0 else "EQUIV_FAIL"}


def run() -> bool:
    print("=" * 78)
    print("D-1 equivalence verification: T.vcast(round_mode='rint') vs the source")
    print("implicit cross-dtype T.copy cast (both claimed IEEE round-to-nearest-even)")
    print("baseline  = torch.Tensor.to(storage_dtype)      [source semantics]")
    print("candidate = exact-rational RNE (fractions.Fraction, definition-based)")
    print("=" * 78)
    ok = True
    for dtype in ("float16", "bfloat16"):
        res = _run_dtype(dtype)
        print(f"\n--- {dtype} ---")
        print(f"{'case':<28}{'n':>8}{'max_ulp':>10}{'viol':>8}   verdict")
        for r in res["rows"]:
            print(f"{r['case']:<28}{r['n']:>8}{r['ulp']:>10}{r['viol']:>8}   {r['verdict']}")
        print(f"TOTAL n={res['total_n']} violations={res['total_viol']} -> {res['verdict']}")
        print("negative controls (rejected modes, must be >0): "
              f"round(half-away)={res['rejects']['round']} diff, "
              f"trunc={res['rejects']['trunc']} diff")
        ok &= res["verdict"] == "EQUIV_PASS"
        ok &= res["rejects"]["round"] > 0 and res["rejects"]["trunc"] > 0
    print("\n" + "=" * 78)
    print("SUMMARY:", "ALL_EQUIV_PASS" if ok else "EQUIV_FAIL")
    print("=" * 78)
    return ok


def main() -> int:
    return 0 if run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
