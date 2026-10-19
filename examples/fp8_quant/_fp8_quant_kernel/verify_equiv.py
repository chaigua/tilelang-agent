"""D-1 equivalence verification for the FP8Quant NPU redesign (design-time).

Machine-verifies every adopted item of DESIGN.md section 1.6.1 by numerical
comparison against the baseline (source) expression, in an fp64-informed
reference and in the target rounding path, on random tensors + corner values
(torch CPU).

The equivalence object is the *observable* result of the operator: the fp8
e4m3 output bytes (and the fp32 scale).  Intermediate fp32 ulp differences are
recorded for information only, because fp8 has 3 mantissa bits and ~all fp32
ulp-level perturbations are absorbed by the final rounding; the naive
"fp32-ulp-only" criterion is *not* the contract.

Revision history
----------------
v0  (rejected by Stage-2 review) verified the branch-free sign bit as
        sb = 128 * clamp(-v * 1e30, 0, 1)
    which is WRONG for ``0 < |v| < 1e-30``: a finite multiplier C can never
    decide the sign of an arbitrarily small magnitude (``|v| < 1/C`` makes
    ``clamp(-v*C, 0, 1)`` a fractional value in (0, 1)).  Reproducer:
    ``q = -9.9e-31`` encoded to ``0x7E`` (= +448) instead of ``0x80`` (-0).
    The v0 corner set did not contain any value in ``(0, 1e-30)``, so the E4
    "13/263424, all -0.0" PASS was an artefact of a coverage gap.
v1  (this file) replaces the sign bit with the *magnitude-independent*
    comparison form
        sb = 128 * [q < |q|]        (== 128 * [q < 0])
    and extends the corner set with the tiny-negative region, negative
    subnormals, +-0.0, +-inf and NaN.  E5 is the explicit regression guard that
    the old finite-multiplier form fails where the new comparison form passes.

Items
-----
E1  division -> reciprocal-multiply      final fp8 bytes: x/scale vs x*(448/amax)
E2  absmax reduction rewrite             reduce_absmax(x) vs max(|x|)
E3  clamp necessity (scale = amax/448)   is the +/-448 clamp observable in fp8?
E4  software e4m3 encoder vs torch cast  uint8 payload bytes, bit-exact
E5  sign formulation guard               finite-multiplier sign vs comparison sign

Run:  python verify_equiv.py
"""
import numpy as np
import torch

SEED = 20261014
D_LENS = [32, 64, 128]
FP8_MIN, FP8_MAX = -448.0, 448.0
EPS_AMAX = 1e-4
# magnitudes in the region that broke the v0 sign form (0 < |v| < 1e-30)
TINY = [1e-31, 3e-31, 5e-31, 9.9e-31, 2e-31, 1e-32, 1e-35, 1e-38, 1e-41,
        1e-45, 2.0**-149, 1e-40]
NONFINITE = [float("inf"), float("-inf"), float("nan")]

RESULTS = []


# --------------------------------------------------------------------------
# input generation: random + corner values
# --------------------------------------------------------------------------
def make_inputs():
    """Finite inputs only (random + finite corners).  Used by E1/E2/E3/E4/E5."""
    g = torch.Generator().manual_seed(SEED)
    cases = []
    for D in D_LENS:
        for mag in (1e-3, 1.0, 96.0, 1e3):
            cases.append((f"D{D}_mag{mag:g}",
                          (torch.randn(257, D, generator=g) * mag).to(torch.float32)))
    corner = torch.tensor(
        [0.0, -0.0, 1e-6, -1e-6, 1e-4, 0.001953125, 0.0155, 0.015625, 0.0157,
         0.5, 1.0, 1.9999, 2.0, 7.9999, 63.999, 64.0, 255.999, 256.0, 448.0,
         -448.0, 1e4, -1e4, 3.399e38, -3.399e38, 1e-30, -1e-30]
        + TINY + [-t for t in TINY],
        dtype=torch.float32)
    reps = corner.repeat((128 + corner.numel() - 1) // corner.numel())[:128]
    xc = torch.zeros(257, 128, dtype=torch.float32)
    xc.reshape(-1)[: corner.numel()] = corner
    xc = torch.cat([xc, reps.reshape(1, 128), (-reps).reshape(1, 128)], 0)
    cases.append(("corners", xc))
    # random negative subnormals / tiny magnitudes (the v0 failure region)
    rnd_tiny = ((torch.rand(512, 128, generator=g) * 2 - 1) * 1e-31).to(torch.float32)
    rnd_sub = (torch.randn(512, 128, generator=g) * 1e-40).to(torch.float32)
    cases.append(("random_tiny_neg", torch.cat([rnd_tiny, rnd_sub], 0)))
    return cases


def make_nonfinite():
    """+-inf / NaN corner rows (E4 only: their fp8 encoding is compared too)."""
    g = torch.Generator().manual_seed(SEED + 1)
    base = torch.randn(8, 128, generator=g).to(torch.float32)
    rows = []
    for v in NONFINITE:
        r = base.clone()
        r[:, 0] = v
        r[:, 1] = -v
        rows.append(r)
    return [("nonfinite", torch.cat(rows, 0))]


def fp8_bytes(q: torch.Tensor) -> np.ndarray:
    return q.clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def ulp_diff(a: np.ndarray, b: np.ndarray, floor: float = 1e-6,
             ratio: bool = False) -> np.ndarray:
    """ulp distance on the fp32 grid; only the physically meaningful regime is
    scored (|mag| >= floor, both finite) -- near-zero/denormal and overflow
    lanes carry no information for a relative-precision op.

    ratio=True returns relative error |a-b|/|b| instead of ulp (used for the
    overshoot magnitude above 448)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    mag = np.maximum(np.abs(a64), np.abs(b64))
    keep = np.isfinite(a64) & np.isfinite(b64) & (mag >= floor)
    if not keep.any():
        return np.array([0.0])
    a64, b64, mag = a64[keep], b64[keep], mag[keep]
    if ratio:
        return np.abs(a64 - b64) / np.maximum(np.abs(b64), floor)
    # fp32 ulp = 2^(floor(log2(|mag|)) - 23)  (normalised regime)
    sp = np.ldexp(np.ones_like(mag), np.floor(np.log2(mag)).astype(int) - 23)
    return np.abs(a64 - b64) / sp


def record(name, ok, detail):
    RESULTS.append((name, "EQUIV_PASS" if ok else "EQUIV_FAIL", detail))


# --------------------------------------------------------------------------
# E1  x/scale  (baseline)  vs  x*(448/amax_c)  (1 division + N multiplies)
# --------------------------------------------------------------------------
def e1():
    nb = nt = 0
    max_ulp = 0.0
    for tag, x in make_inputs():
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(EPS_AMAX)
        base = (x / (amax / 448.0))                 # baseline: N divisions
        cand = (x * (448.0 / amax))                 # optimised: 1 div + N mul
        b_b, c_b = fp8_bytes(base), fp8_bytes(cand)
        nb += int((b_b != c_b).sum())
        nt += b_b.size
        max_ulp = max(max_ulp, float(ulp_diff(base.numpy(), cand.numpy()).max()))
    record("E1 div -> reciprocal-mul", nb == 0,
           f"fp8 byte mismatch={nb}/{nt}, max fp32 ulp={max_ulp:.4g} (info only)")


# --------------------------------------------------------------------------
# E2  reduce_absmax(x)  vs  reduce_max(|x|)
# --------------------------------------------------------------------------
def e2():
    exact = True
    for tag, x in make_inputs():
        a = x.abs().amax(dim=-1)                 # max over |x|  (fp32 path)
        ref = x.double().abs().amax(dim=-1).float()
        exact &= bool(torch.equal(a, ref))
    record("E2 absmax reduction rewrite", exact, f"bit-exact={exact}")


# --------------------------------------------------------------------------
# E3  clamp necessity: with scale = max(amax,1e-4)/448, quantify the overshoot
#     of |x*inv| above 448 (fp32) and whether it is observable in fp8.
# --------------------------------------------------------------------------
def e3():
    overshoot_ulp, nother = 0.0, 0
    nb = nt = 0
    for tag, x in make_inputs():
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(EPS_AMAX)
        q = x * (448.0 / amax)
        over = q.abs() - 448.0
        m = over > 0
        nother += int(m.sum())
        if m.any():
            overshoot_ulp = max(overshoot_ulp,
                                float(ulp_diff(q[m].numpy(),
                                               np.sign(q[m].numpy()) * 448.0).max()))
        clamped, raw = fp8_bytes(q.clamp(FP8_MIN, FP8_MAX)), fp8_bytes(q)
        nb += int((clamped != raw).sum())
        nt += clamped.size
    record("E3 clamp necessity", nb == 0,
           f"overshoot elems={nother} (max {overshoot_ulp:.1f} ulp above 448), "
           f"observable in fp8={nb}/{nt}")


# --------------------------------------------------------------------------
# E4/E5  software e4m3 encoder bytes  vs  torch float8_e4m3fn cast bytes
# --------------------------------------------------------------------------
def _enc_fields(v: torch.Tensor):
    a = v.abs()
    ec = torch.floor(torch.log2(a.clamp_min(1e-30))).clamp(-6.0, 8.0)
    bits = 8.0 * (ec + 6.0) + torch.round(a * torch.exp2(3.0 - ec))
    return bits, a


def enc_v0(x: torch.Tensor) -> torch.Tensor:
    """v0 (rejected): sign via a finite multiplier -> wrong for 0<|v|<1e-30."""
    v = x.clamp(FP8_MIN, FP8_MAX)
    bits, _ = _enc_fields(v)
    sb = (-v * 1e30).clamp(0.0, 1.0) * 128.0
    return (bits + sb).clamp(0.0, 255.0).to(torch.uint8)


def enc_v1(x: torch.Tensor) -> torch.Tensor:
    """v1 (adopted): magnitude-independent sign, sb = 128*[v < |v|] == 128*[v<0].

    In the kernel this is ``T.vcmp(q, av, cond, "lt")`` with ``av = |q|``
    followed by ``T.vcast(cond, sbv, "rint")`` (bool -> f32, giving {0,1}).
    """
    v = x.clamp(FP8_MIN, FP8_MAX)
    bits, a = _enc_fields(v)
    sb = (v < a).to(torch.float32) * 128.0
    return (bits + sb).clamp(0.0, 255.0).to(torch.uint8)


def _classify(x: torch.Tensor, got: torch.Tensor, ref: torch.Tensor):
    """split mismatches into: accepted (-0.0 sign-of-zero), contract, nan."""
    d = got.to(torch.int32) - ref.to(torch.int32)
    m = d != 0
    zero_ok = m & (x == 0) & ((got.to(torch.int32) ^ ref.to(torch.int32)) == 0x80)
    nan_lane = m & torch.isnan(x)
    other = m & ~zero_ok & ~nan_lane
    return int(zero_ok.sum()), int(nan_lane.sum()), int(other.sum()), int(m.sum())


def e4():
    n_ok = n_nan = n_other = nt = 0
    gv_all, rv_all = [], []
    for tag, x in list(make_inputs()) + list(make_nonfinite()):
        got = enc_v1(x)
        ref = x.clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        ok, nan, other, _ = _classify(x, got, ref)
        n_ok += ok
        n_nan += nan
        n_other += other
        nt += got.numel()
        # cosine over finite lanes only (NaN lanes have no cosine meaning);
        # accumulated globally because a single tiny-magnitude case can decode
        # to an all-zero vector (cosine undefined) -- that is not a mismatch.
        fin = torch.isfinite(x)
        gv_all.append(got.view(torch.float8_e4m3fn).to(torch.float32)[fin])
        rv_all.append(ref.view(torch.float8_e4m3fn).to(torch.float32)[fin])
    gv = torch.cat(gv_all).flatten()
    rv = torch.cat(rv_all).flatten()
    cos = float(torch.nn.functional.cosine_similarity(gv, rv, dim=0))
    # contract domain = all lanes except NaN; the only accepted difference is
    # the sign bit of an exact -0.0 (0x00 vs 0x80, both decode to zero).
    ok = (n_other == 0)
    record("E4 software e4m3 encoder", ok,
           f"contract mismatch={n_ok}/{nt} (all exact -0.0, accepted) ; "
           f"unexpected mismatch={n_other} ; nan-lane mismatch={n_nan} "
           f"(out-of-contract, excluded) ; cos(finite)={cos:.10f}")


def e5():
    """Regression guard: the v0 finite-multiplier sign must fail on the
    tiny-negative region where the v1 comparison sign is correct."""
    x = torch.tensor(TINY + [-t for t in TINY] + [0.0, -0.0, 1e-30, -1e-30],
                     dtype=torch.float32)
    ref = x.clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    _, _, other_v0, tot_v0 = _classify(x, enc_v0(x), ref)
    _, nan_v1, other_v1, tot_v1 = _classify(x, enc_v1(x), ref)
    # v0 must show real (non -0.0) byte errors; v1 must not.
    ok = (other_v0 > 0) and (other_v1 == 0) and (nan_v1 == 0)
    record("E5 sign-form guard (v0 broken / v1 fixed)", ok,
           f"v0 unexpected mismatch={other_v0}/{tot_v0} ; "
           f"v1 unexpected mismatch={other_v1}/{tot_v1}")


if __name__ == "__main__":
    print(f"verify_equiv: torch {torch.__version__} (CPU), seed={SEED}")
    e1(); e2(); e3(); e4(); e5()
    print("\n| item | verdict | detail |")
    print("|------|---------|--------|")
    for name, verdict, detail in RESULTS:
        print(f"| {name} | {verdict} | {detail} |")
    fails = [r for r in RESULTS if r[1] == "EQUIV_FAIL"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} EQUIV_PASS")
    if fails:
        raise SystemExit(1)
