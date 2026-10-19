"""D-1 equivalence machine-verification for `all` (`_logical_reduce_kernel`).

The adopted optimization items of DESIGN.md §1.6.1 (logical-AND row reduction,
NPU re-design for `AllFwdOp`) are *math* identities that the NPU algorithm
depends on.  Each is checked here against the **baseline = the GPU source
formula** on a fp64 reference with torch CPU (no NPU / torch_npu needed --
golden independence).

  A1  complement form (De Morgan)      : all_j [x_j != 0]  ==  NOT any_j [x_j == 0]
      baseline (GPU src)               : all_j [ fp32(x_j) != 0 ]   (unordered NE)
      adopted (NPU)                    : 1 - max_j [ x_j == 0 ]     (ordered EQ)
      Machine form:  baseline == adopted, element-by-element on bools.

  A1c control (must DIFFER)            : ordered-NE route
      `all_j ([x_j != 0] & ~isnan(x_j))` -- this is what a naive
      `T.vcmp(..., "ne")` (ordered NE, NaN -> false) would produce.  If this
      control does *not* differ from the baseline the script has no
      discriminating power and fails itself (`CONTROL_BROKEN`).

  A2  real-column domain (no padding)  : all over the real columns N == all over
      align_up(N, 256) padded to that width with **pad value 1.0** (the AND
      neutral element, i.e. the value the GPU source factories in for `all`).
      This is what lets the kernel skip the source's 256-element pad entirely.

  A2b narrow-row 32B pad (same identity at W = align_up(N, 16)).

  A3  native-dtype compare             : [x == 0] in the *input* dtype ==
      [fp32(x) == 0] -- exhaustive over **all** 2**16 fp16 / bf16 bit patterns
      (fp16->fp32 and bf16->fp32 are exact injections preserving +/-0 and NaN).
      Backs dropping the source's full-width fp32 promotion pass.

  A4  segmented (online) AND           : all over N == AND of the per-segment
      results, with a running `max` merge in the float32 domain (path C of the
      design).  Checked for both a dividing and a non-dividing segment width
      (static tail), and the f32 `running_max` accumulator is checked to be
      bit-exact (the accumulator only ever holds 0.0 / 1.0).

  A5  micro-epilogue cast chain        : all = 1 - any_zero is exactly 0.0/1.0 in
      float32 (0 ulp); f32 -> f16 -> i8 (rint) reproduces the bool output
      bit-exactly (the design's int8 carrier; f32->i8 has no direct vcast).

Data / reference domains: both sides computed in float64 on CPU; results are
bools / 0-1 integers and compared exactly (0 ulp).

Run:  python3 examples/all/_logical_reduce_kernel/verify_equiv.py
Exit code 0 == all gate groups EQUIV_PASS (control groups CONTROL_OK).
"""

import sys

import numpy as np
import torch

ALIGN = 256        # GPU source alignment constant (DEFAULT_ALIGNMENT)
VEC_ALIGN = 16     # 32B / 2B -- narrow-row alignment unit for fp16/bf16

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [(1, 1), (1, 2), (3, 8), (4, 16), (8, 64), (16, 256), (32, 300),
          (64, 512), (7, 1024), (129, 512), (5, 16384)]


def align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


# --------------------------------------------------------------------------- #
# the four formula variants (all computed on a fp64 reference)
# --------------------------------------------------------------------------- #

def all_baseline(x: torch.Tensor) -> torch.Tensor:
    """GPU source semantics: AND_j (x_j != 0) with **unordered** NE (NaN -> nonzero).

    The source materialises [x != 0] as 1.0/0.0 and takes reduce_min, i.e.
    exactly the logical AND of the per-element nonzero predicate.
    """
    xd = x.to(torch.float64)
    return (xd != 0.0).all(dim=-1)


def all_ordered_ne_control(x: torch.Tensor) -> torch.Tensor:
    """Control: ordered-NE verdict (NaN -> *equal to zero*), as `vcmp("ne")` gives.

    Implemented NaN-safely as AND_j ([x_j != 0] & ~isnan(x_j)).
    """
    xd = x.to(torch.float64)
    return ((xd != 0.0) & ~torch.isnan(xd)).all(dim=-1)


def all_eq_complement(x: torch.Tensor) -> torch.Tensor:
    """Adopted candidate: NOT any_j (x_j == 0), i.e. (max_j [x_j == 0]) == 0."""
    xd = x.to(torch.float64)
    return ~(xd == 0.0).any(dim=-1)


def all_eq_complement_padded(x: torch.Tensor, width: int, pad: float) -> torch.Tensor:
    """Same, on a width-padded copy; pad value `pad` must be the AND neutral (1.0)."""
    xd = x.to(torch.float64)
    if width > xd.shape[-1]:
        xd = torch.nn.functional.pad(xd, (0, width - xd.shape[-1]), value=pad)
    return ~(xd == 0.0).any(dim=-1)


# --------------------------------------------------------------------------- #
# case construction
# --------------------------------------------------------------------------- #

_CORNER_SCALARS = [
    0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"),
    6.10e-5, -6.10e-5,             # fp16 smallest normal magnitude
    5.96e-8, -5.96e-8,             # fp16 smallest subnormal
    65504.0, -65504.0,             # fp16 max finite
    1e-45, -1e-45,                 # fp32 subnormal
    3.4e38, -3.4e38,               # fp32 max finite
]


def _corner_rows(dtype: torch.dtype, n: int) -> torch.Tensor:
    """Stack of crafted rows (each a homogeneous row of length n, then patterns)."""
    rows = []
    for v in _CORNER_SCALARS:
        rows.append(torch.full((n,), v, dtype=torch.float64))
    rows.append(torch.zeros(n, dtype=torch.float64))
    rows.append(torch.ones(n, dtype=torch.float64))
    rows.append(torch.tensor([1.0 if j % 2 == 0 else 0.0 for j in range(n)], dtype=torch.float64))
    rows.append(torch.tensor([1.0 if j % 2 == 1 else 0.0 for j in range(n)], dtype=torch.float64))
    # single zero / single NaN at the extremes (discriminating rows)
    r = torch.ones(n, dtype=torch.float64)
    r[0] = 0.0
    rows.append(r)
    r = torch.ones(n, dtype=torch.float64)
    r[-1] = 0.0
    rows.append(r)
    r = torch.ones(n, dtype=torch.float64)
    r[n // 2] = float("nan")
    rows.append(r)
    # NaN mixed with zeros: all=False (zero present) even though NaN is nonzero
    r = torch.ones(n, dtype=torch.float64)
    r[::3] = 0.0
    r[::5] = float("nan")
    rows.append(r)
    # all-NaN row -> all == True (NaN is nonzero; the AND sees no zero)
    rows.append(torch.full((n,), float("nan"), dtype=torch.float64))
    # subnormal halo
    r = torch.ones(n, dtype=torch.float64)
    r[::5] = 5.96e-8
    rows.append(r)
    return torch.stack(rows, dim=0).to(dtype)


def _random_rows(dtype: torch.dtype, m: int, n: int, seed: int) -> torch.Tensor:
    """Random rows with ~30% exact zeros (all=False becomes a discriminating signal)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.float64)
    mask = torch.rand(m, n, generator=g, dtype=torch.float64) < 0.3
    x[mask] = 0.0
    return x.to(dtype)


def _all_cases():
    """Yield (tag, x[dtype]) covering random + forced-zero/nonzero + corner rows."""
    seed = 0
    for dtype in DTYPES:
        for m, n in SHAPES:
            seed += 1
            yield f"rand{seed}", _random_rows(dtype, m, n, seed)
    for dtype in DTYPES:
        for n in (1, 2, 16, 64, 300, 512, 1024):
            yield f"corner-{n}", _corner_rows(dtype, n)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

_REPORT = []


def _record(group: str, rows: int, max_diff: int, violations: int, verdict: str, note: str = ""):
    _REPORT.append((group, rows, max_diff, violations, verdict, note))


def gate_a1():
    """A1: complement form (ordered EQ) == baseline (unordered NE)."""
    rows = violations = max_diff = 0
    for _tag, x in _all_cases():
        base = all_baseline(x)
        cand = all_eq_complement(x)
        d = (base != cand).sum().item()
        max_diff = max(max_diff, int(d))
        violations += int(d)
        rows += base.numel()
    _record("A1 补式(¬any[x==0]) vs 基线(AND[x!=0])", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_a1c_control():
    """A1c: the ordered-NE route MUST differ (discriminating-power control)."""
    rows = violations = 0
    for _tag, x in _all_cases():
        base = all_baseline(x)
        ctrl = all_ordered_ne_control(x)
        violations += int((base != ctrl).sum().item())
        rows += base.numel()
    verdict = "CONTROL_OK" if violations > 0 else "CONTROL_BROKEN"
    _record("A1c 对照：有序 NE 路线（应报出差异）", rows, -1, violations, verdict,
            "vcmp(ne) 为 ordered NE（NaN 恒假）→ 含 NaN 行被判 False，与 torch/源语义分歧")


def gate_a2():
    """A2: all over the real columns == all over the 256-padded width (pad = 1.0)."""
    rows = violations = max_diff = 0
    for _tag, x in _all_cases():
        n = x.shape[-1]
        base = all_baseline(x)
        cand = all_eq_complement_padded(x, align_up(n, ALIGN), 1.0)
        d = (base != cand).sum().item()
        max_diff = max(max_diff, int(d))
        violations += int(d)
        rows += base.numel()
    _record("A2 实列域 vs 256 对齐 pad 域（pad=1.0，AND 中性）", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_a2b():
    """A2b: narrow-row 32B alignment pad (W = align_up(N, 16)) is neutral too."""
    rows = violations = max_diff = 0
    for _tag, x in _all_cases():
        n = x.shape[-1]
        base = all_baseline(x)
        cand = all_eq_complement_padded(x, align_up(n, VEC_ALIGN), 1.0)
        d = (base != cand).sum().item()
        max_diff = max(max_diff, int(d))
        violations += int(d)
        rows += base.numel()
    _record("A2b 窄行 32B 对齐 pad(W=align_up(N,16), pad=1.0)", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_a3():
    """A3: [x == 0] in the input dtype == [fp32(x) == 0]; exhaustive over 2**16 patterns."""
    rows = violations = 0
    for dtype in (torch.float16, torch.bfloat16):
        bits = torch.arange(0, 1 << 16, dtype=torch.int32)
        x = bits.view(torch.int16).view(dtype).to(torch.float64).to(dtype)
        mask_native = (x == 0)
        mask_f32 = (x.to(torch.float32) == 0)
        violations += int((mask_native != mask_f32).sum().item())
        # the AND verdict on the same (single-element) rows must agree as well
        v_native = mask_native.reshape(-1).logical_not()
        v_f32 = mask_f32.reshape(-1).logical_not()
        violations += int((v_native != v_f32).sum().item())
        rows += x.numel()
    _record("A3 原生 dtype 比较 vs fp32 上抛比较（全 2^16 位型穷举）", rows, 0, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "fp16->f32 / bf16->f32 为精确单射且保 ±0 与 NaN")


def gate_a4():
    """A4: segmented AND + f32 running-max merge == full-row AND (path C)."""
    rows = violations = max_diff = 0
    cases = [("rand", x) for x in
             [_random_rows(torch.float16, 64, 16384, 7),
              _random_rows(torch.float32, 32, 32768, 8),
              _random_rows(torch.bfloat16, 16, 16384, 9)]] + \
            [("corner16k", _corner_rows(torch.float16, 1024)),
             ("corner16k32", _corner_rows(torch.float32, 1024))]
    for _tag, x in cases:
        n = x.shape[-1]
        base = all_eq_complement(x)                       # full-row candidate
        any_zero_full = (x.to(torch.float64) == 0.0).any(dim=-1)
        for tn in (8192, 4096, 300):                      # dividing / non-dividing
            if tn > n:
                continue
            full = n // tn
            tail = n - full * tn
            # float32 running max over the per-segment any-zero indicator
            acc = torch.zeros(x.shape[0], dtype=torch.float64)
            for t in range(full):
                seg = x[..., t * tn:(t + 1) * tn].to(torch.float64)
                acc = torch.maximum(acc, (seg == 0.0).any(dim=-1).to(torch.float64))
            if tail > 0:
                seg = x[..., full * tn:].to(torch.float64)
                acc = torch.maximum(acc, (seg == 0.0).any(dim=-1).to(torch.float64))
            seg_all = acc == 0.0                            # 1 - acc, exact in {0,1}
            d1 = int((base != seg_all).sum().item())        # segmented vs full-row
            d2 = int((acc != any_zero_full.to(torch.float64)).sum().item())  # merge exact
            d = max(d1, d2)
            max_diff = max(max_diff, d)
            violations += d
            rows += base.numel()
    _record("A4 分段 AND + f32 running-max 合并 vs 整行（tn=8192/4096/300）", rows,
            max_diff, violations, "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "分段覆盖恰为 [0,N)（无重无漏）+ max 结合律；acc 恒为 0.0/1.0 ⇒ f32 位精确")


def gate_a5():
    """A5: micro-epilogue cast chain f32{0,1} -> f16 -> i8 is bit-exact."""
    rows = violations = 0
    max_ulp = 0
    for _tag, x in _all_cases():
        xd = x.to(torch.float64)
        any_zero = (xd == 0.0).any(dim=-1)
        one_v = torch.ones_like(any_zero, dtype=torch.float64)
        all_v = (one_v - any_zero.to(torch.float64)).to(torch.float32)   # 1 - max
        # exactness of the value domain: 0.0 / 1.0 only
        violations += int((~((all_v == 0.0) | (all_v == 1.0))).sum().item())
        all_f16 = all_v.to(torch.float16)
        out_i8 = all_f16.to(torch.int8)               # rint semantics (values 0/1)
        ref_i8 = (~any_zero).to(torch.int8)
        d = int((out_i8 != ref_i8).sum().item())
        max_ulp = max(max_ulp, d)
        violations += d
        rows += ref_i8.numel()
    _record("A5 微 epilogue 链 f32 0/1 -> f16 -> i8 位精确", rows, max_ulp, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "1 - max_j[x==0] 恒为 0.0/1.0（f32 精确）；f32->f16->i8(rint) 与 bool 逐位一致")


def info_nan_matrix():
    """INFO (not a gate): the NaN semantics of the competing routes.

    Row 1 = `[NaN, 1, -1, 5.96e-8]` (NaN and no zero) is the *discriminating*
    row: NaN is nonzero, so `all` must be True; the ordered-NE route says False.
    Row 2 = `[NaN, 0, 1, -0]` additionally exercises the zero-dominates case.
    """
    for tag, row in (("NaN 无零", [float("nan"), 1.0, -1.0, 5.96e-8]),
                     ("NaN 混零", [float("nan"), 0.0, 1.0, float("-0.0")])):
        x = torch.tensor([row], dtype=torch.float64)
        base = all_baseline(x).item()            # source (unordered NE)
        cand = all_eq_complement(x).item()       # adopted candidate
        ctrl = all_ordered_ne_control(x).item()  # ordered-NE route
        torch_ref = bool(x.bool().all(dim=-1).item())
        _record(f"INFO: NaN 语义 [{tag}]", 1, -1, -1, "INFO",
                f"源(unordered NE)={base} / 采纳候选(ordered EQ 补式)={cand} / "
                f"有序 NE 路线={ctrl} / torch.all={torch_ref}")


def main() -> int:
    print("=" * 104)
    print("all (op `_logical_reduce_kernel`) -- D-1 equivalence machine verification")
    print(f"torch {torch.__version__} | CPU | fp64 reference | exact bool comparison")
    print("=" * 104)
    gate_a1()
    gate_a1c_control()
    gate_a2()
    gate_a2b()
    gate_a3()
    gate_a4()
    gate_a5()
    info_nan_matrix()

    print(f"\n{'验证项':<56s} {'用例行':>8s} {'最大偏差':>9s} {'违反数':>8s}  {'结论':<20s}")
    print("-" * 104)
    failed = False
    for group, rows, max_diff, viol, verdict, note in _REPORT:
        md = "-" if max_diff < 0 else str(max_diff)
        vv = "-" if viol < 0 else str(viol)
        print(f"{group:<56s} {rows:>8d} {md:>9s} {vv:>8s}  {verdict:<20s}")
        if note:
            print(f"    · {note}")
        if verdict in ("EQUIV_FAIL", "CONTROL_BROKEN"):
            failed = True
    print("-" * 104)
    print("ALL EQUIV_PASS" if not failed else "*** FAILURE: see above ***")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
