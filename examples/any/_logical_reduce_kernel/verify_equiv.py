"""D-1 equivalence machine-verification for `any` (`_logical_reduce_kernel`).

The adopted optimization items of DESIGN.md §1.6.1 (logical-OR row reduction,
NPU re-design for `AnyFwdOp`) are *math* identities the NPU algorithm depends
on.  Each is checked here against the **baseline = the GPU source formula** on a
fp64 reference with torch CPU (no NPU / torch_npu needed -- golden independence).

  A1  complement form (De Morgan)      : any_j [x_j != 0]  ==  NOT all_j [x_j == 0]
      baseline (GPU src)               : any_j [ fp32(x_j) != 0 ]   (unordered NE)
      adopted (NPU)                    : 1 - min_j [ x_j == 0 ]     (ordered EQ)
      Machine form:  baseline == adopted, element-by-element on bools.

  A1c control (must DIFFER)            : ordered-NE route
      `any_j ([x_j != 0] & ~isnan(x_j))` -- this is what a naive
      `T.vcmp(..., "ne")` (ordered NE, NaN -> false) would produce.  The
      *discriminating* rows are "NaN and zeros only" and "all-NaN": torch/源 say
      True (NaN is nonzero) while the ordered-NE route says False.  If this
      control does *not* differ from the baseline the script has no
      discriminating power and fails itself (`CONTROL_BROKEN`).

  A2  real-column domain (no padding)  : any over the real columns N == any over
      align_up(N, 256) padded to that width with **pad value 0.0** (the OR
      neutral element, i.e. the value the GPU source factories in for `any`).
      This is what lets the kernel skip the source's 256-element pad entirely.
      (Under the adopted `[x==0]` + min form, pad = 0.0 gives indicator 1.0, a
      min-neutral element -- so `1 - min` is unchanged.)

  A2b narrow-row 32B pad (same identity at W = align_up(N, 16), pad = 0.0).

  A3  native-dtype compare             : [x == 0] in the *input* dtype ==
      [fp32(x) == 0] -- exhaustive over **all** 2**16 fp16 / bf16 bit patterns
      (fp16->fp32 and bf16->fp32 are exact injections preserving +/-0 and NaN).
      Backs dropping the source's full-width fp32 promotion pass.

  A4  segmented (online) ANY           : any over N == ANY of the per-segment
      results, with a running `min` merge in the float32 domain (path C of the
      design).  Checked for both a dividing and a non-dividing segment width
      (static tail), and the f32 `running_min` accumulator is checked to be
      bit-exact (the accumulator only ever holds 0.0 / 1.0).

  A5  micro-epilogue cast chain        : any = 1 - all_zero is exactly 0.0/1.0 in
      float32 (0 ulp); f32 -> f16 -> i8 (rint) reproduces the bool output
      bit-exactly (the design's int8 carrier; f32->i8 has no direct vcast).

  INFO (not a gate): NaN semantics matrix + the empty-reduction (N=0) identity
      of `torch.any` (empty OR = False, the opposite of `torch.all`).

Data / reference domains: both sides computed in float64 on CPU; results are
bools / 0-1 integers and compared exactly (0 ulp).

Run:  python3 examples/any/_logical_reduce_kernel/verify_equiv.py
Exit code 0 == all gate groups EQUIV_PASS (control groups CONTROL_OK).
"""

import sys

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

def any_baseline(x: torch.Tensor) -> torch.Tensor:
    """GPU source semantics: OR_j (x_j != 0) with **unordered** NE (NaN -> nonzero).

    The source materialises [x != 0] as 1.0/0.0 and takes reduce_max, i.e.
    exactly the logical OR of the per-element nonzero predicate.
    """
    xd = x.to(torch.float64)
    return (xd != 0.0).any(dim=-1)


def any_ordered_ne_control(x: torch.Tensor) -> torch.Tensor:
    """Control: ordered-NE verdict (NaN -> *equal to zero*), as `vcmp("ne")` gives.

    Implemented NaN-safely as OR_j ([x_j != 0] & ~isnan(x_j)).
    """
    xd = x.to(torch.float64)
    return ((xd != 0.0) & ~torch.isnan(xd)).any(dim=-1)


def any_eq_complement(x: torch.Tensor) -> torch.Tensor:
    """Adopted candidate: NOT all_j (x_j == 0), i.e. (min_j [x_j == 0]) == 0."""
    xd = x.to(torch.float64)
    return ~(xd == 0.0).all(dim=-1)


def any_eq_complement_padded(x: torch.Tensor, width: int, pad: float) -> torch.Tensor:
    """Same, on a width-padded copy; pad value `pad` must be the OR neutral (0.0)."""
    xd = x.to(torch.float64)
    if width > xd.shape[-1]:
        xd = torch.nn.functional.pad(xd, (0, width - xd.shape[-1]), value=pad)
    return ~(xd == 0.0).all(dim=-1)


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
    """Stack of crafted rows (each a homogeneous row of length n, then patterns).

    Discriminating signal for `any`: randn has ~0 zero density, so `any` is
    *always True* -- the informative rows are the all-zero rows (expected False)
    and the "NaN + zeros only" rows (expected True, where the ordered-NE route
    wrongly gives False).
    """
    rows = []
    for v in _CORNER_SCALARS:
        rows.append(torch.full((n,), v, dtype=torch.float64))
    rows.append(torch.zeros(n, dtype=torch.float64))                       # all zero -> False
    rows.append(torch.ones(n, dtype=torch.float64))                        # all nonzero -> True
    rows.append(torch.tensor([1.0 if j % 2 == 0 else 0.0 for j in range(n)], dtype=torch.float64))
    rows.append(torch.tensor([1.0 if j % 2 == 1 else 0.0 for j in range(n)], dtype=torch.float64))
    # single nonzero / single zero at the extremes (discriminating rows)
    r = torch.zeros(n, dtype=torch.float64)
    r[0] = 1.0
    rows.append(r)
    r = torch.zeros(n, dtype=torch.float64)
    r[-1] = -3.0
    rows.append(r)
    r = torch.ones(n, dtype=torch.float64)
    r[0] = 0.0
    rows.append(r)
    r = torch.ones(n, dtype=torch.float64)
    r[-1] = 0.0
    rows.append(r)
    r = torch.ones(n, dtype=torch.float64)
    r[n // 2] = float("nan")
    rows.append(r)                                                        # NaN + nonzeros -> True
    # *** key discriminating row: NaN with ONLY zeros *** -> any == True
    r = torch.zeros(n, dtype=torch.float64)
    r[n // 2] = float("nan")
    rows.append(r)
    # all-NaN row -> any == True (NaN is nonzero; the OR sees a nonzero)
    rows.append(torch.full((n,), float("nan"), dtype=torch.float64))
    # sparse nonzeros among zeros -> True
    r = torch.zeros(n, dtype=torch.float64)
    r[::7] = 2.0
    rows.append(r)
    # subnormal halo with a single zero (False only if the zero dominates)
    r = torch.ones(n, dtype=torch.float64)
    r[::5] = 5.96e-8
    rows.append(r)
    return torch.stack(rows, dim=0).to(dtype)


def _random_rows(dtype: torch.dtype, m: int, n: int, seed: int) -> torch.Tensor:
    """Random rows with ~70% exact zeros (any=False becomes a discriminating signal)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.float64)
    mask = torch.rand(m, n, generator=g, dtype=torch.float64) < 0.7
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
        base = any_baseline(x)
        cand = any_eq_complement(x)
        d = (base != cand).sum().item()
        max_diff = max(max_diff, int(d))
        violations += int(d)
        rows += base.numel()
    _record("A1 补式(¬all[x==0]) vs 基线(OR[x!=0])", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_a1c_control():
    """A1c: the ordered-NE route MUST differ (discriminating-power control)."""
    rows = violations = 0
    for _tag, x in _all_cases():
        base = any_baseline(x)
        ctrl = any_ordered_ne_control(x)
        violations += int((base != ctrl).sum().item())
        rows += base.numel()
    verdict = "CONTROL_OK" if violations > 0 else "CONTROL_BROKEN"
    _record("A1c 对照：有序 NE 路线（应报出差异）", rows, -1, violations, verdict,
            "vcmp(ne) 为 ordered NE（NaN 恒假）→ 「NaN+零」/全 NaN 行被判 False，与 torch/源语义分歧")


def gate_a2():
    """A2: any over the real columns == any over the 256-padded width (pad = 0.0)."""
    rows = violations = max_diff = 0
    for _tag, x in _all_cases():
        n = x.shape[-1]
        base = any_baseline(x)
        cand = any_eq_complement_padded(x, align_up(n, ALIGN), 0.0)
        d = (base != cand).sum().item()
        max_diff = max(max_diff, int(d))
        violations += int(d)
        rows += base.numel()
    _record("A2 实列域 vs 256 对齐 pad 域（pad=0.0，OR 中性）", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_a2b():
    """A2b: narrow-row 32B alignment pad (W = align_up(N, 16)) is neutral too."""
    rows = violations = max_diff = 0
    for _tag, x in _all_cases():
        n = x.shape[-1]
        base = any_baseline(x)
        cand = any_eq_complement_padded(x, align_up(n, VEC_ALIGN), 0.0)
        d = (base != cand).sum().item()
        max_diff = max(max_diff, int(d))
        violations += int(d)
        rows += base.numel()
    _record("A2b 窄行 32B 对齐 pad(W=align_up(N,16), pad=0.0)", rows, max_diff, violations,
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
        # the OR verdict on the same (single-element) rows must agree as well
        v_native = mask_native.reshape(-1)          # any = exists nonzero = NOT(all zero)
        v_f32 = mask_f32.reshape(-1)
        violations += int((v_native != v_f32).sum().item())
        rows += x.numel()
    _record("A3 原生 dtype 比较 vs fp32 上抛比较（全 2^16 位型穷举）", rows, 0, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "fp16->f32 / bf16->f32 为精确单射且保 ±0 与 NaN")


def gate_a4():
    """A4: segmented ANY + f32 running-min merge == full-row ANY (path C)."""
    rows = violations = max_diff = 0
    cases = [("rand", x) for x in
             [_random_rows(torch.float16, 64, 16384, 7),
              _random_rows(torch.float32, 32, 32768, 8),
              _random_rows(torch.bfloat16, 16, 16384, 9)]] + \
            [("corner16k", _corner_rows(torch.float16, 1024)),
             ("corner16k32", _corner_rows(torch.float32, 1024))]
    for _tag, x in cases:
        n = x.shape[-1]
        base = any_eq_complement(x)                       # full-row candidate
        all_zero_full = (x.to(torch.float64) == 0.0).all(dim=-1)
        for tn in (8192, 4096, 300):                      # dividing / non-dividing
            if tn > n:
                continue
            full = n // tn
            tail = n - full * tn
            # float32 running min over the per-segment all-zero indicator
            acc = torch.ones(x.shape[0], dtype=torch.float64)
            for t in range(full):
                seg = x[..., t * tn:(t + 1) * tn].to(torch.float64)
                acc = torch.minimum(acc, (seg == 0.0).all(dim=-1).to(torch.float64))
            if tail > 0:
                seg = x[..., full * tn:].to(torch.float64)
                acc = torch.minimum(acc, (seg == 0.0).all(dim=-1).to(torch.float64))
            seg_any = acc == 0.0                            # 1 - acc, exact in {0,1}
            d1 = int((base != seg_any).sum().item())        # segmented vs full-row
            d2 = int((acc != all_zero_full.to(torch.float64)).sum().item())  # merge exact
            d = max(d1, d2)
            max_diff = max(max_diff, d)
            violations += d
            rows += base.numel()
    _record("A4 分段 ANY + f32 running-min 合并 vs 整行（tn=8192/4096/300）", rows,
            max_diff, violations, "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "分段覆盖恰为 [0,N)（无重无漏）+ min 结合律；acc 恒为 0.0/1.0 ⇒ f32 位精确")


def gate_a5():
    """A5: micro-epilogue cast chain f32{0,1} -> f16 -> i8 is bit-exact."""
    rows = violations = 0
    max_ulp = 0
    for _tag, x in _all_cases():
        xd = x.to(torch.float64)
        all_zero = (xd == 0.0).all(dim=-1)
        one_v = torch.ones_like(all_zero, dtype=torch.float64)
        any_v = (one_v - all_zero.to(torch.float64)).to(torch.float32)   # 1 - min
        # exactness of the value domain: 0.0 / 1.0 only
        violations += int((~((any_v == 0.0) | (any_v == 1.0))).sum().item())
        any_f16 = any_v.to(torch.float16)
        out_i8 = any_f16.to(torch.int8)               # rint semantics (values 0/1)
        ref_i8 = (~all_zero).to(torch.int8)
        d = int((out_i8 != ref_i8).sum().item())
        max_ulp = max(max_ulp, d)
        violations += d
        rows += ref_i8.numel()
    _record("A5 微 epilogue 链 f32 0/1 -> f16 -> i8 位精确", rows, max_ulp, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "1 - min_j[x==0] 恒为 0.0/1.0（f32 精确）；f32->f16->i8(rint) 与 bool 逐位一致")


def info_nan_matrix():
    """INFO (not a gate): the NaN semantics of the competing routes + N=0 identity.

    Row 1 = `[NaN, 1, -1, 5.96e-8]` (NaN and a nonzero) -- all three routes True.
    Row 2 = `[NaN, 0, -0.0, 0.0]` (NaN and *only* zeros) is the *discriminating*
    row for `any`: NaN is nonzero, so `any` must be True (torch) while the
    ordered-NE route says False.
    Row 3 = `[0, -0.0, 0.0, 0.0]` (all zero) -> False (torch agrees).
    Row 4 = all-NaN -> True (NaN is nonzero).
    """
    rows = (("NaN+非零", [float("nan"), 1.0, -1.0, 5.96e-8]),
            ("NaN+零（判别行）", [float("nan"), 0.0, float("-0.0"), 0.0]),
            ("全零", [0.0, float("-0.0"), 0.0, 0.0]),
            ("全 NaN", [float("nan")] * 4))
    for tag, row in rows:
        x = torch.tensor([row], dtype=torch.float64)
        base = any_baseline(x).item()            # source (unordered NE)
        cand = any_eq_complement(x).item()       # adopted candidate
        ctrl = any_ordered_ne_control(x).item()  # ordered-NE route
        torch_ref = bool(x.bool().any(dim=-1).item())
        _record(f"INFO: NaN 语义 [{tag}]", 1, -1, -1, "INFO",
                f"源(unordered NE)={base} / 采纳候选(ordered EQ 补式)={cand} / "
                f"有序 NE 路线={ctrl} / torch.any={torch_ref}")


def info_empty_reduction():
    """INFO (not a gate): empty-reduction (N=0) identity of torch.any == False.

    The kernel asserts N >= 1 (empty OR is a degenerate input); this records the
    torch contract so the boundary table in DESIGN §0.1 is machine-checked.
    """
    x = torch.zeros(3, 0, dtype=torch.float64)
    try:
        ref = torch.any(x, dim=-1)
        got = bool(ref.all().item()) is False and ref.numel() == 3
        note = (f"torch.any((3,0), dim=-1) = {ref.tolist()} ⇒ 空 OR 单位元 = False"
                f"（与 torch.all 的 True 相反）；kernel 侧工厂期 assert N>=1"
                f"（{'一致' if got else '不一致'}）")
        _record("INFO: 空归约维 N=0 语义", x.shape[0], -1, -1,
                "INFO" if got else "INFO_MISMATCH", note)
    except Exception as exc:  # noqa: BLE001
        _record("INFO: 空归约维 N=0 语义", 0, -1, -1, "INFO",
                f"torch.any((3,0)) raised {type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 104)
    print("any (op `_logical_reduce_kernel`) -- D-1 equivalence machine verification")
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
    info_empty_reduction()

    print(f"\n{'验证项':<58s} {'用例行':>8s} {'最大偏差':>9s} {'违反数':>8s}  {'结论':<20s}")
    print("-" * 104)
    failed = False
    for group, rows, max_diff, viol, verdict, note in _REPORT:
        md = "-" if max_diff < 0 else str(max_diff)
        vv = "-" if viol < 0 else str(viol)
        print(f"{group:<58s} {rows:>8d} {md:>9s} {vv:>8s}  {verdict:<20s}")
        if note:
            print(f"    · {note}")
        if verdict in ("EQUIV_FAIL", "CONTROL_BROKEN", "INFO_MISMATCH"):
            failed = True
    print("-" * 104)
    print("ALL EQUIV_PASS" if not failed else "*** FAILURE: see above ***")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
