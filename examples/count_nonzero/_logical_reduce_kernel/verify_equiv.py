"""D-1 equivalence machine-verification for count_nonzero (`_logical_reduce_kernel`).

Adopted optimization items of DESIGN.md §1.6.1 (all four are *math* identities
that the NPU re-design depends on) are checked here against the baseline
(= the GPU source formula) with a **fp64 reference** on torch CPU:

  E1  complement form      : count = W - sum_j [x_j == 0]      (ordered EQ)
      baseline (GPU src)   : count = sum_j [x_j != 0]          (unordered NE)
  E1c control (must fail)  : count = sum_j [x_j != 0] restricted to non-NaN
                             (i.e. an *ordered*-NE hardware verdict) -- must be
                             reported as DIFFERENT, otherwise this script has no
                             discriminating power and fails itself.
  E2  fp32 count domain    : float32 accumulation of a 0/1 vector is exact for
                             any length <= 2**24 (all partial sums are exactly
                             representable integers); divergence is shown beyond
                             that bound (this is what the factory assert guards).
  E3  no padding           : sum over the real columns N == sum over
                             align_up(N, 256) with zero padding (the GPU source
                             computes over the padded width with pad value 0.0).
  E4  native-dtype compare : [x == 0] in the *input* dtype == [fp32(x) == 0]
                             (fp16->fp32 and bf16->fp32 are exact injections) --
                             exhaustive over **all** 2**16 fp16/bf16 bit patterns.

Data / reference domains: both sides computed in float64 on CPU; counts are
compared exactly (0 ulp).  No NPU / torch_npu needed (CPU golden independence,
`TRAP-npuir-ref-backend-golden-capability`).

Run:  python3 examples/count_nonzero/_logical_reduce_kernel/verify_equiv.py
Exit code 0 == all gate groups EQUIV_PASS (control groups CONTROL_OK).
"""

import sys
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

ALIGN = 256          # GPU source alignment constant (DEFAULT_ALIGNMENT)
VEC_ALIGN = 16       # 32B / 2B  -- narrow-row 32B alignment unit for fp16/bf16


def align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def count_nonzero_baseline(x: torch.Tensor) -> torch.Tensor:
    """GPU source semantics: sum of (x != 0), i.e. **unordered** NE (NaN -> nonzero)."""
    xd = x.to(torch.float64)
    return (xd != 0.0).sum(dim=-1).to(torch.int64)


def count_nonzero_ordered_ne(x: torch.Tensor) -> torch.Tensor:
    """Control: ordered-NE verdict (NaN -> *not* counted), as an ordered `vcmp "ne"` would give.

    Implemented NaN-safely as (x != 0) & ~isnan(x) -- this is the naive
    "compare against zero with NE" design that §1.6.1-E1 rejects.
    """
    xd = x.to(torch.float64)
    return ((xd != 0.0) & ~torch.isnan(xd)).sum(dim=-1).to(torch.int64)


def count_nonzero_complement(x: torch.Tensor, width: int) -> torch.Tensor:
    """Adopted candidate: width - number of elements *ordered-equal* to zero.

    `x` must already be zero-padded to `width` (padding = +0.0), which is exactly
    the narrow-row alignment padding of the NPU design.
    """
    xd = x.to(torch.float64)
    zc = (xd == 0.0).sum(dim=-1).to(torch.int64)
    return torch.full_like(zc, width) - zc


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
    2.0 ** 24, 2.0 ** 24 + 1.0,    # exact-integer domain boundary
]

_SHAPES = [(1, 1), (1, 2), (3, 8), (4, 16), (8, 64), (16, 256), (32, 300),
           (64, 512), (7, 1024), (129, 512), (5, 16384)]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _corner_rows(dtype: torch.dtype, n: int) -> torch.Tensor:
    """A stack of corner rows (each a crafted row of length n)."""
    rows = []
    for i, v in enumerate(_CORNER_SCALARS):
        row = torch.full((n,), v, dtype=torch.float64)
        rows.append(row)
    # all zero / all one / alternating / single nonzero at the extremes
    rows.append(torch.zeros(n, dtype=torch.float64))
    rows.append(torch.ones(n, dtype=torch.float64))
    rows.append(torch.tensor([1.0 if j % 2 == 0 else 0.0 for j in range(n)], dtype=torch.float64))
    rows.append(torch.tensor([1.0 if j % 2 == 1 else 0.0 for j in range(n)], dtype=torch.float64))
    r = torch.zeros(n, dtype=torch.float64)
    r[0] = 1.0
    rows.append(r)
    r = torch.zeros(n, dtype=torch.float64)
    r[-1] = 1.0
    rows.append(r)
    # NaN mixed with zeros (the discriminating row for E1c)
    r = torch.zeros(n, dtype=torch.float64)
    r[::3] = float("nan")
    rows.append(r)
    # subnormal halo
    r = torch.zeros(n, dtype=torch.float64)
    r[::5] = 5.96e-8
    rows.append(r)
    stacked = torch.stack(rows, dim=0)
    return stacked.to(dtype)


def _random_rows(dtype: torch.dtype, m: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=g, dtype=torch.float64)
    # sparsify: 40% exact zeros, to make the count non-trivial
    mask = torch.rand(m, n, generator=g, dtype=torch.float64) < 0.4
    x[mask] = 0.0
    return x.to(dtype)


def _all_cases():
    """Yield (tag, x[dtype]) covering random + forced-zero/nonzero + corner rows."""
    seed = 0
    for dtype in DTYPES:
        for m, n in _SHAPES:
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


def gate_e1():
    """E1: complement form (ordered EQ) == baseline (unordered NE)."""
    rows = 0
    violations = 0
    max_diff = 0
    for tag, x in _all_cases():
        n = x.shape[-1]
        base = count_nonzero_baseline(x)
        cand = count_nonzero_complement(x, n)
        d = (base.long() - cand.long()).abs()
        max_diff = max(max_diff, int(d.max()))
        violations += int((d != 0).sum())
        rows += base.numel()
    _record("E1 补式(W-Σ[x==0]) vs 基线(Σ[x!=0])", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_e1c_control():
    """E1c: the ordered-NE route MUST differ (discriminating-power control)."""
    rows = 0
    violations = 0
    for tag, x in _all_cases():
        base = count_nonzero_baseline(x)
        ctrl = count_nonzero_ordered_ne(x)
        d = (base.long() - ctrl.long()).abs()
        violations += int((d != 0).sum())
        rows += base.numel()
    verdict = "CONTROL_OK" if violations > 0 else "CONTROL_BROKEN"
    _record("E1c 对照：有序 NE 路线（应报出差异）", rows, -1, violations, verdict,
            "有序 NE 把 NaN 判为「零」→ 与 torch/源语义分歧；判别力由本行保证")


def gate_e2():
    """E2: float32 accumulation of a 0/1 vector is exact while the count <= 2**24."""
    rows = 0
    violations = 0
    max_diff = 0
    rng = np.random.default_rng(20261016)
    for n in (1, 256, 4096, 16384, 2 ** 20, 2 ** 23, 2 ** 24):
        for tag in ("ones", "rand", "sparse"):
            if tag == "ones":
                v = np.ones(n, dtype=np.float32)
            elif tag == "rand":
                v = (rng.random(n) < 0.5).astype(np.float32)
            else:
                v = (rng.random(n) < 0.01).astype(np.float32)
            exact = int(v.sum(dtype=np.int64))
            # sequential (the worst case for rounding) via float32 prefix sum,
            # and a pairwise/tree order -- both are legitimate accumulation orders
            seq = np.float32(np.cumsum(v, dtype=np.float32)[-1])
            tree = v.copy()
            while tree.size > 1:
                half = tree.size // 2
                head = tree[:half] + tree[half:2 * half]
                tree = np.concatenate([head, tree[2 * half:]]) if tree.size % 2 else head
            for got in (int(np.float64(seq)), int(np.float64(np.float32(tree[0])))):
                d = abs(got - exact)
                max_diff = max(max_diff, d)
                violations += 1 if d != 0 else 0
            rows += 1
    _record("E2 fp32 计数域精确性 (N <= 2^24)", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "f32 顺序前缀和（cumsum，即逐元素序）+ 树形序两种累加；部分和恒为整数且 <= 2^24 → 精确")

    # boundary demonstration: once the *count* (not just N) exceeds 2**24 the f32
    # accumulator can no longer represent every partial sum -> divergence.
    for n, density, tag in (
        (2 ** 25 + 1, 1.0, "density=1.0 恰奇数 > 2^24（结构上必偏差）"),
        (2 ** 26, 0.5, "density=0.5 期望计数 ~2^25 > 2^24"),
    ):
        if density == 1.0:
            v = np.ones(n, dtype=np.float32)
        else:
            v = (rng.random(n) < density).astype(np.float32)
        exact = int(v.sum(dtype=np.int64))
        seq = np.float32(np.cumsum(v, dtype=np.float32)[-1])
        d = abs(int(np.float64(seq)) - exact)
        _record(f"E2-boundary 计数 > 2^24 应偏差 [{tag}]", 1, d, 1 if d != 0 else 0,
                "BOUNDARY_CONFIRMED" if d != 0 else "BOUNDARY_NOT_REPRODUCED",
                f"n={n} exact={exact} f32seq={int(np.float64(seq))}")


def gate_e3():
    """E3: counting over the real columns == counting over the 256-padded width (pad=0)."""
    rows = 0
    violations = 0
    max_diff = 0
    for tag, x in _all_cases():
        n = x.shape[-1]
        n_padded = align_up(n, ALIGN)
        base_padded = count_nonzero_baseline(
            torch.nn.functional.pad(x.to(torch.float64), (0, n_padded - n)).to(x.dtype)
        )
        base_real = count_nonzero_baseline(x)
        d = (base_padded.long() - base_real.long()).abs()
        max_diff = max(max_diff, int(d.max()))
        violations += int((d != 0).sum())
        rows += base_real.numel()
    _record("E3 实列域计数 vs 256 对齐 pad 域（pad=+0.0）", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL")


def gate_e3b_narrow_pad():
    """E3b: narrow-row 32B alignment padding (W = align_up(N, 16)) is neutral."""
    rows = 0
    violations = 0
    max_diff = 0
    for tag, x in _all_cases():
        n = x.shape[-1]
        w = align_up(n, VEC_ALIGN)
        xp = torch.nn.functional.pad(x.to(torch.float64), (0, w - n)).to(x.dtype)
        base = count_nonzero_baseline(x)                  # real columns only
        cand = count_nonzero_complement(xp, w)            # padded width, complement form
        d = (base.long() - cand.long()).abs()
        max_diff = max(max_diff, int(d.max()))
        violations += int((d != 0).sum())
        rows += base.numel()
    _record("E3b 窄行 32B 对齐 pad(W) 的补式抵消", rows, max_diff, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "pad 区全为 +0.0 → 被 EQ 判为「零」→ 与实列域计数精确抵消")


def gate_e4():
    """E4: [x == 0] in the input dtype == [fp32(x) == 0]; exhaustive over 2**16 patterns."""
    rows = 0
    violations = 0
    for dtype in (torch.float16, torch.bfloat16):
        bits = torch.arange(0, 1 << 16, dtype=torch.int32)
        x = bits.view(torch.int16).view(dtype).to(torch.float64).to(dtype)
        mask_native = (x == 0)
        mask_f32 = (x.to(torch.float32) == 0)
        violations += int((mask_native != mask_f32).sum())
        # count equality on the same rows (all rows have length 1 here)
        c_native = mask_native.to(torch.int64).reshape(-1)
        c_f32 = mask_f32.to(torch.int64).reshape(-1)
        violations += int((c_native != c_f32).sum())
        rows += x.numel()
    _record("E4 原生 dtype 比较 vs fp32 上抛比较（全 2^16 位型穷举）", rows, 0, violations,
            "EQUIV_PASS" if violations == 0 else "EQUIV_FAIL",
            "fp16->f32 / bf16->f32 为精确单射且保 ±0 与 NaN")


def info_nan_matrix():
    """INFO (not a gate): NaN semantics of the three routes."""
    x = torch.tensor([[float("nan"), 0.0, 1.0, float("-0.0")]], dtype=torch.float64)
    base = count_nonzero_baseline(x).item()               # 源 unordered NE
    cand = count_nonzero_complement(x, 4).item()          # 采纳候选
    ctrl = count_nonzero_ordered_ne(x).item()             # 有序 NE 路线
    _record("INFO: NaN 语义（4 元素行 [NaN,0,1,-0]）", 1, -1, -1, "INFO",
            f"源(unordered NE)={base} / 采纳候选(ordered EQ 补式)={cand} / "
            f"有序 NE 路线={ctrl}（torch.count_nonzero=2 → 采纳候选与 torch 及源一致）")


def main() -> int:
    print("=" * 100)
    print("count_nonzero (op `_logical_reduce_kernel`) — D-1 equivalence machine verification")
    print(f"torch {torch.__version__} | CPU | fp64 reference | exact integer comparison")
    print("=" * 100)
    gate_e1()
    gate_e1c_control()
    gate_e2()
    gate_e3()
    gate_e3b_narrow_pad()
    gate_e4()
    info_nan_matrix()

    print(f"\n{'验证项':<52s} {'用例行':>8s} {'最大偏差':>9s} {'违反数':>8s}  {'结论':<22s}")
    print("-" * 100)
    failed = False
    for group, rows, max_diff, viol, verdict, note in _REPORT:
        md = "-" if max_diff < 0 else str(max_diff)
        vv = "-" if viol < 0 else str(viol)
        print(f"{group:<52s} {rows:>8d} {md:>9s} {vv:>8s}  {verdict:<22s}")
        if note:
            print(f"    · {note}")
        if verdict in ("EQUIV_FAIL", "CONTROL_BROKEN", "BOUNDARY_NOT_REPRODUCED"):
            failed = True
    print("-" * 100)
    print("ALL EQUIV_PASS" if not failed else "*** FAILURE: see above ***")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
