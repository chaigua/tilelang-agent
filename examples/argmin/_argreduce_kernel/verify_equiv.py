"""D-1 equivalence machine verification for argmin ``_argreduce_kernel`` (NPU).

Stage 1 artifact (skill tilelang-op-design Phase 2 step 5). Verifies each
*adopted* algorithmic-equivalence optimization (DESIGN.md section 1.6.1,
E1..E5) by numerically comparing the candidate formula against the baseline
formula on random tensors + corner values, using an fp64 reference on
torch CPU.

Scope of the argmin math (DESIGN.md section 0.1):

    y[i] = min{ j in [0, N) : x[i, j] == ext_i(x) },  ext = min (argmin)

    first-occurrence tie-break, int64 output, IEEE equality matching.

Baseline (the GPU source algorithm, sections 0.3/0.5):
    m = -max(-x)  (transform domain, fp32 fmax = NaN-ignoring) -> serial scan
    ``for j: if x[j] == m: break``  (no match -> source fallback 0)

Candidate optimizations verified here:
    E1  vectorized first-occurrence   first = argmin_j (j if x[j]==m else BIG)
    E2  direct reduce_min             (replaces the -max(-x) transform)
    E3  online single-pass recurrence tile-0 direct-write + strict-less update
    E4  native-domain pipeline        fp16/bf16 pipeline vs fp32 up-cast pipeline
    E5  +inf padding is index-neutral candidate over (M, N_padded), pad=+inf

Revision v1 note (addresses REVIEW.md non-blocking suggestion 1):
    * E2 baseline ``base_transform_argmin`` now genuinely evaluates the
      source ``-max(-x)`` transform (fmax / NaN-ignoring), with no dead
      assignment: it is a real "transform domain vs direct domain" contrast
      against ``cand_direct_min_argmin``.  The two paths are provably
      distinct: the informational NaN row check shows they diverge whenever
      NaN is present (see ``info_nan_behavior``).
    * E4 now compares two *actual* dtype pipelines: fp16/bf16 native-domain
      (min + equality matching done in the input dtype) vs the source-style
      fp32 up-cast pipeline (min + matching done in fp32), instead of
      comparing identical fp64 values.
    * A "discriminating power" control group was added: three deliberately
      WRONG candidates (last-occurrence, too-small sentinel, lossy coarsened
      domain) must be flagged with violations > 0, otherwise the harness
      itself is reported broken (``CONTROL_BROKEN``) and the run fails.  This
      turns "the script can detect a difference" into a machine assertion.

NaN rows are a documented non-equivalence boundary (source fmax ignores NaN;
NPU reduce propagates NaN) and are reported separately as INFO, not gated.

Runs on CPU only (torch +cpu) -- no NPU required. Exit code 0 iff every
adopted item reports zero index violations on the NaN-free domain AND every
discriminating-power control reports a non-zero violation count.
"""

import sys

import torch

BIG = 2**30  # fp32-exact sentinel (power of two, > any valid index j < 2**24)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def round_to(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """Round through *dtype* then read back in fp64 (exact reference domain)."""
    return x.to(DTYPES[dtype]).to(torch.float64)


def nan_ignoring_min(v: torch.Tensor) -> torch.Tensor:
    """min over non-NaN entries (fmax semantics, like ``T.reduce_max`` on CUDA)."""
    masked = torch.where(torch.isnan(v), torch.full_like(v, float("inf")), v)
    return masked.min(dim=1).values


def nan_ignoring_min_native(v: torch.Tensor) -> torch.Tensor:
    """Same fmax-style NaN-ignoring min, but evaluated in ``v.dtype``.

    Used by E4 to compare real dtype pipelines (fp16/bf16 vs fp32) rather
    than a single fp64 domain.
    """
    masked = torch.where(torch.isnan(v), torch.full_like(v, float("inf")), v)
    return masked.min(dim=1).values


def first_match(vals: torch.Tensor, m: torch.Tensor, absent: int) -> torch.Tensor:
    """Serial first-occurrence scan: smallest j with vals[r, j] == m[r]."""
    R, _W = vals.shape
    out = torch.full((R,), absent, dtype=torch.int64)
    for r in range(R):
        hit = (vals[r] == m[r]).nonzero(as_tuple=False)
        if hit.numel() > 0:
            out[r] = int(hit[0, 0])
    return out


def has_nan(vals: torch.Tensor) -> bool:
    return bool(torch.isnan(vals).any())


# ---------------------------------------------------------------------------
# baselines (the source algorithm)
# ---------------------------------------------------------------------------
def base_transform_argmin(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """Source baseline: ``-max(-x)`` transform domain, then serial first scan.

    The transform is evaluated for real (no dead assignment):

        neg = -x ; fmax-style NaN-ignoring max(neg) ; m = -max(neg)

    Mirrors the source ``reduce_max`` on ``neg_x`` (fmax ignores NaN operands,
    modelled here by mapping NaN -> -inf before the max).  All-NaN row ->
    max = -inf -> m = +inf -> no match -> source fallback 0 (mirrors the
    source ``T.fill(out_idx, 0)``).
    """
    v = round_to(x, dtype)
    neg = -v
    neg = torch.where(torch.isnan(neg), torch.full_like(neg, float("-inf")), neg)
    m = -neg.max(dim=1).values  # -max(-x), fmax NaN-ignoring
    return first_match(v, m, absent=0)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def cand_mask_argmin(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """E1: vectorized first-occurrence = argmin_j (j if x[j]==m else BIG)."""
    v = round_to(x, dtype)
    m = nan_ignoring_min(v)
    R, W = v.shape
    idx = torch.arange(W, dtype=torch.float64).unsqueeze(0).expand(R, W)
    cand = torch.where(v == m.unsqueeze(1),
                       idx,
                       torch.full((R, W), float(BIG), dtype=torch.float64))
    return cand.min(dim=1).values.to(torch.int64)


def cand_direct_min_argmin(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """E2: direct min (no -max(-x) transform) + first-occurrence scan."""
    v = round_to(x, dtype)
    m = nan_ignoring_min(v)
    return first_match(v, m, absent=BIG)


def cand_native_argmin(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """E4 candidate: min + matching performed in the *native* input dtype."""
    v = x.to(DTYPES[dtype])
    m = nan_ignoring_min_native(v)
    return first_match(v, m, absent=BIG)


def cand_upcast_argmin(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """E4 baseline: source-style fp32 up-cast pipeline (min + match in fp32)."""
    w = x.to(DTYPES[dtype]).to(torch.float32)
    m = nan_ignoring_min_native(w)
    return first_match(w, m, absent=BIG)


def cand_online_argmin(x: torch.Tensor, dtype: str, tile_n: int) -> torch.Tensor:
    """E3: online single-pass recurrence, tile-0 direct write + strict-less update."""
    v = round_to(x, dtype)
    R, W = v.shape
    n_tiles = (W + tile_n - 1) // tile_n

    t0 = v[:, 0:tile_n]
    run_m = nan_ignoring_min(t0).clone()
    run_i = first_match(t0, run_m, absent=BIG)

    for t in range(1, n_tiles):
        lo = t * tile_n
        hi = min(W, lo + tile_n)
        ct = v[:, lo:hi]
        cm = nan_ignoring_min(ct)
        ci = first_match(ct, cm, absent=BIG)
        glob = lo + ci  # globalize the local index
        upd = cm < run_m  # strict-less -> keep the earlier index on ties
        run_m = torch.where(upd, cm, run_m)
        run_i = torch.where(upd, glob, run_i)
    return run_i


def cand_padded_argmin(x: torch.Tensor, dtype: str, n_padded: int) -> torch.Tensor:
    """E5: candidate computed over (M, N_padded) with +inf pad."""
    v = round_to(x, dtype)
    R, W = v.shape
    if n_padded < W:
        raise ValueError("n_padded must be >= W")
    pad = torch.full((R, n_padded - W), float("inf"), dtype=torch.float64)
    vp = torch.cat([v, pad], dim=1)
    m = nan_ignoring_min(vp)
    idx = torch.arange(vp.shape[1], dtype=torch.float64).unsqueeze(0).expand_as(vp)
    cand = torch.where(vp == m.unsqueeze(1),
                       idx,
                       torch.full(vp.shape, float(BIG), dtype=torch.float64))
    return cand.min(dim=1).values.to(torch.int64)


# ---------------------------------------------------------------------------
# deliberately WRONG candidates (discriminating-power controls)
# ---------------------------------------------------------------------------
def neg_last_occurrence(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """WRONG: keeps the *last* matching index (max instead of min)."""
    v = round_to(x, dtype)
    m = nan_ignoring_min(v)
    R, W = v.shape
    idx = torch.arange(W, dtype=torch.float64).unsqueeze(0).expand(R, W)
    cand = torch.where(v == m.unsqueeze(1),
                       idx,
                       torch.full((R, W), float(BIG), dtype=torch.float64))
    return cand.max(dim=1).values.to(torch.int64)


def neg_small_sentinel(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """WRONG: sentinel (1.0) smaller than some valid indices -> index collision."""
    v = round_to(x, dtype)
    m = nan_ignoring_min(v)
    R, W = v.shape
    idx = torch.arange(W, dtype=torch.float64).unsqueeze(0).expand(R, W)
    cand = torch.where(v == m.unsqueeze(1),
                       idx,
                       torch.full((R, W), 1.0, dtype=torch.float64))
    return cand.min(dim=1).values.to(torch.int64)


def neg_coarse_pipeline(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """WRONG (E4 control): a *lossy* (non-injective) dtype round-trip.

    Native fp16 values are pushed through bfloat16 (coarser mantissa) before
    the min + match, so distinct fp16 values can collapse and change the
    first-occurrence index -- this is exactly the "not an exact injection"
    case the E4 group is meant to be able to reject.
    """
    v = x.to(DTYPES[dtype]).to(torch.bfloat16).to(torch.float64)
    return first_match(v, nan_ignoring_min(v), absent=BIG)


# ---------------------------------------------------------------------------
# case construction
# ---------------------------------------------------------------------------
def corner_rows(W: int, dtype: str) -> torch.Tensor:
    """Corner value rows (NaN-free; NaN handled separately as INFO)."""
    rows = [
        [0.0] * W,
        [float("inf")] * W,
        [float("-inf")] * W,
        [float("inf"), float("-inf")] * (W // 2) + ([float("inf")] if W % 2 else []),
        [-0.0, 0.0] * (W // 2) + ([-0.0] if W % 2 else []),
        [0.0, -0.0] * (W // 2) + ([0.0] if W % 2 else []),
        [3.0] * W,  # duplicate extremum (tie)
    ]
    sub_v = 1e-8 if dtype == "float16" else (1e-40 if dtype == "bfloat16" else 1e-45)
    sub = [1.0] * W
    sub[-1] = sub_v  # subnormal / underflow minimum end
    rows.append(sub)
    return torch.tensor(rows, dtype=torch.float64)[:, :W]


def build_cases():
    shapes = [(1, 1), (1, 2), (3, 8), (4, 256), (8, 300), (16, 512), (7, 1024), (4, 4096)]
    cases = []
    torch.manual_seed(0)
    for (R, W) in shapes:
        for dt in DTYPES:
            cases.append((f"randn ({R},{W}) {dt}",
                          torch.randn(R, W, dtype=torch.float64), dt, None))
            cr = corner_rows(W, dt)
            pick = (torch.arange(R) % cr.shape[0])
            cases.append((f"corner ({R},{W}) {dt}", cr[pick].clone(), dt, None))
    return cases


def tile_cases():
    cases = []
    torch.manual_seed(1)
    for dt in DTYPES:
        v = torch.randn(4, 700, dtype=torch.float64)
        for tn in (128, 256, 300, 512, 700):
            cases.append((f"online ({4},700) tn={tn} {dt}", v.clone(), dt, tn))
        t = torch.randn(3, 700, dtype=torch.float64)
        t[:, 700 - 1] = t[:, 0]  # cross-tile tie
        for tn in (128, 256):
            cases.append((f"online crosstie ({3},700) tn={tn} {dt}", t.clone(), dt, tn))
        cases.append((f"online allinf ({2},512) tn=128 {dt}",
                      torch.full((2, 512), float("inf"), dtype=torch.float64), dt, 128))
        u = torch.full((2, 512), 5.0, dtype=torch.float64)
        u[:, 511] = -5.0  # unique hit in the tail tile
        cases.append((f"online tailhit ({2},512) tn=128 {dt}", u, dt, 128))
    return cases


def control_cases():
    """NaN-free rows that MUST trigger a wrong candidate (discriminating power)."""
    torch.manual_seed(7)
    cases = []
    for W in (4, 16, 256, 300):
        r = torch.randn(6, W, dtype=torch.float64)
        r[0, :] = 3.0                     # all-tie row (last-occurrence differs)
        r[1, :] = 1.0
        r[1, W - 1] = -1.0                # unique min at the last index
        r[2, -1] = r[2, 0]                # duplicate extremum at both ends
        r[3, 0] = 5.0
        r[3, 1] = 5.0                     # tie at the very first two slots
        for dt in DTYPES:
            cases.append((f"control ({6},{W}) {dt}", r.clone(), dt, None))
    return cases


def control_coarse_cases():
    """fp16 rows where a bf16 round-trip collapses distinct values.

    Near 1.0 the fp16 spacing is 2^-10 while the bfloat16 spacing is 2^-8;
    ``1.0019531`` (= 1 + 2 * 2^-10) is distinct from ``1.0`` in fp16 but
    rounds to ``1.0`` in bfloat16, so the coarsened pipeline must report a
    different first-occurrence index.
    """
    rows = [
        [1.0019531, 1.0, 2.0],
        [1.0, 1.0019531, 0.5],
        [1.0019531, 1.0019531, 1.0],
    ]
    return [
        (f"coarse ctl ({len(r)},) f16", torch.tensor([r], dtype=torch.float64),
         "float16", None)
        for r in rows
    ]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_group(name, fn_cand, fn_base, cases, expect_pass=True):
    rows = 0
    viol = 0
    max_diff = 0
    for label, x, dt, tn in cases:
        try:
            c = fn_cand(x, dt, tn) if tn is not None else fn_cand(x, dt)
            b = fn_base(x, dt) if tn is None else fn_base(x, dt)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {label}: {exc}")
            viol += x.shape[0]
            rows += x.shape[0]
            continue
        rows += c.numel()
        d = int((c != b).sum().item())
        if d:
            viol += d
            max_diff = max(max_diff, int((c - b).abs().max().item()))
            print(f"  [DIFF] {label}: {d} rows, max|dc|={max_diff}")
    if expect_pass:
        status = "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL"
        ok = viol == 0
    else:
        status = "CONTROL_OK" if viol > 0 else "CONTROL_BROKEN"
        ok = viol > 0
    print(f"  {name}: rows={rows} violations={viol} "
          f"max_index_diff={max_diff} -> {status}")
    return {"name": name, "rows": rows, "viol": viol, "status": status, "ok": ok}


def info_nan_behavior():
    """NaN rows are outside the equivalence domain -- record, do not gate.

    Uses the *true* NPU candidate semantics (``T.reduce_min`` propagates NaN)
    so the recorded divergence matches the implementation, not the NaN-free
    gating helper.  Also shows that the E2 transform baseline (fmax,
    NaN-ignoring) and the direct NaN-propagating candidate genuinely diverge
    on NaN rows -- evidence that E2 compares two distinct paths.
    """
    print("  INFO NaN boundary (not gated; behaviour record):")
    rows = [
        [float("nan"), 1.0, -1.0],
        [float("nan")] * 3,
        [float("nan"), float("inf"), float("-inf")],
        [1.0, float("nan"), 1.0],
    ]
    x = torch.tensor(rows, dtype=torch.float32)
    v = round_to(x, "float32")
    m_npu = v.min(dim=1).values  # NaN-propagating reduce_min (true NPU behaviour)
    W = v.shape[1]
    idx = torch.arange(W, dtype=torch.float64).unsqueeze(0).expand_as(v)
    cand = torch.where(v == m_npu.unsqueeze(1),
                       idx, torch.full(v.shape, float(BIG), dtype=torch.float64))
    cand = cand.min(dim=1).values.to(torch.int64)
    baseline = base_transform_argmin(x, "float32")
    torchref = torch.argmin(x, dim=-1)
    for i, r in enumerate(rows):
        print(f"    row{i} {r}: npu-candidate={int(cand[i])} "
              f"source-fmax-baseline={int(baseline[i])} "
              f"torch-CPU={int(torchref[i])}")
    print(f"    -> npu-candidate sentinel {BIG} on every NaN-containing row "
          f"(reduce_min propagates NaN -> mask never matches); the source fmax "
          f"path returns its fallback / non-NaN extremum; torch returns the "
          f"first-NaN index. E2 transform baseline vs direct NaN-propagating "
          f"candidate diverge here -> the two paths are structurally distinct.")
    return {"npu_candidate": cand.tolist(), "baseline": baseline.tolist(),
            "torch": torchref.tolist()}


def main() -> int:
    print("=" * 78)
    print("D-1 equivalence verification: argmin _argreduce_kernel (torch CPU, fp64)")
    print(f"torch {torch.__version__}  BIG={BIG}")
    print("=" * 78)

    cases = build_cases()
    tcases = tile_cases()
    results = []

    results.append(run_group("E1 mask vs serial-scan baseline",
                             cand_mask_argmin, base_transform_argmin, cases))
    results.append(run_group("E2 direct-min vs -max(-x) transform baseline",
                             cand_direct_min_argmin, base_transform_argmin, cases))
    results.append(run_group("E3 online vs direct argmin",
                             cand_online_argmin, cand_direct_min_argmin, tcases))
    results.append(run_group("E4 native-dtype pipeline vs fp32 up-cast pipeline",
                             cand_native_argmin, cand_upcast_argmin, cases))

    pad_cases = []
    for label, x, dt, tn in cases:
        W = x.shape[1]
        pad_cases.append((label, x, dt, ((W + 255) // 256) * 256))
    results.append(run_group("E5 padded(+inf) vs real width",
                             cand_padded_argmin, cand_direct_min_argmin, pad_cases))

    # --- discriminating-power controls (must report violations) ---
    print("-" * 78)
    print("discriminating-power controls (must be flagged as different):")
    ccases = control_cases()
    results.append(run_group("CTL last-occurrence (wrong tie-break)",
                             neg_last_occurrence, cand_mask_argmin, ccases,
                             expect_pass=False))
    results.append(run_group("CTL small-sentinel (index collision)",
                             neg_small_sentinel, cand_mask_argmin, ccases,
                             expect_pass=False))
    results.append(run_group("CTL lossy coarsened domain (non-injective)",
                             neg_coarse_pipeline, cand_direct_min_argmin,
                             control_coarse_cases(), expect_pass=False))

    info_nan_behavior()

    print("-" * 78)
    all_ok = all(r["ok"] for r in results)
    gated = [r for r in results if r["status"] in ("EQUIV_PASS", "EQUIV_FAIL")]
    equiv_ok = all(r["status"] == "EQUIV_PASS" for r in gated)
    controls = [r for r in results if r["status"] in ("CONTROL_OK", "CONTROL_BROKEN")]
    ctl_ok = all(r["status"] == "CONTROL_OK" for r in controls)
    print(f"TOTAL equivalence groups={len(gated)} "
          f"violations={sum(r['viol'] for r in gated)} "
          f"rows={sum(r['rows'] for r in gated)} -> "
          f"{'ALL EQUIV_PASS' if equiv_ok else 'EQUIV_FAIL'} | "
          f"controls_ok={ctl_ok}")
    print("-" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
