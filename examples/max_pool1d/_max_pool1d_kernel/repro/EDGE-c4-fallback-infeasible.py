# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""EDGE repro: the windowed-pool "C4 capacity fallback" is **not reachable** --
when a whole-line window candidate exceeds UB, the general-grid path exceeds it
too, so the shape has no UB-feasible plan at all.

Finding (asserted below, device-independent for the tiers and then run on NPU):

  For the C4 whole-line specialisation the general-grid span equals the whole
  input line (``LSPAN == L_in`` by construction: the C4 condition requires the
  kernel to span the line).  The per-element UB cost of the general path is
  ``2*eb + 9`` bytes (row_src + row_dst + row_f32 + eq_mask + eq_f32) while C4
  costs only ``eb + 9`` (flat + flat32 + eq_mask + eq_f32).  Hence

      general_bytes(CH, L_in, BL)  >  global_bytes(CH, L_in)

  whenever ``LSPAN == L_in`` and ``BL*CH*(25+2*eb)`` does not compensate, i.e.
  C4 failing its capacity check implies the general path fails it as well.

  Consequences:
    * DESIGN.md section 8.3 L0-11 expects an oversized C4 candidate (e.g.
      ``(1,8,4096) k4096 s1p0``) to "fall back to the general path".  That
      fallback cannot be served: the honest behaviour is a factory rejection
      ("no UB-feasible plan"), which is what ``_max_pool1d_kernel`` does.
    * The C4b (identity) capacity fallback **is** reachable, because there the
      general-grid span is ``LSPAN = (BL-1)*s + (k-1)*d + 1`` with ``k == 1``
      and ``s == 1``, i.e. ``LSPAN = BL `` (32), independent of ``L_in`` -- so
      ``(1,8,16384) k1 s1p0`` falls back and runs correctly.

  This is a design-math bound, not a compiler bug; it is recorded so that future
  pool migrations do not assume "degenerate path too big -> general path saves
  it".  For op parameter spaces where ``k == L_in`` and ``L_in`` is large, the
  only correct outcome is a documented rejection (or a smaller-CH redesign).

Toolchain: tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388 (commit
417dae9) / torch 2.9.0 / Ascend910B3.
origin_task: max_pool1d-_max_pool1d_kernel Stage 3, first_impl attempt 1.

Run: /workspace/venv/bin/python repro/EDGE-c4-fallback-infeasible.py
Exit code 0 == every assertion below holds.
"""

import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _max_pool1d_kernel import (  # noqa: E402
    _general_bytes,
    _global_bytes,
    _plan,
    pool_output_dim,
)

# ---------------------------------------------------------------------------
# Tier 1: the C4 candidate (1, 8, 4096) k4096 s1 p0 d1, fp16.
# ---------------------------------------------------------------------------
N, C, L_IN, KW, SW, PW, DW = 1, 8, 4096, 4096, 1, 0, 1
out_l = pool_output_dim(L_IN, KW, SW, PW, False, DW)
assert out_l == 1, out_l
c4_condition = (out_l == 1 and PW == 0 and DW == 1 and (KW - 1) * DW + 1 >= L_IN)
assert c4_condition, "premise: the shape must satisfy the C4 sufficient condition"

eb = 2
CH = 8
cap = 196_608
inflate = 1.7
gen = _general_bytes(CH, L_IN, 1, eb) * inflate
glob = _global_bytes(CH, L_IN, eb) * inflate
assert glob > cap, f"C4 must not fit UB: {glob} <= {cap}"
assert gen > cap, f"general must not fit UB either: {gen} <= {cap}"
assert gen > glob, (gen, glob)
print(
    f"[EDGE] C4 (l_in={L_IN}, k={KW}): global={glob:.0f}B > cap; "
    f"general={gen:.0f}B > cap; general > global -> no feasible path"
)

# The factory must reject instead of emitting an overflowing kernel.
try:
    from _max_pool1d_kernel import _max_pool1d_kernel

    _max_pool1d_kernel(N, C, L_IN, KW, SW, PW, DW, False, "float16")
except ValueError as exc:
    assert "no UB-feasible plan" in str(exc), exc
    print(f"[EDGE] oversized C4 candidate rejected as expected: {exc}")
else:  # pragma: no cover
    raise AssertionError("oversized C4 candidate was accepted")

# ---------------------------------------------------------------------------
# Tier 2: the C4b candidate (1, 8, 16384) k1 s1 p0 d1, fp16.
# ---------------------------------------------------------------------------
N, C, L_IN, KW, SW, PW, DW = 1, 8, 16384, 1, 1, 0, 1
plan = _plan(N, C, L_IN, KW, SW, PW, DW, False, "float16")
assert plan["path"] == "general", plan["path"]
assert plan["lspan"] == plan["bl"], (plan["lspan"], plan["bl"])
assert plan["lspan"] < L_IN, plan["lspan"]
print(
    f"[EDGE] C4b fallback reachable: path={plan['path']} BL={plan['bl']} "
    f"LSPAN={plan['lspan']} (general span << L_in)"
)

# Run it end to end on NPU to prove the fallback is correct, not just tagged.
x = torch.randn(N, C, L_IN, dtype=torch.float16)
out = _max_pool1d_kernel(N, C, L_IN, KW, SW, PW, DW, False, "float16")(None)(
    x.npu()
).cpu()
assert torch.equal(out, x), "identity fallback must be bit-exact"
print("[EDGE] C4b -> general fallback ran bit-exactly on NPU")

print("[EDGE] ALL ASSERTIONS HOLD")
