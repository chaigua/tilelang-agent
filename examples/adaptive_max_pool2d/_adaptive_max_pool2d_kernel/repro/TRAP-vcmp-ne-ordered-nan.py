# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: ``T.vcmp(..., "ne")`` is an *ordered* not-equal on npuir -- it
returns false whenever either operand is NaN, so it cannot probe NaN.
``"eq"`` / ``"le"`` / ``"ge"`` are IEEE-correct.

Entry: traps-compiler.md ``TRAP-vcmp-ne-ordered-nan``.  Single experience
(one phenomenon, one file): the cmp_mod NaN behaviour.

Phenomenon (asserted below, on NPU):
  * ``vcmp(x, x, "ne")`` on rows containing NaN returns 0 at *every* lane,
    the NaN lanes included -- i.e. it behaves like an ordered not-equal
    (``lt | gt``), not like IEEE ``!=`` (true for NaN operands);
  * ``vcmp(x, x, "eq")`` is IEEE-correct: 1 for numbers, 0 at NaN lanes;
  * ``"ne"`` works normally for NaN-free operand pairs (ordered, not broken),
    so the trap only bites when the intent is NaN probing.

Sanctioned form (asserted below): probe with the IEEE-correct equality mode
and invert it.  Per row materialise ``eq = (x == x)`` as an fp32 {0,1} mask,
multiply it into a flag accumulator initialised to 1 (0 absorbing, 1 neutral);
``has_nan = (flag == 0)``.  This is the form used by
``_adaptive_max_pool2d_kernel`` (general grid; the C4 whole-plane path reduces
the same eq mask with ``T.reduce_min``).

Why it matters: the adaptive-max kernel needs "a NaN anywhere in the window
wins".  With the ``"ne"`` probe the flag never fires, so NaN results came only
from ``T.vmax`` / ``T.reduce_max`` NaN propagation -- undocumented behaviour
the DESIGN explicitly forbids relying on.

First verified / toolchain: 2026-10-13, tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5
+ CANN 9.0.0, Ascend910B3 (origin_task: adaptive_max_pool2d-_adaptive_max_pool2d_kernel
Stage 3, first_impl attempt 1).  Re-verification log: -.

Run: /workspace/venv/bin/python repro/TRAP-vcmp-ne-ordered-nan.py
Exit code 0 == trap and sanctioned form both confirmed.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

K, CH = 4, 16


@tilelang.jit(out_idx=[2], target="npuir")
def cmp_rows(K: int, CH: int, mod: str):
    """out[k] = fp32(vcmp(x[k], y[k], mod)); pass x for y to compare a row to itself."""

    @T.prim_func
    def main(
        x: T.Tensor((K, 1, CH), "float16"),
        y: T.Tensor((K, 1, CH), "float16"),
        out: T.Tensor((K, 1, CH), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            ra = T.alloc_ub((1, CH), "float16")
            rb = T.alloc_ub((1, CH), "float16")
            a32 = T.alloc_ub((1, CH), "float32")
            b32 = T.alloc_ub((1, CH), "float32")
            cond = T.alloc_ub((1, CH), "bool")
            c32 = T.alloc_ub((1, CH), "float32")
            for k in T.serial(K):
                T.copy(x[k, 0:1, 0:CH], ra[0:1, 0:CH])
                T.copy(y[k, 0:1, 0:CH], rb[0:1, 0:CH])
                T.vcast(ra, a32, round_mode="rint")
                T.vcast(rb, b32, round_mode="rint")
                T.vcmp(a32, b32, cond, mod)
                T.vcast(cond, c32, round_mode="rint")
                T.copy(c32, out[k, 0:1, 0:CH])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def flag_accum(K: int, CH: int):
    """Sanctioned NaN probe: per-row eq mask, product accumulation (init 1)."""

    @T.prim_func
    def main(x: T.Tensor((K, 1, CH), "float16"), out: T.Tensor((1, CH), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            row = T.alloc_ub((1, CH), "float16")
            row32 = T.alloc_ub((1, CH), "float32")
            cond = T.alloc_ub((1, CH), "bool")
            eq32 = T.alloc_ub((1, CH), "float32")
            flag = T.alloc_ub((1, CH), "float32")
            one = 1.0
            T.vbrc(one, flag)
            for k in T.serial(K):
                T.copy(x[k, 0:1, 0:CH], row[0:1, 0:CH])
                T.vcast(row, row32, round_mode="rint")
                T.vcmp(row32, row32, cond, "eq")
                T.vcast(cond, eq32, round_mode="rint")
                T.vmul(flag, eq32, flag)
            T.copy(flag, out)

    return main


x = torch.randn(K, 1, CH, dtype=torch.float16)
x[1, 0, 3] = float("nan")
x[3, 0, 11] = float("nan")
is_nan = torch.isnan(x.reshape(K, CH))

ne = cmp_rows(K, CH, "ne")(x.npu(), x.npu()).cpu().reshape(K, CH) > 0
eq = cmp_rows(K, CH, "eq")(x.npu(), x.npu()).cpu().reshape(K, CH) > 0
print(
    "nan lanes per row      :",
    [is_nan[k].nonzero().flatten().tolist() for k in range(K)],
)
print(
    "ne self-compare is-nan :", [ne[k].nonzero().flatten().tolist() for k in range(K)]
)
print(
    "eq self-compare is-nan :",
    [(~eq[k]).nonzero().flatten().tolist() for k in range(K)],
)
assert not bool(ne.any()), "TRAP NOT reproduced: vcmp 'ne' detected a NaN lane"
assert bool(torch.equal(~eq, is_nan)), "vcmp 'eq' is not the IEEE complement of isnan"

# NaN-free operand pairs degrade to plain ordered "ne" (not a broken op).
z = torch.randn(K, 1, CH, dtype=torch.float16)
ne2 = cmp_rows(K, CH, "ne")(z.npu(), (z + 1).npu()).cpu() > 0
print("ne distinct NaN-free   : all-true =", bool(ne2.all()))
assert bool(ne2.all()), "vcmp 'ne' failed on differing NaN-free operands"

flag = flag_accum(K, CH)(x.npu()).cpu().reshape(CH)
print("eq-mask product flag   :", flag.tolist())
assert bool(torch.equal(flag == 0, is_nan.any(dim=0))), "sanctioned flag form failed"
print("TRAP confirmed (ordered 'ne' cannot probe NaN); sanctioned eq-inversion OK")
