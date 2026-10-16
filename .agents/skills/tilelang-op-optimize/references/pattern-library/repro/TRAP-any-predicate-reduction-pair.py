# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: for ``any`` (logical OR) the predicate mode and the reduction
operator form a *pair* -- ``vcmp("ne")`` + ``reduce_max`` (the natural reading of
"does any non-zero exist") agrees with ``torch.any`` everywhere **except** rows
that hold a NaN and no other non-zero.

Entry (new proposal): traps-compiler.md ``TRAP-any-predicate-reduction-pair``.
Single experience: which row shapes expose a mis-paired predicate/reduction.

Phenomenon (asserted below, run on NPU):
  * correct: ``vcmp("eq")`` -> ``reduce_min`` -> ``1 - min`` == ``torch.any`` on
    all 8 corner rows, NaN rows included (``eq`` is IEEE-correct, so a NaN is
    never counted as a zero and the row stays True);
  * wrong-A: ``vcmp("ne")`` (ordered NE -- false at NaN lanes) -> ``reduce_max``
    differs from the correct result on **exactly** the two NaN rows that carry
    no other non-zero (``NaN + zeros`` and ``all NaN``); it agrees on every
    other row, so a ``randn``-only or zero-free test set cannot expose it;
  * wrong-B: the sibling ``all`` direction (``eq`` -> ``reduce_max`` ->
    ``1 - max``) computes "some zero exists" and differs on every row whose
    verdict is not a fixed point of that inversion.

Why it matters: the ``all`` -> ``any`` migration inverts four things at once
(``max->min``, ``1-max->1-min``, running ``vmax->vmin``, identity ``0.0->1.0``);
wrong-B is the copy of ``all``, wrong-A the "shortest" rewrite, and only wrong-A
is subtle -- its exposure set is exactly the two NaN rows.

First verified / toolchain: 2026-10-16, tilelang
0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba + CANN 9.0.0, Ascend910B3
(origin_task: any-_logical_reduce_kernel Stage 3, first_impl attempt 1).
Re-verification log: -.

Run: /workspace/venv/bin/python repro/TRAP-any-predicate-reduction-pair.py
Exit code 0 == the exposure-set trap is confirmed.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401
from tvm import tir

ROWS, N = 8, 256


def build(cmp_mod, reduce_mode, complement):
    """``cmp_mod`` in {"eq","ne"}; ``reduce_mode`` in {"min","max"}.

    ``complement=True`` adds the ``1 - red`` epilogue: with ``min`` it is the
    ``any`` form ("every element is zero"), with ``max`` it is the sibling
    ``all`` form ("some zero exists").  ``complement=False`` is the direct
    "some non-zero exists" reading, paired with ``ne`` for wrong-A.
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((ROWS, N), "float32"), out: T.Tensor((ROWS,), "int8")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((ROWS, N), "float32")
                z_ub = T.alloc_shared((ROWS, N), "float32")
                mask = T.alloc_shared((ROWS, N), "bool")
                ind = T.alloc_shared((ROWS, N), "float32")
                red = T.alloc_fragment((ROWS, 1), "float32")
                flat = T.alloc_fragment((ROWS,), "float32")
                one = T.alloc_fragment((ROWS,), "float32")
                zero = T.alloc_fragment((ROWS,), "float32")
                dst = T.alloc_fragment((ROWS,), "float32")
                dst16 = T.alloc_fragment((ROWS,), "float16")
                out_ub = T.alloc_shared((ROWS,), "int8")
                T.fill(z_ub, tir.const(0, "float32"))
                T.fill(one, tir.const(1, "float32"))
                T.fill(zero, tir.const(0, "float32"))
                T.copy(x, x_ub)
                T.vcmp(x_ub, z_ub, mask, cmp_mod)
                T.vcast(mask, ind, round_mode="rint")
                if reduce_mode == "min":
                    T.reduce_min(ind, red, dim=1, clear=True)
                else:
                    T.reduce_max(ind, red, dim=1, clear=True)
                T.reshape(red, flat)
                if complement:
                    T.vsub(one, flat, dst)
                else:
                    T.vsub(flat, zero, dst)
                T.vcast(dst, dst16, round_mode="rint")
                T.vcast(dst16, out_ub, round_mode="rint")
                T.copy(out_ub, out)

        return main

    return _f


nan = float("nan")
rows = [
    torch.zeros(N),  # 0 any=False
    torch.ones(N),  # 1 any=True
    torch.ones(N),  # 2 any=True  (one zero among non-zeros)
    torch.full((N,), -0.0),  # 3 any=False
    torch.zeros(N),  # 4 any=True  KEY: NaN + zeros only
    torch.full((N,), nan),  # 5 any=True  KEY: all NaN
    torch.zeros(N),  # 6 any=True  (a single non-zero)
    torch.ones(N),  # 7 any=True  (zeros + NaN + non-zeros)
]
rows[2][7] = 0.0
rows[4][N // 2] = nan
rows[6][0] = 1.0
rows[7][::3] = 0.0
rows[7][N - 1] = nan
x = torch.stack(rows)
ref = torch.any(x, dim=-1)

correct = build("eq", "min", True)()(x.npu()).bool().cpu()
wrong_a = build("ne", "max", False)()(x.npu()).bool().cpu()
wrong_b = build("eq", "max", True)()(x.npu()).bool().cpu()

mis_a = (correct != wrong_a).nonzero().flatten().tolist()
mis_b = (correct != wrong_b).nonzero().flatten().tolist()
print("torch.any ref   :", ref.tolist())
print("eq + min (ok)   :", correct.tolist())
print("ne + max misrows:", mis_a)
print("eq + max misrows:", mis_b)

assert torch.equal(correct, ref), "the eq + reduce_min complement is not torch.any-exact"
assert mis_a == [4, 5], f"wrong-A exposure set {mis_a} != [4, 5] (NaN rows only)"
assert not bool(wrong_a[4]) and not bool(wrong_a[5]), "wrong-A did not mis-read the NaN rows"
assert mis_b == [2, 4, 6, 7], f"wrong-B ('some zero exists') exposure set {mis_b} != [2, 4, 6, 7]"
print("TRAP confirmed: ne+max is exposed only by the NaN rows; eq+max inverts the OR")
