# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: an IR-form gate that bans ``scf.for`` / ``if_then_else`` on the
lowered npuir kernel must be scoped to the *resident* path.

Entry (new proposal): traps-compiler.md ``TRAP-npuir-ir-gate-loop-scope``.
Single experience: the lowered-MLIR shape of the two capacity paths.

Phenomenon (asserted below; compile-only, no data movement, no device needed):
  * the **resident** path (whole row block resident in UB, one reduce per task)
    lowers with **no** ``scf.for`` and no ``if_then_else``;
  * the **column-tiled** path (static ``T.serial(num_full)`` tile loop + no-alias
    ``vmin`` running merge) legitimately lowers **with** an ``scf.for`` -- that
    loop *is* the tile loop, not a scalarised per-element loop;
  * both lower the row reduction as ``hivm.hir.vreduce <min>`` with
    ``reduce_dims = [1]`` and no ``arith.cmpf``.

Why it matters: ``TRAP-parallel2d-bm-ge2-scalarize`` asks an IR gate to fail on
``scf.for`` / ``if_then_else`` (the shape a scalarised ``T.Parallel`` +
``if_then_else`` judgement degrades to).  Copying that ban verbatim onto the
column-tiled variant reports a **false failure** on a correctly vectorised
kernel.  Scope the ban to the resident variant and assert the vector ops on
both.

First verified / toolchain: 2026-10-16, tilelang
0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba + CANN 9.0.0, Ascend910B3
(origin_task: any-_logical_reduce_kernel Stage 3, first_impl attempt 1).
Re-verification log: -.

Run: /workspace/venv/bin/python repro/TRAP-npuir-ir-gate-loop-scope.py
Exit code 0 == the loop-scope trap is confirmed.
"""

import os
import re

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
from tvm import tir

M, DTYPE = 32, "float32"
N_RES, TILE_N, NUM_FULL = 512, 8192, 4


def build(N, row_block, tile_n, num_full):
    """Resident when ``num_full == 1`` else column-tiled with a vmin merge."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(x: T.Tensor((M, N), DTYPE), out: T.Tensor((M,), "int8")):
            with T.Kernel(M // row_block, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((row_block, tile_n), DTYPE)
                z_ub = T.alloc_shared((row_block, tile_n), DTYPE)
                mask = T.alloc_shared((row_block, tile_n), "bool")
                ind = T.alloc_shared((row_block, tile_n), "float32")
                mn = T.alloc_fragment((row_block, 1), "float32")
                acc = T.alloc_fragment((row_block, 1), "float32")
                nxt = T.alloc_fragment((row_block, 1), "float32")
                flat = T.alloc_fragment((row_block,), "float32")
                one = T.alloc_fragment((row_block,), "float32")
                dst = T.alloc_fragment((row_block,), "float32")
                dst16 = T.alloc_fragment((row_block,), "float16")
                out_ub = T.alloc_shared((row_block,), "int8")
                T.fill(z_ub, tir.const(0, DTYPE))
                T.fill(one, tir.const(1, "float32"))
                start = cid * row_block
                if start < M:
                    real_m = T.min(row_block, M - start)
                    T.fill(acc, tir.const(1, "float32"))
                    for t in T.serial(num_full):
                        T.copy(
                            x[start : start + real_m, t * tile_n : (t + 1) * tile_n],
                            x_ub[0:real_m, 0:tile_n],
                        )
                        T.vcmp(x_ub, z_ub, mask, "eq")
                        T.vcast(mask, ind, round_mode="rint")
                        T.reduce_min(ind, mn, dim=1, clear=True)
                        T.vmin(acc, mn, nxt)
                        T.copy(nxt, acc)
                    T.reshape(acc, flat)
                    T.vsub(one, flat, dst)
                    T.vcast(dst, dst16, round_mode="rint")
                    T.vcast(dst16, out_ub, round_mode="rint")
                    T.copy(out_ub[0:real_m], out[start : start + real_m])

        return main

    return _f


# Resident: one tile per row block (row_block == M => a single logical core).
ir_res = build(N_RES, M, N_RES, 1)().mlir_content
# Column-tiled: bm=1 + static tile loop + no-alias vmin running merge.
ir_tiled = build(TILE_N * NUM_FULL, 1, TILE_N, NUM_FULL)().mlir_content

for name, ir in (("resident", ir_res), ("column-tiled", ir_tiled)):
    print(f"[{name}] len={len(ir)} scf.for={'scf.for' in ir} if_then_else={'if_then_else' in ir}")
    assert "hivm.hir.vcmp" in ir, f"{name}: vcmp missing"
    assert "hivm.hir.vcast" in ir, f"{name}: vcast missing"
    assert "vsub" in ir, f"{name}: vsub missing"
    assert "vreduce <min>" in ir, f"{name}: the reduction is not <min>"
    assert re.search(r"reduce_dims\s*=\s*\[\s*1\b", ir), f"{name}: reduce_dims != [1]"
    assert "arith.cmpf" not in ir, f"{name}: judgement degraded to a scalar compare"

assert "scf.for" not in ir_res, "resident path unexpectedly contains scf.for"
assert "scf.for" in ir_tiled, "column-tiled path lost its static tile loop"
print(
    "TRAP confirmed: the `scf.for` / `if_then_else` IR ban must be scoped to the "
    "resident variant; the column-tiled variant legitimately emits scf.for"
)
