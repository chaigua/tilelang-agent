#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-parallel-affine-cast-plane -- Home file: traps-compiler.md (new entry).

Phenomenon
    A 2-D `T.Parallel(bn, bm)` loop whose body is an affine index expression
    with a per-operand cast

        for i0, j0 in T.Parallel(bn, bm):
            plane[i0, j0] = T.Cast("float32", j0) - T.Cast("float32", i0)

    silently drops the row term for part of the tile on this toolchain: for a
    (16, 32) fp32 plane the last 6 lanes of the last row read back as `j0`
    (e.g. (i=15, j=26) -> 26 instead of 11).  Casting the *difference* instead

        plane[i0, j0] = T.Cast("float32", j0 - i0)

    is exact, as is a nested `T.serial` scalar build.

Assertions
    (A) per-operand-cast form: mismatch count > 0    (trap present, asserted)
    (C) nested serial scalar form: mismatch count == 0  (workaround, asserted)
    (B) cast-of-difference form is printed only (recorded): it came out exact in
    one isolated probe and corrupt in this one -> the `T.Parallel` plane build is
    *context dependent*, so only the serial-nest form can be relied on.

First verified
    tilelang 0.1.2 dev root build + CANN 9.0.0 + Ascend910B3, 2026-10-16
origin_task
    multi_head_attention / _gqa_bwd_kernel Stage 3 (causal mask plane)
Re-verify
    python repro/TRAP-parallel-affine-cast-plane.py   (exit 0 == trap + workarounds hold)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BN, BM = 16, 32


@tilelang.jit(target="npuir")
def plane_kernel(bn: int, bm: int, form: int):
    @T.prim_func
    def main(OUT: T.Tensor((bn, bm), "float32")) -> None:  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, _):
            plane = T.alloc_shared((bn, bm), "float32")
            if form == 1:      # per-operand casts (trap)
                for i0, j0 in T.Parallel(bn, bm):
                    plane[i0, j0] = T.Cast("float32", j0) - T.Cast("float32", i0)
            elif form == 2:    # cast of the difference (workaround)
                for i0, j0 in T.Parallel(bn, bm):
                    plane[i0, j0] = T.Cast("float32", j0 - i0)
            else:              # nested serial scalar (workaround)
                for i0 in T.serial(bn):
                    for j0 in T.serial(bm):
                        plane[i0, j0] = T.Cast("float32", j0 - i0)
            T.copy(plane, OUT)

    return main


def bad_lanes(form):
    kernel = plane_kernel(BN, BM, form)
    out = torch.zeros((BN, BM), dtype=torch.float32, device="npu")
    kernel(out)
    torch.npu.synchronize()
    i = torch.arange(BN).view(BN, 1).float()
    j = torch.arange(BM).view(1, BM).float()
    return int((out.cpu() != (j - i)).sum().item())


if __name__ == "__main__":
    a, b, c = bad_lanes(1), bad_lanes(2), bad_lanes(3)
    print(f"A per-operand cast (trap)  : {a} mismatched lanes")
    print(f"B cast of difference (fix) : {b} mismatched lanes")
    print(f"C nested serial (fix)      : {c} mismatched lanes")
    assert a > 0, "trap not reproduced: the per-operand-cast plane came out exact"
    assert c == 0, f"serial-nest workaround is not exact ({c} lanes)"
    print(f"(B is recorded only: {b} lanes -- the T.Parallel plane build is context dependent)")
    print("ALL ASSERTIONS PASS")
