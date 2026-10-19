#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-vec-cube-ub-scratch-role-conflict -- Home file: traps-compiler.md (new).

Phenomenon
    Developer mode, one UB buffer used by BOTH engines inside a block:

        buf  <- vector op            (e.g. mask/temp scratch)
        buf  <- T.gemm(..., buf, initC=True)
        out  <- vector op reading buf

    the consumer reads a *corrupted* `buf`: none of the 2048 cells equal the gemm
    result and none equal the pre-gemm vector value (a mixture/wrong lowering),
    so the gemm's contribution is destroyed.  The corruption is silent -- there is
    no compile error and no NaN.

    Giving the vector scratch its own buffer, or writing `buf` only from the gemm,
    is exact.  Same class of failure shows up at kernel scale when a tail/mask
    penalty plane reuses the buffer that a later GEMM writes (in the operator the
    causal branch read the mask penalty instead of `dO V^T`, producing dk up to
    ~1e4).

Assertions
    (A) gemm-only                 -> out exact                (control)
    (B) vector write then gemm    -> out has 0 cells equal to the truth
    (C) separate scratch buffer   -> out exact                (workaround)

First verified
    tilelang 0.1.2 dev root build + CANN 9.0.0 + Ascend910B3, 2026-10-16
origin_task
    multi_head_attention / _gqa_bwd_kernel Stage 3 (dsum used as mask temp + g2 dst)
Re-verify
    python repro/TRAP-vec-cube-ub-scratch-role-conflict.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BN, BM, K = 32, 64, 64


@tilelang.jit(target="npuir")
def conflict(mode: int, bn: int, bm: int, k: int):
    """mode 0 = gemm only, 1 = vector write then gemm (trap), 2 = separate scratch."""

    @T.prim_func
    def main(
        A: T.Tensor((bn, k), "float16"),  # type: ignore
        B: T.Tensor((bm, k), "float16"),  # type: ignore
        ROW: T.Tensor((bn, 1), "float32"),  # type: ignore
        OUT: T.Tensor((bn, bm), "float32"),  # type: ignore
    ) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_s = T.alloc_shared((bn, k), "float16")
            b_s = T.alloc_shared((bm, k), "float16")
            buf = T.alloc_shared((bn, bm), "float32")
            scratch = T.alloc_shared((bn, bm), "float32")
            row = T.alloc_shared((bn, 1), "float32")
            T.copy(A, a_s)
            T.copy(B, b_s)
            T.copy(ROW, row)
            if mode == 1:
                T.vbrc(T.float32(7.0), buf)      # vector op owns the gemm dst
            elif mode == 2:
                T.vbrc(T.float32(7.0), scratch)  # separate scratch
            T.gemm(a_s, b_s, buf, b_transpose=True, initC=True)
            T.vsub(buf, row, buf)
            T.copy(buf, OUT)

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(BN, K, dtype=torch.float16, device="npu")
    b = torch.randn(BM, K, dtype=torch.float16, device="npu")
    row = torch.randn(BN, 1, dtype=torch.float32, device="npu")
    truth = (a.cpu().float() @ b.cpu().float().T) - row.cpu()
    stats = {}
    for mode in (0, 1, 2):
        out = torch.zeros(BN, BM, dtype=torch.float32, device="npu")
        conflict(mode, BN, BM, K)(a, b, row, out)
        torch.npu.synchronize()
        got = out.cpu()
        exact = int((torch.abs(got - truth) < 1e-3).sum().item())
        stats[mode] = exact
        print(f"mode={mode}: cells matching truth = {exact}/{got.numel()}  "
              f"max_diff={(got - truth).abs().max().item():.4e}")
    assert stats[0] == BN * BM, "control (gemm only) is not exact"
    assert stats[1] == 0, "trap not reproduced: the shared scratch came out exact"
    assert stats[2] == BN * BM, "separate-scratch workaround is not exact"
    print("ALL ASSERTIONS PASS")
