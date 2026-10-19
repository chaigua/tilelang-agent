#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-bishengir-crash-s32 -- Home file: traps-compiler.md (new entry).

Phenomenon
    A Developer-mode attention-backward-style kernel (persistent `T.Kernel` +
    `T.serial` task loop + `T.Pipelined` query loop + runtime-extent `T.copy`
    into pre-cleared operand buffers + `T.gemm` + per-row `T.atomic_add`) makes
    ``bishengir-compile`` (CANN 9.0.0) **segfault** for ``S == 32``:

        NPU IR opt failed: PLEASE submit a bug report ... Stack dump:

    while ``S`` in {16, 24, 40, 48, 56, 64, 96} compile.  The crash is
    independent of the tile ((bm,bn) = (64,64) or (32,32)), of `heads`, of
    `dim` in {64,128} and of `is_causal`.  Manifest shapes (S multiple of 64)
    are unaffected.  This is an upstream backend defect, not a kernel-logic
    error: the same source compiles for every other tested `S`.

    (If this script reports ``S=32: COMPILED`` the defect has been fixed by a
    toolchain upgrade -- refresh the stamp and close the entry.)

Assertions
    (A) S == 32 -> compilation raises (bishengir-compile crash / bug report)
    (B) S == 40 -> compiles (control)
    (C) S == 24 -> compiles (control)

First verified
    tilelang 0.1.2 dev root build + CANN 9.0.0 (bishengir-compile
    /usr/local/Ascend/cann-9.0.0/bin/bishengir-compile) + Ascend910B3, 2026-10-16
origin_task
    multi_head_attention / _gqa_bwd_kernel Stage 3 (L2 warn-only record)
Re-verify
    python repro/TRAP-bishengir-crash-s32.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

H, DIM, BM, BN = 4, 64, 64, 64


@tilelang.jit(target="npuir")
def tiny_bwd(h: int, s: int, dim: int, bm: int, bn: int):
    """Reduced attention-backward skeleton: same structural recipe as the
    operator (persistent tasks + pipelined query loop + runtime-extent copies
    into cleared operands + gemm + per-row atomic write-back)."""

    @T.prim_func
    def main(
        q: T.Tensor((s, h, dim), "float16"),  # type: ignore
        k: T.Tensor((s, h, dim), "float16"),  # type: ignore
        dq: T.Tensor((s, h, dim), "float32"),  # type: ignore
    ) -> None:
        with T.Kernel(h, is_npu=True) as (bx, _):
            k_s = T.alloc_shared((bm, dim), "float16")
            q_s = T.alloc_shared((bn, dim), "float16")
            qkt = T.alloc_shared((bn, bm), "float32")
            pcast = T.alloc_shared((bn, bm), "float16")
            dq_blk = T.alloc_shared((bn, dim), "float32")
            for by in T.serial(1):
                r0 = by * bm
                real_m = T.min(bm, s - r0)
                T.clear(k_s)
                T.copy(k[r0 : r0 + real_m, bx, 0:dim], k_s[0:real_m, 0:dim])
                for kq in T.Pipelined(0, 1, num_stages=1):
                    c0 = kq * bn
                    real_n = T.min(bn, s - c0)
                    T.clear(q_s)
                    T.copy(q[c0 : c0 + real_n, bx, 0:dim], q_s[0:real_n, 0:dim])
                    T.gemm(q_s, k_s, qkt, b_transpose=True, initC=True)
                    T.vexp(qkt, qkt)
                    T.vcast(qkt, pcast, round_mode="rint")
                    T.gemm(pcast, k_s, dq_blk, initC=True)
                    for r in T.serial(real_n):
                        T.atomic_add(dq[c0 + r, bx, 0:dim], dq_blk[r, 0:dim],
                                     [1, dim])

    return main


if __name__ == "__main__":
    outcomes = {}
    for s, name in ((32, "A S=32"), (40, "B S=40"), (24, "C S=24")):
        try:
            tiny_bwd(H, s, DIM, BM, BN)
            outcomes[s] = "compiled"
            print(f"{name}: COMPILED")
        except Exception as exc:  # noqa: BLE001
            txt = str(exc)
            outcomes[s] = "crash" if "bug report" in txt else "other"
            print(f"{name}: FAILED [{outcomes[s]}]")
    assert outcomes[32] == "crash", (
        f"S=32 did not hit the backend crash ({outcomes[32]}); the defect may be "
        "fixed -- refresh the toolchain stamp"
    )
    assert outcomes[40] == "compiled", "S=40 control did not compile"
    assert outcomes[24] == "compiled", "S=24 control did not compile"
    print("ALL ASSERTIONS PASS")
