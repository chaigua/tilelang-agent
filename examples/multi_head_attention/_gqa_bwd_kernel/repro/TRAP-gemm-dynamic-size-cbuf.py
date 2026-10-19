#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-gemm-dynamic-size-cbuf -- Home file: traps-compiler.md (new entry).

Phenomenon
    Developer mode: a `T.gemm` whose `size=[m, k, n]` override carries a
    *runtime* extent (e.g. the tail bound `rn = T.min(bn, S - c0)` with
    `S % bn != 0`) and which sits *inside a `T.Pipelined` loop* makes the
    BishengIR pipeline die with a nonsensical message

        cbuf overflow, requires 12009693044328448 bits while 4194304 bits available

    i.e. an L1 (cbuf = 512 KB) allocation blows up although the declared tiles
    are ~100 KB.  Dropping the `size` override (buffer extents) or passing a
    static `size` compiles; so does a runtime `size` *outside* the pipelined
    loop.  Both ingredients -- a runtime extent in `size` AND the pipelined
    frame -- are required.

Assertions
    (A) pipelined + runtime size  -> compilation raises "cbuf overflow"
    (B) pipelined + static size   -> compiles
    (C) runtime size, no pipeline -> compiles   (isolates the pipeline as a
        required ingredient; dynamic `T.copy` extents are fine everywhere)

Workaround used by the operator
    Runtime tail bounds stay only on the GM<->UB copies and the row-atomic
    loops; the GEMMs use full-tile extents (no `size` override) and the operand
    buffers are pre-zeroed (`T.clear`) so lanes outside `[0, real_*)` contribute
    exactly 0.

First verified
    tilelang 0.1.2 dev root build + CANN 9.0.0 + Ascend910B3, 2026-10-16
origin_task
    multi_head_attention / _gqa_bwd_kernel Stage 3 (tail blocks, S % 64 != 0)
Re-verify
    python repro/TRAP-gemm-dynamic-size-cbuf.py   (exit 0 == trap + workarounds hold)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

S, DIM, BM, BN, NQ = 100, 64, 64, 64, 2


@tilelang.jit(target="npuir")
def gemm_tail(mode: int, s: int, dim: int, bm: int, bn: int, nq: int):
    """mode 0 = pipelined + runtime size, 1 = pipelined + static size,
    2 = runtime size outside the pipeline."""

    @T.prim_func
    def main(
        A: T.Tensor((s, dim), "float16"),  # type: ignore
        B: T.Tensor((s, dim), "float16"),  # type: ignore
        C: T.Tensor((s, s), "float32"),  # type: ignore
    ) -> None:
        with T.Kernel(2, is_npu=True) as (cid, _):
            a_s = T.alloc_shared((bn, dim), "float16")
            b_s = T.alloc_shared((bm, dim), "float16")
            c_s = T.alloc_shared((bn, bm), "float32")
            r0 = cid * bm
            rm = T.min(bm, s - r0)
            T.clear(b_s)
            T.copy(B[r0 : r0 + rm, 0:dim], b_s[0:rm, 0:dim])
            if mode == 2:                       # no pipeline
                c0 = 0
                rn = T.min(bn, s - c0)
                T.clear(a_s)
                T.copy(A[c0 : c0 + rn, 0:dim], a_s[0:rn, 0:dim])
                T.gemm(a_s, b_s, c_s, b_transpose=True, initC=True,
                       size=[rn, dim, rm])
                T.copy(c_s, C[c0 : c0 + rn, r0 : r0 + rm])
            else:
                for kq in T.Pipelined(0, nq, num_stages=1):
                    c0 = kq * bn
                    rn = T.min(bn, s - c0)
                    T.clear(a_s)
                    T.copy(A[c0 : c0 + rn, 0:dim], a_s[0:rn, 0:dim])
                    if mode == 0:               # runtime size (trap)
                        T.gemm(a_s, b_s, c_s, b_transpose=True, initC=True,
                               size=[rn, dim, rm])
                    else:                       # static size
                        T.gemm(a_s, b_s, c_s, b_transpose=True, initC=True,
                               size=[bn, dim, bm])
                    T.copy(c_s, C[c0 : c0 + rn, r0 : r0 + rm])

    return main


if __name__ == "__main__":
    results = {}
    for mode, name in (
        (0, "A pipelined + runtime size"),
        (1, "B pipelined + static size "),
        (2, "C runtime size, no pipeline"),
    ):
        try:
            gemm_tail(mode, S, DIM, BM, BN, NQ)
            results[mode] = None
            print(f"{name}: compiled")
        except Exception as exc:  # noqa: BLE001
            results[mode] = str(exc)
            flag = "cbuf overflow" if "cbuf overflow" in str(exc) else "other error"
            print(f"{name}: FAILED [{flag}]")
    assert results[0] is not None and "cbuf overflow" in results[0], (
        "trap not reproduced: the pipelined runtime-size gemm did not overflow"
    )
    assert results[1] is None, "static-size workaround did not compile"
    assert results[2] is None, "runtime size outside the pipeline did not compile"
    print("ALL ASSERTIONS PASS")
