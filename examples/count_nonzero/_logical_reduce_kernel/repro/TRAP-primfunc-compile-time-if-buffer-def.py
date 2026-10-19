# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-primfunc-compile-time-if-buffer-def -- Home file: pattern-library/traps-compiler.md

Phenomenon: inside ``@T.prim_func``, a **Python-level (parse-time) ``if`` that
allocates buffers** is not usable.  Names bound inside such a branch do not
escape into the enclosing prim_func scope, so a later reference from a sibling
branch either fails to resolve (``ParserError: Undefined variable: <name>``) or
-- when the condition is false so the branch body is skipped entirely --
segfaults the compiler process (SIGSEGV, no diagnostic).  Workaround: hoist the
structural condition to the *factory* level and emit one dedicated
``prim_func`` per variant (the pattern used by migrated row-reduction kernels
that need an optional static-width tail block).

Assertions (trap kind: reproduce the phenomenon + workaround passes):
  A. ``tail > 0``: branch-local buffer definitions + sibling-branch reference ->
     the build raises, and the rendered exception chain matches
     ``Undefined variable``.
  B. ``tail == 0``: the very same source builds cleanly (subprocess exits 0) --
     the defect is **latent and shape-dependent**: a kernel that compiles for
     every divisible width breaks as soon as the optional width becomes
     non-zero.  (In this task's larger combined form -- a full guarded tail-tile
     body plus guarded epilogue branches -- the tail==0 case was also observed
     to segfault the compiler process; that minimal trigger was not pinned, so
     this repro asserts only the reliable parse-error signature.)
  C. Workaround: two dedicated prim_funcs selected at factory level compile,
     and (when an NPU device is present) the compiled kernel returns exact
     per-row non-zero counts -- the legal form is bit-exact.

First verified: 2026-10-16, tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892
+ CANN 9.0.0 + Ascend910B3 (AIV=40 / UB 196608 B).
origin_task: examples/count_nonzero/_logical_reduce_kernel (Stage 3; count_nonzero
factory, path C column-tiled with an optional static tail block).
Re-verify: ``/workspace/venv/bin/python <this file>``; a toolchain that fixes the
parser flips assertions A/B and must re-stamp this entry.

Run: python repro/TRAP-primfunc-compile-time-if-buffer-def.py
"""

import os
import subprocess
import sys
import traceback

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tvm import tir  # noqa: E402

M, N, DTYPE, TILE_N, BM = 8, 8192, "float16", 8192, 1


@T.macro
def _chain_body(x, out, cid, block_m, x_ub, zero_ub, mask, ind, zc, cnt, n_v, out_ub):
    """One count_nonzero tile: copy -> vcmp(eq) -> vcast -> reduce_sum -> N - zc."""
    start = cid * block_m
    if start < M:
        T.copy(x[start : start + block_m, 0:TILE_N], x_ub)
        T.vcmp(x_ub, zero_ub, mask, "eq")
        T.vcast(mask, ind, round_mode="rint")
        T.reduce_sum(ind, zc, dim=1, clear=True)
        T.reshape(zc, cnt)
        T.vsub(n_v, cnt, cnt)  # zero-count complement -> non-zero count
        T.vcast(cnt, out_ub, round_mode="rint")
        T.copy(out_ub, out[start : start + block_m])


def build_conditional(tail):
    """The trap shape: a parse-time ``if`` that defines a buffer."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_kernels = min((M + block_m - 1) // block_m, 40)

        @T.prim_func
        def main(x: T.Tensor((M, N), DTYPE), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, TILE_N), DTYPE)
                zero_ub = T.alloc_shared((block_m, TILE_N), DTYPE)
                mask = T.alloc_shared((block_m, TILE_N), "bool")
                ind = T.alloc_shared((block_m, TILE_N), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")
                if tail > 0:  # parse-time: binds a name inside the branch scope
                    zero_tail = T.alloc_shared((block_m, tail), DTYPE)
                T.fill(zero_ub, tir.const(0, DTYPE))
                T.fill(n_v, tir.const(N, "float32"))
                if tail > 0:  # sibling branch: the name does not resolve here
                    T.fill(zero_tail, tir.const(0, DTYPE))
                _chain_body(x, out, cid, block_m, x_ub, zero_ub, mask, ind, zc, cnt, n_v, out_ub)

        return main

    return _func(BM)


def build_workaround(with_tail):
    """The workaround: one dedicated prim_func per structural variant."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_kernels = min((M + block_m - 1) // block_m, 40)

        @T.prim_func
        def tail_pf(x: T.Tensor((M, N), DTYPE), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, TILE_N), DTYPE)
                zero_ub = T.alloc_shared((block_m, TILE_N), DTYPE)
                mask = T.alloc_shared((block_m, TILE_N), "bool")
                ind = T.alloc_shared((block_m, TILE_N), "float32")
                zero_tail = T.alloc_shared((block_m, 808), DTYPE)
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")
                T.fill(zero_ub, tir.const(0, DTYPE))
                T.fill(n_v, tir.const(N, "float32"))
                T.fill(zero_tail, tir.const(0, DTYPE))
                _chain_body(x, out, cid, block_m, x_ub, zero_ub, mask, ind, zc, cnt, n_v, out_ub)

        @T.prim_func
        def plain_pf(x: T.Tensor((M, N), DTYPE), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, TILE_N), DTYPE)
                zero_ub = T.alloc_shared((block_m, TILE_N), DTYPE)
                mask = T.alloc_shared((block_m, TILE_N), "bool")
                ind = T.alloc_shared((block_m, TILE_N), "float32")
                zc = T.alloc_fragment((block_m, 1), "float32")
                cnt = T.alloc_fragment((block_m,), "float32")
                n_v = T.alloc_fragment((block_m,), "float32")
                out_ub = T.alloc_shared((block_m,), "int64")
                T.fill(zero_ub, tir.const(0, DTYPE))
                T.fill(n_v, tir.const(N, "float32"))
                _chain_body(x, out, cid, block_m, x_ub, zero_ub, mask, ind, zc, cnt, n_v, out_ub)

        return tail_pf if with_tail else plain_pf

    return _func(BM)


def _assert_trap():
    # A: parse-time-true branch -> unresolved name at the sibling reference.
    # The parser error surfaces through the exception chain (DiagnosticError
    # wrapping ParserError), so scan the rendered chain.
    try:
        build_conditional(808)
    except Exception as exc:  # noqa: BLE001
        chain = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        assert "Undefined variable" in chain, f"A: unexpected error: {chain[:300]}"
        print(f"A PASS: tail>0 build raises -> Undefined variable: zero_tail")
    else:
        raise AssertionError("A FAIL: conditional buffer definition did not raise")

    # B: parse-time-false branch -> the same source builds cleanly (isolated
    # subprocess, so the trap's latency is demonstrated without parser state).
    done = subprocess.run(
        [sys.executable, __file__, "--tail0"], capture_output=True, text=True
    )
    assert done.returncode == 0 and "BUILD_OK" in done.stdout, (
        f"B FAIL: tail==0 build rc={done.returncode}: {done.stderr[-300:]}"
    )
    print("B PASS: tail==0 builds cleanly -> defect is latent/shape-dependent")

    # C: the workaround compiles; run it exactly when a device is present.
    import torch  # noqa: PLC0415

    for with_tail in (False, True):
        kernel = build_workaround(with_tail)
        try:
            import torch_npu  # noqa: F401,PLC0415

            x = torch.zeros(M, N, dtype=torch.float16, device="npu")
            x[:, ::3] = 1.0
            y = kernel(x).cpu()
            ref = torch.count_nonzero(x.cpu(), dim=-1).to(torch.int64)
            assert torch.equal(y, ref), f"C FAIL: tail={with_tail} counts differ"
            print(f"C PASS: workaround (with_tail={with_tail}) compiles and is exact")
        except ImportError:
            print(f"C PASS: workaround (with_tail={with_tail}) compiles (no NPU device)")


if __name__ == "__main__":
    if "--tail0" in sys.argv:
        build_conditional(0)
        print("BUILD_OK")
        sys.exit(0)
    print(f"tilelang {tilelang.__version__}")
    _assert_trap()
    print("ALL ASSERTIONS PASS")
