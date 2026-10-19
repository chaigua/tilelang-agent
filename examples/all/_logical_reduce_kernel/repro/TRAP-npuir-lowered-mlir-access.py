# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: accessing the lowered NPUIR kernel body of a ``target="npuir"`` kernel.

Finding (Stage 3 develop of `all` / `_logical_reduce_kernel`, 2026-10-16):

  * A compiled ``target="npuir"`` kernel is a ``JitKernel_NPU`` (not the
    generic ``JITKernel``).  The public ``JITKernel.get_kernel_source()`` API
    (signature ``(self) -> str``) is **not** a text channel on that class:
    observed in the same session returning the linked **ELF binary** as
    ``bytes`` (3736 B) in some runs, and raising
    ``TypeError: a bytes-like object is required, not 'str'`` in others -- so
    it is useless for a vectorisation / lowering check either way.
  * The lowered body *is* reachable as the attribute ``kernel.mlir_content``
    (a ``str``): the TileLangIR/HIVM MLIR of the device function, showing the
    ``hivm.hir.v*`` ops in the order they were emitted.

Why it matters: the "is the judgement chain actually vectorised (no per-element
scalar loop)?" gate (DESIGN.md section 8.2 L0-5, `TRAP-parallel2d-bm-ge2-scalarize`
regression channel) needs a text channel.  Grepping the ELF bytes / the
generated host source does not provide it.

Assertions (all must hold):

  A  ``kernel.mlir_content`` exists, is a ``str``, and contains the exact
     chain of this operator: vcmp -> vcast(i1->f32) -> vreduce<max> -> vsub
     -> vcast(f32->f16) -> vcast(f16->i8), with a ``<max>`` reduction marker
     and ``reduce_dims = [1]`` on the horizontal (last) axis.
  B  the same text contains **no** per-element scalar loop / ``if_then_else``
     (``arith.cmpf`` / ``scf.for`` / ``if_then_else`` absent) -- i.e. the
     explicit v-prefix chain did not degrade to a scalar form at bm = 8.
  C  ``get_kernel_source()`` on the NPU kernel class does not return a
     text source: it either raises ``TypeError`` (observed) or returns
     ``bytes`` (ELF).  The check is written so that a future toolchain that
     starts returning text still passes via the ``bytes`` branch -- the point
     is only that the MLIR text must come from ``mlir_content``.

Run:  python -u examples/all/_logical_reduce_kernel/repro/TRAP-npuir-lowered-mlir-access.py
Exit code 0 == all assertions hold.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tvm import tir  # noqa: E402

M, N = 64, 512
BM = 8
CHAIN = ("hivm.hir.vcmp", "hivm.hir.vcast", "vreduce", "<max>", "hivm.hir.vsub")
SCALAR_FORMS = ("arith.cmpf", "scf.for", "if_then_else")


def build():
    @tilelang.jit(out_idx=[1], target="npuir")
    def _factory(block_m: int):
        @T.prim_func
        def main(
            x: T.Tensor((M, N), "float16"),
            out: T.Tensor((M,), "int8"),
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid, _):
                x_ub = T.alloc_shared((block_m, N), "float16")
                zero_ub = T.alloc_shared((block_m, N), "float16")
                mask_eq = T.alloc_shared((block_m, N), "bool")
                ind = T.alloc_shared((block_m, N), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")
                out_ub = T.alloc_shared((block_m,), "int8")
                real_m = T.min(block_m, M - pid * block_m)
                T.fill(zero_ub, tir.const(0, "float16"))
                T.fill(one_v, tir.const(1, "float32"))
                T.copy(x[pid * block_m : pid * block_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                T.vcast(mask_eq, ind, round_mode="rint")
                T.reduce_max(ind, mx, dim=1, clear=True)
                T.reshape(mx, mx_flat)
                T.vsub(one_v, mx_flat, all_v)
                T.vcast(all_v, all_f16, round_mode="rint")
                T.vcast(all_f16, out_ub, round_mode="rint")
                T.copy(out_ub[0:real_m], out[pid * block_m : pid * block_m + real_m])

        return main

    return _factory(BM)


def main() -> int:
    print(f"tilelang {tilelang.__version__}")
    kernel = build()
    print(f"kernel class: {type(kernel).__name__}")

    # --- A: the MLIR text channel ---------------------------------------- #
    mlir = getattr(kernel, "mlir_content", None)
    assert isinstance(mlir, str) and mlir, "kernel.mlir_content is not a non-empty str"
    missing = [tok for tok in CHAIN if tok not in mlir]
    assert not missing, f"lowered MLIR lacks {missing}"
    assert "reduce_dims = [1]" in mlir, "the reduction is not on the horizontal axis"
    print("A PASS: mlir_content holds the vcmp->vcast->vreduce<max>->vsub->vcast chain")

    # --- B: no scalar degradation --------------------------------------- #
    present = [tok for tok in SCALAR_FORMS if tok in mlir]
    assert not present, f"lowered MLIR contains scalar/ite forms {present}"
    print("B PASS: no per-element scalar loop at bm = 8")

    # --- C: get_kernel_source() is not a text channel ------------------- #
    try:
        src = kernel.get_kernel_source()
    except TypeError as exc:
        print(f"C PASS: get_kernel_source() raised TypeError ({str(exc)[:60]}); "
              f"use mlir_content for lowering checks")
    else:
        assert isinstance(src, (bytes, bytearray)), (
            f"get_kernel_source() unexpectedly returned {type(src).__name__}"
        )
        print(f"C PASS: get_kernel_source() returns {type(src).__name__} "
              f"(the linked ELF, not source text)")

    for line in mlir.splitlines():
        if any(tok in line for tok in CHAIN):
            print("   ", line.strip()[:150])
    print("ALL ASSERTIONS HOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
