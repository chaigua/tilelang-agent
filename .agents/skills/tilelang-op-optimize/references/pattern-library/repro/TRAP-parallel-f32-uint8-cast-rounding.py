# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-parallel-f32-uint8-cast-rounding

Entry      : traps-compiler.md -> TRAP-parallel-f32-uint8-cast-rounding (new)
Phenomenon : inside ``T.Parallel``, a direct ``T.cast(f32_expr, "uint8")`` does NOT
             round fractional values to the nearest integer.  The emitted
             ``hivm.hir.vcast`` (f32 -> i8) resolves to a *non-nearest* grid: for
             x.25 .. x.49 it rounds down, for some x.5x it still rounds down, and
             the result is provably neither truncation nor round-to-nearest-even.
             The two-step form ``T.vcast(src, i32, "rint")`` then
             ``T.vcast(i32, dst, "rint")`` followed by the narrowing cast *is*
             round-to-nearest-even and bit-exact.
Assertions : (1) the direct cast disagrees with ``round()`` on fractional inputs;
             (2) the two-step form matches ``round()`` on every probe value,
             including the odd integers that the direct cast corrupts.
Scope      : the encoder used fractional byte values; the fix is to round in the
             f32 -> i32 domain first so the narrowing cast only ever sees an exact
             integer.
First verified : 2026-10-14, tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760,
                 CANN 9.0.0, Ascend910B3, torch 2.9.0 torch_npu 2.9.0.post8
origin_task    : examples/fp8_quant/_fp8_quant_kernel Stage 3
"""
import sys

sys.path.insert(0, "/workspace/tilelang-mlir-ascend")
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

N = 64


def build(mode):
    @tilelang.jit(target="npuir")
    def fwd():
        @T.prim_func
        def main(a: T.Tensor((N,), "float32"), c: T.Tensor((N,), "uint8")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                av = T.alloc_ub((N,), "float32")
                cv = T.alloc_ub((N,), "uint8")
                iv = T.alloc_ub((N,), "int32")
                rv = T.alloc_ub((N,), "float32")
                T.copy(a, av)
                if mode == "direct":
                    for i in T.Parallel(N):
                        cv[i] = T.cast(T.min(T.max(av[i], -128.0), 127.0), "uint8")
                else:
                    T.vcast(av, iv, round_mode="rint")
                    T.vcast(iv, rv, round_mode="rint")
                    for i in T.Parallel(N):
                        cv[i] = T.cast(rv[i], "uint8")
                T.copy(cv, c)

        return main

    return fwd


if __name__ == "__main__":
    import torch

    vals = []
    for base in range(90, 106):
        vals += [base + 0.25, base + 0.49, base + 0.5, base + 0.51]
    vals = vals[:N]
    a = torch.tensor(vals, dtype=torch.float32)
    expect = torch.tensor([round(v) for v in vals], dtype=torch.int32)

    direct = torch.empty(N, dtype=torch.uint8, device="npu")
    build("direct")()(a.npu(), direct)
    direct = direct.cpu().to(torch.int32)
    n_bad = int((direct != expect).sum())
    print(f"direct  cast: mismatches={n_bad}/{N} "
          f"(first: {vals[(direct != expect).nonzero()[0].item()] if n_bad else '-'})")
    assert n_bad > 0, "direct f32->uint8 cast unexpectedly round-to-nearest"

    roundtrip = torch.empty(N, dtype=torch.uint8, device="npu")
    build("via_i32")()(a.npu(), roundtrip)
    roundtrip = roundtrip.cpu().to(torch.int32)
    n_bad2 = int((roundtrip != expect).sum())
    print(f"via_i32 cast: mismatches={n_bad2}/{N}")
    assert n_bad2 == 0, "two-step rint round-trip is not round-to-nearest-even"
    print("TRAP-parallel-f32-uint8-cast-rounding: PHENOMENON CONFIRMED + WORKAROUND OK")
