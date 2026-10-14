# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-vexp2-not-exact-pow2

Entry      : traps-compiler.md -> TRAP-vexp2-not-exact-pow2 (new)
Phenomenon : the tile-level ``T.vexp2`` is implemented as ``exp(x * ln2)`` -- it is
             NOT an exact power of two.  For integer exponents it returns values
             like ``2^-4 -> 0.0624999962747097`` (6e-8 relative error).  When that
             value scales a mantissa that must be rounded to nearest even, every
             exact half-way mantissa is pushed to the wrong side of the tie
             (observed: 17/16384 fp8 bytes off, all at ``n.5`` mantissas).
             ``2^k`` with k = ec + 6 built from ``T.vshl(1, k)`` (int32) and cast
             back to f32 is bit-exact, and dividing by it is exact too.
Assertions : (1) the ``vexp2`` route disagrees with the exact power of two for
             every probed exponent (trap reproduced);
             (2) the ``vshl`` route reproduces ``2.0**k`` bit-exactly for
             k = 0..14, so ``x * 2^9 / 2^(ec+6)`` is an exact mantissa scale.
Scope      : any software float encoder that needs ``2^k`` (fp8/bf16/int8
             quantization, exponent re-scaling) with round-to-nearest-even.
First verified : 2026-10-14, tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760,
                 CANN 9.0.0, Ascend910B3, torch 2.9.0 torch_npu 2.9.0.post8
origin_task    : examples/fp8_quant/_fp8_quant_kernel Stage 3
"""
import sys

sys.path.insert(0, "/workspace/tilelang-mlir-ascend")
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

K = 15  # exponents 0..14


@tilelang.jit(target="npuir")
def pow2():
    @T.prim_func
    def main(kv: T.Tensor((K,), "float32"),
             exp2_v: T.Tensor((K,), "float32"),
             shl_v: T.Tensor((K,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            kv_u = T.alloc_ub((K,), "float32")
            pv = T.alloc_ub((K,), "float32")
            tv = T.alloc_ub((K,), "float32")
            iv = T.alloc_ub((K,), "int32")
            one = T.alloc_ub((K,), "int32")
            pw = T.alloc_ub((K,), "int32")
            T.copy(kv, kv_u)
            for i in T.Parallel(K):
                one[i] = 1
            T.vexp2(kv_u, pv, tv)                       # exp(k * ln2) -- inexact
            T.vcast(kv_u, iv, round_mode="rint")
            T.vshl(one, iv, pw)                         # 1 << k -- exact
            T.vcast(pw, tv, round_mode="rint")
            T.copy(pv, exp2_v)
            T.copy(tv, shl_v)

    return main


if __name__ == "__main__":
    import torch

    k = torch.arange(K, dtype=torch.float32)
    exp2_v = torch.empty(K, dtype=torch.float32, device="npu")
    shl_v = torch.empty(K, dtype=torch.float32, device="npu")
    pow2()(k.npu(), exp2_v, shl_v)
    exp2_v, shl_v = exp2_v.cpu(), shl_v.cpu()
    exact = torch.pow(torch.tensor(2.0), k)

    n_exp2_bad = int((exp2_v != exact).sum())
    n_shl_bad = int((shl_v != exact).sum())
    rel = ((exp2_v - exact).abs() / exact)
    print(f"vexp2: mismatches={n_exp2_bad}/{K} max_rel_err={rel.max().item():.3e}")
    print(f"vshl : mismatches={n_shl_bad}/{K} values={shl_v[:5].tolist()}")
    assert n_exp2_bad > 0, "T.vexp2 became exact -- re-verify entry"
    assert n_shl_bad == 0, "T.vshl-based 2^k is not bit-exact"
    print("TRAP-vexp2-not-exact-pow2: PHENOMENON CONFIRMED + WORKAROUND OK")
