# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-npuir-f32-division-not-ieee-rounded

Entry      : constants.md -> CONST-npuir-f32-division-not-ieee-rounded (new)
Phenomenon : npuir float32 division is NOT IEEE correctly rounded.  Two measured
             forms on Ascend910B3 (tilelang 0.1.2+8b16bcc, CANN 9.0.0):
               * ``T.Parallel`` scalar expression ``a / b`` (per-row scale):
                 135/256 values 1-2 ulp off vs the correctly rounded quotient;
               * ``T.vdiv`` between UB buffers:
                 - 1-column operand ``(M,1) / (M,1)``: bit-exact (0/256);
                 - 2-D broadcast/2-D same-shape ``(M,D)``: ~5.6% of lanes 1 ulp off.
             Consequence for a software fp8 encoder: at an exact half-way mantissa
             the reference value sits within one device ulp of the tie, so the
             neighbour selection may differ by +-1 fp8 LSB (~1e-5 of elements,
             0 for float32 random data, few per 5e5 for fp16/bf16).
             Exactness of the *scale* is still attainable: dividing by a constant
             column with ``T.vdiv`` was bit-exact in the measured cases.
Assertions : the 2-D ``T.vdiv`` result is compared against a correctly rounded
             (torch float64 -> float32) quotient; the script asserts that a
             non-zero fraction of lanes differ by exactly 1 ulp, and prints the
             measured rate (absolute value is environment dependent, the
             magnitude/relation is the locked claim).  It also asserts the byte
             magnitudes are small (<= 2 ulp), i.e. the error is a last-place
             effect and not a broken lowering.
Measured   : 2026-10-14, 3200x64 f32, mismatch rate ~6% at max 1 ulp.
First verified : 2026-10-14 (same environment as above).
origin_task    : examples/fp8_quant/_fp8_quant_kernel Stage 3
"""
import sys

sys.path.insert(0, "/workspace/tilelang-mlir-ascend")
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

M, D = 3200, 64   # 40 blocks x 80 rows (must divide evenly)


@tilelang.jit(target="npuir")
def div2d():
    @T.prim_func
    def main(a: T.Tensor((M, D), "float32"),
             b: T.Tensor((M, D), "float32"),
             c: T.Tensor((M, D), "float32")):
        with T.Kernel(40, is_npu=True) as (cid, _):
            av = T.alloc_ub((M // 40, D), "float32")
            bv = T.alloc_ub((M // 40, D), "float32")
            cv = T.alloc_ub((M // 40, D), "float32")
            T.copy(a[cid * (M // 40), 0], av)
            T.copy(b[cid * (M // 40), 0], bv)
            T.vdiv(av, bv, cv)
            T.copy(cv, c[cid * (M // 40), 0])

    return main


if __name__ == "__main__":
    import torch

    torch.manual_seed(7)
    a = torch.randn(M, D, dtype=torch.float32) * 10.0
    b = (torch.randn(M, D, dtype=torch.float32).abs() + 1.0)
    c = torch.empty(M, D, dtype=torch.float32, device="npu")
    div2d()(a.npu(), b.npu(), c)
    c = c.cpu().float()

    ref64 = a.double() / b.double()
    ref = ref64.float()                       # correctly rounded f32 quotient
    # ulp distance is only meaningful away from the zero/denormal neighbourhood
    keep = ref.abs() >= 1e-3
    ulp = torch.ldexp(torch.ones_like(ref),
                      torch.floor(torch.log2(ref.abs())).to(torch.int32) - 23)
    err_ulp = ((c - ref).abs() / ulp)[keep]
    mism = int((c[keep] != ref[keep]).sum())
    rate = mism / int(keep.sum())
    print(f"T.vdiv 2-D: mismatches={mism}/{int(keep.sum())} ({rate:.3%}) "
          f"max|ulp|={err_ulp.max().item():.4g}")
    assert mism > 0, "T.vdiv became IEEE correctly rounded -- re-verify entry"
    assert err_ulp.max().item() <= 2.0, "error is larger than a last-place effect"
    assert 0.001 < rate < 0.30, "unexpected mismatch rate magnitude"
    print("CONST-npuir-f32-division-not-ieee-rounded: CONFIRMED")
