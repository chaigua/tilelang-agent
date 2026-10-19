"""repro/CONST-vcast-rint-downcast-bitexact.py

Entry ID  : CONST-vcast-rint-downcast-bitexact (proposed; nearest home pattern-library/elementwise.md PL-1.17-subfp32-fp32-transit)
Phenomenon: On Ascend910B3 / CANN 9.0.0 (tilelang 0.1.2 dev root build), ``T.vcast(src, dst, round_mode="rint")`` for the
  DOWN-converting directions f32->f16 / f32->bf16 is bit-identical to the CPU oracle ``torch.Tensor.to(...)`` over fp32
  corners (+-inf, NaN, +-0, 65504, overflow threshold 65520, min normal 6.103515625e-05, min subnormal 5.96e-08 and its
  half = RNE tie, ties 1+2**-11 / 1+2**-8, fp32 extremes and fp32 subnormals) and mixed random domains.  The same-dtype
  direction f32->f32 is NOT the identity (silent round-to-integer; entry TRAP-vcast-same-dtype-rounds, re-verified here
  as the contrast assertion).  So an element-wise down-conversion may use vcast(rint) as a golden-aligned cast (what
  makes a bit-exact gate legal), but the source and destination dtype MUST differ; same-dtype movement must use T.copy.
Assertions: (1) f32->f16 == torch.to(float16) bit-exact; (2) f32->bf16 == torch.to(bfloat16) bit-exact (each on corners
  + mixed); (3) f32->f32 != src on >half of non-integral inputs; (4) f16->f32 == torch.to(float32) bit-exact (control).
First verified: 2026-10-16, Stage 3 of the origin task -> SUMMARY: ALL ASSERTIONS PASS.  origin_task:
  examples/multi_head_attention/_flashattn_bwd_postprocess_kernel (Stage 3, mode=first_impl, ASCEND_MODE=Developer).
  Re-verify: append below.  Run: python repro/CONST-vcast-rint-downcast-bitexact.py
Side trap : ``TILELANG_ASCEND_MODE`` must be set BEFORE the first JIT factory call, else jit_npu appends
  ``--disable-hivm-tensor-compile=true`` and any Developer-style ``T.alloc_shared`` + ``T.copy`` GM<->UB kernel dies in
  bishengir-compile with the misleading text "'hivm.hir.store' op only support store ub to gm currently!".  Standalone
  probes / repros that import tilelang directly do not inherit the operator module's os.environ.setdefault line.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)

ROWS, COLS = 16, 128  # 2048 elements; cols is a multiple of the fp16 vector width
TD = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
CORNERS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, float("inf"), float("-inf"), float("nan"), 65504.0,
    65519.0, 65520.0, 65536.0, -65504.0, -65520.0, 6.103515625e-05, 3.0517578125e-05,
    5.960464477539063e-08, 2.9802322387695312e-08, 1.0e-45, 1.0 + 2.0**-11, 1.0 + 2.0**-8,
    1.0 + 3.0 * 2.0**-11, 1.5 + 2.0**-11, 3.4028235e38, -3.4028235e38, 1.0e-40, 1.0e30,
]


@tilelang.jit(out_idx=[1], target="npuir")
def cast_kernel(rows, cols, src_dtype, dst_dtype):
    @T.prim_func
    def main(src: T.Tensor((rows, cols), src_dtype), dst: T.Tensor((rows, cols), dst_dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            sbuf = T.alloc_shared((rows, cols), src_dtype)
            dbuf = T.alloc_shared((rows, cols), dst_dtype)
            T.copy(src, sbuf)
            T.vcast(sbuf, dbuf, round_mode="rint")
            T.copy(dbuf, dst)

    return main


def run_cast(src, src_dtype, dst_dtype):
    """Run one cast on the device; ``src`` is fp32, converted to ``src_dtype`` first."""
    kernel = cast_kernel(ROWS, COLS, src_dtype, dst_dtype)
    assert tuple(src.shape) == (ROWS, COLS), f"unexpected input shape {tuple(src.shape)}"
    return kernel(src.to(TD[src_dtype]).npu()).cpu()


def bit_mismatch(x, y):
    """Differing bit patterns between two equal-shaped tensors; NaN==NaN counts equal."""
    assert x.dtype == y.dtype and x.shape == y.shape
    view = torch.int16 if x.dtype in (torch.float16, torch.bfloat16) else torch.int32
    return int(((x.view(view) != y.view(view)) & ~(torch.isnan(x) & torch.isnan(y))).sum().item())


def _inputs():
    corners = torch.tensor(CORNERS, dtype=torch.float32)
    gen, third = torch.Generator().manual_seed(0), (ROWS * COLS + 2) // 3
    mixed = torch.cat([torch.randn(third, generator=gen),
                       torch.rand(third, generator=gen) * 7.0e4 - 3.5e4,
                       torch.randn(third, generator=gen) * 2.0e-5])[: ROWS * COLS]
    idx = torch.arange(ROWS * COLS, dtype=torch.int64) % corners.numel()
    return corners[idx].reshape(ROWS, COLS).contiguous(), mixed.reshape(ROWS, COLS).contiguous()


def main():
    corner_in, mixed = _inputs()
    for dst_dtype, ref_dtype, label in [("float16", torch.float16, "f32->f16"),
                                        ("bfloat16", torch.bfloat16, "f32->bf16")]:
        for tag, values in (("corners", corner_in), ("mixed", mixed)):
            n = bit_mismatch(run_cast(values, "float32", dst_dtype),
                             torch.Tensor.to(values, ref_dtype))
            assert n == 0, f"{label}/{tag}: {n}/{ROWS * COLS} bit mismatches vs torch.to"
            print(f"[{label}/{tag}] bit-exact vs torch.to: 0/{ROWS * COLS} mismatches")

    out32 = run_cast(mixed, "float32", "float32")
    n_diff = int((out32 != mixed).sum().item())
    assert n_diff > mixed.numel() // 2, f"f32->f32 vcast close to identity ({n_diff})"
    print(f"[f32->f32 contrast] round-to-integer: {n_diff}/{mixed.numel()} differ from source")

    half = torch.Tensor.to(corner_in, torch.float16)
    n_up = bit_mismatch(run_cast(half.to(torch.float32), "float16", "float32"),
                        torch.Tensor.to(half, torch.float32))
    assert n_up == 0, f"f16->f32: {n_up} bit mismatches vs torch.to"
    print(f"[f16->f32/control] bit-exact vs torch.to: 0/{ROWS * COLS} mismatches")
    print("SUMMARY: ALL ASSERTIONS PASS")


if __name__ == "__main__":
    main()
