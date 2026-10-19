# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-out-idx-uint8-dtype-fallback

Entry      : traps-runtime.md -> TRAP-out-idx-uint8-dtype-fallback (new)
Phenomenon : ``@tilelang.jit(out_idx=[...])`` allocates the output torch tensors
             from ``tilelang/jit/jit_npu.py::_tvm_dtype_to_torch``.  That mapping
             has entries for float16/32/64, bfloat16, int8/16/32/64 and bool --
             but **no uint8** -- and `mapping.get(dtype_str, torch.float32)` falls
             back to ``torch.float32``.  A prim_func whose output buffer is
             ``uint8`` therefore receives a float32 tensor of the same logical
             shape: the kernel writes 1-byte lanes into a 4-byte-per-element
             buffer and the returned tensor is unusable as a byte carrier.
Assertions : (1) with ``out_idx`` the returned output dtype is float32 even though
             the prim_func declares uint8 (trap reproduced);
             (2) passing an explicitly allocated ``torch.uint8`` tensor as an
             ordinary input argument round-trips the bytes bit-exactly
             (workaround: no ``out_idx`` for byte carriers).
Scope      : any npuir kernel whose output is a raw byte carrier (fp8 software
             encoding, quantization payloads).
First verified : 2026-10-14, tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760,
                 CANN 9.0.0, Ascend910B3, torch 2.9.0 torch_npu 2.9.0.post8
origin_task    : examples/fp8_quant/_fp8_quant_kernel Stage 3
"""
import sys

sys.path.insert(0, "/workspace/tilelang-mlir-ascend")
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

R, D = 256, 64


@tilelang.jit(out_idx=[1], target="npuir")
def with_out_idx():
    @T.prim_func
    def main(x: T.Tensor((R, D), "float16"), y: T.Tensor((R, D), "uint8")):
        with T.Kernel(4, is_npu=True) as (cid, _):
            xv = T.alloc_ub((64, D), "float16")
            yv = T.alloc_ub((64, D), "uint8")
            T.copy(x[cid * 64, 0], xv)
            for i, j in T.Parallel(64, D):
                yv[i, j] = T.cast(xv[i, j], "uint8")
            T.copy(yv, y[cid * 64, 0])

    return main


@tilelang.jit(target="npuir")
def explicit_out():
    @T.prim_func
    def main(x: T.Tensor((R, D), "float16"), y: T.Tensor((R, D), "uint8")):
        with T.Kernel(4, is_npu=True) as (cid, _):
            xv = T.alloc_ub((64, D), "float16")
            yv = T.alloc_ub((64, D), "uint8")
            T.copy(x[cid * 64, 0], xv)
            for i, j in T.Parallel(64, D):
                yv[i, j] = T.cast(xv[i, j], "uint8")
            T.copy(yv, y[cid * 64, 0])

    return main


if __name__ == "__main__":
    import torch

    torch.manual_seed(3)
    # integer-valued fp16 inputs below 128 so the i8-lowered uint8 carrier does
    # not saturate (uint8 is lowered to i8 by the codegen)
    x = torch.randint(0, 120, (R, D)).to(torch.float16)

    got = with_out_idx()(x.npu())
    print("out_idx dtype:", got.dtype, "shape:", tuple(got.shape))
    assert got.dtype == torch.float32, (
        "uint8 output buffer no longer falls back to float32 -- re-verify entry")

    y = torch.empty(R, D, dtype=torch.uint8, device="npu")
    explicit_out()(x.npu(), y)
    ref = x.to(torch.float32).to(torch.uint8)
    assert torch.equal(y.cpu(), ref), "explicit uint8 carrier is not bit-exact"
    print("explicit uint8 carrier: bit-exact OK")
    print("TRAP-out-idx-uint8-dtype-fallback: PHENOMENON CONFIRMED + WORKAROUND OK")
