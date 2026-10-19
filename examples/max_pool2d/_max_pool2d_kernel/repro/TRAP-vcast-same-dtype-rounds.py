# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: ``T.vcast`` with the **same** float dtype (f32 -> f32) does not
transfer the source -- it lowers to a round-to-integer conversion.  Use
``T.copy`` for same-dtype transfers.

Entry: traps-compiler.md ``TRAP-vcast-same-dtype-rounds`` (sibling of
``TRAP-vcast-shape-and-gm-source``, which covers the shape / GM-source
constraints only).

Phenomenon (asserted below, on NPU):
  * ``T.vcast(src_f32, dst_f32, round_mode="rint")`` with identical dtype and
    identical shape yields ``dst == round(src)`` element-wise (the fractional
    part is destroyed: -1.1258 -> -1.0, 0.8487 -> 1.0, -0.4339 -> 0.0).  No
    error, no warning -- silently wrong values as soon as the source is not
    integral.  The ``round_mode`` operand is apparently applied as an *integer*
    conversion even when both operands are already f32.
  * ``T.copy(src_f32, dst_f32)`` moves the same data bit-exactly (control /
    sanctioned form).
  * ``T.vcast`` with a *genuine* dtype change (f16 -> f32) works correctly, so
    the trap is specific to the same-dtype form, not to ``vcast`` in general.

Sanctioned form: choose the transfer by dtype at trace time -- ``T.copy`` when
the dtypes are equal, ``T.vcast`` only for an actual dtype change.
``_max_pool2d_kernel`` uses exactly that split for its float32 path (the
f32 -> f32 ``vcast`` in the C4 whole-plane branch returned 100% wrong values,
off by up to 0.49 absolute / 0.19 relative, before the switch to ``T.copy``).

Why it matters: any kernel whose comparison / accumulation domain is fp32 (e.g.
a max-pool with an fp32 compare domain) needs the fp32 input rows in UB as fp32;
the "harmless" f32 -> f32 ``vcast`` is the natural way to write that transfer,
and it silently rounds the data.

First verified / toolchain: 2026-10-14, tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c
(dev root build) + torch_npu 2.9.0.post8 + CANN 9.0.0, Ascend910B3
(origin_task: max_pool2d-_max_pool2d_kernel Stage 3, first_impl attempt 1).
Re-verification log: -.

Run: /workspace/venv/bin/python repro/TRAP-vcast-same-dtype-rounds.py
Exit code 0 == trap and sanctioned form both confirmed.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

R, CH = 4, 8


@tilelang.jit(out_idx=[2], target="npuir")
def probe(R: int, CH: int):
    """out[0] = f32->f32 vcast (trap), out[1] = f32->f32 copy (control),
    out[2] = f16->f32 vcast (genuine dtype change)."""

    @T.prim_func
    def main(
        src32: T.Tensor((R, CH), "float32"),
        src16: T.Tensor((R, CH), "float16"),
        out: T.Tensor((3, R, CH), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            s32 = T.alloc_ub((R, CH), "float32")
            s16 = T.alloc_ub((R, CH), "float16")
            dst = T.alloc_ub((R, CH), "float32")
            sentinel = T.alloc_ub((R, CH), "float32")
            T.copy(src32, s32)
            T.copy(src16, s16)
            T.vbrc(T.float32(7.0), sentinel)
            # Path A: same-dtype vcast (the trap).  dst is pre-filled with a
            # sentinel so that a no-op would also be detectable.
            T.copy(sentinel, dst)
            T.vcast(s32, dst, round_mode="rint")
            T.copy(dst, out[0, 0:R, 0:CH])
            # Path B: same-dtype copy (the control / sanctioned form).
            T.copy(sentinel, dst)
            T.copy(s32, dst)
            T.copy(dst, out[1, 0:R, 0:CH])
            # Path C: genuine dtype change (vcast is the right tool).
            T.copy(sentinel, dst)
            T.vcast(s16, dst, round_mode="rint")
            T.copy(dst, out[2, 0:R, 0:CH])

    return main


torch.manual_seed(0)
src32 = torch.randn(R, CH, dtype=torch.float32)
src16 = torch.randn(R, CH, dtype=torch.float16)
got = probe(R, CH)(src32.npu(), src16.npu()).cpu()

same_dtype_vcast, same_dtype_copy, real_cast = got[0], got[1], got[2]
print("src32            :", src32.flatten()[:6].tolist())
print("same-dtype vcast :", same_dtype_vcast.flatten()[:6].tolist())
print("round(src32)     :", torch.round(src32).flatten()[:6].tolist())
print("same-dtype vcast == round(src) :", bool(torch.equal(same_dtype_vcast, torch.round(src32))))
print("same-dtype vcast == src        :", bool(torch.equal(same_dtype_vcast, src32)))
print("same-dtype copy  == src        :", bool(torch.equal(same_dtype_copy, src32)))
print("f16 -> f32 vcast == src16      :", bool(torch.equal(real_cast, src16.float())))

assert bool(torch.equal(same_dtype_copy, src32)), "control copy is not bit-exact"
assert bool(torch.equal(real_cast, src16.float())), "f16 -> f32 vcast is broken"
assert not bool(torch.equal(same_dtype_vcast, src32)), "TRAP NOT reproduced"
assert bool(torch.equal(same_dtype_vcast, torch.round(src32))), (
    "same-dtype vcast is neither a transfer nor a round-to-integer conversion -- "
    "the destination is in an unexpected state"
)
print(
    "TRAP confirmed: f32 -> f32 vcast silently rounds to integers "
    "(T.copy is the sanctioned same-dtype transfer; vcast works for genuine "
    "dtype changes)"
)
