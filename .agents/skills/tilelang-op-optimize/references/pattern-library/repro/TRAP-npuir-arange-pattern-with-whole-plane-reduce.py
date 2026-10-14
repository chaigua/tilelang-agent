# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: an in-kernel ``arange``-derived flat-index pattern is read back
corrupted when the same kernel also consumes it through ``T.vselect`` +
``T.reduce_min`` (whole-plane argmax).

Phenomenon (asserted below, on NPU, for int32 and fp32 carriers):
  * control kernel -- build the ``(CH, H, W)`` pattern ``k = h*W + w`` by
    ``T.arange(1-D) -> T.reshape -> T.vbrc -> T.reshape`` and copy it out:
    the pattern is exact ([0, 1, ..., H*W-1] per channel).
  * trap kernel -- same pattern, but the kernel also runs
    ``T.vcmp(f32, gbrc, "eq") -> T.vselect(eqb, kj, bigk, cand) ->
    T.reduce_min(dim=2) -> T.reduce_min(dim=1)``: the pattern is read back as
    ``w`` repeated on every row (``[0,1,2,3, 0,1,2,3, ...]``) and the derived
    first-hit index is wrong (it is a column index, not a flat index).
  The corruption is independent of how the pattern is built (2-D chain, 4-D
  chain, or a materialised ``vbrc`` straight into ``(CH, H, W)``), of the
  carrier dtype, of the reshape direction, and of inserting
  ``T.pipe_barrier("PIPE_ALL")`` around the producer/consumer.

Why it matters: the whole-plane (``out_H == out_W == 1``) argmax
specialisation of ``_adaptive_max_pool2d_with_indices_kernel`` derives its
first-hit / last-NaN index from exactly this pattern.  The *values* stay exact
while the *indices* are silently wrong, so a value-only gate cannot see it
(the trap was found because an int64 index oracle is exact).  The kernel
therefore routes ``1x1`` output grids through the general path, which is
DESIGN.md section 4.5's sanctioned semantics-preserving fallback.

Run: /workspace/venv/bin/python repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py
Exit code 0 == control exact and trap reproduced.

Entry: traps-compiler.md TRAP-npuir-arange-pattern-with-whole-plane-reduce
First verified: 2026-10-14, tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2
  (dev root build) + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3.
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z.
Registration note: this header block (entry id + version stamp + origin_task) was added by
  tilelang-skill-evolver during ED-C promotion -- the task-side repro carried the phenomenon,
  assertions and Run line but no entry id / version stamp. Body unchanged (byte-identical to the
  task-side file before this block).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

CH, H, W = 8, 3, 4
HW = H * W


def build_control(dt):
    @tilelang.jit(out_idx=[1], target="npuir")
    def f():
        @T.prim_func
        def main(x: T.Tensor((CH, H, W), dt), out_kj: T.Tensor((CH, H, W), dt)):
            with T.Kernel(1, is_npu=True) as (cid, _):
                i1 = T.alloc_ub((HW,), dt)
                i2 = T.alloc_ub((1, HW), dt)
                iw = T.alloc_ub((CH, HW), dt)
                kj = T.alloc_ub((CH, H, W), dt)
                T.arange(i1, [1], 0)
                T.reshape(i1, i2)
                T.vbrc(i2, iw)
                T.reshape(iw, kj)
                T.copy(kj, out_kj)

        return main

    return f()


def build_trap(dt):
    @tilelang.jit(out_idx=[1, 2, 3], target="npuir")
    def f():
        @T.prim_func
        def main(
            x: T.Tensor((CH, H, W), "float16"),
            out_kj: T.Tensor((CH, H, W), dt),
            out_cand: T.Tensor((CH, H, W), dt),
            out_firstk: T.Tensor((CH, 1, 1), dt),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                flat = T.alloc_ub((CH, H, W), "float16")
                f32 = T.alloc_ub((CH, H, W), "float32")
                gbrc = T.alloc_ub((CH, H, W), "float32")
                eqb = T.alloc_ub((CH, H, W), "bool")
                mid = T.alloc_ub((CH, H, 1), "float32")
                gmax = T.alloc_ub((CH, 1, 1), "float32")
                kj = T.alloc_ub((CH, H, W), dt)
                cand = T.alloc_ub((CH, H, W), dt)
                bigk = T.alloc_ub((CH, H, W), dt)
                midk = T.alloc_ub((CH, H, 1), dt)
                firstk = T.alloc_ub((CH, 1, 1), dt)
                i1 = T.alloc_ub((HW,), dt)
                i2 = T.alloc_ub((1, HW), dt)
                iw = T.alloc_ub((CH, HW), dt)

                T.copy(x, flat)
                T.vcast(flat, f32, round_mode="rint")
                T.arange(i1, [1], 0)
                T.reshape(i1, i2)
                T.vbrc(i2, iw)
                T.reshape(iw, kj)
                if dt == "int32":
                    T.vbrc(T.int32(HW), bigk)
                else:
                    T.vbrc(T.float32(HW), bigk)
                T.reduce_max(f32, mid, dim=2, clear=True)
                T.reduce_max(mid, gmax, dim=1, clear=True)
                T.vbrc(gmax, gbrc)
                T.vcmp(f32, gbrc, eqb, "eq")
                T.vselect(eqb, kj, bigk, cand)
                T.reduce_min(cand, midk, dim=2, clear=True)
                T.reduce_min(midk, firstk, dim=1, clear=True)
                T.copy(kj, out_kj)
                T.copy(cand, out_cand)
                T.copy(firstk, out_firstk)

        return main

    return f()


def main():
    torch.manual_seed(0)
    x = torch.randn(CH, H, W, dtype=torch.float16)
    x[0, 0, 0] = 5.0  # make channel 0's maximum land on a known flat index
    x[0, 2, 3] = 9.0  # flat 11
    xf = x.float().reshape(CH, HW)
    gmax = xf.max(dim=1).values
    kk = torch.arange(HW).reshape(1, HW)
    true_firstk = (
        torch.where(xf == gmax[:, None], kk, torch.full_like(kk, HW)).min(dim=1).values
    )

    for dt, tt in (("int32", torch.int32), ("float32", torch.float32)):
        expect = torch.arange(HW, dtype=tt).reshape(1, H, W).expand(CH, H, W)
        control = build_control(dt)(torch.zeros(CH, H, W, dtype=tt).npu()).cpu()
        kj, cand, firstk = [r.cpu() for r in build_trap(dt)(x.npu())]
        print(f"[{dt}] control pattern exact      : {bool(torch.equal(control, expect))}")
        print(f"[{dt}] trap pattern exact         : {bool(torch.equal(kj, expect))}")
        print(f"[{dt}] trap pattern row 0         : {kj[0].flatten().tolist()}")
        print(f"[{dt}] expected row 0             : {list(range(HW))}")
        print(
            f"[{dt}] trap first-hit index       : {firstk.flatten()[:4].tolist()} "
            f"(true {true_firstk[:4].tolist()})"
        )
        assert torch.equal(control, expect), (
            f"{dt}: control pattern is not exact -- the trap is not isolated"
        )
        assert not torch.equal(kj, expect), (
            f"{dt}: TRAP NOT reproduced -- the pattern survived the reduce chain"
        )
        assert int(firstk[0, 0, 0]) != int(true_firstk[0]), (
            f"{dt}: derived first-hit index was correct; corruption not observable"
        )

    print(
        "TRAP confirmed: the arange index pattern survives standalone but is "
        "corrupted inside the whole-plane vselect/reduce argmax chain"
    )


if __name__ == "__main__":
    main()
