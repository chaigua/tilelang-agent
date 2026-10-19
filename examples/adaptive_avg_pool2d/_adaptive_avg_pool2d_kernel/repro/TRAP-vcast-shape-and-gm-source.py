# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: two `T.vcast` constraints hit while building the
adaptive-average-pool accumulator path (both are real lowering failures, not
style preferences).

TRAP-1 -- `T.vcast` requires equal (broadcast-compatible) src/dst shapes; it
does NOT reshape.  Rounding an `(out_W, CH)` fp32 accumulator back to the
output dtype in NCHW order therefore needs an explicitly allocated same-shape
buffer plus `T.transpose([1, 0])` for the layout change:
    T.vcast(acc_(out_W,CH)_fp32, acc_t_(CH,out_W)_dtype)
  -> "The input and output shapes do not match for broadcast."

TRAP-2 -- `T.vcast` whose *source* is a global (GM) memref region
(`T.vcast(x[...], ub_fp32)`) lowers to a `hivm.hir.vcast` with a
`tensor.empty()` destination and aborts later in the BiShengHIR pipeline:
    "NPU IR opt failed: ... error: Failed to run BiShengHIR pipeline"
Independently of a following transpose / copy.  Sanctioned form: always
stage GM -> UB with a same-dtype `T.copy` first, then `T.vcast` UB -> UB
(this is what `_adaptive_avg_pool2d_kernel` does for every row / plane load).

Companion behaviours verified in the same probe (both on the critical path):
  * `T.vmul` accepts a *runtime* let-bound fp32 scalar as `src2` (the C1
    reciprocal multiply `sum * fl(1/(bh*bw))` of DESIGN.md section 1.6.1 --
    the docs' "scalar operand must be defined inside T.Kernel" does not imply
    "compile-time constant");
  * a 3D `T.reduce_sum` chain (dim=2 then dim=1) works -- the primitive the C4
    global-pooling specialisation (DESIGN.md section 1.6.1 C4) relies on.

Run:
    /workspace/venv/bin/python repro/TRAP-vcast-shape-and-gm-source.py
Exit code 0 == both traps and all companion behaviours confirmed.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import traceback

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M, N = 6, 8


# --- TRAP-1: mismatched vcast shapes ----------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def shape_trap_kernel(M: int, N: int, dtype: str = "float16"):
    """vcast fp32 (M,N) -> dtype (N,M): shape mismatch, lowering aborts."""

    @T.prim_func
    def main(x: T.Tensor((M, N), dtype), y: T.Tensor((N, M), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), "float32")
            b = T.alloc_ub((N, M), dtype)
            T.vcast(x, a, round_mode="rint")
            T.vcast(a, b, round_mode="rint")
            T.copy(b, y)

    return main


# --- TRAP-2: vcast with a GM source ----------------------------------------
@tilelang.jit(out_idx=[1], target="npuir")
def gm_source_trap_kernel(M: int, N: int, dtype: str = "float16"):
    """vcast straight from a GM region: BiShengHIR pipeline failure."""

    @T.prim_func
    def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), "float32")
            a16 = T.alloc_ub((M, N), dtype)
            T.vcast(x, a, round_mode="rint")  # <- GM source
            T.vcast(a, a16, round_mode="rint")
            T.copy(a16, y)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def fixed_kernel(M: int, N: int, dtype: str = "float16"):
    """T.copy GM->UB, then vcast UB->UB; runtime scalar; transpose out."""

    @T.prim_func
    def main(x: T.Tensor((M, N), dtype), y: T.Tensor((N, M), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((M, N), dtype)
            a = T.alloc_ub((M, N), "float32")
            a16 = T.alloc_ub((M, N), dtype)
            b = T.alloc_ub((N, M), dtype)
            # runtime int scalar -> let-bound fp32 scalar (not a constant)
            d = cid + 7
            inv = 1.0 / T.cast(d, "float32")
            T.copy(x, src)  # GM -> UB, same dtype (required staging)
            T.vcast(src, a, round_mode="rint")  # UB -> UB
            T.vmul(a, inv, a)
            T.vcast(a, a16, round_mode="rint")
            T.transpose(a16, b, permutation=[1, 0])
            T.copy(b, y)

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def reduce3d_kernel(C: int, H: int, W: int, dtype: str = "float16"):
    """3D reduce chain (dim=2 then dim=1) == whole-plane mean."""

    @T.prim_func
    def main(x: T.Tensor((C, H, W), dtype), y: T.Tensor((C, 1, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            flat = T.alloc_ub((C, H, W), dtype)
            flat32 = T.alloc_ub((C, H, W), "float32")
            mid = T.alloc_ub((C, H, 1), "float32")
            acc = T.alloc_ub((C, 1, 1), "float32")
            g16 = T.alloc_ub((C, 1, 1), dtype)
            inv = 1.0 / float(H * W)
            T.copy(x, flat)
            T.vcast(flat, flat32, round_mode="rint")
            T.reduce_sum(flat32, mid, dim=2)
            T.reduce_sum(mid, acc, dim=1)
            T.vmul(acc, inv, acc)
            T.vcast(acc, g16, round_mode="rint")
            T.copy(g16, y)

    return main


def _expect_failure(fn, needle, label):
    try:
        fn()
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        assert needle in tb, f"{label}: expected {needle!r} in traceback, got:\n{tb[-1500:]}"
        print(f"[TRAP] CONFIRMED ({label}): {needle}")
    else:
        raise AssertionError(f"{label}: compiled, but the trap was expected to fail")


def main():
    # --- TRAP-1 -------------------------------------------------------------
    _expect_failure(
        lambda: shape_trap_kernel(M, N),
        "shapes do not match for broadcast",
        "vcast shape mismatch",
    )

    # --- TRAP-2 -------------------------------------------------------------
    _expect_failure(
        lambda: gm_source_trap_kernel(M, N),
        "Failed to run BiShengHIR pipeline",
        "vcast with a GM source",
    )

    # --- sanctioned form + runtime scalar + transpose -----------------------
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32).to(torch.float16)
    y = fixed_kernel(M, N)(x.npu()).cpu()
    ref = (x.float() / 7.0).to(torch.float16).transpose(0, 1).contiguous()
    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
    md = (y.float() - ref.float()).abs().max().item()
    print(
        "[FIX]  CONFIRMED: T.copy GM->UB + T.vcast UB->UB + runtime let-bound "
        f"scalar as T.vmul src2 + T.transpose round-trip (max_diff={md:.3e})"
    )

    # --- 3D reduce chain (C4 global-pool primitive) -------------------------
    C, H, W = 8, 5, 7
    x3 = torch.randn(C, H, W, dtype=torch.float32).to(torch.float16)
    y3 = reduce3d_kernel(C, H, W)(x3.npu()).cpu()
    ref3 = x3.float().mean(dim=(1, 2), keepdim=True).to(torch.float16)
    torch.testing.assert_close(y3, ref3, rtol=1e-3, atol=1e-3)
    md3 = (y3.float() - ref3.float()).abs().max().item()
    print(f"[FIX]  CONFIRMED: 3D T.reduce_sum chain (dim=2 then dim=1) max_diff={md3:.3e}")

    print("\nvcast-shape / vcast-GM-source REPRO PASS")


if __name__ == "__main__":
    main()
