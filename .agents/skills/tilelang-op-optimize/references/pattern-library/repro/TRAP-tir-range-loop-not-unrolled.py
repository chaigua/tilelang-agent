# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: a bare `for i in range(K)` inside `@T.prim_func` is NOT a
Python-level compile-time unroll -- it becomes a TIR serial loop whose loop
variable is a `tir.Var`.

条目: pattern-library/traps-compiler.md TRAP-tir-range-loop-not-unrolled
      （ED-C 机械转正自 examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/
      repro/TRAP-tir-range-loop-not-unrolled.py，2026-10-13 蒸馏，逐字节拷贝）
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z
First verified: 2026-10-13, tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b
      (dev root build), CANN 9.0.0, Ascend910B3
      （Stage 3 attempt 1 trace-time 失败 + 本 repro 全过 exit 0）
断言: ① `range` 循环变量为 `tir.Var`（Python 列表索引在 trace 期报
      `list indices must be integers or slices, not Var`）；② 静态界
      `T.serial(max)` + 运行时 `if k < extent` 守卫形态编译且 max_diff=0。
重验记录: (追加式；工具链变更后 trap 不再复现 = 条目被推翻的信号)

Why this matters
----------------
DESIGN.md for `_adaptive_avg_pool2d_kernel` plans "Python `range` compile-time
expansion" for the W-bin / output-column loops (DESIGN.md section 1.6.2 row #8
and section 6.1: "编译期展开为向量 op 序列").  On this toolchain the tilelang
frontend parses `for x in range(n)` as a serial TIR loop, so:
  * Python-level indexing of a compile-time list (a precomputed bin-bound
    table, `bins_w[ow]`) raises
    `list indices must be integers or slices, not Var`;
  * `int(...)` / `float(...)` conversion of the loop var fails likewise;
  * slice offsets built from the loop var stay runtime expressions.

Sanctioned replacement (used by the shipped kernel)
---------------------------------------------------
Mirror the source GPU kernel's form (and DESIGN.md section 0.4 #1): static
`T.serial(max_extent)` bound plus a runtime `if k < extent` guard, with the
bin bounds computed inline from the loop var (`adaptive_bin(o, ...)`).  The
same form is used for the H and W bin axes.

Run:
    /workspace/venv/bin/python repro/TRAP-tir-range-loop-not-unrolled.py
Exit code 0 == the trap and the sanctioned fix are both confirmed.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import traceback

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401


def adaptive_bin(o, size_in: int, size_out: int):
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def max_adaptive_bin_extent(size_in: int, size_out: int) -> int:
    ext = (adaptive_bin(o, size_in, size_out) for o in range(size_out))
    return max(e - s for s, e in ext)


M, SIZE_IN, SIZE_OUT = 8, 7, 3
_DTYPE = "float16"
_BINS = [adaptive_bin(o, SIZE_IN, SIZE_OUT) for o in range(SIZE_OUT)]

N, C, H, W, OH, OW = 1, 8, 4, 5, 2, 2


@tilelang.jit(out_idx=[1], target="npuir")
def trap_kernel(M: int, size_in: int, size_out: int, dtype: str = "float16"):
    """Python-level list index of the loop var: fails at trace time."""

    @T.prim_func
    def main(x: T.Tensor((M, size_in), dtype), y: T.Tensor((M, size_out), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, _):
            row = T.alloc_ub((1, size_in), "float32")
            acc = T.alloc_ub((1, size_out), "float32")
            T.clear(acc)
            T.vcast(x[cid : cid + 1, 0:size_in], row, round_mode="rint")
            for o in range(size_out):
                s, e = _BINS[o]  # <- o is a tir.Var, not a Python int
                for k in range(e - s):
                    T.vadd(
                        acc[0:1, o : o + 1],
                        row[0:1, s + k : s + k + 1],
                        acc[0:1, o : o + 1],
                    )
            T.vcast(acc, y[cid : cid + 1, 0:size_out], round_mode="rint")

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def fixed_kernel(N: int, C: int, H: int, W: int, OH: int, OW: int, dtype: str = "float16"):
    """Static bound + runtime guard form (as shipped): compiles and is exact."""
    max_bin_h = max_adaptive_bin_extent(H, OH)
    max_bin_w = max_adaptive_bin_extent(W, OW)

    @T.prim_func
    def main(x: T.Tensor((N, C, H, W), dtype), out: T.Tensor((N, C, OH, OW), dtype)):
        with T.Kernel(N * OH, is_npu=True) as (cid, _):
            acc = T.alloc_ub((OW, C), "float32")
            row_src = T.alloc_ub((C, W), dtype)
            row_dst = T.alloc_ub((W, C), dtype)
            row_f32 = T.alloc_ub((W, C), "float32")
            acc16 = T.alloc_ub((OW, C), dtype)
            acc_t = T.alloc_ub((C, OW), dtype)
            n_i = cid // OH
            oh = cid % OH
            ih0, ih1 = adaptive_bin(oh, H, OH)
            bh = ih1 - ih0
            T.clear(acc)
            for kh in T.serial(max_bin_h):
                if kh < bh:
                    T.copy(x[n_i, 0:C, ih0 + kh, 0:W], row_src)
                    T.transpose(row_src, row_dst, permutation=[1, 0])
                    T.vcast(row_dst, row_f32, round_mode="rint")
                    for ow in T.serial(OW):
                        iw0, iw1 = adaptive_bin(ow, W, OW)
                        bw = iw1 - iw0
                        for kw in T.serial(max_bin_w):
                            if kw < bw:
                                T.vadd(
                                    acc[ow : ow + 1, 0:C],
                                    row_f32[iw0 + kw : iw0 + kw + 1, 0:C],
                                    acc[ow : ow + 1, 0:C],
                                )
            for ow in T.serial(OW):
                iw0, iw1 = adaptive_bin(ow, W, OW)
                inv = 1.0 / T.cast(bh * (iw1 - iw0), "float32")
                T.vmul(acc[ow : ow + 1, 0:C], inv, acc[ow : ow + 1, 0:C])
            T.vcast(acc, acc16, round_mode="rint")
            T.transpose(acc16, acc_t, permutation=[1, 0])
            T.copy(acc_t, out[n_i, 0:C, oh, 0:OW])

    return main


def main():
    # --- 1. the trap: range() loop var is a tir.Var -------------------------
    try:
        trap_kernel(M, SIZE_IN, SIZE_OUT, _DTYPE)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        needle = "list indices must be integers or slices, not Var"
        assert needle in tb, tb[-2000:]
        first = [ln for ln in tb.splitlines() if needle in ln][0].strip()
        print("[TRAP] CONFIRMED: `for o in range(n)` loop var is a tir.Var;")
        print("       Python list indexing of it fails at trace time:", first)
    else:
        raise AssertionError(
            "trap_kernel compiled -- the `range` loop was Python-unrolled on this "
            "toolchain; re-verify the DESIGN.md section 1.6.2 #8 assumption"
        )

    # --- 2. the sanctioned fix: static bound + runtime guard ----------------
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16)
    kernel = fixed_kernel(N, C, H, W, OH, OW, _DTYPE)
    y = kernel(x.npu()).cpu()

    bins_h = [adaptive_bin(o, H, OH) for o in range(OH)]
    ref = torch.stack(
        [
            torch.stack([x[:, :, hs:he, ws:we].float().mean(dim=(2, 3)) for ws, we in _BINS_W()], dim=2)
            for hs, he in bins_h
        ],
        dim=2,
    ).to(torch.float16)
    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
    md = (y.float() - ref.float()).abs().max().item()
    print(
        f"[FIX]  CONFIRMED: static `T.serial(max_bin)` + `if k < bw` guard "
        f"compiles and matches (max_diff={md:.3e})"
    )
    print("\nTRAP REPRO PASS (trap confirmed + sanctioned fix verified)")


def _BINS_W():
    return [adaptive_bin(o, W, OW) for o in range(OW)]


if __name__ == "__main__":
    main()
