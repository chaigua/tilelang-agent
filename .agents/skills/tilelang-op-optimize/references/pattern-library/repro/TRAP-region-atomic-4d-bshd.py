#!/usr/bin/env python3
"""TRAP-region-atomic-4d-bshd: rank-reduced 4-D BSHD 区域原子静默写错。

`T.atomic_add(dst[bz, r0:r0+ROWS, h, 0:D], src_ub, [ROWS, D])` 在 dst=(B,S,H,D)、
H>1、D != H*D 时把目标区当作**连续平铺块**写出：

  static  extent -> 目标区只 256/2048 cell 命中，区外 1792 cell 被写坏
  dynamic extent -> 前端报错 `int() argument must be ... not 'Var'`

可用替代 = 把张量声明为 (B*S, H*D) 视图后写 rank-2 显式切片跨步区域（7x 更快）。
运行：python TRAP-region-atomic-4d-bshd.py --mode static|dynamic
      打印 TRAP reproduced / FIXED。

origin_task: TileOPs-_gqa_bwd_kernel-20261018T145922Z（DESIGN §0.6 R3 的 910B3 结论在 910B4 重验）
toolchain:   tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4 / 2026-10-18
First verified: 2026-10-18, tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4
"""
"""Probe: rank-reduced 4-D region `T.atomic_add` on the strided BSHD target.

Stage 1 recorded this form as "compiles with a static extent but silently
writes the wrong region" (910B3 + CANN 9.0.0).  Re-verified here on the Stage 4
device because the per-row fallback is 7x more expensive than a region atomic
(probe_atomic.py: 224 ns/row vs 32 ns/row).

Modes:
  static : T.atomic_add(dst[bz, r0:r0+ROWS, h, 0:D], src, [ROWS, D])
  dynamic: same with a runtime extent (T.min(ROWS, S-r0))

Assertions (exact): every cell of dst[0, r0:r0+ROWS, h, :] == 1.0, every other
cell (all bz, all s, all h != target) == 0.0.  Prints PASS/FAIL per mode.
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

B, S, H, D = 2, 64, 8, 64
ROWS = 32
R0 = 16
H_TARGET = 5  # interior; H > 1 and D != H*D so a contiguous misinterpretation shows


@tilelang.jit(target="npuir")
def _probe(mode: str):
    @T.prim_func
    def region_atomic(
        src: T.Tensor((ROWS, D), "float32"),
        dst: T.Tensor((B, S, H, D), "float32"),
    ) -> None:
        with T.Kernel(1, is_npu=True) as (cid, _):
            s = T.alloc_shared((ROWS, D), "float32")
            T.copy(src, s)
            if mode == "static":
                T.atomic_add(dst[0, R0 : R0 + ROWS, H_TARGET, 0:D], s, [ROWS, D])
            else:
                real = T.min(ROWS, S - R0)
                T.atomic_add(dst[0, R0 : R0 + real, H_TARGET, 0:D], s, [real, D])

    return region_atomic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="static", choices=["static", "dynamic"])
    args = ap.parse_args()
    src = torch.ones(ROWS, D, dtype=torch.float32, device="npu")
    dst = torch.zeros(B, S, H, D, dtype=torch.float32, device="npu")
    _probe(args.mode)(src, dst)
    torch.npu.synchronize()
    d = dst.cpu()
    mask = torch.zeros_like(d)
    mask[0, R0 : R0 + ROWS, H_TARGET, :] = 1.0
    inside = int((d[mask.bool()] == 1.0).sum().item())
    n_inside = int(mask.sum().item())
    outside = int((d[mask == 0] != 0.0).sum().item())
    print(f"[{args.mode}] inside {inside}/{n_inside} == 1.0 ; outside_nonzero={outside}")
    print(f"[{args.mode}] {'PASS' if inside == n_inside and outside == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
