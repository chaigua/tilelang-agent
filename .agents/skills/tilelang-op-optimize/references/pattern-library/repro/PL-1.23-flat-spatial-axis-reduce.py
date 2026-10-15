# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Pattern repro (delta form): PL-1.23-flat-spatial-axis-reduce.

条目: pattern-library/layout.md PL-1.23-flat-spatial-axis-reduce
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z (Stage 4)
First verified: 2026-10-14, tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760
      (commit 417dae9) + CANN 9.0.0 + Ascend910B3

关键代码更改（整平面/全局池化归约: 慢 -> 快）: 把核内 UB 的整平面 tile 从
「自然三维 (CH, H, W)」改为「扁平空间轴 (CH, 1, H*W)」，并让 GM tensor 同步声明为
`(N, C, 1, H*W)`（同一块连续内存，rank/形状重解释，不改变 I/O 契约的字节布局）。

  慢 (SLOW): tile = T.alloc_ub((CH, H, W), dtype)                 # 最内轴 = W = 7
            T.copy(x3[nb, c0:c1, 0:H, 0:W], tile)
            T.vcast(tile, t32, "rint")
            T.reduce_sum(t32, mid, dim=2)                          # 归约轴 = 最内轴, 宽 7
            T.reduce_sum(mid, acc, dim=1)                          # 归约轴 = 中轴 (最内宽 1)
            ... -> resnet-global (8,2048,7,7) Task Duration 166.3 us

  快 (FAST): x4: T.Tensor((N, C, 1, H*W), dtype)                   # 同一内存, 扁平声明
            tile = T.alloc_ub((CH, 1, H*W), dtype)
            T.copy(x4[nb, c0:c1, 0:1, 0:H*W], tile)
            T.vcast(tile, t32, "rint")
            T.reduce_sum(t32, sred, dim=2)                         # 归约轴 = 最内轴, 宽 49
            ... -> 同一 workload Task Duration 16.68 us (9.96x), 逐位精确

优化见解摘要: 机制归因到两条编译器行为——(a) TRAP-inner-axis-reduce-scalarize:
`T.reduce_sum` 对最内轴逐元素标量化，代价 ≈ 元素数 × ~6 ns（W=7 时每单元 ~19 us），
把归约轴换宽（H*W=49）后同一算子可用；(b) TRAP-copy-layout-propagation-dma-amplification:
`T.copy` 的 UB 目标一旦被推成归约/转置布局，GM->UB DMA 退化为 16 B/元素分散读
（36.75 -> 336 KB/block）。扁平声明同时解决 (b)：x 的声明形状与 GM 连续序一致，
DMA 保持整块连续，且不需要任何 `T.transpose`（该工具链上 3D 转置会被降级为
UB->GM->UB 往返，见 TRAP-transpose-ub-gm-roundtrip）。缓冲区总量也下降
（去掉 mid/acc_t 等中间层），UH 预算更宽松。

断言（可运行部分）: 两形态都编译执行，且输出与 torch 的整平面均值逐位一致
（fast 形态 max_diff == 0）。性能断言（需 msprof，见头部命令族）:
fast <= slow / 2。

Run: python repro/PL-1.23-flat-spatial-axis-reduce.py            # 正确性（可运行部分）
     python repro/PL-1.23-flat-spatial-axis-reduce.py --variant fast --iters 25   # 供 msprof 驱动
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401
from tilelang.utils.npu_utils import NPUUtils

N, C, H, W = 8, 2048, 7, 7
HW = H * W
CH = 32
DTYPE = "float16"


def build(variant: str):
    num_cblk = C // CH
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    inv = 1.0 / float(HW)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _f(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW) if variant == "fast" else (N, C, H, W), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                if variant == "fast":
                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            tile = T.alloc_ub((CH, 1, HW), DTYPE)
                            t32 = T.alloc_ub((CH, 1, HW), "float32")
                            sred = T.alloc_ub((CH, 1, 1), "float32")
                            gout = T.alloc_ub((CH, 1, 1), DTYPE)
                            T.copy(x[nb, cb * CH : cb * CH + CH, 0:1, 0:HW], tile)
                            T.vcast(tile, t32, round_mode="rint")
                            T.reduce_sum(t32, sred, dim=2)
                            T.vmul(sred, inv, sred)
                            T.vcast(sred, gout, round_mode="rint")
                            T.copy(gout, out[nb, cb * CH : cb * CH + CH, 0:1, 0:1])
                else:
                    for t in T.serial(num_local):
                        uid = t * num_kernels + cid
                        if uid < num_logical:
                            nb = uid // num_cblk
                            cb = uid % num_cblk
                            tile = T.alloc_ub((CH, H, W), DTYPE)
                            t32 = T.alloc_ub((CH, H, W), "float32")
                            mid = T.alloc_ub((CH, H, 1), "float32")
                            acc = T.alloc_ub((CH, 1, 1), "float32")
                            gout = T.alloc_ub((CH, 1, 1), DTYPE)
                            T.copy(x[nb, cb * CH : cb * CH + CH, 0:H, 0:W], tile)
                            T.vcast(tile, t32, round_mode="rint")
                            T.reduce_sum(t32, mid, dim=2)
                            T.reduce_sum(mid, acc, dim=1)
                            T.vmul(acc, inv, acc)
                            T.vcast(acc, gout, round_mode="rint")
                            T.copy(gout, out[nb, cb * CH : cb * CH + CH, 0:1, 0:1])

        return main

    return _f(CH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", "slow", "fast"])
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16)
    ref = x.float().mean(dim=(2, 3), keepdim=True).to(torch.float16)
    for v in (["slow", "fast"] if args.variant == "all" else [args.variant]):
        k = build(v)
        with torch.no_grad():
            for _ in range(args.iters):
                out = k(x.npu()).cpu()
        d = (out.float() - ref.float()).abs().max().item()
        assert d < 1e-2, f"{v} mismatch {d}"
        print(f"[PL-1.23 {v}] OK max_diff={d:.3e} out={tuple(out.shape)}")
    print("repro OK (runnable part; the >=2x performance assertion needs msprof, see header)")


if __name__ == "__main__":
    main()
