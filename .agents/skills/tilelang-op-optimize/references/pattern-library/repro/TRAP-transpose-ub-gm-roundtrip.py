# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: three layout/lowering costs that dominate the C4 global-pool path.

条目: pattern-library/traps-compiler.md
      TRAP-transpose-ub-gm-roundtrip
      TRAP-inner-axis-reduce-scalarize
      TRAP-copy-layout-propagation-dma-amplification
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z (Stage 4)
First verified: 2026-10-14, tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760
      (commit 417dae9) + CANN 9.0.0 + Ascend910B3

现象（msprof op 口径, N=8 C=2048 H=W=7, CH=64, 40 blocks, Task Duration median/20）:
  * control   : GM->UB 连续拷贝 + 拷回 GM                   ->   5.4 us
  * + vcast   : 整块 fp16->fp32                              ->   5.6 us  (逐元素算子几乎免费)
  * + reduce  : 追加 T.reduce_sum(dim=2)（最内轴 W=7）       -> 155.1 us (单个 op !)
                 GM_to_UB_datas 36.75 -> 336 KB/block（布局传播把连续 DMA
                 退化为 16 B/元素分散读, ~9x 扇出）
  * + transpose: 追加 T.transpose([2,1,0])                    -> 138.2 us
                 UB_to_GM_datas 48 KB/block, aiv_mte3_time 43-58 us
                 （hivm.hir.vtranspose 被降级为 UB->GM->UB 往返）
对照: (64,56)->(56,64) f16 二维转置（两边 >=8 且为 8 的倍数）在同一工具链仍走
      UB 内实现（UB_to_GM 仅 8.5 KB/block，见 spp-6x6 的 general 路径）——
      转置代价强依赖 shape/dtype 形态，不可一概而论。

断言（可运行部分，默认模式，NPU 上执行）:
  control  : 拷回结果与输入逐位一致；
  reduce   : out[...,0:H,0:1] 等于 torch 沿 W 轴求和（含 rint 舍入，容差 1e-2）；
  transpose: 两次 [2,1,0] 往返后与输入逐位一致。
性能断言（需 msprof，见下方命令）: reduce/transpose 形态 >= 10x control 的
  Task Duration，且 GM_to_UB / UB_to_GM 出现量级放大。

msprof 测量命令（人工/工具链升级时按需执行；repro_runner 只验证可运行部分）:
  msprof op --kernel-name=main --output=/tmp/repro_tr --launch-count=20 --warm-up=5 \
    --dump=off --aic-metrics=BasicInfo,PipeUtilization,Memory \
    python repro/TRAP-transpose-ub-gm-roundtrip.py --variant {control|reduce|transpose} --iters 25
  读 <out>/OPPROF_*/main/*/OpBasicInfo_*.csv 的 Task Duration(us) 与同目录
  Memory_*.csv 的 GM_to_UB_datas(KB) / UB_to_GM_datas(KB)。

Run: python repro/TRAP-transpose-ub-gm-roundtrip.py    # 三形态正确性（可运行部分）
     python repro/TRAP-transpose-ub-gm-roundtrip.py --variant reduce --iters 25   # 供 msprof 驱动
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
CH = 64
DTYPE = "float16"


def build(variant: str):
    num_cblk = C // CH
    num_logical = N * num_cblk
    num_kernels = min(num_logical, NPUUtils.get().get_aicore_num() * 2)
    num_local = (num_logical + num_kernels - 1) // num_kernels

    @tilelang.jit(out_idx=[1], target="npuir")
    def _f(ch_: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, H, W), DTYPE),
            out: T.Tensor((N, C, H, W), DTYPE),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((CH, H, W), DTYPE)
                f32 = T.alloc_ub((CH, H, W), "float32")
                mid = T.alloc_ub((CH, H, 1), "float32")
                mid16 = T.alloc_ub((CH, H, 1), DTYPE)
                tr = T.alloc_ub((W, H, CH), DTYPE)
                tr2 = T.alloc_ub((CH, H, W), DTYPE)
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        T.copy(x[nb, cb * CH : cb * CH + CH, 0:H, 0:W], flat)
                        if variant == "control":
                            T.copy(flat, out[nb, cb * CH : cb * CH + CH, 0:H, 0:W])
                        elif variant == "reduce":
                            T.vcast(flat, f32, round_mode="rint")
                            T.reduce_sum(f32, mid, dim=2)
                            T.vcast(mid, mid16, round_mode="rint")
                            T.copy(mid16, out[nb, cb * CH : cb * CH + CH, 0:H, 0:1])
                        else:  # transpose (round trip -> restores the input)
                            T.transpose(flat, tr, permutation=[2, 1, 0])
                            T.transpose(tr, tr2, permutation=[2, 1, 0])
                            T.copy(tr2, out[nb, cb * CH : cb * CH + CH, 0:H, 0:W])

        return main

    return _f(CH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", "control", "reduce", "transpose"])
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16)
    xn = x.npu()
    for v in (["control", "reduce", "transpose"] if args.variant == "all" else [args.variant]):
        k = build(v)
        with torch.no_grad():
            for _ in range(args.iters):
                out = k(xn).cpu()
        if v == "control":
            assert torch.equal(out, x), "control copy is not bit-exact"
        elif v == "reduce":
            ref = x.float().sum(-1).to(torch.float16)
            got = out[:, :, :, 0].float()
            assert (got - ref.float()).abs().max().item() < 1e-2, "inner-axis reduce mismatch"
        else:
            assert torch.equal(out, x), "transpose round trip is not bit-exact"
        print(f"[repro {v}] OK  out={tuple(out.shape)}")
    print("repro OK (runnable part; performance assertions in the header, msprof-driven)")


if __name__ == "__main__":
    main()
