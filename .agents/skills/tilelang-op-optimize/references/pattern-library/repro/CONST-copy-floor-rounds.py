# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-copy-floor-rounds.py -- global 类拷贝地板由「每核串行轮数」主导（delta 形态）

条目: pattern-library/elementwise.md PL-1.6-copy-floor（update：轮数扫描维度）
关联: constants.md CONST-aiv-time-decomposition（Task ≈ max(mte2,vec)+scalar）

现象/优化点:
  整平面连续搬运的 copy-floor 不由带宽/流量决定，而由**每核串行轮数**
  `num_local = ceil(num_logical / num_kernels)` 决定（同流量、同 DMA 粒度）：

    | 栅格（resnet-global [8,2048,7,7] fp16, CH×49 元素/轮） | 轮数 | Task(us) | mte2(us) |
    |---|---|---:|---:|
    | CH=32（512 单元 / 40 核） | 13 | 10.99 | 3.85 |
    | CH=64（256 单元 / 40 核） | 7  | 6.85  | 2.13 |
    | CH=128（128 单元 / 40 核）| 4  | 5.09  | 1.21 |

  流量恒为 1.6 MB ⇒ 带宽差 3.3x，但 mte2 只差 3.2x 而 Task 差 2.2x：差额全部是
  **每轮 DMA→落盘 的串行延迟**（≈0.7-0.9 µs/轮）。故「copy-floor」必须按**将要
  落地的轮数**标定，不能只测一个粒度就外推；70% 校准线 = 标定地板 ÷ 0.7 也随之
  随轮数变化（6.85/0.7=9.79 µs vs 5.09/0.7=7.27 µs）。

优化见解摘要（机制归因）:
  纯 Vector persistent kernel 每轮是「GM→UB 拷贝 → 落盘」的依赖链，轮间无重叠
  （自动多缓冲在此形态未生效，实测关/开无差异）；每轮固定开销摊销不掉，因此
  **减少轮数（加大每轮 CH）是降低地板的唯一杠杆**，直到受 UB 或归约单价限制
  （见 CONST-reduce-ub-per-element）。

断言: delta 形态（效应只在完整栅格规模显现）——本文件只做 py_compile 语法校验；
  测量部分需 NPU + msprof，命令见头部。

首次验证: 2026-10-15 adaptive_max_pool2d-_adaptive_max_pool2d_kernel Stage 4
  (tilelang 0.1.2+489385e9 / CANN 9.0.0 / Ascend910B3)
  测量命令（在含 `calib_bench.py` 的任务工作区）:
    msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off \
      --aic-metrics=BasicInfo,PipeUtilization,Memory \
      python calib_bench.py --case global --ch {32|64|128}
重验记录: (追加式)
"""

# ---- 轮数自变量骨架（同流量、同 DMA 粒度，只改 CH ⇒ 只改轮数）----
CALIB_TEMPLATE = """
import tilelang.language as T

def copy_floor(x, out, CH, PLAIN, num_cblk, n, num_kernels, num_local):
    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
        tile = T.alloc_ub((CH, 1, PLAIN), "float16")
        for t in T.serial(num_local):          # <- the rounds variable
            uid = t * num_kernels + cid
            if uid < n * num_cblk:
                nb = uid // num_cblk
                cb = uid % num_cblk
                real_c = T.min(CH, C_IN - cb * CH)
                T.copy(x[nb, cb * CH: cb * CH + real_c, 0:1, 0:PLAIN],
                       tile[0:real_c, 0:1, 0:PLAIN])
                # 1-element store keeps the copy alive without adding the
                # strided-column write amplification of the real epilogue.
                T.copy(tile[0:1, 0:1, 0:1], out[nb, cb * CH: cb * CH + 1, 0:1, 0:1])
"""

# num_local is the single independent variable:
#   n=8, C_IN=2048, PLANE=49, num_cblk=ceil(C_IN/CH), num_kernels=min(n*num_cblk, 40)
#   CH=32  -> num_cblk=64  -> units=512 -> num_local=13
#   CH=64  -> num_cblk=32  -> units=256 -> num_local=7
#   CH=128 -> num_cblk=16  -> units=128 -> num_local=4
ROUNDS_TABLE = """
CH=32:  units=512 num_local=13   Task=10.99us mte2=3.85us
CH=64:  units=256 num_local=7    Task= 6.85us mte2=2.13us
CH=128: units=128 num_local=4    Task= 5.09us mte2=1.21us
"""

if __name__ == "__main__":
    import py_compile
    import tempfile

    src = ("C_IN = 2048\n" + CALIB_TEMPLATE).replace("x, out, CH,", "x, out, CH,")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        py_compile.compile(path, doraise=True)
        print("[CONST-copy-floor-rounds] calibration skeleton: py_compile PASS")
    finally:
        import os

        os.unlink(path)
    print(ROUNDS_TABLE.strip())
    print("NOTE: the Task/mte2 values are NPU+msprof measurements (see header).")
