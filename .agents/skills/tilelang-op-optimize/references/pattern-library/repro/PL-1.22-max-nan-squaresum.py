# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.22-max-nan-squaresum.py -- C4/整平面 max 归约的 NaN 追踪等代价形态

条目: pattern-library/layout.md PL-1.22-maxreduce-nan-tracking（update：C4/整平面域）
关联: constants.md CONST-reduce-ub-per-element（本形态的 UB 单价，32 vs 96-128 B/元素）

现象/优化点:
  整平面（全局池化）NaN 追踪有两种数学等价写法，但 UB 代价差 3-4x：
    慢（掩码链）: vcmp(x, x, "eq") -> vcast(bool -> f32) -> reduce_min(f32 mask)
                  BishengHIR 归约 lowering 需 96-128 B / 平面元素（CH 被压到 <=32）
    快（平方和）: vmul(x, x, x)  [原位] -> reduce_sum -> isnan(sum)
                  需 32 B / 平面元素（解锁 CH<=120）
  数学等价：x*x 仅在该元素为 NaN 时为 NaN，其余值 >= 0 ⇒ 求和不会因 +inf-inf
  产生假 NaN（这正是不能直接 reduce_sum(x) 的原因）；±inf 的平方为 +inf，求和
  仍为 +inf（非 NaN）⇒ 正确判为「无 NaN」。

优化见解摘要（机制归因）:
  NaN 是「吸收元」，任何含 NaN 的加法链结果都是 NaN ⇒ 求和天然可作 NaN 探测器；
  但「原始值求和」会被 +inf + (-inf) 污染成假 NaN，故先平方把值域压到 [0,+inf]。
  相比之下 vcmp 掩码链要额外物化 bool 平面 + fp32 掩码平面，触发 BishengHIR 归约
  lowering 的高价 UB 记账（见 CONST-reduce-ub-per-element），是真正的瓶颈。

断言（本文件为 torch CPU 数值等价断言，不需要 NPU）:
  在含 NaN / ±inf / 二者同平面的随机数据上，isnan(sum(x*x)) 与「平面含 NaN」
  逐元素一致；并显式展示 sum(x) 在 {+inf,-inf} 平面上的假阳性。

首次验证: 2026-10-15 adaptive_max_pool2d-_adaptive_max_pool2d_kernel Stage 4
  (tilelang 0.1.2+489385e9 / CANN 9.0.0 / Ascend910B3)
  kernel 侧实测: global-1x1 320.83 -> 19.16 us（掩码链 96-128 B/元素 -> 平方和
  32 B/元素；CH 上限 32 -> 120；落档 CH=64）。
重验记录: (追加式)
"""

import torch

PLANE = 49  # 整平面元素数（H*W）


def nan_by_mask(x: torch.Tensor) -> torch.Tensor:
    """vcmp(x,x,'eq') 的 IEEE 补集：mask==0 <=> x 是 NaN。"""
    eq = (x == x).to(torch.float32)
    return eq.min(dim=-1).values == 0.0


def nan_by_squaresum(x: torch.Tensor) -> torch.Tensor:
    """isnan(reduce_sum(x*x))：本条目采纳的等代价形态。"""
    q = (x * x).sum(dim=-1)
    return torch.isnan(q)


def main():
    torch.manual_seed(0)
    rows = []
    for _ in range(64):
        rows.append(torch.randn(4, PLANE, dtype=torch.float32) * 100.0)
    x = torch.cat(rows, dim=0)
    # inject the semantic corner cases
    x[0, 0] = float("nan")
    x[1, :] = float("inf")            # +inf plane
    x[2, :] = float("-inf")           # -inf plane
    x[3, 3] = float("inf")
    x[3, 7] = float("-inf")           # {+inf,-inf}, no NaN
    x[4, 1] = float("nan")
    x[4, 5] = float("inf")            # NaN and +inf in the same plane
    x[5, :] = float("nan")            # all NaN

    truth = torch.isnan(x).any(dim=-1)
    m_mask = nan_by_mask(x)
    m_sq = nan_by_squaresum(x)
    print(f"planes={x.shape[0]} with_nan={int(truth.sum())} "
          f"mask_agree={bool((m_mask == truth).all())} "
          f"sq_agree={bool((m_sq == truth).all())}")
    assert bool((m_mask == truth).all()), "vcmp eq mask chain disagrees"
    assert bool((m_sq == truth).all()), "squaresum form disagrees"

    # the form we must NOT use: raw sum is fooled by +inf together with -inf
    raw = x.sum(dim=-1)
    fp = torch.isnan(raw) & (~truth)
    print(f"raw reduce_sum false positives (inf-inf, no NaN): {int(fp.sum())}")
    assert int(fp.sum()) > 0, "expected raw-sum false positives to demonstrate"

    # explicit corner-case table
    for name, i in (("+inf plane", 1), ("-inf plane", 2), ("inf,-inf plane", 3),
                    ("nan+inf plane", 4), ("all-nan plane", 5)):
        print(f"  {name:>16}: truth_nan={bool(truth[i])} "
              f"squaresum_nan={bool(m_sq[i])}")
    print("\n[PL-1.22] PASS: isnan(reduce_sum(x*x)) is exactly 'plane has NaN' "
          "(incl. NaN together with +inf); raw sum is not.")
    print("NOTE: the kernel-side (NPU) UB-cost claim needs msprof/compile; see "
          "repro/TRAP-reduce-ub-per-element.py for the compile-time sweep.")


if __name__ == "__main__":
    main()
