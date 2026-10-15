# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Pattern repro (delta form): PL-1.4-tiling-heuristic -- single-wave CH rule.

条目: pattern-library/layout.md PL-1.4-tiling-heuristic（第二数据点式续证）
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z (Stage 4)
First verified: 2026-10-14, tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760
      (commit 417dae9) + CANN 9.0.0 + Ascend910B3（纯 Vector 物理核 40）

单波判据（优化点 = CH 选择）：纯 Vector general-grid 路径的 channel block `CH` 不是
按「C 大小」或「越大越好」选，而是按**逻辑单元数是否装进一波**选——逻辑单元
`units = N * out_H * ceil(C / CH)`，`num_kernels = min(units, vector_cores)`、
`num_local = ceil(units / num_kernels)`。`num_local > 1` 意味着有核要串行跑第二波，
代价超过「更宽向量」的收益。

  实测（msprof op Task Duration median/20）:
      spp-6x6   C=128, N=2, out 6x6 : CH=64 -> 24 units -> num_local=1 -> 16.31 us
                                       CH=32 -> 48 units -> num_local=2 -> 24.32 us (+49%)
      nondiv    C=64,  N=2, out 7x7 : CH=64 -> 14 units -> num_local=1 -> 14.45 us
                                       CH=32 -> 28 units -> num_local=1 -> 13.14 us (-9%)
                                       CH=16 -> 56 units -> num_local=2 -> 22.17 us (+53%)

  采纳规则（`_pick_ch_general`，本 repro 即该函数的最小形态）:
      base = round_up8(min(C, 64))
      if units(base) * 2 > vector_cores: return base          # 已用掉过半物理核
      half = max(8, base // 2)
      return half if units(half) <= vector_cores else base     # 仅当折半后仍单波

优化见解摘要: 机制归因到「纯 Vector persistent 分核」的波次代价（PL-1.21 模板）：
`T.serial(num_local)` 的第 2 波要与第 1 波串行，而折半 CH 只把每个单元的工作量减半、
op 数不变（向量宽度变窄），因此「多一波」的代价 > 「窄向量」的收益；反之，当物理核
大量空闲时（units 远小于核数），折半 CH 仍保持单波而把更多单元摊到空闲核上，净赚。

断言（可运行部分，纯 Python，无需 NPU）: 对上面两个代表 shape 断言分派结果等于
实测最优档（spp -> 64，nondiv -> 32），并断言「折半后多一波」的形态不会被采纳。

Run: python repro/PL-1.4-single-wave-ch.py
"""

VECTOR_CORES = 40  # Ascend910B3: aicore_num(20) x 2 for a pure-Vector op

MEASURED = {
    # (N, C, out_H): measured Task Duration(us) per CH
    (2, 128, 6): {64: 16.31, 32: 24.32},
    (2, 64, 7): {64: 14.45, 32: 13.14, 16: 22.17},
}


def round_up8(v: int) -> int:
    return max(8, ((v + 7) // 8) * 8)


def units(n: int, c_in: int, out_h: int, ch: int) -> int:
    return n * out_h * ((c_in + ch - 1) // ch)


def pick_ch_general(c_in: int, n: int, out_h: int, cap: int = 64) -> int:
    base = round_up8(min(c_in, cap))
    if units(n, c_in, out_h, base) * 2 > VECTOR_CORES:
        return base
    half = max(8, base // 2)
    return half if units(n, c_in, out_h, half) <= VECTOR_CORES else base


def single_wave(n, c_in, out_h, ch) -> bool:
    u = units(n, c_in, out_h, ch)
    return u <= min(u, VECTOR_CORES)


def main():
    for (n, c_in, out_h), table in MEASURED.items():
        chosen = pick_ch_general(c_in, n, out_h)
        best = min(table, key=table.get)
        print(f"shape N={n} C={c_in} out_H={out_h}: pick={chosen} "
              f"(units={units(n, c_in, out_h, chosen)}) measured={table} best={best}")
        assert chosen == best, f"dispatch {chosen} != measured best {best}"
        assert single_wave(n, c_in, out_h, chosen), "the dispatched CH is not single-wave"
    print("repro OK: single-wave CH rule reproduces the measured optimum on both manifest shapes")


if __name__ == "__main__":
    main()
