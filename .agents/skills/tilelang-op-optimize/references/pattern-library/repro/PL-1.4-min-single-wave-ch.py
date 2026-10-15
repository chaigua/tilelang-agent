# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.4-min-single-wave-ch.py -- general-grid CH 的「最小可行 CH」判据 (纯 Python)

条目: pattern-library/layout.md PL-1.4-tiling-heuristic（第六数据点）
关联: PL-1.21-vector-persistent-split（units = N*out_H*ceil(C/CH) 的分波判据）

现象/优化点:
  纯 Vector general 路径的 channel block 不是由 C 大小决定，而是由
  「逻辑单元数 units = N * out_H * ceil(C / CH) 与物理核数 vector_cores 的关系」
  决定。**取满足 units <= vector_cores 的最小 CH**（= 最多核 × 最短最长块）：
  更小的 CH 会把 units 推过物理核数 → 多一个串行波（代价最大）；更大的 CH
  会减少占用核数（并行度不足）。判据与 C 大小无关。

断言（相对关系，绝对值仅打印）:
  spp-6x6  (C=128, N=2, out 6x6) -> CH=48（units 36，最长块 48）
  nondiv   (C=64,  N=2, out 7x7) -> CH=32（units 28，最长块 32）
  CH 再小一档即越过 40 核（spp CH=40 -> 48 units；nondiv CH=24 -> 42 units）。

首次验证: 2026-10-15 adaptive_max_pool2d-_adaptive_max_pool2d_kernel Stage 4
  (tilelang 0.1.2+489385e9 / CANN 9.0.0 / Ascend910B3; vector_cores = 20 aicore * 2)
  实测（msprof op Task Duration）: spp CH=64(24 units) 20.42us -> CH=48(36 units)
  18.97us；CH=32(48 units, 两波) 29.47us(+44%)；nondiv CH=64(14 units) 18.32us ->
  CH=32(28 units) 15.45us；CH=48(28 units, 块失衡 48/16) 17.07us。
重验记录: (追加式)
"""

VECTOR_WIDTH = 8
VECTOR_CORES = 40  # Ascend910B3: 20 aicore * 2 (纯 Vector 核数翻倍)


def round_ch(limit: int, cap: int) -> int:
    ch = min(limit, cap)
    return max(VECTOR_WIDTH, ((ch + VECTOR_WIDTH - 1) // VECTOR_WIDTH) * VECTOR_WIDTH)


def pick_min_single_wave_ch(c_in: int, n: int, out_h: int,
                            vector_cores: int = VECTOR_CORES, cap: int = 64) -> int:
    """最小可行 CH：满足 N*out_H*ceil(C/CH) <= vector_cores 的最小 8 的倍数。"""
    base = round_ch(c_in, cap)
    for ch in range(VECTOR_WIDTH, base + 1, VECTOR_WIDTH):
        units = n * out_h * ((c_in + ch - 1) // ch)
        if units <= vector_cores:
            return ch
    return base  # 连最窄块都装不下一波：只能接受多波


def units(c_in: int, n: int, out_h: int, ch: int) -> int:
    return n * out_h * ((c_in + ch - 1) // ch)


if __name__ == "__main__":
    cases = [
        # (label, C, N, out_H, expected CH, measured best Task(us))
        ("spp-6x6", 128, 2, 6, 48, 18.97),
        ("nondiv-7x7", 64, 2, 7, 32, 15.45),
    ]
    print(f"| shape | C | N | out_H | CH | units | max block | Task(us) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, C, N, oh, exp, us in cases:
        ch = pick_min_single_wave_ch(C, N, oh)
        print(f"| {label} | {C} | {N} | {oh} | {ch} | {units(C,N,oh,ch)} | "
              f"{ch} | {us} |")
        assert ch == exp, f"{label}: expected CH={exp}, got {ch}"
        # the next narrower block would need more than one wave
        narrower = max(VECTOR_WIDTH, ch - VECTOR_WIDTH)
        assert units(C, N, oh, narrower) > VECTOR_CORES, (
            f"{label}: CH={narrower} unexpectedly still single wave")
        # the returned block is single wave, and wider blocks use fewer cores
        assert units(C, N, oh, ch) <= VECTOR_CORES
        wider = ch + VECTOR_WIDTH
        if wider <= round_ch(C, 64):
            assert units(C, N, oh, wider) <= units(C, N, oh, ch)
    print("\n[PL-1.4] PASS: min-single-wave CH dispatch reproduces the measured "
          "optimum for spp-6x6 and nondiv-7x7 (relative relation only).")
