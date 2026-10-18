#!/usr/bin/env python3
"""PL-causal-skip-task-mapping: causal 块级跳过的负载模型（`by` 必须最慢）。

机制：`T.Pipelined` 的 runtime start 在本工具链不可用（伪 cbuf overflow），但循环体
可以加 runtime `if c0 + real_n > r0:`；整块位于掩码三角内的迭代 P ≡ +0.0，跳过逐位精确。

关键：跳过后每任务工作量 `work(by) = n_q - floor(by*bm/bn)` 随 `by` **单调递减**。
若沿用 `uid = it*cores + cid` 的轮转（`by` 是最快下标），20 个核拿到的 `by` 组合不同：
max/mean 工作量 1.35x（S=512 档），把 1.78x 的跳过收益压到 1.31x；
把 `by` 换成**最慢**下标（`by = uid // (B*H)`）后 max/mean = 1.01–1.02x，实测多拿 ~21%。

本脚本是纯解析模型（无需 NPU）：断言两种映射的 max/mean 比值，并打印每档的
"理论迭代数下降"与"含负载不均后的有效加速"。

origin_task: TileOPs-_gqa_bwd_kernel-20261018T145922Z
toolchain:   tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4 / 2026-10-18
             （实测：r1 9195.5 us -> r2 7254.3 us，8b-short/fp16）
First verified: 2026-10-18, tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4
"""

import math

CORES = 20


def work(by, n_q, bm, bn):
    return n_q - (by * bm) // bn


def simulate(n_logical, n_kv, n_q, bm, bn, mode, cores=CORES):
    tot = [0] * cores
    if mode == "round_robin":            # uid = it*cores + cid, by 最快
        for it in range(math.ceil(n_logical / cores)):
            for cid in range(cores):
                uid = it * cores + cid
                if uid < n_logical:
                    tot[cid] += work(uid % n_kv, n_q, bm, bn)
    elif mode == "by_slowest":           # by = uid // (B*H)
        for it in range(math.ceil(n_logical / cores)):
            for cid in range(cores):
                uid = it * cores + cid
                if uid < n_logical:
                    tot[cid] += work(uid // (n_logical // n_kv), n_q, bm, bn)
    else:
        raise ValueError(mode)
    mean = sum(tot) / cores
    return max(tot) / mean, sum(tot)


def main():
    cases = {
        "8b-short (4,512,32,128)": (4 * 32, 512 // 64, 512 // 64),
        "8b-long  (2,2048,32,128)": (2 * 32, 2048 // 64, 2048 // 64),
    }
    print(f"{'case':26s} {'iter drop':>10s} {'rr max/mean':>12s} "
          f"{'by-slow max/mean':>17s} {'rr speedup':>11s} {'by-slow speedup':>16s}")
    for name, (bh, n_kv, n_q) in cases.items():
        n_logical = bh * n_kv
        full = n_logical * n_q
        rr, total = simulate(n_logical, n_kv, n_q, 64, 64, "round_robin")
        bs, total2 = simulate(n_logical, n_kv, n_q, 64, 64, "by_slowest")
        assert total == total2, "both mappings must cover the same work"
        drop = full / total
        # effective speedup = iteration drop / load imbalance
        print(f"{name:26s} {drop:10.2f}x {rr:12.2f} {bs:17.2f} "
              f"{drop / rr:11.2f}x {drop / bs:16.2f}x")
        assert bs < 1.05, f"by-slowest mapping not balanced: {bs}"
        assert rr > 1.05 and rr > bs * 1.05, (
            f"round-robin imbalance should exceed the balanced mapping: {rr} vs {bs}"
        )
    print("\nPASS: by-slowest removes the causal-skip load imbalance "
          "(matches the measured 9195.5 -> 7254.3 us, -21%).")


if __name__ == "__main__":
    main()
