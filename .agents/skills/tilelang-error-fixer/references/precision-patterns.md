# 精度与失配定位模式（precision-patterns.md）

> 本文件收录**已验证的精度/失配定位手法与判据**（P 类，Tier 1：证据链齐全入队 + 2 次独立证据后由 `tilelang-skill-evolver` 合入）。
> 条目自包含（关键数字 + 适用条件 + 结论内嵌正文）；溯源（origin_task 与出处）允许失效；代码证据唯一合法形态是知识域自包含脚本（`tilelang-op-optimize` skill `references/pattern-library/repro/`，`repro-missing` 表示待回填）。
> 尚未合入的同域提案（在 `.agents/evolution/queue.md` 的 Pending Tier 1）：VP-2026-0006 精度失败定征顺序 / VP-2026-0017 bf16 golden 噪声放大位形 / VP-2026-0038 误差结构定征 / VP-2026-0052 分层中间量导出——合入时置于对应条目邻位并互链。

## 索引输出 kernel 的失配行特征三指纹 + `torch.equal` 免费 oracle

- **领域**：argmax / argmin / 带索引池化（`return_indices=True`）等**输出 int64 索引**的算子。
- **免费 oracle（核心结论）**：索引输出经 `torch.equal` 精确比对（非容差比对）——任何 tie-break / 哨兵 / 状态携带 / 分区边界缺陷立即暴露，且零成本；**值侧 `max_diff == 0` 完全不能证明索引链正确**。两次独立任务实证：
  - argmax reduce（`_argreduce_kernel`，2026-09-24）：`torch.equal` 在 Stage 3 首跑直接暴露原位 vselect 状态丢失（输出全落 tile-0 局部索引）。
  - adaptive max pool with indices（`_adaptive_max_pool2d_with_indices_kernel`，2026-10-14）：整平面路径 **values 全对而 14050/16384 条索引错**——仅因索引 `torch.equal` 精确才被发现（值-only 门禁会漏；该形态的根因见 pattern-library traps-compiler.md `TRAP-npuir-arange-pattern-with-whole-plane-reduce`）。
- **失配行的空间/时间特征先于逐元素排查定根因类别**（指纹速查）：
  ① 输出**全落哨兵值**（2^30）⟹ 掩码匹配链断裂（NaN 传播 / `vbrc` 窄 N 垃圾）；
  ② 错误行数**跨 run 变化** ⟹ 竞态（shared 多消费者超预算）；
  ③ 输出**全落 tile-0 / 首块局部索引** ⟹ loop-carried 状态未携带（原位 vselect 类）；
  ④ 固定行错位（行 0 对、行 ≥1 错）⟹ 步长 / 声明宽度错位（TileOPs reduce 族 pad 契约类，见 queue VP-2026-0128）；
  ⑤ **索引呈列号逐行重复**（`[0,1,…,W-1]` 重复）⟹ `arange` 派生平面下标 pattern 在 in-chain 消费时被读回损坏（2026-10-14 新增指纹，参见 traps-compiler.md 同名条目）。
- **与同域条目互补**：VP-2026-0006 定征顺序（dtype 隔离 → 违反率无关性 → golden opmath）与 VP-2026-0038 误差结构定征（均匀 vs 部分核）管**值侧**；本条是**索引输出域**的指纹版。合入时间 2026-10-14（2/2；两次证据来自不同任务、不同算子族）。
