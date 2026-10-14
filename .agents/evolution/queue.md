# 进化提案队列（Evolution Queue）

> TileLang-Op-Conductor 自进化机制的提案队列。由 `tilelang-skill-evolver` 维护（唯一写入者）。
>
> - **用途**：暂存未达合入阈值的 P 类（模式方法）提案与待人工审批的 R 类（流程规则）diff 提案；D/C 类（实测数据/案例索引）不进本队列，由 evolver 直接合入 pattern-library。
> - **字段规范**：见 `.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。
> - **状态**：`pending`（待确认/待审批）→ `verified`（Tier 1 证据达阈值待合入）→ `merged`；或 `rejected` / `expired`（90 天未决）/ `conflict`（与现有条目矛盾且无法自动裁决，留人工）。
> - **审阅方式**：Tier 2（R 类）提案经用户批准后由 conductor 调度 evolver（`mode=apply`）执行写入；Tier 1 达阈值后由 evolver 在下次蒸馏时合入。

## 字段速查

```yaml
proposal_id: VP-{YYYY}-{NNNN}     # 唯一 ID，由 evolver 递增分配
type: D / P / R / C               # D 实测数据 / P 模式方法 / R 流程规则 / C 案例索引
title: <一句话标题>
evidence:                         # provenance（ED-A 语义，2026-09-10 起）：来源 task_id + 出处描述——允许失效，不做路径存在性核验
  - <task_id + 出处（如 该任务 opt_log round 3 / 路径#定位——路径允许失效）>
repro: <知识域自包含脚本路径（pattern-library/repro/<id>.py，须存在且可执行）或 repro-missing 或 none>
toolchain_stamp: <tilelang build/commit + 设备 + CANN 版本，或 none>
target_doc: <目标文件仓库相对路径（pattern-library 相关指向 pattern-library/ 主题文件）>
delta: <五种合法动作之一：add / update / consolidate / negate / deprecate + 条目正文>
status: pending / verified / merged / rejected / expired / conflict
confirmations: {n}/{2}            # Tier 1 合入阈值：2 次独立证据（不同任务）；E-3 快速通道：另一上下文 repro_runner/ab_test 复现成功计 2/2
created_by: <task_id + 日期>
decided_by: evolver / human / -   # 裁决者
```

> **历史条目路径映射（2026-09-10 pattern-library 拆分）**：本文件中既有条目的 `target_doc: .../references/pattern-library.md` 与 delta 中的「§1/§2/§4」章节引用，合入时按 pattern-library/INDEX.md 的历史引用对照映射到主题文件（§1→layout/elementwise/attention.md、§2→traps-compiler/traps-runtime.md、§4→cases.md、§1.6 常数→constants.md）——历史提案文本不改写，合入锚点按映射解析。
>
> 完整字段规范：`.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。

---

## Pending

> **条目迁移（2026-10-14 apply）**：VP-2026-0176/0177/0178/0179 经用户批准（E-3 审批简报，2026-10-14）于 `mode=apply` 落盘（E-4 预验证 PASS），完整记录移至下方 `## Decided` 归档区。

> **条目迁移（2026-10-14 apply 批次 2）**：VP-2026-0150 / 0047 / 0020 / 0054 / 0116 经用户批准（E-3 审批简报，2026-10-14）于 `mode=apply` 落盘（E-4 预验证 PASS），完整记录移至下方 `## Decided` 归档区。

### Tier 1（P 类，2 次独立证据后合入）

> **条目迁移（2026-10-14 蒸馏，max_pool1d_with_indices 双输出 1D 迁移）**：VP-2026-0159（双输出池化索引契约）/ VP-2026-0160（索引归约极值方向）第二证达标（两证均来自 task `max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z`，不同任务/不同算子）→ 分别合入 `pattern-library/traps-runtime.md` TRAP-pool-index-contract-dual-output / TRAP-index-reduce-extremum-direction；VP-2026-0175（纯选择算子 Ratio 判读口径）第二证达标（同上任务）→ 合入 `profile-collection.md` §4.1；VP-2026-0180（1D 长轴窗口池化必需结构）第二证达标（同上任务，输出个数不同的变体）→ 合入 `algorithm-candidates.md` ALG-pooling。完整记录移至下方 `## Decided` 归档区。

> **条目迁移（2026-10-13 蒸馏）**：VP-2026-0003 第二证达标（adaptive_avg_pool2d 迁移），已合入 `pattern-library/layout.md` PL-1.21-vector-persistent-split，完整记录移至下方 `## Decided` 归档区。

> **条目迁移（2026-10-14 蒸馏，max_pool1d 1D 窗口池化迁移）**：VP-2026-0172（窗口池化退化特化充分条件）/ VP-2026-0173（NPU 参考后端 golden 能力缺口）/ VP-2026-0174（容量驱动 CH 选择）第二证达标（三证均来自 task `max_pool1d-_max_pool1d_kernel-20261014T082309Z`，不同任务/不同算子/新工具链 417dae9），已分别合入 `algorithm-candidates.md` ALG-pooling / `pattern-library/traps-runtime.md` TRAP-npuir-ref-backend-golden-capability / `pattern-library/layout.md` PL-1.4-tiling-heuristic，完整记录移至下方 `## Decided` 归档区。

> **条目迁移（2026-10-14 蒸馏，argmin 共享 ArgreduceKernel 迁移）**：VP-2026-0125（argreduce 边界语义三基准分野）第二证达标（第二证来自 task `argmin-_argreduce_kernel-20261014T151121Z`，不同任务/算子/设备 910B3+CANN 9.0.0）→ 合入 `pattern-library/traps-runtime.md` TRAP-argreduce-edge-semantics；VP-2026-0128（TileOPs reduce 族 pad 契约）第二证达标（同上任务）→ 合入 `algorithm-candidates.md` ALG-reduction R4 附注。完整记录移至下方 `## Decided` 归档区。

## VP-2026-0004

- type: P
- title: BP_grid_stride_vs_contig_chunk：带宽饱和区连续分块映射回退 +3.2%（256M），grid-stride 聚集窗口对 HBM 混合流更优
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-3（v3_op2：256M fp16 1765.01 vs 1710.07，+3.2%；1M/16M tie）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/round3/v3op2_256m_fp16_bs4096
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：msprof op --kernel-name=main --launch-count=15 ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v3_op2.py --dtype float16 --N 268435456 --block-size 4096（对照 grid-stride 基线）
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_grid_stride_vs_contig_chunk：
    触发信号：带宽饱和区（mte2_ratio ≥0.95）考虑把 grid-stride 任务映射改为每核连续分块以改善「DRAM 局部性」。
    反证/判定：lerp_tensor 256M 实测连续分块 +3.2% 回退、小 N tie——grid-stride 的 N 核聚集移动窗口（fewer open pages / 行缓冲命中）对 HBM 混合读写流更优，「连续流局部性」直觉在混合流下反向；数据侧见 pattern-library §1.6 访问模式条。
    动作：饱和区维持 grid-stride；连续分块仅在纯读/写单一流向或 L2 友好 workload 复测后考虑。
    验证：Task Duration 非回退 + mte2_ratio 不恶化。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0005

- type: P
- title: 标量/分支削减优化在 MTE2 饱和区无效的判定式：mte2_ratio ≥0.95 时标量已被搬运隐藏，先查饱和再立项
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 静态尾块全档 tie/-0.6%；guard-free 机制证据 defer <0.3%）
  - 同文件 Baseline 表（≥16M mte2_ratio 0.95–0.99；1M 档 0.70–0.84 流水浅区）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：msprof op ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v2_op2.py --dtype float16 --N 1048576 --block-size 4096
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    update BP_ub_traffic_floor 触发信号，追加：「per-tile 标量/分支开销（动态尾块 T.min、if-guard、地址计算）立项削减前，先查 mte2_ratio——≥0.95（搬运饱和）时标量已被隐藏，实测静态尾块/去 guard 全档 tie/-0.6%（lerp_tensor 2026-09-07）；仅 mte2_ratio <0.9 的流水浅区（小 N 档 0.70–0.84）可能有收益」。数据侧见 pattern-library §1.6 标量削减判定式条。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0006
- type: P
- title: 精度失败定征顺序：dtype 隔离 → 违反率与 shape/block/核数无关性 → golden opmath 域小实验（避免盲目排查分核/内存层级）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Transferable Lessons（定征顺序实证有效）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py（定征输入：fp16 ~0.125% 违反、bf16 逐位一致、fp32 全过、违反率与 shape 无关）
- repro: 对 attempt-1 基线分 dtype 跑 --level L0，观察违反的 dtype 隔离性与违反率 shape 无关性
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24；golden torch 2.9.0+cpu
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新文件首条目（合入时创建 references/precision-patterns.md）「精度失败定征顺序」：
    ① dtype 隔离：哪个 dtype 挂 / 哪些全过（全 dtype 挂 → 结构/同步缺陷；单 dtype 挂 → 舍入路径统计性质）。
    ② 无关性检查：违反率与 shape/block_size/核数无关 → 纯舍入路径性质，排除 tiling/分核/同步排查方向。
    ③ golden opmath 域小实验：数行脚本核实 golden（torch CPU op）在目标 dtype 的实际计算域（fp16 常见 fp32 opmath + 单次舍回），再决定 kernel 原生域 vs fp32 中转（模式见 pattern-library §2 opmath 行 + VP-2026-0002）。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0007
- type: D
- title: T.Kernel(..., threads=) 在 npuir 无效果（KernelLaunch NPU 分支仅消费 grid_size）——设计侧约束表路由
- evidence:
  - src/ir.cc L175–203（NPU 分支无 block_size→threadIdx 消费路径）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 附#28
  - tilelang/language/kernel.py L218–223（Python 签名仍接受并透传）
- repro: grep -n -A 30 "is_npu_kernel_frame" src/ir.cc
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07，源码通读）
- target_doc: .agents/skills/tilelang-op-design/references/ascend-constraints.md
- delta: |
    add §1「本项目已知限制」表新行：
    | GPU `threads=` kwarg 无效果 | `T.Kernel(..., threads=)` 的 block_size 在 npuir KernelLaunch 不生成 threadIdx 绑定（Python 签名仍透传） | GPU 源码迁移残留 `threads=` 无声无效，threads/npt 语义需显式落到 block_size | 迁移时直接移除 `threads=`，折叠为单一 block_size 由 wrapper 参数传入（实证见 pattern-library §2 threads= 行；lerp_tensor 2026-09-07） |
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: 三件套齐备；因 target_doc 为 op-design references（Tier 1 写域）而入队——设计期（GPU→NPU 迁移）是该事实的主要消费场景，事实本体已 Tier 0 合入 pattern-library §2。

## VP-2026-0015
- type: P
- title: Expert 跨引擎 workspace 的跨任务槽位复用握手：per-task 槽位重启 + FLAG_TASKDONE 边界（避免 causal 变长 NK 的全局块前缀索引）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（flag 协议安全性传递性论证）+ #§7.2（四族 flag + FLAG_V 写侧 staging 补全）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：三族 + 单槽 + 跨任务五条论证独立推演全部成立）
  - 〔2026-09-16 第六轮限定，task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z〕TASKDONE 屏障在两相位 v3 结构经 WAR 全链审计判定**冗余**并删除（全域 −7~−21%）：ws_s/ws_p/ws_o 的跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合（attention.md PL-1.12）——握手模式是保守安全默认而非必要条件，**本条合入时须附该限定**（结构特定的程序序闭合可免除屏障；另见 VP-2026-0080 屏障冗余审计 BP 提案，两案互链）
- repro: 任一 per-task 变长内循环（causal NK 变长）的 Expert staggered 流设计
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（Expert 流水模式条目〔VP-2026-0013〕的握手子模式，合入时紧随其后）：
    模式：per-task 槽位重启（slot = i % slots 任务内重计数、跨任务从 0 重启）+ FLAG_TASKDONE 跨任务握手（Vector epilogue 完成后 set、Cube 下一任务首个 S 存储前 wait）——跨任务槽位复用安全，避免 causal 变长 NK 下的全局块前缀索引（数据依赖、不可静态化）。
    配套：flag 协议安全性按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手族单列；写侧 staging（FLAG_V 族，highperf V0S→C1L 同构）补全读侧无序窗口。
    实测：L2-gap100 四例 + kv32-guard（M7 单块退化共存位形）全过（DESIGN §8.2 / debug_log.md Final results）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0016

- type: P
- title: slots≥2 分块装载的尾块守卫从不变式推导（nk_total ≤ slots）而非 shape 枚举——枚举法漏浅块数域；判据同源反推测试域
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E7 + #§8.2（缺口域 4 例：S_kv=100 ∈ (bn,2bn)、ns=2，条件与测试同源）+ #§7.2（Vector 零填充预备段）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 3：v0 守卫仅覆盖 nk_total==1，slots=2/nk_total=2 时部分块为槽位首次使用——L2 套件 kv32/tail520 均不在缺口域，缺口靠检视推导而非测试暴露）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md（Final results：L2-gap100 四例 causal/non-causal × fp16/bf16 全过）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/multi_head_attention/_gqa_prefill_fwd_kernel/_gqa_prefill_fwd_kernel.py --level all（L2 段 gap 四例即判据同源测试域）
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「多槽分块装载的尾块守卫不变式」：
    判据推导：部分块 p=nk_total−1 的残留安全性 ⟺「该槽位 p%slots 此前已被整块装载」⟺ nk_total > slots；唯一破坏点 = 核启动后槽位首用即部分块（残留为未初始化 L1/GM〔torch.empty〕，0×NaN=NaN 污染有效行）——正确守卫条件 = nk_total ≤ slots（含跨任务归纳：槽位尾行状态恒 ∈ {零, 有限真实数据}）。
    枚举法反例：从 shape 枚举（seq_len_kv < bn ⟺ nk_total==1）出发漏浅块数域（nk_total=2、slots=2）且测试域与条件不同源；判据同源反推测试域 = 按判据取边界值 S_kv ∈ (bn,2bn) + 必须触发 slots=2。
    配套：Vector 零填充预备段 + FLAG_V 握手（无它则 staging 写与 Cube 读无序）；守卫面随路线结构性简化（删载体族后 GM 未初始化域缺口自动消失——路线裁决的「正确性面」第四维）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0017
- type: P
- title: bf16 输出精度门禁设计须评估 golden 自身 P 量化噪声的放大位形——tier-2 绝对下限 max(5e-3, 2·flip_rel·vmax) + tier-3 窄域分类机制
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md#D6（L1-70b-long-bf16：1/16777216 元素超 tier-2 1.1%；fp64 逐元素验证 |kernel−true|=4.3e-4 vs |golden−true|=5.8e-3——golden 是偏离方；P 1-ulp 翻转经 causal 早行 p≈0.33×|v|≈2.06 放大 → 5.4e-3 > tier-2）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md（Stage 3 Precision-gate 行）
  - 交叉引用：queue VP-2026-0006（精度失败定征顺序——同文件合入时互链）、pattern-library §2 torch golden opmath 行（golden 侧域差异同类现象）
- repro: bf16 randn 域 attention：causal 早行（有效位少 × P 量化翻转 × 大 |v|）位形对照 tier-2 门限；元素级 fp64 对照定位偏离侧
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；golden = torch SDPA NPU
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「bf16 输出门禁的 golden 噪声放大位形与 tier-3 分类」（合入时与 VP-2026-0006 定征顺序条目同文件、互相交叉引用）：
    机理：golden 自身 P 量化到输入 dtype 的 1-ulp 翻转，在 causal 早行（有效位置少、p 大、|v| 大）放大为输出偏差 5.4e-3 > tier-2（5e-3）——kernel 比 golden 更接近 fp64 真值（4.3e-4 vs 5.8e-3）仍可能超差；该机理是源算法 P-quantization-to-input-dtype 的固有性质，非 kernel 缺陷。
    门禁设计：tier-2 绝对下限应取 max(5e-3, 2·flip_rel·vmax)（bf16 randn 域 ≈ 4e-2）；未重校门禁前的过渡机制 = tier-3 窄域分类：发生率 ≤0.01% + 放大包络（2·flip_rel·|v|max）+ lse 门不受影响 + 逐例上报——真实缺陷（如丢 mask：79.8% mismatch）仍为响亮失败。
    定位手法：元素级 fp64 对照判定偏离侧（golden vs kernel 谁离真值远）再定机制，避免把 golden 噪声当 kernel 缺陷排查（亦见 pattern-library §2 零输入探针行的 fp64 对照法）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0018
- type: P
- title: AIV/lane 半分类设计的参数化标量阈值按最劣 lane 全局坐标独立重推 + 双 lane 端点数值例（单 AIV 思维自检必然逃逸）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1：v0 L220/L536 K_blk = base − row0，正确为 base + row0；smoke bx=4/块 4/vid=1 行 288 数值例复现——正确 j≤32 vs 错误 j≤−32）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3（v1 修复：行映射式 q_pos = s_lo + row0 + i 先行 + 阈值 PrimExpr 标量）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → DESIGN.md 修订链
- repro: 任一含 vid/row0 半分 + 块级 mask/阈值构造的 Expert 设计：对每个 per-lane 标量公式代入 vid=0 / vid=1 两个端点数值例
- toolchain_stamp: 设计/检视期证据（无运行时依赖）；审查环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（并入 VP-2026-0014 向量 mask 构造链小节的「分界/阈值推导模板」子条，合入时紧随其后）：
    修复模板：① 先写行映射式（lane 的全局坐标，如 q_pos = s_lo + row0 + i）再代源条件推导阈值（K_blk = base + row0）；② 阈值走 PrimExpr 标量而非动态 offset（T.arange.md offset 参数文档类型为 int，运行时 PrimExpr 支持无文档）；③ 共用分界（K_A 类）对非推导 lane 标注保守性（仅性能不损正确性——取整方向反例见 §4 expert attention 案例行：ceildiv 版对 vid=0 是激进上界、floordiv 版精确）。
    逃逸机理：以 vid=0（row0=0）代入的阈值在 vid=1 恰好相差 2·row0 且符号方向隐蔽（该 +row0 写成 −row0 时 vid=0 完全正确、vid=1 全错）——单 AIV 思维的自检必然逃逸，双端点代入是最低成本拦截。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0034
- type: P
- title: BP 跨引擎 flag 串行链：dual-Scope persistent 结构下块宽摊减优先于流水深度（bn↑ 同削 flag 往返/向量 op/gemm 碎片，ns/slots 在宽块上失效）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-1（busy 核分解：aic_scalar 41%/aiv_scalar 33% = flag 等待自旋，aic_cube 仅 9% 近峰值——ns=1→2 在 bn=64 上零变化）+ #Iteration-3（bn=512 单槽 -15~-16% vs G2@bn=256；bn=512 双槽 L1 溢出 invalid_compile）+ #Final-Summary（跨轮 busy 核画像主线：flag 等待 25–47% 恒存，块宽摊减是唯一有效方向）
  - pattern-library §1.9（数据侧：块宽摊减律 + cbuf=512KB 上限公式 + wide 单槽模式 + H=1 平衡公式；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10（2026-09-09 同谱系第二轮续调补充〔task 20260909T033622Z，按 VP-2026-0035 同 op 谱系先例不计独立确认〕：v10_dp bn=256 深流水全 case 回退 +11~12%——向量发射开销定律〔每 op ~0.5µs 固定成本〕给出块宽摊减律的机制解释；另实测第三硬上限 L0C=128KB）
  - 〔2026-09-17 适用边界数据点（不同结构类，不计确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕mamba 族 MixCV baseline 的串行形态为 **task 级 ping-pong 零重叠**（Vector(T) 产完 → Cube(T) 消费，非块内 flag 自旋）——该位形下**深度 2 任务流水是第一杠杆**（v1_pipe −39.6%，attention.md PL-1.12 update 消费侧前导 set 形态），块宽摊减居次（v6_bn128 −10.5%）：BP_cross_engine_serial_chain 的「块宽优先」结论适用于块内 flag 自旋类；task 级零重叠类先解除流水串行
- repro: 同结构 kernel（Expert dual-Scope persistent + per-slot flag）扫 bn 64→256→512 × ns∈{1,2}，fa2048/fa4096 non-causal fp16 对照 msprof op Task Duration
- toolchain_stamp: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_cross_engine_serial_chain（目录行置于 BP_pipeline_overlap 之后，正文紧随 BP_pipeline_overlap 节，交叉引用之）：
    触发信号：Expert dual-Scope persistent kernel（Cube/Vector + GM workspace 多槽 + per-slot flag 跨引擎流水）busy 核 aic_scalar 或 aiv_scalar 占比 30%+ 且归因为 flag 等待自旋；cube/mte 已近峰值而总时延远超计算分量；加深流水（slots/ns↑）零变化或被抵消。
    常见反证/不确定点：非跨引擎结构的 MTE/compute 串行归 BP_pipeline_overlap；bn 增大触发 L1(cbuf)=512KB / UB=192KB 上限、bm 增大触发 L0C(cc)=128KB（bm≤51@bn=512，2026-09-09 第二轮实测；上限公式见 pattern-library §1.9）——上限不否定方向，转 wide 单槽（L1 单槽 + GM ws/flag 双槽）继续换块宽。
    推荐动作：先块宽摊减（bn↑ 同时削减向量 op 数/块、flag 往返数/列、gemm 碎片化）再评估流水深度——宽块上 ns=1≈ns=2（双槽收益被单槽 L1 v-load↔gemm2 WAR 串行抵消）；配 bm 任务平衡（wall-rows = ceil(ceildiv(S,bm)/24)·bm）。机制基础（2026-09-09 第二轮实测）：向量算子按发射开销计费——每 op ~0.5µs 固定成本、f16≈f32 逐元素速率，同元素总量下窄块 op 翻倍使 vec-active 翻倍（v10_dp bn=256 实测 +12% 回退）。
    验证指标：Task Duration 下降 + busy 核 scalar（flag 自旋）占比下降 + causal/full 等非目标 dispatch 非回退（full 变体逐字节保留）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0038
- type: P
- title: 跨引擎/在线递推 kernel 精度失败的误差结构定征：全块均匀且可复现 → 排除竞态、指向跨迭代状态生命周期（rescale 因子覆盖）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10-OP2（v10_dp 首版精度 FAIL：全块均匀误差 → alpha/ellcur 跨 defer-2 生命周期被 V1(j) 覆盖；3 路 ping-pong〔平铺 buffer + 运行时三分支〕修复后 L0 全过）
  - 同文件#Skill-Retrospective-第二轮（「先看误差结构再定位」纪律：无它会把递推因子覆盖误判为竞态而放弃深流水方向）
  - pattern-library §1.9 sync_block_wait 行（修复形态交叉引用：v-op codegen 拒绝多维 UB 切片操作数，平铺多 buffer + 运行时三分支为合法形态）
- repro: 任一含 online 递推状态（running max/sum + rescale 因子）跨流水 deferral 生命周期的 Expert kernel：首版精度 FAIL 时先记录误差的空间分布（全块均匀 vs 部分核/非确定）再定排查方向
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「误差结构定征（均匀 vs 部分核）」（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声放大位形同文件并互相交叉引用）：
    判据：精度 FAIL 的误差空间分布是竞态/同步缺陷与系统性状态缺陷的判别器——全块均匀且可复现（各块/各核一致）→ 排除竞态假设，指向跨迭代状态覆盖（online 递推的 rescale 因子〔running max/sum 的 α、ell〕在流水 deferral 生命周期内被后继迭代写入）；部分核/非确定分布 → 竞态/同步方向（flag 协议、握手窗口）。
    修复形态：递推状态跨 deferral 需按流水深度 ping-pong 多路；v-op codegen 拒绝多维 UB 切片操作数——平铺多 buffer + 运行时索引三分支是合法形态（pattern-library §1.9）。
    定位顺序与 §2 零输入探针行的元素级 fp64 对照法互补：先判分布（均匀/部分）定缺陷类别，再下钻元素级对照定偏离侧。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0043
- type: D
- title: layernorm 矩式方差（var=E[x²]−E[x]²）fp32 计算域失败定量：mean≈3000/σ≈0.5 时 fp64err 1.4e3（两遍式 6.3e-4，差 6 个量级）、违反率 25%，fp16 粗栅格 83%——「须评估 catastrophic cancellation」从定性升定量
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（moment_form + run_rejection_evidence 段——torch CPU 设计期脚本，无 NPU 依赖）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Value Point Proposals 首行）
- repro: repro-missing（知识域最小 repro 待同族任务回填；矩式 vs 两遍式差分最小化仅需数行 torch CPU 脚本）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（输出「C2 rejection evidence」段）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-layernorm R1 行（R1 已有指向本条的 pending 指针——VP-2026-0044 三件套合入时预写；本条闭环时把定量句写入 R1 并把指针转为条目引用）：
    R1 单遍式矩方差的 catastrophic cancellation 定量边界（ada_layer_norm 2026-09-10 verify_equiv 实测，torch CPU）：mean≈3000/σ≈0.5（pad 校正小 N 大均值位形）时矩式 fp64err 1.4e3 vs 两遍式 6.3e-4（差 6 个量级）、容差违反率 25%；fp16 粗栅格（6e4 行）违反率 83%——两遍式为默认候选。配套：GPU 源的 256-pad 方差校正在 NPU 精确 N 式下可整体舍弃且数值更优（恒等式机器验证 0 违反，pad 列贡献 m² 显式减去）。
- status: pending
- confirmations: 1/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: 因 target 为 Tier 1 写域（algorithm-candidates.md）+ 缺知识域 repro 而入队（VP-2026-0007 先例形态）；R1 行的 pending 指针已随 VP-2026-0044 合入预写。

## VP-2026-0052
- type: P
- title: 分层中间量导出定位法（系统性 vs 随机误差判别）：out_idx 多输出落 GM 对照 fp64 + violation 率 dtype 指纹（bf16 全过/fp16 ~6%/fp32 ~87% ⇒ 常数相对误差 ~1e-3 = 近似指令特征）单次运行定位到具体指令
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Value Point Proposals 第三行：探针 stage A/B 对照——S1/mean/d 精确 1.1e-7 而 rstd 偏 1.566e-3，误差独占定位在 vrsqrt 指令）
  - 同文件 Stage 3 首跑三 dtype violation 率分布（12 失败用例的 violations/N 统计）
  - pattern-library TRAP-vrsqrt-plain-precision（2026-09-10 Tier 0 合入；「定位手法」段即本方法的首次实证）
- repro: 复现条件——任一含统计链中间量（mean/var/rstd 等）的精度失败定位：中间量经 out_idx=[1,2] 多输出落 GM 对照 fp64 精确值逐级 dump；无固定脚本（方法形态）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声、VP-2026-0038 误差结构定征同文件互链）
- delta: |
    add 新条目「中间量分层导出定位」（合入时置于 VP-2026-0006 定征顺序条目之后）：
    手法：kernel 统计链中间量（S1/mean/d/S2/var/rstd）经 out_idx 多输出落 GM，对照 fp64 精确值逐级比较——单次运行把误差定位到具体指令段（ada 实证：mean/d 精确 1.1e-7 而 rstd 偏 1.57e-3 ⇒ 独占于 vrsqrt）。
    指纹判据（先于探针预判根因类别）：「violation 率 vs 容差」的 dtype 分布——bf16 全过 / fp16 ~6% / fp32 ~87% ⇒ 常数相对误差 ~1e-3 量级 = 近似指令特征（非数据依赖、非同步 bug）；与 §2 零输入探针行的元素级 fp64 对照法互补（先判分布定类别，再下钻元素级定偏离侧）。
- status: pending
- confirmations: 1/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: 原提案 target 为 tilelang-debug-helper SKILL.md（Tier 2 写域）；知识本体为精度定位手法（数据/模式类），改路由至 error-fixer precision-patterns.md（P 类 Tier 1，与 VP-2026-0006/0017/0038 同域互链）——保守序内降档处理，如需保持原 target 请在审批时说明。

## VP-2026-0057
- type: P
- title: 两相位结构的 causal 多头域适配机制集（persistent 变长任务循环 / 尾块钳位 + 装载不满宽掩码门控 / K_A 对角链承接 / flag 预算-bn 耦合守卫 / per-n-block flag 跨任务复用条件）——fa 域母本的 causal 化五机制
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md v3 §0.6 E1/E2/E3/E6/E7 + §5.2/§7.2（机制设计）+ §6.4（flag 协议传递性论证）
  - Stage 3 实证：同目录 RETROSPECTIVE.md Stage 3（2026-09-15）章节——`--level all` 29/29 全绿（L0 4 + L1 8 + L2 10 + Boundary 7；E7 缺口域四例 + M7 单块退化 + GQA 位形覆盖）
  - 母本对照：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py `_builder_2phase`（fa 域 B=1/H=1/整除/非 persistent 特化——五机制均为其不具备的 causal 多头增量）
  - 〔2026-09-16 第六轮机制集存活验证，task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z——同 op 谱系不计独立确认（VP-2026-0035 先例）〕跨工具链迁移 6797758→a13585dc 后 stale 重验（probe_ub.py 直跑 4 点复校，config 空间封闭性结论全部维持）+ 第六轮全部门禁绿；机制⑤ 预言的双槽 id-offset 变体（flag id = i + slot×nk_total，2×nk_total ≤ 15）由 r7f 实现并实测（短域 −7.4%）；机制④ E6 钳位域在 S_kv=4096 首次触发并经 r9 守卫收窄修复（联合 UB 核算判据）。**合入注意**：pipe2 slot-offset 与 E6 钳位分析已写入 attention.md PL-1.12，本条合入时以五机制清单为主体、机制③⑤细节指向 PL-1.12 防重复
- repro: repro-missing（机制效应只在完整 kernel 规模显现；ED-B delta 骨架待同族任务回填）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 PL-1.9-twophase-causal（置于 PL-1.9-twophase 之后；attention.md 已近 16KB 预算，合入时触发 consolidate）：
    **两相位结构 causal 多头域适配五机制**（fa 域母本 → causal 多头域的必要增量，Stage 3 29/29 实证）：
    ① per-task 变长 NK 的 persistent 任务循环 + 轮转均衡（母本为非持久定长）；② 尾块分形下限钳位（M/N≥16、K≥32 的 tmc/tnc 公式）+ **尾带 OOB 掩码门控按「装载不满宽」口径**——Vector 装载宽是 tn_real 而非分形宽，统一门控 `has_kv_tail = (S_kv % bn_eff != 0)` + 运行时 `i >= K_A`（non-causal 时 K_A=kv_full；按「分形带存在」判定会漏 rem≥32 且 16|rem 的不满宽残留）；③ causal 对角 mask 链平移进 pass-1（K_A floordiv 两段式 + per-AIV K_blk = base + row0 阈值）；④ flag 预算-bn 耦合守卫：per-n-block 下标 flag 方案要求 NK_max + TASKDONE ≤ 16 → 工厂期强制 `bn_eff = max(bn_caller, ceil16(ceildiv(S_kv, 15)))`——守卫恒生效、设计默认仅替换 wrapper 默认值（config 契约分层：正确性守卫不可被用户 config 绕过）；⑤ per-n-block 同 id 三次握手的跨任务复用安全条件（单槽）：Cube 到达任务 t+1 的 set_S(i) 前程序序必经 pass-2(t) 的 wait_P(i)（同 id 前一生命周期消费点）——id 事件在任务边界前排空；双槽变体需 id-offset（2·NK_max+2 ≤ 16 才合规）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0058
- type: D
- title: T.ceildiv 负数域 lower 静默错误（divsi 截断 + 后续 pass 复活恒正改写）——persistent 空核越界任务 MTE 崩溃；绕法 = cid = T.min(task_id·N+kernel_id, num_logical-1) 幂等钳制
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Skill Flow Issues 首行 + Value Point Proposals 首行
  - 复现链：one-task (1,1,1,64,64) 崩溃（num_logical < 24 时空核执行越界任务 cid ≥ num_logical → s_lo ≥ S_q → MTE write-out-of-range）+ dump_ir2.log L130-134/L371-374（多 pass 快照对照：lower 成 `divsi(x,N)+1`，x∈(-N,0] 时返回 1 而非 0；恒正参数的 `floordiv(n-k+N-1,N)` 改写被后续 pass 重新规范化回错误形态——纯表达式改写不可靠）
  - 幂等性验证：two-batch (2,1,1,64,64) 钳制下 PASS（幽灵任务重算最后一个合法任务，同输入同指令序位级一致；各核独立 ws 槽 + 核内 flag 通道保证安全；饱和域 num_logical ≥ 24 零开销）
- repro: repro-missing（NPU 编译探针 + dump IR 对照待同族 persistent 任务回填；任务内复现条件 = 任一 num_logical < 24 的 persistent case 如 (1,1,1,64,64)）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-ceildiv-negative-lower：
    **`T.ceildiv` 对非正参数的 lower 是静默错误**：`T.ceildiv(n-k, N)` lower 成截断除法 `divsi(x,N)+1`，x∈(-N,0] 时返回 1 而非 0——persistent kernel 空核（kernel_id ≥ num_logical，仅 num_logical < 物理核数 24 时存在）执行越界任务直接 MTE write-out-of-range 崩溃；更隐蔽的是后续 BishengIR pass 会把恒正参数的 `floordiv(n-k+N-1, N)` 改写重新规范化回 divsi+1 错误形态（纯表达式改写不可靠）。**绕法 = cid 钳制幂等方案**：`cid = T.min(task_id*N + kernel_id, num_logical-1)` 把幽灵任务钉到最后一个合法任务——幂等安全（同输入同指令序位级一致重算 + 各核独立 ws 槽）、饱和域零开销、比 if 守卫（parser var-table 风险）与表达式改写（pass 复活）都稳。官方 persistent 模板（docs/开发指南.md §3.3）在 num_logical ≥ 核数域从未触发该分支。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0060
- type: D
- title: lse [B,H,S,1] 视图直写形态实测可用——`T.copy(ub[0:rm,0:1], lse[bz,by,s:s+rm,0])`（2D src → 1D dst 切片）bit-exact；[N,1] 源 MTE DDR fault 限定于特定 size= 组合而非该直写形态的普遍性质
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Skill Flow Issues 末行（probe_lse_store.py 三形态 A/B/C 全过 bit-exact + 全量 29/29 lse 门通过；E6 transpose-free 方案落地无障碍）
  - 被澄清条目：pattern-library traps-runtime.md TRAP-T-copy-region-semantics 规则③（"[N,1] UB 源 + size=[1,N] 按步长 N 越界读——lse 需先 T.transpose 到 [1,N]"，源自 Sep-07 任务 E1-E7 impl note 2）；traps-compiler.md TRAP-transpose-epilogue-poison 的绕法（[B,H,S,1] 视图 + 2D 区域拷贝）即本形态，第三轮已生产使用
- repro: repro-missing（NPU 探针；任务内复现条件 = probe_lse_store.py 三形态对照——session-local provenance，关键发现镜像于 RETROSPECTIVE Stage 3）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    update 条目 TRAP-T-copy-region-semantics 规则③（边界澄清，原规则保留）：
    ③ [N,1] UB 源 + **size=[1,N]** 组合按步长越界读（MTE DDR fault）——fault 限定于该 size= 组合；**2D src → 1D dst 切片直写 `T.copy(ub[0:rm, 0:1], gm[b, h, s:s+rm, 0])` 实测 bit-exact 可用**（2026-09-15 探针三形态全过 + 29 用例 lse 门；Sep-07 记载的「[N,1] UB 源 MTE DDR fault」不构成对该直写形态的否定）——行向量写 GM 尾维连续区首选 [B,H,S,1] 增维视图 + 2D 区域直写（transpose-free lse，与毒化条目绕法一致），T.transpose 到 [1,N] 仅在必须 base+size 形态时需要。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0061
- type: P
- title: 错误指纹速查——lse 恒等于 log2(S_kv) ⇒ S 通道读到未初始化零页（伴生指纹：前几行 NaN = 越界更远脏页），秒级定位上游 workspace 生产者/槽位错误
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 2 行 + Skill Flow Issues 第 2 行（two-task (1,1,1,128,128)：lse=7.0=log2(128) 恒值直接锁定 ws_s 通道；bx=1 任务全错〔lse 恒值 + 4 行 NaN〕而 bx=0 完美 = per-core 槽 row0 索引 bug——全局/本地行基巧合掩盖）
- repro: 复现条件——two-task (1,1,1,128,128) 类位形（S 通道读到 torch.empty 零页时指纹必现）；无固定脚本（指纹判据形态）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 诊断指纹条目（置于 TRAP-zero-input-crash 之后，同「部分行数据错」诊断族）：
    **lse 恒值 log2(S_kv) 指纹**：lse = log2(ell) + m·LOG2E——输入全零分数（S 通道读到未初始化零页 workspace）时 ell=S_kv、m=0 → lse 精确恒值 log2(S_kv)；「前几行 NaN = 越界更远读到脏页」为伴生指纹。判据指向「上游生产者未写 / 读错槽位」（workspace 槽位索引、flag 时序），先查 Cube 是否写/Vector 是否读对槽位再查计算链，无需逐指令分析（2026-09-15 per-core ws 槽全局行基误用案例：bx=0 任务「全局=本地」巧合掩盖、bx≥1 全错，指纹秒级锁定）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0062
- type: P
- title: MTE 越界类三层崩溃定位法（崩溃边界矩阵 → 错误值指纹 → TILELANG_DUMP_IR + 新 TILELANG_CACHE_DIR）——tilelang 缓存命中跳过编译也跳过 IR 打印，dump 必须换 cache 目录
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 3 行 + Transferable Lessons（one-task 崩溃 40 分钟定位：矩阵 5 分钟锁定「nl<24 崩 / bx≥1 错 / NK 无关」→ IR divsi 对照 → pass 复活确认；probe_nd2nz_offset.py 排除法）
  - 工具行为实证：TILELANG_CACHE_DIR 命中缓存时跳过编译也跳过 IR 打印——TILELANG_DUMP_IR=TRUE 对已缓存 kernel 无输出
- repro: 复现条件——任一 MTE 越界崩溃的 persistent kernel（任务数 × NK × bx 三维隔离矩阵 + dump IR 对照）；cache 行为 = 对已缓存 kernel 设 TILELANG_DUMP_IR=TRUE 观察无输出、换新 TILELANG_CACHE_DIR 后有输出
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-debug-helper/references/mlir-dump-guide.md
- delta: |
    add 增补两节（置于环境变量段之后）：
    **① dump IR 必须配新 TILELANG_CACHE_DIR**：tilelang 缓存命中时跳过编译也跳过 IR 打印——`TILELANG_DUMP_IR=TRUE` 对已缓存 kernel 无输出，须指向新 cache 目录强制重编译。
    **② 崩溃边界矩阵法（MTE 越界类三层定位）**：一层 崩溃边界矩阵（任务数 × NK × bx 三维隔离小用例各跑一次，快速锁定崩溃维度——如「nl<24 崩 / bx≥1 错 / NK 无关」）→ 二层 错误值指纹（恒值/NaN/部分行分布——如 lse 恒 log2(S_kv) ⇒ S 通道零页，见 traps-runtime 指纹条目）→ 三层 TILELANG_DUMP_IR（新 cache 目录）看 TIR/MLIR 双层指令形态（lower 结果 vs pass 后形态对照，确认被哪层改写）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0080
- type: P
- title: BP_sync_barrier_redundancy：persistent/两相位结构的同步屏障冗余审计前置（列出每个 barrier 保护对象集，逐对象找更细粒度已有信号覆盖）——TASKDONE 全 drain 屏障删除实测全域 −7~−21%，纯程序顺序论证零实验成本
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4）：opt_log Phase 1 诊断归因（单槽 ws + TASKDONE 全 drain 屏障使每 task 边界完全串行；WAR 全链审计结论：屏障在数据竞争角度冗余——ws_s/ws_p/ws_o 的所有跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合）+ Round 6 r6a（删除后全域 −7~−21%：8b-long 1189.18→1021.58 / 8b-short 397.79→315.58 / smoke 49.97→46.26µs；全引擎利用率等比提升 cube 20.4→23.7% / vec 49.9→57.4% / mte2 52.1→62.1% 而 GM 流量不变——节约纯等待）
  - 知识域锚点：attention.md PL-1.12「TASKDONE 屏障冗余审计」段（方法 + 逐对象论证 + 数字自包含）
  - 交叉引用：queue VP-2026-0015（FLAG_TASKDONE 跨任务握手模式——保守安全默认，本条是其「可审计免除」限定，两案合入时互链）；VP-2026-0034（BP_cross_engine_serial_chain——块宽/深度之外的第三杠杆）
- repro: repro-missing（delta 最小形态 = 删除 task 边界的 TASKDONE set/wait 同步对〔2 个 sync op〕，效应依赖完整 kernel 的跨 task 指令流衔接；机制归因与逐对象 WAR 论证自包含于 PL-1.12——知识域骨架待同族 persistent 任务回填）
- toolchain_stamp: tilelang 0.1.2+a13585dc / CANN 8.5.0 / Ascend910B2C / 2026-09-16
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_sync_barrier_redundancy（目录行 + 正文，追加于 BP_run_state_bimodality 之后）：
    触发信号：persistent / 两相位（跨引擎 task 循环 + GM workspace）结构存在 task 边界全 drain 屏障（TASKDONE 类：下一 task 的生产等待上一 task 全部消费完成）；时长因子分解显示 per-task 固定成本 F 主导（引擎利用率低而 GM 流量已定、busy 核 scalar 占比高但非 flag 自旋类）。
    审计方法（零实验成本）：列出每个 barrier 保护的对象集（ws buffer × 读写者对），逐对象找更细粒度的已有信号覆盖——同引擎单指令流程序顺序（天然保序）、消费链 flag（P-ready/S-ready 类既有握手）、AIV 串行流。全部对象被覆盖 ⇒ 屏障在数据竞争角度冗余、可删。
    实测（attention 第六轮 r6a，a13585dc）：TASKDONE 删除全域 −7~−21%，引擎利用率等比提升而 GM 流量不变（归因成立的判据：节约的是纯等待）。
    边界：删除前提是 WAR 链逐对象闭合论证（数据竞争角度），非盲目删同步；跨引擎同 buffer WAR 无同步原语的形态不可删（CG-2026-0009 / PL-1.12「Vec 侧同构改造 blocked」教训——引擎独占 buffer 是流水深度 >1 的隐含前提）。
    验证：全域 dispatch 非回退（--level all 全绿）+ 利用率等比提升 + GM 流量不变。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0081
- type: P
- title: msprof roofline Ratio 采数管线规则——--aic-metrics=Roofline 与 --kernel-name 过滤缺一不可（metrics 集不含 Roofline 不产出 visualize_data.bin；无过滤全捕获把 ZerosLike 等辅助 op 当目标，Ratio 0.15–0.65% 伪影实录）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4）：opt_log 第六轮头部「度量管线」段（TileOPs bench 框架 2026-09-16 修复：kernel-name 经 JitKernel_NPU 对象图解析 + 多 op 检测 + duration 匹配；禁止回退无过滤全捕获——profile_run_msprof_20260915_105716.log 的 0.15–0.65% 即该伪影，描述的是 ZerosLike 等辅助 op）+ Phase 1（修复后管线验证：--aic-metrics=Roofline + --kernel-name 过滤 → 单 op 捕获 mix_aic 无 ZerosLike 污染、bin roofline 条目可提取——第五轮 metrics 集不产出 bin 的问题确认并规避）+ DESIGN.md §11.2 指标结构锚定（Ratio = Perf/359.33 TOps/s，Computility 按 48 核计而实机 24 核——口径基数须实测锚定核对）
  - 现行 profile-collection.md 已有 target_kernel_name 命令模板与 captured_op 追溯要求（基础规则存在）；本条补 Roofline-metrics↔bin 耦合、全捕获伪影实录与口径基数核对（2026-09-15 会话产出过伪影 log——规则存在下的执行复发）
- repro: repro-missing（需 msprof + 目标算子环境；伪影对照 = 同 workload 无过滤 vs --kernel-name 过滤双跑，Ratio 条目差异即伪影量级）
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-16（采数侧 tilelang 0.1.2+a13585dc）
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（captured_op 追溯要求行扩展）
    定位锚: "- `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。**roofline Ratio 口径采数（2026-09-16 attention 第六轮实证）**：① `--aic-metrics=Roofline` 单命令同时产出 Task Duration + 诊断 CSV + `visualize_data.bin` 预计算 roofline 条目（Ratio =「GM/L2 · GM Read + Write」条目 ratio×100）——metrics 集不含 Roofline 时不产出 bin（第五轮实证）；② `--kernel-name` 过滤缺一不可：无过滤全捕获会把 ZerosLike 等辅助 op 当目标 kernel（实录：Ratio 0.15–0.65% 伪影，描述的是辅助 op 而非目标）；③ Ratio = Perf/Computility，Perf = msprof 计数 FLOPs / Task Duration——指标奖励执行利用率、不奖励 padding 缩减（flops 与时长同降时 Ratio 不动），Computility 归一基数以实测锚定核对（如 359.33 TOps/s 按 48 核计而实机 24 核——attention.md PL-1.12 指标结构段）。
    动机: Ratio 类硬目标的采数链路三处坑（bin 不产出 / 辅助 op 污染 / 口径基数错）任一都会使目标判定失真——2026-09-15 会话已产出伪影 log，2026-09-16 修复管线后才可靠；规则化使下一次 Ratio 目标任务第一轮即采对。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0083
- type: D
- title: SSD/衰减类因子的 cast 侧选择——因子乘在「值随因子缩小的操作数」一侧再 cast 噪声随真值缩小；移到对侧（K 侧因子移动）fp16 绝对噪声放大 38–135×
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py（OPT-3 否决行 + 回归 case——K 侧因子化 cast(x·exp·dt) + cb 直读 gemm 的 max 2.9e-3 > atol 1e-3，噪声放大 38–135×；cb 侧因子化 cast(cb·exp·exp) 噪声随 exp 缩小）+ DESIGN.md §1.6.1 否决表
  - 〔2026-09-20 跨工具链强化证据（同 op 谱系重跑，不计独立确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z，tilelang 0.1.2+4515de8 / torch CPU〕verify_equiv.py OPT-X 回归行：fp16 下 exp(anchor−dA_s) 正大指数 → inf → cast inf → ×exp(负) = **直接 NaN**（此前记录为 38–135× 噪声放大），bf16 1e26 量级（3.4e+22/2.9e+24 实测行）；OPT-Y（对角 L-side 因子化）NaN×2 同步机器复证——否决项保留 FAIL 回归记录的设计使跨任务复证零成本（RE-1 建议）
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现命令见 evidence provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py（含否决项回归 case）
- toolchain_stamp: torch CPU（2.x，脚本自带）；设计工具链 tilelang 0.1.2+1990aa9fe4 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（SSD/mamba 族新增条目）
- delta: |
    add SSD/mamba 族条目注意事项：衰减/门控类因子（exp(−Δ)、softmax 权重）做 dtype cast 时，因子乘在**值随因子缩小的操作数**上再 cast，绝对量化噪声随真值同步缩小；把因子移到对侧（值不缩小的操作数）会放大绝对噪声并可能超出绝对 atol 门禁——纸面「乘法交换律恒等」≠「容差内等价」，须机器验证（verify_equiv 首轮否决即此形态，返工集中在叙述层）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0084
- type: P
- title: L-side anchor 因子化（exp_l·exp_as）存在 fp32 溢出边界——对角块内 s ≥ l0 ⟹ exp_as ≥ 1，块内衰减 > ~88 溢出污染下三角；直接差分 + 掩码前置是无条件安全 fallback
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：DESIGN.md §1.6.1 否决行 #5 / §1.6.3 候选 #9；源码 ssd_chunk_scan.py L350–371 微块 M32 锚点选择注释（anchor 机制正是为此）
- repro: repro-missing（分析型——边界可由 verify_equiv.py deep_decay 案例扩展复现）
- toolchain_stamp: 分析基于 fp32 exp 上限 ~e^88.7（无工具链依赖）；核对环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（SSD 族注意事项，与 VP-2026-0083 同条目或紧邻）
- delta: |
    add SSD 族数值安全注意：L-side anchor 因子化的安全性依赖「两个因子指数均 ≤ 0」的单调论证；对角块（因子定义域重叠，s ≥ l0 ⟹ exp_as ≥ 1）内该论证失效——块内衰减 > ~88（fp32 exp 上限 e^88.7）溢出污染下三角（真实分布 24σ 罕见但非零）。直接差分形式下三角指数恒 ≤ 0、上三角被掩码，无条件安全；源实现的微块 M32 锚点选择即该边界的工程规避。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0085
- type: D
- title: 累加 gemm 的 A 操作数必须覆盖完整 K 维（N 维）——n-loop 复用单块 [bl,bn] 在 N > bn 时静默算错（max_diff 0.40 符号翻转），小 N 用例全绿是假象
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：Stage 3 修复链——history 路径 c_scaled = C·exp(dA_l) 按 [bl,bn] 单块 + n-loop 复用，N=128/bn=64 时 n-block 1 读到第 0 块（静默算错，仅 L1-256 暴露）；修复 = c_scaled 全维 [bl,N] + Cube n-loop 逐块 T.copy（修复前 L1-256-full max_diff 3.9e-1 → 修复后 1.4e-4）
- repro: repro-missing（NPU kernel 级；任务内复现 = python3 _ssd_chunk_scan_fwd_kernel.py --level all 观察 L1-256-full FAIL→PASS——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 语义陷阱条目（置于 TRAP-L1-band-dst-tail-overrun 之后）：**累加 gemm 的 A 操作数 K 维覆盖**——history/state 类 Σ_n A[:,n]·B[:,n] 的 n-loop 每块读的是**不同的 A 列段**，只算一块 [bl,bn] 复用会静默算错（N > bn 时才暴露；N_tiles=1 的 L0 用例全绿是假象——验证用例须含 N 维 > block_n 的 case）。修复形态：A 侧因子按 [bl,N] 全维物化，Cube n-loop 逐块 copy 消费。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0086
- type: D
- title: Expert 模式 T.copy 通用路径不支持跨 dtype（编译报错 "does not support element type casting"）——与 TRAP-C12 的 Developer 静默 VCast 互补；三条落地口径
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：Stage 3 三处跨 dtype copy 报错修复链（prev_states(fp32)→state_t(dtype)、cb/C/dt(dtype)→f32 因子）——显式 dtype 中转 buffer + T.vcast(..., round_mode="rint")（f16/bf16→f32 仅 rint）+ gemm B 的 fp32→dtype cast 放 host wrapped() 内 .to(dtype)（对齐源码 T.cast 语义，golden 仍 fp32）
  - docs/Tilelang.language/内存操作/T.copy.md §2.3 条 2（通用路径 dtype 一致要求）
  - 被补充条目：traps-compiler.md TRAP-C12-copy-dtype-cast（Developer GM→UB 隐藏 VCastOp 静默转换——同一 API 两模式行为相反）
- repro: repro-missing（NPU 编译探针；任务内复现 = grep -n "vcast\|\.to(torch_dtype)" 于任务 kernel——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-C12-copy-dtype-cast（追加 Expert 模式互补段）：Developer GM→UB copy 静默插 VCastOp（原文保留）；**Expert 通用路径要求 src/dst dtype 一致**——跨 dtype copy 直接编译报错（"T.copy does not support element type casting"），须显式中转：① 同 dtype copy + T.vcast(..., round_mode="rint")（f16/bf16→f32 仅 rint）；② gemm B 操作数的 fp32→dtype cast 可在 host .to(dtype) 完成；③ 5D 输入 Vector scope GM→UB 动态 extent copy 触发 codegen "cannot find variable" 时关 TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION（PL-1.16 硬边界）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0087
- type: D
- title: 向量二元 op 广播形态速查——vmul/vsub/vadd 文档化仅四形（无 [M,N]*[1,N]）、M==N 时方向写反静默算错、vmin 族内原生支持一般列广播（规则不族内统一）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：Stage 2 REVIEW.md 问题 1（v0 列因子 [bs,1] 错向——bl==bs=64 时编译不报错静默算错，本任务设计修订 3 阻塞之首）+ re_review Value Point Proposals（T.vmin.md §2.2.2 明示 src1 (1,N,K)/src2 (M,N,K)→(M,N,K) 一般列广播合法，vs T.vmul/T.vsub/T.vadd §2.2.2 仅四形：[M,N]*[M,N] / [M,N]*[M,1] 行因子 / [1,N]*[1,1] 退化 / [M,N]*scalar——无 [M,N]*[1,N]）
  - docs/Tilelang.language/数学操作/T.vmul.md#§2.2.2、T.vsub.md、T.vadd.md；比较操作/T.vmin.md#§2.2.2；shape操作/T.vbrc.md#§2.2.2（列因子合法路径：vbrc (1,N,K)→(M,N,K) 展开后全形运算）
  - 数值演示（torch）：cb*f[64:128].unsqueeze(-1) vs unsqueeze(0) 两形不等、差值 O(数据)（repro 命令可复现）
  - 反例先例：GQA _gqa_prefill_fwd_kernel.py L401/L405/L645（真行因子 [half,1]）与 L430（j 轴依赖改 T.arange 全形物化）；全仓 9 处 T.vmul 无列因子广播用例
  - 关联已合入：layout.md PL-1.15-broadcast-perf-tax（性能税半边——本条补形态合法性半边，两半互补成完整速查）
- repro: repro-missing（文档口径核对 + torch 数值演示；知识域最小脚本待同族任务回填）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / Ascend910B2C / npu-smi 26.0.rc1（文档口径，未上机）；数值演示 torch 2.7.1+cpu / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    add「向量二元 op 广播形态速查」条目（与 PL-1.15 互补，合入时互链）：**行因子 [n,1] 沿尾轴广播（随 i 变化）；随列 j 变化的因子必须 [1,n]——而 [M,N]*[1,N] 不在 vmul/vsub/vadd 文档化 Shape 清单内**（仅四形），列因子须 T.vbrc 展开为 [M,N] 全形再运算（vbrc.md §2.2.2 合法形 src (1,N,K)→dst (M,N,K)）；**M==N（如 bl==bs）时方向写反不报 shape 错、静默算错**。族内不统一：T.vmin §2.2.2 原生支持 (1,N,K)→(M,N,K) 一般列广播——逐 op 读 §2.2.2 是唯一可靠做法，禁止按族类推。配套已核实（供 SSD/causal 族直接引用）：T.vmin/T.vadd 均导出（tilelang/language/__init__.py L73/L79）、fp32 √ / bf16 ×、标量形有生产先例（engram_fwd.py L76）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0088
- type: P
- title: BP_aiv_duplication_check——MixCV（Mix Block Dim = 2× Block Dim）画像两 AIV 子块指标相同 ⟹ Vector 程序重复执行 ⟹ subid 分片为第一候选（本轮 −34% 单点）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：opt_log §9 流程观察 1（发现花了 4 轮——前 3 轮把 vector0/vector1 相同指标读作"分摊或镜像"，R4 才用「若分摊则 ratio 应减半」反证确认）+ R4/R5（v8_aivsplit w2 −34.4%；数据侧已回写 attention.md PL-1.13）
- repro: repro/PL-1.13-aiv-dup-subid-split.py（知识域，2026-09-17 转正——分片骨架 + 机制归因 + 判据）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_aiv_duplication_check（目录行置于 BP_compute_granularity 之后）：触发信号：MixCV kernel（Mix Block Dim = 2× Block Dim）的 msprof PipeUtilization 中两 AIV 子块 vec/mte ratio **完全相同**而非各半（相同时每 AIV 承担 100% 向量工作——AIV 产能 2× 浪费）。反证：ratio 各半 ⟹ 已分摊（GQA v11 类先例已分片）。动作：subid 边界表达式分片（蛇形均衡式见 PL-1.13；禁 if 守卫——TRAP-tvm-parser-rules）为第一候选；分片维度须与 ws 写区域正交。验证：per-AIV vec/mte2/mte3 时间减半 + Task Duration 下降（ssd 实测 −28~−36%）。机制与完整形态：attention.md PL-1.13（互链，不复制）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0095
- type: D
- title: L1→L1（cbuf→cbuf）`T.copy` 硬编译失败（Expert 最小探针实证）——L1 域内重排/组装须单 buffer GM→L1 列偏移直载（dst 侧尾块裁剪联防）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 复检轮：REVIEW.md（复检轮）机械复核段编译探针表 + 附录（探针 1/2 关键代码与报错全文——第三轮 REVIEW 覆盖后残留描述见该任务 RETROSPECTIVE.md Stage 2 revision 章节）；DESIGN.md v1→v2 §1.4 band 组装修订（L347–353 两跳 → 单跳）
  - docs/Tilelang.language/内存操作/T.copy.md §2.3 搬运方向表无 L1→L1 方向（GM→UB/UB→GM/UB→UB/GM→L1/L0C→GM）；全仓 examples 零先例
  - 〔provenance，允许失效〕探针重建最小形态：`T.Kernel(1, is_npu=True)` + `T.Scope("Cube")` + 两个 `T.alloc_L1([64,64],"float16")` + `T.copy(l1_a, l1_b)` → 预期 bishengir-compile err code 1 报 'Unsupported copy from cbuf to cbuf!'；探针 2（GM→L1 列偏移直载 + 单条 band gemm）预期编译通过
  - 关联：capability-gaps CG-2026-0002（cbuf→cbuf 不支持，2026-09-20 occurrences 2 升级 recurring——vbrc lowering + Expert T.copy 双形态）；traps-runtime.md TRAP-L1-band-dst-tail-overrun（dst 侧 ts 裁剪联防）；traps-compiler.md TRAP-C12（dtype 面，互补）
- repro: repro-missing（知识域最小 repro 待同族任务回填——探针原件 transient 丢失；重建最小形态与预期报错见 evidence provenance 项）
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 新条目 TRAP-L1-L1-copy-unsupported（置于 TRAP-C12 之后）：**`T.copy` L1→L1（cbuf→cbuf）方向硬编译失败**：`'hivm.hir.copy' op Unsupported copy from cbuf to cbuf!`（bishengir-compile err code 1）——docs 方向表无此方向、全仓 examples 零先例、最小 Expert 探针实证（三证合一，2026-09-20，4515de8）。**绕法**：L1 域内「重排/组装」意图落为单 buffer GM→L1 列偏移 dst 直载（`T.copy(ws[s_blk,0:bl,0:ts], l1_band[0:bl,s0:s0+ts])`，src/dst 双侧尾块裁剪 ts=T.min(bs,Q−s0)——与 TRAP-L1-band-dst-tail-overrun 联防）+ 单条 band gemm `size=[tmc,l0+tl,tnp]`（探针 2 编译通过），或经 GM 中转；设计期引入新搬运方向先过三关（方向表在列 + examples 先例 + 最小编译探针）。与 TRAP-C12 构成 T.copy 方向/dtype 双约束；缺口本体 CG-2026-0002（recurring）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 设计期「写实占位伪代码」自创两跳结构引入未证实原语、复检探针证伪（修订写实对照纪律提案见 VP-2026-0104）；报错文本与 CG-2026-0002 首证（Developer vbrc lowering，2026-09-04）同源。

## VP-2026-0096
- type: D
- title: 深度 2 任务流水 slot-free wait 的正确位置 = Vector 任务头（先于任何 ws[slot] 写）——尾置形态在「prologue 前导 set + 事件计数」协议下产生 off-by-one WAR 竞争
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 revision：REVIEW.md（首轮）问题 1 令牌推演（CONS0 set 序 {prologue, C 完成任务 0, 2, …} × 尾置 wait 序 {T=0 尾, 2 尾, …} ⟹ T=2 写 ws[0] 只受 prologue 放行、不受 C 消费任务 0 约束）；DESIGN.md v1 §7.2 行 #4 头置形态
  - 对齐基准：`git show 4515de8:examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py` L356–362（旧 verified 终版注释 "from task 2 on it waits for Cube's consumption of task (T-2)"）
  - 4515de8 复刻头置形态首编即过 + L0–Boundary 全绿（同任务 Stage 3）
- repro: repro-missing（令牌推演为纸面证明——set 序×wait 序×受保护对象写序位置逐对配对，无需运行即可复核；verified 形态经 git show 对照）
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.12-task-pipeline-depth2 的「消费侧前导 set」update bullet，追加：**wait 置位位置是深度 N 流水的正确性字段（非实现细节）**——slot-free wait 必须 Vector 任务头前置（先于任何 ws[slot] 写）；尾置形态在「prologue 前导 set + 事件计数」协议下产生 off-by-one WAR 竞争（写只被上上次同槽令牌放行，稳态下快引擎覆写慢引擎正在读的槽位）；前置形态下任务 W 写同槽受 C 完成任务 W−2 约束（V 领先 ≤1 任务）。后续算子复刻该结构时 wait 位置须与条目形态逐行比对（2026-09-20 重跑任务 v0 曾以尾置形态逃逸设计自检、检视期令牌推演才检出）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 检视侧令牌推演检查项已由 VP-2026-0029 承载（本任务为其第二任务独立证据）；本条为数据侧位置规则条目化。

## VP-2026-0097
- type: D
- title: band 域 stale 列掩码双判据必要性机器证明——stale 列 NaN 可跨下三角（i ≥ j ≥ ts），加法掩码污染下三角、单一因果 vselect 亦泄漏，`(j≤i)&(j<ts)` 组合才干净
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1：verify_equiv.py stale-NaN 专项案（输出 add_polluted=True / causal_leak=True / full_clean=True）+ DESIGN.md §1.6.1 OPT-B2 行（band-carrying 域形态）
- repro: repro-missing（verify_equiv.py 为任务过程文件；stale-NaN 案三形态对照可由数行 torch 脚本复现——下次同族任务顺手转正）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.11-causal-mask-scalartrap 的「NaN 免疫边界」bullet：band-carrying 域的 vselect 守卫须**双判据** `(j≤i)&(j<ts)`（列 OOB 守卫与因果守卫缺一不可）——机器证明（2026-09-20 verify_equiv stale-NaN 案）：stale 列 NaN 可跨越下三角（i ≥ j ≥ ts 位），① 加法掩码 NaN+x=NaN 污染下三角；② 单一因果 vselect（j≤i）不足——stale 列与下三角交集的 NaN 被选路保留；③ 组合判据干净。与既有「band-carrying 保留 vselect 选择语义」合并为完整判据。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 与 TRAP-L1-band-dst-tail-overrun（dst 侧裁剪）互为 Vector 侧掩码对偶——合入时互链。

## VP-2026-0098
- type: D
- title: Cube gemm 分形 K 维下限「K ≥ 32」为保守口径——K=16（size=[tmc,16,tnp]）在 4515de8 编译且数值正确（max_diff 2.75e-5，静默无 stale 带污染）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 3：Boundary B-n16 用例 PASS（history gemm N=16 作 K 维）；DESIGN.md §5.3 注意 3 / §9.1（原断言「N=32 贴分形下限」为安全最小，未覆盖 N=16）
  - 被细化口径：.agents/skills/tilelang-op-design/references/decision-tree.md L74–75（L0A: M ≥ 16, K ≥ 32 / L0B: K ≥ 32, N ≥ 16）
- repro: repro-missing（端到端 kernel 级——B-n16 用例；K<16 与 M/N 下界未测，结论边界开放）
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/references/decision-tree.md
- delta: |
    update 分形限制条目（L74–75 邻域）追加实测注记：K=16 在 4515de8 静默有效（编译 + 数值正确，2026-09-20 ssd 重跑任务 B-n16 实证）——「K ≥ 32」按保守口径使用（设计仍可取 32 为安全最小），但不再作为硬下限否定 K∈[16,32) 形态；K<16 未测。合入时在 queue VP-2026-0057 机制②（尾块钳位 K≥32 公式）加同口径注。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 单点边界实测，泛化边界（更小 K / M/N）开放——条目按「保守口径 + 实测下界」双标注，防过度泛化。

## VP-2026-0099
- type: P
- title: Expert MixCV kernel 工厂闭包自包含约定——返回闭包内自分配 workspace + 输入 dtype cast，对 wrapper 暴露与 GPU 源一致的张量签名 → 集成零胶水（attempts=0 首过）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 5：examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/_ssd_chunk_scan_fwd_kernel.py L461–484（`_inner`/`wrapped` 内部分配 ws_c/ws_lcb + prev_states fp32→dtype cast）；integration_log.md（attempts=0——首次 import 冒烟 + 4/4 正确性 + 11/11 benchmark 一次通过）
- repro: 复现条件——任一 Expert kernel 工厂把 workspace 分配与输入 cast 封装进返回闭包（对外签名与 GPU 源一致）后走 integrate_kernel.py 全流程
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir（4515de8 谱系）; Ascend910B2C; CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md（Expert 族结构约定 bullet，合入时与预算 consolidate 协调）
- delta: |
    add bullet（PL-1.16 结构形态段邻位）：**工厂闭包自包含约定**：kernel 工厂返回的闭包内部完成 workspace 分配（ws_c/ws_lcb 等）与输入 dtype cast（如 prev_states fp32→dtype），对 wrapper/测试/benchmark 暴露与 GPU 源算子一致的张量签名——wrapper 零胶水改动即可切换集成（2026-09-20 ssd 重跑任务 Stage 5 attempts=0 实证：smoke + 4/4 正确性 + 11/11 bench 一次通过）；workspace 需求不构成改 wrapper 的理由（与 PL-1.16 wrapper 契约兼容条呼应）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 与 VP-2026-0093（config 口径衔接）/ VP-2026-0074（config-契约范式）构成 Stage 5 集成约定三件——合入时互链。

## VP-2026-0100
- type: D
- title: TileOPs report 新集成算子首跑的 N 条 "missing tileops candidate record" 告警 ≠ benchmark 失败（N=workload 数）——有效性看 run.json summary 字段，不数 warnings
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 5：examples/TileOPs/reports/tileops/20260920_144100_174551_SSDChunkScanFwdOp/run.json（summary：benchmark_passed=true, msprof_case_count=11, profiler_fallback_count=0, invalid_profile_count=0；warnings：11 条 candidate record 告警）+ report.md#Warnings；integration_log.md 备注
  - 〔provenance，允许失效〕判定命令：TileOPs 根目录以 reporting CLI 对任一新集成 Op 跑 msprof 采数后查 run.json 的 warnings 与 summary 两段
- repro: repro-missing（需 TileOPs + 新集成 Op 环境；判定口径自包含于 delta——run.json summary 三字段，判定命令见 evidence provenance 项）
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir; Ascend910B2C; CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 小条目（TRAP-BENCH-CONFIG-CALIBRATION 之后，采数判读族）：新集成算子首跑 report 的 "missing tileops candidate record" 告警（每 workload 一条）是候选基线记录缺失的提示性告警，不影响 status=passed 与 msprof 数值有效性（fallback=0 / invalid_profile=0）——判定 benchmark 有效性看 run.json summary 字段（benchmark_passed / msprof_case_count / invalid_profile_count），不数 warnings（2026-09-20 ssd 重跑任务 11 条告警 + 11/11 全 msprof 实证）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: 消费场景在 Stage 5 integrator 侧——若审批倾向改 integrator known-fixes（Tier 2 域）而非 traps-runtime 条目，可在审批时改路由（事实本体不变）。

## VP-2026-0110
- type: P
- title: BP-engine-criticality-drift——persistent 双引擎 kernel 的引擎临界性随对侧优化漂移，「引擎 X 非关键」结论每轮 winner 合并后重算（任务周期 vs 引擎 busy 逐 workload 核算）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log.md#round-4 现象节（逐引擎任务周期核算：w2 任务周期 6.58µs vs Cube mte2 busy 4.45µs/任务 ⟹ w2 已 AIV-bound——旧档 PL-1.18 的「AIV vec 62% 非关键」在 prevhoist+l0c2x 两轮 Cube 优化后不再成立，vbrchoist 因此获益 −3.5%）
  - 方法有效性的自然实验：同文件 R4（vbrchoist 获益分布 w2 −3.45% / w4 −3.30% 响应，w3 −0.27% 不响应——AIV-bound/AIV 共临界/Cube-bound 三态与逐 workload 约束判定逐一互证）
  - 〔2026-09-21 重建会话方向翻转级证据（同 op 谱系不计确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z〕vbrchoist 单点在无 prevhoist 的 base 上 H=80 族回退 −4.2~−6.7%（双引擎 mte2 +22~28%）、叠加 prevhoist 后全转正 +2.2~+10.2%——不只收益幅度，效应方向本身随对侧累积序翻转（本条「临界性漂移」的更强形态，opt_log R1 机制分析 + R2 复测）；姊妹 BP 提案 VP-2026-0115（累积序检验步骤），两案合入时互链
  - 数据侧落点：attention.md PL-1.18 update③ 获益分布注（2026-09-21 已回写）；流程侧姊妹提案 VP-2026-0112（iteration-diagnosis.md 已知迭代结论补注——方法/流程两 Tier 拆分，合入时互链）
- repro: none（方法类：判据与核算式自包含于 delta；效应载体为 attention.md PL-1.18 + repro/PL-1.18-floor2-wins.py 骨架）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_engine_criticality_drift（目录行置于 BP_run_state_bimodality 之后）：
    触发信号：Expert/persistent 双引擎（Cube+Vector）kernel 的 profile 或前轮结论含「引擎 X 非关键路径 / 非瓶颈」定性；或本轮 winner 改变了某一引擎的负载/串行结构（字节削减、链解串行、op 削减类合并）。
    核算方法（每轮 winner 合并后执行）：逐 workload 比较「任务周期（Task Duration ÷ 每核任务数）」与「各引擎 per-task busy 时间」——引擎 busy 之和逼近或超过任务周期 ⟺ 该 workload 转为该引擎 bound；临界性是随结构状态漂移的动态量，不是一次性判定（ssd 实证：AIV「非关键」结论在两轮 Cube 优化后失效，w2 转 AIV-bound 使 Vector 侧 hoist 重新有收益）。
    推荐动作：临界性反转的 workload 上重启「此前因非关键被否决」的对侧候选（旧否决注须核对其归因是否含形态税等可修正因素——ssd v11 vbrc hoist 失败实为 vsub dst=src2 alias 税，干净形态胜出）；Cube-bound workload 对 Vector 侧优化不响应是判定正确的佐证。
    验证指标：候选获益分布与逐 workload 临界性判定一致（响应/不响应互证）；任务周期与引擎 busy 核算在合并后重新闭合。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0115
- type: P
- title: BP_mixcv_cumulative_order——MixCV 跨引擎候选单点验证的累积序调制：与先验矛盾的 workload 族分裂（方向翻转）先检验累积序假设（合入对侧已验证胜出后复测），再谈 per-shape 条件路径
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z（重建会话 Stage 4）：opt_log.md#Round-1-机制分析 + #Round-2-结论与机制（round1_vbrchoist 在无 prevhoist 的 base 上 H=80 族〔2.7B，G=1 下 cb 读放大 3.3×于 H=24〕回退 −4.2~−6.7%〔双引擎 mte2 传输时间 +22~28%，L2 局部性劣化〕；同一形态叠加 prevhoist 之上 model-scale 全转正 +2.2~+10.2%——回退消失；判定「Round 1 的 H 族分裂是单点叠加顺序的伪象，非 vbrchoist 形态本身的缺陷」；本会话曾据 H 分裂考虑 trace-time H 条件分支，累积序复测后证伪该需求——避免了无谓的核内分派复杂度）
  - 数据侧落点：attention.md PL-1.18 update②（2026-09-21 optimizer 已回写、evolver 复核对账相符——机制归因自包含：AIV 侧候选的单点效应受 Cube 侧 base 状态调制，prevhoist 削减 Cube mte2 压力后 L2/总线交互改变）
  - 姊妹条目：queue VP-2026-0110（BP_engine_criticality_drift——临界性随对侧优化漂移；本条是其「方向翻转级」补全：不只收益幅度，效应方向本身可随累积序翻转）+ VP-2026-0112（流程侧每轮重算规则——三案合入时互链）
- repro: repro/PL-1.18-floor2-wins.py（知识域结构骨架——三胜出与累积序效应载体；效应数据见该任务 perf_records round1/round2〔provenance，允许失效〕）
- toolchain_stamp: tilelang 0.1.2+96f287eeaa（与 15ad002 delta 仅 .agents 文档、tilelang 源码零改动）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_mixcv_cumulative_order（目录行与正文置于 BP_engine_criticality_drift 之后——若 VP-2026-0110 尚未合入则置于 BP_run_state_bimodality 之后，两案互链）：
    触发信号：MixCV / 跨引擎（Cube+Vector）kernel 的单侧（Vector 或 Cube）候选单点验证出现与先验证据矛盾的 workload 族分裂——尤其方向翻转（某 workload 族稳定回退而先验记录该形态有效，或反之）。
    检验步骤（先于 per-shape 条件路径）：① 刻画分裂族的结构特征（H/G/读放大比等维度）；② 把候选叠加到「已合入对侧已验证胜出」的 current best 上复测（复现既有累积序）；③ 分裂消失 ⟹ 单点结论依赖 base 状态（叠加顺序伪象）；分裂仍在 ⟹ 才评估核内 trace-time 条件分支。
    机制：单侧候选的单点效应受对侧 base 状态调制（对侧字节削减/链解串行改变 L2/总线交互与双引擎 mte2 传输速度）——「单点分支从 current best 派生」的纪律须以已合入的累积序为基。
    实测（ssd 重建会话，96f287e）：vbrchoist 单点 H=80 回退 −4.2~−6.7%（双引擎 mte2 +22~28%）→ prevhoist 之上 +2.2~+10.2%（回退消失）；机制归因与画像 diff 见 attention.md PL-1.18 update②（互链，不复制）。
    验证指标：族分裂消失 + 全域（model-scale）非回退；与 BP_engine_criticality_drift 的逐 workload 临界性核算结论一致（响应/不响应互证）。
- status: pending
- confirmations: 1/2
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0119
- type: D
- title: reduce 混合 dtype（fp16 src→fp32 dst）静默产出错误极值——文档示例与实现矛盾（T.reduce_max.md §2.4 示例按实测产出垃圾值）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/2：design_probe_argmax_dbg.py 变体 B（kern_mixed，fp16 输入 + fp32 规约输出）extreme=[0.75,117,13,−152] vs 真值 [3.09,2.81,3.98,2.89]，全部落 2^30 哨兵；Stage 2 实机重放复现（dbg-B 哨兵 / dbg-C 同 dtype PASS 对照）；对照 docs/Tilelang.language/规约操作/T.reduce_max.md §2.4（dtype≠accum_dtype 示例）与 §2.2.1——文档示例与实现矛盾
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/argmax/_argreduce_kernel/history_version/design_probe_argmax_dbg.py（看 kern_mixed 输出行）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-reduce-mixed-dtype-silent-corrupt：T.reduce_max/T.reduce_min 的 dst dtype 与 src 不一致时静默产出错误极值（非 cast、非报错——fp16 src→fp32 dst 实测 extreme 全落 2^30 哨兵级垃圾值）；docs/Tilelang.language/规约操作/T.reduce_max.md §2.4 的 dtype≠accum_dtype 示例与实现矛盾（文档示例不可信）。配套纪律（bf16 面）：reduce dtype 表不含 bf16——bf16 输入先 T.vcast(..., round_mode="rint") 到 fp32 再 reduce。诊断指纹：极值列全落哨兵/异常量级 + 索引主路径正确 ⟹ 混合 dtype reduce 而非搬运/索引链缺陷。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: Stage 2 重放为同任务非独立证据（证据链追加性质）；capability-gaps 候选（文档示例错误类），暂不重复登记。

## VP-2026-0120
- type: D
- title: bm=1 时 T.Parallel 融合循环条件引用 (·,1) 不变操作数被误 lower 为 arith.fptoui(f16→i1)——MLIR verify fail；先 T.vbrc 同形化修复；bm≥2 正常
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：P2 首版编译失败 IR（`arith.fptoui` f16→i1 广播，/tmp 探针日志）；触发条件 `for i,j in T.Parallel(1,N): ite(x[i,j]==m[i,0],...)`（m 形状 (1,1)——条件中的 (·,1) 不变操作数）；vbrc 同形化（先 T.vbrc(m, ext) 展开）后 PASS；DESIGN.md §9.1-C-2
- repro: repro-missing（知识域最小 repro 待同族任务回填；条件形态自包含于 delta）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-parallel-cond-unit-operand-fptoui：bm=1 的 T.Parallel 融合循环条件引用形状 (·,1)/(1,1) 的不变操作数（如 m[i,0] 与标量/布尔比较组合）时被误 lower 为 `arith.fptoui(f16→i1)` 广播——MLIR verify 编译期失败；bm≥2 同形态正常。绕法 = 先 T.vbrc 把 (1,1) 缓冲同形化展开到 (1,N) 再做元素对元素比较。与 TRAP-vbrc-same-shape-empty-broadcast（同形 vbrc 空广播维）构成 bm=1 形态族的两面：前者是「该 vbrc 而未 vbrc」，后者是「不该 vbrc 而 vbrc」。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: -

## VP-2026-0121
- type: D
- title: shared 多消费者缓冲在 auto-multi-buffer 预算超限时静默数据竞争（非确定错误值跨 run 变化）+ 未用 T.alloc_fragment 不被 DCE——fragment 计算形态 + ≤64KB/block 手工预算双纪律
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/2：design_probe_argmax.py p3c/p3d 首版（fp16/bf16 bm=2，shared 缓冲喂 ≥2 个算子 + 含未用 ext_brc）非确定 BIG 行 15–25/2048、3 run 变化（竞态指纹）；p3a/p3d/p3e fragment 版 3/3 稳定 PASS；DESIGN.md §4.5/§9.1-C-3/C-4；Stage 2 实机重放 p3a bm=2@hidden-state 3/3 稳定复证
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现 = python3 history_version/design_probe_argmax.py --case p3d〔bm=2〕多 run 观察非确定 mismatch——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-shared-multiconsumer-silent-race：alloc_shared 缓冲被 ≥2 个算子消费（多消费者）且 auto-multi-buffer 双缓冲使预算超限时发生**静默数据竞争**——错误值非确定（同行跨 run 变化，15–25/2048 级），无报错无警告，是「最难查的一类」；同场实证：未用的 T.alloc_fragment 缓冲不被 DCE（计入 UB 规划挤占双缓冲预算）。规避双纪律：① 计算缓冲一律 alloc_fragment、alloc_shared 仅作 GM staging（bm/dtype 分派用独立 prim_func，不写「无条件 alloc + 条件使用」）；② 手工 UB 预算 ≤64KB/block（×2 双缓冲 = 128KB ≤ 184KB 可用）。诊断指纹：错误行数跨 run 变化 ⟹ 竞态（vs 全哨兵 = 匹配链断裂、固定行错 = 步长/索引）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0122（CONST-capacity 预算数据点）联动：本条是「超限后果」，该条是「超限阈值实测」。

## VP-2026-0122
- type: D
- title: CONST-capacity-910B2C 追加 argmax 结构膨胀数据点——宽 N 1.60×/2.0×、窄内维 N<8 ~4×（(2048,4) bm=2048 手工 64KB→requires 256KB 溢出，bm=1024 通过）、bm=32@N_padded=256 ~2× 编译实证
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z：DESIGN.md §4.5 预算表（fp16 bm=4@4096 手工 160KB → requires 2097408 bits=256.03KB=1.60×；128KB→256KB=2.0×——宽 N 定标）；Stage 3 首跑 (4096,4)/(524288,4) fp16 bm=2048 手工 64KB → requires 2097408 bits（~4×，窄 N）编译溢出、降档 bm=1024 后 3/3 PASS；Stage 5 集成 run.json：bm=32@N_padded=256 手动预算 80KB、×2 约 160KB < 192KB 编译通过（latency 5145µs，profiler_valid=true）
- repro: repro-missing（知识域最小 repro 待同族任务回填；任务内复现 = _select_config 阶梯复算 + 窄 N bm 阶梯 cap 逻辑——provenance 允许失效）
- toolchain_stamp: tilelang 0.1.2（dev root build 2026-09-24 + 集成 013dbbf5）+ CANN 8.5.0 + Ascend910B2C / 2026-09-24~28
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md
- delta: |
    update 条目 CONST-capacity-910B2C（auto-multi-buffer 膨胀系数段）追加数据点：① 宽 N（N≥256）argmax 结构实测 1.60×（fp16 bm=4@4096：160KB→256.03KB）与 2.0×（128KB→256KB），与既有 ×1.7 量级互证；② **窄内维 N<8 膨胀 ~4×**（(2048,4) fp16 bm=2048：手工 64KB → requires 256KB）——窄 N 的扩展阶梯（小 N 大 M）预算解除不能沿用宽 N 的 64KB/block 口径（Stage 3 实测 bm 2048→1024 才过）；③ bm=32@N_padded=256 手动 80KB ×2 ≈160KB < 192KB 编译通过（集成环境实证）。交叉注记 TRAP-UB-multibuffer-inflation（膨胀条目）与 VP-2026-0121（超限静默竞态——预算红线不仅是编译溢出，超限先于溢出即竞态）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 数据点为多 attempt 汇总（设计探针 + Stage 3 首跑 + Stage 5 bench），同一任务内互证但非跨任务独立确认。

## VP-2026-0123
- type: D
- title: TRAP-vbrc-same-shape-empty-broadcast 追加窄 N 垃圾索引形态——bm=1 vbrc (1,1)→(1,N) 在 N<8 产出垃圾/哨兵索引（N=2..6 全错、N=6→2^30），N=1 编译失败（同形空广播），N=8/bm≥2 正常
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 3：调试 (1,N) bm=1——N=8 PASS、N=2..6 FAIL（N=6 → 1073741824=2^30 全哨兵）、N=1 compile fail（empty broadcast dims，与 TRAP-vbrc-same-shape-empty-broadcast 同族）；bm=8 时 N=3/5 奇数窄列亦失败、N=4/8 通过；契约内最小 N=4 已由 L0-6 覆盖故未阻塞
- repro: repro-missing（调试脚本在 /tmp 探针（可转正）；条件形态自包含于 delta）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-vbrc-same-shape-empty-broadcast（追加窄 N 形态段，原文保留）：除同形空广播维编译失败（N=1 形态）外，**vbrc (1,1)→(1,N) 在 N<8 时产出垃圾索引**（cand 全落 BIG 哨兵/错误 int64；N=6 实测全 2^30）——窄 N 向量化对 vbrc 广播与 candidate 循环的边界在 N=8；bm≥2 直接索引 + N=4 正常。窄 N（N<8）负载的候选索引构造避开 (1,1)→(1,N) vbrc 形态。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: -

## VP-2026-0124
- type: D
- title: T.Parallel 内标量直接写 GM 输出张量元素（无 UB staging）产出错误值——同行 int64 主路径经 alloc_shared + T.copy 出栈正确（单次观察，待二次隔离证据）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2：design_probe_argmax_dbg.py kern_same 的 `dbg[pid*bm+i,0] = T.cast(ext16[i,0],"float32")`（L69-70）重放输出 extreme=[-0.845,0.452,-1.135,-0.852] vs torch=[3.36,3.08,3.10,2.67] 而同行 ok=True（经 out_ub+T.copy 的索引主路径正确）；对照 kern_mixed 的 T.copy(ext32, dbg[...]) 导出路径（该变体错值来自混合 dtype reduce 非导出路径）
- repro: repro-missing（隔离实验未做：把 dbg 改成 UB staging + T.copy 后是否恢复——单次观察，合入前需第二次隔离证据；任务内复现命令见 evidence provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：cd examples/argmax/_argreduce_kernel/history_version && python3 design_probe_argmax_dbg.py（看 [C-same] 的 extreme 行 vs ok 行）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-parallel-scalar-direct-gm-write（待二次隔离证据后合入）：T.Parallel 循环体内标量表达式直接写 GM 输出张量元素（无 UB staging）实测产出错误值；同一 kernel 内经 alloc_shared + T.copy 出栈的主路径正确。隔离归因待补：UB staging + T.copy 化后是否恢复（当前不排除与调试 buffer 布局的其他交互）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 提案源头（Stage 2 复盘）自标「单次观察，合入前需第二次隔离证据」——本条即按该口径入队等待。

## VP-2026-0126
- type: D
- title: multitile bm>1 组合触发编译器 SIGSEGV（触发矩阵：bm=1 任意 tile ✓ / bm>1 单 tile ✓ / bm>1 多 tile ✗）——前次任务 repro 被清理后靠 pyc 考古恢复；CG-2026-0014 互链
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：capability-gaps CG-2026-0014（本任务 §9.4-R9 补录——前次任务已登记但条目随工件清理丢失，源 repro .py 仅存 repro/__pycache__ pyc）；DESIGN.md §9.1-C-5 触发矩阵引用（本任务引用结论未复现崩溃本身）；pyc 反序列化 docstring 考古恢复路径见 Stage 1 复盘 Info-source 行
- repro: repro-missing（源 repro 已清理仅存 pyc——重建触发矩阵三形态最小 kernel 待同族任务回填：bm=1 多 tile / bm>1 单 tile / bm>1 多 tile）
- toolchain_stamp: 前次任务 2026-09-23（tilelang 0.1.2 dev build + CANN 8.5.0 + Ascend910B2C）；本任务引用环境 2026-09-24 同源
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-multitile-bm-gt1-segfault：num_tiles>1 的 T.serial tile 循环 + bm>1 组合触发编译器 SIGSEGV（rc 139，无报错文本）；触发矩阵：bm=1 任意 tile 数 ✓ / bm>1 单 tile ✓ / bm>1 多 tile ✗。绕法：tiled 路径强制 bm=1（工厂期保证）；小 M 大 N 负载改 N-split 两段 kernel（elementwise.md PL-1.20）。关联：capability-gaps CG-2026-0014（open）——本条使其进入 kb_search 检索域（本次知识流失-考古循环的教训：CG 登记簿不在 pattern-library 检索索引内，设计期 kb_search 查不到该陷阱）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0141（scaffolder 清理知识保全）联动：本条是清理致知识流失的实证案例。

## VP-2026-0129
- type: P
- title: reduction 族小 N 大 M 负载设计规则集——workload (M,N) 按 Op 层 reshape 规则推导（dim=0 → M=prod(其余)、N=该维长度）；block_m 由发射项决定 + 门控扩展阶梯 {16..2048}（窄 N 膨胀 4× 换算 + cap bm≤M）；tile_n 整除优先 + 裕度后置校正
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2/revision/3/5：REVIEW.md 问题 1（v0 (512,4) 手抄 vs manifest+Op 层真值 (524288,4) 差 1024×——错误根因是抄错后无推导式可自检）；DESIGN.md v1 §5.2/§5.5（门控代码：基础阶梯 {1,2,4,8}〔GPU 48KB SMEM 承袭〕下 block 数 > 48×2 时启用 {16..2048} 档、UB 预算过滤器封顶——3d (524288,4) fp16：bm 8→2048、65536→256 block、发射项 4.8ms→18.7µs）；Stage 3 修正（bm=2048 窄 N 4× 膨胀溢出 → 实取 1024；(1,3)/(1,5) bm=8 垃圾行 → cap min(block_m,M)）；Stage 5 bench（bm=32@N_padded=256 编译通过）；tile_n 规则（DESIGN v1 §5.2：候选集 = N 因子 ∩ 256 倍数 ∩ ≤cap ∩ slab·elem·tn ≤ 0.9×UB_MANUAL_BUDGET 取最大，空集回退 cap+静态尾 tile——复现 lm-head fp16 5120 / bf16 4096）
- repro: 纯工厂期算术复算（host 侧、设备无关）：65536/(4×8)=2048；ceildiv(524288,2048)=256；256×7op÷48×0.5µs≈18.7µs；反事实 ceildiv(524288,8)→≈4.8ms；factors(102400)≤6400 且 5×2×tn≤58982 → 5120
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C（发射常数 CONST-vector-launch-overhead）/ 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-reduction 行 R4 追加小 N 大 M 负载设计规则集：① **(M,N) 推导**——manifest 负载一律按 Op 层 reshape 规则重算（dim=0/中间维 → M=除规约维外全维乘积、N=该维长度；movedim(dim,−1) 后 reshape），不按 x_shape 字面读（v0 差 1024× 实证）；② **bm 由发射项决定**——小 N 大 M 负载 UB 预算允许的 bm 远大于 GPU 承袭阶梯 {1,2,4,8}（block 数爆到数万、每 block 几个向量 op、0.5µs/op 发射主导：65536 block ≈4.8ms vs bm=2048 256 block ≈18.7µs）；门控扩展阶梯 {16..2048}（blocks > aicore×2 时启用）+ UB 预算过滤器封顶；**窄内维 N<8 的预算解除按 ~4× 膨胀换算**（宽 N 2× 口径在 (2048,4) 实测溢出，降档 1024 才过）且 **cap block_m = min(block_m, M)**（bm>M 尾行是 UB 未初始化垃圾、窄 N 下被向量化器污染有效行）；③ **tile_n 整除优先 + 裕度后置**——候选集 = N 的因子 ∩ 256 倍数 ∩ ≤ tile_n_cap ∩ slab·elem·tn ≤ 0.9×手工 UB 预算，取最大（零尾 tile + ≥10% 裕度）；空集回退 cap + 静态尾 tile（不依赖 compute_tile_n 的除数接受判据——其「除数优先」仅在不增加 tile 数时生效）。kb_links 补 CONST-vector-launch-overhead / VP-2026-0122（窄 N 膨胀）。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0044（row-reduction 基准形态，已合入）互补：0044 覆盖 N 可驻留的常规形态，本条覆盖小 N 大 M 与 N 超驻留两端的阶梯/分派规则。

## VP-2026-0130
- type: P
- title: 调优迭代效率两手法——① 归因「二选一」嫌疑列表当轮用变体对照消歧（不留到后续轮——argmax 标量化归因跨轮证伪实证）② 高成本负载先微基准定标（copy-only 行宽-效率曲线 W≥32 渐近）再上真 shape 采数
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 4：opt_log Iteration 4（session-1 对 baseline 标量化留两个嫌疑〔(bm,1) 列广播操作数 / varange 生成失败〕未闭合，round 4 r4a 操作数变体对照一次证伪两者——真触发器是 2D T.Parallel(bm,N) 融合循环本身；证伪更正留痕 canonical §2-3）；Iteration 2（probe_gran copy-only 微基准定标行宽曲线：W=4→9.74µs / W=8→6.28 / W=16→5.56 / W=32→4.52µs，W≥32 达渐近线——r2a 窄行宽视图设计直接建立在该曲线上，避免真 shape 1632µs 负载多轮高成本采数）；SESSION_TIMING_ANALYSIS §3（诊断消歧并行化建议 + 3d baseline 采数最贵）
- repro: none（方法类：判据与实例自包含于 delta；效应载体 = traps-compiler.md TRAP-parallel2d-bm-ge2-scalarize 证伪更正段）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24~28
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    add「迭代效率两手法」小节：① **嫌疑列表当轮消歧**——Phase 1 诊断产出「二选一」式归因嫌疑时，当轮即对每个嫌疑派一个操作数/形态变体实验分支做对照（各 ~1 分支成本），不跨轮次延迟：argmax 实证（session-1 两嫌疑留到 round 4 才由 r4a 变体对照双双证伪——若 round 1 当轮消歧，链式形态可提前 3 轮锁定；证伪更正按 canonical §2-3 留痕：误判根因 + 合法形态 + 新数据）；② **高成本负载微基准定标先行**——baseline 采数最贵的 workload（多 wave/读膨胀/诊断链长）先用更小的快速回归 shape 或 copy-only 微基准定标关键常数（行宽-效率曲线、膨胀系数），再上真 shape——argmax 3d 负载的窄行宽视图设计直接建立在 probe_gran W 曲线（W≥32 渐近 4.52µs/4MB）上，避免 1632µs 级负载的多轮盲扫。
- status: pending
- confirmations: 1/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: ①的实证即 TRAP-parallel2d-bm-ge2-scalarize 的证伪更正链（Stage 4 已回写知识侧），本条是其流程侧（当轮消歧纪律）的条目化。

## VP-2026-0142
- type: D
- title: avg-pool 倒数乘（`sum/d` → `sum·fl(1/d)`，倒数 fp32 计算一次）的容差内等价实测数据与判据分档（≤1 ulp of target dtype）
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/verify_equiv.py（C1 行 + 45 用例表：spp (2,128,56,56)→6×6 fp16 bitdiff=15 / viol=0 / max_ulp=1.00、max_abs_diff 1.22e-04；其余形状与 bf16 全 0 差——与 atol/rtol 1e-3/1.6e-2 完全相容）（任务工作区，未上库，provenance 允许失效）
  - 同目录 DESIGN.md §1.6.1 C1 行 + Stage 2 复检复盘 Value Point Proposals 第二行；交叉引用 layout.md PL-1.5（乘倒数行）
- repro: repro-missing（verify_equiv.py 为任务过程文件；知识域最小 repro 待同族任务回填——逐元素 `sum/d` vs `sum·fl(1/d)` 取舍入对照仅需数行 torch CPU 脚本）
- toolchain_stamp: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b + CANN 9.0.0 + Ascend910B3 / 2026-10-13；verify_equiv 为 torch CPU（fp64 参照）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    update 条目 PL-1.5-quickref 的「乘编译期常数倒数」行（原文保留，追加定量与判据分档）：倒数在 fp32 计算一次的 `sum·fl(1/d)` 对 `sum/d` 为**容差内等价**——adaptive_avg_pool2d 2026-10-13 实测（45 用例，torch CPU fp64 参照，双式均按目标 dtype 舍入）：spp (2,128,56,56)→6×6 fp16 有 15/9216 元素位级差（10 元素差 1 ulp，max_abs_diff 1.22e-04）、viol=0、max_ulp=1.00；其余形状与 bf16 全 0 差；与 atol/rtol 1e-3/1.6e-2 完全相容。**判据分档纪律**：等价性判据须按优化项的等价强度分档——逐位等价项用 `viol==0`，容差内等价项用「舍入到目标 dtype 后 ≤1 ulp」并与逐位判据分列（同任务 verify_equiv `_ULP_TOL={C4:0,C4b:0,C1:1}` 实现）；单一 `viol==0` 判据会把容差内等价项误判为 FAIL。
- status: pending
- confirmations: 1/2
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: -
- decided_note: 缺知识域 repro 降级 Tier 1（merge-policy §1 ED-A）；目标 PL-1.5 为既有 verified 条目，合入时用 update。

## VP-2026-0145
- type: P
- title: 设计盘点两漏点与核对项——① 累加器零初始化（`for o_w: for c: acc=0`）常被排除在「循环/标量点盘点」外、可整块 `T.clear` 消除；② `ceil(C/CH)>1` 的通道块尾须 GM/UB 两侧动态切片仅写回有效通道
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/REVIEW.md 问题 1（v0 §1.6.2/§6.1 均无 `acc` 零初始化行，§3.3 行 605-607 有）；DESIGN.md v1 §1.6.2 #0 / §6.1 / §3.3 `T.clear(acc)`；docs/Tilelang.language/内存操作/T.clear.md；Stage 3 L1 bf16 `(1,96,28,30)→(7,5)` 与 `(1,96,5,7)` 恒等两例（`real_c = T.min(CH, C - cb*CH)` + `[0:real_c]`）0 差（任务工作区，未上库）
- repro: none（文档级/实现级证据）
- toolchain_stamp: tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-design/references/quality-checklist.md
- delta: |
    add/update quality-checklist 分块与向量化盘点段两级核对项：① **累加器零初始化**（`for o_w: for c: acc[..]=0` 类标量写）常被排除在「循环/标量点盘点」之外——可整块 `T.clear(acc)` 消除（降级为 vbrc 类整 buffer 操作），盘点时以 §3.3 伪代码逐 `for`/元素级赋值与 §1.6.2/§6.1 建立一一映射；② **通道块尾**：`CH = min(C, 64)` 且 `ceil(C/CH)>1` 时须 GM↔UB 两侧动态切片 `real_c = T.min(CH, C - cb*CH)` 且仅写回 `[0:real_c]`（UB 未初始化尾 lane 只要不写回即不影响结果）——设计 checklist 的分块章节增该核对项与 L0/L1 用例形态。
- status: pending
- confirmations: 1/2
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: -
- decided_note: ①的目标亦被 queue VP-2026-0147（design-review 维度 8 可操作清单）覆盖——本条落在设计 skill 的 references（P 类 Tier 1），VP-2026-0147 落在检视 SKILL（R 类 Tier 2），合入时互链。

## VP-2026-0153
- type: P
- title: 「门禁用例全过 ≠ 语义机制生效」——含独立判决量（NaN 掩码/饱和/选择）的 kernel 须给机制级证据（中间量观测或最小探针），否则端到端 PASS 可能是相邻机制偶然传播导致的假阳性
- evidence:
  - task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z Stage 3 near-miss：首版 `vcmp(...,"ne")` + `vmax` 累积，L0-10 NaN/±inf 语义门禁**假阳性通过**（标志位恒假，NaN 输出实际来自 `vmax`/`reduce_max` 的 NaN 传播）；5 组最小探针（`/tmp/probe_vcmp*.py`）逐层剥离后定位；改用 `eq` 取反后语义判定变为确定性路径、全部用例复跑仍全过且 max_diff=0（任务工作区 RETROSPECTIVE.md Stage 3 + history_version/..._impl_s3_attempt1.py 首版快照，允许失效）
  - repro/TRAP-vcmp-ne-ordered-nan.py 的 `ne self-compare is-nan == []` 段（标志位失效的直接观测）+ `flag==0 ⟺ 列含 NaN` 断言
  - 交叉引用：queue VP-2026-0152（该 NaN 追踪形态）、VP-2026-0006 / VP-2026-0052（精度失败定征顺序 / 分层导出——本条补 "语义机制生效性" 面）
- repro: repro/TRAP-vcmp-ne-ordered-nan.py（机制失效的直接观测段 + 绕法通过断言）
- toolchain_stamp: tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「语义机制生效性的机制级证据要求」（合入时与 VP-2026-0006 定征顺序、VP-2026-0052 分层导出定位同文件互链）：
    判据：当 kernel 含会独立影响输出的**判决量**（NaN 掩码、饱和/钳位、选择条件），"端到端门禁用例通过"**不构成**该机制生效的证据——可能存在另一条会把结果 "顺带修对" 的路径（本例 `T.vmax`/`T.reduce_max` 的未文档化 NaN 传播）掩盖失效的判决量。要求：除端到端用例外，须给出机制级证据——判决量本身（中间量）的观测、或最小 repro（如 `repro/TRAP-vcmp-ne-ordered-nan.py` 的 `flag==0 ⟺ 列含 NaN` 断言）。诊断动作：对 "看似已生效" 的语义机制，构造一个**该机制被刻意关掉后结果是否仍正确**的最小探针——仍正确 ⟹ 存在假阳性风险，机制未被真实验证。
- status: pending
- confirmations: 1/2
- created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13
- decided_by: -
- decided_note: -

## VP-2026-0158
- type: P
- title: 本分支（`target="npuir"`）`tilelang.jit` 无 `compile_flags` 形参——迁移设计里的 `compile_flags=["-O3","-DENABLE_BF16"]` 应弃用；bf16 能力不依赖 `-DENABLE_BF16`
- evidence:
  - task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z Stage 3 Skill Flow Issues 第四行：`DESIGN.md` §4.7/§3.5.2/§10.5 要求保留 `compile_flags`，但本分支 `tilelang/jit/__init__.py` 无该形参（`grep -n "compile_flags" tilelang/jit/__init__.py` 无命中）；实现不传，bf16 用例（L0-4/L0-9/L1 多例）实测全过
  - 第二源（不同任务，代码事实）：同族前作 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` L224 同款静默弃用（任务工作区，未上库，允许失效）
  - 交叉引用：queue VP-2026-0007（`T.Kernel(threads=)` 在 npuir 无效果——同属「GPU 源 kwarg 残留、npuir 前端签名以本分支为准」族）
  - 〔2026-10-14 第三现（不同任务，pool 族 + Stage 3 实现侧消费），task max_pool2d-_max_pool2d_kernel-20261014T062320Z〕Stage 3 Skill Flow Issues 首行：`DESIGN.md` §3.5.2 表断言源 GPU 的 `compile_flags=["-O3","-DENABLE_BF16"]` 为「设备无关」应**保留**，照抄即 `TypeError`（`tilelang/jit/__init__.py` 签名 `(out_idx, target, target_host, execution_backend, verbose, pass_configs, debug_root_path)` 无该形参，tilelang/jit/__init__.py#L180-190）；同族已验证 kernel（`_adaptive_max_pool2d_kernel` L294 / `_gqa_prefill_fwd_kernel`）一律只用 `target="npuir"`——**迁移设计模板侧的处置口径仍未更新**（该行应自「保留（设备无关）」改为「移除」，与既有 `threads`/`autotune` 两行同类），属「已三现事实未进设计模板」的传播面缺口
- repro: repro-missing（API 约束核对；复现条件 = `grep -n "compile_flags" tilelang/jit/__init__.py` 无命中，观察迁移设计与实现是否保留该 kwarg）
- toolchain_stamp: tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-design/references/ascend-constraints.md
- delta: |
    update §1「本项目已知限制」表追加行：
    | `compile_flags=` kwarg 不存在 | `target="npuir"` 的 `tilelang.jit` 无 `compile_flags` 形参（`tilelang/jit/__init__.py` 无该参数）——GPU 源迁移设计里的 `compile_flags=["-O3","-DENABLE_BF16"]` 无法透传 | 迁移设计沿用该 kwarg 会与实际前端签名不符；bf16 能力不依赖 `-DENABLE_BF16`（未传 flag 的 bf16 用例实测通过） | 迁移时直接移除 `compile_flags`；前端签名以本分支 `tilelang/jit/__init__.py` 为准（模式同 VP-2026-0007 `threads=` 无效果） |
- status: pending
- confirmations: 1/2
- created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13
- decided_by: -
- decided_note: 第二源为同族前作代码事实（未独立任务化）——保守按 1/2 待第三证；target 为 op-design references（Tier 1 写域）。

## VP-2026-0161
- type: D
- title: 共享累加器的「比较-更新次序」是正确性字段——`vmax(acc,v)` 先于 `v > acc` 则判据恒假、极值/索引输出静默退化为初值；公式级 D-1 全 PASS **不覆盖**实现 op 次序（最小反例 `arange(12)→(2,2)` 的 `bin(1,1)`：正确 11 / 错误 6）
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 2（检视 #1 不通过）+ Stage 1 revision v1：`REVIEW.md#问题 1`（含反例 `arange(12).reshape(1,1,3,4)→(2,2)` 的 `bin(1,1)` 应得 11、按 v0 次序得 6）；`DESIGN.md` v0 §3.3/§6.2（`T.vmax(acc,...)` 写在 `T.vcmp(...,"gt")` 之前）vs §1.6.1 N3；`history_version/design_v0.md`；v1 §1.6.1 `N3-ord`（342 checks 全 `EQUIV_PASS`，general 矩阵 9779 个可检出元素非空）（任务工作区，未上库）
  - 判据脱钩实证：v0 的 `verify_equiv.py` 6 项全 `EQUIV_PASS` 却完全未覆盖该次序——公式级验证与 §3.3 实现次序脱钩（verification-implementation gap）
- repro: repro-missing（知识域最小 repro 待同族任务回填；最小复现 = `python -c "import numpy as np; a=np.array([-np.inf]); v=5.0; a=np.maximum(a,v); print(v>a)"` → `False`，或把命令式候选式改回先更新后比较的次序）
- toolchain_stamp: 语义/逻辑事实（无运行时依赖）；设计侧证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-shared-accumulator-cmp-update-order（实现/lowering 次序陷阱类）：把「`val > max_val` 条件覆写」改写为「共享累加器 + 掩码候选」的向量式后，**比较与累加的先后次序是实现正确性的一部分**——若 `T.vmax(acc, x)` 先于 `T.vcmp(x, acc, "gt")` 执行，则比较恒假（`x > max(acc,x)` 恒假）⟹ 候选恒为哨兵/初值 ⟹ 极值或索引输出**静默退化**（无报错，values 侧可能同时正确）。公式级 D-1（候选式 vs 基线式）无法发现该错误——须把实现伪代码的 op 相对次序纳入验证（双向：正确次序复现基线 + 错误次序必被检出 + 最小反证 shape）。最小反例：`arange(12)→(2,2)` 的 `bin(1,1)` 正确 11 / 先更新后比较得 6；实证 342 checks 全 `EQUIV_PASS` 的次序守卫。与 TRAP-tir-range-loop-not-unrolled 等实现形态类陷阱同族（管 §3.3 伪代码 → §6 循环结构的落地形态）。
- status: pending
- confirmations: 1/2
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 流程侧（D-1 增实现次序守卫项）见 VP-2026-0164（op-design SKILL）/ VP-2026-0166/0167（design-review SKILL）——知识侧与流程侧分层合入；缺知识域 repro 降级 Tier 1。

## VP-2026-0162
- type: P
- title: 「值错 vs 核 fault」分离的探针定位链——control（仅填充+拷贝）→ 逐项加 op（vcmp/vselect/reduce）→ 每步一子进程；配 pattern 类操作数的 standalone 精确 + in-chain 精确**双向**断言
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 3：probe6（整文件跑 → 首个 int32 fill case core dump，前面所有 print 丢失）vs probe7/8（一 case 一进程 → 7 个 case 各自定性）；probe12/13（pattern standalone 精确）→ probe14 v6/v10/v11（链内损坏，构造方式无关）——约 5 分钟把「C4 索引错」从「reshape bug / 同步 bug / reduce bug」收敛到「pattern 在链内被损坏」，并排除 `pipe_barrier("PIPE_ALL")` 与 reshape 方向（任务工作区 `history_version/` 与 RETROSPECTIVE.md Stage 3，允许失效）
  - repro 载体：`repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py`（control vs trap 双向断言）
  - 交叉引用：queue VP-2026-0062（MTE 越界类三层崩溃定位法：崩溃边界矩阵 → 错误值指纹 → 新 cache 目录 IR dump）——本条补「未初始化 UB / 向量核 fault」分支与 pattern 类操作数的双向断言纪律
- repro: repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py（双向断言段即该手法载体；知识域已登记）
- toolchain_stamp: tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-debug-helper/references/mlir-dump-guide.md
- delta: |
    add 小节（置于「崩溃边界矩阵法」小节之后并互链）：**值错 vs 核 fault 的分离定位链**——① control：先只保留填充 + 拷贝（去掉消费链）；② 逐步加入 vcmp / vselect / reduce 等消费者，每步做 **standalone 精确 + in-chain 精确双向断言**（单向不足：`arange` 派生平面下标 pattern standalone 逐位精确而在整平面 vselect+reduce 链内读回损坏，2026-10-14 实证）；③ 每步 **一 case 一子进程**（`timeout` + 单 case argv）——向量核 fault（acl 507034 "Vector core execution timed out"，由读未初始化 UB 触发）会 core dump 整个进程并吞掉 stdout 缓冲，整文件跑会丢失此前全部结论（实测 probe6 全丢 vs probe7/8 七 case 各自定性）。判据：值错（可复现错值）与核 fault（进程级崩溃）两类现象物理分离，避免把 fault 误诊为数值 bug。
- status: pending
- confirmations: 1/2
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0062 同为 debug-helper 定位法族（不同文件域：本条落在 mlir-dump-guide.md 的定位法小节，合入时置于该小节之后并互链）。

## VP-2026-0163
- type: D
- title: int32 与 fp32 载体在元素级 UB 算子上的行为等价性（本工具链实测）——`vbrc`(typed 标量)/copy/vmax/vcmp/vselect/transpose/`i32→i64 vcast`/rank-2 slice 累加逐位正确；端到端全 int32 索引链 `torch.equal` 0/104487 mismatch ⟹「本仓无 int32 转置先例」不构成能力缺口
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 3：probe9（`i32 fill/vmax/chain` ch=8/32/64 全 OK）；本算子 `_func_general` 全 int32 索引链在 L0/L1 全量 `torch.equal` **0/104487 mismatch**（值 max_diff=0）；单算子 report `run.json` 本算子 rollup 7/7、`avg_max_abs_err=0.00e+00`（任务工作区 + 集成包）
  - 被澄清的旧论断：同任务 `DESIGN.md` §9.1 曾把「本仓无 int32 转置先例」列为风险并预置 fp32 承载回退（+ `h_in*w_in ≤ 2**24` 守卫）；`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 的 int32 ✓ 在实测下成立，回退路径未启用（保留源的 `< 2**31` 界即可）
- repro: repro-missing（知识域最小 repro 待同族 argmax/索引类任务回填——`i32` 元素级链最小 kernel 逐位对照 fp32 载体）
- toolchain_stamp: tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    update PL-1.5-quickref 追加行（或紧邻新增「int32 元素级链」条目，合入时按 layout.md 现状择优，不新开重复行）：**int32 与 fp32 载体在元素级 UB 算子上的行为等价性（910B3 实测，2026-10-14）**——`T.vbrc`（typed `T.int32` 标量）/`T.copy`/`T.vmax`/`T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast(i32→i64 rint)`/rank-2 slice 累加在 int32 载体上逐位正确（ch=8/32/64），端到端全 int32 索引链 `torch.equal` 0/104487 mismatch（值 `max_diff=0`）——`T.transpose` 的 int32 支持（docs §2.2.1）在实测下成立，「无 int32 转置先例」不构成能力缺口，无需降级到 fp32 载体（也就无需 `h_in*w_in ≤ 2**24` 守卫，保留源的 `< 2**31` 界）。适用条件：索引 dtype 需与源一致（int64 索引经 `T.vcast` i32→i64 落 GM）；`T.vbrc` 的标量 operand 须为 typed PrimExpr（见 traps-compiler.md TRAP-npuir-vbrc-region-dropped）。
- status: pending
- confirmations: 1/2
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 缺知识域 repro 降级 Tier 1；与 constants.md CONST-arange-scalar-materialize / traps-compiler TRAP-vcast-shape-and-gm-source（同族 dtype/形状约束）互链。

## VP-2026-0181
- type: D
- title: 窗口池化退化路径的「容量回退」不总可达——整线特化（C4）超 UB 时一般路径的 span 亦等于 `L_in` 且每元素成本更高（`2·eb+9` > `eb+9`）⇒ 该 shape 无 UB 可行路径、正确行为是如实拒绝；恒等特化（C4b）是反例（回退可达）
- evidence:
  - task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 3：`examples/max_pool1d/_max_pool1d_kernel/repro/EDGE-c4-fallback-infeasible.py`（**真跑断言**：`global=613129B > cap=196608` ∧ `general=724567B > cap` ∧ `general > global`（fp16、CH=8、`L_in=4096`、`k=4096`、膨胀系数 1.7）；factory 抛 `ValueError: no UB-feasible plan`；C4b 反例 `(1,8,16384) k1s1p0` 回退一般路径且 NPU 实跑 `torch.equal(out, x)` 逐位一致——任务工作区，未上库，provenance 允许失效）
  - 同任务 `_max_pool1d_kernel.py` L0-11a4/L0-12b 用例；DESIGN.md §8.3 L0-11 原写「超容即回退一般路径」被实现期推翻（Stage 3 Skill Flow Issues）
  - 交叉引用：hardware-cost-model.md §5 ①（VP-2026-0146——「退化路径容量放大 → 编译期判据决定特化或回退」；本条是该「回退」的**可达性**限定）；queue VP-2026-0183（develop 侧核算门禁）
- repro: repro-missing（任务内 repro 非自包含：import 任务 kernel 模块 + 头部缺条目 ID/版本戳 ⇒ 未达知识域转正要求不进 repro/；纯算术部分可数行重写——`general_bytes(CH, L_in, BL=1, eb) > global_bytes(CH, L_in, eb)`，或按公式直接列 `CH·L_in·(2eb+9) > CH·L_in·(eb+9) + CH·(25+eb)` 恒成立域）
- toolchain_stamp: tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ torch 2.9.0 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/_shared/standards/hardware-cost-model.md（§5「退化/几何特化路径的容量缩放」）
- delta: |
    update §5 ①（原文保留，追加限定与判据）：**「容量不可行即回退一般路径」不是普适兜底**——回退目标本身可能同样超容，须逐回退分支做**带字节数的可行性核算**后取「拒绝」而非「回退」：整线/整平面特化（覆盖集合 = 整轴）失败时，一般路径的 span 亦等于整轴长（该特化条件本就要求核覆盖整轴），而一般路径每元素驻留成本更高（max 池化实测 `2·eb+9` = row_src+row_dst+row_f32+eq_mask+eq_f32 > 特化 `eb+9` = flat+flat32+eq_mask+eq_f32）⇒ `general > global > cap` 恒成立域（fp16/`L_in=4096`/`k=4096`/CH=8/×1.7：general 724 567 B > global 613 129 B > cap 196 608 B，factory 抛 `no UB-feasible plan`）。反例：恒等特化（`k==1 ∧ s==1`）的一般路径 span = `BL`（与 `L_in` 无关），回退可达且逐位正确。⇒ 设计/实现须对每条「容量回退」分支核算回退路径的字节数，不可行时如实拒绝（`ValueError`）并记录。
- status: pending
- confirmations: 1/2
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: -
- decided_note: 承接 VP-2026-0146（已合入 §5 ①）的「回退」措辞限定；因缺知识域 repro 降级 Tier 1（merge-policy §1 ED-A 语义），任务内 repro 与知识域自包含要求不符（import 任务模块 + 头部缺条目 ID）故未机械转正——该判定已如实披露。

### Tier 2（R 类，结构化 diff 提案，待人工批准后 mode=apply 执行）

## VP-2026-0008
- type: R
- title: op-design Phase M0 第一问增加 golden 计算域核实检查项（fp16 golden 常见 fp32 opmath，勿假设与源算子舍入路径一致）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Skill Flow Issues（设计/检视期隐含假设未被拦截，L0 实测才暴露）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/.task_timeline.jsonl（Stage 3 attempt-1 precision fail，2485s）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py
- repro: torch.lerp CPU fp16（fp32 opmath+单次舍回）vs 逐步 fp16 舍入三步链：N=2^24 差 2–3 ulp、违反率 ~0.125%（pattern-library §2 opmath 行）
- toolchain_stamp: torch 2.9.0+cpu；tilelang-mlir-dev dev root build（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（扩展一行）
    定位锚: "2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。**golden 计算域核实**：golden 若为 `torch.*` CPU 实现，须用数行最小实验核实其在目标 dtype 下的实际 opmath 域——fp16 常见 fp32 opmath + 单次舍回（torch.lerp 实证：NPU 原生域逐步舍入差 2–3 ulp、容差违反率 ~0.125%，pattern-library §2）；「忠实源算子原生 dtype 舍入」与「过 golden 容差」冲突时以 golden 为准（源仓测试 ref 即 golden），kernel 侧用 fp32 中转对齐。
    动机: lerp_tensor Stage 3 attempt-1 precision fail（2485s）源于设计期「golden 舍入 = 源算子原生舍入」隐含假设；REVIEW 维度 8 预警了 v-prefix 侧舍入语义缺口但未覆盖 golden 侧 opmath 差异——设计期一道 3 行实验即可消除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0009
- type: R
- title: design-review 维度 2 UB 预算检查项补多 dtype 混合字节复算口径（dtype 中转路径禁沿用单一 dtype N-buffer 口径）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 维度 2 勘误（DESIGN.md §4.5 注与表自相矛盾：bf16 误写 fp16 同款 24576，正确 196608/24=8192）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 2 Value Point Proposals
- repro: 含 vcast 升精度中转路径的 block_size 上界 = UB容量 / Σ(bytes_i × count_i)（bf16 中转 24B/elem→8192；fp16 中转 20B/elem→9830 保守 8192）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；A2/A3 UB 192KB
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（改一行）
    定位锚: "| UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB）；含 dtype 中转（vcast 升精度）路径须按 Σ(elem_bytes × buffer_count) 混合字节宽复算，不得沿用单一 dtype 的 N-buffer 口径（lerp_tensor 实证：bf16 中转 24B/elem 上界 8192，误沿 fp16 口径 24576 需 480KB UB） |
    动机: lerp_tensor DESIGN.md §4.5 出现口径混用且注表自相矛盾，检视维度 2 原口径只查总和未查混合字节宽——勘误后才发现；若未拦截，Stage 3 assert 与 Stage 4 block_size 扫描范围都会错。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0011
- type: R
- title: conductor optimize 场景预检 baseline 字段补测量口径标注：Stage 5 events（host 侧 wall）数据禁作 headroom 依据
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md（Stage 5 bench events 口径 1M fp16 182.83us「慢于 torch」vs torch msprof 15.22us）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/.stage_state.json user_requirement（任务框定被 events 基线误导：「中小 N 慢于 torch，派发开销主导」）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md Baseline P1（msprof kernel-only 1M 7.30us 快于 torch 2.0–2.3x，全档快于 torch 1.65–2.3x）
- repro: 同 workload 双口径对照：events（host wall，含派发）vs msprof op Task Duration（kernel-only）
- toolchain_stamp: torch 2.9.0+cpu / torch_npu 2.9.0.post2；msprof cann-8.5.0；tilelang 0.1.2+ed787bb（2026-09-07 build）
- target_doc: .opencode/agents/conductor-scenarios/optimize.md
- delta: |
    动作: update（扩展一行）
    定位锚: "- 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）。**baseline 口径标注**：来自 Stage 5 bench（`prof_mode=events`，host 侧 wall、含端到端派发开销）的数值仅趋势参考、禁止作为 headroom 依据——kernel 时延唯一口径为 `msprof op Task Duration`（pattern-library §2 已弃用 event 口径；两口径方向可相反：lerp_tensor 2026-09-07 events 判 1M 慢于 torch、msprof kernel-only 判快 2.0–2.3x）；透传给 optimizer 的 events 基线须附此标注；
    动机: lerp_tensor optimize 任务的 user_requirement 即被 Stage 5 events 基线误导框定（「中小 N 慢于 torch」实为端到端派发开销失真）；靠 optimize skill 强制 msprof 唯一口径才免于方向错误——调度层透传时即标注可根除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0012

- type: R
- title: integrator 第一步后补 lint/format 自检步骤（TileOPs rule set B 对集成包生效，B023 类闭包绑定违反须在 pytest 前修复并留档）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md History mod 1（B023 "Function definition does not bind loop variable" + format.sh ruff-format auto-join；幂等重跑 integrate_kernel.py 会覆盖修复）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_s5_attempt1.py（pre-fix 备份）
  - 同款再现（2026-09-10 蒸馏追加，不同任务）：examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md Issues 第 1 条 + op 级 RETROSPECTIVE.md Transferable 末条（ALIGNMENT/_align_up 胶水补丁被幂等重跑整文件覆盖——「重跑前先备份 history_version/，重跑后按 Debug history 复放补丁」）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：cd examples/TileOPs && bash format.sh --files tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py（pre-fix 报 B023）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；TileOPs pyproject rule set B；torch 2.9.0+cpu / torch_npu 2.9.0.post2
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（第一步 bullet 列表末尾追加一条）
    定位锚: "- 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。
  - 脚本完成后、第二步 pytest 前，对集成的 kernel 文件执行 `bash format.sh --files <integrated files>`（TileOPs rule set B 对集成包生效，conductor 产物目录不受其约束）并修复 lint 违反（如 B023 循环变量闭包绑定用 default-arg 绑定：`def _to_npu(t, _dt=torch_dtype): ...`）；format/lint 修改记入 `integration_log.md` History（mod 条目）并在 `history_version/` 留 pre-fix 备份——幂等重跑 integrate_kernel.py 会整文件覆盖这些修复，重跑前须先备份；若修改触及 pytest 不覆盖的内嵌 L0 测试代码，须单独复跑内嵌 L0。
    动机: lerp_tensor 集成后 format.sh 才暴露 1 处 B023（内嵌测试闭包），且 format 自动改动使集成副本与 conductor 产物出现双向分歧；pytest 不覆盖内嵌 L0 代码，须单独复跑。前置自检把该类修改收敛为标准流程步骤。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0019
- type: R
- title: gate 1 S1-PLACEHOLDER 模板变量正则对中文数学集合记法误判——含分隔符的 {零, 有限真实数据} 类花括号集合被判「疑似模板变量未替换」造成 Stage 1 无效重试
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 两次 runtime fail：2026-09-07T12:44:47Z〔2960s〕与 13:47:06Z〔1518s〕，stage_retry_count 1→2）
  - .agents/tools/gate_lint.py#L36（TEMPLATE_BRACE_RE 定义）+ #L208-215（S1-PLACEHOLDER 判定）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`TEMPLATE_BRACE_RE.finditer("槽位尾行状态恒 ∈ {零, 有限真实数据}")` 命中 `{零, 有限真实数据}` → fail；误判文本出现在被门禁拒绝的中间版本上（归档 design_v0/v1/DESIGN.md 已无中文花括号——修订后通过）；发生账户来自 conductor 终态钩子输入（复盘工件未覆盖该两次 fail 的 gate 侧原因，同时构成复盘缺口）
  - 〔2026-09-15 复发，task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z〕新形态两处误报（Stage 1 gate 修复重试的 2×S1-PLACEHOLDER）：`e^{块内分数 − m_cur}`（CJK 数学上标，无列表分隔符）与 `_{\text{Cube 部和，f16 物化回传}}`（LaTeX \text{中文} 下标）——**前者不受本提案正则排除**（负向前瞻仅排除 [,、;；∈≤≥]，无分隔符的 `^{中文 − var}` 形态仍命中），审批时建议扩大排除集（brace 组前缀为 `^`/`_` 或含 `\text{` 时跳过）；writer 侧排版规则另立 VP-2026-0066 互补（改写消除模式命中：括号形式 / 注解移出公式，语义零变化）；纯 ASCII 花括号（`e^{m_prev − m_cur}`）不触发
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}', '恒 ∈ {零, 有限真实数据}')])"
- toolchain_stamp: gate_lint.py 现行版本（statectl gate 1 机械核对层）；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换一行正则定义）
    定位锚: 'TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")'
    old 文本: |
      TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    new 文本: |
      # CJK-brace groups that look like unfilled template slots. Set/math
      # notation with CJK members and list separators (e.g. "{零, 有限真实数据}")
      # is legitimate prose, not a template variable: skip such groups.
      TEMPLATE_BRACE_RE = re.compile(r"\{(?![^{}\n]*[,、;；∈≤≥])[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    动机: 负向前瞻排除含列表分隔符/数学关系符的花括号组——已验证 `{零, 有限真实数据}`/`{保留、舍弃}` 不再命中，`{算子名称}`/`{如: xxx}` 类真模板变量仍命中；本任务 Stage 1 两次 runtime 重试中 gate 误判贡献了无效重试（设计文档的集合记法是合法内容）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0023
- type: R
- title: DESIGN 伪代码单一权威源规则——伪代码与实现注/设计步骤矛盾时必须先消歧再冻结（单边权威声明不够）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（VP 行 4：v0 §3.3 伪代码 α 行 vsub(ub_m, ub_mcur) 与实现注①「快照式 m_prev」矛盾——修复 = 伪代码改写为注的形态而非加「以注为准」标记）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Skill Flow Issues Design-internal 行：§3.3 伪代码 mask→softcap 与 E3/§3.1 softcap→mask 矛盾——照抄伪代码会破坏 P(OOB)=0 不变式，实现被迫自行消歧）
- repro: 复现条件——含伪代码 + 实现注双源且顺序/操作数不一致的设计文档（本任务 v0 两处实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表后、Phase 5 前插入附注）
    定位锚: |
      11. 交付清单

  ```
  ### Phase 5：质量自检
  ```
  old 文本: |
      11. 交付清单

  ```
### Phase 5：质量自检
```
  new 文本: |
      11. 交付清单

      > **伪代码单一权威源**：§3 伪代码即权威实现形态，实现注只解释不纠偏——两者矛盾时修正伪代码本身（改写为注的形态），而非加「以注为准」标记；伪代码与其他章节（设计步骤/论证）多源矛盾时必须先消歧再冻结——单边权威声明不够（2026-09-07 attention expert 任务实证：mask/softcap 顺序矛盾若照抄伪代码会破坏 P(OOB)=0 不变式，softcap(−1e38)≈−softcap 非精确 0）。

```
### Phase 5：质量自检
```
  动机: 同一设计文档两处伪代码-注/步骤矛盾（α 行操作数顺序、mask/softcap 顺序），一处靠检视拦截、一处漏到 Stage 3 由实现者自行消歧——冻结前的消歧义务是缺口。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0024
- type: R
- title: tilelang-op-design Phase 6 修订纪律三条——多路线裁决三步法+正确性面第四维 / 公式修复「公式-论证-自述」三同步 / 概念删除与计数预算标注 grep 清零判据
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → design_v1.md → DESIGN.md（v0→v1→v2 修订链）+ DESIGN.md 修订记录（v1/v2 两节「为何不会再犯」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（路线裁决三步法 + 第四维 + grep 清零 + 三同步各 VP 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（三同步模板 + 计数/预算标注同步义务：「8 参」「6 输入」需带边界 pattern 防子串误命中）
  - 〔2026-09-20 第三任务证据 + 建议并入第 4 条，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕designer 修订头「问题编号 → 调整 → 为何不会再犯」三列表（DESIGN.md v1/v2 两轮均合格）使复检可逐项对账——第三轮复检据此对六步修复逐项核实；建议审批时把「修订头三列表」并入本提案 Phase 6 修订纪律（新增第 4 条：修订版头部必须带该表，防复检对账退化）
- repro: 复现条件——任一 Stage 2 不通过后的针对性修订（本任务 2 轮修订、牵连面 10+ 章节实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境同上
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 6 小节扩展）
    定位锚: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  old 文本: |
      ### Phase 6：针对性修订

  ```
仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  new 文本: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

  检视不通过后的针对性修订（revision mode）附加纪律（2026-09-07 attention expert 任务 v0→v1→v2 修订链实证）：

1. **多路线修复必须写明路线裁决**：检视给出多条修复路线时，修订版写明裁决依据与回退设计，不默认选改动面小的路线。裁决步骤：① 证据分级（文档规格 > 前序设计断言 > 形态缺口——缺口降级为「未实证假设」而非「不支持」）；② 结构收益量化（如直连使 dtype 同构、删 flag/ws 族）；③ 回退完整性（被否决路线降级为 R-x 回退而非删除）；④ 正确性面作为独立第四维（flag 族数、未初始化域、握手竞态数随路线的削减）。
2. **公式修复的「公式-论证-自述」三同步**：修正分界/阈值/取整公式（ceildiv↔floordiv 类）时同步恢复依赖该公式的整条下游论证链（保守性论证 / 收益前提 / 量化自述占比）——只替换公式 token 而不同步论证会保留旧矛盾或制造新矛盾；修复前执行代数重推 + 整除/不整除双数值例。
3. **概念删除与计数/预算标注的完成判据 = token grep 清零**：删除某概念（API/buffer/flag/参数）或切换预算基准后，以旧 token（含「N 参」「N 输入」「NKB」类标注）全文 grep 清零为完成判据（子串误命中用带边界符号的精确 pattern；合法语境白名单 = 回退方案/修订记录/历史版引用）。
```
  动机: 本任务两轮修订中：路线裁决无模板（自行三步法完成）、v1 删参未同步计数（第 2 轮问题 2/3）、K_A 修复若无三同步会留下 §5.4 占比与 M5 前提的内部矛盾——三条纪律均已被 3 轮检视逐项验证有效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0025
- type: R
- title: 弃选论证证据规则补第 6 条「前序实证的证据边界」——归档工件「N/M 实证通过」只证明实测路径可行，不得外延为替代路径不可行
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（问题 2：上一轮 23/23 载体链实证被用作「直连 bf16 不可行」的证据；负向断言「T.gemm bf16 ×」两轮原样继承且引用同一文档——两轮均未打开文档核对表格）
  - docs/Tilelang.language/线性代数操作/T.gemm.md#§2.2.1/§2.3（实际 bf16 √ + 实测可用——断言被推翻的对照证据）
  - examples/multi_head_attention/_prev_task_20260904_developer/_gqa_prefill_fwd_kernel/debug_log.md（前序实证的原始出处——provenance 交叉核对）
- repro: pytest testing/npuir/bf16_support_ops/test_gemm_bf16.py（被误读为「bf16 不可行」的反例证据链复核）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07 两轮设计 + 本轮复核）
- target_doc: .agents/skills/_shared/standards/negative-claim-evidence.md
- delta: |
    动作: update（§2 规则要点追加第 6 条）
    定位锚: "，与实测矛盾即 fail。"
    old 文本: |
      ，与实测矛盾即 fail。
    new 文本: |
      ，与实测矛盾即 fail。
      6. **前序实证的证据边界**：归档工件/前序任务的「N/M 实证通过」只证明其实测过的实现路径可行，不得外延为替代路径不可行——引用前序结论做负向断言时降级为「该路径已验证」并为替代路径另行举证（文档表格/测试/先例）；从前序任务继承的负向断言同样必须亲自打开所引文档复核到具体表格行/限制条款（「引用了正确路径、转述了相反内容」= 两轮设计同款失效形态，2026-09-07 attention expert 任务 bf16 断言实证：T.gemm.md §2.2.1 实为 bf16 √，前序断言驱动了整条载体链重设计后被推翻）。
    动机: bf16 负向断言跨两轮任务存活并驱动 E5 全链设计（载体链 +32KB UB、2 个 flag 族、Vector 载体负载），本轮文档+测试亲核推翻后直连形态 bf16/fp16 延迟差 ≤0.2%（pattern-library §1.8）——继承断言不亲核的代价是整条设计路线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0026
- type: R
- title: design-review 维度 8 补「公式与参数化标量验证」强制块——⟺ 分界声明代数重推+双数值例 / 量化自述对账 / per-lane 双端点代入 / 继承负向断言亲核
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 2 轮问题 1：K_A ceildiv 假等价在 v0/v1 两轮存活——`块 n 全有效 ⟺ n < ceildiv(a, bn)` 的 +1/−1 补偿直觉使公式「看起来对」；8 个 bx 数值枚举 design−precise=1、免 mask 域违例 2016 位置/任务）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（VP 行：自述即测试向量——§5.4「每任务 1–2 块走 mask」与公式联立 NK−K_A∈[1,2] 直接暴露公式错误）+ #Stage-2-r1（per-lane 双端点行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md#L221（同款公式两轮存活证据）
- repro: 复现条件——任一含「块级分界 + 条件跳过」的 DESIGN 检视（causal K_A / sliding window / 守卫域判定）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07，3 轮检视）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 引言 blockquote 追加一段）
    定位锚: "不得只复算数字而继承设计的前提。"
    old 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。
    new 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。**公式与参数化标量验证（强制）**：① 一切「⟺」分界/阈值/取整类声明（causal 分界、sliding window 边界、守卫域判定）执行代数重推 + 整除/不整除双数值例（a=65/bn=64/n=1 类反例）——ceildiv↔floordiv 差 1 的补偿直觉使公式「看起来对」，文字论证无法发现（K_A 反例两轮存活）；② 设计文档内的量化自述（块数/占比/频次）是与公式对账的不变式，自述-公式联立矛盾即线索；③ AIV/lane 半分类的参数化标量阈值代入两个 lane 端点数值例（vid=0/vid=1——单 AIV 思维推导在另一 lane 符号方向隐蔽出错）；④ 从前序任务/归档工件继承的负向断言必须亲自打开所引文档复核到表格单元格——前序实证只对其实测过的路径有效。
    动机: 本任务 3 轮检视中 K_A 假等价公式在第 1 轮维度 0/8 下 pass（文字论证自洽）、per-lane 符号错误靠 smoke 数值例才发现——四项核对均为「文字论证全对、公式错」形态的最低成本拦截器。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0027
- type: R
- title: 检视报告修改建议中的替代公式/参数写入前必须先经独立数值验证——检视建议是 Stage 1 revision 的逐字输入，建议错误的修复成本高于漏检
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1 修改建议③把 ceildiv 版 K_A 误背书为「保守下界」；第 2 轮问题 1 代数重推 + 反例 c=64/bn=64/n=1 + 8 个 bx 数值枚举确证为假等价——ceildiv 对 vid=0 是激进上界）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（v1 曾把建议③逐字固化进「K_A 保守性说明（防 Stage 3 误改）」——错误由「待核公式」升级为「已论证事实」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（Skill Flow Issues 首行 + Transferable「检视者对自己前轮输出的建议负有验证义务」）
- repro: 复现条件——检视报告给出具体替代公式/取整方向且被修订版逐字采纳（本任务 v1 实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 4 追加一段）
    定位锚: "按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。"
    old 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。
    new 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

  ```
  修改建议中给出的具体替代公式/参数/取整方向，写入前先经独立数值验证（代数重推或数值例）——检视建议是 Stage 1 revision 的逐字输入，建议自身携带错误会被修订版固化升级为「已论证事实」（2026-09-07 attention expert 任务实证：第 1 轮建议③把 ceildiv 版 K_A 误背书为「保守下界」，v1 逐字固化，第 2 轮代数重推 + 8 个 bx 数值枚举才确证为假等价）；检视者对自己前轮输出的建议负有与设计同等的验证义务——重审时必须重推建议本身的正确性，「建议被执行了」与「建议的目标达成了」是两回事。
```
  动机: 建议③误背书使错误公式获得检视权威加持、跨 v0/v1 两版存活且被「防误改」说明保护——该形态比漏检更难被发现，写入前验证是唯一低成本拦截位。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0028
- type: R
- title: design-review 修订版重审工作流——复用白名单/公式重推黑名单 + diff 锁定深查范围 + 新引入 API 亲验 + 建议原文逐句验收清账 + 三态判定
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮：v1→v2 diff 仅 8 处与修订记录声明一致；「复用 R2」标注 + K_A 全链重推策略；问题 4 三态判定「核心诉求成立 + 残留建议级」实践）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2/r3（复用白名单/重推黑名单行、diff 全量比对法、不变量对账清单、API 亲验行、原建议文本即 requirements 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v1.md（diff 比对基准）
  - 〔2026-09-17 第二证（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕RETROSPECTIVE.md re_review 章节——section-diff 复用形态独立复现：v0↔v1 按章节字节 diff 得 10 节 SAME / 9 节 CHG，SAME 节沿用首轮结论（同时证明修订无附带损伤）、CHG 节全量复核；**「修订残留只能靠全文 grep 抓、changelog 不可信作覆盖面依据」再证实**：v1 changelog 自称清理 6 处残留，复检仍独立发现 7 处未同步（§5.3/§9.1/§9.2 旧 UB 数字与 v1 直接矛盾等）——补充机械步骤见 VP-2026-0092（grep 关键词扫描 + 验证脚本 md5 比对）
  - 〔2026-09-20 第三任务独立复现，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕复检三段式：§0.1–§0.4 与 v0 逐字一致（diff 确认）→ 首轮源码行号级核对结论安全沿用；§0.5–§0.7 与 v1 逐字一致 → 上轮修复保留；变更半径审计（v1→v2 diff 28删/40增全落修复域，无越界改动/决策翻案）+ 决定性探针复跑（band 直载形态编译复证）——section-diff 复用 + diff 锁定深查范围第三任务实证；建议合入时把「变更半径审计 / 决定性探针复跑」两步骤并入本提案 new 文本的重审工作流
  - 〔2026-10-13 第四任务独立复现 + 补充「v_prev 问题闭合表」证据，task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z〕Stage 2 两次复检均以人工「v_prev 问题逐项闭合表」完成：v0 的 2 阻塞 + 4 建议在 v1 被逐项核对闭合（CH/膨胀系数/收益分母/PL-1.4 四处口径），据此才把注意力放到新问题（v1 新发现 C4b 恒等路径 1 阻塞为 v0/v1 通用 Tiling 核对均未触及）；第二轮复检确认 v1 遗留 1 阻塞 + 1 建议闭合、v0 六项回归保持闭合。**补充证据**：只重跑机械检查会漏掉「旧问题改了一半」的情形（本任务复检靠闭合表拦截）。建议把「复检模式强制产出 v_prev 问题逐项闭合表（阻塞必列 + 建议列，每项给本版证据句，未闭合照常 fail）」并入本提案的重审工作流（与既有复用白名单/diff 锁定/三态判定并列）
  - 〔2026-10-14 第五任务独立复现（不同任务，pool 族），task max_pool2d-_max_pool2d_kernel-20261014T062320Z〕Stage 2 复检轮由 conductor prompt 专项要求「**先建立 v0 问题闭合表，再查新问题**」，Reviewer 据此产出「上一轮（v0）检视问题闭合表」小节（5 项逐项闭合、1 处命名残留另列）并以 69 case-group 复跑为基线（v0 为 57 组）——闭合表形态第五次独立复现。同轮两条新发现落在本提案未覆盖的检查面：① **跨章节复述项取值漂移**——§3.2（API 映射表）的 C4 判决 `vcmp(nmax,zero,"gt")` 与 §3.3/§4.4 的 `"eq"` 矛盾，照抄 §3.2 会**反转 NaN 判决**（须对每个跨章节复述的谓词/比较模式/单位 `grep` 全部出现处核对取值一致）；② **负向守卫的子句隔离覆盖缺口**——C4x 两配置均对称 dilation（`dH==dW`），只删 `dW==1` 子句时断言不触发（「移除任一子句即失败」对该子句是空真）。**建议**：把「复检模式（Phase 0'：存在上一轮 REVIEW.md 时先逐项产出闭合表〔问题→证据句→状态〕再进入维度检视）+ §5 模板增『上一轮问题闭合表』小节（仅复检时输出）」并入本提案的重审工作流；①的动作面并入 VP-2026-0147（跨节一致性检查项）；②见 queue VP-2026-0177（特化谓词检查项含子句隔离覆盖）
  - 〔2026-10-14 证据链追加（task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z，pool 族双输出 1D 变体）〔第六任务复现〕Stage 2 **复审**（revision v1）由 reviewer 自发补建「前次阻塞项闭环核对」——2 条前次阻塞项（§4.5 容量系数 / §0.1 空窗口域不变量）逐项**独立复算**确认闭环（未采信 DESIGN.md 的「相对上一版调整表」自述）；并专项核「修复爆炸半径」时发现 §1.6.2 分析表仍留旧的两缓冲形态（`cfinal`/`cfinal2`）与 §3.2/§3.3/§4.3 的单一 `cfinal` 不一致（非阻塞 warn）——「跨节 buffer 清单一致性」是本提案工作流未覆盖的检查面（现由 VP-2026-0147 承担）。本任务的复审仍须自行 Read 上一版 REVIEW.md 重建阻塞项清单（skill 侧无 `Phase 0（revision 复审）` 步骤）〕
- repro: 复现条件——Stage 2 第 N 轮（N≥2）修订版重审（本任务 R2/R3 两轮实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 2 追加重审规则）
    定位锚: "按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。"
    old 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。
    new 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。

  ```
  修订版重审（第 N 轮，N ≥ 2）附加规则（2026-09-07 attention expert 任务 3 轮检视实证）：
- **复用白名单 / 重推黑名单**：前轮已验证的事实核对类结论（文档原文/源码行号/路径存在性/数字复算）可复用（标注「复用 R{N-1}」）；一切分界/阈值/取整公式与保守/激进的定性判断无论修订是否触碰均须重推——修订未触碰的段落不等于正确的段落（K_A 公式 v0/v1 两轮存活，唯一新阻塞恰在「复用区」）。
- **diff 锁定深查范围**：对「其余内容逐字保持上一版」类声明，以 `diff design_v{N-1}.md DESIGN.md` 全量比对验证——既验证声明属实（无未声明改动的隐性风险面），又自动锁定本轮深查范围（diff 命中处 = 必查）。
- **修订版新引入的 API 一律亲验存在性**（伪代码/映射表中首次出现者）——设计自述的「实查/亲验」记录不构成佐证，重跑同款 3 行验证才是检视证据。
- **修复验证对照建议原文逐句清账**：原建议文本是修复验收的 requirements 文档（每个验收子句独立判定达成/残留）；多目标建议部分达成时判「核心诉求成立 + 残留建议级」三态（记维度 warn + 修改方向），不整体放行也不整体阻塞。
```
  动机: 本任务 R2 漏检根因 = 复用策略未区分结论类型（事实核对 vs 推演判断）；R3 以 diff 锁定 + 公式全量重推 + 三态判定完成收口——规则均经实战验证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0029
- type: R
- title: design-review 维度 5 新增「flag 协议安全性」检查行——per-slot flag 逐族传递性推演法（写 i+slots 前读 i 已完成）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（三族 + 单槽 + 跨任务五条传递性论证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：本轮对五条论证独立推演全部成立）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（VP 行 5：推演法可直接复用）
  - 〔2026-09-20 第二任务独立证据，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕令牌推演法双用实证：① v0 检视据此检出尾置 slot-free wait 的 off-by-one WAR 竞争（T=2 写 ws[0] 只受 prologue 前导令牌放行——set 序 × wait 序逐对配对 + 写序位置核对，REVIEW.md 问题 1）；② v1→v2 修复正确性据此纸面复核（无需运行）。wait 任务头前置结论与 GQA/旧 ssd verified 形态一致——位置规则的数据条目化提案见 VP-2026-0096（合入时互链）
- repro: 任一 per-slot flag + 多槽 workspace 的 Expert 设计检视
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 5 表追加一行）
    定位锚: "| 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |"
    old 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
    new 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
      | flag 协议安全性 | per-slot flag + 多槽 workspace 设计按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手（TASKDONE 类）单列核对（2026-09-07 attention expert 任务 §6.3 五条论证推演法） |
    动机: Expert 跨引擎流水的正确性核心在 flag 协议而非循环结构——维度 5 原三行（循环/同步模式/尾块）不覆盖 flag 族传递性，本轮靠检视者自发推演完成，方法已验证可复用。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0030
- type: R
- title: design-review 维度 7 新增「统计行对账」检查行——四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮维度 7 warn：统计行分项归类与表体处置列标签 2 处错位——表体字面分布 6/7/6/4 vs 统计 5/8/5/5，跨 v0/v1/v2 三版存活）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r3（Skill Flow Issues 首行：前三轮检视只数总数 23=23，未做分项-标签逐行比对）
- repro: 复现条件——含四态/多态分类统计行的设计文档（本任务 §0.5 处置表三版存活实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 7 表追加一行）
    定位锚: "| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |"
    old 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
    new 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
      | 统计行对账 | 四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）——总数对齐但分项归类错位（6/7/6/4 vs 5/8/5/5）在 v0/v1/v2 三版存活，仅靠总数对账无法发现（2026-09-07 attention expert 任务实证） |
    动机: 总数对账只能发现双计类错误；分项归类与标签的错位需逐行比对——低成本高收益动作，防止错误统计被后续章节继承。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0031
- type: R
- title: conductor 修订调度 prompt 附加仲裁规则——检视建议跨轮冲突时显式标注「以最新轮为准 + 禁止恢复项」；design_error_summary 位置冲突以 REVIEW.md 为准
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（「以第 2 轮报告的 floordiv 公式为准，严禁按第一轮检视建议③的字面恢复」——本轮 conductor 调度输入显式指令防止了修订者回退到已推翻建议）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（Skill Flow Issues 两行：仲裁输入行 + design_error_summary 位置引用笔误行——「M5 收益前提（§1.6.2 #5）」实位于 §1.6.1）
- repro: 复现条件——Stage 2 多轮检视且后轮推翻前轮建议（本任务 v0→v1→v2 实证）
- toolchain_stamp: 流程规则（conductor 调度层）；证据环境 2026-09-07
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（设计修订循环步骤 3 追加仲裁规则）
    定位锚: "3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。"
    old 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。
    new 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。修订调度 prompt 附加仲裁规则：① 前轮检视建议被后续轮次推翻时，显式标注「以最新轮为准 + 禁止恢复项」（如「严禁按第 1 轮建议③字面恢复」）——无此指令时修订者可能因「上一轮建议已被执行过」而复用错误公式（2026-09-07 attention expert 任务实证：误背书的 ceildiv 建议曾逐字固化进 v1）；② `design_error_summary` 的位置引用与 REVIEW.md 位置列表冲突时以 REVIEW.md 为准（或摘要只引问题编号不重复位置）——摘要位置笔误会误导修订定位。
    动机: 检视建议跨轮冲突时修订者面对两个「检视权威」；显式仲裁输入是被实证有效的防回退机制（本任务 v2 修订记录明示该指令起了作用）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0032

- type: R
- title: integrate_kernel.py 补 stale perf_opt lineage 告警——换代重集成不清理上一任务 perf_opt/，占位符会静默解析到旧谱系调优 kernel（工厂签名兼容、测试不拦截）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md（§Bench observations 末条 + Files 清单：perf_opt/ 保留的 developer 谱系产物〔mtime 09-07 04:51，43KB〕早于 expert 基线 kernel〔09-07 17:22，52KB〕）
  - examples/multi_head_attention/RETROSPECTIVE.md#Stage-5（Skill Flow Issues 行 + Transferable 首条）
  - 〔2026-09-15 预警场景实际发生，task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z〕换代重集成（单遍 expert→两相位 v3）再次命中且以另一形态：integrate_kernel.py 幂等检查误判 already integrated（正则匹配到 r1-r3 调优轮注释掉的 baseline 行），wrapper 未重写、Sep-07 谱系 perf_opt import 静默保持激活（`w._gqa_prefill_fwd_kernel.__module__` 修复前解析到 perf_opt 旧谱系 kernel）；证据：integration_log.md Round 2 Integration steps Step 1（`wrapper_rewritten: false`）+ Step 2 人工胶水修复记录 + op 级 RETROSPECTIVE.md Stage 5 round-2 Skill Flow Issues 首行。互补缺陷另立 VP-2026-0064（幂等检查剥离注释行——修「不重写」侧；本条修「重写后不告警」侧），同 VP-2026-0055/0056 集群建议同批审批
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：换代重集成后 `ls -la tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/` 对比基线源 mtime——本任务实测旧谱系早 12.5 小时
- toolchain_stamp: tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0；TileOPs pyproject 环境
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（main 复制循环内追加谱系检查）
    定位锚: |
      (main() 的 sources 复制循环内)
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    old 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    new 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
          perf_dst = target_dir / "perf_opt" / src.name
          if perf_dst.exists() and perf_dst.stat().st_mtime < src.stat().st_mtime:
              print(
                  f"[warn] stale perf_opt lineage: {perf_dst} predates the "
                  f"baseline source {src}; it likely derives from a previous "
                  f"task lineage -- verify lineage (mtime / origin task) "
                  f"before activating the perf_opt import"
              )
    动机: 换代重集成（developer→expert）整文件覆盖基线但保留旧 perf_opt/；wrapper 的 perf_opt import 占位符切换后会解析到旧谱系 kernel——工厂签名兼容、smoke/full/bench 均不拦截，只有谱系核对能发现（本任务实测两谱系相差 12.5 小时与 9KB/640↔949 行）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0033
- type: R
- title: op_template.py 的 _run_case 加长签名传参规范注释——位序敏感 case 参数（≥4 个或多个同类整型）用 kwargs 或 case 元组，防 heads↔seq 写反
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Test-harness 行：6 参位序 (batch, heads, heads_kv, Sq, Skv, dim) 在 13 处手写调用中写反 2 类〔heads↔seq〕，1 次 SDPA GQA 崩溃 + 1 次 ValueError；全 MHA 位形静默通过、GQA 位形才炸）
  - /tmp/opencode/full_run2.log#L48（RuntimeError: The size of tensor a (1024) must match the size of tensor b (16)——位序写反的崩溃实录，session-local）
- repro: 复现条件——长位序 case 签名手写调用 ≥10 处（本任务 13 处写反 2 类实证）
- toolchain_stamp: 测试工程规则（无运行时依赖）；环境 torch_npu 2.x + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-develop/templates/op_template.py
- delta: |
    动作: update（_run_case 定义行前追加规范注释）
    定位锚: "def _run_case(M, N, dtype, tag):"
    old 文本: |
      def _run_case(M, N, dtype, tag):
    new 文本: |
      # Case-signature rule: when a kernel's case parameters grow beyond ~4
      # positional args, or contain several same-type integers (shape / dims /
      # head counts), pass them as kwargs (or wrap a case tuple) at every call
      # site -- transposed positional args (heads<->seq) pass silently on
      # all-MHA shapes and only crash on GQA/dispatch variants (2026-09-07
      # attention expert task: 2 of 13 hand-written calls transposed).
      def _run_case(M, N, dtype, tag):
    动机: 位序写反在 GQA 位形（heads≠heads_kv）才暴露，全 MHA 测试全绿掩盖错误——模板级规范使新算子测试套件从第一处调用就免于此类浪费。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0036
- type: R
- title: optimize SKILL.md Phase 2 步骤 7 补 blocked/invalid 分支归宿——perf_records.jsonl 仅收录完成验证与测量的分支（duration_us 恒为实测数、不得 null），结构性不可行证据归 opt_log invalid 行
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-3-注记（gate 4 窄修复记录：4 个 blocked-compile 分支〔bn=512 双槽 L1 溢出〕曾以 duration_us: null 写入 perf_records.jsonl，按契约移除、证据归 opt_log invalid_compile 行）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/.task_timeline.jsonl（Stage 4 attempt 1 fail 7546s〔gate S4-PERF-RECORDS-SCHEMA〕→ attempt 2 窄修复 complete 314s）
  - .agents/skills/_shared/standards/signal-registry.md#§5（字段契约：duration_us 类型 number；「每轮每个实验分支验证后追加一行」）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#终局验证-注记（2026-09-09 第二轮同主题实证〔task 20260909T033622Z，并入不新开 id〕：final causal 首行 3779.0 为手抄转录笔误，append-only 纪律下以更正行 3773.39 + opt_log 注记收尾——手抄数字是 null 之外第二类结构化记录失真源）
- repro: 复现条件——调优轮出现 blocked-compile/invalid 分支时的记录行为（本任务 Round 3：bn=512 双槽 4 分支全部编译失败无测量值）
- toolchain_stamp: gate_lint.py 现行 S4-PERF-RECORDS-SCHEMA 规则；发生环境 tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（步骤 7 扩展）
    定位锚: "7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。"
    old 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。
    new 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。**blocked-compile / invalid 分支不写入 perf_records.jsonl**——`duration_us` 恒为实测数值、不得为 null（signal-registry §5 契约，gate 4 `S4-PERF-RECORDS-SCHEMA` 强校验）；其结构性不可行证据（报错文本、L1/UB 溢出公式）记入 opt_log 当轮实验分支表的 invalid/blocked 行，证据不丢失。**行内数值一律从 msprof 提取输出直接复制（或管道）生成，禁止手抄**（2026-09-09 第二轮实证：final causal 首行 3779.0 转录笔误，靠 append-only 更正行 + opt_log 注记收尾）。
    动机: 契约的「验证后追加」语义未在写入指令处显式表达负面清单，optimizer 把 blocked 分支也「忠实记录」进结构化文件——2026-09-08 expert Stage 4 实证：4 个 bn=512 双槽 L1 溢出分支误记 null 致 gate 失败一次 attempt（7546s），窄修复又耗 314s；显式归宿规则可省整次重试。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0037
- type: R
- title: profile-collection.md msprof 输出目录权限规则从叶子目录扩展到父目录——group-writable 父目录下仅 chmod 700 输出目录采集仍被拒（8/8 baseline 首采失败实证）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP-R1 proposal：本轮 8 个 baseline 采集全部因 group-writable 父目录失败一次后才修正，前置检查可省一轮）
  - .agents/skills/tilelang-op-optimize/references/profile-collection.md#L96（现行规则仅 chmod 输出目录自身，未覆盖父目录）
- repro: 复现条件——在 group-writable 父目录（如默认 umask 002 的共享 checkout）下新建 msprof 输出目录、仅对叶子目录 chmod 700 后执行 msprof op 采集
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（权限规则行扩展）
    定位锚: "新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。"
    old 文本: |
      新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。
    new 文本: |
      新建 `msprof op` 输出目录后，先确保该目录**及其父目录**（至少直接父目录）group/other 不可写，例如 `chmod 700 {output_dir}` 并同步检查父目录；msprof 的权限校验覆盖父目录——仅叶子目录 700 而父目录 group-writable 时采集仍被拒（2026-09-08 expert Stage 4 实证：8/8 baseline 首采失败，父目录收紧后一次通过）。
    动机: 现行规则只覆盖输出目录自身；本任务 8 个 baseline 采集全部因 group-writable **父目录**失败一次后才修正——预检规则覆盖父目录可直接省一轮重采。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: BP-R1 整合去重：optimizer 自提的「profile-collection.md 补权限预检」经查现行文件已有叶子目录规则（67db6f3 期引入），本任务证据表明其不充分——按 update 形态提案而非 add。

## VP-2026-0039
- type: R
- title: profile-collection.md 补「探针 kernel 测硬件上限」标准形态——复刻目标数据流多模式对照 + 编译报错反推容量 + 真实 kernel 反推双证
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-8（probe_kvbw 四模式〔single/shared/distinct/pp〕同为 ~110.8µs kernel-only → L1 端口 r+w 148–154GB/s/核；bm=64@bn=512 `cc overflow` 报错反推 L0C=128KB；fabric ≥1.73TB/s 聚合）
  - 同文件#Skill-Retrospective-第二轮 有效手法①（硬上限测绘先行——一轮内把设计空间不确定性收敛为可算术的复合地板，直接决定 DESIGN_LIMIT 判定的证据质量）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md（三硬上限构成 [DESIGN_LIMIT] 2.09x/2.27x 缺口归因的实测基础）
  - pattern-library §1.9（数据侧落点：L0C/L1 端口/发射定律/fabric 条目，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z）
- repro: 复刻目标 kernel 数据流（同结构装载/gemm 模式）的最小探针 + 编译期容量压限（`overflow, requires N bits while M bits available` 报错文本）+ 真实 kernel msprof 计数器反推（流量÷时间）三方对照
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（文件末尾追加一节）
    定位锚: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
  ```
  ```
  old 文本: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
```
  ```
    new 文本: |
      返回：

  ```
```text
  PERF_DATA_COLLECTED
```

---

## 6. 探针 kernel 测硬件上限（DESIGN_LIMIT 证据 / 设计空间测绘）

  适用：调优目标涉及硬天花板判定（设计层归因、[DESIGN_LIMIT] 反馈、结构方向投入前的可行性测绘）时，在迭代候选之外用探针 kernel 把硬件/编译器上限测成可算术的数字。标准形态（2026-09-09 attention expert 第二轮实证：一轮测绘直接支撑 DESIGN_LIMIT 判定）：

  1. **复刻目标数据流的最小探针**：与目标 kernel 同结构的装载/gemm/访问模式，多模式对照（single / shared / distinct / ping-pong 双槽）区分 fabric、端口、容量因子——四模式同耗时 ⇒ 端口读写串行（双槽不能解除），非 fabric 争用。
  2. **编译失败报错反推容量上限**：刻意压限触发 BishengIR `overflow, requires N bits while M bits available` 报错（L0C cc / L1 cbuf / UB），从报错文本反推硬容量与 bm/bn 联合约束式。
  3. **真实 kernel 计数器反推双证**：探针值须与真实 kernel 的 msprof 计数器反推值互证（如 L1 r+w 流量 ÷ (mte2+mte1) 时间 = 端口带宽）；单源探针结论降级为待证。
  4. **探针纪律**：跨引擎探针必须带 Cube→Vector flag 同步（缺同步的数值崩坏是探针自身竞态而非 API 缺陷，pattern-library §1.9）；探针 kernel 保留 ≥1 个输入 tensor（§2 零输入行）。
  5. **归宿**：探针结论回写 pattern-library §1（数据侧）；触及编译器能力缺口时登记 `.agents/evolution/capability-gaps.md`；探针 kernel 脚本与日志入 `perf_opt/logs/`（jsonl 不收录非本算子分支的探针行，证据存日志与 opt_log 当轮表）。
```
  动机: DESIGN_LIMIT 判定与结构方向裁决的证据质量取决于硬上限数字的可信度；本轮 round 8 以该形态一轮完成 L0C/L1 端口/fabric 三项测绘 + 复合地板算术，直接封死无效调参方向（bm 上调/深流水/两遍 softmax）并支撑 2.09x/2.27x 缺口归因——无此形态需多轮试错反推。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0040
- type: R
- title: iteration-diagnosis.md「已知迭代结论」补续调任务前置基线复核——上轮交付驱动完整性（手改漂移检测）+ 基线重采对账（偏差 ≤ 噪声阈值）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#第二轮-前置修复（上轮终版测试驱动被用户手改：run_fa_tuned 仅余 fa4096、main 绕过 --level 分派；恢复后 `--level all` 全过〔logs/round8/driver_restore_level_all.log〕——手改漂移重新验证 ✓）
  - 同文件#Round-8-Diagnosis-1（基线复核：4 case 重采 18.19/29.69/77.62/263.93µs，与上轮终值记录偏差 ≤0.4%——谱系/环境一致性验证后才开调）
- repro: 复现条件——任一以「上轮终值为基线」的续调 optimize 任务（用户在手改过的 perf_opt 交付物上发起第二轮）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「已知迭代结论」小节扩展）
    定位锚: |
      ### 已知迭代结论

  ```
每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
  ```
  ```
  old 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
```
  ```
  new 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
本轮 profile 相比上一轮 profile 哪些指标发生变化
```

**续调任务（基线 = 上轮终值）的额外前置**（2026-09-09 attention expert 第二轮实证）：开调前完成两项复核——① 上轮交付测试驱动完整性：确认 run/case 分派逻辑与上轮终版一致（用户可能手改交付物——实证：run_fa_tuned 仅余 1 case、main 绕过 --level 分派；恢复后 `--level all` 全过再开调）；② 基线重采对账：重采上轮终值 workload 与上轮记录对账，偏差 ≤ 噪声阈值（实证 4 case ≤0.4%）才算谱系/环境一致，超阈值先查谱系与环境再开调。
```
  动机: 续调任务若直接信任被手改的测试驱动或未复核的基线，后续所有测量与归因都建立在失真地基上；本轮前置修复 + 重采对账各耗数分钟，拦截的是整轮方向的偏移风险。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0041
- type: R
- title: iteration-diagnosis.md 补「参考实现对照（morph ladder）」标准定位法——结构移植后性能不达预期时在参考实现上逐级叠加契约增量二分定位（op 级无罪 ≠ 组合无罪）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective-第三轮（有效手法① morph 阶梯决定性 + skill 流程问题①：形态学排查消耗 4 个 no_gain 分支后才启动，应更早切换；流程问题②：第二轮 [DESIGN_LIMIT] 地板归因未做同 contract 最小增量对照——有则 -20% transpose 税第二轮即被发现）
  - 同文件#Round-11-Diagnosis（morph 14 点矩阵：MORPH=0 参考原版 95.8 → MORPH=1 4D BSHD 输入 96.1 无罪 → MORPH=2c +lse 254.5 爆炸 → write-only 97.2 无罪 → 单 op 全无罪（vlog2/vmul+vadd/死源 transpose/4 个平凡 vadd）→ 任意含「活跃源 transpose」组合全部 253–255——唯一毒化点定位）
  - pattern-library §2 活跃源 transpose 毒化行（组合阈值效应数据侧，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（负结论推翻闭环的同款方法论缺口）
- repro: 复现条件——任一「结构移植/重写后性能不达预期（>1.3x）且仓库内存在同族参考实现」的调优轮（本轮实证：v11 初版 250.68 vs 参考 95.8–99µs，逐字转录参考 body 仍 255.53——上下文级差异形态学排查完全无效）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 代码结构观察」小节末条之后、「### 已知迭代结论」标题之前插入新小节）
    定位锚: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 已知迭代结论
  ```
    old 文本: （即上述定位锚原文）
  new 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 参考实现对照（morph ladder）

  仓库内存在同族参考实现（`examples/` 同 API 结构先例）且本方 kernel 结构移植/重写后性能显著不达预期（>1.3x）时，在逐项排查 op 形态 / pass_configs / 装载写法（含逐字转录参考 body）均无差异后，**优先切换 morph 阶梯定位，不在形态学排查上继续消耗实验分支**（2026-09-09 attention expert 第三轮实证：4 个形态学分支 no_gain 后 morph 阶梯一轮定位毒化点；且 [DESIGN_LIMIT] 地板归因若做此对照，-20% transpose 毒化税在第二轮即被发现）：

1. 在参考实现上逐级叠加本方契约增量（输入布局 → 输出契约 → 逐 op），每级一个实测点，二分定位差异源——**单 op / 单开关的无罪结论不可外推为组合无罪**（编译级阈值效应：活跃源 transpose 单 op 无罪、死源无罪、4 个平凡 vadd 无罪，任意含活跃源 transpose 的组合全部 2.6x 劣化，pattern-library §2）。
2. 对照点分级：契约级（输入/输出布局）先于 op 级；write-only 探针（只写不算）区分「计算成本」与「数据流形态触发」。
3. 纪律：perf-only 探针（故意破坏输出只测性能）在记录中显式标注；morph/探针脚本与实测点矩阵记入 opt_log 当轮（perf_records.jsonl 不收录非本算子 dispatch 的探针行）；scratch 探针结论须镜像回 opt_log 才可作为证据引用。

### 已知迭代结论
```
  动机: 「逐字转录参考仍慢 2.6x」类上下文级差异只有 morph 阶梯能定位（形态学排查对该类完全无效）；组合阈值效应使 op 级排除法结构性漏检——两条均指向该方法应为标准动作而非末位手段。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0045
- type: R
- title: op-design Phase 2 第 5 步等价性验证补双判据——基线式自身数值退化（pad 校正消灾难/大中间量结合序）时「候选 vs 基线」单判据假性 FAIL；补「候选对 fp64 误差 ≤ 基线误差（候选不得更差）」判据
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（compare() 双准则实现 + 首跑输出 randn-N2 fp32 案例 crit=b：基线 fp64err 1.58e-4 vs 候选 1.9e-7——候选更接近精确数学却被判 FAIL，首跑 3/8 违反）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Skill Flow Issues 首行 + Value Point Proposals 第二行：补判据后 45 案例全过且不掩盖真实回归）
- repro: 复现条件——任一基线式含 pad 校正/大中间量结合序的等价性验证（双判据形态自包含于 delta）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（输出「cases passing via criterion (b)」行）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（第 5 步「对照对象」bullet 扩展）
    定位锚: "   - **对照对象**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **对照对象与判定双判据**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）。**基线式自身数值退化时（GPU pad 校正的 catastrophic cancellation、大中间量乘法结合序）单判据「候选 vs 基线」会产生假性 FAIL**——补充判据 (b)：候选对 fp64 精确数学的误差 ≤ 基线误差（候选不得更差）；判定 = (a) 候选 vs 基线容差内，或 (b) 成立；(b) 路径通过须在结果表标注角点与双侧 fp64err（ada_layer_norm 实证 2026-09-10：randn-N2 fp32 基线 fp64err 1.58e-4 vs 候选 1.9e-7，补判据后 45 案例全过且不掩盖真实回归）；
    动机: D-1 验证的对照口径只写了「候选式 vs 基线式（同 dtype 舍入路径）」，基线退化场景把更优的候选判 FAIL——首跑假性 FAIL 3/8 案例浪费一轮排查；双判据以「候选不得更差」守住回归底线。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0046
- type: R
- title: op-design Phase 4 内存规划章节补 auto-multi-buffer 膨胀预算标定规则——按结构类平台值（TRAP-UB 第三证）而非「resident × 系数」，无同构先例时显式标 estimate + Stage 3 compile-probe 验证步骤
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Skill Flow Issues 首行：DESIGN §4.5 按「驻留 × 1.7」估膨胀，§8.3 L0-1 据此选 fp32 bm=3 首跑即编译失败 requires 245,760B > 192KB，一轮编译探针才修正为 20 B/elem 律并降档 bm 3→2）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md（Iteration 2/3 探针：transit 20.01 / fp32 26.00 / nl=4 22.00 B/elem——平台值精测）
  - pattern-library traps-compiler.md TRAP-UB-multibuffer-inflation（第三证，2026-09-10 已合入）
- repro: compile-probe 复现条件：persistent serial（num_local_tasks≥2）+ 20×bm×N > 196608 → bishengir 报文 `ub overflow, requires <bits> while 1572864 bits available`（报文 requires bits 即精确实测 UB 需求）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表第 4 项扩展）
    定位锚: "4. 数据规格与内存规划"
    old 文本: |
      4. 数据规格与内存规划
    new 文本: |
      4. 数据规格与内存规划（UB 膨胀预算按结构类平台值标定：persistent serial auto-multi-buffer 下 transit ≈20 B/elem、fp32 ≈26 B/elem、nl≥4 或静态 extents +2（pattern-library TRAP-UB-multibuffer-inflation 第三证）——不以「resident × 系数」估算；无同构先例的结构显式标 estimate 并写明 Stage 3 compile-probe 验证步骤：`ub overflow` 报文 requires bits 即该配置精确实测 UB 需求，读数反推系数后修正工厂 guard 与默认 bm 表）
    动机: ada_layer_norm DESIGN 按 ×1.7 估算（当时无同构先例），Stage 3 首跑编译失败一轮（245,760B > 192KB）才被 compile-probe 修正为 20 B/elem 律——设计期标定规则 + 探针验证步骤可把该修复前移或至少把验证步骤预置于设计中。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0048
- type: R
- title: core-split-strategy.md §2.2 reviewer 侧物理核数实查的替代验证条款——reviewer 无 NPU 设备连接时以「实查记录存在 + 常数表交叉 + 同款先例」三证放行，设备变更场景标注复核责任在 Stage 3/4
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 第三行：reviewer 无法独立重放 NPUUtils 实查〔需设备连接〕，本次以设计实查记录 §5.5 + CONST-aicore-910B2C + lerp 同款查询三方交叉佐证放行）
  - .agents/skills/_shared/standards/core-split-strategy.md §1 ②（实查记录要求，无 reviewer 侧独立验证手段条款）
  - 〔2026-09-28「有设备时直接重放」对照数据点，task argmax-_argreduce_kernel-20260924T031005Z〕reviewer 有设备场景：`NPUUtils.get().get_aicore_num()` = 24 于 v0 检视（05:15）与 v1 复审（06:34）两次独立重放一致；另五轮 Stage 1 探针实机重放（P1/P2/P2b/p3a×3/dbg，~12min）把 C-1~C-4 与冻结配置 bm=2 全部独立定证——「有设备优先重放实查、无设备走三证放行」的完整双分支口径（REVIEW.md v0 机械复核段 + Stage 2 复盘 VP 行，此前 VP 记录的是无设备分支）
- repro: 只读核对（无运行时依赖）——reviewer 环境无 NPU 设备连接时核数实查不可重放的位形
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/_shared/standards/core-split-strategy.md
- delta: |
    动作: update（§2.2 透传文本扩展）
    定位锚: "> 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      > 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。reviewer 环境无 NPU 设备连接、实查不可独立重放时，以三证放行：设计实查记录存在（查询代码 + 返回值）+ constants.md 物理核数条目交叉（CONST-aicore-910B2C）+ 同款先例（examples/ 同族查询记录）；目标设备非 910B2C 时三证链失效，核数复核责任显式移交 Stage 3/4（首跑核数断言或 profile Block Dim 核对）。
    动机: reviewer 侧无设备是常态位形（检视可在无 NPU 环境执行），现行条款只有 designer 侧实查义务、无 reviewer 侧替代验证手段——三证放行把该场景的核对义务显式化，避免形同虚设或过度阻塞。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0049
- type: R
- title: design-review Phase 1 第 7 步 verify_equiv 重跑核对补口径——「与内嵌表一致」须逐格对上确界（汇总格 = 逐案例输出的上确界），聚合口径不当的汇总值不构成不一致但须记建议级微瑕
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Value Point Proposals 首行：bf16 extreme-mod 案例逐输出 max_ulp=1 而 DESIGN §1.6.1 内嵌汇总格写 0——violation 计数与 EQUIV_PASS 判定不受影响，但「与内嵌表一致」的核对结论未逐格对上确界）
  - .agents/skills/tilelang-design-review/SKILL.md#L139（现行口径「执行结果须与 §1.6.1 内嵌结果表一致且全部 EQUIV_PASS」——「一致」未定义粒度）
  - 〔2026-09-28 第二形态（不同任务）：行数粒度对账，task argmax-_argreduce_kernel-20260924T031005Z〕重跑 verify_equiv 的 per-item 行数（E3=33）与 DESIGN 内嵌表（40 行）不符而总数与结论一致——逐行核对发现行数差异记为建议级（不因总数相符跳过）；「一致」口径除上确界外还须覆盖**行数粒度**（内嵌表行数 vs 脚本输出行数），出处：该任务 Stage 2 复盘 Transferable Lessons 末条 + REVIEW.md 等价性段
- repro: 复现条件——任一含内嵌汇总格的 verify_equiv 结果表重跑核对（汇总格 vs 逐案例输出上确界）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（bf16 段 extreme-mod 行 vs 内嵌表）
- toolchain_stamp: torch CPU + tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步首条 bullet 扩展）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）。「一致」的核对口径：汇总格（如「逐位一致 max ulp=0」）须等于逐案例输出的**上确界**（ada_layer_norm 实证：bf16 extreme-mod 逐案例 max_ulp=1 而汇总格写 0——violation 计数与判定不受影响，记建议级微瑕并要求修正汇总格，不判 fail）；
    动机: 「与内嵌表一致」未定义粒度时，聚合口径差异（min vs sup）与真实漂移不可区分——逐格对上确界是零成本口径收紧。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0050
- type: R
- title: design-review 维度 1「API 存在性」补 docs 无文档 API 的 examples 佐证豁免口径——T.min 类 API 在 docs/Tilelang.language/ 全子目录无独立文档（含未映射目录已枚举），3 处 examples 实调构成合法佐证
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 第二行：`ls docs/Tilelang.language/` 无 T.min.md；examples/flash_attention/flash_attn_npuir_dev.py L86 / lerp / logsumexp 三处实调）
  - docs/Tilelang.language/ 全子目录枚举（含创建操作/索引与元素操作/条件操作/排序操作/逻辑操作/原子操作/编译器提示操作——负向断言举证标准要求 docs 路径，但该 API 连正向使用也无文档可引）
- repro: 复现条件——语言文档全部子目录枚举无该 API 独立条目，且仓库示例代码实调检索得 ≥2 处跨算子目录的使用
  - 〔provenance，允许失效〕任务内核对命令：ls docs/Tilelang.language/数学操作/ | grep -i min（无 T.min 条目）；grep -rn "T\.min(" examples/ --include="*.py" | head（多处实调）
- toolchain_stamp: 仓库 docs 现状 2026-09-10；与 tilelang 版本无关
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 1 表「API 存在性」行扩展）
    定位锚: "| API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |
      | docs 缺文档 API 的佐证口径 | API 在 docs/Tilelang.language/ 全子目录（含未映射目录）无独立文档时（实证：T.min，2026-09-10 枚举），≥2 处 `examples/` 实调（跨算子目录）构成存在性佐证，检视记录注明「docs 无条目 + N 处实调」——负向断言举证标准不因文档缺口而弃用，但正向存在性不要求 docs 条目 | 仅有 docs 条目而无任何实调（文档与实现漂移风险） |
    动机: 负向断言举证标准（须引 docs 路径）遇到「API 无文档但实际存在」时举证责任无法满足——T.min 实证靠 3 处 examples 佐证放行；口径明示化避免同类场景每次重新论证。配套（超出本提案写域，需人工/contributor 处理）：docs/Tilelang.language/数学操作/ 补 T.min 条目（标量 min，含尾块 real_mode 用法）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0051
- type: R
- title: develop SKILL.md Phase 4 结果收集补失败分类要求——compile/runtime-error 与 precision-fail 分类上报（混排同列会使 conductor 的 precision_fix 路由把编译错误判为精度错）
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Skill Flow Issues 第二行：首跑 12 个 L0 失败混有两类根因〔1 个编译期 UB 溢出 RUNTIME-ERROR + 11 个 vrsqrt 精度 FAIL〕，汇总输出未分类）
  - 同任务首跑输出（`[collected] L0-1 RUNTIME-ERROR` 与 `[L0-5] FAIL: max_diff=...` 混排同列）
  - .agents/skills/tilelang-op-develop/SKILL.md#L48（现行「3. 收集结果：max_diff、失败用例 shape、层级」——无失败分类要求）
- repro: 复现条件——任一 L0 首跑同时含编译失败与精度失败的混合位形（本任务实证）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 收集步骤扩展）
    定位锚: "3. 收集结果：max_diff、失败用例 shape、层级。"
    old 文本: |
      3. 收集结果：max_diff、失败用例 shape、层级。
    new 文本: |
      3. 收集结果：max_diff、失败用例 shape、层级；失败须按 **compile/runtime-error vs precision-fail 分类**上报（分类计数 + 各类代表案例），不得混排同列——直接影响四出口判定与 conductor 路由精度（ada_layer_norm 实证 2026-09-10：首跑 12 失败 = 1 编译期 UB 溢出 + 11 精度 FAIL，未分类时 precision_fix 路由会把编译错误误判为精度错）。
    动机: 四出口判定与 conductor 路由按失败类型分流；混合失败位形下无分类的汇总使路由依据失真。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0053
- type: R
- title: optimize SKILL.md Phase 2 第 6 步「每个实验分支跑 L0 精度回归」补分支类型分层——参数分支（同 kernel 不同 config）可用 bench 内嵌 golden check 等价替代，结构分支必须全量套件
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP proposal：参数扫描分支〔同 kernel 不同 bm〕的精度门禁 = bench `--check` 精确 (M,N,dtype,bm) golden 比对；结构分支 = 全量 `--level all`——该分层未显式定义，「每个实验分支跑 L0」对参数分支语义模糊，内嵌 L0 表不覆盖被扫的 bm）
  - 同文件 Iteration 1/2/3（参数分支 18+16+8 行记录全走 --check、结构分支走 --level all 的实际执行形态）
  - 〔2026-09-17 证据扩展（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕**结构分支 L0 回归须含 bf16 × 非整除 shape 组合**：v5_xband 引入的 L1 band 尾块越界（`s0:s0+bs` 未按 `ts=min(bs,Q−s0)` 裁剪）在 fp16 Q=96 靠 L1 布局运气通过 L0、潜伏 4 轮，仅 Boundary 层 bf16×Q=96 拦截（traps-runtime.md TRAP-L1-band-dst-tail-overrun；opt_log §6.1/§9.2）——「结构分支必须全量 --level all」若不含 bf16×非整除组合仍是假门禁
- repro: 复现条件——任一含参数扫描分支（同 kernel 不同 bm/config）的调优轮（本任务 round 1/3 实证形态）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 2 第 6 步扩展）
    定位锚: "6. 每个实验分支跑 L0 精度回归。"
    old 文本: |
      6. 每个实验分支跑 L0 精度回归。
    new 文本: |
      6. 每个实验分支跑 L0 精度回归，按分支类型分层：**参数分支**（同 kernel 仅改 config，如 bm/num_kernels 扫描）可用 bench 内嵌 golden check（精确 (shape, dtype, config) 比对，如 bench.py `--check`）等价替代完整 L0 套件——内嵌 L0 表不覆盖被扫的 config 值；**结构分支**（kernel 代码改动）必须全量 `--level all`（ada_layer_norm 2026-09-10 实证形态：round 1 参数扫描 18 行全 --check、round 2 结构改动全量 + 分支 `--level all`）。
    动机: 「每个实验分支跑 L0」对参数分支语义模糊（L0 表按基准 config 参数化，不覆盖扫描值）——显式分层消除歧义，避免参数分支误跑不覆盖的全量（浪费）或结构分支误用单点 check（漏检）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0055
- type: R
- title: integrate_kernel.py 产物侧校验两缺陷——① wrapper 契约名在产物中无顶层定义时 gen_init_py 静默降级 [warn]（__init__.py 仍声称 __all__ 含该名，smoke 才 ImportError 且无修复指引；幂等重跑覆盖集成侧补丁）；② gen_kernel_source_block 的 stem fallback 把非函数名（常量/helper）当模块路径生成无意义 perf_opt 占位行
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md（Debug history attempt-1：首跑 `[warn] ALIGNMENT: not found in integrated modules` ×2 → `[error] import smoke test FAILED`；Issues 第 1/2 条）+ ada_layer_norm.py L61/L63（`# from .ada_layer_norm_kernel.perf_opt.ALIGNMENT import ALIGNMENT` 类占位行）
  - examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L158-162（defining_module==None → [warn] 注释行）+ #L211（`stem = defining_module(name, integrated_files) or name`）
  - 交叉引用：queue VP-2026-0032（同文件 stale lineage 告警——缺陷集群）、VP-2026-0012（幂等重跑覆盖集成侧修复——本任务 Stage 5 同款再现，证据链互链）
- repro: 重跑 python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/norm/ada_layer_norm/.migration_meta.json（覆盖集成副本后 smoke 必现 ImportError——provenance，允许失效）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（两处）
    定位锚 1: |
      (gen_init_py 函数体内)
              stem = defining_module(name, integrated_files)
    （其后 [warn] 分支，L158-162）
    old 文本 1: |
              stem = defining_module(name, integrated_files)
    new 文本 1: |
              stem = defining_module(name, integrated_files)
    （[warn] 分支升级为可操作 [error]：parse_wrapper_imports 得到的 wrapper 契约名在产物顶层定义缺失时——提示两条修复路径：集成包内逐字 re-export GPU extracted 的同名定义（Stage 5 胶水先例，见 ada_layer_norm integration_log attempt-1）/ 调整 scaffolder-integrator 契约透传 helper 名；并保留已修补的非产物段不再整文件覆盖〔幂等重跑保护〕）
    定位锚 2: |
      (gen_kernel_source_block 函数体内)
              stem = defining_module(name, integrated_files) or name
    old 文本 2: |
              stem = defining_module(name, integrated_files) or name
    new 文本 2: |
              stem = defining_module(name, integrated_files)
              # Non-function names (constants/helpers) have no per-func
              # perf_opt module: fall back to the single integrated module
              # stem and import by name, or emit no placeholder line --
              # never use the bare name as a module path.
    动机: 契约名缺失从「静默 warn + smoke 期 ImportError 无指引」升级为前置可操作 error（修复手法已在 ada 案例验证）；占位行 stem fallback 对常量/helper 生成语义错误的模块路径（注释态无害但误导后续 wrapper 切换）。两缺陷与 VP-2026-0032/0012 同属 integrate_kernel.py 幂等与校验缺口集群，建议同批审批。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0056
- type: R
- title: integrator 已知修复目录「import / 路径错误」行补 extracted-import 契约缺口形态——wrapper 从 GPU extracted 模块导入的全部名字（含常量/helper）构成集成 re-export 契约；NPU 重设计丢弃对应机制时最小修复 = 集成副本逐字 re-export 同名定义
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md（Debug history attempt-1 完整修复链：ALIGNMENT/_align_up 逐字 re-export + __init__ 修正 re-export；安全判别——名字只影响被 NPU 工厂接受但忽略的 flag 或无消费者的 config 选择器路径）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py L13-21（逐字源：GPU extracted _ada_layer_norm_fwd_kernels.py）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/RETROSPECTIVE.md（Stage 5 Transferable Lessons：wrapper 政策函数与 pytest 政策测试仍校验 GPU 语义，属 wrapper 胶水契约——集成期不顺手 NPU 化）
- repro: 复现条件——任何 wrapper extracted-import 含非 kernel 名字（常量/helper）且 NPU 产物不定义它的迁移（重跑 integrate_kernel.py 覆盖集成副本 → smoke 必现 ImportError）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（已知修复目录表「import / 路径错误」行扩展）
    定位锚: "| import / 路径错误 | 检查聚合 `__init__.py` 与相对导入 |"
    old 文本: |
      | import / 路径错误 | 检查聚合 `__init__.py` 与相对导入 |
    new 文本: |
      | import / 路径错误 | 检查聚合 `__init__.py` 与相对导入。**extracted-import 契约缺口**（ImportError 指向常量/helper 名而非 kernel 名时）：wrapper 从 GPU extracted 模块导入的全部名字构成集成包 re-export 契约；NPU 重设计丢弃对应机制（padding/cp.async 类）致产物不顶层定义这些名字时，最小修复 = 集成副本逐字 re-export GPU extracted 的同名定义（语义零漂移，wrapper 政策函数与 pytest 政策测试继续校验 GPU 语义——集成期不顺手 NPU 化）；安全判别：名字只影响被 NPU 工厂接受但忽略的 flag 或无消费者的 config 选择器路径。修复后幂等重跑会覆盖补丁（同 VP-2026-0012/0055 caveat） |
    动机: 该形态在本任务以 1 attempt 闭环（修复手法验证有效），但检索无门——已知修复目录只有泛化的「检查 __init__」；契约语义（名字清单 = 契约）与最小修复形态入目录后，同类迁移任务可一次定位。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0064
- type: R
- title: integrate_kernel.py 幂等检查只认未注释的激活 import——rewrite_wrapper_import 的 "already integrated" 探测正则匹配到被注释的 baseline 行（调优轮 perf_opt 激活态遗留），wrapper 不重写、旧谱系 perf_opt kernel 静默保持生效
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 Integration steps Step 1（`[wrapper] ... already integrated`〔wrapper_rewritten: false〕时 wrapper L65 baseline 被注释、L67 perf_opt import 生效；`[smoke]` 仅 import 冒烟不区分激活源）+ Step 2 人工胶水修复（`w._gqa_prefill_fwd_kernel.__module__` 修复前解析到 perf_opt 旧谱系）
  - examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L224-226（text 全文 re.search，不剥离注释行）
  - 交叉引用：queue VP-2026-0032（stale perf_opt lineage 告警——预警场景本轮实际发生的另一形态，修「重写后不告警」侧）、VP-2026-0055/0056（同文件幂等与校验缺口集群，建议同批审批）
- repro: 复现条件——wrapper 处于 perf_opt 激活态（调优轮遗留：baseline import 被注释）时重跑 integrate_kernel.py → 输出 already integrated 且 wrapper_rewritten=false，旧谱系 perf_opt import 保持激活
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15；脚本现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（幂等检查剥离注释行后匹配）
    定位锚: |
      (rewrite_wrapper_import 函数体内)
          text = wrapper_path.read_text(encoding="utf-8")
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
              return False
    old 文本: |
          text = wrapper_path.read_text(encoding="utf-8")
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
              return False
    new 文本: |
          text = wrapper_path.read_text(encoding="utf-8")
          # Idempotency must consider only ACTIVE (uncommented) imports: a
          # commented-out baseline line (left by a perf_opt-activation tuning
          # round) must not satisfy this probe -- otherwise the wrapper is
          # never rewritten and the stale perf_opt import stays active
          # (factory-compatible signature; import smoke does not detect it).
          active_lines = [
              line for line in text.splitlines()
              if not line.lstrip().startswith("#")
          ]
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", "\n".join(active_lines)):
              return False
    动机: 换代重集成遇 perf_opt 激活态 wrapper 时被注释的 baseline 行骗过幂等检查——旧谱系调优 kernel 静默保持生效且 smoke 不拦截（2026-09-15 实际发生，靠人工 __module__ 核对才发现）；与 VP-2026-0032 成对互补。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0065
- type: R
- title: develop SKILL.md Phase 4 L1 门禁补「wrapper-default 派发路径」——带 config 替换语义的 kernel，出厂 dispatch 的 traced 变体必须 ∈ 被门禁编译验证过的变体集
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/RETROSPECTIVE.md Stage 5 round-2 Skill Flow Issues 第 2 行（Stage 3 L1 门禁用逐字 config `fn(64,64,ns=2)` → bn_eff∈{64,144} 验证 manifest 域，wrapper 实际派发走 E6 替换路径 (64,64,1)→(64,256)；L0-3/L0-4 仅 dim=64 smoke 验证替换路径）
  - integration_log.md Round 2 §Design-layer finding（E6 替换路径在 dim=128 causal 域的首次编译发生在 Stage 5 bench 且直接 UB 溢出 8/10；pytest 4 参数全 causal=False 三重掩盖）
  - 〔2026-09-16 第二证（不同任务），task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z r9 precision_fix〕同类失效再发生：r7e per-shape bm=80 分派 × E6 bn 钳位（S_kv≥3841 → bn_eff=288）的组合变体从未被任何门禁编译（standalone 29 例 S_kv≤2048 / manifest S≤2048 / 此前 pytest 域未覆盖该 S），首次编译发生在 TileOPs pytest full-fwd-bf16 且直接 UB 溢出 13472B（bishengir-compile exit 1）——「出厂 dispatch 的 traced 变体 ∉ 门禁验证集」的同一失效类，且两个守卫各自安全、组合超限（联合核算判据 + repro：attention.md PL-1.12 r9 update / repro/PL-1.12-bn-clamp-bm-guard.py）。Tier 2 人工门不因证据数豁免，供审批参考；与 VP-2026-0074（修复侧 config-契约范式）互补
  - 〔2026-09-21 第三现（不同任务、kernel 内配置分支亚型），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 重建会话〕tuned bn=128 经 min(bn,N) 钳位使 N_TILES 恒为 1——kernel 自带 L0/L1 门禁（TUNED config 口径）**永不触达 N_TILES_MULTI 代码路径**，而 tileops wrapper 的 GPU 启发式默认 bn=min(64,N)（H=80 时 N_tiles=2）会走该路径；结构改动涉及配置分支时须补 bn<N 探针（本会话 `perf_opt/probe_bn64.py` 4 case：双 dtype × 尾块 × 双 pp 全过）。**建议 apply 时把 new 文本的「门禁变体集」口径扩为「默认/tuned 双配置 × 关键配置分支形态」的笛卡尔积**（含 tilelang-op-optimize SKILL.md Phase 2 分支验证清单同句）；数据侧已回写 attention.md PL-1.18 update③（BP_config_path_coverage）
- repro: 复现条件——任一 kernel 工厂带 config 替换/分派语义的迁移任务（wrapper default_config 派发的 traced 变体未入 L1 变体集时，出厂首编后移到 Stage 5 bench 暴露）
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 第 2 步扩展）
    定位锚: "2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。"
    old 文本: |
      2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。
    new 文本: |
      2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。**门禁变体集覆盖 wrapper-default 派发路径**：kernel 工厂带 config 替换/分派语义时，wrapper `default_config` 实际派发的 traced 变体（含替换路径与惰性旋钮组合）必须被 L1 在目标域（dtype/dim/causal 维度）编译验证——至少一组与 wrapper `default_config` 全同的调用 + 替换路径的目标域案例（2026-09-15 反例：E6 替换路径 bn_eff=256 在 dim=128 causal 域首次编译发生在 Stage 5 bench 且 UB 硬溢出 8/10，L1 逐字 config + L0 dim=64 smoke + pytest 全 non-causal 三重掩盖）。
    动机: 出厂 dispatch 变体不在门禁变体集内时，唯一编译门控后移到 Stage 5 bench——集成期编译失败按功能缺陷查根因的成本远高于 L1 期发现。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0066
- type: R
- title: design SKILL.md Phase 4 补 CJK 排版规则——数学记号中 CJK 只出现在圆括号或正文、不进 {...}（含 LaTeX \text{中文} 下标与 e^{中文 − var} 上标），规避 S1-PLACEHOLDER 误报；命中合法记号时改写消除模式命中而非申诉豁免
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 gate-修复重试章节 Skill Flow Issues 首行（DESIGN.md L41/L446 两处误报 `e^{块内分数 − m_cur}`、`_{\text{Cube 部和，f16 物化回传}}`；修复 = 括号形式改写 + 注解移出公式，`grep -P '\{[^}]*[\x{4e00}-\x{9fff}][^}]*\}'` 复核清零，语义零变化）
  - 交叉引用：queue VP-2026-0019（gate_lint 正则侧修复——本条为 writer 侧互补；2026-09-15 新形态 `e^{中文 − var}` 无列表分隔符，VP-2026-0019 现提案正则不覆盖，见其证据链追加）
- repro: 复现条件——DESIGN.md 数学记号含 CJK-in-braces 形态（`grep -P '\{[^}]*[\x{4e00}-\x{9fff}][^}]*\}' DESIGN.md` 有命中）时跑 gate 1
- toolchain_stamp: gate_lint.py 现行版本；误报发生于 2026-09-15 任务
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一条 blockquote）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **CJK 排版规则（S1-PLACEHOLDER 误报规避）**：DESIGN.md 数学记号中的 CJK 只出现在圆括号或正文，不进 `{...}`（含 LaTeX `\text{中文}` 下标与 `e^{中文 − var}` 形态上标）——gate 1 的 CJK-in-braces 模式会命中此类记号判「疑似模板变量未替换」。命中合法记号时的修复姿势 = 改写消除模式命中（括号形式 `e^(...)` / 注解移出公式为正文括注），语义零变化、门禁可机械复核，不申诉豁免（2026-09-15 实证：两处误报改写后 `grep -P` 复核清零；纯 ASCII 花括号不触发）。
    动机: gate 1 误报消耗整段修复重试（本任务 Stage 1 重试的 2/4 项）；writer 侧预防与 gate 侧正则修复（VP-2026-0019）双管齐下，正则修复未落地前排版规则是唯一防线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0067
- type: R
- title: design SKILL.md「超长文档分段落盘」从条件建议升级为阈值无条件触发——VP-2026-0022 于 2026-09-11 apply 后 2026-09-15 仍复现首写截断（「预计超过」的自行判断是失效点），heredoc 分段追加为可靠替代
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节 Skill Flow Issues 首行（1116 行 / ~95KB DESIGN.md 首次 write 即截断——上版建议已合入仍复现且更早失败；bash heredoc 6 段追加成功）
  - .agents/skills/tilelang-op-design/SKILL.md Phase 4 blockquote（VP-2026-0022 2026-09-11 apply 文本——条件式「预计超过 ~60KB」）
  - 旁证（同任务 Stage 3 侧 sibling）：.task_timeline.jsonl Stage 3 attempt 1 fail 809s（空返回——单次 LLM 响应生成整个大文件被 provider 截断；VP-2026-0021 同样 2026-09-11 apply 后仍首发失败，conductor 增量写入流程指令后才成功）——分段纪律的条件式表述在两个 agent 上同日失效
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-15 本任务首设计 attempt，规则已合入状态下复现）
- toolchain_stamp: 会话层行为（无运行时依赖）；复现环境 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（替换 Phase 4「超长文档分段落盘」blockquote 措辞）
    定位锚: "> **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。"
    old 文本: |
      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
    new 文本: |
      > **超长文档分段落盘（>60KB 一律执行）**：DESIGN.md 预计超过 ~60KB / ~1200 行时一律分段落盘（bash heredoc 分段追加为可靠替代——单次 Write 截断后 heredoc 不受工具 JSON 体积限制），不以单次整文件 Write 交付，不以「预计不会超」的自行判断跳过——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 首证：1249 行首写即截断；**2026-09-15 复发**：本规则 2026-09-11 合入后 designer 仍先单次 Write 且首写即截断〔~95KB〕，heredoc 6 段恢复——条件式「预计超过」判断是失效点）。
    动机: 条件式建议在两个 agent（design/develop）上同日失效（Stage 3 侧 VP-2026-0021 同款复发）——阈值触发改为无条件 + 明示 heredoc 替代，把「是否分段」从 agent 自由裁量变为机械规则。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0068
- type: R
- title: design SKILL.md Phase 2 等价性机器验证补「判定口径」bullet 三条——ULP 阈值按恒等式域界第一性推导（勿拍小常数）/ 逐项隔离显式关闭其他量化开关 / 探针行与判定行分离
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 gate-修复重试章节 Value Point Proposals 后三行（M1 首跑 15 ulp 对拍定 8/16 阈值假 FAIL → 域界推导 |t|≤87 → 上界 ~120 ulp，阈值 128 后 viol 双口径 0；M14 沿用默认管线被 M12 的 ~2.6e-3 相对误差污染 viol 87% 假 FAIL → 隔离后 0；基线式失败演示只进 info 不参与判定）+ verify_equiv.py（13/13 EQUIV_PASS，双跑 diff 一致）
- repro: 复现条件——任一含 softcap/深尾域或多项量化开关的 verify_equiv 编写（任务内复现命令见 evidence provenance 项，torch CPU）
- toolchain_stamp: torch 2.7.1+cpu / 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 5 项「对照对象」bullet 后追加一条）
    定位锚: "   - **对照对象**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）；"
    old 文本: |
      （即上述定位锚原文，保持不变）
    new 文本: |
      （定位锚原文保留）其后追加：
         - **判定口径**：① ULP 阈值按恒等式结构第一性推导，不拍小常数——exp 类恒等式的 ulp 差与指数入参域界成正比（|δt| ≤ c·u·|t|max → P 域 ulp ≤ c·|t|max·ln2；softcap 域 |t|≤87 → 上界 ~120 ulp，实测 15——拍 8/16 在深尾假 FAIL）；② 逐项隔离对照显式关闭其他量化/物化开关（被验证项之外全 off，否则他项误差污染本项参照门——M14 实测 viol 87% 假 FAIL → 隔离后 0）；③ 基线式失败模式演示（溢出/NaN/发散）只进 info 不进 PASS/FAIL（探针行与判定行分离）（2026-09-15 attention 两相位 verify_equiv 实证，13/13）。
    动机: 首跑两类假 FAIL（阈值拍脑袋 + 隔离缺失）都发生在「脚本已产出、口径不当」阶段——判定口径入 skill 可在首次编写时避开，省一轮 gate 往返（本任务 gate 重试 4 项中 2 项属 EQUIV-EXEC 类）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0069
- type: R
- title: algorithm-research.md §6 信息源层「本仓 examples 同类实现」补全目录清单完成标准——结构先例检索以 grep -rl 族关键词显式列举命中清单并逐一归类为完成，不以命中第一个目录为完成
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节（Research-flow 行：本轮以 `grep -rl "softmax\|attention" examples/` 显式列举 12+ 文件并逐一归类〔结构原型 / tuned 母本 / 机制先例 / 证据档案〕才闭环；仅按 AGENTS.md 关键词路由会漏 deepseek/mixcv/torch_tl_ops 侧先例）+ Transferable 首条
  - 前序根因（provenance）：examples/TileOPs/.../perf_opt/perf_feedback.md 修正附录（上版漏检两遍式先例 → 三轮调优 + [DESIGN_LIMIT] 才纠正）
- repro: 复现条件——任一迁移/重设计任务的结构先例检索（对照「已调研」断言 vs grep -rl 清单）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（§6 第 3 层扩展）
    定位锚: "3. 本仓 `examples/` 同类实现所用算法；"
    old 文本: |
      3. 本仓 `examples/` 同类实现所用算法；
    new 文本: |
      3. 本仓 `examples/` 同类实现所用算法——**完成标准 = 全目录清单**：结构先例检索以 `grep -rl "<族关键词>" examples/` 显式列举命中文件并逐一归类（结构原型 / tuned 母本 / 机制先例 / 证据档案），不以命中第一个目录（如关键词路由的 flash_attention/）为完成（2026-09-15 实证：上版漏检 deepseek/mixcv 侧两遍式先例致 [DESIGN_LIMIT] 才纠正，本轮清单法 12+ 文件闭环；检索动作可复现、清单可检视，优于「已调研」断言）；
    动机: 调研漏检的代价是整条结构选型错误（上版三轮调优 + DESIGN_LIMIT 才暴露）；清单化完成标准使检视可机械化对账。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0070
- type: R
- title: algorithm-research.md §3 R4「流水可融合性」补行为维度注记——flag wait 是否阻塞发射流属结构属性须引实测档案佐证；候选间流量近似（<10%）不构成结构裁决依据，裁决权移交行为维度
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节（Info-source 行 + Transferable 第 2 条：R3 流量口径下单遍链与两相位仅差 ~10%〔2.51 vs 2.78GB〕，实测性能差 2.31×，决定项是 R4 的发射流行为〔wait 与依赖对齐方式〕——R3/R4 分工「流量筛选 vs 行为定序」应显式提示）
- repro: 复现条件——任一候选间 fabric 流量差 <10% 的结构裁决（纸面流量对比无法区分的位形）
- toolchain_stamp: 流程规则；证据环境 2026-09-15（2.31× 锚点 = fa 域 perf_records 实测，tilelang 0.1.2+3a214cde 2026-09-09）
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（§3 R4 检查清单第 5 项扩展）
    定位锚: "  5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行；"
    old 文本: |
      5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行；
    new 文本: |
      5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行——flag wait 是否阻塞发射流属**结构属性**（非 tiling 可解），结论须引用 examples/ 实测档案（msprof pipe 利用率 / aic_scalar 占比）而非纸面推断；R3 类流量近似（候选间差 <10%）不构成结构裁决依据，此时裁决权移交本维度行为侧（2026-09-15 实证：单遍链 vs 两相位 fabric 流量差 ~10% 而实测差 2.31×，决定项是发射流行为）；
    动机: 上版误选单遍结构的根因之一即「流量相近时按流量拍板」——R3（流量筛选）与 R4（行为定序）分工显式化后，流量平局候选自动进入行为维度举证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0071
- type: R
- title: T.sync_block_set.md 补「同一 event_id 多次 set 的聚合语义」条款——2-AIV 同 id 双生产（各 set 一次、Cube 单次 wait）的放行条件（双 AIV 均完成 vs 任一完成）无文档规定，设计只能以先例背书
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 2（2026-09-15）Skill Flow Issues 首行（T.sync_block_set.md §1 仅「同一 block 中的其他执行单元」一句、无多生产者语义条款；reviewer 只能以双生产先例〔E1-E7 29/29 + v11nt fa-tuned〕背书；设计侧列为 R-1 风险项 + 拆双 id 回退）
  - docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1（现行文本无该条款）
  - 〔2026-09-17 生产先例追加（不同任务不同族），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕mamba 族 MixCV persistent（AIV subid 蛇形分片后**双 AIV 仍执行相同 set/wait 序列**、同 flag id）全量门禁 + Stage 4 六轮 + TileOPs 集成 11/11 bench 通过——「one-set-multi-wait」形态第二个跨族生产实证（GQA dual-producer 之外）
- repro: 只读核对（无运行时依赖）——docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1 无聚合语义条款
- toolchain_stamp: 仓库 docs 现状 2026-09-15；与 tilelang 版本无关（文档条款缺失）
- target_doc: docs/Tilelang.language/同步管道操作/T.sync_block_set.md
- delta: |
    动作: update（§1 简介段后追加语义说明段）
    定位锚: "简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。"
    old 文本: |
      简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。
    new 文本: |
      简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。

      **同一 event_id 的多次 set 语义**：同一 block 内两个执行单元（如 2-AIV 半分类）对同一 event_id 各 set 一次、消费侧单次 wait 的聚合/覆盖语义（双生产者均完成才放行，还是任一完成即放行）本文档未规定——依赖该形态的设计应以仓内可运行先例佐证（examples/flash_attention/flash_attn_npuir.py 两相位、examples/multi_head_attention/ 双生产形态），并在设计文档中单列风险项与拆双 id 回退方案。
    动机: 2-AIV 半分类是 attention 族 Expert 设计的常见形态，聚合语义无文档时 reviewer 无法独立核验 flag 协议安全性（只能先例背书）——条款化（哪怕先标注「未规定 + 推荐单生产者」）使该风险项可在 docs 层闭环。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0072
- type: R
- title: T.gemm.md §1 签名模式标注与 Expert 生产用法口径不一致——L1/L0C 形态在 Scope("Cube") 内生产使用 T.gemm 名称（flash_attn_npuir / fp8_lighting_indexer 先例），与 [Expert mode] T.npuir_dot 标注的关系未澄清（两任务连续检视提出）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 2（2026-09-15）Skill Flow Issues 第 2 行（T.gemm.md L8-9 两行签名标注 vs examples/flash_attention/flash_attn_npuir.py L102-109 Scope("Cube") 内 `T.gemm(l1, l1, l0c, initC, b_transpose, size)` 生产使用——reviewer 依先例判定可行但文档层无法自洽闭环）
  - 前序独立证据：同文件 Sep-07 Stage 2 round 2 Skill Flow Issues 第 3 行（examples/deepseek_v32/fp8_lighting_indexer.py L72-79 同款形态落差——两轮检视连续提出）
- repro: 只读核对（无运行时依赖）——T.gemm.md §1 两行签名标注与仓内 Expert 生产先例的形态落差（先例路径见 evidence）
- toolchain_stamp: 仓库 docs 与 examples 现状 2026-09-15；与工具链版本无关（文档-先例落差）
- target_doc: docs/Tilelang.language/线性代数操作/T.gemm.md
- delta: |
    动作: update（§1 签名区追加关系说明行）
    定位锚: "T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]"
    old 文本: |
      T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Developer mode]
      T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]
    new 文本: |
      T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Developer mode]
      T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]

      > 注：Expert 模式 `Scope("Cube")` 内以 L1/L0C 张量为操作数的形态（如 `T.gemm(l1, l1, l0c, initC, b_transpose, size)`）在仓内先例（examples/flash_attention/flash_attn_npuir.py、examples/deepseek_v32/fp8_lighting_indexer.py）中以 `T.gemm` 名称生产使用；上两行签名标注按 fragment（Developer）形态理解，L1/L0C 形态下 `T.gemm` 与 `T.npuir_dot` 的别名/等价关系以先例为准。
    动机: 同一形态落差在两轮独立检视中被连续提出（reviewer 每次都须以先例对抗文档标注）——一句关系说明消除文档-先例张力，使维度 1 API 存在性核对可从 docs 层闭环。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0073
- type: R
- title: develop SKILL.md Phase 2 补「母本移植逐项甄别」检查项——结构母本只提供 API 形态与程序序，布局假设/整除域特化/非 persistent 假设须按目标任务重推导；移植后首个回归须含 S_q > bm 的多头用例
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 4 行 + Transferable 第 2 条（v11_2phase _builder_2phase 为 fa 域〔B=1/H=1/整除/非 persistent〕特化实验形态：全局行 ws 布局/无 mask 链/无 TASKDONE/无钳位在 persistent+多头+causal+尾块域全部需适配——ws 槽全局行基误用即直接诱因之一：bx=0 任务「全局=本地」巧合掩盖、bx≥1 全错〔lse 恒 log2(S_kv) 指纹〕，row0 索引修复后 29/29）
- repro: 复现条件——任一从实验文件/先例 kernel 移植结构的开发任务（对照母本假设清单 vs 目标任务域）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 清单追加第 4 项）
    定位锚: "3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。"
    old 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
    new 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
      4. **母本移植逐项甄别**：从实验文件/先例 kernel 移植结构时，「结构母本」只提供 API 形态与程序序——布局假设（全局行 vs per-core 本地行）、整除域特化（尾块/掩码/TASKDONE 缺失）、非 persistent 假设须按目标任务域重推导并显式列差异清单（2026-09-15 实证：v11_2phase 全局行 ws 布局直接移植致 per-core 槽 row0 索引 bug——bx=0 任务巧合掩盖、bx≥1 全错，lse 恒 log2(S_kv) 指纹锁定；移植后首个回归须含 S_q > bm 的多头用例）。
    动机: 母本的隐藏域假设（整除、全局行基、无尾块）只在目标域位形暴露——差异清单 + 定向回归用例把该类 bug 从「bx≥1 用例碰运气发现」变为「首个回归必现」。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0074
- type: R
- title: integrator 已知修复目录「config 参数（block/tile）不适配 NPU」行补 config-契约集成范式——kernel 带 config 替换语义时 wrapper default_config 是出厂契约决策：惰性旋钮路由到门禁验证过的逐字路径（1 行胶水修复恢复整个域可编译）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §调试历史 attempt 1（分类命中该已知修复行 → 根因链 5 项 → 最小复现双路径 A fail / B ok → 修复 = wrapper default_config num_stages 1→2：(64,64,2) 在 E6 替换集外 → 逐字路径 bn_eff∈{64,144} = Stage 3 L1 门禁验证过的 traced 变体；num_stages 为结构惰性旋钮〔单槽两相位 stage 不变，DESIGN R-7〕语义零影响 → smoke 4/4 + full 6/6 + bench 10/10 复核）
  - .opencode/agents/tilelang-op-integrator.md L102（现行已知修复行——未展开带替换语义 kernel 的契约层选法）
- repro: 复现条件——任一 kernel 工厂带 config 替换/分派语义的集成期编译失败（wrapper 还原 (64,64,1) 时 `pytest "benchmarks/ops/bench_multi_head_attention.py::test_mha_fwd_bench[llama-3.1-8b-short-float16]" -v --tb=short` 复现 UB 溢出）
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（已知修复目录表「config 参数」行扩展）
    定位锚: "| config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件） |"
    old 文本: |
      | config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件） |
    new 文本: |
      | config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件）。**config-契约范式（kernel 带 config 替换语义时）**：wrapper `default_config` 是出厂契约决策而非 GPU 遗产直搬——替换集外的**逐字路径 + 惰性旋钮**（如 num_stages，结构 stage 不变的设计保留项）是集成层对齐「Stage 3 门禁验证过的 traced 变体」的合法工具：改惰性旋钮选中逐字路径即路由回门禁变体集（2026-09-15 实证：num_stages 1→2 使 (64,64,2) 落在 E6 替换集外 → bn_eff∈{64,144} 逐字路径，1 行胶水修复恢复 manifest 域可编译，pytest/bench 全绿复核）；修复后以工厂解析检查（`w.{func}.__module__`）确认激活源 |
    动机: 该修复形态已验证（1 attempt 闭环）但检索无门——现行行只说「调整 default_config」，未展开替换语义 kernel 的契约层选法（何时改旋钮、为何逐字路径合法）；范式入目录后同类集成失败可一次定位。Stage 3 门禁对 wrapper-default 派发路径的覆盖缺口另立 VP-2026-0065（防患侧），本条为修复侧。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0075
- type: R
- title: optimize SKILL.md 证伪协议第 4 条「标量占比触发换轴」前置编译警告排查——aiv_scalar 高占比与 flag 自旋同形，`will execute by scalar instruction` 警告是两者的区分器（vcmp int16 标量化占壁钟 45% 实证）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）：examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/opt_log.md#Phase-1（baseline aiv_scalar 94.5–96.8% 全 24 核均匀、两引擎 >90% 时间等待、aic_cube 0.7–1.4%——非引擎吞吐非带宽瓶颈）+ #判别实验（TILELANG_DUMP_IR + stderr 定位 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low efficiency`，仅 causal trace 有、每变体 2 处）+ #Round-2（根因定案：vcmp int16 全形态标量化 236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%；修复后 14.2x）
  - pattern-library/traps-compiler.md TRAP-expert-v-operands 第五轮补充段（「与 flag 自旋同形，须查编译警告区分」）+ attention.md PL-1.11（数据面档案）
  - 误诊路径对照：bottleneck-patterns BP_cross_engine_serial_chain（VP-2026-0034）触发信号即「busy 核 scalar 占比 30%+ 且归因为 flag 等待自旋」——两种根因（标量化陷阱 / flag 自旋）在 aiv_scalar 指标上同形，归因步骤缺区分器是误诊源头
- repro: `TILELANG_DUMP_IR=TRUE`（配新 `TILELANG_CACHE_DIR`，见 VP-2026-0062——缓存命中跳过编译也跳过 IR 打印）重编译目标 kernel，stderr grep `will execute by scalar instruction`；命中则热点段含被降级的 v-op（对照 msprof aiv_scalar 占比与「链宽 × 实测单链耗时」估算贡献占比）
- toolchain_stamp: tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（证伪协议第 4 条原位扩展）
    定位锚: "4. **诊断信号强制触发换轴分析**：msprof 显示热点段标量执行占比 > 50% 时，强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **诊断信号强制触发换轴分析（先排查标量化，再评估换轴）**：msprof 显示热点段标量执行占比 > 50% 时，第一步先用 `TILELANG_DUMP_IR=TRUE`（配新 `TILELANG_CACHE_DIR`）重编译并 grep stderr 的 `will execute by scalar instruction` 警告——命中说明热点 v-op（如 vcmp int16）被降级为标量指令执行（traps-compiler.md TRAP-expert-v-operands：与 flag 等待自旋在 profile 上同形，第五轮实证占壁钟 ~45%），按对应 TRAP 条目绕法处理；未命中且归因为等待/自旋时再强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。
    动机: 第五轮 baseline aiv_scalar 94–97% 与良性 flag 自旋同形，若直接走「换轴」路径会漏掉 vcmp int16 全形态标量化这一占壁钟 45% 的根因（本轮靠 IR dump 警告 + 微探针才定案，修复即 14.2x）；编译警告 grep 是零成本区分器，前置可避免一类高代价误诊。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0076
- type: R
- title: iteration-diagnosis.md 注意事项补微探针输出分片规范——`T.Kernel` 多 AIV 并发写同一 GM 输出缓冲存在竞态，探针结论（「链坏」类）可能是探针自身 artifact 而非被测机制缺陷
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）：examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/probe_penalty_chain.py（P9 修正前后对照）+ opt_log.md#Round-2-实现要点（调试链：P9 探针因 T.Kernel(1) 双 AIV 写同一 GM 输出而竞态，产出误导性「链坏」结论，浪费一轮调试）
  - 同族先例（不同形态）：pattern-library/attention.md PL-1.9-blockwidth「探针教训」——缺 Cube→Vector flag 同步的数值崩坏是探针自身竞态而非 API 缺陷（跨引擎形态）；本条补多 AIV 写竞态形态，两形态同属「探针异常先怀疑探针自身」
- repro: 微探针 kernel 以 `T.Kernel` 启动 ≥2 AIV 且输出为单一 GM 张量时，对照按 subid 分片（`out[subid, ...]` 各写各区）或单 AIV 版本的输出稳定性（P9 实证：分片前结果不可复现地「坏」、分片后稳定）
- toolchain_stamp: tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「## 注意事项」列表末尾追加一条）
    定位锚: "- 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。
  - 微探针（判别实验 kernel）的输出缓冲按 AIV 分片（`out[subid, ...]` 各写各区）或显式单 AIV——`T.Kernel` 多 AIV 并发写同一 GM 输出区域存在竞态，会产出误导性的「机制坏」结论（2026-09-15 实证：双 AIV 竞态使向量链探针误报「链坏」，浪费一轮调试；同族形态见 pattern-library PL-1.9 探针教训——探针数值异常先怀疑探针自身同步/写冲突，再怀疑被测机制）。
    动机: 微探针是第五轮定案 vcmp 标量化的关键手段，但其自身的多 AIV 写竞态曾误导一轮调试——判别实验的可信性以探针正确性为前提，分片输出是零成本规范。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0077
- type: R
- title: optimize SKILL.md Phase 3 TileOPs 集成算子条目补「perf_opt 双镜像同步」规则——conductor 工作区与 tileops 集成目录为独立 inode，单侧修复时 pytest 门禁测旧拷贝（错误位级相同无法区分源与镜像）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：DESIGN.md §11.2 集成拓扑注意（「TileOPs wrapper import 的是 tileops/kernels/.../perf_opt/ 镜像拷贝（独立 inode），perf_opt 侧任何修改须双镜像同步，否则 pytest 门禁测的是旧拷贝」——conductor 收束补记）
  - 同任务 opt_log Round 10（r9 precision_fix：修复 conductor 工作区 perf_opt 后手动同步 TileOPs 集成镜像 `tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py`，wrapper 的 baseline/perf_opt 切换块不动；同步后 TileOPs pytest 6/6 passed）
  - 双 inode 实核（2026-09-16 蒸馏会话）：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` 与 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` 两份 81708B 拷贝并存（mtime 06:22 / 06:26）
- repro: 复现条件——conductor 工作区与 TileOPs 集成目录存在两份独立 perf_opt 拷贝的 optimize/precision_fix 任务，单侧修改工作区 kernel 后直接跑 TileOPs pytest（门禁测的是未同步的集成镜像，错误位级相同无法区分源与镜像）
- toolchain_stamp: tilelang 0.1.2+a13585dc + CANN 8.5.0 + Ascend910B2C / 2026-09-16；TileOPs pyproject 环境
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 3 条扩展）
    定位锚: "3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。**双镜像同步（conductor 工作区 + TileOPs 集成目录）**：任务工作区（`examples/{project}/{op}/perf_opt/`）与 TileOPs 集成目录（`tileops/.../perf_opt/`）是独立 inode 的两份拷贝——凡修改任务工作区 perf_opt kernel（含 precision_fix 会话），须同步拷贝到 TileOPs 集成镜像（wrapper 的 baseline/perf_opt 切换块不动）后再跑 TileOPs pytest/回归门禁；单侧修复时门禁测的是旧拷贝，且位级相同的错误使门禁无法区分源与镜像（2026-09-16 r9 precision_fix 实证：同步后 pytest 6/6）。
    动机: TileOPs wrapper import 的是集成目录镜像而非任务工作区文件；r9 precision_fix 若只修工作区，pytest 门禁会在旧拷贝上复现位级相同的 UB 溢出——既浪费修复验证又可能误判修复无效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0078
- type: R
- title: statectl.py 状态迁移命令加并发互斥锁（flock）——双 conductor 会话并行驱动同一任务目录时 load→modify→write 交错，产出零产物 runtime fail 与交叉状态写入（4 次实证）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：.task_timeline.jsonl（6 次 fail 中 4 次 runtime 归因双会话竞争：03:38–06:15 期间另一窗口与本会话交错驱动——attempt 1「03:38 并行调度 #1 已 runtime 失败（819s 零产物）」记录于 .stage_state.json user_requirement；attempt 2 期间出现外来 set/snapshot 写入〔03:58:15〕；stage_retry_count 被推至 6 污染 E-2 指标）
  - .agents/tools/statectl.py 现行版本（Ctx.load→write 为无锁 read-modify-write；cmd_start 无活跃 attempt 互斥；atomic_write_json 只保证单次写原子性不保证序列化）
- repro: 复现条件——两个进程同时对同一 op 目录执行 statectl start/set/fail（本任务 2026-09-16 03:38–06:15 实录：interleaved set/snapshot + 819s 零产物 runtime fail ×1 + 竞争性 fail ×3）
- toolchain_stamp: statectl.py 现行版本；发生环境 2026-09-16（与 tilelang 工具链无关）
- target_doc: .agents/tools/statectl.py
- delta: |
    动作: update（main() 分发加互斥锁 + 新增 acquire_op_lock 辅助函数）
    定位锚: |
      def main(argv=None) -> int:
          parser = build_parser()
          args = parser.parse_args(argv)
          if getattr(args, "func", None) is None:
              parser.error("missing command")
          return args.func(args)
    old 文本: （即上述定位锚原文）
    new 文本: |
      MUTATING_ACTIONS = {"init", "start", "fail", "complete", "set", "snapshot", "repair"}


      def acquire_op_lock(op_dir: str):
          """Exclusive lock for state-mutating commands.

          Two conductor sessions driving the same op dir interleave
          load->modify->write cycles: silent state corruption, phantom
          runtime fails and cross-written snapshots (2026-09-16 gqa
          task: 4 competing runtime fails, first one 819s zero-artifact).
          flock converts that into a loud immediate E-BUSY.
          """
          import fcntl

          lock_path = os.path.join(op_dir, ".stage_state.json.lock")
          fh = open(lock_path, "w", encoding="utf-8")
          try:
              fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
          except OSError:
              try:
                  holder = open(lock_path, encoding="utf-8").read().strip() or "?"
              except OSError:
                  holder = "?"
              fh.close()
              emit(
                  {
                      "action": "busy",
                      "errors": [
                          err(
                              "E-BUSY",
                              f"另一 statectl 会话持有 {lock_path}（holder pid {holder}）"
                              "——同一任务目录禁止并行驱动；确认无活跃会话后可删除锁文件重试",
                          )
                      ],
                  },
                  1,
              )
              return None
          fh.write(f"{os.getpid()}\n")
          fh.flush()
          return fh


      def main(argv=None) -> int:
          parser = build_parser()
          args = parser.parse_args(argv)
          if getattr(args, "func", None) is None:
              parser.error("missing command")
          lock_fh = None
          if args.command in MUTATING_ACTIONS:
              lock_fh = acquire_op_lock(os.path.abspath(args.dir or os.getcwd()))
              if lock_fh is None:
                  return 1
          try:
              return args.func(args)
          finally:
              if lock_fh is not None:
                  lock_fh.close()  # releases flock
      （注：只读命令 show/verify/gate/timeline-summary 不加锁；migration 子树若同构需要可另行评估，本提案只覆盖主状态命令）
    动机: 并行双会话的每一次交错都产生真实成本（819s 零产物 attempt + 交叉写入使时间线不可信 + stage_retry_count 污染 E-2 北极星指标）；flock 是 OS 级机械守卫，比「conductor 自觉不并行」可靠——同任务目录的第二驱动者立即收到 E-BUSY 而非静默竞争。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0079
- type: R
- title: statectl.py cmd_fail 补「未闭合 start 前置校验」——重复 fail 记账（无中间 start 的第二个 fail）虚增 stage_retry_count，污染重试指标与 E-2 抽取
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：.task_timeline.jsonl L14–15（06:15:32 fail 971s / 06:15:39 fail 978s——第二个 fail 前无任何 start，duration 均回溯到同一 start#5，stage_retry_count 5→6 虚增；conductor 归因记账侧重复调用）
  - .agents/tools/statectl.py 现行版本（cmd_fail 非 BLOCKED 路径无「本 stage 存在未闭合 attempt」校验；stage_attempt_duration 只回溯最近 start，第二个 fail 复用同一 start 的 duration）
- repro: 复现条件——对同一 stage 连续调用两次 statectl fail（无中间 start）：第二条 fail 照常追加时间线并递增 stage_retry_count（本任务 2026-09-16 06:15 实录，7 秒间隔双记账）
- toolchain_stamp: statectl.py 现行版本；发生环境 2026-09-16
- target_doc: .agents/tools/statectl.py
- delta: |
    动作: update（cmd_fail 非 BLOCKED 路径头部插入校验块）
    定位锚: |
      （cmd_fail 函数体内，E-TERMINAL 守卫块之后、design_revision 分支之前）
          if state["phase"] in TERMINAL_PHASES:
              errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)
    old 文本: （即上述定位锚原文）
    new 文本: |
          if state["phase"] in TERMINAL_PHASES:
              errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)

          # A fail must close an OPEN attempt: the most recent start/fail/
          # complete event for this stage has to be a start. Duplicate fail
          # invocations (2026-09-16: two fails 7s apart, no intervening
          # start) otherwise inflate stage_retry_count and pollute the
          # E-2 retry metrics.
          last_stage_action = None
          for ev in read_timeline_events(ctx.op_dir):
              if ev.get("stage") == n and ev.get("action") in (
                  "start", "fail", "complete"
              ):
                  last_stage_action = ev.get("action")
          if last_stage_action != "start":
              errors.append(
                  err(
                      "E-NO-OPEN-ATTEMPT",
                      f"Stage {n} 无未闭合的 start（最近事件为 {last_stage_action}）"
                      "——重复 fail 记账拒绝，不递增 stage_retry_count",
                  )
              )
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)
    动机: fail 是「闭合一个 attempt」的语义事件；无 start 的第二个 fail 只能来自记账侧重复调用——照单全收会虚增重试计数（本任务 6→实为 5），并使 duration 回溯到已消费的 start 产出误导性 duration_s。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0082
- type: R
- title: optimize SKILL.md T-4 实验批处理 runner 补适用边界——结构重构类分支（指令流重排/流水深度/同步协议）默认手工 diff 串行，生成脚本化仅用于参数扫描类
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：opt_log Skill Retrospective（第六轮）第 6 条（T-4 runner 未搭建的决策依据：分支数 8、依赖交互调试多，手工串行可控性更高；r8g 结构分支的生成脚本引入 3 次脚本 bug，叠加 2 次编译错 + 1 次精度失败 + 1 次 flaky 超时，消耗 ~40% 轮次时间）
  - 〔2026-09-21 第二任务证据（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z〕opt_log Skill Retrospective Skill Flow Issues 第 4 行：str.replace 型代码生成分支时同构 body（even/odd/tail）文本相同导致多点命中——round3 cband 生成脚本 + round7 xmid 两度返工共 3 次工具事故均为此类（单点实验纪律本身无缺口）；作者建议的缓解工具（run_experiments 模板提供 branch-from-base 结构化 diff 生成）作为本提案 apply 时的可选配套一并裁决
- repro: 复现条件——以脚本生成结构重构类分支（指令流重排/多槽 buffer/flag 协议变更）且需多轮交互调试的调优轮（本任务 r8g 实证形态）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 tilelang 0.1.2+a13585dc / 2026-09-16
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（T-4 段末尾追加一句）
    定位锚: "脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。**适用边界**：runner 适用于参数扫描/配置类分支（同 kernel 改 config）；**结构重构类分支（指令流重排 / 流水深度 / 同步协议 / 多槽 buffer 变更）默认手工 diff 串行**——此类分支依赖完整 kernel 上下文且需交互调试，脚本生成的变体一旦携带 bug，其调试成本高于手工串行（2026-09-16 attention 第六轮 r8g 实证：生成脚本引入 3 次脚本 bug，叠加编译/精度/flaky 共消耗 ~40% 轮次时间）。
    动机: T-4 的收益模型（交互轮次下降 60%+）建立在「分支可批量执行」的前提上；结构分支不满足该前提，脚本化反而放大失败面——边界标注使 runner 用在参数类刀刃上。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0089
- type: R
- title: pattern-library 条目时效性判定口径——以条目自身 `status:` + front-matter 重验行为准，kb_stale_check 全局 stale_count 可因工具链戳批量失配虚高（stale_count=83 误伤 verified 条目致掩码主备倒置）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 2 Skill Flow Issues 第 3 行 + Value Point Proposals——设计把 PL-1.11（front-matter `status: verified` + L117 载 2026-09-16 a13585dc 重验维持行）记作「stale 线索」并据此把实测已根治的病态形态（int16 vcmp 标量化占壁钟 45%）选为主选，Stage 2 独立复核才发现；根因 = kb_stale_check 全局 stale_count=83（工具链 0.1.2+1990aa9fe4 与条目戳批量失配）被当作单条目判定依据
  - Stage 1 first_design 同源问题：全局 stale 使设计期 roofline 只能按"估算口径"引用关键常数（CONST-capacity / CONST-vector-launch-overhead / CONST-mte2），削弱 D-2 定量性
  - pattern-library/attention.md#L110-L131（PL-1.11 status: verified + 重验行）vs DESIGN.md §1.6.3 候选 #3「stale 线索」定性
  - 〔2026-09-20 正确消费实例，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕重跑任务设计以条目自身 status + 重验行消费 PL-1.11（DESIGN §1.6.1 OPT-B 主选依据引「PL-1.11 verified，2026-09-16 a13585dc 重验维持」），未被全局 stale_count 误导；其流程化产物（stale 条目两分法消费指引）另立 VP-2026-0101（判后如何消费 vs 本条如何判 stale，两案互补互链）
  - 〔2026-09-21 第三现（规模峰值），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 重建会话〕stale_count=125 几乎全库假阳性——根因为 repo HEAD（96f287e，evolution 文档 commit）被当作 tilelang 工具链戳：`git diff --stat 15ad002..96f287e` 仅 7 个 .agents/ 文件、tilelang 源码零改动；任务内以 docs-only delta 论证 + probe 锚点复现（217.58 vs 217.65，−0.03%）自行消解，未造成错误决策，但 Phase 0 被迫逐条 git diff 自证。**工具侧根因修复提案见 VP-2026-0116**（stamp diff 过滤非源码路径——本条与其构成消费侧/工具侧互补，三案〔含 0101〕建议同批审批）
- repro: python3 .agents/tools/kb_stale_check.py（观察全局 stale_count 与单条 status 的口径差）；grep -n -A3 "^id: PL-1.11" .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md（读 status 与重验行）
- toolchain_stamp: kb_stale_check.py 现行版本；发生环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 第 5 条「弃选论证查证」末尾追加一句）
    定位锚: "代价类论断未引用 pattern-library 已有实测数据的，要求补证（该量已有实测则不再是"未实证假设"）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      代价类论断未引用 pattern-library 已有实测数据的，要求补证（该量已有实测则不再是"未实证假设"）。**条目时效性判定口径**：pattern-library 条目是否 stale 以**条目自身 front-matter `status:` 字段 + 重验行**为准；`kb_stale_check` 的全局 stale_count 可能因工具链版本戳批量失配而虚高（2026-09-17 ssd 实证：stale_count=83 误导设计把 `status: verified` 且已跨工具链重验维持的 PL-1.11 降级为「stale 线索」，致掩码主备倒置、实测已根治形态被重选为主选）——不得据全局计数把 verified 条目降级，工具链戳失配只触发「待重验」标注而非结论失效。
    动机: 同一口径建议同步写入 tilelang-op-design 的弃选论证证据规则（Phase 2/Phase 3 引用 pattern-library 处）——本提案主锚 design-review，op-design 侧同步属同语义扩散，审批时可一并裁决。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0090
- type: R
- title: 机器验证脚本的「模型形态」须与设计规格 API 级操作数同形核对——EQUIV_PASS 只覆盖数学、不覆盖 API 形态偏差（[1,bs] 脚本正确 + [bs,1] 规格错误仍全绿放行）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 2 Skill Flow Issues 首行 + Value Point Proposals——verify_equiv.py 重跑全 EQUIV_PASS 且与 §1.6.1 内嵌表逐项相符（skill Phase 1 第 7 步判定条件全满足），但脚本用 unsqueeze(0)（[1,bs] 列因子）建模、DESIGN.md §3.2/§3.3 却写 [bs,1]——数学被验证正确、API 形态错误，「假安心」直到 Stage 2 形态对读才发现
- repro: 复现条件——verify_equiv 类脚本与 DESIGN §3.2/§3.3 的形态对读（任务内对读命令见 evidence provenance 项）
  - 〔provenance，允许失效〕任务内对读命令：grep -n "unsqueeze" examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/verify_equiv.py 与 grep -n "0:1\]" examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/DESIGN.md 对读
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「重跑等价性验证」bullet 扩展）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）。**模型形态同形核对**：抽查验证脚本中各因子/操作数的 shape 形态（unsqueeze 方向、[n,1] vs [1,n]）与 §3.2/§3.3 的 API 级操作数是否同形——形态不一致时机器验证不覆盖该偏差（2026-09-17 ssd 实证：脚本 [1,bs] 正确 + 规格写 [bs,1]，EQUIV_PASS 全绿仍放行至 Stage 2），须在维度 8 单列；
    动机: 配套（同语义扩散，审批时可一并裁决）：tilelang-op-design SKILL.md Phase 2 机器验证产出规范补一句「verify_equiv 脚本注释须标注每个因子的 shape 形态（如 `# [1,bs]`），供检视侧与 §3.2/§3.3 交叉核对」——形态标注是同形核对的可机械化前提。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0091
- type: R
- title: 编程模式选型硬约束——persistent 分核（核内 T.serial 多任务 + gemm）⟹ Expert 模式；design-review 维度 4 增「模式与分核策略一致性」必查行
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 3 Skill Flow Issues 首行——DESIGN §2.1/§0.7 声明 Developer 模式（user_requirement 亦指定）与自身 §0.6 R1 persistent 24 核分核策略不兼容，Stage 2 检视未拦截；Developer 首实现运行时崩溃（"unaligned UUB addresses"），Expert 同结构正常（repro/TRAP-DEVMODE-PERSIST-GEMM.py 双模式对照 + CG-2026-0010）
  - 判定依据：全仓 24 处 is_npu=True 中 Developer 仅用于非 persistent 网格（flash_attn_npuir_dev.py、fp8_lighting_indexer.py），persistent 混合算子（GQA、sparse_mla_fwd_exp.py）全部 Expert
  - 〔2026-09-20 知识生效先例，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕重跑任务设计期直接按 Expert 双 Scope 展开（算子结构落在 CG-2026-0010 崩溃域即判定），未再为 Developer 形态设计 fallback——省一版无效设计（该任务 Stage 1 Transferable Lessons）；本条 pending 期间的注入生效证据（kb_search 预注入命中 CG-2026-0010/PL-1.16）
- repro: python3 .agents/skills/tilelang-op-optimize/references/pattern-library/repro/TRAP-DEVMODE-PERSIST-GEMM.py（Expert 数值断言 + Developer 崩溃双模式对照）
- toolchain_stamp: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表第 2 项扩展）
    定位锚: "2. 编程模式选型"
    old 文本: |
      2. 编程模式选型
    new 文本: |
      2. 编程模式选型（**persistent 硬约束**：分核策略含核内 `T.serial` 多任务 + `T.gemm`（persistent 混合算子）⟹ **Expert 模式**（显式 `T.Scope("Cube")`/`("Vector")` + alloc_L1/L0C/alloc_ub + sync_block_set/wait）——Developer 模式对该结构类运行时崩溃（"unaligned UUB addresses"，pattern-library TRAP-DEVMODE-PERSIST-GEMM / CG-2026-0010，2026-09-17 ssd 实证）；Developer 仅适用于非 persistent 简单网格；user_requirement 指定 Developer 而结构落入该类时，在设计文档记录实测仲裁依据后切换并披露）
    动机: 模式选型错误不在 Stage 2 拦截（检视只看文档不编译），以 Stage 3 运行时崩溃形式暴露成本最高（本任务白耗一次 attempt + 模式重写）。配套（同语义扩散）：tilelang-design-review SKILL.md 维度 4 表追加一行 `| 模式-分核一致性 | §2.1 编程模式与 §0.6/§5 分核策略一致（persistent + gemm ⟹ Expert，TRAP-DEVMODE-PERSIST-GEMM） |`（锚：维度 4 表末行 "| L0C 溢出 | 已在内存层级维度处理 |" 之后）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0092
- type: R
- title: 修订复检两机械步骤——v0 错误值/表述提炼成 grep 关键词全文扫描（区分历史说明 vs 未同步残留）+ 验证脚本 md5 比对（防「改脚本过关」）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md re_review 章节——v1 changelog 自称清理 6 处残留，复检独立发现 **7 处未同步**（§5.3/§9.1/§9.2 旧 UB 数字 115/129/196KB 与 v1 的 99/111/168KB 直接矛盾、§6.4/§3.5.1 的 `tk ≥ 32` 旧钳位、3 处 verified 条目仍标 stale）；grep 关键词组：`115KB\|129KB\|196KB\|贴限`、`tk ≥ 32\|K ≥ 32`、`stale 线索\|stale 条目`；另复检额外做了 verify_equiv.py md5 比对（dd0f0a8f… 与首轮一致）+ 形态同形核对确认分叉已闭合——均为 skill 未要求的自发动作
  - 交叉引用：queue VP-2026-0028（修订版重审工作流——diff 锁定深查范围；本条补其未覆盖的两个机械步骤）、VP-2026-0090（形态同形核对）
- repro: 复现条件——任一 Stage 2 不通过后修订版的复检（本任务 7 处残留实证）
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「重跑等价性验证」bullet 后追加同层 bullet）
    定位锚: "   - **算术复算**：`python3 .agents/tools/design_calc_check.py --design <path>`——逐项消费其 JSON 输出："
    old 文本: （即上述定位锚原文所在 bullet 起始行）
    new 文本: |
       - **算术复算**：`python3 .agents/tools/design_calc_check.py --design <path>`——逐项消费其 JSON 输出：
       - **修订复检（revision re-review）两机械步骤**（2026-09-17 ssd 实证：changelog 自称清理 6 处、复检 grep 仍抓出 7 处未同步，其中 2 处与修订版主体直接矛盾）：① 对每个已修复的阻塞/建议项，把 v0 的错误值与错误表述提炼成 grep 关键词在修订版全文扫描，命中处逐一判定「历史说明（changelog/风险条目/源码描述节，合法）」vs「未同步残留（现行规格节 §3–§6/§9 结论句，记建议级）」——不得以 changelog 修复清单作覆盖面依据；② 复检时记录并比对机器验证脚本（verify_equiv.py 等）与首轮的 md5——脚本未变而结论仍 PASS 才是有效证据，脚本有变更须判定变更性质（新增 case 合法、放宽阈值或改变建模形态需重新独立推演）。
    动机: 修订残留与「改脚本过关」是复检的两个结构性盲区——摘要/风险/循环章节最易漏改（本任务 7 处分布实证），md5 是零成本防篡改核。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0093
- type: R
- title: Stage 5 集成两防线——① wrapper `default_config` 与 kernel 内嵌 tuned 默认的口径衔接（翻转切换块激活 perf_opt 时同步核对，二者取一或标注口径差异）② 集成前工厂签名 diff 核对（零成本防线）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：op 级 RETROSPECTIVE.md Stage 5——wrapper 脚手架 `default_config`（baseline 启发式硬编码 block_n=64/num_stages=3）在 perf_opt 激活后仍显式传参，覆盖调优版 kernel 内嵌 `TUNED_DEFAULT_CONFIG(block_n=128/num_stages=2)`（custom_op 显式传参优先级更高），Stage 4 tuned config 集成态不生效；integration_log.md「Bench 观察项 #2」（bench 输出 config 与 opt_log §final 不一致）+ 工厂签名核对段（`diff <(sed -n '/^def .../,/^):/p' ...)` 输出 SIGNATURE_IDENTICAL——签名逐参一致使 wrapper 胶水零改动完成切换）
  - 〔2026-09-28 第二证（不同任务，reduction 族，wrapper 侧 GPU 启发式亚型），task argmax-_argreduce_kernel-20260924T031005Z〕Stage 5 集成：wrapper `default_config` 沿用 GPU 时代 `SHARED_MEMORY_BUDGET_BYTES=49152` smem 启发式选 bm=4，kernel 探针校准阶梯 `_select_config` 给 bm=2（含 min(block_m,M) 钳制与窄 N 预算减半，外部启发式均不掌握）——bm=4×4096 直接 `ub overflow, requires 2097408 bits while 1572864 bits available`；修复 = `default_config` 转调 `_select_config(self.M, self.N_padded, self.dtype_str)`，同一 report 42/42 通过（修复前后 run：20260928_024504 vs 20260928_030100）。诊断指纹：`ub overflow, requires N bits` 报文先查「谁传的 block_m、是否经 kernel 校准」。出处：examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/integration_log.md#attempt-1 + op 级 RETROSPECTIVE.md Stage 5 VP 行；工具侧联动见 VP-2026-0140（integrate_kernel.py 自动搬运 `_select_config` 成对 import——本次手工补行，翻转切换块时易漏翻）
  - 〔2026-10-14 第三现（不同任务，reduction 族同 kernel，跨工具链/跨设备），task argmin-_argreduce_kernel-20261014T151121Z〕Stage 5 首跑 report 失败 40/42（2 失败用例 100% 落 N_padded=4096：fp16 `ub overflow, requires 2359808 bits while 1572864 bits available` / bf16 `2361856 bits`）+ lm-head `[4,102400]` 硬 `ValueError`（single row exceeds 49152-byte budget，`49152 // 204800 == 0`）——同一根因（wrapper 继承 GPU 48 KiB SMEM 启发式 override 与 kernel `_select_config` 冲突）在**与首证同源的共享 kernel、不同工具链（CANN 9.0.0 / 910B3）**上逐字节复现；修复 = `argmin.py#default_config` 转调 `from .argmin_kernel._argreduce_kernel import _select_config` 取 `["block_m"]`，kernel/tests/manifest/op 零改动，修复后 42/42 + bench 9/9（run `20261014_160246_645105_ArgminFwdOp`）。**新增两条落地细节**：① 集成侧 wrapper 的 `default_config` 委托核对可机制化为「装配后工厂 config 来源检查」——本任务人工定位≈0 成本（指纹直接命中），但**集成脚本未做该核对**（本任务失败由 bench 期暴露）；② 聚合 `argmin_kernel/__init__.py` 只 re-export kernel factory，wrapper 委托须 import 私有模块路径 `.argmin_kernel._argreduce_kernel`（暴露面缺口见 VP-2026-0140 证据链追加）。出处：examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/integration_log.md#attempt-1 + examples/argmin/RETROSPECTIVE.md Stage 5 Skill Flow Issues 首行。
- repro: 复现条件——任一 perf_opt 激活态集成（wrapper default_config 显式传参 vs kernel TUNED_DEFAULT_CONFIG 内嵌默认的优先级差）
- toolchain_stamp: tilelang npuir dev build（1990aa9fe4 谱系）+ CANN 8.5.0 / 2026-09-17
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（第一步 bullet 列表末尾追加两条）
    定位锚: "- 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。"
    old 文本: （即上述定位锚原文）
    new 文本: |
   - 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。
   - **工厂签名 diff 核对（集成前零成本防线）**：对 baseline 与 perf_opt 的同函数工厂签名段做 sed+diff 比对（`diff <(sed -n '/^def {func}/,/^):/p' <baseline> <perf_opt>)`）——签名逐参一致时 wrapper 胶水零改动即可切换，不一致时先对齐签名再翻转（2026-09-17 ssd 实证：SIGNATURE_IDENTICAL）。
   - **config 口径衔接（翻转切换块激活 perf_opt 时）**：核对 wrapper `default_config` 与调优版 kernel 内嵌 tuned 默认（TUNED_DEFAULT_CONFIG 类模块级常量）——custom_op 显式传参优先级高于内嵌默认，wrapper 旧 config 会静默覆盖 tuned config 使 Stage 4 调优集成态不生效；二者取一（改 wrapper 引用 tuned 常量，或保持 wrapper 默认并在 integration_log 标注口径差异与不可同口径对比的 bench 边界，2026-09-17 ssd 形态）。
    动机: Stage 4 调优结论与 Stage 5 bench 数值的可比性依赖 config 口径一致；本任务 bench 11 dispatch 全部跑在 wrapper 默认 config 上（与 tuned 不同口径），不核对则调优收益在集成态静默丢失且无告警。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0094
- type: R
- title: 关键决策引用须抄录一手来源限定条款原文（形态/数字/状态三类均适用）+ §3/§4/§6 三方对读细化为可 grep 的机械对读点
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z：函数级 RETROSPECTIVE.md Stage 1 revision 章节——v0 三个阻塞错误共同根因 = 引用证据未打开一手来源核对：① 列因子形态凭直觉写 [bs,1]（未对照 T.vmul.md §2.2.2 四形清单）；② intra FLOPs 手算 6.46G 未与 manifest 权威公式 9.7G→6.44G 交叉核对（2× 高估直送审）；③ PL-1.11 只看 kb_stale_check 全局计数即降级（未读条目自身 status + 重验行）——形态/数字/状态三类各踩一个；另 v0 §3.3 伪代码与 §6.1 循环表、§4.5 预算表与 §3.3 buffer 使用两处自相矛盾（成文未做三方对读终检，质量清单 #21 执行流于形式）
- repro: 复现条件——任一引用 docs/pattern-library/manifest 三类信息做关键决策的设计期（本任务 3 阻塞 + 2 内部矛盾实证）
- toolchain_stamp: 设计层规则（无运行时依赖）；证据环境 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 3 步骤 0.5 段落之后追加一条机械规则）
    定位锚: "> 迁移任务的额外信息源：Phase M0/R/M1 的源算子解读、算法调研结论与迁移决策（优先级介于 `examples/` 同类实现与外部参考实现之间——迁移决策界定"算法该怎么设计"，`examples/` 界定"API 怎么用"）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      > 迁移任务的额外信息源：Phase M0/R/M1 的源算子解读、算法调研结论与迁移决策（优先级介于 `examples/` 同类实现与外部参考实现之间——迁移决策界定"算法该怎么设计"，`examples/` 界定"API 怎么用"）。

      > **引用抄录规则**：关键决策引用的每个 docs / pattern-library / manifest 来源，引用时须抄录其限定条款原文——三类信息都是被引用对象：**形态类**（shape 支持清单，如 T.vmul.md §2.2.2 四形）、**数字类**（实测倍数/复杂度，须与 manifest 权威公式双源交叉）、**状态类**（条目 front-matter status + 重验戳）。「负向断言须引用条款」已有，正向选型引用同样适用（2026-09-17 ssd 实证：v0 三个阻塞错误恰好各踩一类——形态凭直觉、数字单源手算 2× 高估、状态只看全局计数）。
    动机: 配套（同语义扩散，审批时可一并裁决）：Phase 2 产出规范「§1.6.3 布局决策与 §3.3/§4/§6 三方一致」细化为三个可 grep 的机械对读点——buffer 名集合（§3.3 vs §4.5 vs §6.1）、循环 API 形态（§3.3 vs §6.1）、因子 shape 形态（§3.2/§3.3 vs §1.6.3）——对读点可机械化后才不流于形式（本任务 v0 两处自相矛盾均属三方对读缺失）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17
- decided_by: -
- decided_note: -

## VP-2026-0101
- type: R
- title: 知识预注入 stale 条目消费两分法——结构性结论（有 HEAD 生产代码佐证）可继承为主选、硬边界结论（编译器/容量/支配性）必列 Stage 3 重验清单
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 first_design Skill Flow Issues 首行 + DESIGN.md §0.6 引言 / §9.2 R-1（两分法消费全记录：PL-1.16/1.12/1.13/1.14 结构性继承 + CG-2026-0010/0011/0012、UB 容量标待重验）
  - 有效性验证：3 轮检视均判定 stale 引用处置恰当；Stage 3 重验四项全过（pass_configs 双关闭 / UB 记账 / 旧 verified 结构首编即过 / golden）；Developer fallback 设计直接跳过（VP-2026-0091 知识生效先例）
  - 交叉引用：queue VP-2026-0089（stale 判定口径——如何判 stale；本条为判后如何消费，互补互链）
  - 〔2026-10-14 第五任务证据（不同任务，pool 族）+ 补充第三档「结构免疫登记」，task max_pool2d-_max_pool2d_kernel-20261014T062320Z〕Stage 1 Skill Flow Issues 首行：`kb_stale_check.py` 把大批 `TRAP-*` 条目标为 stale（工具链已推进到 `0d3950aa`），但 skill 未给「stale 陷阱是否仍需逐条重验」的可操作判据；本次按**设计对陷阱结构免疫**处置（不重跑 repro，在 `DESIGN.md` §9.4 第 3 条做**结构免疫登记**）。⇒ 两分法之外需第三档：**无依赖 trap 可只做结构免疫登记**（判据 = 逐条核对设计的每一实现步骤是否消费该陷阱条目；有依赖者仍走硬边界重验）。建议把该第三档并入本条 new 文本（结构性结论继承 / 硬边界列重验清单 / 无依赖 trap 结构免疫登记）
  - 〔2026-10-14 第六任务证据（不同任务，reduction 族；第三档的实例与反向补充），task argmin-_argreduce_kernel-20261014T151121Z Stage 5〕知识预注入把 `CASE-argmax-argreduce-integration` 标为 **stale**（工具链戳 910B2C/CANN 8.5.0 ≠ 本任务 910B3/CANN 9.0.0），但其缺陷 ② 在本任务新工具链上**逐字节复现**（同报文 `ub overflow, requires … bits`、同 wrapper GPU 启发式 override 根因、同修复形态），integrator 按「共享类相同」直接消费该案例并一次闭环（attempts=1）。⇒ 对**结构性/契约类**案例（CASExx 集成契约、跨引擎结构约定），stale 标记只看工具链戳差异不足以否证其适用性——第三档「结构免疫登记」应明确覆盖**案例索引类条目**（不只 trap 条目）：消费前判据 = 该案例的失效面是否依赖具体工具链（编译器行为/容量/支配性）还是依赖契约与接口形态（多数集成契约属后者）。出处：examples/argmin/RETROSPECTIVE.md Stage 5 Skill Flow Issues 末行 + examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/integration_log.md#attempt-1；本条与 VP-2026-0089（stale 判定口径）/ VP-2026-0116（stamp diff 过滤非源码路径）构成 stale 治理三案。
- repro: 复现条件——工具链升级后重跑已知先例的迁移/重做任务（本次 4515de8 vs 旧任务 1990aa9fe4：E-1 预注入条目全部标 stale）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（步骤 0.5 段落末尾追加）
    定位锚: "该库是任务实测的积累（如核内融合转置链实测代价、C 轴切片累加已验证形态、跨步系数阻碍向量化案例），其优先级高于 docs 规格与 examples 先例（见 info-sources.md 优先级表）；§1.6.3 的候选枚举与代价模型必须先对照该库，库内已有的实测数字不得当作"未实证常数"重新假设。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + "**stale 条目两分法消费**（工具链升级后重跑先例任务时；2026-09-20 ssd 重跑实证——3 轮检视收敛 + Stage 3 重验四项全过）：结构性结论（双 Scope/流水深度/分片/布局类，有 HEAD 生产代码或先例 kernel 佐证）可继承为主选结构；硬边界结论（编译器行为/容量上限/pass 支配性类，CG-2026-00xx 与 TRAP-*/CONST-* 条目）必须列入 Stage 3 首日重验清单——避免全盘照抄（旧结论失效风险）与全盘重验（丢先例红利）两个极端。条目是否 stale 以条目自身 front-matter status + 重验行为准（VP-2026-0089），全局 stale_count 不作单条目判定依据。"
    动机: 重跑任务的预注入条目在工具链升级后全部标 stale——无消费指引时要么盲信旧结论、要么整体降级重推；两分法使本次重跑零无效重设计（Developer fallback 直接跳过）且硬边界全部实测重验。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0102
- type: R
- title: 迁移重做任务检视前置一步——先 `git status --short -- examples/` 识别 Stage 0 脚手架删除的旧工件，先例引用一律经 `git show HEAD:<path>` 核对提交版本（仅查工作树会误判「引用不存在」或漏核关键先例）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 first review Process 行：Stage 0 脚手架删除旧任务集成工件（`ssd_chunk_scan_kernel/` 整目录 D 状态），而设计大量引用旧集成版/perf_opt 终版作生产先例——host-cast 先例、wait 置位 verified 形态均只存于 git；REVIEW.md 元信息「工件注入披露」+ 问题 1/7 的 git 证据链
- repro: 复现条件——任一迁移重做任务（Stage 0 脚手架重建清理旧任务集成产物）的 Stage 2 检视
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 步骤 2 扩展）
    定位锚: "2. **迁移任务**（`DESIGN.md` 含 §0）：Read `source_op_path` 指向的源算子代码全文；§0 与源码不一致的项在维度 0 中逐条记录证据（源码行/语句 + DESIGN.md 章节）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      2. **迁移任务**（`DESIGN.md` 含 §0）：Read `source_op_path` 指向的源算子代码全文；§0 与源码不一致的项在维度 0 中逐条记录证据（源码行/语句 + DESIGN.md 章节）。**迁移重做任务附加**：先 `git status --short -- examples/` 识别 Stage 0 脚手架删除的旧任务工件（整目录 D 状态常见）——设计引用的旧集成版/perf_opt 终版等先例只存于 git，一律以 `git show HEAD:<path>` 核对提交版本；仅查工作树会误判「引用不存在」（维度 7 假 fail）或漏核关键先例（2026-09-20 ssd 重跑任务：host-cast 先例与 wait 置位 verified 形态均仅存于 git）。
    动机: 重做任务的先例引用合法性无法在工作树核验——不识别删除态会把合法引用误判为断链，或让设计的关键先例逃过核对。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0103
- type: R
- title: design SKILL Phase 5 伪代码自检两条——① Expert 双域伪代码逐 flag 令牌推演（set 序×wait 序×写序位置配对，并与 verified 实现行级比对）② 设计自定 L0 边界用例的伪代码形状走查（非整除/P_tiles≥2/退化域逐一过 copy/gemm src/dst 形状与循环变量绑定）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 revision 两轮 Skill Flow Issues：v0 尾置 wait 在 §0.6 R7 论证文字正确的情况下被写出（论证与伪代码脱节——WAR 竞争逃逸 Phase 5 自检、Stage 2 令牌推演才检出，REVIEW.md 问题 1）；v0 x 装载（0:P 全宽、pp 循环外）与 Vector 域 pp 未绑定只在 P=64（当前 workload 全集）侥幸成立——P=128 是设计自定 L0-2 门禁用例却未过伪代码走查（REVIEW.md 问题 2/3）
- repro: 复现条件——任一 Expert 双域流水设计的 Phase 5 自检（本任务两类缺陷均逃逸自检、检视期拦截）
- toolchain_stamp: 设计层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 5 段末追加两条自检）
    定位锚: "按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。**所有任务必须通过算法级检查项（算法调研完整（§1.6.0 调研四问齐全、结论有依据）/ 数学等价优化分析完整 / 向量化替代分析完整）**；**迁移任务必须额外通过清单中的迁移专属检查项**（源算子解读完整性 / 耦合性判定有依据 / 重设计有语义保持论证 / §1–§7 与迁移决策一致）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + 两条新增自检：
      > **Expert 双域伪代码令牌推演**：对伪代码中每个 `sync_block_set/wait` 做令牌推演（同 id 的 set 序与 wait 序按事件计数逐对配对 + 受保护对象的写发生在 wait 之后的哪个位置），并与引用的 verified 实现行级比对——论证文字正确而伪代码尾置 wait 的脱节形态无法靠通读发现（2026-09-20 ssd 重跑任务 v0 实证：WAR 竞争逃逸 Phase 5、Stage 2 才检出）。
      > **伪代码形状走查**：对设计自定的 L0 边界用例（非整除 / P_tiles≥2 / 单 tile 退化域）逐一在伪代码上走一遍 copy/gemm 的 src/dst 形状与循环变量绑定——「当前 manifest workload 全整除/全 P=64」的侥幸域会掩盖形状矛盾（同任务 v0：P=128 是自定门禁用例却未走查，形状矛盾逃逸到检视期）。
    动机: 两类缺陷（协议脱节/形状矛盾）都属「通读自检必然通过、机械推演才暴露」形态；走查域取设计自定的 L0 用例使自检与测试计划同源。与 VP-2026-0029（检视侧维度 5 行）构成设计自检↔检视双防线。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0104
- type: R
- title: 修订「写实占位伪代码」对照纪律——写实依据 = 占位符对应 verified 实现的真实形态（逐行比对搬运方向/buffer 数）；写实引入新搬运原语前置三关检查（docs 方向表在列 + examples 先例 + 最小编译探针）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 1 revision_index=2 Skill Flow Issues：v1 修问题 5① 时把 band 组装从占位式（`T.gemm(l1_lcb_band...)`）写实为自创 `T.copy(l1_lcb, l1_band[...])` L1→L1 两跳——方向表无此方向、全仓零先例、检视探针实测硬失败（'Unsupported copy from cbuf to cbuf!'）；旧 verified 实为单 buffer GM→L1 列偏移直载；DESIGN.md v1→v2 §1.4 对比（L347–353 两跳 → 单跳）
  - 错误谱系：v0 协议/形状缺陷 → v1 写实自创原语 → v2 清零（「写实化」是修订链最后一个高危环节）
- repro: 复现条件——任一修订轮的「把占位伪代码写实/具体化」动作（本任务该环节引入唯一新阻塞）
- toolchain_stamp: 设计层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 6 段末追加）
    定位锚: |
      ### Phase 6：针对性修订

      仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
    old 文本: （即上述定位锚原文）
    new 文本: |
      ### Phase 6：针对性修订

      仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

      > **「写实占位伪代码」对照纪律**：修订把占位伪代码写实/具体化时，若该占位符对应某个 verified 实现（HEAD 旧终版/先例 kernel），写实的唯一依据是该实现的真实形态——逐行比对搬运方向与 buffer 数，不在「让它更具体」的名义下发明 verified 实现里不存在的结构（两跳/中间 buffer）；写实引入**新搬运原语**时前置三关检查：docs 方向表在列 + 全仓 examples 先例 + 最小编译探针（~1–2 分钟）——任一关不过即回退到 verified 形态（2026-09-20 ssd 重跑任务实证：写实自创 L1→L1 拷贝，探针硬失败，一轮修订白费）。
    动机: 修订链的「写实化」环节是新增缺陷高发区；三关检查使原语合法性在设计期闭合，不留给检视兜底（同任务检视侧的编译探针手段见 VP-2026-0105）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0105
- type: R
- title: design-review Phase 1 第 4 步补「编译探针」为可选决定性手段——API 可行性争议时编写最小 kernel 仅编译不运行（~1–2 分钟/个），同时探针「设计形态」（证伪）与「建议修复形态」（证可行），使阻塞问题的修改建议自带可行性证明
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 复检轮：问题 1 处置全程（探针 1 证伪 L1→L1 两跳、探针 2 证单跳直载可行——修复建议自带编译级证明，Stage 3 零试错）；REVIEW.md（复检轮）机械复核段「编译探针」表；第三轮探针 2 复跑再证（"COMPILED OK"）
- repro: 复现条件——设计形态无文档/先例佐证的 API 可行性争议（本任务 L1→L1 copy：方向表无此方向 + 全仓零先例）
- toolchain_stamp: 检视层规则；证据环境 tilelang 0.1.2+4515de8 / CANN 8.5.0 / 2026-09-20
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 4 步扩展）
    定位锚: "4. 必要时 Grep `tilelang/language/` 确认 API 是否有导出佐证（仅静态文本核对，不执行）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. 必要时 Grep `tilelang/language/` 确认 API 是否有导出佐证（仅静态文本核对，不执行）。**编译探针（可选决定性手段）**：设计形态/修复建议无文档与先例佐证、或负向断言需实证时，编写最小 kernel（仅含争议原语，Expert/Developer 按设计模式）**仅编译不运行**（~1–2 分钟/个）——同时探针「设计形态」（证伪）与「建议修复形态」（证可行），使阻塞问题的修改建议自带可行性证明（2026-09-20 ssd 重跑任务实证：探针 1 证伪 L1→L1 拷贝 + 探针 2 证单跳直载可行，修订一轮收敛、Stage 3 零试错；探针代码与报错全文内嵌 REVIEW.md 附录）。
    动机: 「检视只看文档不编译」使 API 可行性争议只能先例背书或留待 Stage 3 暴露——编译探针把该类争议在检视期闭合。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0106
- type: R
- title: design-review Phase 1 补充核对三则——① 同结构多表述位一致性（§1.4/§3.2/§3.3/§6.2/§4.4 平行表述位作为一个核对单元，占位符/写实化位置是重点风险区）② 复杂度表 ws/中继流量按声明形状全维乘积复算（per-task 口径）③ 数字联动核对含标定域外的外推/假设句
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z：① Stage 2 复检轮 Review-method 行（v0 检视只标 §1.4 占位式命名，漏 §3.2 步骤 7 口径矛盾——v1 写实后才暴露为字面结果错误）② Stage 2 first review 问题 4（R3 表 ws 流量 per-l-tile 口径低估 2.5×：87 vs 216MB@w2，独立复算拦截）③ Stage 2 第三轮维度 2 观察项（§4.5「Q=512 → 144KB」外推行漏改〔应为 160KB〕，两轮均未联动——非当前 workload、结论方向不变）
- repro: 复现条件——任一多章节 DESIGN 检视（三则均本任务三轮实证：字面结果错误 / 2.5× 流量低估 / 外推数字漏改）
- toolchain_stamp: 检视层规则（无运行时依赖）；证据环境 tilelang 0.1.2+4515de8 / 2026-09-20
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步之后追加第 8 步）
    定位锚: "   - 两项机械结果摘要（含 skip 项）附入 REVIEW.md「机械复核」段——与维度 8 的「独立推演」互补而非替代。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + 新步骤：
      8. **补充核对三则**（2026-09-20 ssd 重跑任务三轮检视实证）：
         - **同结构多表述位一致性**：同一结构（协议/搬运/循环）在设计文档的全部平行表述位（§1.4 伪代码 / §3.2 API 映射 / §3.3 / §6.2 循环表 / §4.4 路径图 / §4.5 预算表）作为一个核对单元整体检视——占位符与「写实化」位置是重点风险区（本任务 §3.2 步骤 7 的口径矛盾在 v0 只被读作命名占位，v1 写实后成为字面结果错误）；
         - **复杂度表流量复算口径**：ws/中继类流量按声明的 ws 形状全维乘积（per-task 口径）复算，不得采信 per-l-tile/per-AIV 局部口径（本任务 R3 表曾低估 2.5×：87 vs 216MB@w2——流量项为地板时会误导 Stage 4 目标）；
         - **外推/假设句数字联动**：数字联动核对须含标定域外的插值/外推句（grep 容量单位与「→」外推句式——本任务「Q=512 → 144KB」两轮漏改，应为 160KB）。
    动机: 三则都是「逐维度检视通过、整体对读才暴露」的缺口——第一则曾放过一个字面结果错误到复检轮，后两则靠人工复算拦截。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0107
- type: R
- title: develop SKILL 两条——① Phase 3 golden 落盘形态 = 函数体内联 torch 计算（薄包装/别名导入被 S3-GOLDEN-TORCH AST 检查拒绝）② Phase 4 首跑建议 --level L0 单层（逐 shape 可见编译结果，定位后切 all）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 3：① gate 3 报 S3-GOLDEN-TORCH（薄包装 `return ssd_chunk_scan_fwd_ref(...)` 函数体无 torch.* 调用——attempt 1 白耗 742s）；内联仓内参考的 torch 计算（纯 fp32 einsum 双路径 + tril 掩码 materialize 照抄）后 pass: true；② L0 各用例 shape 不同触发多次编译、逐一可见 "AscendNPU IR compile success"（本任务 L0 首跑全绿省去分层二分）
  - 〔2026-09-28 新亚型（不同任务，reduction 族），task argmax-_argreduce_kernel-20260924T031005Z Stage 3〕**张量方法形态**：golden 用张量方法 `x.argmax(dim=...)` 而非模块函数 `torch.argmax(x, dim=...)`——函数体同样无 `torch.*` 前缀调用，S3-GOLDEN-TORCH AST 检查判 fail（attempt 1 runtime fail 2731s → 修复 attempt 2 complete 70s，修法 = torch.* 命名空间直调）。建议 apply 时在 new 文本 1 的第 3 条补一句：「张量方法调用（`x.argmax`/`x.max` 等）同样不含 torch.* 前缀，须改写为 `torch.argmax(x, ...)` 模块函数直调（2026-09-28 argmax 实证：2731s 白耗）」；出处：examples/argmax/SESSION_TIMING_ANALYSIS.md §3 第 3 条 + .task_timeline.jsonl Stage 3 fail 事件
- repro: 复现条件——① 迁移任务复用仓内参考函数作 golden（`import ... as` 别名与薄包装委托两种形态均被拒）；② 任一多 shape L0 套件首跑
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20；gate_lint S3-GOLDEN-TORCH 现行版本
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（两处）
    定位锚 1: "2. Golden 必须在 CPU 上可独立运行（不依赖 torch_npu）。"
    old 文本 1: （即上述定位锚 1 原文）
    new 文本 1: |
      2. Golden 必须在 CPU 上可独立运行（不依赖 torch_npu）。
      3. **golden 函数体必须内联 torch 计算**（迁移任务复用仓内参考函数时的正确落盘）：把参考函数的 torch 计算逐段抄进 `def golden_...` 函数体（函数体直接含 `torch.einsum/tril/exp` 等调用）——`import ... as` 别名或 `return ref(...)` 薄包装的函数体无 torch.* 调用，会被 gate `S3-GOLDEN-TORCH` 的 AST 检查判 fail（2026-09-20 ssd 重跑任务实证：薄包装被拒白耗一次 attempt 742s；内联后语义与参考逐位一致，且无需 sys.path 指向参考目录）。
    定位锚 2: "1. 跑 L0：`python {op}.py --level L0`。"
    old 文本 2: （即上述定位锚 2 原文）
    new 文本 2: |
      1. 跑 L0：`python {op}.py --level L0`。（Expert 双域 kernel 首跑建议保持单层 L0——各用例 shape 不同会触发多次编译，逐一可见编译结果，定位到具体 shape 的编译失败后再切 `--level all`。）
    动机: golden 形态是 gate 交互问题（AST 检查只认函数体内的 torch.* 调用，委托形态语义等价也拒）；L0 分层是零成本定位习惯。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0108
- type: R
- title: develop SKILL Phase 2 补「迁移重做任务实现基线」——先 `git show HEAD` 取旧 verified 终版（perf_opt/）作结构模板逐行复刻，只做规格差异适配（pass_configs 按新规格 / golden 内联 / 测试套件按新 L0 计划）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 3：实现直接复刻 `git show 4515de8:examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py` 完整结构形态 + 三处差异适配——首次编译即通过、零调试往返（L0–Boundary 全绿，比从伪代码重推省 ~90% 时间）；RETROSPECTIVE.md Stage 3 Value Point Proposals 第 3 行 + Transferable 首条（后续 mamba 族未迁移成员直接以该文件为结构模板）
- repro: 复现条件——迁移重做任务（同 op 旧 verified 终版存在于 git HEAD、工作树副本被 Stage 0 删除）的 Stage 3 实现
- toolchain_stamp: tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 清单追加第 4 项）
    定位锚: "3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
      4. **迁移重做任务**（同 op 旧 verified 终版在 git HEAD、Stage 0 已删工作树副本）：实现基线 = `git show HEAD:<旧 perf_opt 终版路径>` 的完整结构逐行复刻（每个搬运方向/flag 置位/slice 裁剪已在旧工具链实证），只做规格差异适配——pass_configs 按新 DESIGN 规格、golden 内联仓内参考、测试套件按新 L0 计划扩充；从伪代码重推是下策（2026-09-20 ssd 重跑任务实证：复刻 + 三处适配首编即过零调试往返，省 ~90% 时间）。
    动机: 重做任务的最大红利是旧 verified 实现（结构层每个 API 形态已验证）——不取用即放弃；差异适配清单使复刻不退化为盲抄（规格演进点显式处理）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0109
- type: R
- title: integrate_kernel.py rewrite_wrapper_import 同步清理被替换 import 正上方的「Part A 提取说明」陈旧注释——避免描述已死 import 的注释误导后续维护者
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 5：examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan.py L56–L58（"# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py / # (pattern B) from {gpu_repo_root}/... / # ---" 注释块保留在 baseline/perf_opt 选择块之上，描述的是已被替换的 GPU 提取 import——且 {gpu_repo_root} 占位符未填充）；op 级 RETROSPECTIVE.md Stage 5 Skill Flow Issues 首行
- repro: 复现条件——任一 wrapper 带脚手架提取说明注释的集成（rewrite_wrapper_import 替换 import 后注释残留）
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir / 2026-09-20；脚本现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（rewrite_wrapper_import 拼接处追加注释清理）
    定位锚: |
      (rewrite_wrapper_import 函数体内)
          block = gen_kernel_source_block(names, integrated_files, op_slug)
          text = text[: m.start()] + block + text[m.end() :]
          wrapper_path.write_text(text, encoding="utf-8")
    old 文本: |
          block = gen_kernel_source_block(names, integrated_files, op_slug)
          text = text[: m.start()] + block + text[m.end() :]
          wrapper_path.write_text(text, encoding="utf-8")
    new 文本: |
          block = gen_kernel_source_block(names, integrated_files, op_slug)
          # Drop the stale "Part A: ... extracted ..." comment block sitting
          # directly above the replaced import: it documents the dead
          # GPU-extraction import (with unfilled {gpu_repo_root} placeholders)
          # and misleads maintainers after the swap (2026-09-20 ssd case).
          pre = text[: m.start()].rstrip("\n")
          part_a = re.search(r"\n(?:#[^\n]*\n)+$", pre)
          if part_a and "extracted" in part_a.group(0):
              pre = pre[: part_a.start()].rstrip("\n") + "\n\n"
          text = pre + block + text[m.end() :]
          wrapper_path.write_text(text, encoding="utf-8")
    动机: 脚手架生成的提取说明注释在 import 被替换后成为死文档；重跑/换代集成时该注释与新的 baseline/perf_opt 切换块并排出现，误导 wrapper 维护者（该文件是 wrapper 胶水的唯一权威说明位）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 2026-09-20
- decided_by: -
- decided_note: -

## VP-2026-0111
- type: R
- title: iteration-diagnosis.md Step 2「代码结构观察」补机械互证检查——继承的段数/流量记账须做「引擎指令数 ÷ 任务数」与 kernel 拷贝语句清单逐项对照（指令数是装载冗余的直接探测器，比散文记账可靠）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log.md §1 末节（Phase 1 代码结构新发现：prev_states 切片对 lt 无依赖却在 lt 循环体内 ⟹ Q=256 时 4× 重读 = 48KB/任务冗余 = Cube mte2 字节 20%；实测 19.0 指令/任务 = 1(x)+Σ_lt[1(ws_c)+1(prev)+(lt+1)(band)] 与代码结构精确互证）
  - 误判代价实证：旧任务（1990aa9fe4 谱系）两套段数拆分共同漏算 ×4（「1216 段不可减」的地板定位错误），本轮据此再挖出几何 1.078× 中的第一项（prevhoist −2.0~−3.4%）；constants.md CONST-mte2 段数更正注 + attention.md PL-1.18 update①（2026-09-21 已回写）
  - .task_timeline.jsonl（Stage 4 无 fail 于该项——发现靠 optimizer 自发代码结构核对，非流程强制；Skill Retrospective Skill Flow Issues 首行自评「旧档数字未被要求做代码结构级互证」）
- repro: python3 .agents/tools/kb_search.py "mte2 段数 指令数 互证"（数据侧落点：constants.md CONST-mte2 / attention.md PL-1.18）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 代码结构观察」bullet 列表末条之后追加一条）
    定位锚: "- 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。"
    old 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。
    new 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。
      - 装载冗余互证（含继承数字）：对每个搬运引擎把「指令数（如 aic_mte2_instructions）÷ 每核任务数」与 kernel 内拷贝语句清单逐项对照——每条 T.copy 对应一条指令，对不上即有漏算/冗余；逐维核对循环体内张量索引的循环不变性（索引不含某循环变量 ⟺ 该装载可提升出该循环）。从旧档/前轮继承的段数/流量记账在用于地板定位前必须过此互证（2026-09-21 ssd 实证：两套旧拆分共同漏算 prev ×4 重读，致「1216 段不可减」错误地板定位跨任务存活，指令数互证一轮发现 20% 可削字节）。
    动机: 散文式段数记账不可机械复核，错误记账一旦进入「继承裁决」链就跨任务存活并封死候选方向；指令数是 msprof 直接给出的机械计数，与代码结构一一对应——把互证设为 Step 2 标准动作可在首轮 profile 即发现记账盲区，而不是依赖 optimizer 自发核对。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0112
- type: R
- title: iteration-diagnosis.md「已知迭代结论」节补注——引擎临界性是动态量，「引擎 X 非关键」类结论每轮 winner 合并后须重算（AIV「非关键」在两轮 Cube 优化后失效、w2 转 AIV-bound）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log.md R2→R4 约束迁移链（prevhoist+l0c2x 后 w2 任务周期 6.58µs vs Cube mte2 busy 4.45µs/任务 ⟹ AIV-bound；vbrchoist 据此立项获益 −3.45%）+ Skill Retrospective Skill Flow Issues 第 2 行（「AIV『非关键路径』结论会随 Cube 侧优化失效」）
  - 知识侧姊妹条目：queue VP-2026-0110（BP_engine_criticality_drift，Tier 1——方法本体与判据；本条为流程侧每轮强制重算规则，两案合入时互链）
  - 〔2026-09-21 重建会话同谱系追加（不计确认），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z〕方向翻转级证据：vbrchoist 单点 H=80 族回退（−4.2~−6.7%）在 prevhoist 合入后消失（+2.2~+10.2%）——单点结论依赖 base 状态的更强形态（数据侧 attention.md PL-1.18 update②）；**建议 apply 时在同节邻位增补「累积序检验」一句**：与先验证据矛盾的 workload 族分裂先合入对侧已验证胜出复测、再谈 per-shape 条件路径（姊妹 BP 提案 VP-2026-0115，合入时互链）
- repro: none（流程规则；核算方法与实证数据自包含于 VP-2026-0110 delta 与 attention.md PL-1.18 update③）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 已知迭代结论」text block 之后追加一段）
    定位锚: |
      本轮 profile 相比上一轮 profile 哪些指标发生变化
      ```
    old 文本: |
      本轮 profile 相比上一轮 profile 哪些指标发生变化
      ```
    new 文本: |
      本轮 profile 相比上一轮 profile 哪些指标发生变化
      ```

      **引擎临界性是动态量（2026-09-21 ssd 实证）**：双引擎（Cube+Vector）kernel 的「引擎 X 非关键路径」结论只对当时的结构状态有效——每轮 winner 合并后须重算逐 workload 的「任务周期（Task Duration ÷ 每核任务数）vs 各引擎 per-task busy」核算，临界性反转时重启此前因「非关键」被否决的对侧候选（ssd：AIV「非关键」在两轮 Cube 优化后失效，w2 转 AIV-bound，Vector 侧 hoist 重新获益 −3.5%）。重算方法与判据见 bottleneck-patterns.md BP_engine_criticality_drift。
    动机: 旧结论按字面继承会把「当前非关键」固化为「永远非关键」，直接封死对侧候选池；重算成本是一次除法对照，收益是整轮优化方向（本任务 vbrchoist 即靠该核算立项）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0113
- type: R
- title: ab_test 交错协议判读规则——3-pair 符号检验统计功效不足（3/3 同向仍 p=0.25 判 tie），采纳判据 = 配对方向一致 + 机制佐证（资源占用严格更优），临界情形 --pairs 5 复测
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：opt_log R1 ab_test（ab_round1_w4：3/3 pair 一致 −2.85/−2.27/−3.02%，合并 −2.84%，verdict 仍 tie——p=0.25 为 3-pair 符号检验固有功效上限）+ R4（ab_round4_w2：3/3 一致 −2.97% 同判 tie）+ R2（ab_round2_w3：B_better 3/3 −3.0~−4.0% 合并 −3.97%——幅度够大时协议能判）
  - 本任务正确采纳路径：靠 iteration-diagnosis.md Step 6 既有「资源占用决胜」规则 + 机制佐证（mte2 指令 19→16/任务、Cube mte2 字节 −20%、busy 3289→2913µs −11.4%）采纳 prevhoist——判据形态有效但未在 T-2 协议条成文，本轮为自发正确应用
  - 工具侧查重确认：`.agents/tools/ab_test.py` 已支持 `--pairs`（L108，默认 3）——缺口在判读协议文档，非工具
- repro: python3 .agents/tools/ab_test.py --a <baseline> --b <candidate> --pairs 5 ...（工具现成；复现条件 = 任一 <5% 增益的 3-pair 交错协议输出 tie 而 pair 方向全一致）
- toolchain_stamp: tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（Step 6 交错 A/B 协议 bullet 末尾追加判读规则）
    定位锚: "工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。"
    old 文本: |
      工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。
    new 文本: |
      工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。**判读规则（2026-09-21 ssd 实证）**：3-pair 符号检验的统计功效上限 p=0.25——**3/3 配对同向仍判 tie 是协议固有，不是「无差异」证据**（两例：3/3 一致 −2.84% / −2.97% 均判 tie）；此形态的采纳判据 = ① 配对方向全一致（≥3/3 同向）**且** ② 机制佐证独立成立（资源占用严格更优：指令数/忙比/字节数等机制性削减指标，与 Task Duration 无关的计数）——两条件同时满足可采纳（本任务 prevhoist：pair 全同向 + mte2 指令 19→16/任务 + busy −11.4%）；仅方向一致而无机制佐证时用 `--pairs 5` 复测提高功效，不得据 3-pair tie 丢弃。
    动机: tie 判定若无配套采纳规则，3/3 同向的真增益会被当噪声丢弃（或反之被主观采纳）——机制佐证把「统计上未决」转化为「机制上可判」，且佐证指标（指令数/字节）不受 run 双态噪声影响。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0114
- type: R
- title: signal-registry.md §5 S4 产物格式契约显式化——perf_records final 的 artifact_path 以 perf_opt/ 为基准（裸文件名），Final Performance Test Data 表为 ##/### 标题 + 恰 5 列纯数字（gate 4 机械解析形态）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z：.task_timeline.jsonl（Stage 4 attempt 1 fail 7681s，verdict=runtime，2026-09-21T02:43:32Z——gate 4 S4-PERF-RECORDS-RECON 对账失败；attempt 2 窄修复 complete 170s）——失败细节来自 conductor 终态钩子输入，opt_log §7 复盘未覆盖该次 fail 的 gate 侧原因（复盘缺口形态，同 stats ⑧⑩ 先例）
  - 两处产物格式错误（attempt 1）：① perf_records.jsonl final 记录 artifact_path 用仓库根相对路径（examples/TileOPs/... 前缀），gate 以 perf_records.jsonl 所在目录（perf_opt/）为基准拼接成双重前缀 → abspath 不匹配 → 「final 未测最终 perf_opt 文件」；② opt_log Final Performance Test Data 用加粗列表项而非 ##/### 标题 + 6 列且数值加粗——gate_lint.py L908 要求 `^#{2,3} Final Performance Test Data\s*$` 标题、L919 要求恰 5 列、L925 要求 float() 可解析（加粗 → ValueError / 6 列 → 行被跳过 → 「Final 表缺少 {key}」）
  - 修复形态（attempt 2，170s）：final 行 artifact_path = 裸文件名 `_ssd_chunk_scan_fwd_kernel.py`（perf_records.jsonl 末 3 行）+ opt_log §6 `### Final Performance Test Data` + 5 列纯数字表
  - .agents/tools/gate_lint.py L882-884（相对路径 join dirname(records_path)）/ L908-927（标题正则 + 5 列 + float 解析）；现行 signal-registry.md §5 已列 5 列字段名但未写标题层级/纯数字/路径基准三要素
- repro: python3 .agents/tools/gate_lint.py（现行 S4-PERF-RECORDS-RECON 规则；负例 = final 行 artifact_path 带仓库根前缀 + Final 表加粗 6 列形态，正例 = 本任务 attempt 2 修复后产物）
- toolchain_stamp: gate_lint.py 现行版本（statectl gate 4 机械核对层）；发生环境 tilelang 0.1.2+15ad002b3d / CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/skills/_shared/standards/signal-registry.md
- delta: |
    动作: update（§5 字段契约段扩展一句）
    定位锚: "`phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。"
    old 文本: |
      `phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。
    new 文本: |
      `phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。**`artifact_path` 相对基准 = `perf_records.jsonl` 所在目录（`perf_opt/`）——final 记录写裸文件名 `{op}.py`，仓库根相对路径会被 gate 以 `perf_opt/` 为基准拼接成双重前缀而判「final 未测最终 perf_opt 文件」（2026-09-21 ssd 二轮调优实证：Stage 4 attempt 1 整轮 gate 对账失败重试）**。Final Performance Test Data 表的机械解析形态：`##`/`###` 标题行 + 恰 5 列 `| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |`，时延列为纯数字（不加粗、不用列表项替代表格行——6 列或加粗数值均使 gate 4 `S4-PERF-RECORDS-RECON` 解析失败或跳行）。
    动机: 契约的字段名已成文但「路径基准 / 标题层级 / 纯数字」三个 gate 解析要素未写——首轮按直觉写仓库根相对路径与富文本表格会整轮 gate 失败（本任务 7681s attempt 的收尾对账失败 + 170s 窄修复 + 一次 stage_retry；与 VP-2026-0036 的 null/手抄亚型同属 S4-PERF-RECORDS 产物契约违例家族，第三亚型）。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0117
- type: R
- title: conductor 终态钩子前置「任务产物保全」步骤——DONE 前对算子目录未提交产物做零工作树影响的 git 保全（临时索引 write-tree + update-ref），防 perf_opt 外部删除致算子断裂与调优档案永久丢失
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z：opt_log.md 任务背景（上一调优会话〔task ...20260921T003531Z〕DONE 后 perf_opt/ 目录被外部删除——wrapper 指向 perf_opt 导致算子断裂；14 个实验分支 + 64 行 perf_records 永久丢失）+ Skill Retrospective Skill Flow Issues 第 3 行 + 重建全程（本会话 mode=full 重建耗 10413s，靠 repro/PL-1.18-floor2-wins.py 骨架才恢复三胜出结构——实验数据与 opt_log 不可恢复）
  - 既有机制缺口：statectl `complete` 的「工件 SHA256 快照」只记哈希不保内容（.stage_state.json artifact_hashes——目录删除后无法恢复）；perf_opt/ 未提交文件在 git 工作区无任何恢复途径
  - 数据侧互补：attention.md PL-1.18 update①（repro 骨架自包含性经工件全失场景检验——ED-A/ED-B 的正反面实证：结构可由骨架重推导，过程数据全失）
- repro: 复现条件——任一 DONE 后算子目录 perf_opt/ 被外部删除（wrapper 断裂 + 档案丢失；恢复路径仅 repro 骨架重推导，实验记录不可恢复）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 2026-09-21（git 仓可用）
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（「### 1. 任务终态蒸馏钩子（必执行）」步骤列表前插入第 0 步）
    定位锚: |
      `phase` 进入 `DONE` 或 `FAILED` 后（最终报告输出后、同一会话内）：

      1. 判断是否存在**可蒸馏信号**（任一为真即有）：
    old 文本: （即上述定位锚原文——intro 行 + 步骤 1 起始行）
    new 文本: |
      `phase` 进入 `DONE` 或 `FAILED` 后（最终报告输出后、同一会话内）：

      0. **任务产物保全（零工作树影响，best-effort）**：对任务算子目录的未提交产物（`perf_opt/` 全部文件、`history_version/`、`integration_log.md` 等）执行 git 保全——`idx=$(mktemp); GIT_INDEX_FILE=$idx git read-tree HEAD; GIT_INDEX_FILE=$idx git add -- <算子目录>; tree=$(GIT_INDEX_FILE=$idx git write-tree); commit=$(git commit-tree $tree -p HEAD -m "task artifacts: {task_id}"); git update-ref refs/task-artifacts/{task_id} $commit; rm -f $idx`（分支头与工作树均不动；恢复 = `git checkout refs/task-artifacts/{task_id} -- <算子目录>`）。失败不阻塞、不重试，记录到最终报告。动机：上一会话 DONE 后 perf_opt/ 被外部删除致 wrapper 断裂 + 14 实验分支/64 行 records 永久丢失（2026-09-21 ssd 重建任务实证——重建耗整个会话且实验数据不可恢复；statectl SHA256 快照只记哈希不保内容）。
      1. 判断是否存在**可蒸馏信号**（任一为真即有）：
    动机: 调优产物是后续蒸馏/续调/断裂修复的唯一载体；「wrapper 指向 perf_opt + 目录可被外部删除 + 内容无 git 保全」三条件叠加使每个 DONE 任务暴露于不可恢复损失——零工作树影响的 ref 保全把恢复成本从「整个重建会话」降到一条 checkout 命令。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: 备选实现（apply 时可裁决）：① statectl `complete` 内置同构保全（工具侧，需改 .agents/tools/statectl.py——机械动作更可靠）；② optimize SKILL.md Phase 4 交付清单加「提示用户 commit perf_opt/」提醒（最轻量）。三形态取一即可。

## VP-2026-0118
- type: R
- title: profile-collection.md 补长采集后台运行形态——`cd X && nohup ... &` 的 `&` 作用于整链，调用方 shell 工具超时 SIGKILL 连带杀死采集进程组；须 setsid 脱离进程组启动
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z：opt_log.md §2 注（perf_records.jsonl 前三行为启动事故——首次后台 baseline 采集被 shell 超时连带杀死前已落 3 行，重启后重测同三案例产生同分布重复记录，append-only 契约下须逐行注记对账）+ Skill Retrospective Skill Flow Issues 第 5 行 + perf_opt/logs/phase1_baseline_all.log（首次启动事故日志）
- repro: 复现条件——以 `cd <dir> && nohup <采集命令> > log 2>&1 &` 形态从带超时的 shell 工具启动分钟级采集，超时触发后观察采集进程被连带 SIGKILL（对照 setsid 形态存活）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 2026-09-21
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（§3 串行运行 msprof op 的注意事项后追加一段）
    定位锚: "注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。"
    old 文本: |
      注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。
    new 文本: |
      注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。

      长采集（全 workload baseline 扫描等分钟级任务）以后台方式运行时，用 `setsid` 脱离当前 shell 的进程组启动（`setsid nohup bash -c '<采集命令>' > {log} 2>&1 &`）——`cd X && nohup ... &` 形态中 `&` 作用于整条链，调用方 shell 工具超时的 SIGKILL 会连带杀死采集进程组，产出半途记录（2026-09-21 ssd 重建任务实证：首次 baseline 采集即被连带杀死、重启后 3 行同分布重复记录，append-only 契约下须逐行注记对账）。
    动机: 采集被中途杀死产生两类成本：半途记录污染 append-only 对账（须注记区分）+ 整轮重采；setsid 是零成本进程组隔离。
- status: pending
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: -
- decided_note: -

## VP-2026-0132
- type: R
- title: op-design SKILL D-3 探针纪律三条——① 预算口径补「同一探针文件的失败-修复迭代计入同一探针，总墙钟超限须在 DESIGN 附录如实记录」② 探针须覆盖冻结目标配置的规模挡位（多迭代 serial 循环 ≥3 次更新；小规模替代形状显式声明未覆盖维度）③ 探针执行 stdout tee 落盘 probe_logs + DESIGN 附录 A 增「日志路径」列
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1/2/3：① D-3 预算「≤2 个、每探针 ≤10min」按文件数合规但 8 轮 case 迭代总墙钟 ~25min（design_probe_argmax.py --case 逐轮演进，DESIGN 附录 A）；② tiled 在线递推探针只验了 num_full==2（P2 (3,700) tn=256 / P2b (96,8192) tn=4096），冻结配置 tn=5120→num_full=20 首跑才首次执行——原位 vselect 状态丢失 bug 恰在多迭代暴露（Stage 3 首跑 (4,102400) 输出全落 tile-0 局部索引）；③ Stage 1 探针无 stdout 落盘，Stage 2 交叉验证 C-1~C-4/C-8 只能重新在设备上跑（P1/P2/P2b/p3a×3/dbg 五轮 ~12min 墙钟）；无设备 reviewer 只能纸面接受声明（本轮探针窗口 2026-09-24 03:50–04:15 零 log 文件）
- repro: 复现条件——任一设计期探针产出被 Stage 2 检视消费的场景（探针清单含多迭代循环结构 / 探针结论被交叉验证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（D-3 段扩展，三处）
    定位锚: "严格预算：**≤2 个探针、每探针 ≤10min**，只测未知常数（最小张量/最小 kernel），不实现完整算子；探针脚本落 `examples/{project}/{op}/history_version/design_probe_*.py`（供 gate 与检视核对）"
    old 文本: （即上述定位锚原文）
    new 文本: |
      严格预算：**≤2 个探针、每探针 ≤10min**（同一探针文件的失败-修复迭代与逐 case 轮次计入同一探针预算；总墙钟超限时在 DESIGN 附录如实记录而非拆分文件规避），只测未知常数（最小张量/最小 kernel），不实现完整算子；探针脚本落 `examples/{project}/{op}/history_version/design_probe_*.py`（供 gate 与检视核对）。**规模挡位覆盖**：对方块循环本身有依赖的候选结构（loop-carried 状态 / 级联更新），探针须用**冻结目标配置的 tile 数/迭代数挡位**跑一遍（多迭代 serial 循环 ≥3 次更新）——小规模替代形状须显式声明其未覆盖的规模维度（argmax 实证：num_full=2 探针全过而 num_full=20 首跑暴露状态丢失）。**探针日志落盘**：探针执行 `tee` 到 `history_version/probe_logs/<case>.log`（含时间戳与工具链戳），DESIGN 附录 A 表增「日志路径」列——Stage 2 检视优先读日志比对，无日志且设备可用时重放、设备不可用时标注不可核验（argmax 实证：无落盘使 reviewer 五轮重放 ~12min / 纸面接受两分支都发生）；
    动机: 三条缺口同源于「探针产出的可消费性」：预算口径可被文件数合规绕过、规模挡位缺口使 bug 逃逸到 Stage 3（2731s attempt）、日志缺失使检视要么重放要么盲信。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: ② 的消费侧（design-review SKILL「有 probe_logs 则先读日志比对」）随本条 new 文本一并生效，review SKILL 侧不再单独立案。

## VP-2026-0133
- type: R
- title: op-design SKILL Phase 3 补「引用仓内既有 Python 原语必须 Read 实现核对参数单位与接受判据」——与 DSL API 查文档同纪律；附 compute_tile_n 参数口径实例（budget=字节 / num_buffers=同形缓冲个数 / elem_bytes=单缓冲元素字节 / 除数优先仅在不增加 tile 数时生效）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1 revision：v0 引用 `tileops/kernels/reduction/_primitives.py` 的 compute_tile_n 未读实现、凭参数名猜语义（budget 折元素数 + 再传 elem_bytes/num_buffers 二次折算）——fp16 lm-head 得 tile_n=512 而非 5120（tile 数 ×10），Stage 2 检视读 L110-196 实现才推翻（`per_buffer = block_m*elem_bytes; max_cols = budget // (num_buffers*per_buffer)`；`if div_tiles <= max_tiles: return best_dividing`）；复算：compute_tile_n(1,2,102400,256,6553,5)=512 vs budget=65536→6400；bf16 elem_bytes=2,num_buffers=7,budget=65536→4608+尾 tile（非零尾 4096）
- repro: 复现条件——任一设计引用仓内 host 侧工具函数（_primitives.py 等）套用公式的场景；复算命令自包含于证据（纯 host 算术，与设备无关）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 dev root build 2026-09-24 / 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 3 信息收集章节补一条）
    定位锚: "### Phase 3：信息收集"
    old 文本: （即上述定位锚原文行——在该节 checklist 末尾追加）
    new 文本: |
      ### Phase 3：信息收集
      （节内追加一条）**引用仓内既有 Python 原语（`_primitives.py` 等 host 侧工具函数）时必须 Read 其实现核对参数单位与接受判据**，与 DSL API 查文档同纪律——凭参数名猜语义是 1024× 级错误的来源之一（argmax 实证 2026-09-24：compute_tile_n 的 budget 为**字节**、num_buffers 为**同形缓冲个数**、elem_bytes 为单缓冲元素字节，预算先折元素数再传参会二次折算使 tile_n 512≠5120〔tile 数 ×10〕；其「除数优先」仅在**不增加 tile 数**时生效——bf16 返回 4608+尾 tile 而非零尾 4096，想要零尾 tile 须 wrapper 侧自有规则）。
    动机: 「引 API 查文档」纪律只覆盖了 DSL API，漏了仓内 host 侧工具函数——v0 凭参数名猜语义导致 tile 数 ×10 的设计错误存活到 Stage 2 才被源码对账推翻。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 知识侧（wrapper tile_n 整除优先规则）已随 VP-2026-0129 入队 algorithm-candidates；本条为流程侧（Read 实现纪律 + 口径注记）。

## VP-2026-0134
- type: R
- title: op-design SKILL DESIGN 产出要求补「集成前置核对」——reduce 族（及一切 Op 层有输入预处理的族）设计阶段必须确认目标 Op 层 `_kernel_handles_padding`/`_pad_value` 行为，kernel 声明宽度与 Op 实际传入宽度一致
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 5：集成前置核对只查 manifest test/bench 与 meta 一致性，未查 kernel 数据入口契约与 TileOPs Op 层 pad 行为一致性——DESIGN 冻结的 raw-N 契约（R3）与 `_ReduceOpBase._prepare_input` 的 F.pad(N_padded, -inf) 冲突直到 Stage 5 report 才暴露（初次 report 17 failed 全落在 N_padded≠N 用例，修复耗一整轮 integrator attempt）；tileops/ops/reduction/reduce.py#_prepare_input + integration_log.md#attempt-1
- repro: 复现条件——任一 reduce 族（或 Op 层含 pad/reshape 预处理）算子的 harness 迁移设计
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ubuntu.22.4.npuir / git 013dbbf5 / 2026-09-28
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（DESIGN 产出 checklist / 交付清单段补一条）
    定位锚: design SKILL.md「11. 交付清单」（Phase 4 checklist 段）
    old 文本: （定位锚节内追加一条，apply 时按当时文件内容锁定精确锚）
    new 文本: |
      - **集成前置契约核对（迁移任务）**：设计阶段确认目标 TileOPs Op 层的输入预处理行为——reduce 族读 `tileops/ops/reduction/reduce.py#_prepare_input`（`_kernel_handles_padding=False` 时输入 pad 到 align_up(N,256)，kernel jit 声明宽度须取 N_padded；翻转 True + kernel 原始 N 是高杠杆优化项）；op_kind 涉及 min/max 的共享 kernel 须核对 `_pad_value` 方向性（argmax −inf / argmin +inf）。kernel 数据入口契约与 Op 层实际行为不一致 = Stage 5 集成期爆雷（argmax 实证 2026-09-28：17/42 failed，步长错位症状行 0 对、行≥1 全错且不报错）。
    动机: 契约冲突在设计期是零成本核对（读一个 Op 基类文件），在 Stage 5 是整轮集成返工 + report 重跑；知识侧条目（CASE-argmax-argreduce-integration + VP-2026-0128）覆盖检索路径，本条把核对动作变成设计产出要求。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0128（知识侧，algorithm-candidates）同一发现的两 Tier 拆分；add-npu-op skill integrate 前置节为备选 target（审批时可改路由）。

## VP-2026-0135
- type: R
- title: design-template.md 三条——§5.2 workload 表首列要求附 (M,N) 推导式（无推导式视为待确认）；§5.2 阶梯段补「扩展阶梯（小 N 大 M）预算解除用窄 N 实测膨胀系数（无实测按 4× 保守）+ 阶梯末 cap bm ≤ M」；§5.5 分核表增可选「反事实锚点」列（沿用最差合法配置的发射项对照）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1 revision/Stage 3：① v0 逐 workload 表数字手抄（无 M/N 推导算式）使 1024× 量级 shape 错误（512 vs 524288）存活到 Stage 2 三源交叉才推翻——错误根因不是算错而是抄错后无推导式可自检（revision Skill Flow Issue #1）；② §9.2-R10 预判 (2048,4) 窄内维未探针覆盖但冻结值仍取 bm=2048，首跑 auto-multi-buffer 4× 膨胀溢出（64KB→256KB）返工降档 1024（Stage 3 Skill Flow Issue #2）；block_m>M 时 (1,3)/(1,5) bm=8 尾行垃圾污染有效行；③ 反事实锚点实践：§1.6.0/§5.5 附「沿用最差合法配置的发射项」对照列（3d 行 bm=8→≈4.8ms），使「规模判定↔bm 取值↔发射估算」不自洽在纸面复算即可暴露（revision VP #4）
- repro: 复现条件——任一含逐 workload 表 + bm 阶梯 + 分核表的 DESIGN 撰写/检视
- toolchain_stamp: 流程规则（模板产出要求）；证据环境 tilelang 0.1.2 dev root build 2026-09-24 / 2026-09-24
- target_doc: .agents/skills/tilelang-op-design/templates/design-template.md
- delta: |
    动作: update（三处）
    定位锚 1: "### 5.2 Block 划分"
    new 文本 1: 节内 workload 表说明追加：「逐 workload 表首列附 (M,N) 推导式（如 `dim=0: N=x.shape[dim]、M=prod(其余)`，按 Op 层 reshape 规则重算——movedim(dim,−1) 后 reshape），无推导式的行视为待确认」。
    定位锚 2: "### 5.2 Block 划分"（阶梯/回退伪代码段）
    new 文本 2: 阶梯段注记：「扩展阶梯（小 N 大 M）的预算解除须用**窄 N 实测膨胀系数**（宽 N 的 2× 口径在 N<8 实测 ~4×——(2048,4) bm=2048 手工 64KB→requires 256KB 溢出，降档 1024 通过）；无实测时按 4× 保守取值或直接取降档档位为冻结值；阶梯末 `block_m = min(block_m, M)`（bm>M 的尾行是 UB 未初始化垃圾、窄 N 下污染有效行）」。
    定位锚 3: "### 5.5 分核策略（物理核数适配）⭐"
    new 文本 3: 节内分核表说明追加：「可选『反事实锚点』列——逐 workload 附『沿用最差合法配置的发射项』对照（如 3d (524288,4) bm=8 → ≈4.8ms vs 选中 bm 2048 → ≈18.7µs），使规模判定↔bm↔发射估算的不自洽在纸面复算即可暴露；跨数字一致性约束（改善倍数 ≤ 流量缩减倍数）是免费 invariant」。
    动机: 三条都把「设计期可机械核对的数值」变成模板硬要求——本任务三处失准（1024× shape / 窄 N 膨胀 / 发射项失配 86×）全部有纸面自检入口可拦。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 知识侧（阶梯规则本体）已随 VP-2026-0129 入队；本条为模板产出要求侧。窄 N 膨胀数据点见 VP-2026-0122。

## VP-2026-0136
- type: R
- title: design_calc_check.py 增 `--manifest <yaml> --op <OpName>` 模式——按 Op 层 M/N 推导规则算出每条 workload 的 (M,N) 真值，与 DESIGN §5.2/§5.5 表的 shape/block_m/num_row_blocks/核内任务数逐格 diff；design-review SKILL Phase 1 机械复核段加该调用
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2：最严重阻塞项（3d-non-last-axis 写成 (512,4)，manifest+Op 层真值 (524288,4)，差 1024×）只能靠人工「manifest workloads → `_ReduceOpBase._prepare_input` 的 M/N 推导（dim=0 → M=除规约维外全维乘积）」两步交叉推出；design_calc_check 的 core_split 检查因解析不到维度直接 skip（`"status":"skip","detail":"insufficient parsed dims {}"`）——shape 正确性完全交给 LLM 复核（REVIEW.md 问题 1）
- repro: 复现条件——任一 harness 迁移 DESIGN 的 workload 表机械核对；复算规则自包含（dim=0/中间维 → M=prod(其余)、N=该维长度）
- toolchain_stamp: design_calc_check.py 现行版本；失效环境 tilelang 0.1.2 dev root build 2026-09-24 / 2026-09-24
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（新增 CLI 模式 + 调用点）
    定位锚 1: design_calc_check.py CLI 参数解析段（现有 --design 模式旁增 --manifest/--op）
    new 文本 1: 新增模式：`python3 .agents/tools/design_calc_check.py check --design <DESIGN.md> --manifest <manifest.yaml> --op <OpName>`——按 Op 层 reshape 规则（含 `_prepare_input` 的 pad/N_padded 推导）算出每条 workload 的 (M,N) 真值，与 DESIGN §5.2/§5.5 表逐格 diff（shape/block_m/num_row_blocks/核内任务数）；不匹配即 fail 并打印真值推导式（manifest workload 行 → movedim/reshape → (M,N)）。
    定位锚 2: .agents/skills/tilelang-design-review/SKILL.md Phase 1「算术复算」调用行
    new 文本 2: 机械复核调用扩展为含 --manifest 模式（harness 任务 manifest 可得时必跑）。
    动机: 1024× 级 shape 错误是本任务 3 项 Stage 1 修订之首（返工成本 7666s：Stage 1 fail+定点修复 + Stage 2 首检不通过）；真值来自 manifest + Op 层规则而非 LLM 复读——与 VP-2026-0047（表形态解析）互补：0047 修「设计表读得进」，本条补「真值算得出」。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0047 联动（其第四现证据即本任务）；合入顺序建议 0047 先（表解析）或同批。

## VP-2026-0137
- type: R
- title: core-split-strategy.md 补 N 向分核（N-split）谓词——分核三要素表与 §2.1/§2.4 透传文本当前只覆盖 M 向分核；小 M 大 N 负载（M≤16 且 M<核数 且 N≥32768 且 N%256==0）应路由到 N-split 两段 kernel 结构，配套实验裁决三件套（分派谓词 + 判定阈值 + 回写路径）作为 §1.6.4 落地范例
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 4 + [DESIGN_LIMIT]：lm-head 类 (4,102400) 单 kernel 主选被 C-5（multitile bm>1 SIGSEGV）钉死 bm=1 → 4/48 核、实测贴 4 核 MTE2 聚合带宽地板；预注册 C6 N-split 备选 A/B 裁决 2.20×/2.60×（0.461×/0.386×，均 « 0.8× 翻转阈值）结构性胜出并采纳（perf_feedback.md [DESIGN_LIMIT] 双门槛达成；DESIGN §1.6.4 实验裁决三件套在续跑会话直接闭环——预注册价值实证）；elementwise.md PL-1.20（Stage 4 已回写结构形态）；遗留：DESIGN §1.4/§3.3 正式化走设计修订路由
- repro: 复现条件——任一 row-reduction 族含小 M 大 N 负载的设计（lm-head 形态）；结构形态与 tn 规则见 PL-1.20（知识域）
- toolchain_stamp: tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24~28
- target_doc: .agents/skills/_shared/standards/core-split-strategy.md
- delta: |
    动作: update（两处）
    定位锚 1: "## 1. 分核策略三要素（Stage 1 门禁核对项）"
    new 文本 1: 节末追加：「**N 向分核（N-split）**：M 向分核对 M < 核数的小 M 大 N 负载（lm-head 形态，M≤16 且 M<物理核数 且 N≥32768 且 N%256==0）失效（核利用率钉死 M/48）——此类负载路由到 N-split 两段 kernel（partial 按 chunk 分核 + merge 单 block，结构见 pattern-library elementwise.md PL-1.20）；分派谓词静态可判，判定阈值（端到端 < 0.8× 主选即翻转）与回写路径（§1.4/§3.3 修订 + Stage 2 复审）按实验裁决三件套预注册（argmax 实证 2026-09-28：2.20×/2.60× 翻转）」。
    定位锚 2: "### 2.4 Stage 4 — optimizer 调优提示（透传文本）"
    new 文本 2: 透传文本补一句：小 M 大 N 负载的单 kernel 地板归因（核利用率 M/核数 + 带宽贴地）时，核对设计是否预注册 N-split 备选裁决；未预注册且结构性天花板明确（>2× 估计）→ 触发 [DESIGN_LIMIT] 路由。
    动机: 分核标准只覆盖 M 向使设计期对小 M 大 N 负载无标准入口；本任务靠 §1.6.4 自建三件套才闭环——预置谓词使同族算子（ArgminFwdOp 等）设计期直接命中。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 知识侧结构形态已 Tier 0（PL-1.20 + CASE-argmax-argreduce-stage4）；本条为分核标准/流程侧。

## VP-2026-0138
- type: R
- title: design-review SKILL Phase 2 复审纪律两条——① 复审场景上一轮 REVIEW.md 的文档引文/实测数字一律视为待核工件（阻塞修复涉及的支持面论断必须重读原文档，不得 diff 两版 REVIEW 文本代替取证）② 阻塞修复涉及的数值及其派生量在 DESIGN 全文检索所有出现点并逐处同口径重算（「修订说明列出的章节」≠「受影响的全部章节」）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2 复审：① v0 REVIEW 引用 T.transpose.md §2.2.1 dtype 表把 uint32 写成 √（文档实际 ×），v1 DESIGN 更正复述与 v0 REVIEW 引文冲突，复审重读原文才裁定 v1 准确——若沿用前轮引文会误判 v1「引入新错误」；② v1 定点修订在未改动章节留下三处数字残留（附录 A 262KB〔应 268MB〕/ §9.2-R8 32768B〔应 28672B〕/ §5.2 269MB〔应 272.6MB〕），靠全文 grep + 逐处重算才捕获（REVIEW.md v1 复审维度 2/3/7 N2/N3/N4）
  - 〔2026-10-14 证据链追加（task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z，pool 族双输出 1D 变体）〕Stage 2 复审实测：复审者独立复算两条阻塞项（容量系数闭式 ∧ 逐 buffer 求和、空窗口域不变量的真实充分条件）后才判「通过」——与①「上一轮 REVIEW.md 的引文/数字一律视为待核工件」一致；②「修复爆炸半径」在本任务以「冻结伪代码 ↔ 分析表 ↔ buffer 清单」跨节缓冲名交叉检索体现（发现 §1.6.2 双缓冲残留）。建议把该跨节检索点写成显式步骤（与 VP-2026-0147 的跨节一致性检查项互补：本条管复审工作流，0147 管维度 3/7/8 检查项）〕
- repro: 复现条件——任一 Stage 1 revision 后的 Stage 2 复审（阻塞修复核验场景）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-24
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 2 复审要点段追加两条）
    定位锚: "### Phase 2：逐维度检视"
    old 文本: （该节复审要点 checklist 处追加，apply 时按当时文件锁定精确锚）
    new 文本: |
      - **复审的「独立」= 证据重新取证，不是文本 diff**：上一轮 REVIEW.md 的文档引文/实测数字一律视为待核工件——阻塞修复涉及的支持面论断（dtype/shape 表、限制条款）必须重读原文档（argmax 实证 2026-09-24：v0 REVIEW 的 dtype 表引文错一列 uint32，沿用会误判 v1 修复引入新错误）；DESIGN 修订说明只当索引用。
      - **阻塞修复的数字体系全文同口径重算扫描**：对每个修复涉及的数值及其派生量（比例/裕度/合计）在 DESIGN 全文检索所有出现点并逐处重算——定点修订的残留高发于修订说明未覆盖的相邻章节（argmax 实证：3 处残留全部在未修订章节）。
    动机: 复审若以「diff 两版文本」代替取证会把前轮引文错误传染为对修复的误判；数字残留的捕获成本（全文 grep）远低于流出到 Stage 3。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0094（引用抄录一手来源）互补：0094 管设计侧引用，本条管复审侧取证。

## VP-2026-0139
- type: R
- title: design-review SKILL 检查项三则——① 维度 8 弃选论证 dtype/shape 支持面论断须同时核对所引文档表与 pattern-library 条目 `dtype:`/`status:` front-matter（实测口径比文档表更强）② 「依赖未实证常数」判定前先检索 `perf_opt/logs/` 该备选的终版日志（不只看首版反例）③ 维度 3 增「冻结工厂逻辑的分支可达性 + 预算口径一致性」核对点（阶梯/回退逐分支代入边界值）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 2：① 设计称 T.transpose dtype 限 fp16，被 T.transpose.md §2.2.1 与 pattern-library PL-1.1 front-matter `dtype: [fp16, fp32, bf16] / status: verified` 双双反驳——现有 skill 文本只点名「dtype 矩阵」未提示核对 pattern-library 实测口径（REVIEW 问题 3）；② 设计引 round8_op9 首版 4/4 mismatch 论证 C6 风险，漏引同 log 族 round9 终版（partial ≈15.7µs + merge ≈1.7µs golden PASS）——该数字正是 C6 收益常数的现成实测，直接改变「未实证常数」判定（REVIEW 问题 S5）；③ v1 阶梯伪代码 p=1 档死分支（`block_m is None and N×B_bm1 ≤ budget` 恒不可达）+ 契约外 N 区间选中 bm=1 驻留但实际 alloc 超自设安全预算（fp32 N=5000：60000≤65536 过滤通过，实际 bm=1 alloc 80000B > 64KB 纪律线）——分支可达性推演 + 边界 N 代入口径核对可纸面判定（复审 VP #1）
- repro: 复现条件——任一含弃选论证/实验裁决计划/工厂期阶梯伪代码的 DESIGN 检视
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-24
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（三处）
    定位锚 1: 维度 8 表「向量化轴与布局决策完整性」行的「弃选论证前提核对」句（"...每个弃选候选的 repack/代价机制前提必须与所引 API 文档亲自核对一致..."）
    new 文本 1: 该句尾追加：「dtype/shape 支持面论断须同时核对所引文档的 DataType 表**与** pattern-library 对应条目的 `dtype:`/`status:` front-matter（实测口径比文档表更强）；两者任一矛盾 → fail（argmax 实证：T.transpose 文档表与 PL-1.1 front-matter 双源反驳设计声称的 fp16 限制）」。
    定位锚 2: 维度 8 表「实验裁决模式完整性」行（"判定裕度依赖未实证常数时"句）
    new 文本 2: 该行判定口径追加：「备选被判『依赖未实证常数』时，先在 `perf_opt/logs/`（同算子前次任务或本任务早期）按 candidate/round 名检索该备选的**终版**日志——首版 mismatch 与终版 golden PASS 可能并存于同一 log 族，已有实测则该常数不再是未实证（argmax 实证：round9 终版 partial/merge µs 直接改变 C6 判定）」。
    定位锚 3: 维度 3（Tiling 策略）检查项表
    new 文本 3: 增检查点：「冻结规格中的工厂期选择逻辑（阶梯/回退/分派伪代码）做**分支可达性 + 预算口径一致性**分析：每条分支代入边界值走一遍（死代码 = 恒不可达分支；口径漂移 = 过滤器用 A 口径而 alloc 用 B 口径——如 p=1 档用 B_bm1 核算而实际 alloc 按 B_multi）——选定值对契约内形状正确不代表选择函数对契约外形状守纪律」。
    动机: ①② 都是「证据源完整性」缺口（文档表 ≠ 全部证据源；首版反例 ≠ 全部日志）；③ 是「选定值正确 ≠ 选择函数正确」的覆盖层缺口。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: ①与 Phase 1 第 5 步「弃选论证查证（强制）」的 pattern-library 对照条款衔接（该步已要求对照 pattern-library——本条把 front-matter dtype/status 字段明确为核对对象并绑定 fail 判据）。

## VP-2026-0140
- type: R
- title: integrate_kernel.py gen_kernel_source_block 检测 kernel 模块内被 wrapper 消费的模块级 helper（`_select_config` 等）自动生成成对 baseline/perf_opt import 行——当前只搬运 extracted_functions 名单内的 kernel factory
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 5：wrapper 胶水需要 `_select_config`（kernel 探针校准阶梯）时只能手工加行；本次手工补 `_select_config` 成对行（tileops/kernels/reduction/argmax/argmax.py import 块）；integrate_kernel.py#gen_kernel_source_block（L275 起）仅按 extracted_functions 名单生成行；perf_opt 采纳翻转时易漏翻（integration_log 遗留说明：「翻转 import 时须同步翻转 _select_config 行」——人工记忆依赖）
  - 〔2026-10-14 第二现 + 新暴露面（不同任务，reduction 族同 kernel），task argmin-_argreduce_kernel-20261014T151121Z Stage 5〕同一手工加行形态再现：`argmin.py#default_config` 委托 `_select_config` 时须 import 私有模块路径 `from .argmin_kernel._argreduce_kernel import _select_config`——**聚合 `argmin_kernel/__init__.py` 只 re-export kernel factory**（`from ._argreduce_kernel import _argreduce_kernel` / `__all__ = ["_argreduce_kernel"]`），`_select_config` 不在 re-export 面内，因此 wrapper 胶水无法经聚合包公共路径消费该 helper。**新暴露面（建议并入本条生成逻辑）**：`gen_init_py` 在生成/维护聚合 `__init__.py` 时，对被 wrapper 消费的模块级 helper（`_select_config` 等）一并 re-export（或至少生成稳定的公共别名），避免「wrapper 依赖私有模块路径」这一脆弱形态；信息取自集成侧 `argmin_kernel/__init__.py`（集成副本，随交付件存在）+ integration_log.md 等价说明。
- repro: 复现条件——任一 kernel 模块含模块级 helper 且 wrapper default_config 消费它的集成/翻转场景；重跑 integrate_kernel.py 覆盖集成副本后 helper import 行消失
- toolchain_stamp: tilelang 0.1.2+ubuntu.22.4.npuir / git 013dbbf5 / 2026-09-28；integrate_kernel.py 现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（gen_kernel_source_block 逻辑扩展）
    定位锚: integrate_kernel.py#L275 def gen_kernel_source_block（按 extracted_functions 名单生成 import 行的循环体）
    old 文本: （按名单生成 kernel factory import 行的现行逻辑）
    new 文本: |
      生成 import 行前对 kernel 源模块做静态扫描：模块级 `def _select_config`（及其他被 wrapper 模板引用的 helper 约定名单）存在时，自动生成成对 baseline/perf_opt import 行（与 kernel factory 行同步翻转注释态）；扫描不到则维持现状。
    动机: wrapper config 权威已确立为 kernel 校准阶梯（VP-2026-0093 argmax 第二证：GPU 启发式直接 ub overflow）——helper import 成对性靠手工维护是翻转漏翻的已知隐患点。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 与 VP-2026-0055（gen_kernel_source_block stem fallback 缺陷）同函数域不同缺陷，合入时互检；与 VP-2026-0093/0074（config 契约族）联动。

## VP-2026-0141
- type: R
- title: 脚手架/工件清理流程的知识保全规则——同算子重迁移（Stage 0 脚手架重建工作区）不得清理 `examples/{op}/{func}/repro/*.py` 与 capability-gaps 登记条目；跨任务同算子知识继承路径（perf_opt/logs + repro + CG）在 scaffolder 流程显式化
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：前次同算子任务的工具链陷阱知识几乎流失——capability-gaps 曾有 CG-2026-0014 登记（trap docstring 自称已登记）但条目丢失；repro .py 源码被清理仅存 `repro/__pycache__/*.pyc` 与 `perf_opt/logs/`；本次靠 pyc 反序列化 docstring + perf 日志考古才恢复两条关键陷阱（multitile bm>1 segfault / vcast f32→f32 corrupt）与性能校准数字（hidden-state fp16 bm=2/bf16 bm=1 终选值与前次完全互证）（Stage 1 复盘 Info-source 行 + DESIGN §9.4-R9 补录记录）
- repro: 复现条件——任一同算子二次迁移任务开始时检查前次 repro/*.py 与 CG 登记的存活状态
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-24（前次任务工件 2026-09-23）
- target_doc: .opencode/agents/tileops-scaffolder.md
- delta: |
    动作: update（执行流程补知识保全条款）
    定位锚: tileops-scaffolder.md「## 执行流程」段（S1-S7 移植步骤列表）
    old 文本: （该流程段末尾追加，apply 时按当时文件锁定精确锚）
    new 文本: |
      6. **前次任务知识保全（同算子重迁移）**：Stage 0 重建/清理工作区时**保留** `examples/{op}/{func}/repro/*.py`（知识域 repro 源）、`examples/{op}/{func}/perf_opt/logs/`（实测锚点）与 `.agents/evolution/capability-gaps.md` 登记条目（不随源码清理）；重迁移 prompt 中附「前次任务知识入口」清单（repro 文件名 + CG gap_id + 校准数字出处），供 Stage 1 设计期直接消费而非 pyc 考古（argmax 实证 2026-09-24：清理致知识流失，恢复成本一整段设计期探针预算）。
    动机: repro/CG 是跨任务知识载体，清理使下一次迁移从零考古；perf_opt/logs 与 repro 的「金矿」价值（校准值互证、失败反例、终版正例）依赖其存活。
- status: pending
- confirmations: -/-
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: -
- decided_note: 清理动作的实际执行方若不在 scaffolder（如 conductor Stage 0 编排或脚本），apply 时改路由至实际 owner；与 VP-2026-0117（终态产物保全）/ VP-2026-0102（迁移重做 git-show 核对）互补三件。

## VP-2026-0147
- type: R
- title: design-review 维度 3/7/8 检视缺口增补——几何特化路径「单元粒度 ↔ num_logical ↔ §3.3 解码」三要素自洽 + num_logical 全文 grep 唯一性 + §3.3 逐 for 对齐 §1.6.2 + 候选采纳结论跨节唯一 + 伪代码切片 rank vs buffer 形状
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：Stage 2 检视三轮复盘（examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/RETROSPECTIVE.md Stage 2 / recheck-1 / recheck-2 的 Skill Flow Issues 与 Value Point Proposals）——本轮 v0/v1 两版检视的通用 Tiling 核对均绕过了 C4b 恒等路径（`(1,8,9,11)` 整平面单元应配 `num_logical=N*num_cblk`，通用式 9 会产生 `n=uid//num_cblk∈0..8` 越界）；`acc` 零初始化漏点靠逐行人工交叉阅读发现；C4/C4b 示意切片与 §4.3 声明 buffer rank 不符（记附注级）；修复面跨 11 处章节靠人工 grep 对齐（REVIEW.md 问题 1/2 + 机械复核段）（任务工作区，未上库，provenance 允许失效）
  - 〔2026-10-14 第五任务独立复现（不同任务，pool 族窗口池化），task max_pool2d-_max_pool2d_kernel-20261014T062320Z〕Stage 2 复检轮两条缺口均落在本条检查面：① **跨章节复述项取值漂移**——§3.2（API 映射表）的 C4 判决 `vcmp(nmax,zero,"gt")` 与 §3.3/§4.4 的 `"eq"` 矛盾（照抄 §3.2 会反转 NaN 判决；`grep -n 'vcmp(.*"gt"\|vcmp(.*"eq"' DESIGN.md` 捕获）；建议在维度 7 通过标准里把「表间复述项取值一致（谓词/比较模式/单位）」单列——仍为建议级，但要求给出 grep 证据（覆盖全部出现处）；② 特化谓词充分条件的依赖变量核对（`out==1×1` 漏 `d==1`）属维度 8 新检查面，另立 queue VP-2026-0177
- repro: repro-missing（检视流程规则；复现条件 = 任一含几何特化/退化路径或候选跨节结论的设计文档走维度 3/7/8 核对）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 3 一处 + 维度 7 一处 + 维度 8 两处）
    定位锚 1（维度 3）: "| 分核策略（物理核数适配） |"
    new 文本 1: 在该行之后插入新行：
      | 几何特化/退化路径分核自洽 | 凡设计含几何特化或退化路径（全局池化、恒等、恒等轴、几何条件分支），须逐路径核对「单元粒度 ↔ `num_logical` 公式 ↔ §3.3 任务解码 `n = uid // …`」三者一致（整平面单元应配 `N*num_cblk` 而非通用式 `N*out_H*num_cblk`——`N=1` 时后者产生越界 `n`）；并对 `num_logical`/`NUM_LOGICAL` 做全文 `grep` 逐处核对取值口径唯一（修复面常跨 §3.3/§4.3/§4.4/§5.2/§5.5/§6.x/§9.2 等 10+ 处） | 特化路径沿用通用 `num_logical` 公式而与整平面单元粒度不符（越界索引风险）；同一分核变量全文出现两种取值口径 |
    定位锚 2（维度 7）: "| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |"
    new 文本 2: 在该行末追加子项：
      | 伪代码切片一致性 | §3.3 伪代码声明的 buffer 形状与其内 `T.copy`/归约切片 rank 须一致（示例：4D 输入切片超 `H_in`、3D 切片喂 2D buffer）；设计显式声明「伪代码为决策示意、局部次序交 Stage 3」时记附注级并在 REVIEW.md 提示 Stage 3，不阻塞 |
    定位锚 3（维度 8，向量化替代完整性行）: "| 向量化替代完整性 | §1.6.2 覆盖方案中**全部**循环 / 标量计算点（与 §3.3 伪代码、§6 循环结构交叉核对，无遗漏）"
    new 文本 3: 在通过标准中追加可操作判据：
      「盘点对象 = §3.3 伪代码逐 `for`/`range`/元素级赋值」——逐条与 §1.6.2 行名、§6.1 表一一对齐，缺失即 fail（高频漏点：累加器零初始化 `for o_w: for c: acc=0`，可整块 `T.clear` 消除）；另核对 §1.6.2 表内标记列与结论段文字/保留计数逐点一致（表/文同点符号相左记建议级）。
    定位锚 4（维度 8，调研→设计一致性行）: "| **调研→设计一致性** |"
    new 文本 4: 在该行末追加子项：
      「候选采纳结论跨节唯一性」——逐候选建立采纳结论表，跨 §1.6.0 R4 行 / §1.6.0 调研结论 / §1.6.1 / §0.6 / §3.3 五处取值须一致（任何分歧即 fail）；采纳与否以 §1.6.1 为唯一决定源，其余章节只引用不重判。
    动机: 本任务两轮修订回退（Stage 2 复检两次不通过、合计 ~1460s 设计修订往返，占会话 33%）的根因即上述四类检视缺口——几何特化路径不在通用 Tiling 核对范围、`num_logical` 跨 11 处人工枚举、候选采纳结论跨 5 节矛盾、`acc` 零初始化不在盘点范围；四项均无机械化支持，靠检视者逐行交叉阅读。
- status: pending
- confirmations: -/-
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: -
- decided_note: 与 VP-2026-0145（设计侧 checklist，P 类）/ VP-2026-0092（复检 grep+md5）/ VP-2026-0028（修订版重审工作流）互补；定位锚为唯一子串，apply 时按锚后插入执行。

## VP-2026-0148
- type: R
- title: op-design Phase 2 补「候选采纳结论单一决定源」+ D-1 等价性判据分档 + §1.6.2 盘点对象可操作判据
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：Stage 1 revision 复盘（examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/RETROSPECTIVE.md Stage 1 design-revision Skill Flow Issues 三行 + Value Point Proposals 第一行）——v0 同一候选（倒数乘 C1）在 §1.6.0 R4（「✅ 采纳」）/ 调研结论 / §1.6.1 C1（「未采纳」）/ §3.3（实际使用）各写不同结论；`acc` 零初始化盘点漏点修复须同步 §1.6.2/§6.1/§3.3/§1.4 四处；C1 采纳后暴露 D-1 判据缺口（`viol==0` 无法覆盖容差内等价，v1 引入 ≤1 ulp 分档）（任务工作区，未上库）
- repro: repro-missing（设计流程规则；复现条件 = 同一候选在多节各写不同采纳结论、或容差内等价项沿用单一 viol==0 判据的设计文档）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 三处）
    定位锚 1（Phase 2 第 2 步）: "2. **向量化替代分析（循环 / 标量消除 → §1.6.2）**：盘点实现方案中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代："
    new 文本 1: 在该行「盘点实现方案中**全部**循环与标量计算点」后追加可操作判据：
      「（盘点对象 = §3.3 伪代码逐 `for`/`range`/元素级赋值，逐条与 §1.6.2 建立一对一映射，并同步 §6.1 表与 §1.4 步骤；累加器零初始化 `for o_w: for c: acc=0` 类标量写属高频漏点，可整块 `T.clear` 消除。）」
    定位锚 2（Phase 2 第 4 步产出与下游约束）: "4. **产出与下游约束**："
    new 文本 2: 在该列表末追加一条：
      - **候选采纳结论的单一决定源**：候选采纳与否以 §1.6.1 为唯一判定处；§1.6.0 R4 行、§1.6.0 调研结论、§0.6、§3.1、§3.3 一律引用（标注「→ §1.6.1 Cx」）而非重判——同一候选跨节结论矛盾会使 D-1 等价性验证范围与「已采纳优化项」不匹配（逐位 vs 容差内）。
    定位锚 3（Phase 2 第 5 步 D-1）: "   - **内容**：每个采纳项的候选式 vs 基线式数值对照——随机张量 + 角点值（±inf/NaN/subnormal/边界 shape）上以 **fp64 参照**比对（torch CPU 实现，两式均以 fp64 计算后按目标 dtype 舍入对照）；SymPy 符号化简可选（仅限可符号化公式）；"
    new 文本 3: 在该 bullet 后插入新 bullet：
      - **判据分档（按采纳项等价强度）**：逐位等价项用 `viol==0`；容差内等价项（如 `sum/d` → `sum·fl(1/d)`）须显式声明判据「舍入到目标 dtype 后 ≤N ulp」并与逐位判据分列（脚本以每 case `kind` 分判据，NaN/finite 不一致恒判失败）——单一 `viol==0` 判据会把容差内等价项误判为 FAIL（adaptive_avg_pool2d 2026-10-13 实证：C1 spp fp16 viol=0 但 bitdiff=15 / max_ulp=1.00）。
    动机: 本任务 v0 的跨节矛盾与盘点漏点各消耗一轮 Stage 2 检视 + Stage 1 修订；容差内等价判据缺口在 v1 采纳 C1 后立即暴露——三处均为设计 skill Phase 2 缺少可操作判据/单一决定源条款所致。
- status: pending
- confirmations: -/-
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: -
- decided_note: 与 VP-2026-0142（判据分档的数据侧）/ VP-2026-0145 / VP-2026-0147 互链。

## VP-2026-0149
- type: R
- title: op-develop Phase 2 补三条实现前置核对——`range` 是 serial TIR 循环（非编译期展开）/ `ceil(C/CH)>1` 通道块尾动态切片 / 退化路径容量判据与回退
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：Stage 3 复盘（examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/RETROSPECTIVE.md Stage 3 Skill Flow Issues 四行）——首轮按 DESIGN §1.6.2 #8/§6.1「`range` 编译期展开」字面实现即 trace 期失败（`list indices must be integers or slices, not Var`，走静态界+守卫修复）；DESIGN §5.2/§5.3 声明「本算子不存在 GEMM 式整除问题」未覆盖通道块尾（C=96 → 64+32，整块搬运即越界）；§4.5 UB 预算只对代表 workload 核算，未覆盖退化路径 `(1,8,32,32)→1×1` 的 ≈393KB（任务工作区，未上库）
- repro: repro-missing（开发流程规则；复现条件 = 源/伪代码含「Python `range` 编译期展开」声明或 `ceil(C/CH)>1` / 几何特化路径的迁移 kernel）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 第 1 步前追加前置核对）
    定位锚: "1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 `@tilelang.jit(target="npuir")` kernel。"
    new 文本: 在该条前插入三条前置核对（生成前执行）：
      ① **`range` 语义核对**：DESIGN/伪代码中凡出现 Python `range`/预计算表索引并声明「编译期展开」，先确认前端语义——本工具链将 `for i in range(n)` 解析为 serial TIR 循环（loop var 为 `tir.Var`），Python 列表索引在 trace 期报 `list indices must be integers or slices, not Var`；等价实现 = 静态上界 `T.serial(max_extent)` + 运行时 `if k < extent` 守卫（见 pattern-library TRAP-tir-range-loop-not-unrolled）。
      ② **通道/维度块尾核对**：凡 `ceil(C/CH)>1`（或任何非整除分块），GM↔UB 两侧用动态 extent（`real_c = T.min(CH, C - cb*CH)`）且仅写回 `[0:real_c]`——UB 未初始化尾 lane 只要不写回即不影响结果。
      ③ **退化路径容量核对**：几何特化/退化路径的驻留缓冲随空间规模放大，实现须带编译期容量判据（容量够才走特化、否则回退一般路径），回退作为纯优化（语义与精度不变）。
    动机: 本任务三条均为「实现层才发现」的形态——首次实现 `range` 字面即失败、块尾与退化容量均为设计文档未覆盖项；前置核对把成本从 Stage 3 现场试错前移到生成前。
- status: pending
- confirmations: -/-
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: -
- decided_note: ①的知识侧数据条目已 Tier 0 合入 traps-compiler.md TRAP-tir-range-loop-not-unrolled（含 repro）；本条为开发 skill 的前置核对流程侧。

## VP-2026-0151
- type: R
- title: add-npu-op Stage 5 前置与 report 时效性口径——`integrate_kernel.py`/report CLI 须先 `source /workspace/env.sh`；`S5-REPORT-STALE` 为 mtime 口径，行为中性 gate 修复的时间戳对账须显式披露
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：examples/adaptive_avg_pool2d/RETROSPECTIVE.md Stage 5 attempt 1/2（Skill Flow Issues 两行 + 第二行 Value Point Proposal）——未 `source /workspace/env.sh` 时 smoke `ModuleNotFoundError: No module named 'tilelang'`（integration 本身已成功）；`S5-REPORT-STALE` 纯 mtime 检查使 report 后任何 wrapper/kernel/`__init__.py` 编辑都判 stale，与 conductor「修 gate 但不重跑正确性/benchmark」冲突；行为中性 edit（加类级字面量）+ 时间戳对账后 report 保持 `status=passed`（任务工作区，未上库）
- repro: repro-missing（集成流程规则；复现条件 = 不 source env.sh 跑 integrate_kernel.py 观察 smoke ModuleNotFoundError；report 后编辑 wrapper 再 `statectl gate 5` 观察 S5-REPORT-STALE）
- toolchain_stamp: tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/SKILL.md
- delta: |
    动作: update（Tier 2 段两处）
    定位锚 1: "#### Tier 2 — Runtime verification (after NPU kernel component rewrites kernels for NPUIR)"
    new 文本 1: 在该小节标题后追加前置条件：
      > **Prerequisite**: `source /workspace/env.sh`（或等效把仓库根加入 `PYTHONPATH`）再运行 `integrate_kernel.py` 与 report CLI——集成 smoke（`cwd=examples/TileOPs`、`sys.executable`）与 report CLI 均需 `tilelang` 可 import；缺少该前置时 smoke 报 `ModuleNotFoundError: No module named 'tilelang'`（环境缺口，非 kernel/胶水缺陷）。
    定位锚 2: "Use the PascalCase manifest key `{op_name}`, not `{op_slug}`."
    new 文本 2: 在该段追加 report 时效性口径：
      **`S5-REPORT-STALE` 口径**：gate 5 以 `run.json` 的 mtime 对比 wrapper / `pkg/__init__.py` / kernel `.py`（`getmtime(run.json) < max(...)` 即 stale）——report 之后对上述文件的任何编辑都会机械地使 report 失效，故 report 宜作为 Stage 5 最后一步；当修复须在 report 之后落地且不允许重跑时，只有行为中性的编辑（如 gate 要求的声明式字面量）能保持 report 有效——此时对账文件时间戳并**显式披露**（不编辑 report、不重跑正确性/benchmark）。
    动机: 两条均为本任务 Stage 5 实际消耗（attempt 1 env 缺口 + attempt 2 gate/时间戳往返 164.9s）；skill 原文只在别处提 `PYTHONPATH`，未给 Stage 5 运行前置与 stale 语义。
- status: pending
- confirmations: -/-
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: -
- decided_note: 与 VP-2026-0150 同属本任务 Stage 5 门禁/集成缺陷集群（涉及文件不同：脚本 vs skill 文档），审批简报时建议同批。

## VP-2026-0164
- type: R
- title: op-design SKILL Phase 2 第 5 步（D-1）补三条——实现次序守卫项（正确次序复现 + 错误次序必被检出 + 最小反证 shape）／守卫负向控制须与被 guard 的伪代码 op 序列对齐／约定无关性检查（结论依赖未文档化语义时两种约定各跑一遍）
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 1 revision v1 + Stage 2 两轮：`DESIGN.md` v0 §3.3/§6.2（`T.vmax(acc)` 先于 `vcmp(gt)`）的公式级 D-1 **6 项全 `EQUIV_PASS` 却完全未覆盖该次序**，v1 新增 `N3-ord`（342 checks；`value_first=True/False` 双向 + `_order_counterexample_ok` 最小反证 `arange(12)→(2,2)` 的 `bin(1,1)` 正确 11 / 错误 6）；重检轮 Reviewer 须**亲读** `_running_first_idx(value_first=True)` 与 design_v0 §3.3 逐行比对才能确认守卫守的是 v0 缺陷；同设计另以 `vmax_is_nan_propagating=False` 分支跑 `N3b` 证「含 NaN 判决由独立掩码路径决定、不依赖 `T.vmax` NaN 约定」（`REVIEW.md#机械复核` / `#维度 8`）
  - 复盘出处：同目录 `RETROSPECTIVE.md` Stage 1（first_design + revision v1）与 Stage 2（不通过 + 重检）的 Skill Flow Issues / Value Point Proposals（任务工作区，未上库，允许失效）
  - 交叉引用：queue VP-2026-0161（知识侧条目）/ VP-2026-0166、VP-2026-0167（检视侧流程）/ VP-2026-0045、VP-2026-0148（D-1 判据分档与对照对象——同节不同锚）
- repro: repro-missing（设计流程规则；复现条件 = 设计含共享累加器/顺序语义向量化、D-1 只有公式对照而无次序守卫项时，走 Phase 2 第 5 步核对）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 5 步「内容」bullet 之后插入三条 bullet）
    定位锚: "   - **内容**：每个采纳项的候选式 vs 基线式数值对照——随机张量 + 角点值（±inf/NaN/subnormal/边界 shape）上以 **fp64 参照**比对（torch CPU 实现，两式均以 fp64 计算后按目标 dtype 舍入对照）；SymPy 符号化简可选（仅限可符号化公式）；"
    new 文本: 在该 bullet 之后插入：
      - **实现次序守卫项（共享状态多次更新时强制）**：凡 §3.3/§6 中共享状态（累加器 / 追踪器）在同一元素上有多次更新，D-1 在公式对照之外还须验证**实现的 op 相对次序**——① 正确次序复现基线；② 错误次序必须被检出（构造「真实值离开窗口首元素」的非退化形状）；③ 硬编码最小反证 shape 作回归。依据：`vmax(acc,v)` 先于 `v > acc` 时判据恒假、输出静默退化为初值，而候选式 vs 基线式的公式对照无法覆盖该次序（实测：v0 公式级 6 项全 PASS 却次序错误；正确次序 342 checks、最小反证 `arange(12)→(2,2)` 的 `bin(1,1)` 正确 11 / 错误 6）。
      - **守卫的负向控制须与被 guard 的形态对齐**：实现次序/状态守卫自建的「错误形态」必须等于它要拦截的 §3.3/§6 op 序列（v0 伪代码），否则守卫可在「守一个不存在的错误」下恒 PASS——产出时在 §1.6.1 注明负向分支对应的伪代码位置。
      - **约定无关性检查（结论依赖未文档化行为时）**：判决依赖某算子未文档化语义（如 `T.vmax`/`T.reduce_max` 的 NaN 传播）时，D-1 以两种约定各跑一遍；两种约定下均与基线逐位一致 ⟹ 结论不依赖该未文档化行为（实测：`N3`/`N3b` 两分支均 `EQUIV_PASS`，证明含 NaN 判决由独立掩码路径决定）。
    动机: 本任务 v0→v1 的设计修订（≈12min）与 Stage 2 首轮不通过（≈9min）都源于 D-1 的验证范围缺口（只验公式不验实现次序、守卫负向控制未与伪代码对齐）；三条均为 D-1 产出物的可操作判据，可在首次设计期零成本补齐。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 定位锚为该 bullet 全行（唯一）；apply 时在其后插入三条 bullet，其余内容不动。与 VP-2026-0148（同节判据分档）锚点不同、内容互补。

## VP-2026-0165
- type: R
- title: op-design SKILL Phase 2 第 1 步（数学等价优化）补「状态机类改写的写入语义判定」——把「顺序条件覆写」改写为「集合取极值」前先判首写生效（`>`）还是末写生效（无条件覆写），再决定取 `min` 还是 `max`
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 1：`DESIGN.md` §1.6.1 N2/N3——首版把 first-hit 当成「值等于最终最大值的**最小**扁平下标」（`vmin`），`verify_equiv.py` N3 首轮 **20426 mismatch**；源的写入语义是严格 `>` 更新（首写生效）⟹ 破纪录集合扁平最大值，改 `vmax` 后 340 例全 `EQUIV_PASS`；`rand` 组在错误方向下 123/144 索引错而 `all_equal`/`zeros`/`all_neg_inf` 全过（任务工作区，未上库）
  - 〔2026-10-14 证据链追加（task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z，pool 族双输出 1D 变体）〕Stage 1 首版设计即按「先判写入语义、再定候选集」执行并写入 §1.6.1：1D 窗口池化 first-hit（严格 `>` 首写生效）= `min{iw : x[iw]==acc}`（**两遍法**：`acc` 定稿后由等式集合取 `min` + `BIG` 哨兵，第二遍与第一遍解耦 ⇒ 消除「判据 vs 累加器更新次序」依赖），last-NaN = `max{iw : isnan(x[iw])}`；`verify_equiv.py` N2N3（8 配置 × 6 kind × 3 dtype 全 `EQUIV_PASS`）+ N2N3-ord（次序颠倒误形态 39/72 检出）。**口径澄清**：单遍式取「破纪录集合的位置最大值」与两遍式取「等式集合的位置最小值」数值等价，但**候选集不同**——判据表述须绑定候选集（混搭即错）；建议插入文本补该澄清，避免后续设计师把 `vmin` 一律当作错误方向〕
  - 复盘出处：同目录 `RETROSPECTIVE.md` Stage 1 Skill Flow Issues 首行（「技能 Phase 2 只要求等价性论证，未强制状态机类改写先写出源的『最后写入语义』再取极值」）
- repro: repro-missing（设计流程规则；复现条件 = 设计把「顺序条件覆写 / 记录」改写为集合取极值时，核对 §1.6.1 是否写明首写/末写判定与所选极值方向）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 1 步引入句之后插入一段强制前置判定）
    定位锚: "1. **数学等价优化（公式级 → §1.6.1）**：对计算公式做恒等变形与结构调整，逐项评估、择优采纳："
    new 文本: 在该句之后插入：
      **状态机类改写的写入语义判定（前置）**：把「顺序条件覆写 / 记录」改写为「集合上取可交换极值」前，先判定源的写入是**首写生效**（严格 `>`，并列取首）还是**末写生效**（无条件覆写，如 NaN 分支覆盖），再决定取 `min` 还是 `max`——如 first-hit（严格 `>`）与「窗口含 NaN 取最后一个 NaN」两种语义在向量化后都取**破纪录 / NaN 位置集合的扁平最大值**（`max`），误取 `min` 会静默给出「窗口内首个局部峰值」的索引（实测：v0 用 `vmin`，D-1 首轮 20426 mismatch，改 `vmax` 后 340 例全 `EQUIV_PASS`）。该判定与所选方向写入 §1.6.1 等价性论证，并在 §1.6.2/§3.3 保持同一口径。
    动机: 该判定缺失使 v0 的索引方向错误只在 D-1 机器验证时暴露（Stage 1 一次修订 ≈12min + 一次 Stage 2 重检）；判定本身两行可写完，属「等价性论证」的必要前提。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 知识侧判据见 queue VP-2026-0160；与 VP-2026-0148（同 Phase 2 不同锚：单一决定源/盘点判据）互补不重叠。

## VP-2026-0166
- type: R
- title: design-review SKILL 维度 8 增「共享累加器的比较-累加先后次序」核对 + 实现次序守卫的负向控制对齐核对
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 2：`REVIEW.md#问题 1`——v0 §3.3/§6.2 把 `T.vmax(acc,...)` 写在 `T.vcmp(...,"gt")` 之前（§1.6.1 N3 的等价性前提是「与更新前的累加器比较」），`gt` 恒假 ⟹ 索引退化为窗口首元素；同设计的 `verify_equiv.py` 验证的是公式（340 例全 PASS）**与 §3.3 的 op 次序脱钩**（verification-implementation gap），只有人工推演才发现；重检轮进一步要求亲读守卫的负向分支确认其等于 v0 §3.3 序列（任务工作区，未上库）
  - 复盘出处：同目录 `RETROSPECTIVE.md` Stage 2（不通过 + 重检）Skill Flow Issues 首行与 Value Point Proposals 首行
  - 交叉引用：queue VP-2026-0161（知识侧）/ VP-2026-0164（设计侧）/ VP-2026-0156（同维度已 apply 的负向断言分级复核——本条锚点不同）
- repro: repro-missing（检视流程规则；复现条件 = 设计把顺序语义改写为共享累加器 + 掩码候选时，核对维度 8 是否检查 §3.3/§6 的比较-累加次序与守卫负向控制）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 检视方式段首句之后插入一段核对项）
    定位锚: "> 检视方式：以 §1.6 为对象逐项核对，并交叉核对 §3.1 / §6 与 §1.6 的双向一致性。"
    new 文本: 在该句之后插入：
      **共享累加器的比较-累加先后次序核对（顺序语义向量化时强制）**：凡把顺序语义（条件覆写 / 记录）改写为「共享累加器 + 掩码候选」的向量式，须核对 §3.3/§6 中**比较与累加的先后次序**是否满足 §1.6.1 的等价性前提——先更新再比较 ⇒ 判据恒假、极值/索引静默退化为初值（实测 2026-10-14）；并检查 §1.6.1 的机器验证是否**覆盖该次序**（公式级全 PASS ≠ 次序正确），以及实现次序守卫的**负向控制建模**是否与其要拦截的历史伪代码 op 序列逐行对齐（守卫须能说明「守的是哪一个 v0 形态」，不得只以「重跑全 PASS」结案）。
    动机: 本任务 Stage 2 首轮不通过的两个阻塞项之一即该次序（另一为退化路径 UB 预算）；维度 8 原口径只核对「§3.1 是否用优化后公式 / §6 是否残留未论证标量循环 / 是否选 A 做 B」，不含共享累加器的 op 相对次序。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 定位锚为维度 8 检视方式段首句（唯一）；与 VP-2026-0156（同维度已 apply，锚为段末负向断言分级复核）互不覆盖。

## VP-2026-0167
- type: R
- title: design-review SKILL Phase 1 第 7 步机械复核补两条——设计正文本的支撑性数字须可由留存脚本机械复现（不可则由 Reviewer 手工复算并记录口径）／实现次序守卫须亲读负向控制建模
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 2 重检轮：`REVIEW.md#机械复核`（`N3-ord` 独立核查段）——设计正文记「守卫非空性证据 9779」而 `verify_equiv.py` 只打印各 item 的 checks/miss/status、不打印可检出元素数，Reviewer 自行复算得 9769（掩码口径差：`oki != seed` vs `(oki != seed) & ~isnan(bval)`），只能手工复算且结论口径不同（任务工作区，未上库）
  - 同轮复盘要求「若设计附实现次序/状态守卫项，须亲读该守卫的负向控制建模，确认其等价于被 guard 的历史错误形态，不得只以『重跑全 PASS』结案」（`RETROSPECTIVE.md` Stage 2 重检 Skill Flow Issues 首行）
  - 交叉引用：queue VP-2026-0049（重跑核对须逐格对上确界 + 行数粒度——同步骤不同锚）
- repro: repro-missing（检视流程规则；复现条件 = 设计正文引用脚本未输出的派生数字，或设计附实现次序/状态守卫项时的 Phase 1 第 7 步核对）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「重跑等价性验证」bullet 之后追加两条 bullet）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    new 文本: 在该 bullet 之后追加：
      - **派生数字可机械复现**：设计正文引用的**支撑性数字**（守卫非空性/覆盖计数、逐案例覆盖计数等）若不由留存脚本输出，Reviewer 须逐一手工复算并在 REVIEW 记录复算口径与差异原因（实测：正文记 9779、脚本未输出该计数、复算 9769，差 10 源于掩码定义口径）——该类数字纳入脚本标准输出后可转为机检项。
      - **守卫负向控制对齐**：设计附实现次序 / 状态守卫项时，须**亲读**该守卫的负向分支（脚本内错误形态建模），确认其等价于被 guard 的历史错误形态（如 v0 §3.3 的 op 序列）；不得只以「重跑全 PASS」结案（守卫可在「守一个不存在的错误」下恒 PASS）。
    动机: 两条均为本任务 Stage 2 重检轮的实际人工成本（一个数字靠手工复算、一个守卫靠亲读代码）；机械复核段原口径为「执行结果须与内嵌表一致且全部 EQUIV_PASS」，未覆盖「数字不由脚本输出」与「守卫守对东西」两种失效。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 定位锚为 Phase 1 第 7 步首条 bullet 全行（唯一）；与 VP-2026-0049（同 bullet 的「一致」粒度口径）互补，apply 时两者按序插入。

## VP-2026-0168
- type: R
- title: op-design SKILL Phase 4 第 4 项（数据规格与内存规划）补「退化/几何特化路径逐 buffer 复算 + 编译期容量判据不等式 + 收窄通道块」硬要求
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 2：`REVIEW.md#问题 2`——v0 §4.5 一般网格行（65.1 KB）复算无误，但 C4 行**低报 46%**（记 ≈90KB / 实算 132.8KB）、C4b 行漏计「模式构造 buffer」（`idx1d/idxp4/idxb4/idxbl` ≈51KB，记 ≈64KB / 实算 112.1KB），使「两档均在容量内」的结论在默认档 ×1.7 下失效（C4 ≈226KB > 192KB）；v1 §4.5 改为逐 buffer 复算表 + 判据不等式 `degenerate_bytes(CH_deg,H,W,dtype) × inflate ≤ 192 KB` + 容量驱动收窄 `CH_deg = min(C,32)`（C4 68 200 B / ×1.7 = 113.2 KB ✓、C4b `CH_deg=8` → 15 048 B / 25.0 KB ✓、`(1,8,32,32)` 实算 ≈340 344 B ≫ 192 KB 回退）（任务工作区，未上库）
  - 同族第三现：hardware-cost-model.md §5（2026-10-14 增补的第三数据点）；机检缺口见 queue VP-2026-0047（第七现：`ub_budget` 对逐 buffer 表 + 编译期分支形态仍 skip）
  - 〔2026-10-14 证据链追加（task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z，pool 族双输出 1D 变体）〕Stage 2 检视 #1 唯一阻塞项即 §4.5 双输出（索引型）1D 容量闭式**分量漏计**：v0 记 `53+2·eb` 而同行分量注释求和为 `69`（漏 `idx32/idx_t` 8 B + `idx64` 8 B）——纯文本即可拦截的自相矛盾；v1 改为 `G = CH·LSPAN·(2·eb+9) + BL·CH·(69+2·eb) + 256` 并逐 buffer 求和 = 闭式值（sincnet fp16 `BL=16` 114 944 B、ecg bf16 `BL=17` 113 792 B，×1.7 均 ≤192 KiB；被否档 `BL=18/20` 130 880/132 800 B ×1.7 超容）。**建议本提案的 1D 形态要求补两条自检**：① 容量闭式括号内分量求和 = 系数（机器可查）；② 档位（`BL`/`CH`）变更后所有派生量（`LSPAN`→`num_oblk`→`num_logical`→`num_local`）与备选档连带重算〕
  - 复盘出处：同目录 `RETROSPECTIVE.md` Stage 1 revision v1 Skill Flow Issues 第二行（「skill Phase 4 未强制特化/退化路径按实际 alloc 清单逐 buffer 复算 + 带字节数的编译期判据不等式」）
- repro: repro-missing（设计流程规则；复现条件 = 设计含几何特化/退化路径而 §4.5 只列代表 workload 容量结论时）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 列表第 4 项扩展）
    定位锚: "4. 数据规格与内存规划"
    new 文本: |
      4. 数据规格与内存规划。**退化 / 几何特化路径的容量核算（含特化路径时强制）**：§4.5 须按 §3.3 的 alloc 清单**逐 buffer 求和**（含常量 / 模式构造 buffer 与 `T.reshape` 视图各自物化的保守口径——低报两种典型形态：只算头部大 buffer、按「不会同时物化」假设省算但未给依据），并写出 `bytes(CH,H,W,dtype) × inflate ≤ 192 KB` 形式的编译期判据（含代表 shape 实算与回退结论）；代表 shape 的容量结论**不覆盖**退化路径。容量不足时为退化路径单独选**收窄的通道块**（如 `CH_deg = min(C,32)`，独立于性能 A/B 的 `CH`）而非直接回退。实测（2026-10-14）：v0 低报 ≈46%，v1 `CH_deg=32` → C4 68 200 B（×1.7 = 113.2 KB ✓）/ C4b `CH_deg=8` → 15 048 B（25.0 KB ✓）/ `(1,8,32,32)` ≈340 344 B ≫ 192 KB → 回退。同族三任务实证与双口径说明见 `.agents/skills/_shared/standards/hardware-cost-model.md` §5。
    动机: 本任务 Stage 2 首轮两个阻塞项之一即退化路径 UB 低报（另一次序项见 VP-2026-0166）；`design_calc_check.py` 的 `ub_budget` 对该表形态持续 skip（VP-2026-0047 已七现），逐 buffer 复算与不等式是当前唯一可执行的拦截口径。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: -
- decided_note: 定位锚「4. 数据规格与内存规划」为 Phase 4 列表项（唯一）；与 VP-2026-0046（同项、锚为该项展开内容）互补，apply 时两者按序处理；知识侧已合入 hardware-cost-model.md §5。

## VP-2026-0169
- type: R
- title: harness 场景 §7.1 Stage 5 session 导出命令随 opencode v2 漂移——`opencode export` 已不存在，等价子命令为 `opencode session export`
- evidence:
  - task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z Stage 5 终态：`examples/adaptive_max_pool2d_with_indices/SESSION_TIMING_ANALYSIS.md` §1 数据来源注——「当前 opencode 为 v2.0.27，harness.md §7.1 记载的 `opencode export <id>` 已不存在，等价子命令为 `opencode session export <id>`；已据此完成导出」；§4 方案 4 记「约 1 min，但影响可复现性——建议同步更新 harness.md §7.1，避免每个 harness 任务重复探测」
  - 目标文件受影响处：`harness.md` §7.1 命令块（`opencode export <session_id> > examples/{op_slug}/full.json`）与同节执行要求第 1/2 条、文件清单表（`full.json` 行）的措辞
  - 交叉引用：queue VP-2026-0151（同场景 Stage 5 前置/时效性口径，目标文件不同）
- repro: repro-missing（文档命令漂移；复现条件 = 在 opencode v2.x 执行 `opencode export <id>` 观察子命令不存在、`opencode session export <id>` 成功）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 opencode v2.0.27 / 2026-10-14
- target_doc: .opencode/agents/conductor-scenarios/harness.md
- delta: |
    动作: update（§7.1 命令块内命令替换；同节同文件另 3 处措辞按同口径一并更新）
    定位锚: "opencode export <session_id> > examples/{op_slug}/full.json"
    old 文本: （即上述定位锚原文）
    new 文本: |
      opencode session export <session_id> > examples/{op_slug}/full.json
    动机: opencode v2.0.27 起命令子命令化（`opencode session export`），`opencode export` 不再存在——harness 任务每个都会重复探测一次（本任务实测消耗约 1 min 且影响步骤可复现性）；同文件另 3 处同口径表述（执行要求第 1/2 条的 `opencode export`、文件清单表 `full.json` 行的「由 `opencode export` 导出」）在 apply 时一并更新（同一处文档命令的完整替换，不改变语义）。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z 2026-10-14
- decided_by: -
- decided_note: 定位锚为 §7.1 命令块内命令行（唯一）；apply 时按 new 文本整行替换，并顺带更新同节同口径的 3 处表述（属同一命令的完整替换，若审批要求严格单点可在 apply 时只替换锚行）。

## VP-2026-0170
- type: R
- title: op-develop SKILL Phase 2 增「同族前作整文件复用」——harness 迁移且 `examples/{op_slug}/` 下有同族/同 op 前作时，以复制前作 kernel 再改增量部分为默认起点（配套 L0 fail-fast）
- evidence:
  - task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z Stage 3：`examples/adaptive_max_pool2d_with_indices/SESSION_TIMING_ANALYSIS.md` §3/§4——Stage 3 单次 developer 调度 **6944.7s（会话墙钟 62.8%）** 用于从零生成 1342 行 kernel + 53 用例并跑完全量分层测试；方案 1 建议「更激进地整文件复用同族前作 `_adaptive_max_pool2d_kernel.py`（同 family/同骨架，仅多一个索引输出），以『复制—改索引路径』替代从零生成」+「L0 编译一通过就立即跑 L0（fail fast），不要等到全部分层测试写完」
  - 同族前作存在性：`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/`（pool 族第 2 个，2026-10-13 迁移，同 Developer 纯 Vector 骨架；C 轴向量轴 / NaN 追踪形态 / 几何特化路径均已实现）
  - 项目基线：`AGENTS.md`「Operator Implementation Baseline（Mandatory）」——优先修改既有算子案例而非从零生成；本任务实际未走该路径
  - 交叉引用：queue VP-2026-0155（conductor 知识预注入扩展至 Stage 3 developer + 同族前作 Transferable Lessons——已 apply，覆盖「教训」面但不含「代码骨架」复用）
- repro: repro-missing（开发流程规则；复现条件 = harness 迁移任务存在同族前作时，观察 Stage 3 是否以复制前作 kernel 为起点）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 列表追加第 4 条）
    定位锚: "3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。"
    new 文本: 在该条之后追加：
      4. **同族/同 op 前作整文件复用（harness 迁移、前作存在时）**：`examples/{op_slug}/` 下存在同族/同 op 的前作任务时，以**复制前作 kernel 文件再改增量部分**（如新增的索引输出路径）为默认起点，替代从零生成——AGENTS.md「优先修改既有算子案例（prefer modifying an existing operator case）」在迁移场景的落地形态。实测（2026-10-14 pool 族第三个算子）：Stage 3 单次调度 6944.7s（会话墙钟 62.8%）用于从零生成 1342 行 kernel + 53 用例；同族前两个算子的骨架、NaN 追踪形态、几何特化路径与分核模板均已存在。配套纪律：**L0 编译一通过即先跑 L0（fail fast）**，通过后再扩展 L1/L2/Boundary，避免在未验证的形态上继续堆砌用例（Phase 4 的执行顺序不变）。
    动机: Stage 3 是会话最大单项成本（62.8%）；同族追加迁移（pool 族已连续三个）是 harness 场景的高频形态，「整文件复用」把从零生成改为增量修改，是零知识风险的提速路径。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z 2026-10-14
- decided_by: -
- decided_note: 定位锚为 Phase 2 第 3 条全行（唯一）；new 文本含 Phase 2 新增一条 + Phase 4 执行顺序的一句纪律（同一处插入）。与 VP-2026-0155（Stage 3 知识预注入，已 apply）互补。

## VP-2026-0171
- type: R
- title: designer 契约补「交付件目录名/文件名逐字核对」——`_with_` 类连接词漏写会白耗一次自纠与 re-dispatch
- evidence:
  - task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z：`examples/adaptive_max_pool2d_with_indices/SESSION_TIMING_ANALYSIS.md` §2 阶段 5「设计交付件目录修正」89.5s（01:25:12–01:26:42，修正误命名目录 `_adaptive_max_pool2d_indices_kernel` → `_adaptive_max_pool2d_with_indices_kernel`，designer 自纠）+ §4 方案 3（建议在调度 prompt 中把目标目录与文件名逐字重复两遍并在设计文档首行自检；或在门禁 S1 增加「算子目录名 == op_name 逐字相等」的机械校验）
  - 交叉引用：queue VP-2026-0047（门禁机检覆盖缺口族）——本条只落 designer 侧契约（机器侧校验作为备选方案记于 decided_note，未成提案）
- repro: repro-missing（设计流程规则；复现条件 = 设计交付件目录名与 conductor 指定的 op/函数名不完全一致时观察是否在写盘前被自检拦住）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-10-14 harness 迁移任务
- target_doc: .opencode/agents/tilelang-op-designer.md
- delta: |
    动作: update（「核心原则 · 输入工件驱动，输出工件落盘」段该 bullet 之后追加子项）
    定位锚: "  - 输出必须写到 conductor 指定的算子目录。"
    new 文本: 在该句之后追加：
      - **路径逐字核对**：写盘前把目标目录名与文件名与 conductor 指定值（`op_requirements` / 调度 prompt 给出的算子名与函数名）**逐字**比对——含 `_with_` 类连接词与下划线位置（实测 2026-10-14：交付件目录曾误写为 `_adaptive_max_pool2d_indices_kernel`〔漏 `with_`〕，自纠 + re-dispatch 耗 89.5s）；可在设计文档首行以「目标目录」自检行明示，便于写盘前一眼核对。
    动机: 纯机械的命名不符会白耗一次自纠与可能的重调度；把核对前移到写盘前（一句自检行）成本近零。
- status: pending
- confirmations: -/-
- created_by: task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z 2026-10-14
- decided_by: -
- decided_note: 定位锚为「核心原则」段内该 bullet（唯一）；机器侧备选方案（门禁 S1 增「交付件目录名 == conductor 指定名」校验，需在 `gate_lint.py` 落地）本次未成提案——如审批倾向工具侧拦截，可在同一批准批次追加工具侧提案。

---

## VP-2026-0182
- type: R
- title: add-npu-op Stage 0 Tier 1 校验增「bench 参数展开冒烟」——`--collect-only` 不构造 bench 参数，标量 workload extras 经 `tuple(...)` 报 `TypeError` 使整段 bench 在 profiling 前 abort、report 降级 `partial`
- evidence:
  - task max_pool1d-max_pool1d-20261014T081737Z（op 级）Stage 5 复盘：`examples/max_pool1d/RETROSPECTIVE.md#Stage-5` Skill Flow Issues 行（benchmark stage 在 Stage 5 edit scope 外，缺陷只能记录不能修）；`examples/max_pool1d/SESSION_TIMING_ANALYSIS.md#§3 方案 3`（建议 Tier 1 增 bench 参数展开冒烟，本次虽不阻断集成但浪费一次完整 report 运行）
  - 缺陷本体：`examples/TileOPs/benchmarks/ops/bench_pool.py:161` `kernel_size = tuple(op_params["kernel_size"])`，而 `examples/TileOPs/tileops/workloads/pool.py` 的 `MaxPool1dWorkload` 以标量 int 提供 extras（`stride`/`padding`/`dilation` 同样包裹，`kernel_size` 先崩）；产物：`examples/TileOPs/reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/run.json` `benchmark_tests.failed=6`（6 例全为本算子）、`status=partial`、`performance_cases=0`；集成侧记录 `.../max_pool1d_kernel/integration_log.md#Issues 1`（含建议修法）
  - 交叉引用：queue VP-2026-0178（Stage 0 golden 可用性自检）/ VP-2026-0157（report per-op rollup 判定口径）/ VP-2026-0151（Stage 5 前置与 report 时效）
- repro: 复现条件——对 scaffolded 1D bench 跑该算子的 bench 用例：6 例在 profiling 前以 `TypeError: 'int' object is not iterable` abort（correctness 不受影响）
  - 〔provenance，允许失效〕任务内复现命令：`cd examples/TileOPs && python -m pytest benchmarks/ops/bench_pool.py -k test_max_pool1d_bench -q`
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/SKILL.md
- delta: |
    动作: update（Stage 0 Tier 1 校验段扩展）
    定位锚: "<Stage 0 Tier 1 结构校验步骤行——collect-only 冒烟>"
    old 文本: （该行原文，apply 时按目标文件实际文本取锚——当前仅 `--collect-only` 级冒烟）
    new 文本: |
      Tier 1 增一步 **bench 参数展开冒烟**（不实跑 profiling）：对 manifest 中每个 workload 走一遍 bench 侧参数构造路径（含 `include_extra` extras 的解包/包裹），任一 workload 抛异常即 Tier 1 失败并修模板——`--collect-only` 只做收集、不执行参数构造，标量 extras 经 `tuple(...)` 的 `TypeError` 会整段 bench 在 profiling 前 abort、使单算子 report 降级 `partial` 并浪费一次完整 Stage 5 report（max_pool1d 2026-10-14 实证：6/6 本算子 bench 例失败）。模板侧对齐：标量 extras 先归一为元组（`(v,) if isinstance(v, int) else tuple(v)`）再包裹，`kernel_size`/`stride`/`padding`/`dilation` 同处理。
    动机: 1D 算子的 scalar workload extras 是脚手架 bench 模板未覆盖的形态；缺陷在 Stage 0 可被零成本拦截，拖到 Stage 5 才暴露则浪费一次完整 report（本任务 ~13 min Stage 5 中的采集段全废、`performance_cases=0`），且 bench 段在 Stage 5 edit scope 外无法自修。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d-max_pool1d-20261014T081737Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0183
- type: R
- title: develop 门禁对设计声明的每条「容量回退」分支做带字节数的可行性核算——回退目标不可行时如实改为「拒绝路径」并记 Issue，不把 L0 计划当契约
- evidence:
  - task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 3 复盘：`examples/max_pool1d/_max_pool1d_kernel/RETROSPECTIVE.md#Stage-3` Skill Flow Issues 首行（develop skill 的测试展开步骤默认接受 DESIGN.md 的 L0 计划，无「设计声明的容量回退是否 UB 可行」的机械核对；该缺陷直到实现期才暴露）+ Value Point Proposals 首行（`repro/EDGE-c4-fallback-infeasible.py` 的不等式实证）
  - 同任务 DESIGN.md §8.3 L0-11 与实现的对立（设计期望「超容回退一般路径」，实现核算得一般路径同样不可行 ⇒ 改如实拒绝）；Stage 2 机械复核对纯 Vector 全 skip（见 VP-2026-0184）使该缺陷无更早拦截位
  - 交叉引用：queue VP-2026-0181（该事实的 D 类条目）/ VP-2026-0184（检视侧兜底）——三案同源，分层落点
- repro: 复现条件——设计文档声明「容量超限 → 回退一般路径」而回退路径字节数 ≥ 特化路径时，实现期按字节数核算即暴露（最小算术形式见 queue VP-2026-0181 的 repro 说明）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 / 测试展开段扩展）
    定位锚: "<Phase 4 分层测试展开段中「按 DESIGN.md 的 L0 计划实现用例」类行>"
    old 文本: （该行原文，apply 时取锚）
    new 文本: |
      对设计声明的每条**「容量回退」分支**（如「特化路径超容 → 回退一般路径」），在实现期用**带字节数的容量不等式**核算回退目标本身是否可行（回退路径逐 buffer 字节求和 × 膨胀系数 vs UB 容量）——不可行时如实改为「拒绝路径」（`ValueError` + 明确报文）并记 Skill Flow Issue，不得静默把 L0 计划当契约（max_pool1d 实证：整线特化超容时一般路径 span 亦为整线且每元素成本更高 ⇒ 无可行路径，设计声明的回退不可达）。
    动机: 设计侧 verify_equiv 只验公式等价、不验容量，检视侧机械复核对纯 Vector 设计大面积 skip ⇒ 「回退分支可达性」在实现期之前没有任何拦截位，缺陷直到 Stage 3 才暴露（一次实现期返工 + 设计文档 L0 计划失准）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0184
- type: R
- title: design-review Phase 1 第 7 步补「纯 Vector 设计的机械复核退化」——design_calc_check 对该类设计三检查全 skip（仅 r3_metrics pass），Stage 2 须人工逐 buffer 复算 UB 预算与分核三要素并写入机械复核段
- evidence:
  - task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 2 复盘：Skill Flow Issues 首行（`python3 .agents/tools/design_calc_check.py --design <DESIGN.md>` → `ub_budget` / `l0c_budget` / `core_split` 三 skip、仅 `r3_metrics` pass）+ Value Point Proposals 首行；REVIEW.md「机械复核」段记录了人工逐 buffer 复算（113 152 / 114 944 B，任务工作区，未上库，provenance 允许失效）
  - 〔2026-10-14 证据链追加（task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z，pool 族双输出 1D 变体）〔第六→第七现〕Stage 2 复审机械复核：`design_calc_check.py --design examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/DESIGN.md` → `ub_budget`/`l0c_budget`/`core_split` **三 skip**、仅 `r3_metrics` pass（纯 Vector 双输出设计，同为「逐 buffer 表 + 闭式不等式」形态）；人工逐 buffer 复算成为唯一拦截位——本任务 Stage 2 **首检**正是靠人工复算拦下闭式系数漏计（`53+2·eb` vs 同行分量注释求和 `69`）。**建议扩展本提案的 delta**：纯 Vector 形态的人工复算口径补两条——① 容量闭式须与 pattern-library 同类记账式交叉核对（本任务以 `layout.md` PL-1.4 单输出版式 `25+2·eb` 为独立佐证判定双输出应为 `69+2·eb`），系数差异逐项解释；② `G×INFLATE` 距判据 < 5% 时须给「再加一个最小 buffer 是否翻转判据」的边界结论（本任务裕度 0.6%，单个 `(BL,CH)` bool 即翻转）〕
  - **修复后仍复发**：VP-2026-0047 的两类表解析扩展已于 2026-10-14 08:10 apply（commit ddf4255）而本任务 08:16 起运行仍三 skip——工具侧修复覆盖不到的表形态仍在，检视侧兜底位独立成立；同族累计第八现（ada / ssd×2 / argmax / pool 族四连）
  - 交叉引用：queue VP-2026-0047（工具侧解析扩展，已 merged）/ VP-2026-0168（设计侧退化路径逐 buffer 复算）/ VP-2026-0136（`--manifest` 真值推导）
- repro: 复现条件——对纯 Vector（无 `block_M×block_N`、无逐 buffer 解析表）的设计跑 `python3 .agents/tools/design_calc_check.py --design <DESIGN.md>`，观察四项检查的 skip 数
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「机械复核」bullet 扩展）
    定位锚: "<Phase 1 第 7 步中机械复核（design_calc_check）相关 bullet>"
    old 文本: （该 bullet 原文，apply 时取锚）
    new 文本: |
      在该 bullet 末尾追加：**纯 Vector 设计的机械复核退化**——`design_calc_check.py` 对无 `block_M×block_N`、无逐 buffer 解析表的设计（纯 Vector / 任务维 Tiling 类）`ub_budget`/`l0c_budget`/`core_split` 普遍 skip（仅 `r3_metrics` 生效），`skip` 不得当 `pass`：此类设计须**人工逐 buffer 求和复算 UB 预算**（`Σ 逐 buffer 字节 × 膨胀系数 ≤ 192 KiB`，给出「逐 buffer 原始字节 + 膨胀后字节 + 统一 KiB」小表）并按单元粒度人工复算分核三要素（`num_logical` / `num_kernels = min(logical, 实查核数×2)` / `num_local`，核数须设备实查），复算过程写入 REVIEW.md 机械复核段（pool 族任务实证：正是该人工复算拦下退化路径 UB 预算低报）。
    动机: 纯 Vector 迁移是池化/归约/逐元素族的常态，而机械检查对这类设计的可机检覆盖率接近零；不写死「人工复算 + 复算表形态」，机械复核段形同虚设（池化族四连全 skip，含 apply 修复后仍 skip 的形态）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0047（工具侧表解析，已 merged 但本任务仍 skip——支持本条按检视侧独立兜底纳入审批）互补。

## VP-2026-0185
- type: R
- title: op-design 的窗口/池化候选清单补两个维度——① 空间轴长超 UB 驻留预算时「沿该轴分块 + 连续 span 载入」是必需维度（非可选优化）；② 同族前作结论须标注轴数前提（整轴驻留 / 守卫外提 / `BH=1` 最优均为 ≥2 空间轴结论）
- evidence:
  - task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 1 复盘：Skill Flow Issues 首行（skill 的「窗口/池化类」候选清单只列「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」两类轴向候选，未提容量维度；2D 骨架「照搬」风险）+ 第三行（1D 与 2D 的「守卫外提」不可平移，skill 的同族前作教训未标注轴数前提，照搬会写出不存在的优化）
  - 〔2026-10-14 证据链追加（task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z，pool 族双输出 1D 变体）〕Stage 1 Skill Flow Issues 首行：**索引型（双输出）算子不得照搬单输出变体的退化特化**——单输出可采纳的整线归约（C4）在双输出上需额外构造整线 argmax 索引，其自然实现依赖 `T.arange` 派生 pattern 在 `vmax/vselect/reduce` 链内消费（`TRAP-npuir-arange-pattern-with-whole-plane-reduce` 的已验证断裂链）；**判据 = 特化的每一路输出都必须独立可证，值半成立不代表索引半成立**，无法证明时走一般路径（本任务 C4/C4b 均记录为不可采纳）。**建议把该维度写入本提案候选清单**：长空间轴维度之外增「索引构造是否独立于 arange 派生 pattern 链」前置核对（窗口/池化族 + argmax/argreduce 族同适用）〕
  - 同任务 DESIGN.md §0.6 R1（为何必须沿 L 分块）/ §1.6.1（1D-2D 守卫差异）；RETROSPECTIVE Stage 1 Transferable Lessons 末条（「照搬前先问该结论依赖几个空间轴 / 轴长上限多少」）
  - 交叉引用：queue VP-2026-0180（该结构的 P 类条目——数据/模式类落点）/ VP-2026-0147（设计盘点漏点族）
- repro: repro-missing（设计流程规则；复现条件 = 窗口/池化类设计的目标空间轴长超出 UB 驻留预算时观察候选清单是否提示分块维度）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（Phase 2 第 3 项「窗口 / 池化类」清单行 + 同族结论前提注）
    定位锚: "<Phase 2 第 3 项窗口/池化类候选清单行>"
    old 文本: （该行原文，apply 时取锚）
    new 文本: |
      在候选清单追加一行候选维度：**「沿空间轴分块 + 连续 span 载入（钳制起点 / 静态长度）」**——当空间轴长超出 UB 驻留预算（`轴长 × CH × 元素宽度 × 膨胀系数 > 192 KiB`）时这是**必需维度而非可选优化**（逐元素 gather 不成立；span 载入 + 起点钳制保证恒在界内且长度静态）；同时在同族前作结论引用处标注**轴数前提**——「整轴驻留」（需轴长可驻留）、「守卫外提」（需 ≥2 个空间轴且判据只依赖该方向循环变量）、「`BH=1` 最优」（2D 专属，1D 应取容量允许的最大 `BL`）三类结论在轴数/轴长上限变化时均需重新推导。
    动机: 池化族从 2D 迁到 1D 时，照搬 2D 骨架会得到错误/溢出的 kernel（2D `W_in ≤ 112` 整行驻留 vs 1D `L_in` 可达 4096），且同族教训缺轴数前提会导致写出不存在的「外提」优化——两条缺口都在设计期零成本可拦。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0186
- type: R
- title: kb_stale_check 增 `--filter <family|id 子串>` 输出 stale 子集——全库 stale 达 166 条时人工筛选本族条目额外耗时
- evidence:
  - task max_pool1d-_max_pool1d_kernel-20261014T082309Z：`examples/max_pool1d/SESSION_TIMING_ANALYSIS.md#§3 方案 4`（本次 `kb_stale_check` 报 166 条 stale，人工筛选 pool/常数/陷阱相关子集需额外时间；建议按 family 聚合 stale 视图或 `--filter <family>` 直出子集。provenance，允许失效）
  - 工具现状（本蒸馏会话 grep 实证）：`.agents/tools/kb_stale_check.py` 的 argparse 仅 `--include-unknown` / `--json`；`--json` 后仍须调用方自行按 family facets 过滤
  - 交叉引用：queue VP-2026-0116（kb_stale_check stamp 由 repo HEAD 改源码树过滤，已 merged——166 为修复后仍 stale 的真实条目）；queue VP-2026-0081（采数管线族）
- repro: 复现条件——`python3 .agents/tools/kb_stale_check.py` 输出全库 stale 列表（当前无 family 过滤选项）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ Ascend910B3 / 2026-10-14
- target_doc: .agents/tools/kb_stale_check.py
- delta: |
    动作: 新增 `--filter <pat>` 参数（对 stale 条目的 id 与 front-matter family facets 做子串匹配，可与 `--json` 组合；缺省行为不变）
    动机: stale 清单的使用场景是「本次任务相关条目是否需要重验」——与 `kb_search` 的 query 检索同源；全库 166 条下按 family 直出子集可把逐条筛选变成一次调用（蒸馏期与任务内预注入两处消费）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0187
- type: R
- title: kb_lint 的 KB-QUEUE-REPRO 检查因 `text.split("## Decided")` 子串切分被头部迁移注记短路而**从未生效**——ED-A 队列 repro 解耦机械门禁实际处于死态
- evidence:
  - 机械复现（本蒸馏会话，task max_pool1d-_max_pool1d_kernel-20261014T082309Z）：`python3 -c` 读 queue.md 后 `text.split("## Decided")[0]` 长度 **1938 B**、其中 `"## VP-" in pre == False`——首个子串出现位置在文件头部「条目迁移」注记行（第 36/38 行，形如「完整记录移至下方 \`## Decided\` 归档区。」），故 `pending_region` 只含头部说明、`re.finditer(r"^## (VP-\d{4}-\d{4})\s*$", pending_region)` 永不命中 → 该检查对任何 pending 条目都不执行（0 failures 假阴性；全库 5 处 `## Decided` 子串中仅 1 处在行首）
  - 代码位置：`.agents/tools/kb_lint.py` `lint_queue()` 首段（`pending_region = text.split("## Decided")[0]`）
  - 影响面实证：本次蒸馏新增的 pending 条目中两条的 `- repro:` 行原含任务工作区命令（`examples/...`），kb_lint 仍报 0 failures——检查已按 ED-A 规则手工整改，但机械门禁未兜底；既有 pending 条目的 repro 字段同理从未被机械校验
  - 交叉引用：queue VP-2026-0116（kb_stale_check stamp 过滤，同类「机械校验工具行为」条目）/ VP-2026-0047（机检覆盖缺口族）
- repro: 复现条件——读 `queue.md` 并断言 `"## VP-" not in text.split("## Decided")[0]`（当前为真 ⇒ 检查域不含任何条目）；或向任一新 pending 条目写入含 `examples/` 的 `- repro:` 行后跑 `python3 .agents/tools/kb_lint.py` 观察仍 0 failures
- toolchain_stamp: 流程规则/工具缺陷（无运行时依赖）；证据环境 2026-10-14 蒸馏会话（kb_lint.py 现行版本，commit ddf4255 工作树）
- target_doc: .agents/tools/kb_lint.py
- delta: |
    动作: update（`lint_queue()` 的检查域切分改为行首匹配，并让迁移注记不再短路）
    定位锚: |
      (lint_queue 函数体首段)
          pending_region = text.split("## Decided")[0]
    old 文本: （即上述定位锚原文）
    new 文本: |
      # Split on a line-start heading only (migration notes in the header
      # quote the archive heading inline, which used to truncate the pending
      # region to the file header and silently disable every queue check).
      pending_region = re.split(r"(?m)^## Decided\b", text)[0]
    动机: 该检查是 ED-A「pending 条目 repro 字段不得是任务工作区命令」的唯一机械门禁；因注入注记引用归档标题而静默失效，等于队列侧解耦规则全靠人工——机械门禁须先自证生效（可加一条自检：pending_region 中至少应出现 1 个 `## VP-` 标题，否则报 fail）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0188
- type: R
- title: op-develop Phase 2 补「实现侧 `_plan` 须逐字复现冻结设计的容量闭式」——不得以「更精确的实现记账」替换设计闭式（否则档位与分核表静默漂移），并给出「plan 输出 vs 设计 §5.x 表」逐字段自查
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 3 Skill Flow Issues 首行 + Value Point Proposals 第二行：首版 `_general_bytes` 按**实现实际 buffer 清单**精确记账（索引型每 `BL·CH` 元素 `53+2·eb`，UB 方向安全）⇒ `BL` 选到更大值，`sincnet BL=18/LSPAN=54/num_oblk=76/num_logical=4 864/num_local=122`（设计 `16/48/86/5 504/138`）、`ecg BL=20/LSPAN=47/num_oblk=52/num_logical=1 664/num_local=42`（设计 `17/41/61/1 952/49`）；修正为逐字复现设计闭式 `G = CH·LSPAN·(2eb+9) + BL·CH·(69+2eb) + 256` 后 plan 与 `DESIGN.md#§5.2`/`#§5.5` **逐字段一致**，L0/L1/L2/Boundary 全过（任务工作区，未上库，provenance 允许失效）
  - 同任务 `RETROSPECTIVE.md` Stage 3 Transferable Lessons 首条（「实现侧的 `_plan` 必须复现冻结设计的容量闭式，而不是更精确的实现记账」）；漂移后果：不影响精度，但 §5.2/§5.5 分核表与实现永久不一致，Stage 4/5 的对照基准失效
- repro: repro-missing（开发流程规则；复现条件 = 设计侧给出容量闭式与 `CH`/`BL` 档位表，而实现侧按实际 buffer 清单自行记账后输出 plan，与设计表逐字段比对）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 第 1 步前追加一条前置核对）
    定位锚: "1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 `@tilelang.jit(target=\"npuir\")` kernel。"
    new 文本: 在该条前插入：
      ⓪ **分块常数须复现冻结设计（前置）**：1D/2D 分块的具体常数（容量闭式系数、`CH`/`BL` 选型规则与档位表）属**冻结设计的一部分**——实现侧的分块计划（`_plan`/`_choose_bl`/`_choose_ch_bl` 及并联的 `num_oblk`/`num_logical`/`num_local`）须**逐字复现设计闭式**，不得以「更精确的实现记账」（按实现实际 buffer 清单精确求字节）替换之：实现实际占用通常**小于**设计的保守闭式（方向安全），但按实际记账会选出更大的 `BL`/`CH`，使分核表与 `DESIGN.md#§5.x` 永久漂移、Stage 4/5 对照基准失效（max_pool1d_with_indices 实证：同一 shape 两版 plan 的 `num_logical` 为 4 864 vs 5 504；改回逐字复现闭式后逐字段一致）。自查 = 打印实现 `_plan` 输出与设计 `§5.2`/`§5.5` 表逐字段比对。
    动机: 本次漂移不影响精度、门禁全过，属「静默」类不一致（Stage 4/5 才会暴露）；一条打印级自查即可闭环，成本≈0。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0149（develop 三条前置核对）/ VP-2026-0183（容量回退分支核算）相邻但锚点与主题不同（本条管「设计闭式的单一定义源」）；定位锚为 Phase 2 第 1 步原文，apply 时按当时文件锁定精确锚。
## VP-2026-0189
- type: R
- title: op-design 等价性脚本产出纪律两条——① D-1 脚本按条件分支采样（禁「全配置矩阵 × 全线公式」全枚举）+ 单命令时限 + 先落 DESIGN.md；② 语义要素实测表须由脚本 `semantics_evidence()` 产出并逐行落 §0.1（与 §1.6.1 公式等价表分开）
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 1 Skill Flow Issues 第二/三行 + `SESSION_TIMING_ANALYSIS.md` §3/§4：首次设计的 `verify_equiv.py` 对 C4/C4b 全枚举（`whole_line_formula` 对 `L_in=4096` 逐 `ow×t` 循环）跑 ~40 min 被取消，占本任务总墙钟 **40.3%**（3 775.4 s / 9 366.4 s）；用户补充 S3 后改按 `(k,s,p,d,ceil)` 分支采样（每分支 ≥1 代表组）+ 特殊化子句 value-only 少量论证，脚本降至 ≈38.5 s / 1 074 项检查（其后两次重跑 12.2 s / 84 s，随宿主负载漂移）
  - 同任务 `DESIGN.md#§0.1`（11 行 torch 语义实测表，由 `verify_equiv.py#semantics_evidence()` 产出）与 `#§1.6.1`（公式等价 item 表）分列；`RETROSPECTIVE.md` Stage 1 revision「D-1 措辞须与脚本实现一致」教训
  - 交叉引用：VP-2026-0132（D-3 探针纪律三条——探针预算口径；本条管 D-1 等价脚本的规模与耗时口径）/ VP-2026-0164（D-1 内容判据三条）——三者同节不同锚
- repro: repro-missing（设计流程规则；复现条件 = 长轴算子（`L_in` 达数千）的 D-1 脚本若对全配置矩阵跑 `O(out_l × L_in)` 的全线公式，单配置即数亿次 Python 迭代）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 5 步 D-1 与迁移解读段各追加一条）
    定位锚 1: "   - **内容**：每个采纳项的候选式 vs 基线式数值对照——随机张量 + 角点值（±inf/NaN/subnormal/边界 shape）上以 **fp64 参照**比对（torch CPU 实现，两式均以 fp64 计算后按目标 dtype 舍入对照）；SymPy 符号化简可选（仅限可符号化公式）；"
    new 文本 1: 在该 bullet 之后（或在 Step D-1 的「脚本规模」说明处）追加：
      - **脚本规模与耗时纪律**：等价性脚本的 `(k,s,p,d,ceil)` 空间按**条件分支采样**（每分支 ≥1 代表组；`padding>0` / `dilation>1` / `ceil_mode` / `stride≠kernel` / 长度不整除 各至少一例），**禁止对全配置矩阵 × 全线公式做全枚举**——长轴算子（`L_in` 达数千）的 `O(out_l × L_in)` 全线 item（如 `for ow: for t: fmax`）单个配置即数亿次 Python 迭代（实测 ~40 min 未完成，占任务墙钟 40%）；长轴配置只进轻量 item（下标记/结构断言），整线/恒等特化未采用时只做少量 value-only 数学论证。**先落 DESIGN.md 再跑长验证**，且任何单条命令的墙钟设硬上限（超限即拆分）。
      - **产物的可复现描述用不变量而非秒数**：脚本规模以「item 组数 + 检查项数 + 分支采样配置数」描述（同一脚本实测耗时随宿主负载在 12–84 s 间漂移）。
    定位锚 2: "**迁移解读**"（`migration-analysis.md` 承载的迁移解读段引入处；apply 时按当时文件锁定精确锚）
    new 文本 2: 在该段追加一句：
      **语义要素须脚本实测并落 §0.1**：索引编码 / tie 方向 / NaN 行为 / 越界跳过 / padding 退化窗等语义要素须以 `semantics_evidence()` 类函数在 torch 上逐个实测，并**逐行**写入 `DESIGN.md#§0.1`（与 §1.6.1 的公式等价表分列——前者是语义定标、后者是公式对照）。
    动机: 首次设计的长跑是本任务最大单点墙钟来源（40.3%）；且语义要素与公式等价混列会降低两类证据的可核性。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 定位锚 1 与 VP-2026-0148/0164 同节不同锚（0148 判据分档、0164 内容判据，本条管规模/耗时与产物描述口径）；定位锚 2 为 migration-analysis 承载段（apply 时锁定）。
## VP-2026-0190
- type: R
- title: design_calc_check.py 增「容量闭式自洽性」纯文本检查——闭式括号内分量求和 ≠ 系数即 fail；1D 形态或于 skip detail 固定附「须人工按同节 buffer 清单重算并对照 pattern-library 同类记账式」
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 2 Skill Flow Issues 首行 + Stage 1 revision Skill Flow Issues 首行（**同任务两次独立提出**）：`design_calc_check.py` 在 1D 池化「逐 buffer 表 + 闭式不等式」形态下 `ub_budget`/`l0c_budget`/`core_split` 三项 skip（仅 `r3_metrics` pass），容量类算术错误的机械拦截位实质失效；本任务 v0 的 `53+2·eb` 与同行分量注释求和 `69` 自相矛盾——**纯文本即可拦截**却漏过，只能靠 Stage 2 人工逐分量手算发现
  - 同任务 `REVIEW.md#问题 1`（含逐 buffer 求和复算表）+ `DESIGN.md#§4.5`（闭式与分量注释并存）
  - 交叉引用：VP-2026-0047（已合入：补两类表格形态解析——本条是其**未覆盖的纯文本自洽子形态**，与 VP-2026-0184（design-review 侧人工复算口径）互补：本条是工具侧机械拦截
- repro: repro-missing（工具规则；复现条件 = 任一含「闭式 + 括号内分量注释」的 §4.5 且分量和 ≠ 系数——`python3 .agents/tools/design_calc_check.py --design <path>` 应报 fail）
- toolchain_stamp: design_calc_check.py 现行版本；失效环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（检查器集合追加一项 + skip detail 文本）
    定位锚: "UB_INFLATION" （现文件内 UB 预算相关常量/检查函数定义处；apply 时按当时文件锁定精确锚）
    new 文本: 新增 `capacity_closed_form_self_consistency` 检查（纯正则/文本，无需解析 buffer 表）：扫描设计文本中形如 `... = <符号表达式> + <术语>·(A + B·eb) + ...` 的容量闭式，把 `(A + B·eb)` 中的常数分量 `A` 与该条闭式旁注的分量逐一求和值比对，不一致即 `fail`（本任务 v0 实证：闭式系数 `53` vs 分量注释求和 `69`，后者正确）。同时在 `ub_budget`/`core_split` 的 `skip` detail 末尾固定追加：**「1D/纯 Vector 形态暂不支持：须人工按同节 buffer 清单逐项求和重算，并对照 pattern-library 同类记账式（layout.md PL-1.4 等）」**——`skip` 不得被当作 `pass`。
    动机: 本次唯一阻塞项是「纯文本即可拦截」的自相矛盾（分量和 ≠ 系数），而同一检查器已能读到该闭式文本；补一条正则级检查成本极低、拦截面覆盖全部 1D/纯 Vector 设计（该族已连续多任务三项全 skip）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0047（表格形态解析，已 apply）/ VP-2026-0136（--manifest 真值推导）/ VP-2026-0184（design-review 人工复算口径）同工具族不同锚；建议与 0184 同批审批。
## VP-2026-0191
- type: R
- title: design-review 维度 0 增「域不变量断言」复核项——凡 §0.x 出现「某参数条件 ⇒ 某集合关系（非空/全覆盖/退化）」的断言，须独立推导真实充分条件并枚举反例（附反例规模）；正例覆盖不足与断言过强同为 fail
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 2 Skill Flow Issues 第三行 + `REVIEW.md#问题 2`：维度 0 的 7 个检查项全为「源语句/章节一致性」类，**无域不变量复核项**——§0.1「`p ≤ ⌊k/2⌋` ⇒ 每个窗口至少含 1 个有效元素」被独立枚举参数空间推翻（`L_in ≤ 13, k ≤ 6, s ≤ 5, p ≤ ⌊k/2⌋, d ≤ 6, ceil ∈ {F,T}` 得 **89 组** `(config, ow)` 反例；最小例 `L_in=1,k=2,s=1,p=1,d=2`，真实充分条件为 `d==1 ∧ p≤⌊k/2⌋ ∧ s>0`）（任务工作区，未上库，provenance 允许失效）
  - 同族复发型：`max_pool2d` 任务同族缺陷（整平面归约充分条件漏 `d`，VP-2026-0172/0177）与 `adaptive_*` 任务的特化条件措辞，均属「参数约束 ⇒ 集合关系」断言类——跨任务已三现，检视侧无对应检查项
  - 交叉引用：VP-2026-0177（维度 8 编译期特化谓词充分条件 + 子句隔离覆盖——锚点在维度 8，本条在维度 0 的域不变量断言）/ algorithm-candidates.md 窗口池化域条件①③
- repro: repro-missing（检视流程规则；复现条件 = 设计 §0.x 含「参数条件 ⇒ 集合关系」断言时，按本项独立推导充分条件并枚举短 shape 反例）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 0 检查项表追加一行）
    定位锚: 维度 0 检查项表的末行（迁移任务「源算子理解」类检查项末行；apply 时按当时文件锁定精确锚）
    new 文本: 追加一行：
      | 域不变量断言 | §0.x 中凡出现「某参数条件 ⇒ 某集合关系（非空 / 全覆盖 / 退化 / 两集合相等）」的断言，须**独立推导其真实充分条件**并**枚举反例**（附反例脚本或手工枚举规模与参数空间）；断言过强（存在反例）与正例覆盖不足（未证明充分性）同为 fail。典型形态：池化窗口「非空」的真实条件是 `d==1 ∧ p≤⌊k/2⌋ ∧ s>0`（`d≥2` 下有 89 组反例）；整平面归约的充分条件须含 `d==1`。引用来源不能替代独立推导。 |
    动机: 本任务 §0.1 的核心边界断言错误由检视者在无专属检查项下自发枚举才发现（一次设计修订 + 一次复审 ≈14% 墙钟）；同族该形态已三现，属检查项缺口而非个案。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0177（维度 8 谓词充分条件）/ VP-2026-0168（op-design 逐 buffer 复算）同族互补、锚点分属不同维度。
## VP-2026-0192
- type: R
- title: op-design 迁移解读段补「参数约束 ⇒ 集合关系」的真实充分条件推导 + 空窗口类退化 regime 的契约（含「退化 regime 不得拿 torch 当 oracle」）+ 源引用须指向本算子行号
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 1 revision Skill Flow Issues 第三行 + `DESIGN.md` v0 §0.1（第 109–110 行「`p≤⌊k/2⌋` ⇒ 窗口非空（无需兜底）」）vs v1 §0.1（逐条件非空窗口保证 + 空窗口契约 `max_idx==BIG ∧ ¬has_nan ⇒ idx=0` 对齐源 `(-inf,0)`）（任务工作区，未上库）
  - 同任务 Stage 1 revision Transferable Lessons：「参数约束不蕴含窗口非空；`d≥2` 时窗口输入覆盖集合是等差子集，须逐窗口判空」「退化 regime 不得拿 torch 当 oracle（空窗口下 torch 返回越界索引）」「引文须指向本算子行号（`§0.4`/`§1.6.0` 曾引用同族单输出工厂第 15–68 行而非本算子第 212–288 行）」
  - 交叉引用：VP-2026-0185（候选清单维度）/ VP-2026-0172（窗口池化退化特化充分条件，已合入 algorithm-candidates）/ queue VP-2026-0159（索引契约）；知识侧落点见 traps-runtime.md TRAP-torch-oracle-maxpool1d-empty-window
- repro: repro-missing（设计流程规则；复现条件 = 迁移设计的源算子含 padding/dilation 且 §0.1 以参数约束推断集合性质时）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/references/migration-analysis.md
- delta: |
    动作: update（迁移解读段追加三条）
    定位锚: 迁移解读段「域/边界语义」相关条目（apply 时按当时文件锁定精确锚）
    new 文本: 追加：
      ① **集合关系断言须逐条件推导**：凡由 shape/参数约束推出集合关系（窗口非空 / 覆盖集合 = 全轴 / 退化恒等），须写出**真实充分条件**并逐子句隔离验证——窗口池化的「非空」条件是 `d==1 ∧ p≤⌊k/2⌋ ∧ s>0`（`d≥2` 时覆盖集合为等差子集 `{o·s−p+k·d}`，反例族存在），不得默认「无需兜底」；② **退化 regime 的契约与 oracle 分野**：须显式写出退化域（空窗口 / 全 padding 窗口 / 越界 lane）的索引/值契约（对齐源初值 vs 对齐 torch），并标注该域是否可用 torch 作 oracle——窗口池化空窗口域 torch 返回**越界索引**（见 pattern-library traps-runtime.md TRAP-torch-oracle-maxpool1d-empty-window），此类用例只能断言与源对齐的文档化行为；③ **引文指向本算子**：引用源机制时指向本算子迁移函数的行号，而非同族其它变体/工厂（语义可同族同构，引文指向会误导读者）。
    动机: 三条均为本任务 Stage 1 修订 v1 的实际修复内容（v0 缺 ① 致 Stage 2 阻塞、② 的 oracle 分野由 `empty_window_evidence()` 逐行留证才闭合、③ 曾被复审记为瑕疵）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0189（同文件/同段，锚点不同：0189 管语义实测表落 §0.1）互补不重叠；知识侧落点 = TRAP-torch-oracle-maxpool1d-empty-window + TRAP-pool-index-contract-dual-output。
## VP-2026-0193
- type: R
- title: add-npu-op reference 模板——纯选择类算子（max/min/argmax/排序，无累加）的 NPU golden 不可用判据不限于「受限参数」，多输出/`return_indices` 后端支持须先实测；纯选择类默认直接走 CPU golden
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z op 级 `RETROSPECTIVE.md` Stage 5 Skill Flow Issues 第二行：脚手架按「受限参数（`dilation != 1`）」预埋 CPU golden 回退，但本任务实测 `F.max_pool1d(..., return_indices=True)` 的 indices 损坏**对所有 dilation 成立**（`dilation==1` 静默返回 int8 mask buffer）⇒ 按受限参数预埋的回退覆盖面不足，首轮 report 失败（per-op 40/47）
  - 同任务集成包 `integration_log.md#调试历史` + `examples/TileOPs/tests/ops/test_pool.py#MaxPool1dIndicesTest.ref_program`（原 `needs_cpu_golden = dilation != 1` → 改为无条件 CPU golden；修复后 per-op 47/47、`max_abs_err=0.0`）（任务工作区/集成包，未上库）
  - 交叉引用：VP-2026-0178（已合入：Stage 0/S5 golden 可用性自检 + 受限参数回退模板——本条是其覆盖面扩展）/ VP-2026-0179（已合入：integrator 失败分类表子类）；知识侧 = traps-runtime.md TRAP-npuir-ref-backend-golden-capability 第三形态
- repro: repro-missing（集成流程规则；复现条件 = 纯选择类算子用 `torch.nn.functional` 多输出 API 作 NPU golden，先实测其 indices 的 dtype/shape 是否等于契约）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/SKILL.md
- delta: |
    动作: update（reference 模板/自检段扩展）
    定位锚: Stage 0/S5「NPU 参考实现（golden）可用性自检」段（VP-2026-0178 apply 后的现存段；apply 时按当时文件锁定精确锚）
    new 文本: 在该段末尾追加：**纯选择类算子（max/min/argmax/排序，无累加）的默认 golden 路径 = CPU golden**——按「受限参数」分支预埋回退不足以覆盖「全参数区间静默损坏」类缺口（实测：torch_npu 的 `F.max_pool1d(..., return_indices=True)` 在 `dilation == 1` 下**不抛异常**却返回 dtype/shape 全错的 indices（`int8 (N,C,k,L_pad)` 内部 mask buffer），`dilation != 1` 才抛 `AclNN_Parameter_Error`）；自检须核对 golden 输出**每个**返回值的 `dtype`/`shape` 是否等于算子契约（「跑通」不构成判据），多输出/`return_indices` 类后端支持一律先实测再定回退范围。含累加算子（sum/avg/conv/gemm）不得照搬该默认。
    动机: 本任务首轮集成失败 100% 源于「按受限参数预埋」的覆盖面假设；把默认值前置到 reference 模板可消除同类缺口的整轮集成返工（≈17 min/次，见 VP-2026-0178）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 承接已合入的 VP-2026-0178（自检机制）与 VP-2026-0179（失败分类），本条只改「默认 golden 路径与覆盖面」；apply 时须先读文件确认 0178 落盘形态再定位锚。
## VP-2026-0194
- type: R
- title: integrator 失败分类表「NPU reference 后端能力缺口」行的识别信号扩为「异常 `AclNN_*_Error` **或** 后端返回 tensor 的 dtype/shape 与算子契约不符（静默损坏）」
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z op 级 `RETROSPECTIVE.md` Stage 5 Skill Flow Issues 首行：现行子类的识别信号只写「异常 + `AclNN_*_Error`（op 未执行、堆栈落在 `torch.nn.functional`）」——本次后端**不抛异常、照常返回**，indices 为 `torch.int8 (N,C,k,L_pad)` 垃圾 buffer，错误直到测试 harness 的 tensor 比较处以 `RuntimeError: size mismatch` 浮出（形似 shape/runtime 失败）；仅按「异常信号」判类会把该形态误归 shape/runtime 类而漏掉 CPU-golden 修复（任务工作区/集成包 `integration_log.md#调试历史`）
  - 同任务修复后 `run.json` 通过轮（per-op 47/47、`max_abs_err=0.0`、benchmark 36/36）
  - 交叉引用：VP-2026-0179（已合入：该子类行本体，用户批准 2026-10-14 apply 落盘于 `.opencode/agents/tilelang-op-integrator.md`）——本条是对该行的**识别信号扩展**
- repro: repro-missing（集成流程规则；复现条件 = 用 torch 多输出 API 作 NPU golden，后端返回 dtype/shape 与契约不符的 tensor 而 op 未被执行/未被判错）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（失败分类表「NPU reference 后端能力缺口」行的「识别信号」列扩展）
    定位锚: VP-2026-0179 落盘后的子类行（含「异常在 ref_program 内、op 尚未执行且报 `AclNN_*_Error`」的单元格；apply 时按当时文件锁定精确锚）
    old 文本: 异常出现在 `ref_program` 内、**op 尚未执行**（堆栈落在 `torch.nn.functional`）且报 `AclNN_*_Error`（如 `aclnnMaxPool2dWithMask` 仅支持 `dilation == 1`）
    new 文本: 异常出现在 `ref_program` 内、**op 尚未执行**（堆栈落在 `torch.nn.functional`）且报 `AclNN_*_Error`；**或** reference 后端**不抛异常、照常返回**但返回值与算子契约不符（dtype 或 shape 错，如 `F.max_pool1d(..., return_indices=True)` 在 NPU 上返回 `int8 (N,C,kernel_size,L_pad)` 内部 mask buffer 而非 `int64 (N,C,L_out)`）——后者错误会延迟到测试 harness 的 tensor 比较处才以 `RuntimeError: size mismatch` 浮现、形似 shape/runtime 类，判类时须核对 golden 每个返回值的 `dtype`/`shape`。两形态处置相同（纯选择类算子 reference 回退 CPU golden）。
    动机: 本次 attempt 1 的分类若按旧信号会漏掉该子类而错走 shape/runtime 排查；扩展后该缺口在集成期一步判类（本任务实际修复成本 = 一次受控 reference 路径修复 + 一次 report 重跑 ≈17 min）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 修订已被用户批准的 VP-2026-0179 落盘行（同一单元格），apply 时须先读 `.opencode/agents/tilelang-op-integrator.md` 现状定位；与 VP-2026-0193（add-npu-op 侧默认 golden）配套。
## VP-2026-0195
- type: R
- title: tilelang-skill-evolution retrospective-schema 补「上游提案数值更正」登记通道——Stage 2/3 发现同函数上游 D 类提案数值错误时以「更正：<上游标题>」登记（evolver 蒸馏时以更正条目优先裁决）
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 2 Skill Flow Issues 第四行（Info-source）：Stage 1 的 `RETROSPECTIVE.md` 已把**被推翻的**数值 `53+2·eb` 作为 D 类 durable 价值点入表（Value Point Proposals 第 3 行 + Transferable Lessons），而 `retrospective-schema.md` §3 价值点表只有「新增提案」形态、**无「更正/撤回」通道**——错误常数有流入 pattern-library 的风险；Stage 2 的 Value Point Proposals 首行以「更正：…系数 = `69+2·eb`（非 `53+2·eb`）」形态记录并指明冲突双方位置（任务工作区，未上库）
  - 蒸馏侧实证（本任务 evolver 本次）：两级复盘中同一数值的正/误两版并存，裁决依赖跨章节人工比对（`DESIGN.md#§4.5` 分量注释求和 vs Stage 1 提案行）；若按机械追加，错误常数会先于更正条目落库
  - 交叉引用：pattern-library INDEX.md §2 证伪协议（证伪更正须留痕）/ merge-policy §3 冲突消解（Tier 0 D 类冲突可直接「新 add + 旧 deprecate」）
- repro: repro-missing（复盘 schema 规则；复现条件 = 同一函数级任务的 Stage 2/3 复盘发现 Stage 1 已登记的数值错误时，检查 §3 表是否有更正形态可承载）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-skill-evolution/references/retrospective-schema.md
- delta: |
    动作: update（§3 价值点表字段说明追加一行）
    定位锚: §3 价值点表（Value Point Proposals）字段说明处（apply 时按当时文件锁定精确锚）
    new 文本: 追加说明：**上游提案的数值更正通道**——Stage 2/3 复盘若发现同一函数级任务上游（Stage 1 等）已登记的提案数值/结论有误，以 `title = 更正：<上游提案标题>` 登记（`vp_type` 按证据类型），并在 `evidence` 列**同时给出冲突双方位置**（上游提案行 + 本次反算依据）；evolver 蒸馏时以「更正」条目优先裁决（实测 > 推演；并可对旧条目执行 `deprecate`），避免错误常数先于更正落库。
    动机: 本任务 v0 的错误容量系数（`53+2·eb`）已被上游复盘登记为 D 类 durable 价值点，而更正只能靠 Stage 2 自发写入 Value Point Proposals；机械追加若无更正通道，错误结论会与更正条目并存甚至先落库（pattern-library 的 D 类冲突虽可自愈，但依赖人工发现）。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 属自进化机制自身的 schema 修改（Tier 2 人工门）；与 merge-policy §3 冲突消解条款衔接。
## VP-2026-0196
- type: R
- title: pattern-library/repro/README.md 补「oracle / 测试设计类陷阱允许纯宿主语言（torch/numpy）实现」——只要可独立执行 + 断言现象存在 + 不 import 任务工作区，且头部标注 `First verified` 验证环境
- evidence:
  - task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z Stage 3 Skill Flow Issues 第二行：本任务新增 repro `repro/TRAP-torch-oracle-maxpool1d-empty-window.py`（96 行，纯 `torch.nn.functional`、无 NPU 依赖、`python repro/...` exit 0）——是 repro 索引中**首个纯宿主语言形态**（现存 12 条均为 NPU 可运行）；`repro/README.md` 的「每个 repro 的要求」只写「可独立执行/可独立检查 + 自描述头 + 断言」，未说明**是否允许非 tilelang / 非 NPU 的 oracle 类陷阱**（任务工作区，未上库）
  - 本次蒸馏已按 ED-C 机械转正该 repro（逐字节拷贝 + 头部核验 + py_compile PASS + `repro_runner --filter` 实跑 PASS），但其形态与 README 现行措辞的默认假设（NPU 可运行）不一致——需要 spec 侧显式承认该形态
  - 交叉引用：repro/README.md「分级断言形态（kind: trap）」段 / `repro_runner.py` 的可运行部分语义（纯 CPU 脚本天然可批量回归）
- repro: repro-missing（repro 规范规则；复现条件 = 提交一个纯 torch/numpy 的 oracle 类陷阱 repro，检查 README 是否有条款承载）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…） + torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/repro/README.md
- delta: |
    动作: update（「每个 repro 的要求」第 1 条后追加一句）
    定位锚: "1. **可独立执行/可独立检查**：`python repro/<条目ID>.py` 直接跑或通过语法检查——输入自造（脚本内构造最小张量/shape）、不 import 任务工作区模块、不读 `examples/{project}/` 路径（kb_lint 的 `KB-REPRO-DECOUPLE` 机械校验此解耦）；"
    new 文本: 在该条之后追加：
      **oracle / 测试设计类陷阱（kind: trap）允许纯宿主语言实现**（torch / numpy，无 NPU、无 tilelang）——只要满足「可独立执行 + 断言现象存在（premise + hazard + control 三段）+ 不 import 任务工作区」，并在头部标注 `First verified` 的验证环境（宿主语言与版本）；此类 repro 对 `repro_runner` 天然是全可运行部分（如 pooling 空窗口域的 torch-oracle 陷阱）。
    动机: 本任务转正的纯 torch repro 与 README 现行措辞的默认假设不一致；spec 不承认会使后续同类转正被 kb_lint/人工核验判为形态例外而反复讨论。
- status: pending
- confirmations: -/-
- created_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14
- decided_by: -
- decided_note: 与已合入条目 repro/TRAP-torch-oracle-maxpool1d-empty-window 配套；仅扩形态许可，不放宽「可独立执行 + 解耦」两条硬要求。

## VP-2026-0197

- type: R
- title: `gate_lint.review_lint` 的 `S2-MIG-DIM0` 用裸子串 `"n/a"` 匹配「维度 0 不适用」——迁移任务必需的源路径连写 `reduction/argreduce.py`（`reductio`+**`n/a`**+`rgreduce`）必然误触，「结论通过」的合规检视报告无法过机械门禁
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 2 首轮检视：`REVIEW.md` 维度 0 正文两处源路径 `ops/reduction/argreduce.py`（设计侧 Stage 1 与 reviewer 双复盘均登记）；`statectl complete 2` 机械门禁失败（`.task_timeline.jsonl` fail 事件 `stage=2 / verdict=design_revision / ts=2026-10-14T15:34:07Z`），conductor 现场定位 ≈80s 后按 `design_revision` 回退整轮；设计按「`reduction` 目录下 `argreduce.py`」分列改写后同报告门禁 0 failure——证明失败与维度 0 质量无关，纯规则误触
  - `.agents/tools/gate_lint.py#L463-475`（`dim0_body = _dim_body(text, 0)` → `if "n/a" in dim0_body.replace(" ", "")` 触发 `S2-MIG-DIM0` fail）；机械复现：`python3 -c "import importlib.util as u; s=u.spec_from_file_location('g','.agents/tools/gate_lint.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.review_lint('examples/argmin/_argreduce_kernel/REVIEW.md', True))"`——含 `reduction/argreduce.py` 连写路径的训练样例返回 failure
  - SESSION_TIMING_ANALYSIS §3 方案 1（量化：P4+P5 ≈900s 中约 460s / 12% 总墙钟为误触导致的空转修订轮，为该任务最大单项可省耗时）
- repro: 复现命令见 evidence 末行（`gate_lint.review_lint` 可按模块 import 调用）；知识域最小复现 = 构造一份维度 0 含 `ops/reduction/argreduce.py` 连写路径的合规 REVIEW 文本跑 `review_lint(migration=True)`（provenance 允许失效）
- toolchain_stamp: 流程规则/工具缺陷（无运行时依赖）；发生环境 tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（`S2-MIG-DIM0` 判定式改词边界/占位词形态匹配）
    定位锚: `            if "n/a" in dim0_body.replace(" ", ""):`
    old 文本: `if "n/a" in dim0_body.replace(" ", ""):`
    new 文本: |
        # 占位词形态匹配（词边界）——裸子串会误触 reduction/arg* 类源路径：
        # "ops/reduction/argreduce.py" 连写后含 "n/a"（reductio[n/a]rgreduce）。
        if re.search(r"(?<![A-Za-z0-9_])n/a(?![A-Za-z0-9_])", dim0_body.replace(" ", "")):
    动机: 迁移任务（reduction 族）的维度 0「源码核对证据」典型写法就是引用 `ops/reduction/argreduce.py`/`reduction/argmax.py` 等路径——裸子串规则使**合规且必须引用源码路径**的报告必然失败，代价是整轮「通过」结论被回退 + 设计空转修订（本任务 ≈460s）。词边界匹配保留「独立占位词 `n/a` 即 fail」的原意，同时放行路径字面。备选（审批时可择一或叠加）：对已知路径前缀（`reduction/`、`ops/`）先做掩码；design/review skill 侧补「维度 0 引用 `reduction/arg*` 路径时分列书写」的写作纪律（本任务 P6 一次即过的规避形态）。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0019/0020/0066（gate lint 误报族）同批；本族已有四起以上实证且本起为「合规报告必然失败」形态，建议审批简报置顶。

## VP-2026-0198

- type: R
- title: `tilelang-design-review` 写盘 REVIEW.md 后、返回前本地自跑 `gate_lint.review_lint(path, migration=True)`（可按模块 import 调用）——把 `S2-*` 全部机械门禁从 `statectl complete 2` 后移到返回前；同一步骤注明 D-1 重跑的解释器选择
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 2 第二轮（post-revision recheck）：reviewer 在写盘后主动 `python3 -c "... gate_lint.review_lint('examples/argmin/_argreduce_kernel/REVIEW.md', True)"` → `FAILURES: 0 WARNINGS: 0`，据此确认返回 `REVIEW_COMPLETED`——证明该自检可在 `statectl complete` 之前完成（本轮用一次即过）
  - 上一轮（同任务 Stage 2 首轮）的对照成本：`complete 2` 才发现 `S2-MIG-DIM0` 误触 → conductor 定位 ≈80s + 整轮 `design_revision` 回退 382s + 重检视 284s（SESSION_TIMING §3 方案 1）
  - 解释器缺口（同任务 reviewer Stage 2 首轮 Skill Flow Issues）：检视环境默认 `python3`（`/usr/bin/python3`、`/usr/local/python3.12.13/bin/python3`）**无 torch**，`verify_equiv.py` 直接 `ModuleNotFoundError`；改用 `/workspace/venv/bin/python`（torch 2.9.0+cpu）后 exit 0 ALL EQUIV_PASS——解释器选择未成文，试错成本 ≈60–120s（SESSION_TIMING §3 方案 2）
- repro: 复现命令 = 对任一 `examples/{project}/{op}/REVIEW.md` 跑 `gate_lint.review_lint(path, migration=True)`（含 `n/a` 裸子串/维度缺失/维度 8 证据缺失各形态时返回对应 failure）；解释器探测 = `find / -maxdepth 9 -path '*site-packages/torch'`（provenance 允许失效）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ac0522ed…（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 4 段落，写盘之后追加一步提交前自检）
    定位锚: "### Phase 4：生成 REVIEW.md\n按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      ### Phase 4：生成 REVIEW.md
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

      **提交前本地机械门禁自检（写盘后、返回前，必做）**：`gate_lint.review_lint` 可独立于 `statectl` 直接 import 调用——`python3 -c "import importlib.util as u; s=u.spec_from_file_location('g','.agents/tools/gate_lint.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.review_lint('examples/{project}/{op}/REVIEW.md', True))"`；`FAILURES: 0` 方可返回 `REVIEW_COMPLETED`（迁移任务尤其核对 `S2-MIG-DIM0`/`S2-DIMS`/`S2-DIM8-EVIDENCE`/`S2-PASS-NO-ISSUES`）。该自检把机械门禁失败从 `statectl complete 2` 之后前移到返回之前（2026-10-14 argmin 实证：一次自检即过；对照上一轮直到 `complete 2` 才暴露，整轮回退 ≈750s）。**D-1 重跑的解释器**：本仓默认 `python3` 可能无 torch——先探测（`find / -maxdepth 9 -path '*site-packages/torch'` 或试 `/workspace/venv/bin/python`）并把所选解释器与版本写进 REVIEW.md「机械复核」段，避免脚本 `ModuleNotFoundError` 被误判为「脚本执行失败」（2026-10-14 argmin 实证：`/workspace/venv/bin/python` exit 0 ALL EQUIV_PASS）。
    动机: 机械门禁的失败信息在 `statectl` 侧，检视 skill 流程不自跑则只能被动等待；本任务第二轮实测证明自检成本≈0、收益为一整轮返工。两条同属「Phase 4 提交前自检」的零成本防线，故合并为一个提案。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0199

- type: R
- title: `tilelang-design-review` 空真守卫扩「对照组判别力」——`violations=0` 不足以证明等价性脚本有判别力：须核对候选与基线不共用同一实现路径（死赋值/精确单射两类退化对照）且存在能触发差异的用例
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 2 首轮 + Stage 1 revision v1：`verify_equiv.py` E2 基线 `base_source_argmin` 第 85 行 `m = -((-v).max(...))` 被第 87 行 `m = nan_ignoring_min(v)` 立即覆盖（死赋值）→ 候选与基线在 NaN-free 用例上字面等价；E4 的 fp32 上抛为**精确单射**→ 两组各 264 元素全部比较却恒 PASS——机械输出 `violations=0` 与真实覆盖无法区分（REVIEW.md 维度 8-④ 独立推演确认属**覆盖缺口非结论错误**）
  - v1 修复形态（判别力自证）：E2 基线显式保留 `-max(-x)`（fmax NaN-忽略、无覆盖）、E4 改为原生 dtype 流水 vs fp32 上抛的真实跨域对照，并加三组故意错误候选（`neg_last_occurrence`/`neg_small_sentinel`/`neg_coarse_pipeline`，`expect_pass=False`）——本轮 reviewer 亲跑输出 `CTL ... -> CONTROL_OK`（`viol>0`）方判脚本有判别力
  - 交叉引用：现有 Phase 1 第 7 步空真守卫只覆盖「比较元素数 = 0」一种退化 PASS（max_pool2d 2026-10-14 实证）；本条是其形态扩展（**非空真但无判别力**）
- repro: 复现条件——任一含 `verify_equiv.py` 的迁移/重做任务：检查每组候选与基线是否共用实现路径（如基线存在被后续语句覆盖的死赋值、或两侧经精确单射往返），以及是否存在 `expect_pass=False` 且 `viol>0` 的故意错误候选（provenance 允许失效）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 torch 2.9.0+cpu（脚本侧）+ tilelang 0.1.2+ac0522ed… + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「空真守卫核对」条目追加第 ③ 点）
    定位锚: "   - **空真守卫核对**：逐组核对 `verify_equiv.py` 的每组 case——① 两侧广播后**比较元素数 > 0**；② case 配置确实满足其声明的前置条件（特化条件成立 / 拒绝条件不成立）。"
    old 文本: （即上述定位锚原文所在条目）
    new 文本: |
      - **空真守卫核对**：逐组核对 `verify_equiv.py` 的每组 case——① 两侧广播后**比较元素数 > 0**；② case 配置确实满足其声明的前置条件（特化条件成立 / 拒绝条件不成立）；③ **结构性可证伪性（判别力自证）**——逐组确认候选与基线**结构上可证伪**：不共用同一实现路径（反例形态：基线语句被后续赋值覆盖的死赋值——两者在目标域字面等价；对照两侧经**精确单射**往返如 fp16/bf16→fp32 上抛——恒真）且存在能触发差异的用例；若脚本含 `expect_pass=False` 的故意错误候选，逐组确认其为 `CONTROL_OK`（`viol>0`），否则脚本判别力未自证。`violations=0` 在「非空真但无判别力」的对照上与真实覆盖不可区分（2026-10-14 argmin 实证：E2 死赋值 + E4 精确单射两组各 264 元素全 PASS 而无信息量；v1 以三组故意错误候选 + 真实跨域对照修复）。
    动机: 空真守卫的现行两条判据都建立在「有比较元素数」上，而本任务暴露的是「有元素、全比较、仍无判别力」的形态——机械 `violations=0` 与「脚本能报出差异」是两件事。补第 ③ 点后可把判别力变成机器断言（`CONTROL_BROKEN` 即脚本失败），检视侧不必靠目检。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0200

- type: R
- title: `tilelang-op-design` Phase 2 D-1 生成规范补两条硬要求——每组候选与基线不得共用同一实现路径；至少一组故意错误候选须被报出（负向对照，`CONTROL_BROKEN` 即脚本失败）
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 1 first_design 产出的 `verify_equiv.py`：E2 基线死赋值（第 85 行被 87 行覆盖）+ E4 精确单射恒真 → 10 组 case 全 `EQUIV_PASS` 而两组无判别力（REVIEW.md 维度 8-④；Stage 2 首轮复盘 Value Point Proposals 第 2 行）
  - revision v1 落地形态：`examples/argmin/_argreduce_kernel/verify_equiv.py`（v1：`base_transform_argmin` / `cand_native_argmin` / `cand_upcast_argmin` / `neg_last_occurrence` / `neg_small_sentinel` / `neg_coarse_pipeline`）+ DESIGN.md §1.6.1 结果表（5 组门禁 + 3 组对照）——执行输出 `ALL EQUIV_PASS | controls_ok=True`，exit 0
  - 解释器约束（同任务）：设计侧机器验证须在带 torch 的解释器下运行（本任务 `/workspace/venv/bin/python`）
- repro: 复现条件——任一含采纳优化项的迁移设计：生成的 `verify_equiv.py` 须含 ≥1 组 `expect_pass=False` 的错误候选并断言其 `viol>0`；脚本内不得出现「基线语句被后续覆盖」的死赋值或纯精确单射往返对照（provenance 允许失效）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 torch 2.9.0+cpu + tilelang 0.1.2+ac0522ed… + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 5 步「等价性机器验证（D-1）」条目追加两条硬要求）
    定位锚: "   - **内容**：每个采纳项的候选式 vs 基线式数值对照——随机张量 + 角点值（±inf/NaN/subnormal/边界 shape）上以 **fp64 参照**比对（torch CPU 实现，两式均以 fp64 计算后按目标 dtype 舍入对照）；SymPy 符号化简可选（仅限可符号化公式）；"
    old 文本: （即上述定位锚原文所在条目）
    new 文本: |
      原条目追加两条：
      - **判别力要求（硬）**：每组候选与基线不得共用同一实现路径——禁用形态包括「基线语句被后续赋值覆盖的死赋值」（两者在 NaN-free 域字面等价）与「对照两侧经精确单射往返」（如 fp16/bf16→fp32 上抛恒真）；本脚本须含**至少一组故意错误候选**（wrong tie-break / 过小哨兵 / 有损粗化域等）且脚本断言其被报出差异（`viol>0`，输出 `CONTROL_OK`；`CONTROL_BROKEN` 即脚本失败）——把「脚本能报出差异」变成机器断言（2026-10-14 argmin 实证：v0 10 组全 PASS 而两组无判别力，v1 补三组负向对照后 `controls_ok=True`）。
      - **执行解释器**：脚本需 torch CPU（fp64 参照）——运行前定位带 torch 的解释器（本仓常见 `/workspace/venv/bin/python`）并把解释器与版本写入结果表，避免 `ModuleNotFoundError` 被误判为等价性失败。
    动机: D-1 的价值在于「有判别力的机器验证」；现行规范只约束内容形态（fp64 参照 + 角点值），未约束**对照的可证伪性**——本任务该缺口导致 Stage 2 首轮两组 case 全 PASS 而无信息量（检视侧靠独立推演才发现，属覆盖缺口）。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0201

- type: R
- title: `tilelang-op-develop` 的 `templates/op_template.py` 用 `out_idx=[-1]`，与迁移 harness 的 `out_idx=[1]` 契约不一致——迁移场景照抄模板会得到错误的多输出切分
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 3 Skill Flow Issues 首行：`.agents/skills/tilelang-op-develop/templates/op_template.py:20` 为 `@tilelang.jit(out_idx=[-1], target="npuir")`，而迁移 harness 的单输出 kernel 契约用 `out_idx=[1]`（本实现 `_argreduce_kernel.py:255,403`；由 wrapper 契约决定，与 GPU 源 `@tilelang.jit(out_idx=[1])` 对齐）
  - 交叉引用：设计模板传播面缺口家族（同族：queue VP-2026-0158 `compile_flags` 第三现 / VP-2026-0107 golden 内联）
- repro: 复现条件——迁移场景按 `templates/op_template.py` 抄写 `out_idx=[-1]` 后与 wrapper 多输出契约对照（切分错位）（provenance 允许失效）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+ac0522ed… + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-develop/templates/op_template.py
- delta: |
    动作: update（`@tilelang.jit` 装饰行后加注释说明两种契约）
    定位锚: `@tilelang.jit(out_idx=[-1], target="npuir")`
    old 文本: `@tilelang.jit(out_idx=[-1], target="npuir")`
    new 文本: |
      # out_idx 契约：new_op 的多参数签名用 [-1]（最后一个参数为输出）；
      # 迁移 harness 的单输出 kernel 用 [1]（与 GPU 源 @tilelang.jit(out_idx=[1]) 及
      # wrapper 契约对齐）——照抄本模板的 [-1] 会得到错误的多输出切分。
      @tilelang.jit(out_idx=[-1], target="npuir")
    动机: 模板是 Stage 3 的默认起点，契约差异属「照抄即错」的静默面；一条注释即可消除（2026-10-14 argmin 实证：harness 侧实作用 `out_idx=[1]`）。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0202

- type: R
- title: AGENTS.md「Keyword to Docs Mapping」的条件操作行落点失真——`docs/Tilelang.language/条件操作/` 下只有 `T.vselect.md`，`if_then_else` 只出现在 `编译器提示操作/T.Parallel.md` 正文，按路由检索会落空并易误判「无文档/无先例」
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 1 Skill Flow Issues 首行（Info-source）：`ls docs/Tilelang.language/条件操作/` → 仅 `T.vselect.md`；`grep -rln if_then_else docs/Tilelang.language/` → 仅 `编译器提示操作/T.Parallel.md`；设计 §3.4 最终只能引 `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py` 作先例
  - 现行 AGENTS.md L122-124 把 `if_then_else` 列在「Comparison and selection ops」映射下并指向 `比较操作/` + `条件操作/`
- repro: 复现命令 = `ls docs/Tilelang.language/条件操作/` + `grep -rln if_then_else docs/Tilelang.language/`（provenance 允许失效）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-10-14（仓库 docs 树现行状态）
- target_doc: AGENTS.md
- delta: |
    动作: update（关键词映射的「条件操作」行加落点注记）
    定位锚: "- Comparison and selection ops (vcmp/T.min/T.max/if_then_else/select):\n    docs/Tilelang.language/比较操作/\n    docs/Tilelang.language/条件操作/"
    old 文本: （即上述定位锚原文）
    new 文本: |
      - Comparison and selection ops (vcmp/T.min/T.max/if_then_else/select):
          docs/Tilelang.language/比较操作/
          docs/Tilelang.language/条件操作/   # 现仅含 T.vselect.md
          # if_then_else 的文档落点为 docs/Tilelang.language/编译器提示操作/T.Parallel.md（正文），
          # 条件操作/ 下无 T.if_then_else.md；按本行检索落空不等于「无文档/无先例」。
    动机: 按路由检索落空会被误读为「无文档/无先例」，进而影响负向断言与选型（本任务设计只能引测试文件作先例）。注记是零成本修法（如需更彻底，可在 options 下新增 `docs/Tilelang.language/条件操作/T.if_then_else.md`——那属仓库文档面，另行立项）。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: -

## VP-2026-0203

- type: R
- title: `kb_search.py` 缺 device facet 过滤/高亮——陷阱与常数条目绑定设备/工具链戳（910B2C/CANN 8.5.0 vs 910B3/CANN 9.0.0）时，「同设备优先」筛选只能逐条人工判定
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 1 Skill Flow Issues 第二行（Info-source）：设备 Ascend910B3 / CANN 9.0.0 与同族 argmax 任务（910B2C / CANN 8.5.0）不同，argreduce 全陷阱族（§9.1 C-1..C-11）须逐条人工判定「同工具链 vs 待重验」，成本高且易漏（DESIGN.md §0 实查表 vs pattern-library 条目的 `device:`/`toolchain:` 字段；DESIGN §9.1 逐条标注）
  - 现行 `kb_search.py` 已解析并打印 front-matter `facets`（L264-283）但无按 facet 过滤/加权的入口
  - 交叉引用：queue VP-2026-0089（stale 判定口径）/ VP-2026-0116（stamp diff 过滤非源码路径）/ VP-2026-0101（stale 消费两分法）——本条补「判 stale 之前先按设备/工具链分域」的检索侧
- repro: 复现条件——用 `python3 .agents/tools/kb_search.py "<query>" --json` 观察返回条目的 facets 含 device/toolchain 但无法按当前设备过滤（provenance 允许失效）
- toolchain_stamp: 流程规则/工具增强（无运行时依赖）；证据环境 2026-10-14 蒸馏会话（kb_search.py 现行版本）
- target_doc: .agents/tools/kb_search.py
- delta: |
    动作: update（新增可选参数与输出标注）
    定位锚: `    ap.add_argument("query", nargs="*", help="search query")`
    old 文本: （即上述定位锚原文行）
    new 文本: |
      `    ap.add_argument("query", nargs="*", help="search query")` 之后新增：
      `    ap.add_argument("--device", help="filter/rank by device facet (e.g. 910B3)")`
      `    ap.add_argument("--toolchain", help="filter/rank by toolchain facet substring (e.g. CANN 9.0.0)")`
      行为：提供时对返回条目按 facets.device / facets.toolchain 子串命中做优先排序（命中前置并标注 `[同设备]`/`[同工具链]`），不提供时维持现状；不改变 bm25 主体与日志格式。
    动机: 跨设备/跨工具链迁移时，「同设备条目优先、异设备条目待重验」是消费纪律（pattern-library 证伪协议第 2 条）；当前只能靠人眼扫 facets 行逐条判，条目量增长后成本线性上升（本任务 argreduce 陷阱族 11 条逐条判定）。filter/rank 是纯增量、不影响既有检索语义。

- status: pending
- confirmations: -/-
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: 与 VP-2026-0089/0101/0116（stale 治理三案）互补：判 stale 之前先按 device/toolchain 分域。

## VP-2026-0204

- type: D
- title: pad 契约（`N_padded = align_up(N,256)`）下 tiled 路径的两条设计事实——静态尾 tile `tail ≡ 0`（不可达，非被覆盖面）；冻结 tiled `tile_n` 预算只在 bm=1 融合候选形态下自洽（显式向量链的 idx 缓冲击穿 64KB）
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 3：`_argreduce_kernel.py:344-347`（`tail = N_padded - num_full*tile_n`）+ `:186-206`（`_pick_tile_n` 自 cap 降至 256 且偏好整除）——实测 tn=5120/4096/2048 均整除 102400/32768/16384，余数列全 0（`L0/L1` 全部 tiled 用例 `tail=0`）；设计 §5.4 的「tiled 静态尾 tile」在本契约下不可达，实现按设计保留该路径（防御性、`tail_w=1` 开销可忽略）但测试无法构造 `tail>0`
  - `_tiled_slab` 实测（`:130-165`）：fp16 5 / bf16 7 / fp32 4 个 slab 单元，恰计 x_ub/x_work/ext_brc/cand 四类 `(1,tile_n)` 大缓冲，与设计 §5.2 逐值一致；若改走驻留路径的显式 `vbrc→vcmp→vselect` 链，`idx_src`+`idx_j`（2×(1,tile_n) fp32 ≈ 4 slab 单元）直接击穿 64KB 手工预算 ⇒ tiled 候选构造须用 bm=1 融合式（`T.Parallel + T.if_then_else`，无 idx 缓冲）
  - 复现命令（任务内，provenance 允许失效）：`python3 -c "import importlib.util as u;s=u.spec_from_file_location('k','examples/argmin/_argreduce_kernel/_argreduce_kernel.py');m=u.module_from_spec(s);s.loader.exec_module(m);print([(n,m.align_up(n,256),m.align_up(n,256)%m._pick_tile_n(m.align_up(n,256),d)) for n,d in [(102400,'float16'),(32768,'float16'),(16384,'float32')]])"`（余数列全 0）
- repro: repro-missing（知识域最小 repro 待同族 reduce 迁移任务回填——判定式与 slab 计数自包含于条目正文；最小形态 = 一组 (N, dtype) 上的「align_up(N,256) % _pick_tile_n(...) == 0」断言 + slab 预算表）
- toolchain_stamp: tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目（tiled 路径预算/可达性族，置于 TRAP-parallel2d-bm-ge2-scalarize 邻域）：**pad 契约下的 tiled 路径设计事实（argmin/argmax 共享 kernel，2026-10-14）**——① **静态尾 tile 不可达**：Op 层 pad 契约使 `N_padded` 恒为 256 倍数，而 `_pick_tile_n` 的候选集含 256 且偏好整除 ⇒ 冻结 `tile_n`（fp16 5120 / bf16 4096 / fp32 2048）下 `tail = N_padded - num_full*tile_n ≡ 0`（102400/32768/16384 三例实测余数 0）——「静态尾 tile」是死代码，不构成需覆盖的回归面（要真尾块须在工厂期强制非整除 `tile_n`）；② **冻结 `tile_n` 预算与候选构造形态耦合**：tiled slab 计数（fp16 5 / bf16 7 / fp32 4 单元）只覆盖 bm=1 融合式的四类 `(1,tile_n)` 大缓冲（x_ub/x_work/ext_brc/cand）；改用驻留路径的显式 `vbrc→vcmp→vselect` 链会新增 `idx_src`+`idx_j`（≈4 slab 单元）并击穿 64KB 手工预算——同族 kernel 复刻该 tiled 路径时，候选构造形态与 slab 预算表须成对引用（与 TRAP-parallel2d-bm-ge2-scalarize 的 bm 形态、CONST-capacity 的膨胀系数联动）。

- status: pending
- confirmations: 1/2
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: D 类但无知识域 repro（判定式自包含于条目正文）→ 按 merge-policy §1 降级 Tier 1 入队；同族 reduce 迁移任务可顺手回填最小 repro。

## VP-2026-0205

- type: D
- title: `npu_loop_vectorize` 对 `T.Parallel` 内 cast 强制 `rint` 并每次 tiled 编译发警告——依赖 `trunc`/`floor` 的索引构造被静默改成四舍五入（整数域恒等时无害，非整数假设下为静默坏值面）
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 3：编译日志 `src/transform/npu_loop_vectorize.cc:801` 每次 tiled 编译命中——「Parallel cast lowering uses fixed round mode 'rint'; non-default floor/trunc cast expectations are not preserved」（本实现 `T.cast(j,"float32")` 与索引基址 `T.cast((t+1)*tile_n,"float32")` 触发）
  - 本任务无影响（索引为整数值，`rint` 与 `trunc` 在精确整数域恒等：`j < N_padded ≤ 2^24`、`t*tile_n` 精确）——价值在「该警告的正确解读」：它是固定 lowering 语义，不是可忽略的噪声
  - 复现命令（任务内，provenance 允许失效）：`python examples/argmin/_argreduce_kernel/_argreduce_kernel.py --level L0`（观察 `npu_loop_vectorize.cc:801` 警告行）
- repro: repro-missing（知识域最小 repro 待同族任务回填——最小形态 = 一个含 `T.Parallel` 内 `T.cast(float→int)` 且依赖 trunc 语义的探针，比对 rint/trunc 差异域）
- toolchain_stamp: tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 小条目（Parallel lowering 族，置于 TRAP-parallel2d-bm-ge2-scalarize 之后并互链）：**`T.Parallel` 内的 cast 被 npuir 强制 `rint`**——编译期固定警告（`npu_loop_vectorize.cc:801`：「Parallel cast lowering uses fixed round mode 'rint'; non-default floor/trunc cast expectations are not preserved」）。含义：Parallel 体内的 fp→int / fp 舍入 cast **不保留** `trunc`/`floor` 期望，一律四舍五入。**判别**：索引在精确整数域（`|x| ≤ 2^24` 的整数运算）时 rint 与 trunc 恒等、无影响（argmin 实测）；任何依赖「向零截断/向下取整」语义的浮点派生索引（含负值域、超 2^24、或非整数中间量）在该 lowering 下静默取整方向错误。同族：本警告与 TRAP-vcast-same-dtype-rounds（vcast 同 dtype 退化 rint）同源——npuir 侧 cast/舍入语义的固定性由 lowering 决定而非 call-site。

- status: pending
- confirmations: 1/2
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: D 类无知识域 repro（现象载体为编译警告 + 整数域恒等判别）→ 降级 Tier 1。

## VP-2026-0206

- type: D
- title: `tileops.reporting.analyzer._is_smoke_record` 把 label 含 "smoke" 的 benchmark 用例从 `run.json#performance.cases` / `report.md` 静默过滤（msprof profile 有效也照样过滤）——数值须引 `benchmark.json`
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 5：passing run `20261014_160246_645105_ArgminFwdOp`——`benchmark.json` 9 条记录（含 3 条 `smoke-argmin [32,256]` fp16/bf16/fp32 = 1.8000/1.8600/1.8800 µs，msprof valid），而 `run.json#performance.cases` 只有 6 条（`summary.benchmark_passed=true`、`invalid_profile_count=0`、`profiler_fallback_count=0`）；`report.md` 同缺——过滤规则见 `tileops/reporting/analyzer.py:27-35,54-56`（`_is_smoke_record`）
  - 判定命令（provenance 允许失效）：对任一新集成 Op 的 passing run 比对 `benchmark.json` 记录数与 `run.json#performance.cases` 条数（差值 = 被过滤的 smoke 用例数）
  - 交叉引用：queue VP-2026-0100（report warnings ≠ benchmark 失败——同属「report 判读口径」族）；TRAP-BENCH-CONFIG-CALIBRATION（采数配置口径族）
- repro: repro-missing（需 TileOPs + 新集成 Op 环境；判定口径自包含于条目——`benchmark.json` 记录数 vs `run.json#performance.cases` 条数）
- toolchain_stamp: tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 小条目（采数判读族，置于 VP-2026-0100 对应条目邻域并互链）：**report 层对 smoke 标签用例的静默过滤**——`tileops.reporting.analyzer._is_smoke_record` 把任何 label 含 "smoke" 的 benchmark 用例从 `run.json#performance.cases` 与 `report.md` 中剔除，**即使其 msprof profile 有效**（2026-10-14 argmin 实证：`benchmark.json` 9 条含 3 条有效 smoke 记录，`run.json` 仅 6 条）；其数值仍完整保留在 `benchmark.json`。判读动作：小 shape 的校准锚点（smoke）数值取 `benchmark.json`；`benchmark_passed` / `invalid_profile_count` / `profiler_fallback_count` 判定不受该过滤影响（`summary` 字段照常覆盖）。这是 reporting 层约定而非采数失败。

- status: pending
- confirmations: 1/2
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: D 类无知识域 repro（需 TileOPs 环境，判定口径自包含）→ 降级 Tier 1；消费场景在 Stage 5 integrator 侧，若审批倾向并入 integrator known-fixes 可在审批时改路由。

## VP-2026-0207

- type: P
- title: 等价性脚本的判别力自证法——`violations=0` ≠ 有效验证：负向对照（故意错误候选须 `viol>0`）+ 两类退化对照反例形态（基线死赋值 / 精确单射往返）
- evidence:
  - task argmin-_argreduce_kernel-20261014T151121Z Stage 2 首轮认定 + Stage 1 revision v1 修复：反例形态两类（① 基线语句被后续赋值覆盖的死赋值 → 候选与基线字面等价；② 对照两侧经**精确单射**往返如 fp16/bf16→fp32 上抛 → 恒真）；修复配方 = 基线显式保留真实语义（E2 用 `-max(-x)` 的 fmax NaN-忽略，无覆盖）、对照改为真实跨域流水（原生 dtype vs fp32 上抛），并加三组故意错误候选（wrong last-occurrence / 过小哨兵 / 有损粗化域）断言必须被报出（`expect_pass=False` → `CONTROL_OK`）
  - 实证输出：`ALL EQUIV_PASS | controls_ok=True`，exit 0（v1）；v0 同脚本 10 组全 `EQUIV_PASS` 而两组无判别力（264 元素/组全部比较）
  - 交叉引用：queue VP-2026-0006（精度失败定征顺序）/ VP-2026-0153（门禁通过≠机制生效——同「机械通过不等于有效」族）
- repro: 复现条件——任一设计期的 `verify_equiv.py`：检查「候选与基线是否共用实现路径」与「是否存在 `expect_pass=False` 且 `viol>0` 的对照组」（方法形态，无固定脚本）
- toolchain_stamp: torch 2.9.0+cpu（脚本侧，与 NPU 工具链无关）/ 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/references/quality-checklist.md
- delta: |
    add 条目「等价性脚本的判别力自证（负向对照 + 退化对照反例形态）」：把「脚本能报出差异」从人工目检变成机器断言——① **负向对照**：脚本须含 ≥1 组故意错误候选（错方向 tie-break / 过小哨兵 / 有损粗化域），断言其 `viol>0`（`CONTROL_OK`），否则脚本本身失败（`CONTROL_BROKEN`）；② **两类退化对照反例形态**（均报 PASS 且元素数 > 0，机械输出无区分度）：基线语句被后续赋值覆盖的**死赋值**（候选与基线字面等价）、对照两侧经**精确单射**往返（如 fp16/bf16→fp32 上抛恒真）；③ 正例的判别力只能靠「结构可分」的证据（如等价组真实分歧只在 NaN 语义域出现时，以 NaN 行为 INFO 行佐证两路径可分，而非用 NaN-free 违反数）。判据自包含；与 precision-patterns.md 的「门禁通过≠机制生效」互补（观测量层）。

- status: pending
- confirmations: 1/2
- created_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14
- decided_by: -
- decided_note: P 类方法条目（判别力自证法）；与 VP-2026-0199（检视侧）/VP-2026-0200（设计侧）为同一发现的三 Tier 拆分（方法本体 P + 两处流程规则 R）——三案合入时互链。

## Decided（merged / rejected / expired / conflict 归档）

## VP-2026-0003
- type: P
- title: 纯 Vector 一维逐元素 persistent 模板：num_kernels=min(ceil(N/block), aicore×2) + T.serial 静态边界 grid-stride + 动态尾块零起点切片
- evidence:
  - 〔第 1 证据，首证〕examples/lerp_tensor/_make_lerp_tensor_kernel/DESIGN.md#5.5（分核三要素）；examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 提案 + Transferable Lessons）；examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md（persistent nc=48 全档贴 copy 地板；grid-stride 实测最优访问模式）；examples/elementwise/vec_add_1d.py L31–L38（尾块处理官方先例）
  - 〔第 2 证据，不同任务，2026-10-13 蒸馏合入〕task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md#§5.5 + _adaptive_avg_pool2d_kernel.py L211-234（`vector_cores = NPUUtils.get().get_aicore_num() * 2` → `num_kernels = min(num_logical, vector_cores)` + `T.serial(num_local)` 静态界 + `uid = t*num_kernels + cid` + `if uid < NUM_LOGICAL` 守卫；多维 unit 形态推广）+ Stage 3 全量门禁绿（任务工作区，未上库，provenance 允许失效）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py --level all
- toolchain_stamp: 首证 tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24（纯 Vector 翻倍 48 vector cores）；第二证 tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b + CANN 9.0.0 + Ascend910B3（实查 20×2=40）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史单文件；按 INDEX 映射 §1 → layout.md）
- delta: |
    add §1 新小节「纯 Vector 一维逐元素 persistent 模板」（核数实查/纯 Vector 翻倍规则已有 ascend-constraints.md「物理核数限制」行，仅交叉引用不重复）：
    模板：num_kernels = min(ceil(N/block_size), aicore×2)；核内 T.serial(ceil(num_logical/num_kernels)) 静态边界（N 工厂特化保证编译期常量，可直接用 T.ceildiv+min 求值，见 §1.5 host 层常量折叠行）+ block_id = i×num_kernels + cid + if block_id < num_logical 屏蔽越界。
    尾块：tail = T.min(block_size, N−t0) + UB 零起点切片 T.copy(x[t0:t0+tail], x_ub[0:tail])——v-prefix 整 buffer 操作对 UB 陈旧段一并计算、只拷出 [0:tail]，正确性无碍。
    提取期注意：TileOPs wrapper 的 E3/E4 适配注释（如 threads= 取代）是提取期笼统预估，核数/grid 取值以 DESIGN.md 分核三要素为准，冲突时设计文档优先。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: evolver
- decided_note: 2026-10-13 蒸馏合入 layout.md PL-1.21-vector-persistent-split（第二证 = adaptive_avg_pool2d 迁移，不同任务/不同算子族/跨工具链 417dae99）；合入形态含多维 unit 粒度推广（unit 总数即 num_logical）与「不产生空核」对照（与 VP-2026-0058 的 T.ceildiv 负数域 empty-core 陷阱形态相区分）；尾块动态切片形态按首证保留并标注为一维 elementwise 形态。

## VP-2026-0002
- type: P
- title: sub-fp32 逐元素算子 fp32 中转模式：vcast(rint) 升 fp32 → fp32 域 v-prefix 链 → vcast(rint) 单次舍回（bf16 dtype 支持 + fp16 golden 对齐双触发）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 bf16 触发 + Stage 3 fp16 修复模式提案）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py（模块「Implementation Notes (attempt-2 precision fix)」等价性论证 + attempt-2 全量 62 PASS）
  - docs/Tilelang.language/数学操作/T.vadd.md §2.2.1（v-prefix 算术 dtype 矩阵不含 bf16）
  - docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 仅 rint；f32→f16/bf16 含 rint；bf16↔f32 仅 rint）
  - 第二证（不同任务，2026-09-17）：task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z——opt_log Round 3 v7_cf16（c_scaled 链 fp16 化变体在 **bf16 workload 编译失败** rollback：`T.vmul` bf16 × 的运行时实证，触发条件①独立复现）+ Stage 3 因子链形态（cb/C/dt(dtype)→f32 因子域 + `T.vcast(..., round_mode="rint")` 末端回写，L0–Boundary 全过——链结构同构第二证）
- repro: repro-missing（合入条目标注 repro-missing，ED-B 显式形态；两证任务内复现命令见 evidence——provenance 允许失效）
- toolchain_stamp: 首证 tilelang-mlir-dev dev root build（2026-09-07）；第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/elementwise.md）
- delta: |
    add 新小节「sub-fp32 逐元素 fp32 中转模式」（原 delta 全文，含双触发条件 / 链结构 / 实测 / UB 预算四段）。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- confirmed_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17（bf16 触发编译实证 + 因子链形态第二证）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（lerp elementwise 2026-09-07 / ssd mamba MixCV 2026-09-17），达 Tier 1 阈值 2/2。第二证确证触发条件①（bf16 编译失败）与链结构；触发条件②（fp16 golden 对齐）未单独复现、由首证保留。本蒸馏周期合入 elementwise.md PL-1.17-subfp32-fp32-transit（front-matter 双任务溯源，repro-missing 显式标注）。

---

## VP-2026-0013
- type: P
- title: attention 族 Developer 性能阻塞（aiv_scalar>50% / CG-2026-0001 崩溃类）时先评估 Expert 双 Scope 形态再定编程模式——结构级绕法含硬边界清单
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0（动因：developer 基线 ~3.5 TOps/s，落后 torch-SDPA 10–25×）+ #§0.6 E2/E3
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#§0/§4/Round-7（developer 谱系阻塞项来源——谱系注明：tilelang 67db6f3 + CANN 26.0.rc1）
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py（Expert 结构全同构先例）
  - pattern-library §1.7（expert vs developer 同门 bench：长 KV 1.57–1.63× / 短 KV ~1.31× 回退）
  - 第二证（不同任务不同族，2026-09-17）：task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 3——**persistent 分核 + gemm + v-prefix 混排在 Developer 模式运行时崩溃**（"unaligned UUB addresses"，repro/DEVMODE_PERSIST_CRASH.py 双模式对照）→ 切换 Expert（显式双 Scope + pass_configs 关闭 TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION）后 L0–Boundary 全过 + Stage 4 调优 2.91×；触发类从「性能阻塞/条件构造崩溃」扩展到「persistent+gemm 模式级不兼容」，Expert 硬边界（pass_configs）获独立第二实证
- repro: pattern-library/repro/TRAP-DEVMODE-PERSIST-GEMM.py（知识域，2026-09-17 转正——Expert 数值断言 + Developer 崩溃双模式对照）
- toolchain_stamp: 首证 tilelang dev build 21586b5（2026-09-07）；第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md）
- delta: |
    add §1 新小节「Expert 双 Scope 流水形态（Developer 阻塞的结构级绕法）」（原 delta 全文：判据 / 结构形态 / Expert 硬边界 / 实测收益代价 / 未文档化假设）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- confirmed_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z 2026-09-17（persistent+gemm 模式级不兼容触发类扩展 + pass_configs 硬边界第二证）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（attention 2026-09-07 / ssd mamba 2026-09-17），达 Tier 1 阈值 2/2。第二证为「方向确认 + 触发条件扩展」关系：核心主张（Developer 阻塞 → Expert 双 Scope 结构级绕法）被独立确认，触发类扩展为三类（新增 persistent+gemm 崩溃，traps-runtime.md TRAP-DEVMODE-PERSIST-GEMM / CG-2026-0010）；user_requirement 指定 Developer 时以实测为仲裁依据切换。本蒸馏周期合入 attention.md PL-1.16-expert-dualscope-bypass。

---

## VP-2026-0014
- type: P
- title: 块内 mask 的向量构造链模式：arrange strides → vsub → vcmp(int16 索引) → vand → vselect；−1e38 有限哨兵 + mask 后置 softcap + 两段式分界零 mask 开销块
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3 / #§1.6.2-#1
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345-354（vcmp 标量 PrimExpr + vand 链）/ #L611（PIPE_V 内 T.arange strides）
  - developer 谱系对照：同函数 T.Parallel 谓词预填形态 aiv_scalar 88–89% 实测（opt_log §1，**谱系注明** tilelang 67db6f3 + CANN 26.0.rc1）
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4——opt_log.md Phase 1 判别链 + Round 2：**vcmp int16 全形态标量化实测**（236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%，probe_vcmp_forms.py P6）——原 delta 链结构中 vcmp(int16) 主路径形态被证伪；核心方向（向量链替代谓词预填/规避标量化）二次实证：**算术惩罚掩码**（clamp+vmax/vmin 标量广播+vmul，band-free trace）修复后 14.2x；两段式分派升级为 trace 级（band-free 算术惩罚 / band-carrying 保留 vcmp/vselect 值替换语义——vcmp 标量化代价在该 trace 仅次要路径）
- repro: 任一含块内 mask 的 attention/causal/sliding-window 算子，以向量链替代谓词预填后对照 aiv_scalar 占比与 Task Duration（第五轮复现：`python perf_opt/bench.py perf_opt/_gqa_prefill_fwd_kernel.py --case 8b-long --use-default-config --msprof-loop 25` 外部 msprof op 同口径）
- toolchain_stamp: 第一证 tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0（developer 对照侧 67db6f3 + CANN 26.0.rc1）；第二证 tilelang 0.1.2+6797758（2026-09-15）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md）
- delta: |
    （第二证修正后合入形态）原 add §1「向量 mask 构造链」小节 → 实际合入：主体（算术惩罚掩码 + trace 两段式分派 + NaN 免疫边界 + 终值）已由第五轮 optimizer 任务内回写 attention.md PL-1.11 + traps-compiler.md TRAP-expert-v-operands 补充段；evolver 补遗两点入 PL-1.11：「−1e38 哨兵与 softcap 顺序」bullet（e^{−1e38−m} 对有限 m 下溢精确 0 规避 −inf−(−inf) NaN 路径 + mask 后置 softcap 保 P(OOB)=0 不变式）。③ K_A 分界公式取整方向推导模板仍归 VP-2026-0018（pending，合入时并入）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：vcmp 标量化证伪 + 算术惩罚掩码 14.2x——同主题不同任务的第二次独立实测，含链形态修正）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（2026-09-07 expert 迁移 / 2026-09-15 第五轮调优），达 Tier 1 阈值 2/2。第二证对原提案是「方向确认 + 形态修正」关系：向量链替代谓词预填的核心主张被 14.2x 二次实证，但 vcmp(int16) 主路径形态被证伪（标量化陷阱）——合入时以修正后形态为准（PL-1.11 算术惩罚掩码 + band-carrying 保留 vselect），原 delta 的 vcmp 链仅适用于 band-carrying 次要 trace。本蒸馏周期执行合入。

---

## VP-2026-0059
- type: D
- title: 逐 buffer 手工 UB 预算 vs BiShengIR 基础分配差 ~26KB@192KB（auto-multi-buffer 关闭仍溢出）——预算表 ≠ 可编译证明，宽 config 上靶前须编译探针
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Design-layer finding（DESIGN §4.5 预算 causal trace (bm=64,bn_eff=256,dim=128) = 178.2KB ≤ 192KB "✓"、§5.3 复述；实际 ~204.1KB 硬编译失败 `ub overflow, requires 1666048–1672192 bits while 1572864 bits available`）
  - 排除 multi-buffer 通胀：`bishengir-compile --enable-auto-multi-buffer=false` 仍溢出 → 基础分配差（疑未记载 extra/sync buffers，Stage 5 未进一步定位）；dim=64 smoke ~161.7KB 可过
  - 双路径复现（/tmp/opencode/s5_repro_mha_config.py，工厂直调 (4,512,32,128) causal fp16）：A `(64,64,1)` → E6 替换 (64,256) FAIL；B `(64,64,2)` → 逐字 bn_eff∈{64,144} OK（out/lse finite）
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4 Phase 1——probe_ub.py 编译探针 4 点标定（工具链 6797758，跨 commit 复现并精化为系数律）：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1% → **BishengIR UB 实际需求 ≈ 真实手工预算 × 1.10–1.12**；另实证 DESIGN §4.5 漏计 ub_cond2 8.2KB（「勿漏计小 buffer」）+ UB diet 绕法（N/D staging 合并 −24.6KB + rowmat 删除 −16.3KB → (64,256) 宽块解锁，解锁后掩码修复红利显形）
- repro: repro-missing（NPU 编译探针；两次任务内复现命令见各 evidence 末行——首证工厂直调双路径对照 / 第二证 probe_ub.py 变体扫描）
- toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD），均 + CANN 8.5.0 / Ascend910B2C（跨 commit 复现）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-UB-multibuffer-inflation（追加第四证/姊妹现象，与 20/26 B/elem 平台律并列）——已执行：正文追加「第四证/姊妹现象（基础分配差）」段（×1.10–1.12 4 点标定 + 预算表≠可编译证明 + 编译探针校准法 + UB diet 绕法实证 + 与 constants.md CONST-capacity-910B2C / CG-2026-0008 互链）+ front-matter origin_task/toolchain 机械更新（追加两任务）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：跨 commit 复现 + 4 点标定精化为系数律 + UB diet 绕法）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务且跨工具链 commit（28783f45 / 6797758 均复现基础分配差），达 Tier 1 阈值 2/2。定量系数（×1.10–1.12）已由第五轮 optimizer 任务内回写 constants.md CONST-capacity-910B2C；本蒸馏周期合入 TRAP-UB-multibuffer-inflation 第四证段（陷阱视角与常数视角互链）。CG-2026-0008 同轮升级 recurring。

---

## VP-2026-0063
- type: D
- title: causal 多头域画像（从未结构化优化，headroom >10×）+ 两相位收益的 config 依赖性——窄 config 短 KV 1.30–1.32x 提速 / 长 KV 0.78x 回退 vs 单遍 expert；宽块解锁后反转（第五轮刷新）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Bench（vs Sep-07 expert 基线同门对照列；profile_run_msprof_20260915_065835.log vs 20260907_173440.log）
  - 基线画像（provenance，Sep-09 采数）：perf_opt/perf_records.jsonl round 11 reg* 行——上一版 full 路径 (64,64,1) reg8bshort 3764.05µs（2.25 TFLOPS）/ reg8blong 13265.27µs（5.17 TFLOPS）vs fa4096 两相位 87.6 TFLOPS
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4——opt_log.md Final Summary：UB diet 解锁 (64,256) 后 config 依赖性画像刷新——长 KV 从 0.78x 回退**反转为 11.1x 领先**（8b-long 16962.44→1191.63µs = 14.23x vs baseline / 11.1x vs 单遍 13265µs）、短 KV 7.25–7.35x、全域几何 8.53x；原 delta 的预言「宽块收益需 mask 链缓冲精简（DESIGN §5.4 Stage 4 候选形态）」被完整实现（UB diet −40.9KB + 掩码修复后宽块红利显形 bn=256 vs 144 再 −40%）
- repro: repro-missing（端到端 msprof bench；任务内复现 = wrapper tuned 分派激活下 `msprof op --kernel-name=_gqa_prefill_fwd_main --launch-count=20 --warm-up=5 ... python perf_opt/bench.py ... --use-default-config`，raw 对账 profiles/{phase1,final}/）
- toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15），均 + CANN 8.5.0 / Ascend910B2C（基线侧 0.1.2+3a214cde 2026-09-09 采数——跨 commit 对比，倍率含版本差异成分）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.9-twophase（附加 causal 域数据点 bullet）——已执行：追加「causal 多头域画像与宽块解锁反转」bullet（headroom >10× 画像 + 窄 config 1.30–1.32x/0.78x config 依赖性 + 第五轮 UB diet 解锁后 11.1x 反转与几何 8.53x + bf16/fp16 平价，完整档案指向 PL-1.11）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：宽块解锁 + 0.78x→11.1x 反转的刷新证据）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（v3 重生成集成期 / 第五轮调优期），达 Tier 1 阈值 2/2。第二证是证据刷新而非矛盾：0.78x 回退系窄 config 下的结论，第五轮在宽 config 下将其反转——同一 config 依赖性主张的完整闭环。本蒸馏周期合入 PL-1.9-twophase causal 域 bullet。

---

## VP-2026-0035
- type: P
- title: Stage 4 交付的工厂内 tuned 分派表模式：TUNED_DEFAULT_CONFIGS 仅替换 wrapper 默认 config（显式 config 一律尊重）+ tuned 目标 case 纳入内嵌测试套件作 blocking 精度层
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py（L100–110 TUNED_DEFAULT_CONFIGS 表 + L187–195 分派逻辑「仅 caller 传 wrapper 默认 (64,64,1) 且 shape 命中时替换」+ L1447–1490 run_fa_tuned blocking 门〔4 目标 case dispatch 路径 tier-1 全 0 flips〕）
  - 同目录 opt_log.md#Iteration-7（最终组装：工厂内 tuned 分派表〔S4-5 模式〕；caller 传 wrapper 默认时生效、显式 config 一律尊重——保 wrapper 契约）
  - 谱系先例（同 op 前任务、非独立计数证据）：examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/opt_log.md（S4-5 工厂内 shape 分派调优 config）
  - 第二证（不同任务，2026-09-10 蒸馏追加）：examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/_ada_layer_norm_kernel.py（TUNED_DEFAULT_BLOCK_M per-shape 表 + select_row_config(M,N) 同形态返回值 + dtype 预算回退）+ 同目录 opt_log.md#Iteration-4（「TUNED_DEFAULT_BLOCK_M 显式逐 shape 记录而非规则化」——泛化假设否定后分派表为正解）+ #Iteration-6（U 曲线闭合验证表条目鲁棒性）+ #Final-Summary（wrapper 切换块引用建议）
- repro: repro-missing（config 级交付形态模式，无单一代码 diff；两次实现的分派表结构镜像于各自 perf_opt 交付文件〔provenance，允许失效〕）
- toolchain_stamp: 第一证 tilelang 0.1.2+3a214cde / CANN 8.5.0 / 2026-09-08；第二证 tilelang 0.1.2+a83118285a / CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md §1.9）
- delta: |
    add §1.9 附注 bullet（合入时紧随「工厂级多 @T.prim_func 变体分派」bullet 之后）：
    「工厂内 tuned 分派表（config 级，S4-5 模式）」：wrapper 契约保持的分派语义——caller 传 wrapper 默认 config 且 shape 命中 TUNED_DEFAULT_CONFIGS 时替换为调优 config；caller 传任何显式非默认 config 一律尊重原值（不静默覆盖用户意图）。tuned 分派覆盖的目标 case 须纳入内嵌测试套件作 blocking 精度层（fa-tuned 层：dispatch 路径 tier-1 翻转数对照基线），防止分派表与全量套件漂移。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- confirmed_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10（norm 族独立第二证：TUNED_DEFAULT_BLOCK_M + 非单调 bm 甜点 + 邻域闭合验证 + wrapper default_config 集成差异）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（attention 2026-09-08 / ada_layer_norm 2026-09-10），达 Tier 1 阈值 2/2，本蒸馏周期合入 pattern-library/attention.md §1.9（附注 bullet 追加于「工厂级多 @T.prim_func 变体分派」之后，含第二证要点：非单调 bm 逐 shape 记录 / 邻域闭合验证 / wrapper default_config 集成差异——合入文本见该文件）。

---

## VP-2026-0044
- type: P
- title: row-reduction 族 NPU 迁移基准形态（三件套）：全 N 驻留 UB + 按 M 行块分核（persistent num_kernels = min(ceil(M/bm), aicore×2)）+ 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限才 N-tile 流式
- evidence:
  - 首证（logsumexp，2026-08）：examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py（L64-74 核实：T.Kernel(ceildiv(M, block_m)) grid 形态 + alloc_shared((block_m, N)) 全 N 驻留 + real_m = T.min(block_m, M − pid_m·block_m) 尾块 + 双显式 slice；DESIGN.md 分层测试计划）
  - 第二证（ada_layer_norm，2026-09-10，persistent 形态）：examples/ada_layer_norm/_ada_layer_norm_kernel/DESIGN.md §5.5/§0.6（persistent min(ceil(M/bm), 48) 分核 + T.serial grid-stride + 48 = 24 AICore × 2 实查）+ _ada_layer_norm_kernel.py（尾块 T.min + 双显式 slice + 垃圾行丢弃）+ Stage 4 调优 2.14x 全组几何平均（persistent 形态下 loads-first 结构收益最大化）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Transferable Lessons 首行：「row-reduction 族 NPU 迁移三件套已两次验证（logsumexp + 本次 ada_layer_norm）」）
- repro: 复现条件——任一 row-reduction 族（norm/规约，M 行 × N 列归约）迁移设计；驻留上限推算依赖 TRAP-UB-multibuffer-inflation 平台值（fp16 约 N>11520 @20 B/elem）
- toolchain_stamp: 首证 2026-08（详见 logsumexp DESIGN 版本戳）；第二证 tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-layernorm / ALG-rmsnorm / ALG-reduction 三行（R4 亲和要点 + kb_links + known_impl）：
    ALG-layernorm R4 追加 row-reduction 迁移基准形态（两任务实测：logsumexp grid 形态 / ada_layer_norm persistent 形态）——全 N 驻留 UB（流量 4MN 最小）+ 按 M 行块分核（persistent num_kernels = min(ceil(M/bm), aicore×2)，纯 Vector 核数翻倍实查）+ 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限才 N-tile 流式（+25% 流量，ada 设计期估算）；kb_links 补 TRAP-vrsqrt-plain-precision / PL-1.10-loads-first-decoupling / CASE-norm-adalayern-stage4，known_impl 补 ada perf_opt 指针。ALG-rmsnorm kb_links 补 TRAP-vrsqrt（rsqrt 直用同样受影响）+ R4 引用 ALG-layernorm。ALG-reduction kb_links 补 CASE-reduction-logsumexp / CASE-norm-adalayern-stage4 + R4 引用。
- status: merged
- confirmations: 2/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- confirmed_by: task logsumexp-migration-2026-08（首证证据经工件追溯，创建时一并附上——VP-2026-0001 先例形态；grid 非 persistent 形态的差异已在条目正文中如实标注）
- decided_by: evolver
- decided_note: 创建即达 Tier 1 阈值 2/2（logsumexp 2026-08 + ada_layer_norm 2026-09-10 两个不同任务），本蒸馏周期合入 algorithm-candidates.md 三行。R4 迁移基准形态的正文中「persistent 强化」仅 ada 单证（logsumexp 为 grid 形态），已在文本中标注两形态来源。

### 2026-09-10 批量实施记录（conductor-improvement-report，用户批准直接实施）

> 依据 `/home/tilelang/l00970450/upload/tilelang-mlir-ascend/conductor-improvement-report.md` 的实施路线图，用户批准 P0+P1+部分 P2 批量直接实施（本节为治理留痕，非 VP 提案）：
> - **T-1/VP-2026-0042（扩展版）**：perf-feedback.md 参照锚定门槛 + optimize SKILL.md [DESIGN_LIMIT] 证据门槛 + plateau 参照结构 diff 检查 + cases.md 参考实现集；
> - **T-2/VP-2026-0010**：iteration-diagnosis.md 交错 A/B 多 run 协议前置 + `.agents/tools/ab_test.py` 工具化；
> - **K-1/K-2/K-5**：pattern-library 拆分为目录（INDEX + 主题文件 + repro/，条目 front-matter，字节预算）；cases.md 参考实现集与 CG 互链；
> - **ED-A/B/C/D/F**：证据三件套语义重定义（distillation-rules/queue-schema/merge-policy）+ repro 规范与首批 5 个转正（实跑 PASS）+ developer/optimizer repro 沉淀步骤 + repro_runner.py/kb_lint.py；
> - **E-1/E-5/E-6（K-3）**：kb_search.py 统一检索层 + conductor 预注入 + kb_stale_check.py；
> - **E-3/E-4**：审批简报节奏 + 强证据快速通道 + apply 预验证（merge-policy §6/§8）；
> - **D-1/D-2/D-3/D-4/D-5**：verify_equiv.py 等价性机器验证 + gate S1-EQUIV-EXEC + hardware-cost-model.md + constants.md + 设计期探针选项 + algorithm-candidates.md + design_calc_check.py；
> - **T-4/T-6/E-2(度量自动化)**：run_experiments 模板 + Final Summary 跨 dispatch 汇总 + stats 指标机械抽取。
> 明确未实施：VP-2026-0019/0020（用户决定暂不应用）；P2 的 T-3 BO/T-5 谱系导航/E-2 opbench/D-4 网络检索增强。

## VP-2026-0042
- type: R
- title: optimize SKILL.md [DESIGN_LIMIT] 判定补「结构不可达」证据门槛——仓库内同族先例检索 + 同口径实测 + morph 对照排除局部地板误判 + 被推翻时修正附录机制（append-only）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（第二轮「对 H=1 大 S 硬目标不可达、fa4096 结构性地板 ≈125–131µs > 100µs」被第三轮同 API 两相位结构推翻：四目标全达成 12.94/18.11/30.88/98.05µs；两处根因 = ① Stage 1 算法调研漏掉仓库内两遍式 S/P 物化先例 examples/flash_attention/flash_attn_npuir.py（同 API 可达结构）② round-10 否决两遍式的「S 需全任务驻留 → flag 预算超 15 上限」系推理错误——全局 [S,S] ws + n-block 下标 flag（nk≤16）即可成立，参考与本轮终版为直接反例）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§11.3（[DESIGN_LIMIT] 结论修正段：局部地板 vs 绝对地板界定——L1 端口 ~150GB/s / L0C 128KB / store_fixpipe 仅 GM 三硬上限存活，被推翻的仅是「上限内不存在达标数据流」的推断）
  - 同目录 perf_opt/opt_log.md#Final-Summary-第三轮（[DESIGN_LIMIT] 修正回填记录）
  - 交叉引用：queue VP-2026-0025（弃选举证边界——设计期同族规则）、VP-2026-0041（morph ladder——最小增量对照的定位方法）
- repro: 复现条件——任一触发 [DESIGN_LIMIT] 双门槛（设计层归因 + >2x 结构性估计）且拟断言「当前工具链/硬件上限内不可达」的调优终局（实证：第二轮断言后第三轮被同 API 结构推翻）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 4 条末尾追加证据门槛段）
    定位锚: "4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。**「当前工具链/硬件上限内不可达」类断言的证据门槛**（2026-09-09 attention expert 三轮闭环实证：第二轮 [DESIGN_LIMIT] 断言 fa4096 结构性地板 ≈125–131µs > 100µs 目标，第三轮两相位重构同 API 达标 98.05µs 推翻）：① 断言前强制检索仓库内同族先例（`examples/`/`testing/` 是否已有同 API 可达结构），找到先例时同口径实测 + morph 式最小增量对照（iteration-diagnosis.md「参考实现对照」）排除「冻结设计族局部地板」误判——实测外推的地板只对实测过的结构族有效（前序实证证据边界，见 negative-claim-evidence.md 第 6 条 / queue VP-2026-0025）；② 否决备选结构的理由须为实测或文档条款，纯推理否决（如「S 需全任务驻留 → flag 预算超限」）标注为未实证假设；③ 已产出的 [DESIGN_LIMIT] 被后续实测推翻时不删改原文（append-only），以「修正附录」回填 perf_feedback.md（结论限定于冻结设计族 + 修正证据表 + 存活硬上限清单 + 根因记录——本轮已验证形态）。
    动机: 结构性负结论驱动 conductor「设计层重做/等待编译器补齐」逆向路由，误判代价是整轮设计方向（第二轮误判使达标结构延后一轮才被实施，参考实现就在仓库内而两轮算法调研均未纳入）；先例检索 + 同口径实测是最低成本拦截位。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: human
- decided_note: 2026-09-10 用户批准随 conductor-improvement-report T-1 批量实施（扩展版：perf-feedback.md §1 参照锚定门槛 + §2 schema 参照锚定章节 + optimize SKILL.md Phase 3 第 4 条证据门槛 + plateau 参照结构 diff 检查 + cases.md 参考实现集；gate S4-PERF-FEEDBACK-ANCHOR 同步）

---

## VP-2026-0010
- type: R
- title: optimize iteration-diagnosis Step 6 补小 kernel 交错 A/B 多 run 测量协议前置规则（<5% 差异即启用，勿等采纳后复核）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 首测 +4.4% 复测翻转为 tie，耗一轮）
  - 同文件 #iteration-4（fp32 采纳项单轮 -5.2% 系双态膨胀，A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证：run 间 ±2–3.5% 环境双态层）
- repro: 16m fp32 final(bs8192) vs baseline(bs2048) 交替 A/B/A/B 各 2 run：B {158.75, 158.59} vs A {163.65, 162.27}，配对次序稳定
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（评估规则列表新增一条）
    定位锚: "- 只有通过必测 dispatch 非回退检查的候选 winner，才能更新为全局 current best。"
    new 文本: 交错 A/B/A/B 多 run 协议前置规则（<5% 差异即启用）。
    动机: lerp_tensor 两次踩坑（v2_op2 与 fp32 采纳项）各消耗一轮复测才校正归因；协议前置可省两轮实验。
- status: merged
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: human
- decided_note: 2026-09-10 用户批准随 conductor-improvement-report T-2 批量实施（iteration-diagnosis.md Step 6 评估规则 + optimize SKILL.md Phase 2 第 10 步/核心防呆 + `.agents/tools/ab_test.py` 工具化——交错多 run + 合并中位差 + 配对符号检验）

---


## VP-2026-0001
- type: P
- title: run 级双态（同 kernel 跨 run ±3–5%）需交错 A/B 多 run 协议裁决小差异——BP_run_state_bimodality
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证 2026-08：run 间 ±2–3.5% 环境双态层、per-process 快/慢双态、<3% 结论须多 run）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-4（第二证 2026-09-07：1M fp16 跨 run 7.30–7.66；fp32 采纳项单轮 -5.2% 双态膨胀 → A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/final/ab_*
- repro: 16m fp32：final(bs8192) vs baseline(bs2048) 交替各 2 run，msprof op --launch-count=15 取 median 后合并比较
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0（lerp）；mish 见其 opt_log 版本戳
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_run_state_bimodality（目录行 + 正文节，追加于 BP_parameter_uncertain 之后）：触发信号（同配置跨 run ±3–5%、<20us 或贴地板 workload 差异落 3% 门槛附近、复测翻转）/ 反证（launch-count 消失→BP_measurement_resolution_limited；大 kernel 单 run 已稳）/ 动作（首次 <5% 差异即前置交错 A/B/A/B：≥3 独立 run、交替次序、合并中位；3% 采纳判断必须合并中位 + 配对次序稳定）/ 验证（run 次序与中位记录、配对方向跨 run 一致）。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- confirmed_by: task mish optimize 2026-08（证据经 mish opt_log 档案追溯，创建时一并附上）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（mish 2026-08 / lerp_tensor 2026-09-07），创建即达 Tier 1 阈值 2/2，本蒸馏周期合入 bottleneck-patterns.md（目录行 + BP_parameter_uncertain 节后追加正文）。

---

## VP-2026-0021
- type: R
- title: Stage 3 大工件会话超限空返回的防再犯——DESIGN.md 分段读取 + kernel 分段落盘纪律写入 developer agent 定义（两次空返回各耗 1579s/930s）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 3 attempt 1 fail 1579s / attempt 2 fail 930s，verdict=runtime；attempt 3 complete 7701s）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（标题行：「前两次会话超限空返回，本次完成」）
  - 第三次成功的「分段读取 + 增量写入」纪律账户来自 conductor 终态钩子输入（复盘工件未展开该纪律细节——provenance 如实标注）
- repro: 复现条件——一次性整读 175KB/1249 行 DESIGN.md + 单次巨型 Write 949 行 kernel 文件的子 Agent 会话（2026-09-07 本任务前两次 attempt）
- toolchain_stamp: opencode Subagent 会话限制（2026-09-07 两次实证）；tilelang 工具链无关
- target_doc: .opencode/agents/tilelang-op-developer.md
- delta: |
    动作: update（first_impl 模式首行 bullet 扩展）
    定位锚: "- Read `DESIGN.md` + `REVIEW.md`。"
    old 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。
    new 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。**大工件分段读写纪律**（会话超限空返回防再犯，2026-09-07 attention expert 任务两次空返回实证〔1579s/930s 白耗〕）：DESIGN.md 超过 ~1200 行 / 150KB 时按章节分段 Read（先目录 + §0.5/§0.6 决策 + §3 伪代码，再按需下钻），不一次性整读；`{op}.py` 生成用分段落盘（逐段写 /tmp 后 cat 合并，或先写骨架再增量追加），不用单次巨型 Write——超限空返回表现为无输出直接失败（runtime verdict），与代码错误同型、无法从 stderr 区分。
    动机: 本任务 Stage 3 前两次 attempt 以完全相同的输入空返回（会话超限），第三次仅改变读写纪律即成功（7701s 完成 949 行 kernel + 29 用例全过）——每次空返回消耗一次 attempt 预算与 ~15–26 分钟墙钟。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 用户批准（E-3 审批简报，2026-09-11，组⑤「大工件/会话纪律」）；mode=apply 落盘——合入锚 .opencode/agents/tilelang-op-developer.md `first_impl` 模式首行 bullet（原行保留 + 大工件分段读写纪律文本行内扩展），E-4 预验证 PASS（副本应用锚点唯一命中 + standards_check 控制组/编辑组 delta 零新增 failure；repro 为会话级复现条件非可执行命令，回归 SKIPPED）。

---

## VP-2026-0022
- type: R
- title: 超长 DESIGN.md（>60KB）单次 Write 因 JSON 体积截断——tilelang-op-design SKILL.md Phase 4 补分段落盘规则
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1（Skill Flow Issues 首行：~1200 行 DESIGN.md 单次 Write 截断失败，5 段 /tmp 拼接后 cat 合并才成功；DESIGN.md 1235/1249 行为证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 attempt 1 fail 2960s——含该 Write 截断的恢复成本）
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-07 本任务首设计 attempt）
- toolchain_stamp: opencode Write 工具 JSON 体积限制（2026-09-07 实证）；tilelang 工具链无关
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一段）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
  动机: 截断失败发生在长任务收尾（写盘即交付前），恢复成本一次完整 attempt；attention/mixed 类算子的 DESIGN 普遍超此规模。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 用户批准（E-3 审批简报，2026-09-11，组⑤「大工件/会话纪律」）；mode=apply 落盘——合入锚 .agents/skills/tilelang-op-design/SKILL.md Phase 4 首行「基于 templates/design-template.md 模板，填充所有章节：」后追加「超长文档分段落盘」blockquote（锚行原文保留），E-4 预验证 PASS（同 VP-2026-0021 流程）。

---

## VP-2026-0143
- type: P
- title: 自适应池化的几何特化路径模式集——`out=1×1` 退化为单次整平面归约（avg 用 3D `T.reduce_sum` 链 / max 用整平面 `reduce_max`）；`out==in` 恒等退化为纯拷贝；退化路径容量随 H·W 缩放需编译期判据否则回退一般路径
- evidence:
  - 首证（task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z）：examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md §1.6.1 C4/C4b + §0.6；_adaptive_avg_pool2d_kernel.py `_func_global`（`T.reduce_sum(flat32, mid, dim=2)` + `T.reduce_sum(mid, gacc, dim=1)`）/ `_func_identity` / `identity_fits`/`global_fits` 编译期容量判据；verify_equiv.py C4/C4b 逐位等价 45/45；Stage 3 L2 `(1,8,32,32)→(1,1)`、`(1,8,64,64)→None` 两例走回退路径仍 0 差（任务工作区，未上库）
  - 第二证（不同任务、不同算子，task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z，2026-10-13 蒸馏追加）：examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md §0.6 R4 + _adaptive_max_pool2d_kernel.py `identity_bytes`/`global_bytes` 编译期容量判据；verify_equiv.py 132 用例全 EQUIV_PASS；Stage 3 L2 两例回退且 worst max_diff=0.0——四点模式（整平面归约 / 恒等拷贝 / 编译期容量判据回退 / 整平面单元粒度）在 max 归约（不同语义、无代数化简）上独立复现成立
  - 3D reduce 链的可执行伴随行为见知识域 repro/TRAP-vcast-shape-and-gm-source.py FIX 段第 3 项
- repro: none（设计期/实现期结构形态；3D reduce_sum 链伴随行为由 repro/TRAP-vcast-shape-and-gm-source.py 覆盖）
- toolchain_stamp: 首证 tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b + CANN 9.0.0 + Ascend910B3 / 2026-10-13；第二证 tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（ALG-pooling）
- delta: |
    update ALG-pooling（R1/R4 行）追加几何特化模式集：① `out=1×1` 全局自适应池化 = 整平面单次归约（avg 用 3D `T.reduce_sum` 链 `dim=2` → `dim=1`；max 用整平面 `reduce_max`——整平面 flatten，比一般 bin 双循环少 ≈H·W 倍向量 op；免 `T.reshape`+transpose 链，后者有 mis-compile 风险见 PL-1.1）；② `out==in` 恒等 = 纯拷贝（bins 全为单元）；③ 退化路径的驻留缓冲随 `H·W` 线性放大 → 以编译期容量判据决定是否走特化，否则回退一般路径（纯优化，语义与精度不变；`(1,8,32,32)→1×1` 的 ≈393KB ≫ 192KB 两任务均回退）；④ 与一般网格的区分以编译期常量分支实现，特化路径的单元粒度须为整平面（`NUM_LOGICAL` 单一定义源，见 queue VP-2026-0147）。
- status: merged
- confirmations: 2/2
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- confirmed_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13（pool 族同族追加、不同算子/不同 task——四点模式在 max 归约上独立复现）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同算子（AdaptiveAvgPool2dFwdOp / AdaptiveMaxPool2dFwdOp），达 Tier 1 阈值 2/2，本蒸馏周期合入 algorithm-candidates.md ALG-pooling（追加几何特化模式集段，含两证来源与 max 侧差异）。**保守限定**：max 任务设计期对照了 avg 前作骨架（同族复用），故本条按「不同 task/不同算子的独立机器验证」计——四点模式的机器可验证部分（verify_equiv + Stage 3 L2）两任务各自独立执行；如审批认为需更强独立性，可在 apply 时改回 pending。

---

## VP-2026-0146
- type: D
- title: 退化（几何特化）路径的 UB 缓冲随 H·W 线性放大——代表 shape 的容量结论不覆盖退化路径；编译期容量判据决定特化或回退（`(1,8,32,32)→1×1` 需 ≈393KB ≫ 192KB）；UB 膨胀系数双口径须设计期并列
- evidence:
  - 首证（task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z）：examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md §4.5（仅两行代表 shape：spp CH=64 = 41KB / resnet C4 = 19KB）/ §9.2；_adaptive_avg_pool2d_kernel.py `identity_fits`/`global_fits` 编译期判据 + 回退一般路径；Stage 3 L2 `(1,8,32,32)→(1,1)`（C4 的 `(CH,H,W)` flat 缓冲 `64·1024·(2+4)·2 ≈ 393KB`）与 `(1,8,64,64)→None` 两例回退且 0 差（任务工作区，未上库）
  - 同任务 Stage 2：REVIEW.md 问题 5 + design_calc_check.py `UB_INFLATION = 1.7` 单档展示 vs constants.md CONST-capacity-910B2C 双口径（默认 auto-multi-buffer ~1.7x / 关闭档 1.10–1.12）
  - 第二证（不同任务、不同算子，task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z，2026-10-13 蒸馏追加）：examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md §0.6 R4 + _adaptive_max_pool2d_kernel.py `identity_bytes`/`global_bytes` 同款编译期容量判据；Stage 3 L2 `(1,8,32,32)→1×1` 与 `(1,8,64,64)` 恒等两例实测回退、max_diff=0
- repro: none（设计期容量核算 + 编译期判据形态）
- toolchain_stamp: 首证 tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13；第二证 tilelang 0.1.2+db1783d9 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/_shared/standards/hardware-cost-model.md
- delta: |
    add §5 两节：① **退化/几何特化路径的容量缩放**——退化路径（全局池化整平面、恒等整平面）的驻留缓冲随 `H·W` 线性放大，设计期以代表 shape 的容量结论不覆盖退化路径；实现层以编译期容量判据（`H·W·CH·(elem+4)·2 ≤ 192KB` 类）决定是否走特化，否则回退一般路径（纯优化，语义不变）。实测：adaptive_avg_pool2d `(1,8,32,32)→1×1` 的 C4 flat 缓冲 ≈393KB ≫ 192KB UB → 回退，回退路径 max_diff 仍 0（adaptive_max_pool2d 同款两例复现）。② **UB 膨胀系数双口径并列**——`constants.md` CONST-capacity-910B2C 的双口径（默认 auto-multi-buffer 开 ~1.7x；`--enable-auto-multi-buffer=false` 下 1.10–1.12）在设计期须并列给出：设计未显式关闭 auto-multi-buffer 却只引 1.10–1.12 会低估上界（`design_calc_check.py` 的 `UB_INFLATION` 当前只展示 1.7 一档，检视者需自行判断设计引用的是哪一档）。
- status: merged
- confirmations: 2/2
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- confirmed_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13（同款容量判据 + L2 两例回退 0 差——① 的第二证；② 仍为 avg 单源）
- decided_by: evolver
- decided_note: ①（退化路径容量缩放 + 编译期判据）两任务独立证据达 2/2，本周期合入 hardware-cost-model.md §5；②（UB 膨胀双口径）仍为 avg 单源——合入文本已显式标注「单源」，待后续任务补证。目标文件为 `_shared/standards`（Tier 1 写域）。

---

## VP-2026-0154
- type: R
- title: op-design Phase 5 后追加「输出前路径引用自检」——以门禁 1 同款正则扫描 DESIGN.md 全部路径 token 并核验存在性（行号/函数名禁 ASCII `:` 紧贴路径、路径禁 `…` 缩写、`#`/`::` 锚后缀并入 token）
- evidence:
  - task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z：Stage 1 gate-1 重试（RETROSPECTIVE.md「Stage 1 (design) Retrospective — gate-1 retry (`S1-REF-PATHS`)」Skill Flow Issues 行）——首版 5 处 token 被判不存在，其中 3 处引用格式为 `examples/TileOPs/tests/ops/test_pool.py:130`（ASCII `:行号` 紧贴）、`.../bench_pool.py:test_adaptive_max_pool2d_bench`（`...` 缩写 + `:函数名`）、`docs/…/T.transpose.md`（全角 `…` 缩写）；仅修 5 处引用格式即通过（设计逐字保留），白耗一次 fail/start 重调度 + 修订
  - examples/adaptive_max_pool2d/SESSION_TIMING_ANALYSIS.md §3 方案 2（估 ≈105s 可省）
  - 交叉引用：queue VP-2026-0020（gate_lint `_ref_path_failures` 未剥离 `#`/`::` 锚后缀与省略号——工具侧修复，pending；本条为设计侧写作自检，两侧互补）、VP-2026-0019（S1-PLACEHOLDER 误报）
- repro: repro-missing（设计流程规则；复现条件 = 设计文档路径 token 紧跟 ASCII `:行号`/`:函数名` 或用 `…`/`...` 缩写路径，跑 gate 1 观察 S1-REF-PATHS 误判）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+db1783d9 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 5 段落末尾追加自检条目）
    定位锚: "按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。"
    new 文本: 在该句之后追加：
      **输出前路径引用自检（门禁同款）**：`DESIGN.md` 定稿后用门禁 1 同款正则 `(?:docs|examples|testing|\.agents)/[^\s...]+` 扫描全部路径 token 并逐一核对存在性——行号/函数名不得用 ASCII `:` 紧贴路径（改用文字「第 N 行」/「函数 `foo`」或全角「：」），路径不得用 `…`/`...` 省略号缩写；`#`/`::` 锚后缀形态亦会被并入 token（工具侧剥离见 queue VP-2026-0020）。
    动机: 本任务 Stage 1 gate-1 因 3 处引用格式（`path:行号` / `path:函数名` / `…` 缩写）被判「引用的仓库路径不存在」，fail/start 重调度 + 修订白耗 ≈105s；设计逐字保留说明这是纯格式问题——输出前同款正则自检可零成本拦截。
- status: merged
- confirmations: -/-
- created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13
- decided_by: human
- decided_note: 2026-10-13 用户批准（E-3 审批简报）；mode=apply 落盘 `.agents/skills/tilelang-op-design/SKILL.md` Phase 5 段：在「按照 quality-checklist 自检清单…确保文档质量。」之后追加「输出前路径引用自检（门禁同款）」句（锚文本唯一命中；原句保留）。E-4 预验证 PASS（副本应用 + 锚点唯一 + standards_check 控制/编辑组 delta 零新增 failure〔两组同为既有 SC-INLINE-DUP 单条〕；repro 为流程规则非可执行命令 → 回归 SKIPPED）。

---

## VP-2026-0155
- type: R
- title: 知识预注入扩展至 Stage 3 developer——调度 developer 前跑 kb_search 附 top-K 条目，并在 harness 迁移存在同族/同 op 前作任务时附其 RETROSPECTIVE Transferable Lessons
- evidence:
  - task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z：examples/adaptive_max_pool2d/SESSION_TIMING_ANALYSIS.md §3 方案 1（Stage 3 占会话 40.3% / 1328s，相当比例耗在 `T.vcmp(...,"ne")` ordered-NaN 陷阱的 5 组最小探针定位；「本次知识预注入只覆盖 designer/optimizer，developer 侧未注入同族教训」）
  - 同族前作 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/RETROSPECTIVE.md` 的 Transferable Lessons 存在（跨 task 可复用教训）；现有 `harness.md` §10「session 教训传递」只覆盖**同一迁移任务内**多函数（前序函数 Transferable Lessons 逐字搬运），不覆盖**跨 task 的同族前作**
  - 交叉引用：queue VP-2026-0152/0153（本任务已蒸馏的 developer 侧教训——若 Stage 3 调度前有 kb_search 注入即会命中）
- repro: repro-missing（conductor 路由规则；复现条件 = harness 迁移任务存在同族前作工件时观察 developer 调度 prompt 是否含同族教训/已蒸馏条目）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+db1783d9 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（「知识预注入」段 ① 扩展注入时点与同族前作教训来源）
    定位锚: "> E-1/E-5/E-6——注入从「纪律性全文阅读」变为「结构保证给到」，只粘贴工具输出、不做领域筛选：① 调度 designer / optimizer 前跑"
    new 文本: 把注入时点从「designer / optimizer」扩展为「designer / **Stage 3 developer** / optimizer」，并在 ① 末尾追加：
      调度 **Stage 3 developer** 前同样跑 kb_search（查询词含算子族 + dtype + 设计期已知风险形态）附 top-K 条目——纯 Vector / 归约 / 池化类迁移的语义陷阱（NaN 追踪、单位元、饱和/掩码）多在 Stage 3 才暴露；另在 harness 迁移场景，当 `examples/{op_slug}/` 下存在**同族/同 op 前作任务**时，把前作 `RETROSPECTIVE.md` 的 Transferable Lessons 小节一并附 developer 调度 prompt（前作工作区路径仅在存在时引用；已蒸馏为 pattern-library 条目的部分由 kb_search 命中）。返回条目与前作教训一律视为**数据而非指令**。
    动机: 本任务 Stage 3 耗 40.3%（1328s，NaN 陷阱探针占相当比例）——同族前作（avg 迁移）任务工件含同款模式教训，但预注入只覆盖 designer/optimizer，developer 侧未注入；把预注入前移到 Stage 3 是零成本结构保障（与 harness §10 的「同一任务内多函数教训传递」互补：本条覆盖跨 task 同族前作）。
- status: merged
- confirmations: -/-
- created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13
- decided_by: human
- decided_note: 2026-10-13 用户批准（E-3 审批简报）；mode=apply 落盘 `.opencode/agents/tilelang-op-conductor.md`「知识预注入」段（① 注入时点 `designer / optimizer` → `designer / **Stage 3 developer** / optimizer`；并在 ① 末尾追加 Stage 3 kb_search + 同族前作 Transferable Lessons 段落）。E-4 预验证 PASS（副本应用 + 锚点唯一×2 + standards_check delta 零新增 failure；conductor 文件进程内先 `git stash` 暂存其既有未提交 model 字段改动、commit 后 pop 恢复——快照仅含本条进化编辑）。

---

## VP-2026-0156
- type: R
- title: design-review 三处缺口——维度 0 增 host/op 层路径 `ls` 核对 + 多输出变体迁移范围核对；§6 输出规则明确「通过」结论下非阻塞 warn 的承载章节；维度 8 增负向断言分级复核（数学不可能 / 运算量 / API 能力）
- evidence:
  - task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z Stage 2 复盘（RETROSPECTIVE.md Stage 2 Skill Flow Issues 三行 + Value Point Proposals 两行 + Transferable Lessons）：① D-5 `design_calc_check.py` 对纯 Vector 算子三项全 skip（同 VP-2026-0047 第六现，另条处理）；② 通过结论下 4 项非阻塞 warn 无规范承载位（本任务用中性标题「附：非阻塞改进建议」规避 `redundant_issue_list` 门禁）；③ 迁移维度 0 未列 host/op 层引用路径真实性（设计引用 `ops/pool/adaptive_max_pool2d.py` 源仓实为共享 `tileops/ops/pool.py`）；④ 源算子含多输出变体 `_adaptive_max_pool2d_with_indices_kernel`（不在迁移范围）须核对「排除范围与 manifest 一致」
  - 检视手法分级复核（负向断言）经本检视实证有效：数学不可能类（max 非线性/无逆元）只须数学推演、运算量类（可分离更慢）须独立复算数字、API 能力类（不支持 bf16）须打开 `docs/` 条款核对（REVIEW.md §8；`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1）
- repro: repro-missing（检视流程规则；复现条件 = 迁移设计引用源仓单算子路径（实为共享模块）/ 源算子含多输出变体 / 通过结论带非阻塞 warn 的设计文档走维度 0 + 输出规则）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 (design env) / Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 0 表格追加两行 + §6 输出规则第 4 条扩展 + 维度 8 检视方式段追加）
    定位锚 1（维度 0 表格）: "| golden 独立性 | §8.1 golden 以 §0.1 源算子语义为依据实现（优先移植源仓参考实现），未复刻 §0.6 的 NPU 算法 | golden 直接翻译 NPU 数据流，精度对比沦为自证 |"
    new 文本 1: 在该行之后追加两行：
      | host/op 层引用路径真实性 | §0/§0.3 引用的源仓 host/op 层路径逐条 `ls` 核对存在（源仓常把同族 op 放在共享模块，如 `tileops/ops/pool.py`；语义解读指向共享实现无碍，但引用的单算子文件路径可能并不存在） | 设计引用 `ops/pool/adaptive_max_pool2d.py` 而源仓实为共享 `tileops/ops/pool.py`（误导读者） |
      | 多输出/变体函数的迁移范围 | 源算子含多输出或变体函数（如 `_*_with_indices_kernel`）时，核对「排除范围」与 manifest 声明（如 `return_indices=False`）一致，无范围蔓延 | 源算子含 indices 变体、manifest 只声明前向，设计未显式声明排除范围 |
    定位锚 2（§6 输出规则）: "4. 通过时不得出现"检视问题列表"章节。"
    new 文本 2: 扩展为：
      4. 通过时不得出现"检视问题列表"章节；通过结论下的**非阻塞 warn 统一写入固定的「非阻塞改进建议」小节**（与「检视问题列表」并列但不同名——`redundant_issue_list` 门禁仅针对「检视问题列表」标题）。
    定位锚 3（维度 8 检视方式段）: "**弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**"
    new 文本 3: 在该段末追加分级复核口径：
      **负向断言分级复核**：数学不可能类（非线性/无逆元/无单调性）只须数学推演、不要求 API 检索；运算量/复杂度类（"可分离更慢""更低复杂度算法不存在"）须独立复算数字（勿继承设计结论）；API 能力/代价机制类（"不支持 bf16""repack 是 GM 级"）须打开 `docs/` 条款或源码核对。分级避免对数学事实机械要求检索、也避免对可复算数字放水。
    动机: 本任务 Stage 2 三项缺口各消耗检视往返/返工：非阻塞 warn 无承载位使 4 项建议散落维度文字（易忽略）、host 路径未核对致设计引用不存在的单算子路径、多输出变体未核对范围；负向断言分级复核经本任务实证有效（避免对 max 数学事实机械要求 API 检索）。
- status: merged
- confirmations: -/-
- created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13
- decided_by: human
- decided_note: 2026-10-13 用户批准（E-3 审批简报）；mode=apply 落盘 `.agents/skills/tilelang-design-review/SKILL.md` 三处：维度 0 表格追加 host/op 层路径 `ls` 核对 + 多输出变体迁移范围两行；§6 输出规则第 4 条扩展（通过结论下非阻塞 warn 固定写入「非阻塞改进建议」小节）；维度 8 检视方式段末追加负向断言分级复核。E-4 预验证 PASS（副本应用 + 锚点唯一×3 + standards_check delta 零新增 failure；repro 为流程规则 → 回归 SKIPPED）。

---

## VP-2026-0157
- type: R
- title: add-npu-op S7 Tier 2 report 判定口径——report CLI `--op X` 运行整个共享 `test_<slug>.py`/`bench_<slug>.py` 文件（兄弟算子用例一并计入 `summary.correctness_tests`），被测算子判定读 `run.json` per-operator rollup
- evidence:
  - task adaptive_max_pool2d-adaptive_max_pool2d-20261013T103402Z：examples/adaptive_max_pool2d/RETROSPECTIVE.md Stage 5 Skill Flow Issues 第二行（`run.json` `summary.correctness_tests=11` 而 `operators.AdaptiveMaxPool2dFwdOp` 为 5/5；11 = 6 avg + 4 max + 1 shared `test_adaptive_pool2d_accepts_chw`）+ Value Point Proposals 第二行
  - examples/TileOPs/reports/tileops/20261013_112226_906143_AdaptiveMaxPool2dFwdOp/run.json（status=passed, correctness 11/11, benchmark_passed=true）
  - 交叉引用：queue VP-2026-0100（report 告警≠失败——同属采数判读口径族）；VP-2026-0151（Stage 5 前置与 report 时效性，同目标文件不同锚）
- repro: repro-missing（集成流程规则；复现条件 = 同族多算子共享 `tests/ops/test_<slug>.py` 时对其中单算子跑 `report CLI run --op <OpName>`，比对 `summary.correctness_tests` 与 `operators` rollup）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+db1783d9 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/SKILL.md
- delta: |
    动作: update（S7 Tier 2 report 判定段扩展）
    定位锚: "runs the full correctness suite, then the operator benchmark if correctness passes."
    new 文本: 在该句之后追加：
      **共享测试/bench 文件的 `--op` 语义**：report CLI 的 `--op X` 选择的是 `tests/ops/test_{op_slug}.py` / `benchmarks/ops/bench_{op_slug}.py` **整个文件**（非单算子参数化过滤）——同族多算子共存于一个文件时（如 pool 的 avg + max），兄弟算子的正确性/benchmark 用例一并执行并计入 `summary.correctness_tests`；被测算子的判定读 `run.json` 的 per-operator rollup（`operators` / `performance.cases[].operator`），不数 `summary.correctness_tests`。
    动机: 本任务 AdaptiveMaxPool2dFwdOp 的 `run.json` `summary.correctness_tests=11` 而 `operators.AdaptiveMaxPool2dFwdOp` 为 5/5（11 = 6 avg + 4 max + 1 shared）——不读 rollup 会误判用例口径与遗漏兄弟算子交互。
- status: merged
- confirmations: -/-
- created_by: task adaptive_max_pool2d-adaptive_max_pool2d-20261013T103402Z 2026-10-13
- decided_by: human
- decided_note: 2026-10-13 用户批准（E-3 审批简报）；mode=apply 落盘 `examples/TileOPs/.agents/skills/add-npu-op/SKILL.md` S7 Tier 2 report 判定段：在「runs the full correctness suite…passes.」之后追加「共享测试/bench 文件的 `--op` 语义」句（锚文本唯一；原句保留）。E-4 预验证 PASS（副本应用 + 锚点唯一 + standards_check delta 零新增 failure；repro 为流程规则 → 回归 SKIPPED）。

## VP-2026-0131
- type: P
- title: 索引输出 kernel 的失配行特征三指纹——全哨兵（2^30）= 匹配链断裂 / 跨 run 行数变化 = 竞态 / 全落 tile-0 局部索引 = loop-carried 状态未携带；int64 索引 torch.equal 精确比对是免费 oracle
- evidence:
  - 首证（task argmax-_argreduce_kernel-20260924T031005Z Stage 1/3）：p3c/p3d shared 多消费者竞态（BIG 行 15–25/2048 跨 run 变化）；Stage 3 首跑 vselect 状态丢失（got=[4779,3248,2345,1833] 全落 tile-0 局部索引）；N=6 vbrc 窄 N（cand 全 2^30 哨兵）；Stage 3 复盘 Transferable Lessons（「失配行特征比数值回归更好定位根因」）
  - 第二证（不同任务、不同算子族，task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 3，2026-10-14 蒸馏追加）：双输出池化（values + int64 扁平索引）——整平面路径 **values 全对而 14050/16384 条索引错**（索引呈列号逐行重复，根因见 traps-compiler.md TRAP-npuir-arange-pattern-with-whole-plane-reduce），仅因 int64 索引 `torch.equal` 精确 oracle 才被发现；一般路径全量 `torch.equal` 0/104487 mismatch 与 values `max_diff=0` 同时成立，且新指纹 ⑤（索引呈列号逐行重复）由本任务补充
- repro: none（指纹判据自包含于条目正文；三个原指纹的载体实例分别见 VP-2026-0121/TRAP-vselect-inplace-carried-state/VP-2026-0123，第四/第五指纹见 VP-2026-0128/TRAP-npuir-arange-pattern-with-whole-plane-reduce）
- toolchain_stamp: 首证 tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C；第二证 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 条目「索引输出 kernel 的失配行特征三指纹 + torch.equal 免费 oracle」（置于定征顺序条目之后，互链）：argmax/argmin 类输出 int64 索引的 kernel，`torch.equal` 精确比对是免费 oracle（任何 tie-break/哨兵/状态携带 bug 立即暴露）；失配行的**空间与时间特征**先于逐元素排查定根因类别——① 输出全落哨兵值（2^30）⟹ 掩码匹配链断裂（NaN 传播 / vbrc 窄 N 垃圾）；② 错误行数跨 run 变化 ⟹ 竞态（shared 多消费者超预算）；③ 输出全落 tile-0/首块局部索引 ⟹ loop-carried 状态未携带（原位 vselect 类）；④ 固定行错位（行 0 对行≥1 错）⟹ 步长/声明宽度错位；⑤ 索引呈列号逐行重复 ⟹ arange 派生平面下标 pattern 在 in-chain 消费时读回损坏。与 VP-2026-0006（定征顺序）、VP-2026-0038（误差结构定征）同族互补：本条是索引输出域的空间指纹版。
- status: merged
- confirmations: 2/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- confirmed_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14（不同任务、不同算子族：双输出池化的值-索引解耦实证 + 新指纹 ⑤）
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务/不同算子族，达 Tier 1 阈值 2/2，本周期合入 `.agents/skills/tilelang-error-fixer/references/precision-patterns.md`（该文件由本次合入创建，条目含五指纹 + 两次实证；文件头登记 Pending Tier 1 同域提案 VP-2026-0006/0017/0038/0052 的待合入位置）。原提案中「三指纹」在合入时按两证扩为五指纹（第四/第五指纹分别来自 VP-2026-0128 与本次新增 traps 条目）。与 VP-2026-0061（lse 恒值指纹）同「错误指纹速查」族——合入时互链（待 0061 合入补链）。


## VP-2026-0127
- type: D
- title: T.vcast f32→f32 非恒等（数据破坏 ~0.5 diff）——同 dtype 缓冲复制用 T.copy；前次任务 pyc 考古恢复
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：repro/__pycache__ pyc docstring 考古（前次任务 2026-09-23 实证）；Stage 1 Transferable Lessons（「禁 T.vcast f32→f32（非恒等、数据破坏 ~0.5 diff）；同 dtype 缓冲复制用 T.copy」）
  - 〔2026-10-14 第二证（不同任务），task max_pool2d-_max_pool2d_kernel-20261014T062320Z〕Stage 3 首轮：fp32 比较域的 C4 整平面分支用 f32→f32 `vcast` 时 8/8 元素错（最大绝对差 0.488、相对差 0.194），换 `T.copy` 后逐位一致；知识域 repro 已转正（`pattern-library/repro/TRAP-vcast-same-dtype-rounds.py`，md5 4c8f72e7…，py_compile + AST PASS）
- repro: repro-missing（知识域最小 repro 待同族任务回填——f32 buffer 经 T.vcast f32→f32 后与 T.copy 对照 diff）
- toolchain_stamp: 前次任务 2026-09-23（tilelang 0.1.2 dev build + CANN 8.5.0 + Ascend910B2C）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-vcast-f32-f32-not-identity：T.vcast 在 src/dst 同为 f32 时非恒等——实测数据破坏（~0.5 量级 diff）；同 dtype 缓冲复制用 T.copy。与 TRAP-C12-copy-dtype-cast（T.copy 跨 dtype 静默转换）构成镜像对：「copy 换 cast 会静默变值」vs「cast 当 copy 用会静默坏值」。
- status: merged
- confirmations: 2/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务（max_pool2d 窗口池化迁移，tilelang 0.1.2+0d3950aa + CANN 9.0.0 + 910B3），且本任务产出知识域 repro——三件套齐备，按 Tier 0 合入 traps-compiler.md，条目 id 采用 repro 命名 **TRAP-vcast-same-dtype-rounds**（原提案名 TRAP-vcast-f32-f32-not-identity）：机制由「非恒等/数据破坏 ~0.5 diff」精化为「`round_mode="rint"` 在两侧同为 f32 时仍按整数转换施加 → dst == round(src)、小数部分静默销毁」，并补合法分派形态（按 dtype 在 trace 期选 `T.copy`/`T.vcast`）与 repro 断言（vcast==round(src)!=src / copy 逐位一致 / f16→f32 vcast 正确）。两证跨工具链（8.5.0/910B2C 首证 → 9.0.0/910B3 第二证）。

---

## VP-2026-0144
- type: D
- title: 池化可分离 2D→1D / 前缀和差分不降低运算量（四口径复算：spp 56→6 每平面 3720/3892 > 直接法 3600；nondiv 55→7、57→7 3892/3918 > 3843）
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md §1.6.1 C2/C3 行（四口径复算：FLOPs / 访存 / 扫描遍数 / 中间缓冲；可分离 + 前缀和差分两候选均因 bin 重叠与逐输出差分点数 ≥ 直接法而否决）（任务工作区，未上库，provenance 允许失效）
  - 〔2026-10-14 第二证（不同任务、窗口池化域），task max_pool2d-_max_pool2d_kernel-20261014T062320Z〕DESIGN.md §1.6.0 R1/R3：max 归约无代数化简（非线性 → 无常数权重卷积 / 无前缀最大值差分），可分离 2D 对 manifest 三档运算量均 ≥ 基线（k3s2 → 比 1.00、k2s2 → 1.50、k3s2 → 1.01）；与 queue VP-2026-0152（max 无代数化简）同向
- repro: none（算术推导，无需运行）
- toolchain_stamp: 分析型（无工具链依赖）；核对环境 tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（ALG-pooling R3）
- delta: |
    update ALG-pooling R3 行追加实测/复算结论：自适应池化的 bin 宽为 `floor`/`ceil(in/out)` 两值且相邻 bin 重叠 ≤1，**可分离 2D→1D 与前缀和差分均不降低运算量**——spp(56→6) 每平面可分离 3720 / 前缀和差分 3892 vs 直接法 3600；nondiv(55→7, 57→7) 3892/3918 vs 3843（四口径复算，两候选均否决）。⇒ R3 的候选枚举中「窗口重叠数据的重复读」类结构优化对变长 bin 的自适应池化不成立，直接 bin 累加（C 轴向量化）为默认候选。
- status: merged
- confirmations: 2/2
- confirmed_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14（不同任务、不同池化域：窗口池化可分离比 1.00/1.50/1.01 ≥ 基线）
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务（窗口池化 max_pool2d 的 manifest 三档运算量复算），达 Tier 1 阈值 2/2，本周期合入 algorithm-candidates.md ALG-pooling R3 行（自适应 + 窗口两域结论并列）。**保守限定**：两证分别为自适应池化（四口径复算）与窗口池化（设计期复算），均属「可分离不降运算量」的同一负向结论；若审批认为窗口域数据点独立性不足，可在 apply 时改回 pending。

---

## VP-2026-0152
- type: P
- title: max 池化/归约的 NPU 向量实现形态集——C 轴向量化 + fp32 比较域 + eq 掩码乘性 NaN 追踪（`flag` 初值 1）+ 累加器单位元 `-inf` + max 无代数化简（仅结构特化 + 归约融合）
- evidence:
  - task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z：examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md#0.6-R2 + #3.2/#3.3（NaN 追踪伪代码）+ Stage 1/Stage 2/Stage 3 复盘 Value Point Proposals（任务工作区，未上库，provenance 允许失效）
  - repro/TRAP-vcmp-ne-ordered-nan.py（NaN 追踪合法式 `flag_accum` 段：eq 掩码 → vcast f32 → in-place vmul 累积〔初值 1〕→ `flag==0` 判决）
  - Stage 3 实测：一般路径（`eq_f32`/`h_acc`/`cfinal`）与 C4 路径（`reduce_min` 收尾）全用例过、fp16/bf16 worst max_diff=0.0；单位元规则 `-inf`（`T.clear` 置 0 在全负窗口出错，逻辑推演 + verify_equiv 语义对照）
  - 〔2026-10-14 第二证（不同任务、不同算子），task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 3〕`_max_pool2d_kernel.py` 一般路径：行级 `(W_in, CH)` eq 掩码 + 窗口内列切片 in-place `T.vmul` 累积（`flag` 初值 1）+ 末尾判决；整平面 C4 判决纠正为 `reduce_min + ==0`（首版 `reduce_max` 配注释「1.0 iff 窗口无 NaN」自相矛盾，部分含 NaN 平面会漏判）；端到端 L0 31 / L1 23 / L2 7 / Boundary 8 全过、三 dtype `max_diff=0.000e+00`（含 NaN / ±inf / 全 NaN 窗）
- repro: repro/TRAP-vcmp-ne-ordered-nan.py（覆盖 ③ 的 NaN 追踪合法式）；②/④/⑤ 部分 repro-missing
- toolchain_stamp: tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    add 条目 PL-1.22-maxreduce-nan-tracking（合入时紧随 PL-1.2-caxis-accum 之后，与 PL-1.2 / VP-2026-0143 互链）：
    **max 池化/归约的 NPU 向量实现形态**（自适应 max 池化迁移实测，910B3 + db1783d）：① 向量轴 = C 轴（核内 NHWC 二轴 `T.transpose` 重排后 C 轴 `T.vmax` 归约），与 avg 同构，差异仅在归约算子与 NaN 追踪；② 比较与掩码在 fp32 域（bf16 先 `vcast` 到 fp32——`vmax`/`reduce_max`/`vsub` 不支持 bf16）；③ **NaN 追踪用 IEEE 正确的 `eq` 掩码乘性累积**——`cond = vcmp(x, x, "eq")` → `vcast(cond, f32)` 得 {0,1} → in-place `vmul(flag, mask, flag)`（`flag` 初值 1：0 吸收 / 1 中性）→ 末尾 `vcmp(flag, 0, "eq")` 或整平面 `reduce_min` 判决（**不用 `vcmp(...,"ne")`**——ordered 不等，NaN 恒假，见 traps-compiler.md TRAP-vcmp-ne-ordered-nan）；④ 累加器单位元为 `-inf`（`T.vbrc(-T.infinity("float32"), acc)`；`T.clear` 置 0 在全负窗口产生错误输出）；⑤ **max 无代数化简**（非线性 → 常数权重卷积/GEMM 不等价；无逆元 → 前缀差分不可用）——"数学等价优化" 对本形态只余结构特化（全局池化整平面 `reduce_max` / 恒等纯拷贝，见 VP-2026-0143）与归约融合（max + NaN 同轮扫描）。**负向依赖**：结论不依赖 `T.vmax`/`T.reduce_max` 的 NaN 传播行为（未文档化，实现须显式追踪确保正确）。
- status: merged
- confirmations: 2/2
- confirmed_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14（不同任务、不同算子：窗口池化 max 同形态独立实现 + 行级掩码细化 + 归约算子集合语义纠正）
- created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z 2026-10-13
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务/不同算子（AdaptiveMaxPool2dFwdOp → MaxPool2dFwdOp 窗口池化），达 Tier 1 阈值 2/2，本周期合入 layout.md **PL-1.22-maxreduce-nan-tracking**（含第二证细化：③' 掩码按行级而非逐元素〔逐元素放大 op 数 out_W·k_W 倍〕、③'' 掩码归约算子 = 判决集合语义〔`reduce_max+==0` 只在全 NaN 平面成立，须 `reduce_min+==0`〕、⑥ 窗口池化 H 维守卫可外提）。**保守限定**：两证为同族不同算子且第二证对照了首证骨架（同 queue VP-2026-0143 先例），机器可验证部分（Stage 3 分层门禁）两任务各自独立执行（首证 worst max_diff=0.0；第二证三 dtype max_diff=0.000e+00）。

---

---

## VP-2026-0176
- type: R
- title: design-review Phase 1 机械复核增「空真守卫」核对——重跑等价性验证后须核对每组 case 的比较元素数 > 0 且 case 确实满足其声明的前置条件（空真 PASS 视为覆盖缺失）
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 2 复检：`verify_equiv.py` 的 C4 组有 1 条配置 `((1,8,5,6), k=8, s=2, p=0)` 输出 `(0,0)`——两侧广播后比较 0 个元素，仍输出 `viol=0 EQUIV_PASS`，与真实通过不可区分（`REVIEW.md#问题 2`；v0 基线 57 组中含 3 条空真行）（任务工作区，未上库，provenance 允许失效）
  - 同任务 Stage 1 revision v1 的处置：新增「边界不成立用例」组（C4x）与负向控制断言，使条件本身可执行；但空真守卫属**脚本结构**要求而非用例内容要求，现行 skill 无对应检查位
  - 交叉引用：queue VP-2026-0028（修订版重审工作流）/ VP-2026-0092（复检 grep + md5）/ VP-2026-0142（D-1 判据分档）
- repro: repro-missing（检视流程规则；复现条件 = 任一含「两侧形状可能不一致」比较的等价性脚本——构造输出尺寸为 0 的 case 观察 `EQUIV_PASS`）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+0d3950aa + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步「机械复核」的重跑等价性验证条目后追加一句）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + 追加一句：
      「**空真守卫核对**：逐组核对 `verify_equiv.py` 的每组 case——① 两侧广播后**比较元素数 > 0**；② case 配置确实满足其声明的前置条件（特化条件成立 / 拒绝条件不成立）。`viol=0` 在 0 元素比较上无信息量（与真实通过不可区分），空真 PASS 记为覆盖缺失（建议级→阻塞级按缺失占比判），并在 REVIEW.md 机械复核段列出空真行序号（max_pool2d 2026-10-14 实证：C4 组 1 条 `((1,8,5,6), k=8,s=2,p=0)` 输出 `(0,0)` 仍报 EQUIV_PASS；v0 基线 57 组含 3 条空真行）。」
    动机: 空真 PASS 使「条件型/形状敏感」等价性脚本给出虚假的覆盖面证据（本任务 Reviewer 手工核对元素数才发现）；在 Phase 1 机械复核处加一句零成本核对即可拦截，且不依赖脚本作者自觉。
- status: merged
- confirmations: -/-
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.agents/skills/tilelang-design-review/SKILL.md` Phase 1 第 7 步「机械复核」——在「重跑等价性验证」条目之后追加「空真守卫核对」子项（锚文本唯一命中；原句与其后条目保留）。E-4 预验证 PASS（副本应用 + 锚点唯一 + standards_check 控制组/编辑组 delta 零新增 failure〔两侧同为 SC-HASH-MISMATCH hardware-cost-model.md + SC-INLINE-DUP tilelang-op-develop/SKILL.md 两项既有 failure，stats 全等〕+ kb_lint 0 failures〔19 warnings 两侧一致〕+ `.agents/tools/tests` 3 failed/109 passed 与对照逐节点相同；repro 为检视流程规则、会话级条件 → 回归 SKIPPED）。

---

## VP-2026-0177
- type: R
- title: design-review 维度 8 增「编译期特化谓词的充分条件」核对行——列出谓词依赖变量全集 + 逐变量验证覆盖集合关系（含 `d` 这类被忽略参数）+ 子句隔离覆盖自检
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 2 首轮（1 阻塞）：v0 §0.6 R4 的整平面归约充分条件漏 `d`——反例 `H=W=5,k=2,s=1,p=0,d=4`（`out=1×1` 且四合取项全成立，窗口仅 4 个角点 → 归约 1.0 vs 正确 0.0）；现行 `design_calc_check.py` 只覆盖容量/核数/r3 算术、`verify_equiv.py` 只验公式数值一致而**不验适用条件是否充分**，该缺口只能靠 Reviewer 手工枚举谓词依赖变量 + 逐变量推演窗口覆盖集合才发现（`REVIEW.md#问题 1`）（任务工作区，未上库，provenance 允许失效）
  - 同任务 Stage 2 复检：C4x 两配置均对称 dilation（`dH==dW`），Reviewer 实跑确认「只删 `dW==1` 子句时断言不触发」——子句隔离覆盖缺口（设计自述「移除任一轴即失败」仅对 H 轴成立）
  - 交叉引用：queue VP-2026-0147（维度 8 跨节/伪代码类增补）/ VP-2026-0166（维度 8 比较-累加次序核对，同一维度不同检查面）/ VP-2026-0164（op-design 侧 D-1 实现次序守卫）
- repro: repro-missing（检视流程规则；复现条件 = 设计含条件型编译期特化谓词时，按本条逐变量枚举 + 子句隔离测试）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+0d3950aa + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 检查项表插入新行）
    定位锚: "| 与下游章节一致 | §3.1 公式拆解以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点不得再出现对应逐元素标量循环） | §3.1 仍用优化前公式；§6 出现 §1.6.2 未论证的标量循环 |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      （在该行**之前**插入新行）
      | **编译期特化谓词的充分条件** | 对 §1.6.1/§3.3/§5 中每个编译期特化谓词（`IS_GLOBAL`/`out==1×1`/`out==in`/恒等轴等），① 列出其**依赖变量全集**（`out/k/s/p/d/in` 等，含「看似不相关」的轴参数）并逐变量验证其断言的**集合关系**（覆盖/等价）——尤其窗口类：窗口输入覆盖集合 = `{k·d}` 受 `p` 平移与 `s` 取窗约束，「输出网格退化」**不**蕴含「覆盖整平面」；② 谓词应「定义一次、引用多处」（目标：§3.3 单一定义源，其余节引用）；③ 负向守卫用例须对条件中**每个子句**有隔离覆盖（至少一条用例仅违反该子句、其余成立），否则「移除任一子句即失败」为空真 | 充分条件漏参数（如漏 `d` 致 `d>1` 时静默错值——反例 `H=W=5,k=2,s=1,p=0,d=4` 正确 0.0 / 特化 1.0）；同一谓词多处重复书写且漂移；守卫用例组对称违反多个子句使单子句断言不触发 |
    动机: 本任务唯一的 Stage 2 阻塞即该检查面缺口（设计 v0 已在自己的复盘里写下「须验证窗口输入覆盖集合」却在同文档再犯）——`design_calc_check.py`/`verify_equiv.py` 均不覆盖「条件充分性」，检视侧加固定动作可把该类缺口从「靠 Reviewer 灵光一现」变为可复核步骤。
- status: merged
- confirmations: -/-
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.agents/skills/tilelang-design-review/SKILL.md` 维度 8 检查项表——在「与下游章节一致」行之前插入「编译期特化谓词的充分条件」行（锚文本唯一命中；原表行保留）。E-4 预验证 PASS（同批副本应用 + 锚点唯一 + standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure；repro 为检视流程规则、会话级条件 → 回归 SKIPPED）。

---

## VP-2026-0178
- type: R
- title: add-npu-op Stage 0/S5 增「NPU 参考实现（golden）可用性自检」——对本算子全部 fixture 参数在 NPU 上跑一次最小 golden smoke，已知受限参数直接用 CPU golden 回退模板
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 5 attempt 1 `[INTEGRATE_FAIL]`（402s）→ 根因核对（23s）→ attempt 2 修复并重跑全量 report（622s）：总往返 ≈1025s / op 级 22.3%——修复本身很小（`ref_program` 的 CPU 回退），成本在「首次诊断 → conductor 扩权 → 重跑全量」的往返（`examples/max_pool2d/SESSION_TIMING_ANALYSIS.md` §3/§4；`integration_log.md#调试历史`）
  - 直接根因：Stage 0 按 GPU 用例逐字移植时未校验 `torch_npu` 参考实现能力（`F.max_pool2d` 经 aclnn 仅支持 `dilation == 1`）——该缺陷可在 Stage 0 以约 0.5 min 的 golden smoke 暴露（知识侧事实见 queue VP-2026-0173）
  - 交叉引用：queue VP-2026-0151（add-npu-op Stage 5 前置 env.sh + S5-REPORT-STALE 口径——同为 Stage 0/S5 前置自检，本条补 golden 侧）/ VP-2026-0150（integrate_kernel.py 缺陷）
- repro: repro-missing（流程规则；复现条件 = 迁移算子的 fixture 含 NPU 参考后端不支持的参数组合时跑 golden smoke）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+0d3950aa + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/SKILL.md
- delta: |
    动作: update（Tier 1 structural verification 段末尾追加第 5 项）
    定位锚: "All Tier 1 checks must pass. This confirms the 7 files are structurally
      correct and the extracted kernel functions are properly imported — ready
      for the NPU kernel component to reimplement for `target=\"npuir\"`."
    old 文本: （即上述定位锚原文）
    new 文本: |
      上述原文 + 追加一段：
      ```
      # 5. NPU reference (golden) availability smoke — for every fixture param combo
      python - <<'PY'
      import itertools, torch, torch_npu  # noqa
      import torch.nn.functional as F
      # iterate the S5 fixture param table (k/s/p/d/ceil/dtype) with a >=2^13-element input
      # and call the reference op on an NPU tensor; record every combo that raises
      for params in FIXTURE_PARAMS:
          try:
              F.max_pool2d(torch.randn(2, 8, 33, 33, device="npu"), **params)
          except Exception as e:  # AclNN_Parameter_Error(EZ1001) etc.
              print("GOLDEN-UNAVAILABLE", params, type(e).__name__, e)
      PY
      ```
      Any combo whose NPU reference raises **before the kernel runs** (stack inside
      `torch.nn.functional`, `AclNN_*_Error`) must be recorded now and handled in
      `ref_program` with the **CPU-golden fallback template** for pure-selection ops
      (`F.x(input.cpu(), **kwargs).to(input.device)` — bit-identical for max/min/
      argmax/sort, `atol=rtol=0` kept; NOT applicable to accumulating ops). Doing
      this at S5 costs ~0.5 min and avoids a Stage 5 fail/diagnose/authorize/re-run
      round trip (≈17 min measured, max_pool2d 2026-10-14).
    动机: 本任务 Stage 5 首轮失败的根因是 Stage 0 未校验 NPU 参考实现能力（`dilation>1` 的 aclnn 缺口），而该缺口与迁移正确性无关、代价却落在集成回合；前置自检把该类缺陷挡在 Stage 0（成本 0.5 min vs 往返 17 min）。
- status: merged
- confirmations: -/-
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `examples/TileOPs/.agents/skills/add-npu-op/SKILL.md` Tier 1 structural verification 段——在「All Tier 1 checks must pass…」之后追加第 5 项「NPU reference (golden) availability smoke」代码块与处置段（锚文本唯一命中；原文保留；代码围栏平衡校验 PASS）。E-4 预验证 PASS（同批副本应用 + 锚点唯一 + standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure；repro 为 Stage 0 流程规则、需带 NPU 参考后端的环境 → 回归 SKIPPED）。

---

## VP-2026-0179
- type: R
- title: integrator 失败分类表新增子类「NPU reference 后端能力缺口」——异常在 ref_program 内且 op 未执行时，在受控闭环内把该用例 reference 回退 CPU 计算（保留用例与严格判据），仅当调度未授权改 tests 时回交
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 5 attempt 1：`test_max_pool2d[full-default-stride-dilation-fp16]` FAILED，按现行分类表只能落到「测试/用例自身问题」→ `[INTEGRATE_FAIL]`（当时 `tests/` 不在授权范围）；但该子类的正确处理是「在受控闭环内修复脚手架测试用例的 reference 计算路径」而非无条件回交——attempt 2 经 conductor 显式授权后修 `MaxPool2dTest.ref_program`（`dilation != 1` 回退 CPU golden），报告一次通过（`examples/TileOPs/reports/tileops/20261014_072101_015168_MaxPool2dFwdOp/run.json`：status=passed、本算子 22/22、`max_abs_err=0.00e+00`、benchmark 24/24）（op 级 `RETROSPECTIVE.md` Skill Flow Issues 首行；`integration_log.md`）
  - 已有专属知识条目（知识侧）：queue VP-2026-0173（torch_npu `aclnnMaxPool2dWithMask` 仅支持 `dilation == 1` + 纯选择类 CPU golden 回退逐位一致）
  - 交叉引用：queue VP-2026-0157（report CLI `--op` 共享文件兄弟算子用例判定口径→add-npu-op，已 apply）/ VP-2026-0178（Stage 0 golden 可用性自检——上游拦截，本条为集成期兜底）
- repro: repro-missing（流程规则；复现条件 = 集成用例的 NPU reference 在 op 执行前抛 `AclNN_*_Error`）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+0d3950aa + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（失败分类表插入新行）
    定位锚: "| 测试/用例自身问题 | 失败根因在 tests/workloads/manifest | 不修改，如实报告，`[INTEGRATE_FAIL]` + 建议 |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      （在该行**之前**插入新行，原行保留）
      | NPU reference 后端能力缺口 | 异常出现在 `ref_program` 内、**op 尚未执行**（堆栈落在 `torch.nn.functional`）且报 `AclNN_*_Error`（如 `aclnnMaxPool2dWithMask` 仅支持 `dilation == 1`） | **受控闭环内修复脚手架测试用例的 reference 计算路径**（本算子新增段）：`dilation != 1` 等受限参数走 CPU golden 回退 `F.x(input.cpu(), **kwargs).to(input.device)`；**仅限纯选择类算子**（max/min/argmax/排序，无累加——逐位一致、`atol=rtol=0` 保持）；用例与其严格判据全保留，禁止 skip/xfail/删例/放宽容差；只改 reference 路径，不动 op/kernel/判据/容差；备份 `history_version/{name}_s5_attempt{N}.bak` 并记入 integration_log；**调度方未授权改 tests 时**按「测试/用例自身问题」回交 `[INTEGRATE_FAIL]` + 建议（含本方案） |
    动机: 现行分类表把「参考后端能力缺口」并入「测试/用例自身问题 → 无条件回交」，导致一次可闭环的小修复走成 fail→扩权→重跑（max_pool2d 实测往返 ≈17 min / op 级 22.3% 墙钟）；分列出该子类并写明处理与边界（纯选择类限定）可让 integrator 在授权范围内一次修对。
- status: merged
- confirmations: -/-
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.opencode/agents/tilelang-op-integrator.md` 失败分类表——在「测试/用例自身问题」行之前插入「NPU reference 后端能力缺口」子类行（锚文本唯一命中；原行保留）。E-4 预验证 PASS（同批副本应用 + 锚点唯一 + standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure；repro 为集成期流程规则、会话级条件 → 回归 SKIPPED）。

## VP-2026-0150
- type: R
- title: integrate_kernel.py 两缺陷——① 聚合 `__init__.py` 生成非幂等（wrapper 已重写后 `parse_wrapper_imports` 返回空 → 重跑丢掉 helper re-export）；② `sync_wrapper_ascend_mode` 在两层类继承时把 `ascend_mode` 写到子类而非直接继承 `Kernel` 的基类
- evidence:
  - task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：examples/adaptive_avg_pool2d/RETROSPECTIVE.md Stage 5 attempt 1/2（Skill Flow Issues + Value Point Proposals）——attempt 1 二次运行 `integrate_kernel.py` 后 `ImportError: cannot import name 'adaptive_bin' from ...adaptive_avg_pool2d_kernel`（修复链见 `examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/integration_log.md` Debug history + `history_version/__init___s5_attempt1.py`）；attempt 2 gate `S5-ASCEND-MODE` 失败（两层类 `AdaptivePool2dKernelBase(Kernel)` + `AdaptiveAvgPool2dKernel(Base)`，`meta.kernel_class_name` 指子类 → 直接 `Kernel` 基类无字面量 → `len(class_modes)!=1`），gate_lint 只看直接基类为 `Kernel` 的类（任务工作区/集成包，未上库）
  - 交叉引用：queue VP-2026-0055（同文件 gen_init_py 静默降级）/ VP-2026-0064（幂等剥离注释行）/ VP-2026-0012（幂等重跑覆盖集成侧补丁）——同缺陷集群
  - 〔2026-10-13 第二现（不同任务，pool 族同族追加），task adaptive_max_pool2d-adaptive_max_pool2d-20261013T103402Z〕同缺陷在本任务 Stage 5 再现：`examples/adaptive_max_pool2d/RETROSPECTIVE.md` Stage 5 Skill Flow Issues 首行——`integrate_kernel.py` 对生成聚合 `__init__.py` 仍非幂等（wrapper 已重写后 `parse_wrapper_imports` 返回空 → 重跑丢失 helper re-export），本任务以「恰好运行一次 + 先 `source /workspace/env.sh`」规避（单次运行产出 `__init__.py` re-export 全 3 名：`_adaptive_max_pool2d_kernel` / `adaptive_bin` / `max_adaptive_bin_extent`；`smoke-ok`）。两任务独立再现 ⟹ ① 的修复（持久化 wrapper 契约名 / already integrated 时跳过 `__init__.py` 重生成）证据链增强——建议审批简报置顶（跨任务第二现）。
  - 〔2026-10-14 第三现（连续第三个 pool 族算子，不同任务），task adaptive_max_pool2d_with_indices-AdaptiveMaxPool2dIndicesFwdOp-20261014T005659Z〕同缺陷第三度出现：`examples/adaptive_max_pool2d_with_indices/RETROSPECTIVE.md` Stage 5 Skill Flow Issues 首行 + `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/integration_log.md` Issues 第 1 条——`integrate_kernel.py` 对生成聚合 `__init__.py` 仍非幂等（wrapper 重写后 `parse_wrapper_imports()` 返回空 → `gen_init_py()` 只按 integrated-file keys 重建 → 丢 helper re-export）；本任务同样以「恰好运行一次 + 先 `source /workspace/env.sh`」规避（单次运行产出 3 名 re-export：`_adaptive_max_pool2d_with_indices_kernel` / `adaptive_bin` / `max_adaptive_bin_extent`，`smoke-ok`）。**连续三任务（avg → max → max-with-indices）独立再现，无一次被修复，两次靠同一句人工纪律规避**——① 的修复已积压三起实证，强烈建议审批简报置顶（与 VP-2026-0064「幂等检查剥离注释行」/ VP-2026-0055「产物侧校验」同批）。
- repro: repro-missing（集成脚本规则；复现条件 = 已重写 wrapper 后重跑 integrate_kernel.py 观察 __init__.py 丢失 helper re-export；或对两层 Kernel 继承 wrapper 跑 `statectl gate 5` 观察 S5-ASCEND-MODE）
- toolchain_stamp: tilelang 0.1.2+417dae99 + CANN 9.0.0 + Ascend910B3 / 2026-10-13
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（两处）
    定位锚 1: "def parse_wrapper_imports(wrapper_path: Path, extracted_module: str) -> list[str]:"
    new 文本 1: 在 `gen_init_py` 调用侧（main 流程）持久化 wrapper 契约名：首次运行把 `parse_wrapper_imports` 得到的名字清单写入 `.migration_meta.json`（如 `wrapper_import_names`）；重跑时该字段非空则以其为 `gen_init_py` 的输入（不再从已重写的 wrapper 重新解析），或在 `rewrite_wrapper_import` 报「already integrated」时跳过 `__init__.py` 重生成——消除「wrapper 已重写 → 解析为空 → `__init__.py` 只剩 `integrated_files` keys → helper re-export 丢失 → import smoke ImportError」。
    定位锚 2: "def sync_wrapper_ascend_mode(wrapper_path: Path, kernel_class: str | None, mode: str) -> bool:"
    new 文本 2: `kernel_class` 给出时把目标类的选择从「名字等于 `kernel_class` 的类」改为「`meta.kernel_class_name` 的 MRO 中直接继承 `Kernel` 的类」（或在无 `kernel_class` 时对全部 bases 含 `Kernel` 的 wrapper 类写值）——两层 wrapper（`Base(Kernel)` + `Subclass(Base)`）场景下 gate_lint `S5-ASCEND-MODE` 只度量直接继承 `Kernel` 的类，把字面量写到子类会留下 direct-`Kernel` 基类无字面量而门禁失败。
    动机: 两缺陷在本任务各消耗一次 Stage 5 返工（attempt 1 幂等重跑破坏 import、attempt 2 gate 失败 + 行为中性修复的时间戳对账 164.9s）；同一脚本内两处，独立于既有 VP-2026-0055（产物侧校验）的根因。
- status: merged
- confirmations: -/-
- created_by: task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z 2026-10-13
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` 两处（① wrapper 契约名持久化到 `.migration_meta.json.wrapper_import_names` → 重跑不再丢 helper re-export；② `sync_wrapper_ascend_mode` 经 MRO 解析到直接继承 `Kernel` 的基类）。锚文本（两处 `def` 行）各唯一命中、原行保留、纯增量（+72 行 / 0 删除行）。E-4 预验证 PASS：repro 为可执行脚本场景（幂等重跑 + 两层类继承）——控制组复现两缺陷（重跑丢 helper re-export / `ascend_mode` 写到子类）、编辑组均修复；standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure。〔原备注〕与 VP-2026-0055/0064/0012 同缺陷集群；②的另一可行解在 gate_lint 侧（解析 `meta.kernel_class_name` 的 MRO），本批采用脚本侧方案。

## VP-2026-0047
- type: R
- title: design_calc_check.py 补两类表格形态解析——「按 N 分行的驻留预算表」（§4.5）与「workload 表格式分核表」（§5.5）当前均 skip，VP-2026-0009 混合字节口径类错误的机械拦截位在这些形态上失效
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 首行：ub_budget=skip、core_split=skip 的机械复核 JSON + REVIEW.md 人工逐行复算补位记录）
  - .agents/tools/design_calc_check.py#L100-103（check_ub_budget 行过滤 `^[a-zA-Z_]|缓冲` 把 N 值数字行当 header 跳过）+ #L156-158（skip detail "no per-level buffer tables parsed"）+ #L231-233（core_split skip "insufficient parsed dims"）
  - 交叉引用：queue VP-2026-0009（UB 预算混合字节口径——本条是其机械拦截位的形态扩展）
  - 〔2026-09-17 第二证（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕design_calc_check 对该 DESIGN.md **两轮（v0/v1）均 skip 3/4**（ub_budget / l0c_budget / core_split），且 v1 §4.5 已含逐行 Bytes 列与「稳态峰值合计」行、§5.2 有 bl/bs/bp/bn 代码块、§5.5 有 w1–w4 逻辑核数列表仍全部无法解析——与 ada_layer_norm（N 分行驻留表形态）构成**两类表格形态的独立第二证**；REVIEW 只能人工复算（UB 115.25KB 逐项加和、L0C 16KB、逻辑核数 8/768/2560/16384）。机械拦截位（VP-2026-0009 类）连续两任务不生效，建议审批时把本条的 §5.5 workload 分核表解析与 ssd 的 w1–w4 列表形态并入 new 文本 2 的识别范围
  - 〔2026-09-20 第三现 + 新子缺陷，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z〕4 项检查 3 skip（ub_budget/l0c_budget/core_split——该任务 §4.3 多张 buffer 表 + §5.2 block 常量代码块 + §5.5 分核三要素列表均无法解析，仅 r3_metrics 生效，全靠人工复算兜底）；**新子缺陷**：l0c_budget 解析取 "block 64x128"（bn=128 被当 block_N），与真实 L0C acc [bl,bp]=[64,64] 不符（结论 pass 侥幸同向）——GEMM 类多 block 参数文档的维度取用需语义识别而非首个数值对，建议并入 new 文本 2 识别范围（优先匹配 alloc_L0C/L0C 容量句上下文中的 [bl,bp] 组合）
  - 〔2026-09-28 第四现（不同任务，reduction 族）+ 两类新表形态，task argmax-_argreduce_kernel-20260924T031005Z〕design_calc_check 对该 DESIGN v0/v1 两轮检视 + v1 复审（同一任务三轮）ub_budget/core_split 全 skip，仅 r3_metrics 生效；人工复算 ~20 组数字（含 12 组分核算术）是发现膨胀系数数据点混用与 C1 缓冲区间漏 fp32 bm=1 的唯一途径。**两类新表形态**（建议并入识别范围）：① §4.5「B/elem 表 + 逐配置预算表」（列名含 手工预算/×2 双缓冲/结论）；② §5.5「逐 workload 分核表」（列名含 num_row_blocks/num_kernels/核内任务数）。REVIEW.md v0/v1 机械复核表（skip 三行 + 人工复算结论）。**合入优先级建议上调**：四任务（ada/ssd×2/argmax）连续命中，reduction 族最严重的阻塞项（workload (M,N) 差 1024×）正落在 skip 的 core_split 域——与 VP-2026-0136（--manifest 真值推导模式）互补：本条修「表解析」，该条补「真值来源」
  - 〔2026-10-13 第五现（不同任务，pool 族首个迁移）+ 新子缺陷，task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z〕纯 Vector、**任务维 Tiling**的设计几乎必然全部 skip：`design_calc_check.py` 四项中三项 skip（`ub_budget`/`l0c_budget`/`core_split`），仅 `r3_metrics=pass`；`core_split` 按 GEMM 形状（`M/bm`、`N/bn`）解析，对任务维命名（`(n, o_H, c_block)`）恒 skip。**新子缺陷**：`core_split` 的 skip detail 把 §5.5 的小节编号「5」误当 declared logical cores（输出 `"insufficient parsed dims {}; declared logical cores = 5"`）——识别器对小节编号与逻辑核数不做语义区分。`ub_budget` 因 §4.5 表用「**总计（spp CH=64）**」粗体行 + 纯文本层级描述而 skip（第三种未覆盖表形态：单行总量行 + 文本层级）。检视侧后果：纯 Vector 设计的 UB 逐 buffer 求和、`num_logical` 逐 workload 复算、物理核数实查全部落人工；「skip 不得当 pass」已由检视者手工写「机械跳过，人工已核」。建议识别范围并入：① 任务维（非 M/bm）分核表；② §5.5 编号与逻辑核数区分；③ 单行总量 UB 表形态
  - 〔2026-10-13 第六现（不同任务，pool 族同族追加），task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z〕纯 Vector、任务维 Tiling 的设计**再次三项全 skip**（`ub_budget`/`l0c_budget`/`core_split`），仅 `r3_metrics=pass`——与第五现（adaptive_avg_pool2d）同形态：`§4.5` 用「**总计（spp CH=64）**」单行总量行 + 纯文本层级描述（UB 逐 buffer 复算 ≈34.4KB / resnet C4 ≈38.5KB 的手工求和）、`num_logical` 逐 shape 复算落人工（256/24/14/1）。纯 Vector 池化/归约类设计连续两任务全 skip，机械拦截位不生效——建议识别范围并入前述 ①②③（两任务独立佐证，pool 族为第二族证据）
  - 〔2026-10-14 第七现（不同任务，pool 族第三个算子、纯 Vector 任务维 Tiling），task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z〕与第五/第六现**同形态第三次**（pool 族三连）：`design_calc_check.py` 对 §4.5「逐 buffer 清单 + 按编译期分支分路径」的表**三项全 skip**（`ub_budget`/`l0c_budget`/`core_split`，仅 `r3_metrics=pass`）——退化路径 UB 预算的机器拦截位持续不生效，人工复算成为唯一途径，且本次人工复算正是 Stage 2 唯一发现「C4 低报 ≈46%、C4b 漏计模式构造 buffer ≈51KB」（v0 记 90KB/64KB vs 实算 132.8KB/112.1KB）的手段（`REVIEW.md#问题 2`；v1 改逐 buffer 复算 + `CH_deg=min(C,32)` 收窄，见 hardware-cost-model.md §5 第三数据点）。pool 族三连 + reduction/ada 已累计七现，本条修复优先级与 VP-2026-0136（`--manifest` 真值推导）同档。
- repro: 复现条件——DESIGN.md 采用「按 N 分行的驻留预算表」（§4.5）或「workload 表格式分核表」（§5.5）形态时跑 design_calc_check 两个检查项（当前均 skip）
  - 〔provenance，允许失效〕任务内复现命令：python3 .agents/tools/design_calc_check.py check --design examples/ada_layer_norm/_ada_layer_norm_kernel/DESIGN.md
- toolchain_stamp: design_calc_check.py 现行版本；失效环境 tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（两处解析扩展）
    定位锚 1: |
      (check_ub_budget 函数体内)
          if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
              continue  # header rows
    old 文本 1: （即上述定位锚 1 原文）
    new 文本 1: |
          if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
              # Row-N residence tables (N / block_m / 驻留字节 columns):
              # numeric first cell (N) + a per-elem-bytes or 驻留 column ->
              # parse as per-N footprint rows (bytes_per_elem * N) for the
              # UB budget check, so mixed-dtype byte widths are still caught.
              if not (cells and re.match(r"^\d", cells[0]) and any(
                  ("B/elem" in c or "字节" in c or "驻留" in c) for c in cells
              )):
                  continue  # header rows
    定位锚 2: |
      (check_core_split 函数体内，skip 分支)
              "detail": f"insufficient parsed dims {nums}; declared logical "
    old 文本 2: （即上述定位锚 2 原文所在 skip 分支）
    new 文本 2: |
      同分支前增加 workload 表格式识别：首列为 workload 名、含 "MxN"/shape 与 bm 列的表格行，从 (M, bm) 重算 num_logical = ceildiv(M, bm) 与表中声明值对照（无法解析全部维度时保留 skip，但 detail 注明已识别的 workload 行数）
    动机: ada_layer_norm 的 §4.5/§5.5 采用「N 分行驻留表 + workload 分核表」形态，两项机械检查双双 skip，混合字节口径错误只能靠人工逐行复算拦截——正是 VP-2026-0009 定义的机械拦截位，扩展解析使该防线对常见表格形态生效。
- status: merged
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.agents/tools/design_calc_check.py`（两处解析扩展：① §4.5「按 N 分行的驻留预算表」放行；② §5.5 workload 表格式分核表按 `ceil(M/bm)` 重算并对照声明值）。锚文本两处唯一命中、原行保留、纯增量（+147 行 / 0 删除行）。E-4 预验证 PASS：repro = design_calc_check 自身（合成 DESIGN 两表形态）——控制组 ub_budget=skip / core_split=skip，编辑组 ub_budget=pass（2 项 4.5 KB）/ core_split=pass（w1 8/8、w2 64/64 对照成立）；真实 pool DESIGN 两侧同为 skip（无回归）；standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure。实现说明：① 在提案文本的识别条件上补「该表存在 B/elem｜字节｜驻留 列」的表头判据（提案原条件仅扫数据行，对「描述符只在表头」的常见形态不生效）；② 采用表头列语义映射实现对照。残留边界（未出提案范围，记 apply 报告 issues）：描述符列缺失时 ① 仍 skip；`_section(text,'5')` 会先命中 §4.5 编号，纯 Vector 设计的 §5.5 表多不可达。
  - 〔2026-10-14 **第八现（修复后仍复发）**，task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 2〕本任务 08:16 起运行（晚于本条 apply commit `ddf4255` 08:10），对纯 Vector 1D 设计仍 `ub_budget`/`l0c_budget`/`core_split` 三 skip、仅 `r3_metrics` pass——本次表形态为「单行总计式 UB 预算（`G = CH·LSPAN·(2·eb+9) + BL·CH·(25+2·eb)`，无 `B/elem` 列）」+ 任务维分核，正落在上述**残留边界**（描述符列缺失 / `_section(text,'5')` 先命中 §4.5）。检视侧独立兜底位另立 queue **VP-2026-0184**；本条不重开（merged 终态），复发事实记 stats.md §3。
  - 〔2026-10-14 **第九现（apply 后仍复发；不同任务，reduction 族第二次）**，task argmin-_argreduce_kernel-20261014T151121Z Stage 2〕本任务 15:25 起运行（晚于 apply commit `ddf4255` 08:10），`design_calc_check` 对 §4.5/§5.5 的 **Markdown 表格式预算表与分核表** 三项全 skip（`ub_budget: skip`〔detail `no per-level buffer tables parsed`〕/ `core_split: skip`〔detail `insufficient parsed dims {}`〕/ `l0c_budget: skip`），仅 `r3_metrics` 生效——与第四现（argmax，reduction 族首现）同形态，reduction 族内第二次；检视侧全部转人工复算并在 REVIEW.md 机械复核段记录（`examples/argmin/_argreduce_kernel/REVIEW.md` 机械复核段 + RETROSPECTIVE.md Stage 2 Skill Flow Issues 第三行）。属「修复覆盖的表形态之外」的复发（本任务 §4.5 为逐 buffer 表 + §5.5 分核表，非 apply 时新增识别的「N 分行驻留表 / workload 表格式分核表」形态）——本条不再重开，复发事实与残留边界一并记 stats.md §3。

## VP-2026-0020
- type: R
- title: gate 1 S1-REF-PATHS 路径存在性核对未剥离 `#`/`::` 锚后缀与省略号——`DESIGN.md::修订记录`/`.py#L345` 类合法引用被判「引用的仓库路径不存在」
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（同 VP-2026-0019 的两次 Stage 1 runtime fail）
  - .agents/tools/gate_lint.py#L64-84（_ref_path_failures：REF_PATH_RE 的排除字符类不含 ASCII `:`/`#`，token 连同锚后缀一起做 os.path.exists 判定）+ #L42-44（REF_PATH_RE 定义）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`examples/.../DESIGN.md::修订记录` 与 `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345` 两个 token 均含锚后缀导致 exists 判 false；发生账户同 VP-2026-0019（conductor 终态钩子输入 + 复盘缺口）
  - 〔2026-10-13 第五现（不同任务，pool 族）+ 两个提案原文未覆盖的形态，task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z〕Stage 1 gate-1 `S1-REF-PATHS` 失败 1 次（3 处 token 误判），新增形态：① `examples/TileOPs/tests/ops/test_pool.py:130`——ASCII `:行号` 紧贴路径（本提案 `rstrip(".,;:!?、。】")` 只剥**尾部**冒号，`:` 处于 token 中段，未被覆盖）；② `.../bench_pool.py:test_adaptive_max_pool2d_bench`——`...` 缩写与 `:函数名` 并存（本提案 `"/..." in token` 可覆盖 `...`，但 `:函数名` 后缀仍在）；③ `docs/…/T.transpose.md`——**全角省略号 `…`（U+2026）**，本提案只匹配 ASCII `...` 故不覆盖。仅修 5 处引用格式即通过（设计逐字保留，SESSION_TIMING_ANALYSIS §3 估 ≈105s）。**本条修复须扩为**：token 内嵌 ASCII `:` 后接数字/标识符的锚后缀剥离（或改用「路径须以已知扩展名/目录结尾」的正向识别）+ `…` 与 `...` 双省略号形态；设计侧写作自检另立 **VP-2026-0154**（两案互补）。
  - 〔2026-09-15 任务登记的同主题复发（S1-PLACEHOLDER/S1-REF-PATHS 误报系列，见 stats.md §3）与本条合并计数：本条提案 pending 期间已积累 ≥5 起实证，建议审批简报置顶〕
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'(?:docs|examples|testing|\.agents)/[^\s\`\x27\"<>()\[\]（）【】：，。；、！？]+', '见 examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md::修订记录')])"
- toolchain_stamp: gate_lint.py 现行版本；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换 token 清理两行）
    定位锚: |
      (于 _ref_path_failures 函数体内)
          token = m.group(0).rstrip(".,;:!?、。】")
    old 文本: |
          token = m.group(0).rstrip(".,;:!?、。】")
    new 文本: |
          # Strip anchor suffixes (path#heading / path::section) before the
          # existence check; elided paths (examples/foo/...) are not concrete.
          token = re.split(r"#|::", m.group(0), maxsplit=1)[0].rstrip(".,;:!?、。】")
          if token.endswith("...") or "/..." in token:
              continue
    动机: 设计文档以 `path#heading`/`path::section` 引用仓库工件是合法引证形态（本任务 DESIGN/REVIEW 多处使用）；现行正则把锚后缀并入 token 判不存在——已验证修复后 `DESIGN.md::修订记录` → `DESIGN.md`、`.py#L345` → `.py` 正确判定，尾部省略号路径 rstrip 后落到父目录存在性判定。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.agents/tools/gate_lint.py`（`_ref_path_failures` 剥离 `#`/`::` 锚后缀；省略号路径不作具体路径判定）。锚文本唯一命中、原行保留、纯增量（+5 行 / 0 删除行；实现取「先 rstrip 后 split」等价顺序以保留原行）。E-4 预验证 PASS：repro = 锚后缀 token 复现——控制组 `DESIGN.md::修订记录` / `DESIGN.md#ref` 各 1 条 S1-REF-PATHS 误报，编辑组 0；3 个真实 DESIGN 的 gate 1 结果两侧均 `[]`（既有合法样例仍通过）；standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure。残留（未出本批 diff 范围）：第五现登记的 ASCII `:行号`/`:函数名` 中置冒号与全角 `…` 形态未覆盖，建议另立提案。

## VP-2026-0054
- type: R
- title: ab_test.py 输出目录创建后内置 chmod 700——msprof 权限校验拒绝 group-writable 目录，工具自建目录不 chmod 时采集必失败（需 umask 077 前置调用才可避开）
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md#Skill-Retrospective-工具/环境问题（ab_test.py 建目录未 chmod 700，msprof 拒写 "writable by any other users or group users"——需 umask 077 前置调用）
  - .agents/tools/ab_test.py#L50-51（os.makedirs 两处，无 chmod）
  - 交叉引用：queue VP-2026-0037（msprof 父目录权限规则——profile-collection.md 规则侧，本条为工具侧互补修复）
- repro: 复现条件——默认 umask 002 的共享 checkout 下以默认权限建 msprof 输出目录后执行采集（本任务实证形态；VP-2026-0037 的 8/8 首采失败为同机制父目录侧实证）
  - 〔2026-09-17 第三现（不同任务），task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z〕opt_log §9.4——ab_test.py 的 msprof 输出目录 group-writable 拒采再现（R5 蛇形裁决 pair 采集），本任务以预创建 pair 目录规避；两提案（0037 规则侧 / 0054 工具侧）均 pending 未 apply 期间的第三次发生
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-10
- target_doc: .agents/tools/ab_test.py
- delta: |
    动作: update（两处 makedirs 后追加 chmod）
    定位锚: |
      (run_pair 函数体内)
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
    old 文本: |
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new 文本: |
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
          # msprof refuses to write into group/world-writable dirs (same
          # check as VP-2026-0037's parent-dir rule); self-created dirs
          # must be tightened explicitly regardless of umask.
          os.chmod(out_dir, 0o700)
    动机: msprof 权限校验覆盖输出目录自身；工具自建目录依赖调用方 umask 077 是隐式契约——内置 chmod 消除对调用环境的依赖（规则侧 VP-2026-0037 已覆盖「须 700」，本条使工具自身满足该规则）。
- status: merged
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.agents/tools/ab_test.py`（`run_msprof` 自建 `out_dir` 后 `os.chmod(out_dir, 0o700)`）。锚文本唯一命中、原行保留、纯增量（+4 行 / 0 删除行）。E-4 预验证 PASS：repro = `run_msprof` 目录权限行为探针（umask 002）——控制组 `out_dir=0o775`（group-writable，msprof 拒采）、编辑组 `0o700`；standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure。规则侧 VP-2026-0037（须 700）本轮未获批准、保持 pending。

## VP-2026-0116
- type: R
- title: kb_stale_check.py 的 stamp diff 过滤非源码路径——repo HEAD（含 .agents/docs 知识库 commit）被当作 tilelang 工具链戳产生全库假阳性（stale_count=125，实际 tilelang 源码零改动）
- evidence:
  - task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z：opt_log.md §0 E-5 段（启动时 kb_stale_check 报 stale_count=125 几乎全库，本 kernel 直接相关条目全部标「待重验」——五项结构否决的 carry-over 论证本可直接成立，被迫逐条 git diff 自证）+ Skill Retrospective Skill Flow Issues 第 2 行
  - 机械复核（2026-09-21 蒸馏会话）：`git diff --stat 15ad002..96f287e` = 7 个文件全部位于 .agents/（queue/stats/INDEX/attention/cases/constants/repro——evolution commit），tilelang/ 源码零改动；kb_stale_check 现跑 stale_count=124——假阳性复现
  - 根因：.agents/tools/kb_stale_check.py detect_current_stamp() 以 `git rev-parse --short=10 HEAD`（repo HEAD）为 tilelang_commit——知识库文档 commit 也推进 HEAD，使全部旧工具链戳条目批量失配
  - 交叉引用：queue VP-2026-0089（消费侧口径——全局 stale_count 不作单条目判定依据，已三现）/ VP-2026-0101（消费侧两分法）——本条为工具侧根因修复，三案互补建议同批审批
- repro: python3 .agents/tools/kb_stale_check.py（观察 stale_count）；git diff --stat 15ad002..96f287e（docs-only delta 复现命令——两相对照即假阳性证据）
- toolchain_stamp: kb_stale_check.py 现行版本；发生环境 tilelang 0.1.2+96f287eeaa / CANN 8.5.0 / Ascend910B2C / 2026-09-21
- target_doc: .agents/tools/kb_stale_check.py
- delta: |
    动作: update（detect_current_stamp 的 HEAD 查询替换为源码树过滤查询）
    定位锚: |
      (detect_current_stamp 函数体内)
          try:
              head = subprocess.run(
                  ["git", "rev-parse", "--short=10", "HEAD"],
                  cwd=REPO_ROOT,
                  capture_output=True,
                  text=True,
                  timeout=10,
              )
              if head.returncode == 0:
                  stamp["tilelang_commit"] = head.stdout.strip()
          except Exception:  # noqa: BLE001 -- advisory tool
              pass
    old 文本: （即上述定位锚原文）
    new 文本: |
          try:
              # The stamp must track the tilelang SOURCE tree, not repo HEAD:
              # knowledge-base/doc-only commits (.agents/, docs/, examples/)
              # also advance HEAD and would flip the stamp, producing
              # whole-library false positives (2026-09-21 ssd rebuild task:
              # stale_count=125 while `git diff --stat 15ad002..96f287e`
              # showed 7 .agents files and zero tilelang source changes).
              head = subprocess.run(
                  [
                      "git", "log", "-1", "--format=%h",
                      "--",
                      "tilelang/", "src/", "tilelangir/", "3rdparty/",
                      "pyproject.toml", "CMakeLists.txt",
                  ],
                  cwd=REPO_ROOT,
                  capture_output=True,
                  text=True,
                  timeout=10,
              )
              if head.returncode == 0 and head.stdout.strip():
                  stamp["tilelang_commit"] = head.stdout.strip()
          except Exception:  # noqa: BLE001 -- advisory tool
              pass
    动机: stale 检测的语义是「tilelang 行为可能变了」；文档 commit 不改变编译器行为，却使每次蒸馏后的全库条目批量假阳性——optimizer Phase 0 被迫逐条 git diff 自证（本任务实证），或更糟：照单全收「待重验」清单浪费调优预算。源码树过滤（tilelang/ src/ tilelangir/ 3rdparty/ + 构建文件——路径集 apply 时可复核增删）使 stamp 只随真实工具链变更推进。
- status: merged
- confirmations: -/-
- created_by: task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z 2026-09-21
- decided_by: human
- decided_note: 2026-10-14 用户批准（E-3 审批简报，2026-10-14）；mode=apply 落盘 `.agents/tools/kb_stale_check.py`（stamp 由 repo HEAD 改为源码树过滤 `git log -1 -- tilelang/ src/ tilelangir/ 3rdparty/ pyproject.toml CMakeLists.txt`；原 HEAD 查询保留为回退）。锚文本（`detect_current_stamp` / `rev-parse` / `--short=10`）保留、纯增量（+21 行 / 0 删除行）。E-4 预验证 PASS：repro = `python3 .agents/tools/kb_stale_check.py` 实跑——控制组 stamp=`f1e59e23c4`（repo HEAD，evolution commit）stale_count=**183**，编辑组 stamp=`417dae9`（源码树）stale_count=**166**（全库假阳性 −17）；standards_check / kb_lint / `.agents/tools/tests` delta 零新增 failure。

---

## VP-2026-0172
- type: P
- title: 窗口池化退化特化的充分条件须核「窗口输入覆盖集合」而非「输出网格退化」——`out==1×1` 需额外 `d_H==d_W==1`（adaptive 特化的 `p==0 ∧ 覆盖长度` 不够）；且特化条件应写成可执行断言 + 反例回归组
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 1 v0：DESIGN.md §0.6 R4 把 `out==1×1` 写成整平面归约的「当且仅当」条件（只核 `k/p` 与覆盖长度，漏 `d`）（任务工作区 `history_version/design_v0.md`，未上库，provenance 允许失效）
  - 同任务 Stage 2 首轮（1 阻塞）：REVIEW.md 问题 1——独立推演 + torch 数值反例 `H=W=5,k=2,s=1,p=0,d=4`：`out=1×1` 且四合取项全成立，但窗口仅 `{(0,0),(0,4),(4,0),(4,4)}` → 整平面归约给 1.0 而 torch 正确值 0.0；窗口输入覆盖集合 = `{k_h·d_H}×{k_w·d_W}`，`d>1` 时 ≠ `[0,H_in)×[0,W_in)`
  - 同任务 Stage 1 revision v1 + Stage 2 复检（通过）：条件补 `d_H==d_W==1` **六处同步**（§0.6 R4 / §1.6.1 C4 / §3.3 IS_GLOBAL / §5.2 / §5.5 / §8.3），`verify_equiv.py` 57→69 case-group 并新增 **C4x 反例组 + 负向控制守卫**（`assert not c4_condition(...)` + 一般路径对照 golden），69 组全 PASS
  - 同任务 Stage 2 复检发现残留：C4x 两配置均对称 dilation（`dH==dW`），只删 `dW==1` 子句时断言不触发——**子句隔离覆盖**缺口（设计自述「移除任一轴即失败」仅对 H 轴成立）
  - 〔2026-10-14 **第二证**（不同任务、不同算子、1D 域、新工具链 417dae9），task max_pool1d-_max_pool1d_kernel-20261014T082309Z〕1D 同构复现 `d==1` 必要性：反例 `L=5, k=2, s=1, p=0, d=4`（窗口仅 `{0,4}`）整线归约 1.0 vs 正确 0.0；`verify_equiv.py` 117 case-group 全 PASS（含 `[C4x-GUARD]` 段）；kernel L0-11b/d4-guard 与 L0-11b3/d4-bit-exact 实跑走一般路径返回 0.0；**触发形态差异**：2D manifest 三档均不触发任何特化，1D 的 `textcnn-global`（`[64,256,128]` k128s1p0 → `L_out=1`）恰好触发 C4 → 回归组须覆盖「触发成立 / 边界不成立」两侧（设计侧 Stage 1 Value Point Proposals 第 2 行 + Stage 3 实测）
  - 交叉引用：queue VP-2026-0143（adaptive 域几何特化模式集，已合入 ALG-pooling）/ VP-2026-0144（可分离不降运算量）
- repro: repro-missing（知识域最小 repro 待同族池化任务回填；任务内复现 = `/workspace/venv/bin/python -c "import torch,torch.nn.functional as F; x=torch.zeros(1,1,5,5,dtype=torch.float64); x[0,0,2,2]=1; print(F.max_pool2d(x,(2,2),(1,1),(0,0),(4,4),False).item(), x.reshape(1,1,-1).max(-1).values.item())"` → `0.0 1.0`（torch CPU，与设备无关）+ `verify_equiv.py` C4x 组）
- toolchain_stamp: 首证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14（反例本身为 torch CPU 数值复现）；第二证 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（ALG-pooling）
- delta: |
    update ALG-pooling 几何特化段，追加「**窗口池化域**的退化特化条件（与 adaptive 域不可平移）」+「条件型特化回归组」模式：
    ① `out_H==out_W==1`（整平面归约）的充分条件 = `out==1×1 ∧ p_H==p_W==0 ∧ d_H==d_W==1 ∧ (k_H−1)d_H+1 ≥ H_in ∧ (k_W−1)d_W+1 ≥ W_in`——**`d==1` 项不可省**：窗口的**输入覆盖集合** = `{k_h·d_H}×{k_w·d_W}`（`p` 平移、`s` 取窗），`d>1` 时只覆盖等差子集，整平面归约静默错值（反例 `H=W=5,k=2,s=1,p=0,d=4` → 1.0 vs 正确 0.0；`d==1` 且该轴 `in==1` 时该项自动成立）；
    ② `out==in`（恒等拷贝）在窗口池化下**不可平移**——窗口形状由 `k/s/p/d` 固定，`out==in` 只是「窗口重叠的多次比较」而非恒等（反例 `H_in=5,k=3,s=1,p=1` 为 3 元素窗）——恒等特化需 `k_H==k_W==1 ∧ d==1 ∧ p==0 ∧ s==1`；
    ③ **条件型特化回归组**（把充分条件变成可执行断言）：对每个特化充分条件除「条件成立用例」外补一组「边界不成立用例」（如 `d>1` 的 `out=1×1`），断言「不满足条件 → 必须走一般路径」并与 golden 对照；子句级隔离覆盖要求：至少一条用例仅违反单个子句（其余成立），否则「移除任一子句即失败」不可验证（对称 `dH==dW` 只锁一轴）。
- status: merged
- confirmations: 2/2
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14（函数级 task_id；op 级 max_pool2d-max_pool2d-20261014T061506Z）
- confirmed_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14（不同任务、不同算子、1D 域：`d==1` 必要性跨维同构复现 + 触发形态差异）
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务/不同算子（MaxPool2dFwdOp → MaxPool1dFwdOp）且跨越新工具链（417dae9），达 Tier 1 阈值 2/2，本周期合入 algorithm-candidates.md ALG-pooling（新增「窗口池化域退化特化条件」段，含两证来源、④ 触发形态差异与逐子句隔离要求）。**保守限定**：两证为同族不同算子且第二证设计期参照了首证的窗口池化域框架（同 queue VP-2026-0143/0152 先例），机器可验证部分（`verify_equiv` + L0 反例守卫）两任务各自独立执行。

---

## VP-2026-0173
- type: D
- title: NPU 参考后端能力缺口——torch_npu 的 `F.max_pool2d` 经 `aclnnMaxPool2dWithMask` 只支持 `dilation == 1`（`dilation>1` 抛 `AclNN_Parameter_Error(EZ1001)`，kernel 未执行）；纯选择类算子的 CPU golden 回退逐位一致（`atol=rtol=0` 保持）
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 5 attempt 1（`[INTEGRATE_FAIL]`）：`examples/TileOPs/reports/tileops/20261014_071437_810771_MaxPool2dFwdOp/run.json`（`correctness_failed=1`）+ `tileops/kernels/pool/max_pool2d/max_pool2d_kernel/integration_log.md#调试历史`（异常在 `ref_program` 内、op 未执行；`torch_npu` 的 `aclnnMaxPool2dWithMask` golden 仅支持 `dilation == 1`）
  - 同任务 attempt 2（`INTEGRATE_COMPLETED`）：`ref_program` 按 `dilation != 1` 回退 CPU golden（`F.max_pool2d(input.cpu(), **kwargs).to(input.device)`），`run.json.operators[0]` 本算子 22/22、`max_abs_err = 0.00e+00`（逐位）；用例与 `dilation>1` 覆盖保留、`atol=rtol=0` 未放宽（`history_version/test_pool.py.pre_attempt2.bak` md5 4364b26d…→f6a1dcbb…）
  - 机理边界：纯**选择**类算子（max/min/argmax/排序，无累加）的 CPU golden 与设备输出逐位一致；含**累加**的算子（sum/avg/conv/gemm）**不可**照搬该回退
  - 〔2026-10-14 **第二证**（不同任务、不同算子、1D 域、新工具链 417dae9），task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 0/5〕脚手架按 max_pool2d 教训预置同款 CPU golden 回退（`MaxPool1dTest.ref_program`，`examples/TileOPs/tests/ops/test_pool.py`；任务工作区/集成包，未上库），Stage 5 单算子 report 本算子 rollup **34/34**、`max_abs_err = 0.00e+00`，`ecg-cnn-dilated`（`dilation=2`）用例实跑走回退且**无** `AclNN_*_Error`。限定：1D 侧实测的是「补救路径跨算子/跨 stamp 泛化成立」，未再触发 1D 的 `dilation>1` 后端报错本身（回退已前置；缺口本体为 2D 侧直接观测）
- repro: repro-missing（需 torch_npu + Ascend 环境；任务内复现命令 `cd examples/TileOPs && python -m pytest "tests/ops/test_pool.py::test_max_pool2d[full-default-stride-dilation-fp16]" -q`——修复前 FAILED〔异常来自 ref_program，op 未执行〕、修复后 1 passed）
- toolchain_stamp: 首证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c + torch_npu 2.9.0.post8 + torch 2.9.0 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；第二证 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-npuir-ref-backend-golden-capability（golden 基准选择类，置于 TRAP-fp16-opmath-golden 邻位并互链）：**NPU 参考后端（torch_npu）自身的能力缺口**与 npuir 前端/pass 缺口不同类——`F.max_pool2d` 在 NPU 上经 `aclnnMaxPool2dWithMask` 实现，**只支持 `dilation == 1`**；`dilation != 1`（int 或 tuple 任一非 1）时在 `ref_program` 内即抛 `AclNN_Parameter_Error(EZ1001): dilation only support 1`，**op 未执行**（堆栈落在 `torch.nn.functional`）。判别信号：异常出现在 `ref_program` 且 op 尚未执行 + `AclNN_*_Error` → 判为参考后端能力缺口而非 kernel/migration 缺陷。**处置形态**：纯选择类算子（max/min/argmax/排序，无累加）把该用例的 reference 计算回退 CPU（`F.x(input.cpu(), **kwargs).to(input.device)`）——结果与设备逐位一致、`atol=rtol=0` 可保持、用例与其严格判据全保留；**含累加算子不可照搬**（sum/avg/conv/gemm 的 CPU/设备浮点序不同）。配套流程侧：Stage 0 脚手架期的 golden 可用性自检见 queue VP-2026-0178。
- status: merged
- confirmations: 2/2
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- confirmed_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14（不同任务、不同算子、1D 域：CPU golden 回退泛化应用且逐位一致）
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证（1D 泛化应用，新工具链 417dae9）成立后合入 traps-runtime.md **TRAP-npuir-ref-backend-golden-capability**（条目文本把两证分列：缺口本体 2D 直接观测 + 1D 补救路径泛化；并显式标注 1D 未复现故障本身这一限定）。**不登记 `capability-gaps.md`**——AGENTS.md 的缺口层枚举（Frontend API / TileLangIR pass / BishengIR / tladapter / runtime）不含 PyTorch-NPU 参考后端。原 `repro-missing` 状态维持（条目仍缺知识域自包含 repro）。

---

## VP-2026-0174
- type: P
- title: 池化类「容量驱动 CH 选择」——`CH` 取满足 `ub_bytes(CH, W_in, out_W, dtype)×1.7 ≤ 192KB` 的最大 8 的倍数（大 `W_in` 被压档：W_in=112 → CH=32、fp32 → CH=16）；防止代表 shape 外推 + 只算头部大 buffer 低报
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 1：DESIGN.md §4.5（逐 buffer 字节表 + 带字节数不等式）；实测 CH——resnet-stem `W_in=112` fp16 CH=32（CH=64 → 244.5KB ✗）、fp32 CH=16；vgg `W_in=56` / alexnet `W_in=55` fp16 保持 CH=64、fp32 CH=32（任务工作区，未上库，provenance 允许失效）
  - 同任务 Stage 3：实现采用行级 `(W_in, CH)` 掩码后 fp32 resnet-stem 的 CH 由 32 落到 16——**实现形式一改容量结论须重算**（本任务 Transferable Lessons）
  - 〔2026-10-14 **第二证**（不同任务、不同算子、1D 域、新工具链 417dae9），task max_pool1d-_max_pool1d_kernel-20261014T082309Z〕1D 形态同一条不等式的联合取档（`G = CH·LSPAN·(2·eb+9) + BL·CH·(25+2·eb)`，`BL ≤ min(L_out,32)` 由 `_choose_bl` 逐级下调试到首个可行值、`CH` 在 BL=1 仍不可行时折半，下限 8）——`_plan` 实测（实现期实跑）：`textcnn (64,256,128) k128s1p0` 整线路径 fp16 CH=64 → **fp32 CH=32**（`num_logical` 256 → 512）；`sincnet`/`ecg` 保持 CH=64；smoke CH=8
  - 交叉引用：layout.md PL-1.4-tiling-heuristic（BH=1 + 最宽 CH 最优；UB 192KB 约束收缩 CH——本条补逐 buffer 带字节数口径与容量驱动取档规则）/ constants.md CONST-capacity-910B2C（×1.7 膨胀系数）；queue VP-2026-0146（退化路径容量判据，已合入 hardware-cost-model.md §5）；queue VP-2026-0180（同族 BL 方向规则）
- repro: none（设计期逐 buffer 字节核算；Stage 3 编译探针可复核）
- toolchain_stamp: 首证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14；第二证 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/layout.md
- delta: |
    update 条目 PL-1.4-tiling-heuristic（UB 约束 bullet 追加容量驱动取档规则，原文保留）：**容量驱动 CH 选择**——把实现将要落地的 buffer 清单逐项按带字节数核，取满足 `ub_bytes(CH, W_in, out_W, dtype) × 1.7 ≤ 192KB` 的最大 8 的倍数作为 `CH`；大 `W_in` 的窗口池化会被压档（`W_in=112`：CH=64 ≈ 244.5KB ✗ → CH=32；fp32 更降到 CH=16），小 `W_in`（55/56）保持 CH=64。两条纪律：① **不得用代表 shape 的头部大 buffer 结论外推**（低报）——须给「逐 buffer 原始字节 + 膨胀后字节 + 统一 KiB」；② **实现形式改变（如掩码由逐元素 `(1,CH)` 改行级 `(W_in,CH)`）后容量结论必须重算**，不得沿用设计档位（max_pool2d 2026-10-14 实证：fp32 resnet-stem CH 32→16）。
- status: merged
- confirmations: 2/2
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- confirmed_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14（不同任务、不同算子、1D 域：同一条不等式的 1D 形式 + CH dtype 感知折半实测）
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证（1D 同口径 + fp16→fp32 折半实测）达 Tier 1 阈值 2/2，本周期合入 layout.md PL-1.4（新增「容量驱动 CH 选择」bullet，含 2D 与 1D 两套记账形态、`BL`/`CH` 联合取档、两条纪律）；front-matter 的 origin_task/toolchain/dtype/device 同步扩展为多证形态。与 VP-2026-0145（设计盘点两漏点）/ VP-2026-0146（退化路径容量）同族互补。

## VP-2026-0159
- type: D
- title: 双输出池化的源语义三要素（torch 2.9.0 CPU 实测）——`indices` = H×W 平面行主序扁平下标（非二维坐标）／并列取首个命中／窗口含 NaN → 值 NaN + 索引取最后命中的 NaN／`{+inf,-inf}` 同窗 → 值 `+inf`
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 1：`DESIGN.md` §0.1 实测表（12 行 torch 2.9.0 CPU 实测）+ §1.6.1；同目录 `RETROSPECTIVE.md` Stage 1 Value Point Proposals 第二行（任务工作区，未上库，provenance 允许失效）
  - 消费侧实证：`examples/TileOPs/tests/ops/test_pool.py` 用例 `test_adaptive_max_pool2d_indices_nan_window`（NaN 窗 → 值 NaN + last-NaN 索引）+ 单算子 report `run.json`（本算子 rollup 7/7、`avg_max_abs_err=0.0`、int64 索引 `torch.equal` 精确）
  - 交叉引用：queue VP-2026-0125（argreduce 边界语义三基准分野：torch-NPU ±0.0 全序 / NPU reduce 传播 NaN → 哨兵 2^30）——本条补池化索引域的三要素
- repro: repro-missing（torch CPU 语义基准脚本待同族池化/argreduce 任务回填；任务内复现命令 = §0.1 表内 12 行 `adaptive_max_pool2d(..., return_indices=True)` 逐行对照）
- toolchain_stamp: torch 2.9.0+cpu（语义基准）+ tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-pool-index-contract-dual-output（golden 基准选择类，置于 TRAP-argreduce-edge-semantics 邻位并互链）：带索引池化（`return_indices=True`）的源语义三要素——① **`indices` = H×W 平面行主序扁平下标 `ih*W_in + iw`**（int64，**非** `(ih,iw)` 二维坐标）；② 值并列时取**首个**命中；③ 窗口含 NaN → 值 NaN 且索引取**最后命中的 NaN**；④ `{+inf,-inf}` 同窗 → 值 `+inf` 且索引取 `+inf` 位的扁平下标。与 torch-NPU 设备 argmax 的 ±0.0 可区分全序、以及 NPU 硬件 reduce 的 NaN 传播（掩码匹配全失败 → 哨兵 2^30）三方语义各不相同：跨设备 golden 对齐须逐个实测记录，不得统一假设（三方分野详见 VP-2026-0125）。实测依据：12 行 torch 2.9.0 CPU 表 + 集成侧 NaN 窗用例。
- status: merged
- confirmations: 2/2
- confirmed_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14（不同任务、不同算子：双输出 1D 变体的索引契约四要素——索引=沿 L 轴位置 iw、tie 取首个、NaN 取最后命中、前 padding 全 -inf 窗取首个有效元素，+ 空窗口契约 (-inf,0)）
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证（pool 族双输出 1D 变体，不同任务/不同算子）成立后合入 traps-runtime.md **TRAP-pool-index-contract-dual-output**——条目把两证并列（2D 行主序扁平索引 + 1D 单轴位置索引），并补 1D 侧两条边界域（前 padding 全 -inf 窗的索引 = 首个有效元素；空窗口契约 (-inf, 0) 且该域 torch 返回越界索引不可作 oracle）。缺知识域 repro 维持 `repro-missing`（与 VP-2026-0173 合入先例一致）。


## VP-2026-0160
- type: D
- title: 索引归约的极值方向判据——「顺序条件覆写 → 可交换归约」改写中 first-hit（严格 `>` 首写生效）与 last-NaN（无条件覆写末写生效）都取**破纪录集合扁平最大值**（`max`，非 `min`）；误用 `min` 静默产出「窗口内首个局部峰值」索引；D-1 用例须含「多个严格递增破纪录元素」的窗口
- evidence:
  - task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z Stage 1：`DESIGN.md` §1.6.1 N2/N3（v0 `vmin` → v1 `vselect(gt, fv, -1)` + `vmax`）+ §1.6.2；`REVIEW.md#问题 1`（first-hit 次序阻塞项）；`verify_equiv.py` N3 首轮 20426 mismatch → 修正后 340 例全 `EQUIV_PASS`（任务工作区，未上库）
  - 用例形态实证（同任务 `verify_equiv.py`）：初版 `vmin` 下 `rand` 组（如 `(1,4,11,13)→6,6`）**123/144 索引错**，而 `all_equal` / `zeros` / `all_neg_inf` 形态全过——单靠这些退化形态无法暴露极值方向错误
- repro: repro-missing（知识域最小 repro 待同族 argmax/池化任务回填：把候选式改回 `torch.minimum` 即复现方向错误，数行 torch CPU 脚本可验证）
- toolchain_stamp: tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-index-reduce-extremum-direction（语义陷阱类，与 VP-2026-0125/0131 邻位互链）：把「顺序条件覆写 / 记录」改写为可交换归约时，极值方向由源的**写入语义**决定——**首写生效**（严格 `>`，tie 取首）⟹ 取「破纪录集合扁平**最大**值」（`vselect(gt, flat_idx, -1)` + `vmax`）；**末写生效**（无条件覆写，如 NaN 分支）⟹ 取「NaN 位置集合扁平**最大**值」。两者都是 `max`；误取 `min`（"第一个破纪录元素"）得到窗口内首个局部峰值的索引——静默错值、无报错，且 values 侧可能同时正确。D-1 用例形态要求：必须包含「多个严格递增破纪录元素」的窗口（`rand` 在 `H*W≫1` 的窗口）；`all_equal` / `zeros` / `all_neg_inf` 形态在错误方向下也可能全过（实测 123/144 错 vs 退化形态全过）。实测：v0 `vmin` 首轮 20426 mismatch → v1 `max` 后 340 例全 `EQUIV_PASS`。
- status: merged
- confirmations: 2/2
- confirmed_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14（不同任务、不同算子：1D 两遍法——acc 定稿后由等式集合取 min + BIG 哨兵，N2N3 8 配置 × 6 kind × 3 dtype 全 EQUIV_PASS、N2N3-ord 误形态 39/72 检出）
- created_by: task adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z 2026-10-14
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证成立后合入 traps-runtime.md **TRAP-index-reduce-extremum-direction**——第二证补齐「两遍法（累加器定稿后等式集合取 min）」等价形态，并把**候选集口径**写入条目：单遍「破纪录集合取位置最大值」与两遍「等式集合取位置最小值」数值等价但候选集不同，判据表述须绑定候选集（混搭即错）——消除后续设计师把 `vmin` 一律当错误方向的误读风险。缺知识域 repro 维持 `repro-missing`。


## VP-2026-0175
- type: P
- title: 纯选择/带宽受限算子的 roofline Ratio 判读口径——FLOPs 记账口径下 Ratio 只反映比较密度（窗口池化 12.1–14.3% / 自适应 2.3–7.8%），性能判据应改用带宽/流量达成率（实测已达 fabric 上限 1.75 TB/s）
- evidence:
  - task max_pool2d-_max_pool2d_kernel-20261014T062320Z Stage 5：单算子 report 本算子 rollup 22/22 + 9 msprof case 全有效；per-op Ratio 区间 **12.11–14.25%**、带宽 **1.75 TB/s**（≈ CONST-fabric-bw ≥1.73 TB/s 聚合上限）——算子带宽受限而 Ratio 低（任务工作区 report `20261014_072101_015168_MaxPool2dFwdOp/run.json`，未上库，provenance 允许失效）
  - 同族量级对照（不同任务）：adaptive_avg_pool2d Ratio 2.30/3.81/2.35%、adaptive_max_pool2d 5.38/7.77/4.69%、adaptive_max_pool2d_with_indices 2.54/4.57/2.56%——池化族 Ratio 随窗口内**比较/归约 op 密度**变化（窗口池化 k3/k2 高于自适应 bin），而非随利用率变化
  - 交叉引用：queue VP-2026-0081（Ratio 采数管线：Roofline metrics + kernel-name 过滤 + 口径基数核对）；pattern-library CONST-fabric-bw / CONST-vector-launch-overhead
- repro: repro-missing（需 msprof + 目标算子；判读口径 = 同 workload 并读 Ratio 与带宽侧（GM/L2 条目）——带宽近 fabric 上限而 Ratio 低即判「带宽受限、Ratio 不作目标」）
- toolchain_stamp: CANN 9.0.0 msprof + Ascend910B3 / 2026-10-14（采数侧 tilelang 0.1.2+0d3950aa）
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    add 判读口径小节（置于 captured_op 追溯要求段之后，与 VP-2026-0081 的采数管线规则互补）：**纯选择 / 带宽受限算子的 Ratio 判读**——`Ratio = Perf/Computility`、`Perf = msprof 计数 FLOPs / Task Duration`，故对以选择/搬运为主（max/min/argmax/池化类，无累加矩阵乘）的算子，Ratio 反映的是**比较/归约 op 的记账密度**而非硬件利用率：实测池化族 Ratio 2.3–14.3%（窗口池化 k3/k2 高于自适应 bin），而同 workload 带宽已到 **1.75 TB/s ≈ fabric 聚合上限（CONST-fabric-bw）**——即「Ratio 低」在这里不构成优化空间信号。判读动作：① 先读 GM/L2 带宽条目核对是否近 fabric 上限（是 → 带宽受限，Ratio 不作目标）；② 只有计算受限（Cube/矩阵乘类）才把 Ratio 当硬目标；③ 给带宽受限算子立项时用**流量/带宽达成率**作判据，避免为抬 Ratio 做无意义的 op 密度操作（对照 VP-2026-0081 的采数口径三条）。
- status: merged
- confirmations: 2/2
- confirmed_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14（不同任务、不同算子：非 smoke 3 例 Ratio 3.404/5.699/4.151%，`bandwidth_tb_s` 均 1.750 ≈ fabric 上限；benchmark 36/36、msprof_case_count=24、invalid=0）
- created_by: task max_pool2d-_max_pool2d_kernel-20261014T062320Z 2026-10-14
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证（pool 族双输出 1D 变体，Ratio 低而带宽贴 fabric 上限）成立后合入 `.agents/skills/tilelang-op-optimize/references/profile-collection.md` **§4.1 纯选择/带宽受限算子的 Ratio 判读**（含首证 2D 窗口池化 12.11–14.25% 与自适应池化 2.3–7.8% 的密度对照、判读动作三条、CONST-fabric-bw 互链）。


## VP-2026-0180
- type: P
- title: 1D / 长空间轴窗口池化的必需结构——「输出块 BL + 连续 span 载入 + 起点钳制」（2D 的整轴驻留结论不可平移）；BL 取容量允许最大值（halo 重读随 BL 下降）；单空间轴无「第二轴判据外提」；三类尾块
- evidence:
  - task max_pool1d-_max_pool1d_kernel-20261014T082309Z Stage 1：`examples/max_pool1d/_max_pool1d_kernel/DESIGN.md#§0.6 R1/R3` + `#§1.6.1 C1` + `#§4.5` + `#§5.2`（长轴分块判定式与钳制载入形态；任务工作区，未上库，provenance 允许失效）
  - 同任务实现 + 机器断言：`_max_pool1d_kernel.py` `_span/_choose_bl/_choose_ch_bl/_plan`——`span=(BL−1)·s+(k−1)·d+1`、`LSPAN=min(span, L_in)`、`s0=max(0,min(base, L_in−LSPAN))`、读出下标 `j=iw−s0`、输出块尾 `BL_real=min(BL, L_out−ow0)`、`BL ≤ min(L_out, 32)` 由容量不等式取档；`verify_equiv.py` C1 组（`BL∈{4,8,16}` × 7 配置 × 3 dtype 逐 lane 断言 `iw−s0 ∈ [0,LSPAN)`，117 case-group 全 PASS）
  - 同任务 RETROSPECTIVE Stage 1 Transferable Lessons：2D→1D 三处结论不可平移（整轴驻留 → 必须分块；守卫外提 → 1D 不适用；`BH=1` 最优 → 1D 取最大 `BL`）；BL 方向数据点：ecg `k5s2p2d2` BL=31 ≈1.11× GM 读 vs BL=1 ≈4.5×
  - 交叉引用：layout.md PL-1.4（容量驱动 CH 取档，本任务第三数据点已合入）/ PL-1.22 ⑥（守卫外提的「≥2 空间轴」前提）；queue VP-2026-0172（退化特化条件——同族互补，已合入）
- repro: repro-missing（结构形态 + 设计期机器断言；知识域最小 repro 待同族任务回填）
  - 〔provenance，允许失效〕任务内复现条件：跑 `verify_equiv.py` 的 C1 组（`BL∈{4,8,16}` × 7 配置 × 3 dtype 逐 lane 断言 `iw−s0 ∈ [0,LSPAN)`）——任务工作区脚本，不随交付上库
- toolchain_stamp: tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9，dev root build）+ torch 2.9.0 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md（ALG-pooling）
- delta: |
    update ALG-pooling 追加「窗口池化域的必需空间分块形态（长轴 / 1D）」段（与已有「几何特化路径模式集」「窗口池化域退化特化条件」并列）：
    ① **判定式**：空间轴长 × CH × 元素宽度 × 膨胀系数 > UB 容量（192 KiB 档）⇒ 必须沿该轴分块，且分块后读的是**连续 span**（非逐元素 gather）——2D 的 `W_in ≤ 112` 整行驻留结论不可外推（1D `L_in` 可达 4096）；
    ② **钳制载入形态**（保证恒在界内 + 长度静态）：`span=(BL−1)·s+(k−1)·d+1`、`LSPAN=min(span, L_in)`、`s0=max(0, min(base, L_in−LSPAN))`、读出下标 `j=iw−s0`；守卫 `0<=iw<L_in` 与「缓冲下标 ∈ [0,LSPAN)」是等价式，可写成可执行断言（逐 lane 验下标落界）；
    ③ **BL 取容量允许的最大值**：halo 重读随 BL 下降（`LSPAN/(BL·sW) → 1`；ecg 实测 BL=31 ≈1.11× GM 读 vs BL=1 ≈4.5×），与 2D 的 `BH=1` 最优方向相反（2D 整行已驻留，BH>1 只增容量）；
    ④ **轴数前提清单**：整轴驻留 / 守卫外提 / `BH=1` 最优三处 2D 结论均要求 ≥2 个空间轴——同族迁移须逐项核对「该结论依赖几个空间轴 / 轴长上限多少」；
    ⑤ **尾块三类**：任务尾块（`uid < num_logical`）、通道尾块（`real_c = min(CH, C−cb·CH)`）、输出块尾（`BL_real = min(BL, L_out−ow0)`；1D 独有，2D 因整行驻留而无）。
- status: merged
- confirmations: 2/2
- confirmed_by: task max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z 2026-10-14（不同任务、不同算子：双输出（argmax 索引）1D 变体复现同一 `span/LSPAN/s0/j` 形态与三类尾块，C1 组 90 结构检查逐 lane 验下标落界）
- created_by: task max_pool1d-_max_pool1d_kernel-20261014T082309Z 2026-10-14
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证（同一 1D 长轴结构形态在**输出个数不同的变体**上复现）成立后合入 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md` ALG-pooling **「窗口池化域的必需空间分块形态（长轴 / 1D）」**段（判定式 / 钳制载入形态 / `BL` 取最大 / **访存削减正确口径**（吸收本任务 C1 更正：非「读次数降 k_W 倍」）/ 轴数前提清单 / 尾块三类）。


## VP-2026-0125

- type: D
- title: argreduce 边界语义三基准分野——torch-NPU 设备 argmax/argmin 对 ±0.0 用可区分全序（vs torch-CPU IEEE 相等首现）；NPU reduce 传播 NaN → 掩码匹配式输出哨兵 2^30（vs torch 首 NaN 索引）
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 1：探针实测（DESIGN.md §9.2-R1/R2）——`[0,-0,0]`→npu argmax=0/argmin=1、`[-0,0,-0]`→argmax=1/argmin=0，CPU 全 0（三 dtype 一致复现）；NaN 行 4 位形：NPU reduce 传播 NaN 使掩码匹配全失败 → 输出哨兵 2^30，torch 返首 NaN 索引；±inf 精确
  - 〔2026-10-14 第二证（不同任务/不同算子/不同设备与 CANN；覆盖 ②），task argmin-_argreduce_kernel-20261014T151121Z Stage 3〕`examples/argmin/_argreduce_kernel/_argreduce_kernel.py:927-946`（L2 NaN record）两次独立实测：`kernel=[1073741824, 63, 33, 152]` vs `torch=[100, 63, 33, 152]`、`kernel=[1073741824, 234, 105, 186]` vs `torch=[100, 234, 105, 186]`——首元素恒为哨兵 `2**30 = 1073741824`，证实 `reduce_min` 的 NaN 传播（非 fmin 忽略语义）；同一行同时证明「掩码链 vs torch 首-NaN 索引」结构可分。910B3 / CANN 9.0.0（首证环境为 910B2C / CANN 8.5.0）。① ±0.0 全序在第二证未复测（保持单源）。
- repro: repro-missing（知识域最小 repro 待同族任务回填；±0.0 复现 = python3 -c "import torch,torch_npu; ..." 对照 CPU/NPU——provenance 允许失效）
- toolchain_stamp: 首证 torch 2.x npu + CPU 对照 + tilelang 0.1.2 dev root build 2026-09-24 + CANN 8.5.0 + Ascend910B2C / 2026-09-24；第二证 tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 条目 TRAP-argreduce-edge-semantics（golden 基准选择类）：① torch-NPU 设备 argmax/argmin 对 ±0.0 用**可区分全序**（argmax 偏好 +0.0、argmin 偏好 −0.0），与 torch-CPU（IEEE 相等取首现）不一致——跨设备 golden 对齐时 ±0.0 混合行是已知分歧点；② NPU 硬件 reduce 传播 NaN，掩码匹配式 argmax（vmax/vcmp 链）在 NaN 输入下全匹配失败输出哨兵 2^30，torch 返首 NaN 索引——含 NaN 的负载须在 L2/边界用例记录行为差（记录不断言崩溃），golden 选择与容差设计据此分野（torch-CPU golden / torch-NPU 设备实现 / 硬件 reduce 语义三方各不相同，设计期逐个实测记录比统一假设更诚实）。
- status: merged
- confirmations: 2/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- confirmed_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14（不同任务 / 不同算子 / 不同设备与 CANN：NaN→哨兵 `2^30` 两次独立实测复现 ②；① ±0.0 全序未复测）
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务（argmin 共享 kernel 迁移，910B3 / CANN 9.0.0），达 Tier 1 阈值 2/2，本周期合入 `pattern-library/traps-runtime.md` 条目 **TRAP-argreduce-edge-semantics**（位于 TRAP-index-reduce-extremum-direction 之后；正文把「判读与处置」并入：`2^30` 恒值列 = 匹配链全失败指纹，须与「匹配链断裂」缺陷指纹区分——本语义行是分野而非缺陷）。合入文本已标注「② 两证、① 单源」，防对 ① 过度泛化。此条目为 `TRAP-pool-index-contract-dual-output` 中「三方分野」段的悬空前向引用（合入后引用闭合）。INDEX.md 同步新增一行。

## VP-2026-0128

- type: P
- title: TileOPs reduce 族 Op 层 pad 契约与 host pad 消除——kernel 声明宽度须等于 Op 层 padded 宽度（错位症状：行 0 对、行≥1 全错不报错）；高杠杆项 = `_kernel_handles_padding=True` + kernel 原始 N 契约（小 N 负载最高 64× 流量改善）；argmin 共享 kernel 时 `_pad_value` 须 +inf
- evidence:
  - task argmax-_argreduce_kernel-20260924T031005Z Stage 5：integration_log.md attempt-1（初次 report 17 failed 与 N_padded≠N 用例 100% 重合；最小复现：声明 (8,300) 喂 (8,300) 全对、Op 层 pad 到 (8,512) 后行 ≥1 全错——步长错位）；修复 = forward/__init__ 传 N_padded，42/42 通过
  - host pad 消除证据：tileops/ops/reduction/reduce.py 模块注释（"until their kernels are converted"——仓库既定方向）+ DESIGN.md §0.6-R3/§5.5 + 前次任务 perf_opt/logs（3d 负载 ~195–204µs 主因 host pad，262KB/4KB 流量比）；Stage 5 集成 bench 对照：padded 路径 bm=32@N_padded=256 = 5145µs vs standalone 原始 N narrow 路径 12.5µs
  - argmin 方向性：argmax.py wrapper __init__ 注释 + integration_log 遗留说明（-inf pad 永不赢 argmax、全 -inf 行 kernel 与 torch 同返 0；argmin 必须 +inf）
  - 〔2026-10-14 第二证（不同任务/不同算子/不同设备），task argmin-_argreduce_kernel-20261014T151121Z Stage 1/3〕`examples/argmin/_argreduce_kernel/DESIGN.md#§0.2` 双契约对照表 + 实读 `examples/TileOPs/tileops/ops/reduction/argmin.py:76-78`（`_pad_value() → +inf`）与 `reduce.py:91,460-472`（默认 `False` + `F.pad`）；**+inf 方向确认**（方向反向 → pad 列在 argmin 中胜出、静默错值）。新增两点：① **pad 契约下的索引中性**（`verify_equiv.py` E5 组 264 行含全 `+inf`/tie/`±inf` 混合 0 违反：实列索引恒 `< N ≤` pad 列索引 ⟹ 全域计算 ≡ 实列计算 ⟹ 无需掩码/尾 tile 特化）；② **高杠杆翻转的负载画像边界**——argmin manifest 以大 N 负载为主、无窄 N 用例时保留 pad 契约（零 Op 层/wrapper 契约改动）优于 raw-N 翻转；同源推论：`N_padded` 恒 256 倍数 + `_pick_tile_n` 候选含 256 且偏好整除 ⟹ tiled 静态尾 tile `tail ≡ 0`（设计 §5.4 的静态尾 tile 在 pad 契约下不可达）。
- repro: repro-missing（集成契约需 TileOPs 环境；错位最小复现 = 构造 (8,300) fp16 输入，声明 (8,300) + Op 层 pad (8,512) 喂入 → 行≥1 错位——provenance 允许失效；知识域锚点 = cases.md CASE-argmax-argreduce-integration 已 Tier 0 合入）
- toolchain_stamp: 首证 tilelang 0.1.2+ubuntu.22.4.npuir / Ascend910B2C / CANN 8.5.0 / git 013dbbf5 / 2026-09-28；第二证 tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-reduction 行 R4 追加注记：**Op 层 pad 契约**——TileOPs reduce 族 Op 层默认把输入 pad 到 align_up(N,256)（`_ReduceOpBase._prepare_input`，`_kernel_handles_padding=False` 时）；kernel jit 声明宽度须取 N_padded（raw-N 契约与 F.pad 冲突症状：行 0 正确、行 ≥1 全错且不报错）；**迁移高杠杆项** = 翻转 `_kernel_handles_padding=True` + kernel 接原始 N（规约单位元 padding 语义上不改变结果），小 N/非对齐 N 的 dim=0/中间维负载最高 64× 流量改善（harness reduce.py 注释明示为仓库既定方向）；op_kind 涉及 min/max 的共享 kernel 须核对 `_pad_value` 方向性（argmax −inf 安全 / argmin 必须 +inf）。kb_links 补 CASE-argmax-argreduce-integration。
- status: merged
- confirmations: 2/2
- created_by: task argmax-_argreduce_kernel-20260924T031005Z 2026-09-28
- confirmed_by: task argmin-_argreduce_kernel-20261014T151121Z 2026-10-14（不同任务 / 不同算子 / 不同设备与 CANN：+inf 方向确认 + 索引中性机器验证 + 负载画像边界）
- decided_by: evolver
- decided_note: 2026-10-14 蒸馏：第二证来自不同任务（argmin 共享 kernel 迁移初例的同一 pad 契约裁决），达 Tier 1 阈值 2/2，本周期合入 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md` ALG-reduction 行 R4 的 **「R4 附注：TileOPs reduce 族的 Op 层 pad 契约（两证）」**段——按原提案 delta 的四个要点展开（契约与声明宽度 / pad 单位元方向 / **pad 契约下的索引中性**〔第二证新增，机器验证〕/ **高杠杆翻转的边界**〔第二证修正：负载画像决定收益，argmin 无窄 N 负载时保留契约优于翻转〕），front-matter `kb_links` 同步补 CASE-argmax-argreduce-integration。原提案「高杠杆项最高 64×」的表述在合入文本中按第二证加负载画像前提，防无条件泛化。
