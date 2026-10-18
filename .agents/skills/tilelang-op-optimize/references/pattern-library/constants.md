# 硬件常数表（设计期 roofline 与调优定量口径）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。**本表是已实测硬件常数的唯一事实源**（D-2：evolver 蒸馏新 D 类常数时追加进本表，Tier 0）；设计期 roofline 估算协议（如何用本表算下界）见 `_shared/standards/hardware-cost-model.md`。条目带 front-matter（schema 见 INDEX.md §3），`source` 指向首次实测条目 ID（provenance，允许失效——常数数值与适用条件已自包含于本表）。
>
> **使用纪律**：① 引用常数须带版本戳（工具链变更后自动待重验，`kb_stale_check.py` 检测）；② 常数绑定设备（当前全部为 Ascend910B2C）；③ 外推前核对适用条件（如 MTE2 曲线绑定 4-set 轮换访问模式）。

---
id: CONST-capacity-910B2C
kind: constant
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z / mish-optimize-2026-08 / multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 r9 反例数据点）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（第三数据点：Expert 显式 alloc 结构依赖确认 + bl=128 容量否决）
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0（L1/L0C）；2026-08-24 build（UB 膨胀）；2026-09-15 重校 6797758；2026-09-16 反例 a13585dc；2026-09-17 ssd 1990aa9fe4
source: PL-1.9-blockwidth / PL-1.9-hardlimits / TRAP-UB-multibuffer-inflation / PL-1.12-task-pipeline-depth2
repro: repro/PL-1.12-bn-clamp-bm-guard.py
---

### 片上容量三硬上限（Ascend910B2C）

- **UB = 192KB/AIV**；auto-multi-buffer 会使 UB 占用**膨胀 ~1.7x**——大 block_size 探索须按膨胀后预算评估（BishengIR `ub overflow, requires N bits while 1572864 bits available` 报错文本反推）。**〔2026-09-15 第五轮重校，工具链 6797758〕auto-multi-buffer=false 下 BishengIR 实际需求仍 ≈ 真实手工预算 × 1.10–1.12**（4 点标定：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%——CG-2026-0008；设计期 UB 表须按此系数放大，且勿漏计小 buffer）。**〔2026-09-16 第六轮 r9 反例数据点，工具链 a13585dc〕系数结构依赖**：显式 alloc 结构（全部 UB buffer 首维 = half 的逐 buffer 显式分配，`--enable-auto-multi-buffer=false`）实测手工核算 214080B vs BishengIR 实际 210080B（ratio **0.981，actual 低于手工**）——×1.10–1.12 上偏系数在该结构不成立，该结构实测 ~12% 余量即足（不必按 1.7x 悲观；正负两向偏差均无文档，逐 buffer 分配可见性诉求见 CG-2026-0008）。档案：attention.md PL-1.12 r9 update；核算镜像 + pre-fix 溢出断言：`repro/PL-1.12-bn-clamp-bm-guard.py`。**〔2026-09-17 ssd_chunk_scan 第三数据点，工具链 1990aa9fe4，Expert 显式 alloc 结构——不计 CG-0008 occurrences，属结构依赖确认〕**：Expert 直差分因子链手工峰值 ~179KB ≈ 上限 93%；v3 实测 +16KB 手工 → 195.1KB 实测报错（偏差 <2%，显式 alloc ≈ 手工核算再证）；bl=128 变体编译实测需 582KB（bs=128）/ 357KB（bs=64）vs 192KB——容量否决（DESIGN §1.6.3 变体 B 以实测常数定格）。
- **L1 (cbuf) = 512KB/核**（`cbuf overflow, requires N bits while 4194304 bits available` 反推，2026-09-08）。双槽 k/v/p 的 bn 上限公式：`2·slots·bn·dim·2 + slots·bm·bn·2 + bm·dim·2 ≤ 512KB`（bm=44 时 bn≤427；wide 单槽模式 bn=512 可行，k+v+p+q=312KB）。
- **L0C (cc) = 128KB**（`cc overflow, requires 1310720 bits while 1048576 bits available` 反推）：`l0c_s[bm,bn]f32 + l0c_o[bm,dim]f32 ≤ 128KB` ⇒ bn=512 时 bm≤51。

---
id: CONST-L1-port-bw
kind: constant
family: [attention, expert, cube]
apis: [T.gemm]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### L1 端口读写带宽 ≈ 148–154 GB/s/核

探针复刻 KV+gemm1 数据流 + 真实 kernel 反推双证。ping-pong 双槽**不能**解除 MTE2（GM→L1 写）与 MTE1（L1→L0 读）的端口读写串行——Cube 操作数流的吞吐地板。用法：persistent 结构的 KV 重读流量（tasks/核 × 2MB × 2）÷ 此率 = 该结构的算子级下界（fa4096 ≈ 125–131µs 实测例）。

---
id: CONST-vector-launch-overhead
kind: constant
family: [attention, expert, elementwise, pooling, window]
apis: [v-prefix ops, T.vmax, T.vcmp, T.vbrc]
dtype: [fp16, f16, fp32, int32]
device: 910B2C, 910B3, 910B4
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z（宽块档 0.5 µs/op）/ max_pool1d-_max_pool1d_kernel-20261016T090104Z（窄 op 档第一证 14.4 ns/op）/ max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261017T235029Z（第二证 + 宽度分档实测 12.3/21.1/23.1 ns/op，910B4）/ all-_logical_reduce_kernel-20261018T070754Z（第三数据点：**行归约 pass 的大宽度分档** 4096/8192/16384 → 0.55–0.60 / 1.00 / 1.00 µs，910B4）
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09（首证 910B2C）；tilelang 0.1.2+c3c39ba（commit c3c39ba）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16（窄 op 档）；tilelang 0.1.2+e8b5d51e（commit e8b5d51）+ CANN 9.0.0 + Ascend910B4（aicore 20 → AIV 40）/ 2026-10-18（宽度分档）；tilelang 0.1.2+4d2e41313438e8b79c25d3fe8020f26078400106（dev root build，.so 构建于 2026-10-13）+ CANN 9.0.0 + Ascend910B4（AIV=40 / UB 196608 B）/ 2026-10-18（第三数据点，`msprof op` Task Duration、`--launch-count=15 --warm-up=3`）
source: PL-1.9-hardlimits; queue VP-2026-0300（2/2）
repro: repro/CONST-vector-launch-overhead.py
---

### 向量算子发射开销 ~0.5µs/op；逐元素速率 f16≈f32

无 f16 打包加速；每 op ~0.5µs 固定发射成本主导——同元素总量下宽块 op（bn=512）恒优于窄块 op 翻倍（bn=256）。用法：向量链时延下界 ≈ op 数 × 0.5µs。

**窄 op 档（2026-10-18 第二证，910B4 / AIV=40；512 深 depend 链，`msprof op` Task Duration median/20，扣同 shape 空 kernel floor）**：单价随**lane 数**分档，`0.5 µs/op` 只适用于宽块（bn≈512 级）——64 lane（`(1,64)`）f32 `vmax` **12.3 ns/op**、512 lane（`(8,64)`）**21.1 ns/op**、1024 lane（`(8,128)`）**23.1 ns/op**（**8× lanes 只贵 1.7×、16× 只贵 1.9×** ⇒ 批处理是省 op 数的杠杆）；同档 data-dependent fp32 `vcmp` **30.5 ns/op**、运行期标量 `T.vbrc` **23.8 ns/op**；**int32 `vcmp` 例外地 1039 ns/op（16.2 ns/lane，标量化，见 traps-compiler.md TRAP-int32-vcmp-scalarized）**；空 kernel Task 地板 **3.12 µs**（`(1,64)`）/ **3.38 µs**（`(8,64)`）。与第一证（910B3，由 `aiv_vec`/op 数反解 **14.4 ns/op/核**）同档 ⇒ **「窄向量 op（≤128 lane）单价 10–25 ns/op」在 910B3/910B4 两设备、两工具链成立**；把宽块单价外推到窄 op 会使设计期「发射项」估算失准一个量级以上（第一证实证：DESIGN 估算 22.2 µs vs 实测 301.936 µs = 13.6×）。**用法（三杠杆取二）**：`Task 下界 ≈ 地板 + op 条数 × 单价 + max(流量项)`——**加宽/批处理**降 op 条数，**改 dtype 域**降单价（int→fp32 判决），**削死代码**同时降两者；`BL` 类参数不改变 op 条数时不要指望收益（第一证 sincnet BL 变化对 op 总数无影响）。

**大宽度档：行归约 pass 按行宽 N 分档（2026-10-18 第三数据点，910B4 / `4d2e413`；`read − copyout` 反解，单平面 `reduce_max(dim=1)`）**：同一 16384 元素总量下，`tn` 宽度直接决定归约 pass 单价——4 个 4096 元素 tile 各 **0.551 / 0.592 / 0.595 µs**（合计 ≈2.37 µs）、1 个 8192 元素 tile **0.999 µs**、1 个 16384 元素 tile **0.999 µs**（8192 与 16384 同价 ⇒ N≥8192 档每元素单价减半后**饱和**，不是线性）。判据意义：加宽 `tn` 确实降低每元素归约代价（本例 −58%），但同批标定的 GM→UB 读单价在 64 KB / W=16384 档由 16 KB 档的 0.129 µs 抬到 **1.508 µs**（见 `CONST-aiv-read-roof-and-tiny-dma-910B4`）且单 tile 必然放弃双缓冲 ⇒ 本形态净亏（32k / 3d 实测 −2.6% / −6.9%）。量纲 = **µs/pass（含该行宽整条归约 lowering）**，与本文窄 op 档的 `ns/op` 不是同一口径、不可互推。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/all/all_kernel/perf_opt/{op_cost_calibration.jsonl（`reduce_pass` 5 行原值）, probe_floor.py, copy_floor_calibration.jsonl}`；工具链见 front-matter 第三数据点。

---
id: CONST-fabric-bw
kind: constant
family: [attention, expert]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### fabric 聚合带宽 ≥1.73 TB/s（24 核 × 72GB/s，pp 探针）

shared=distinct=pp 无争用惩罚。fa4096 级 workload 现聚合 1.41TB/s 未触顶——地板因子排序通常为 L1 端口 / 向量发射率先于 fabric。

---
id: CONST-msprof-ratio-compute-roofline
kind: constant
family: [pooling, reduction, general, migration]
apis: [msprof]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（Stage 4 首证）/ max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 第二证，不同任务/不同算子） / adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z（round-3 分母塌缩分支补充）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15；第二证 tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-16
source: profile-collection.md §4.1（纯选择/带宽受限算子 Ratio 判读口径）/ queue VP-2026-0081（采数管线）/ VP-2026-0227（本条来源提案）
repro: repro-missing
---

### msprof roofline `Ratio` 是 compute-roofline 利用率，**不与 latency 成反比**——结构改写会同时压低分子 `total_fops`

- **口径（官方 report 复现，`tileops.benchmark.msprof_roofline.compute_from_csvs`，与 `visualize_data.bin` Method 1 一致）**：`perf = total_fops / TaskDuration`（`total_fops` = `ArithmeticUtilization.csv` 实测累加的 `aiv_vec_fops`，**非** manifest 理论 flops）；`AI = total_fops / total_bytes`（`total_bytes` = L2 read-miss 数 × 128B cacheline）；`peak = min(AI × gm_bw(1.8 TB/s), computility)`（`computility` = 器件算力峰值按实测 dtype 占比加权）；**`Ratio(%) = perf / peak`**。
- **结论**：**分子与分母都会被结构改写影响**——改写若删减 op 数（窄归约/掩码平面/转置等），`total_fops` 同向下降，时延收益被部分抵消。因此「时延降 X 倍 ⇒ Ratio 升 X 倍」的推论不成立（与 profile-collection.md §4.1 的判读动作衔接）。
- **两证实测（不同任务 / 不同算子，工具链亦不同）**：
  - 首证（`adaptive_avg_pool2d` C4 路径，2026-10-15 report）：时延 169.2634 → 16.7203 µs（**10.12x**）而 Ratio 2.3008% → 2.9020%（**1.26x**）——`total_fops` 16.86M → 2.20M（7.66x）、`performance` 0.1043 → 0.1315 TOps/s（1.32x），分母 `computility` 4.5326 TOps/s 固定。
  - 第二证（`max_pool1d` general/窗口路径，task …20261016T090104Z Stage 4）：`_CH_MAX` 64→128 的结构改写使 sincnet 时延 301.936 → 188.454 µs（**1.60x**）而 Ratio 13.727% → 14.776%（**1.076x**）——`total_fops` 153.4M → 106.0M（−30.9%）；若 fops 不变该比值应为 ≈22.0%。同任务 textcnn/ecg 分别 18.334%→20.053%（时延 1.09x）与 12.229%→15.306%（时延 1.37x）。
  - **边界解释（`performance × duration` 反解 `total_fops`）**：分子**恒定**时本条退化为 `Ratio ∝ 1/duration`（fp8_quant 2026-10-15 实测：`total_fops` 保持 ≈17.83 M ⇒ 21.2x 时延 vs 21x Ratio）；分母随 op 混合比**塌缩**时 Ratio 可虚高（adaptive_avg_pool2d round-3：删 fp32 平面 ⇒ `computility` 4.53→0.28 TOps/s、Ratio 2.81%→45.40% 而时延仅 −2.9%）。判别动作 = 由 `performance × duration` 反解 `total_fops` 并与基线对比。
- **可达性筛查（设 Ratio 硬目标前必算，零成本）**：`ratio_max ≈ (total_fops / 该 workload 的时延地板) / computility`。首证算例：地板 ≈4 µs ⇒ 上限 ≈0.55/4.5326 ≈ **12%** ⇒「Ratio → 70%」对无矩阵乘的池化/选择类算子在 compute-roofline 口径下结构不可达。第二证的算式（窗口路径）：sincnet 70% 要求 T ≤ `105.98M / (0.7×3.806e12)` = **39.8 µs**，而实测 `mte2` 单独 55.482 µs 已超该线；textcnn（带宽受限支）70% 要求 `4.194 MB / (0.7×1.8 TB/s)` = **3.33 µs < 空 kernel 地板 3.37 µs**（`CONST-msprof-task-floor-910B3`）。判据：上限 < 目标时改以 **Task Duration / 流量达成率** 为判据，或按 `perf-feedback.md` §1 走口径不可达处置。
- **口径纪律**：`Ratio` 是「比较/归约 op 的记账密度」，不是硬件利用率；`Ratio` 硬目标与 latency 目标可能直接冲突（round-3 实证：CH=120 使 Ratio 68.97% 而时延 +2.9% 回退）。引用本条时须同注 `computility` 与 `total_fops` 两个口径来源（`ArithmeticUtilization.csv`），并用同一个 Task Duration 口径（msprof op median）。
- **证明**：`repro-missing`——结论来自完整 workload 的端到端 report，无法当场最小化为知识域脚本；最小化形态 = 同一 workload 结构改写前/后各采一次 report（或读 `run.json` 的 `total_fops` / `computility` / `duration` 字段），断言 ① `ratio ≈ (total_fops/duration)/computility`；② `ratio 提升 ≪ latency 提升`；③ `(total_fops/时延地板)/computility < 目标 ⇒ 结构不可达`。溯源（任务工作区，允许失效）：首证 `examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/opt_log.md#§12` + `examples/TileOPs/reports/tileops/20261015_003854_809663_AdaptiveAvgPool2dFwdOp/`；第二证 `examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/opt_log.md#§2.2` + `perf_feedback.md`。

> 〔2026-10-16 蒸馏，Tier 1 合入：queue **VP-2026-0227**（D，`repro-missing`）`confirmations` 1/2 → 2/2（第二证 = task `max_pool1d-_max_pool1d_kernel-20261016T090104Z`，不同任务 / 不同算子 / 不同工具链，且满足该条目自定的第二证条件「以 Ratio 为硬目标且结构改写改变 op 数」）→ 按 merge-policy §2 合入 constants.md；条目正文自包含两证数字与可达性算式。〕

---

id: CONST-mte2-degradation
kind: constant
family: [elementwise]
apis: [T.copy]
dtype: [fp16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z / ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（L2 驻留口径补充）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（指令维度口径补充）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z（段数记账更正）
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0；L2 口径 tilelang 0.1.2+a83118285a / 2026-09-10；指令维度口径 tilelang 0.1.2+1990aa9fe4 / 2026-09-17；段数更正 tilelang 0.1.2+15ad002b3d（与 4515de8 同源，git diff 零改动）/ 2026-09-21
source: PL-1.6-copy-floor / CASE-norm-adalayern-stage4
repro: repro-missing
---

### MTE2 有效带宽随总流量退化（4-set 轮换，fp16 bs4096，48 vector core）

1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；256M 的 3:1 R/W 混合全流量 **~1.26 TB/s 为该设备地板**——"~1.8 TB/s 峰值参考"对混合读写流量不可达（勿据其估算大 N headroom）；1M 档读带宽含 L2 成分。grid-stride（48 核聚集移动窗口）实测优于每核连续分块 +3.2%（256M）。

**L2 驻留口径（ada_layer_norm 2026-09-10 补充）**：sets=1 访问模式（同输入张量逐 launch 复用，Stage 5 bench harness 同款）下 MTE2 有效读带宽 43.1 GB/s/core（聚合 ~2.07 TB/s，2048×4096 fp16，43.1×48 核）——远高于上表 HBM 侧曲线同量级值；带宽数字须注明数据驻留状态。设计期 roofline 按上表 HBM 口径估 L2 驻留 workload 会高估时延（ada 实证：DESIGN 估算 53–67µs vs 实测 90.5µs 基线〔失准主项为发射/重叠〕，调优后 40.5µs 优于估算下界——sets=1 下带宽不是地板项，瓶颈转为 Vector/UB 流量 vec_ratio 0.91）。

- **update（2026-09-17 ssd_chunk_scan Stage 4 追加指令维度口径）**：GM→L1 nd2nz 指令代价 ≈ **300ns/指令**、≈ **4ns/128B 段**（w2 标定：16 指令/任务×32 任务 = 156µs mte2 busy；段数 = ws_lcb 640 + ws_c 256 + x 256 + prev 64/任务）——小块搬运是**指令数/段数受限**而非带宽受限（41GB/s ≪ 148GB/s L1 端口）；减少指令数（合并装载）或增大行宽（128B→256B 段）是仅有的两个削减方向。AIV 重复执行时两 AIV 子块指标镜像（判据见 PL-1.13）。**拆分口径注（2026-09-20 ssd 重跑任务 Stage 2 检视登记）**：与 attention.md PL-1.18 段数墙构成的拆分互斥（彼处记 ws_c 128 + prev 128，本条记 ws_c 256 + prev 64）——总和一致（≈1216 段）而分项矛盾，源出两任务不同估算/标定路径；引用以总段数为准，待 Stage 4 段数墙 profile 复核厘清。**〔2026-09-21 复核解决，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z Stage 4，tilelang 0.1.2+15ad002b3d（与 4515de8 同源）〕段数记账更正**：ssd 每任务 Cube mte2 真实流量 = **240KB（1920 段）**，非旧记 1216 段——prev_states 拷贝在 lt 循环内 4× 重读（512 段/任务）被两套旧拆分共同漏算（本条 2026-09-17 的「16 指令/任务」w2 标定同为误记，实测基线 19.0/任务）；互证：aic_mte2_instructions 19.0/任务 = 1(x)+Σ_lt[1(ws_c)+1(prev)+(lt+1)(band)]。prevhoist 后 192KB（1536 段）/16 指令。段≈128B、~2.9-3.2ns/段的量级不变；引用分项拆分以指令数互证口径为准（attention.md PL-1.18 update①）。

---
id: CONST-copy-floor-method
kind: pattern
family: [elementwise]
apis: [T.copy, T.vadd]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0
source: PL-1.6-copy-floor
repro: repro/CONST-copy-floor-method.py
---

### copy-floor 标定法（方法学常数）

与目标 kernel 同结构（同 persistent/分块/映射）的 3 载入 + 2 vadd 保活（DCE-proof）+ 1 写出探针，一步判定计算链暴露量。实测例：lerp_tensor 7-pass fp32-transit 链 ≥16M 全 dtype delta ≤1.1µs（16M 仅 0.34µs）——auto multi-buffer 将全部向量 pass 隐藏于 MTE2 窗口内。标量削减判定式：mte2_ratio ≥0.95 时 per-tile 标量/分支开销已被隐藏。

---
id: CONST-flag-id-budget
kind: constant
family: [attention, expert]
apis: [sync_block_set, sync_block_wait]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-twophase（VP-2026-0042 证据链）
repro: repro-missing
---

### flag id 预算 ≤ 15/核

跨引擎 per-slot flag 的 id 数量预算上限（第二轮 round-10 否决两遍式时的推理输入之一——该推理本身被第三轮"全局 [S,S] ws + n-block 下标 flag（nk≤16）"推翻，但预算上限本身实测存活）。两相位结构用 n-block 下标作 flag（nk≤16）贴上限可行。

---
id: CONST-store-fixpipe-gm-only
kind: constant
family: [attention, expert, cube]
apis: [store_fixpipe]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits（[DESIGN_LIMIT] 修正附录存活清单）
repro: repro-missing
---

### store_fixpipe 仅支持 L0C→GM（存活硬上限）

S/P 类中间矩阵无法经 fixpipe 直写 L1/UB——跨引擎传输强制 GM 往返（S-f16 传输是字节减半的缓解，非消除）。store_fixpipe f32(L0C)→f16(GM) 为正确数值转换（max_rel 4.7e-4 = 1 f16 ulp）。

---
id: CONST-aicore-910B2C
kind: constant
family: [general]
apis: []
dtype: []
device: 910B2C, 910B3
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z / adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z（跨型号数据点：910B3 = 20）/ argmin-_argreduce_kernel-20261014T151121Z（910B3 第二数据点 + `aivector_core_num` 直查入口）
toolchain: 2026-09 build；910B3 实查 tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 / 2026-10-13；910B3 第二数据点 tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 / 2026-10-14
source: PL-1.7-expert-persistent-boundary
repro: none
---

### 物理核数 = 24 AI Core（Ascend910B2C）；Ascend910B3 实查 = 20

persistent 结构按 24 核设计（logical tasks = ceildiv(S,bm)·H·B；wall-rows = ceil(ceildiv(S,bm)/24)·bm——bm=44 使 S=1024/2048/4096 得 44/88/176 行）。实查命令：`NPUUtils.get().get_aicore_num()`（设计期分核三要素之一，禁止假设值）。

**跨型号数据点（adaptive_avg_pool2d 迁移 2026-10-13，tilelang 417dae99 + CANN 9.0.0，设备 `npu-smi` 显示 910B3）**：`NPUUtils.get().get_aicore_num()` 实查 = **20**（底层 `rtGetAiCoreCount`，`tilelang/utils/npu_utils.cpp:127`），纯 Vector 算子核数翻倍 = **40**——与本条 910B2C 的 24（翻倍 48）不同。核数常数绑定设备型号：跨型号复用本值会得到错误核数（29% 偏差级），实查入口的另一约束是须在项目 venv（如 `/workspace/venv/bin/python`）下执行——系统 `python3` 缺 `torch`/`tqdm` 会使 `import tilelang` 失败而误报为「环境无 tilelang 运行时」。溯源（任务工作区，未上库）：`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md#§5.5`。

**910B3 第二数据点（argmin 迁移 2026-10-14，tilelang ac0522ed + CANN 9.0.0）**：`AscendArch()` 直查三值——`get_aicore_num()` = **20**、`AscendArch().aivector_core_num` = **40**、`ub_cap` = **196608 B**（=192 KB，与 CONST-capacity 的 910B2C 名义值同）；实查入口 `python -c "from tilelang.utils.npu_arch import AscendArch; a=AscendArch(); print(a.aivector_core_num, a.ub_cap)"`（项目 venv 内执行）。**纯 Vector 核数直接读 `aivector_core_num`，无需手算 `get_aicore_num()×2`**——两者在 910B3 上互证一致（40 = 20×2），直查口径少一次心算（分核三要素的核数项）。溯源（任务工作区，未上库）：`examples/argmin/_argreduce_kernel/DESIGN.md#§0 实查表/§5.5`。

---
id: CONST-arange-scalar-materialize
kind: constant
family: [reduction, elementwise]
apis: [T.arange, T.vbrc]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_arange_scalar.py
---

### T.arange (M,W) 物化按标量执行 ~1ns/elem；(1,N)+首轴 vbrc 绕法

`T.arange(idx_j, [0,1], 0)` 把 (M,W) 索引缓冲逐元素标量写入（(2,4096) 实测 8.56µs ≈ 1.05ns/elem，占 hidden-state 链式 47µs 的 18%）。绕法：`T.arange(idx_row, [0,1],0)` 于 (1,N) + `T.vbrc(idx_row, idx_j)` 首轴广播 (1,N)→(M,N)（文档合法 src (1,N,K)→(M,N,K)），实测 46.97→38.41µs（-18.2%）。bm=1 时两形相同——同形 vbrc 触发空广播维 verify fail（见 TRAP-vbrc-same-shape-empty-broadcast）。判据：链式 kernel 的固定开销 ∝ 首个 (M,W) 缓冲元素数且 scalar 占比高。

---
id: CONST-gm-to-ub-bw-dilution
kind: constant
family: [reduction, elementwise, general]
apis: [T.copy]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_strided_colblock_read.py
---

### gm_to_ub_bw 是总时长稀释口径；跨步列块 2D 读本身不慢（误读更正记录）

**误判更正**：曾把 partial 的 gm_to_ub_bw 0.80–1.11 GB/s/core 读成「(M,tn) 跨步列块读慢」——实为**标量化计算拉长总时长、带宽被稀释**（同 profile 的 mte2_active_bw 一直有 24.9GB/s；mte2_ratio 仅 0.05–0.15）。copy-only A/B（(M,tn) 2D 列块读 vs per-row (1,tn) 读，同写出量）：墙钟比 0.98×——**读形态无差异**。方法论：判断搬运瓶颈用 `mte2_ratio × aiv_time`（活跃时长）与 `*_active_bw`，勿用总量/总时长口径；(M,tn) 2D 跨步列块 copy 在段宽 ≥ 数百字节时性能正常（对照：窄「行」(bm,4) 的 16× 是**流量**膨胀〔GM_to_UB_datas 超采〕，非带宽下降——两类现象勿混）。

---
id: CONST-reduce-dims0-skinny
kind: constant
family: [reduction]
apis: [T.reduce_max, T.reduce_min]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_dims0_skinny.py
---

### dims=0 归约在长条形 (K,4) 上逐列串行（文档合法但灾难性慢）

`T.reduce_max(val_f (nchunk,M), g, dim=0)` 在长第一轴窄第二轴（(400,4)/(80,4)）实测 33.4/8.8µs——逐列串行 walk（~21ns/elem）。绕法：shared staging + `T.transpose((K,M)→(M,K), permutation=[1,0])` + dims=1 链（PL-1.1 家族），同数据 3.3µs。dims=1（行向归约）为快路径；dims=0 仅在第二轴足够宽时可用。

---
id: CONST-npuir-f32-division-not-ieee-rounded
kind: constant
family: [quantization, elementwise, general]
apis: [T.vdiv]
dtype: [fp32]
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（op 级 fp8_quant-fp8_quant-20261014T152721Z）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ torch 2.9.0 / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/CONST-npuir-f32-division-not-ieee-rounded.py
---

### npuir fp32 除法不是 IEEE correctly rounded（无正确舍入除法原语）

- **实测（3200×64 f32，Ascend910B3 / CANN 9.0.0）**：`T.vdiv` 于 **2-D 同形/广播** `(M,D)` 形态 **≈6.1% lane 偏 1 ulp**（对照 fp64→fp32 正确舍入商）；`T.Parallel` 体内标量除法式（逐行 scale）**135/256 值偏 1–2 ulp**；`T.vdiv` 的 **(M,1)/(M,1) 列形态逐位精确**（0/256）。偏差随操作数形态变化，非统一 lowering。
- **对量化/编码类算子的后果**：任何与 CPU/torch 参考做**逐字节**比对的量化算子，在参考值精确落在 round-to-nearest-even tie 上时，设备的 ≤1 ulp 商偏差把量化值推向另一侧邻居，产生 ±1 LSB 字节差（本任务 fp16 4/16384、bf16 18/16384 全为 mantissa `n.5` tie；fp32 随机数据 524288/524288 逐字节一致）。
- **绕法**：(a) per-row/per-group 的 scale 除法改走 `T.vdiv` **列形态**（实测精确），逐元素量化值除法保留与参考同形的 `x/scale`（**勿换 reciprocal-mul**——乘法倒数在 tie 处与除法不等价，见 TRAP-vexp2-not-exact-pow2 同族与 queue VP-2026-0212）；(b) 字节级判据须允许「参考 mantissa 落在 tie 的 ≤1 device-ulp 邻域内时取另一侧邻居」，把容忍度显式写进分类器（勿放宽为随机容差）。
- **证明**：`python repro/CONST-npuir-f32-division-not-ieee-rounded.py`（2-D `T.vdiv` 对正确舍入商断言非零 1-ulp 比例并打印实测率；绝对率随环境漂移，锁量级与相对关系）。
- 能力缺口本体见 `.agents/evolution/capability-gaps.md#CG-2026-0021`。溯源（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/{RETROSPECTIVE.md Stage 3, repro/CONST-npuir-f32-division-not-ieee-rounded.py}`。

---

id: CONST-copy-floor-dma-granularity
kind: constant
family: [reduction, elementwise, general]
apis: [T.copy]
dtype: [fp16]
device: 910B3
status: verified
origin_task: argmin-_argreduce_kernel-20261015T023310Z
toolchain: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3（npu-smi 26.1.1；aicore 20 → aivector 40）/ 2026-10-15
source: PL-1.20-nsplit-perrow-partial / CONST-copy-floor-method
repro: repro/CONST-copy-floor-dma-granularity.py
---

### 同形纯拷贝地板随 DMA 粒度变化（(2048,4096) fp16，910B3 / CANN 9.0.0）

- **口径**：`msprof op` Task Duration(us)，`--launch-count=20 --warm-up=5`；同一 persistent 结构（grid = 40 Vector 核、per-core 串行 task 环）、同一 16.78 MB 读 + 16 KB 写流量、标量消费者单元素读保活（避免 DCE）——**唯一变量 = 单次 DMA 的连续粒度**。
- **实测**：16 KB 粒度（(2,4096) 行块）= **15.82 µs**（16.78 MB / 15.82 µs = **1.06 TB/s**，按 1.75 TB/s 聚合口径其 Ratio 上限 ≈ 60.7%）；64 KB 粒度（flat (32768,) 块）= **13.61 µs**（**1.23 TB/s**，Ratio ≈ 70.5%）；64 KB 行块（(8,4096)）= 14.03 µs（与 flat 同量级）。任务内 opt_log 另汇总 32 KB 档 ≈14.6 µs（探针族未单列该档，provenance 见 origin_task）。
- **粒度是真实杠杆，其余同族形态不是**：16 KB 档上的输出 UB 累积/每核一次写（16.31）、每核连续行映射（16.53）、`T.Pipelined(num_stages=3)`（16.05）、GM→fragment 直取（16.14）、每核连续 + Pipelined(2)（15.25）全部落在噪声或 no_gain 内（−3.6% 至 +4.3%），即「同样 16 KB 搬运结构内的重排」无收益，**只有把单次 DMA 粒度做大才降地板**。
- **粒度上限与 UB 预算耦合（本常数的适用边界）**：64 KB 粒度需要 ~4× UB 占用（flat 32768 f16 = 64 KB/缓冲）；而 resident 行块 + 显式向量链的正确算法实测足迹 ≈ **16 B/elem**（bm=4 @N=4096 = 262 KB > 192 KB 档，编译报 `ub overflow, requires 2097408 bits while 1572864 bits available`）——故「大粒度 DMA 地板」与「算法缓冲」在同一 kernel 内**不可共存**，算法形态实际回落到 16 KB 档地板。
- **用法（设计期 roofline）**：估「同数据通路的时延下界」时，地板取值须**同时绑定 DMA 粒度与 UB 预算**（16 KB 档 1.06 TB/s vs 64 KB 档 1.23 TB/s，差 ~16%）；按 64 KB 档地板评估需要 4× UB 的算法形态会低估下界。与 `CONST-copy-floor-method`（标定法与 mte2_ratio 判定式）、`CONST-mte2-degradation`（带宽随总流量退化——本条是**粒度**维度，两者正交）配合使用。
- **证明**：`python repro/CONST-copy-floor-dma-granularity.py`（测量脚本 + 头部断言说明；msprof 重测按头部命令人工执行）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/{opt_log.md#Iteration-1, profiles/round1/, logs/round1/{c_c0_base,d_d1_bm8,d_d3_flat32}.log}`。
- **低端外推（2026-10-16 追加指针）**：连续块粒度之外还有一条**行长（burst）/行距**轴——2D 跨步行块读（行长 128–156 B、行距 8 KB，窗口类 span 载入形态）实测仅 **≈0.38 TB/s**（比 16 KB 档低 ≈2.8×），见 `CONST-dma-burst-rowstride-floor`（同族不同轴，互链）。

---

id: CONST-dma-burst-rowstride-floor
kind: constant
family: [pooling, window, layout, general]
apis: [T.copy, T.transpose]
dtype: [fp16]
device: 910B3
status: verified
origin_task: max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 optimize，`probe_tr` 决定性实验）
toolchain: tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3（`get_aicore_num()` 实查 20 → 纯 Vector 40 / UB 196608 B）/ 2026-10-16
source: CONST-copy-floor-dma-granularity（连续块粒度轴；本条是**行长/行距**轴）/ TRAP-transpose-ub-gm-roundtrip（转置误判的负向对照）
repro: repro/CONST-dma-burst-rowstride-floor.py
---

### 2D 跨步行块读的地板由**行长（burst 长度）与行距之比**决定，不由带宽上限决定（(32,80,4096) fp16 窗口载入形态，910B3 / CANN 9.0.0）

- **口径**：`msprof op` Task Duration(us)（`--launch-count=20 --warm-up=5 --dump=off`，读 `OpBasicInfo.csv` median）；单 kernel 纯搬运探针（`T.Kernel(40)` persistent，逐 unit 载入 `(CH=80, L=64)` fp16 = 80 行 × 128 B、行距 `L_IN=4096` 元素 = 8 KB，做同形 copy 或 `T.transpose([1,0])` 后写出等量数据）。
- **实测**：行长 128 B / 行距 8 KB 的读 `mte2 ≈ 51 µs / 520 KB·核` ⇒ **≈0.38 TB/s**（40 核聚合 20.8 MB / 55.5 µs）；`copy` 档 Task **55.461 µs**（`logs/probe_tr_copy.log` 原始 median 56.061 µs，同档）。对照 `CONST-copy-floor-dma-granularity` 的**连续块粒度**轴（同形纯拷贝、dense 行）：16 KB 档 1.06 TB/s、32 KB 档 ≈14.6 µs、64 KB 档 1.23 TB/s ⇒ 本数据点比 16 KB 档低 **≈2.8×**。
- **同探针的负向对照（排除「窗口类慢是因为要转置」）**：同一探针内 `(80,64)→(64,80)` 的 UB 内 2 轴 `T.transpose` `aiv_vec` 仅 **6.594 µs / 2080 次**，`trans` 档 Task 51.311 µs ≈ `copy` 档 55.461 µs ⇒ **UB 内 2 轴转置本身近乎免费**（`TRAP-transpose-ub-gm-roundtrip` 的降级形态在本 shape 上不触发），时延由**行长 + 行距**决定。
- **设计期用法（量纲）**：窗口/滑窗类的 span 载入 `x[nb, cb:cb+ch, s0:s0+LSPAN]` 的**行长 = `LSPAN·eb`、行距 = `L_IN·eb`**（`eb` = 元素字节）。容量不等式驱动的短 `LSPAN`（sincnet 合并版 `LSPAN=64` 元素 = 128 B）**同时**把 DMA 突发长度压进 128–156 B 档 ⇒ 该路径的搬运地板被容量取档间接决定。估下界时须把 `LSPAN` 当 **DMA 粒度**用，而不是只当 halo 度量（`LSPAN/(BL·sW)→1`）。
- **适用域与边界**：适用于「行块 + 大行距」的 2D 区域读（窗口/滑窗、跨行 stride 采样）；**不适用于整平面/连续块**搬运（其地板见 `CONST-copy-floor-dma-granularity`）。本数据点只覆盖 `行长·eb ≤ 156 B`；**≥512 B 档未测**，不外推（中间档待补）。
- **证明**：`python repro/CONST-dma-burst-rowstride-floor.py --variant {copy,trans}`（测量型：构建 + 跑通 + 打印输出形状；时延按头部 msprof 命令读取，断言为量级/相对关系）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/{probe_tr.py, probe_run_tr.sh, logs/probe_tr_{copy,trans}.log, profiles/probe/tr_{copy,trans}}`；归因侧见该任务 `perf_opt/perf_feedback.md`（`[DESIGN_LIMIT]` 触发项 ② 与「影响面」）。

> 〔2026-10-16 蒸馏，D 类 Tier 0 合入（provenance + 版本戳 + repro 三件套齐备）；repro 为 ED-C 机械转正（逐字节拷贝 `probe_tr.py` + 头部登记块，代码与断言零改动，登记后 `py_compile` PASS）。与 `CONST-copy-floor-dma-granularity` 同族不同轴（该条 = 连续块粒度，本条 = 行长/行距），互链不重复；该条正文已补反向指针。〕

---

id: CONST-aiv-time-decomposition
kind: constant
family: [reduction, elementwise, general]
apis: [T.copy, T.reduce_min]
dtype: [fp16, fp32]
device: 910B3
status: verified
origin_task: argmin-_argreduce_kernel-20261015T023310Z / max_pool2d-_max_pool2d_kernel-20261017T043853Z（sum vs max 判别追加，Stage 4 optimize）
toolchain: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3（aicore 20 → aivector 40）/ 2026-10-15；sum/max 判别数据 tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025 + CANN 9.0.0 + Ascend910B3 / 2026-10-17
source: PL-1.20-nsplit-perrow-partial / CONST-copy-floor-method
repro: repro/CONST-aiv-time-decomposition.py
---

### `Task Duration ≈ max(mte2, vec) + scalar`：AIV 时间分解口径（910B3 / CANN 9.0.0）

- **拟合式（本机三处同形实测）**：`Task Duration ≈ max(mte2, vec) + scalar`——纯拷贝档 `aiv_time 13.10 = mte2 10.92 + scalar 1.59`；完整链式档 `29.05 = 18.86 + 9.73`（MTE2 10.33 已被 auto-multi-buffer 流水隐藏，不入 max）；「纯拷贝 + 1 次 fp16 全宽 reduce_min」档 `17.56 = 10.71 + 5.86`。
- **ratio 的读法**：`aiv_vec_ratio` / `aiv_scalar_ratio` 是 **`aiv_time` 的分解分量**（同档实测 0.649 + 0.335 = 0.984 ≈ 1），不是 Task Duration 的占比；`aiv_mte2_ratio` 属另一条 pipe（0.356），它是「已用掉多少搬运窗口」而非「占了多少总时长」。把 ratio 当总时长占比会把「计算受限」误读为「搬运受限」。
- **判据（本档的用途）**：当 `vec + scalar > mte2` 时该 workload 是**计算受限**——此时即使把搬运完全隐藏，下界仍由该 DMA 粒度的纯拷贝地板给出（见 CONST-copy-floor-dma-granularity），压时延的唯一有效方向是削减 vec/scalar 分量（pass 数、标量地址/边界计算），而不是继续优化搬运。
- **sum vs max 判别（2026-10-17 追加，max_pool2d Stage 4）**：`dur ≈ max(mte2, vec) + scalar` 只在**已发生重叠**时成立；当实测 **`dur ≈ vec + mte2 + mte3`**（三 pipe 忙时之和 ≈ 总时延）时，这个「和」是**串行化签名而非 roofline 下界**——把它当下界读会得出「无可优化项」的错误结论，正确动作是评估软件流水（预取/双 scratch）。实证：resnet-stem f16 `231.5 ≈ 125.5 + 59.2 + 25.7 µs`（三 pipe 零重叠）；行级 ping-pong 预取后同 body 探针 201.0 → **156.6 µs**（vec 108.5 / mte2 98.9，MTE2 被向量工作覆盖），实装 6 例 **−7.0%~−25.0%**。机械判据：`|dur − (vec+mte2+mte3)| / dur < 0.10` ⇒ 串行化；`|dur − (max+scalar)| / dur < 0.10` ⇒ 已重叠。机制与代码形态另见 `repro/PATT-row-pingpong-pipeline.py` 与 queue **VP-2026-0303**。
- **证伪更正（本轮纠正的误判模型）**：本轮一度用「UB 字节吞吐」模型解释链式代价并据此实现 fused 候选，实测证伪——把候选链缓冲（idx/sent/cand）由 fp32 改 fp16 仅 **−1.7%**（30.686 vs 31.229 µs），说明链代价**不是字节主导**，而是向量 pass 数与标量分量主导（与 `CONST-vector-launch-overhead` 的 ~0.5 µs/op 量级一致）。旧模型下「换更窄 dtype 就能大降」的推论不成立。
- **证明**：`python repro/CONST-aiv-time-decomposition.py`（逐 op 类标定脚本 + 头部断言说明；msprof 重测按头部命令人工执行）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/{opt_log.md#Baseline-现象结论, logs/round1/q_*.log}`。

> 〔新增 2026-10-15 蒸馏，origin_task: argmin-_argreduce_kernel-20261015T023310Z（Stage 4 / scenario=optimize）；两条均为 D 类 Tier 0 合入，repro 为 ED-C 机械转正（信息域自包含脚本，代码零改动、仅头部登记）。〕

---
id: CONST-msprof-task-floor-910B3
kind: constant
family: [general, elementwise, reduction, quantization]
apis: [msprof, T.Kernel]
dtype: []
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel optimize Stage 4（2026-10-15）
toolchain: tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40）
source: fp8_quant Stage 4 floor probes（perf_opt/profiles/probe/）+ perf_records.jsonl
repro: repro/CONST-msprof-task-floor-910B3.py
---

### msprof op `Task Duration` 固定地板 + 逐元素 pass 单价（910B3 / AIV=40 / 8k×64 fp16 形）

- **空 kernel 地板**（只写 `(block_m,)` scale 向量、不读 X、无 2D 计算）：**Task Duration = 3.37 µs**（`aiv_time` 1.01 µs）⇒ **平台固定发射/调度开销 ≈ 2.4 µs**（与 adaptive_avg_pool2d Stage 4 的 `~3.6 µs` 固定地板先例同量级、互为独立数据点）。
- **必需流量地板**（读 4KB/tile fp16 + cast + u8 store，无编码器）：**Task = 4.25 µs**（`aiv_time` 1.85，`aiv_vec_time` 0.88）。
- **逐元素 pass 单价**：`(block_m, dim)` = (32/64, 64) UB tile 上单个 tile 级或 2D-loop 逐元素 op ≈ **0.23–0.3 µs/pass /核**（本任务 4 次 ablation 差值：`vlog2`(VLog+VMul) −0.52 µs、`vexp2`(VMul+VExp) −0.34 µs、`T.reduce` dim=64 max −1.06 µs、2D `vdiv` −0.22 µs；~20 pass 的软件 e4m3 编码链合计 ≈ 7 µs）。
- **拟合式（两档同形）**：`Task Duration ≈ aiv_time + 2.4…2.6 µs`（baseline 236.5→238.5；final 9.3→11.3）。可达性下界 ≈ `2.4 + max(流量地板, pass数×单价)`。
- **判据**：给定时延目标 T 时，先扣固定地板与流量地板再谈「编码/计算 pass 预算」——本算子 70%-Ratio 目标（≤5.52 µs）留给编码器的预算仅 1.27 µs（≤4 pass），低于软件 e4m3 编码的 ~14 pass 下限 ⇒ 结构性不可达（见 `perf_opt/perf_feedback.md`）。
- **例外**：`T.vshl` 不适用 0.3 µs/pass 单价——UB tile 上被标量化，实测 ≈ 20 ns/元素（见 traps-compiler.md TRAP-vexp2-not-exact-pow2 代价边注）。
- **证明**：`python repro/CONST-msprof-task-floor-910B3.py --mode {empty,rw}`（构建 + 跑通；时延由头部 msprof 命令读取）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/quantization/fp8_quant/fp8_quant_kernel/perf_opt/probes/probe_floor.py`。

---
id: CONST-reduce-ub-per-element
kind: constant
family: [pooling, reduction, layout]
apis: [T.reduce_max, T.reduce_sum, T.reduce_min, T.vcmp, T.vmul, T.vcast]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015（Stage 4）/ count_nonzero-_logical_reduce_kernel-20261016T005019Z（2-D 行归约形态第二数据点）/ all-_logical_reduce_kernel-20261016T024920Z（第三数据点：bf16 dtype 分档 + 最紧档位运行期通过）/ any-_logical_reduce_kernel-20261016T062400Z（第四数据点：reduce_min 链同价 + 独立重标定 E_cap）/ count_nonzero-_logical_reduce_kernel-20261018T043104Z（第五数据点：`(W,bm)` 档位差 12.03 vs 8.00 B/元素；第六数据点：`W=4096` 档两 dtype 单价相同 ⇒ dtype 分档不可跨 `(W,bm)` 引用）/ all-_logical_reduce_kernel-20261018T070754Z（第七数据点：f32 16384 元素单 tile 12.14 B/元素、不可编译）
toolchain: tilelang 0.1.2+489385e9b98806365b585b6e1f984a5dae337154（commit 489385e）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15；2-D 行归约数据点 tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892 + CANN 9.0.0 + Ascend910B3 / 2026-10-16；第三数据点 tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16；第四数据点 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16；第五/第六数据点 tilelang 0.1.2+c316b0e6b8a1ddd13a6927b70f38a614e8b0474d（源树 417dae9）+ CANN 9.0.0 + Ascend910B4（AIV=40 / UB 196608 B）/ 2026-10-18（部分重验：第五数据点仅 `W=8192/bm=2` 与 `W=23552/bm=1` 两档；第六数据点 `W=4096` × bf16/fp16/fp32 三 dtype 编译探针）；第七数据点 tilelang 0.1.2+4d2e41313438e8b79c25d3fe8020f26078400106（dev root build）+ CANN 9.0.0 + Ascend910B4（AIV=40 / UB 196608 B）/ 2026-10-18（`W=16384, bm=1` f32 编译探针，仅编译期）
repro: repro/TRAP-reduce-ub-per-element.py
---

### BishengHIR 归约 lowering 的 UB 单价（按输入 tile 元素数线性记账）

- **量纲**：B / 平面元素（`required_bytes ÷ (ch × H×W)`）。整平面归约（`(CH,1,H*W)` tile，`reduce dim=2`）在 **ch=32/64/96/128 上单价恒定**（严格线性、无固定项；实测 ch=64 与 ch=128 差异 <5%）——即 UB 只与「归约链形态 × 元素数」有关，**与真实 dtype 缓冲清单无关**。
- **单价表（fp16 平面，`out` 为 `(CH,1,1)`，整数运算在 fp32 域）**：

  | 归约链形态 | B/元素 | 198 KB 预算下 H*W=49 的 CH 上限 |
  |---|---:|---:|
  | 仅 `reduce_max`（无 NaN 追踪） | < 24 | ≥128 |
  | `reduce_max` + **原位** `vmul(flat,flat,flat)` + `reduce_sum` | **≈ 32** | **≈ 120** |
  | `reduce_max` + 非原位平方平面 + `reduce_sum` | ≈ 64 | ≈ 62 |
  | `reduce_max` + fp16 掩码平面 + `reduce_min` | ≈ 96 | ≈ 41 |
  | `reduce_max` + `vcmp(平面自比较)` + bool 平面 + `reduce_min` | ≈ 128 | ≈ 31 |

- **判据**：整平面归约路径的 CH 上限 = `UB_BYTES ÷ (单价 × H*W)`（本设备 `UB_BYTES = 196608`）。**每减少一个「全平面中间量」或降低其位宽**，单价按比例下降 ⇒ 这是解锁更大 CH 的唯一杠杆。
- **纪律（DCE 陷阱）**：UB 探针必须让**每个中间结果都被最终输出消费**，否则 DCE 会删掉未被消费的 reduce 并给出**虚低单价**（实测教训：未消费的 `reduce_min` 让探针给出 32 B/元素 vs 真实的 128 B/元素；同形 `probe_ub.py` level 3/4 全部失真）。
- **机制**：`hivm.hir.vreduce` lowering 为归约链物化多份中间张量（IR 可见 `memref.alloc` + `tensor.insert_slice` + `hivm.hir.vreduce`），规模随输入 tile 元素数增长。关闭 auto-multi-buffer（`pass_configs={"npuir.enable_auto_multi_buffer": False}`）**不改变**该记账（实测 open/close 数字完全相同）。
- **伴随实测（同任务）**：整平面 `reduce` 与 GM→UB `T.copy` **不重叠**——global-1x1 实测 `mte2 = 9.10 µs / 1.92 MB = 211 GB/s`，而同栅格同粒度的纯拷贝标定为 690 GB/s（`T.Pipelined(2/3)` 复测无改善，±1% 噪声内）。
- **证明**：compile-time sweep 见 `repro/TRAP-reduce-ub-per-element.py` 头部（失败信息含 `ub overflow, requires N bits while M bits available`）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/perf_opt/{opt_log.md#3.2, probe_ub2.py}`。
- **形态适用域（2026-10-16 追加，Tier 1 合入 VP-2026-0263 两证：count_nonzero + all 两个 logical_reduce 迁移）**：上表单价绑定 **3-D 整平面 `(CH,1,H*W)` + `reduce dim=2`**。**2-D `(bm,tn)` + `reduce dim=1`** 的行归约链实测单价显著更低、且**按 dtype 分档**：

  | 2-D 行归约链 | 单价（B/元素） | 单 block 元素上限（196608 B） |
  |---|---:|---:|
  | `vcmp(eq) → vcast(i1→f32) → reduce_sum(dim=1)`（fp16 / fp32） | **精确 8.00** | 24576 |
  | `vcmp(eq) → vcast(i1→f32) → reduce_max(dim=1)`（bf16） | **12.00** | 16384 |
  | `vcmp(eq) → vcast(i1→f32) → reduce_min(dim=1)`（fp16 / fp32；**第四数据点**） | **精确 8.00** | 24576 |
  | `vcmp(eq) → vcast(i1→f32) → reduce_min(dim=1)`（bf16；**第四数据点**） | **12.00** | 16384 |

  - **第四数据点（谓词 OR / `any` 类迁移，独立重标定、未借用姊妹数值）**：2-D 行归约 `vcmp(eq) → vcast(i1→f32, rint) → reduce_min(dim=1)` 链（`any` 的补式形态）单价与 `reduce_sum` / `reduce_max` 链**同价**——fp16/fp32 = 8.00、bf16 = 12.00 B/元素，即 `reduce_min` 不额外定价（`vreduce <min>` 与 `<max>` / `<sum>` 同记账）。本单元由 compile-only 探针**独立标定**（`E_cap = 196608 // 2 // 12 = 8192`，按最坏 dtype 12 B 记账），与姊妹 `all` / `count_nonzero` 数字一致但依据自本单元链形态——非数字平移。溯源：`examples/any/_logical_reduce_kernel/{DESIGN.md#§4.5, history_version/design_probe_ub.py 第二节}`（任务工作区，未上库，允许失效）；工具链 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16。
  - **正交维度：归约轴对齐带（2026-10-16，`multi_head_attention` bwd preprocess）**：本条的单价分带自变量是**链形态/dtype**；与 dtype 正交的第二个自变量——**归约轴长度是否为向量宽整数倍**——实测再分两带（对齐 `D % 8 == 0` ≈ 10–12 B/元素 vs 非对齐 **96 B/元素**）。该细分因独立 repro 与独立测量维度单列为 `CONST-reduce-ub-price-axis-alignment`（非重复立项），设计期 UB 预算须同时按 dtype 与对齐性两维取值。
  - **单价口径 = 通过点反解上界（非严格线性常数）**：同一 dtype 在**通过点**（`196608 / 通过元素数`）与**溢出点**（`requires bits / 元素数`）反解可给出不同值——实测 fp16 `bm=1/tn=32768` 溢出仅超 UB 96 B（`requires 1573632 bits` = 196704 B）⇒ 溢出点反解 6.00，而同 32768 元素在 `bm≥2` 档通过点反解 8.00（bf16 两档一致 12.00）。引用该常数做**线外推**前先做两类反解的一致性核对；容量判据以**通过点**（每个配置有直接编译通过点）为准——探针 ladder 的逐 `(dtype,bm,tn)` 通过比「单价 × 元素数」纸面推算可靠。

  - **第五数据点（910B4 / 2026-10-18：单价还随 `(W, bm)` 变化，不只随 dtype）**：同一条 `vcmp(eq) → vcast(i1→f32) → reduce_sum(dim=1)` 链在 **fp16** 上两个档位给出不同单价——`W=8192, bm=2`（16384 元素/块）编译失败 `ub overflow, requires 1577216 bits while 1572864 bits available` ⇒ **197152 B ÷ 16384 = 12.03 B/元素**（仅溢出 544 B）；同任务的 `W=23552, bm=1` 单 tile 通过点 188416 B ⇒ **8.00 B/元素**（与上表 fp16 档一致）。即 **fp16 也能落到 12.0 B/元素档**（与 bf16 档数值同量级但机制不同：这不是 dtype 上抛 f32，而是 `(W,bm)` 组合的记账差）⇒ 按 8 B/元素做线性外推在 `W=8192/bm=2` 档**低估 1.5×**、会漏判 overflow；容量判据必须在目标 `(W, bm)` 上直接探针（与 `CONST-ub-price-grid-dependent` 的栅格维度、本条上文「通过点 vs 溢出点」两条同源）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero_kernel/perf_opt/{logs/round1/*op3*.log, op_cost_calibration.jsonl}`；工具链见 front-matter 第五数据点。
  - **第六数据点（910B4 / 2026-10-18 round-4：`W=4096` 档两 dtype 单价相同 ⇒ dtype 分档不是跨 `(W,bm)` 的常数）**：`W=4096` 的 2-D 行归约指示链上 **bf16 与 fp16 的单价几乎相同**——bf16 `bm=4`（16384 元素）编译失败 `ub overflow, requires 1574400 bits`（= 196800 B）⇒ **12.012 B/元素**（仅溢出 192 B），fp16 同点 round-1 实测 `197152 B` ⇒ **12.03 B/元素**；三 dtype 的**通过点同为 `W=4096, bm=3`（12288 元素）**。⇒ 上表「fp16/fp32 8.00 / bf16 12.00」的 dtype 分档**只在 `(W=23552, bm=1)` 这类通过点成立**，不可跨 `(W,bm)` 引用：`W=4096` 上两者同为 ≈12 B/元素，故 **UB 单价不能用来解释 bf16 的时延差**（该 shape 上 bf16 31.14 vs fp16 26.01 的差全部来自 `vcmp` 的 f32 lift 带来的 2 个额外 vec pass，见 queue `VP-2026-0318`）。判别意义：单价只决定容量档（`_ub_elems`：bf16 15701 vs fp16 24576 元素的驻留上限），时间差看 vec pass 数；容量判据仍须在目标 `(dtype, W, bm)` 上直接编译探针（本轮新增探针形态 = 按显式 `block_m` 绕开 `_select_config` 阶梯读 compiler 报文，58 行 / 3 个配置 / ≈3 s each）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero_kernel/perf_opt/{probe_bf16_ub.py, opt_log.md#§9.5.1}`。
  - **dtype 分档的机制**：bf16 上 `vcmp` 的操作数被 lowering 上抛 f32（IR 可见 `vcast bf16→f32` 作用于两操作数，多 2 个 f32 平面 ⇒ 单价 ×1.5）；「含 `vcmp` 的归约链」的容量估算依赖对目标 dtype 的逐档实测（compile-only 探针约 3 s/配置）——按「单价与 dtype 无关」外推（沿 8 B/元素对 bf16 记账）在负载上直接 UB 溢出。**（第六数据点收窄：×1.5 只在部分 `(W,bm)` 档显现——`W=4096` 上两 dtype 同为 ≈12 B/元素，该档的 `(W,bm)` 记账差与 dtype 机制不可区分；引用时按目标档实测，不得据机制跨档外推。）**
  - **实测通过点**（compile-only 探针反解 + 运行期闭合）：fp16 24576 / fp32 16384 / bf16 16384 元素；竞态安全档 `E_cap = 8192`（按最坏 dtype 12 B 记账）。**最紧档位运行期通过**：bf16 `(256,4096)`、`bm=2` ⇒ 8192 元素 × 12 B = 96 KB = UB/2（恰触双缓冲余量上限）在设备上逐位精确、无需降阶或切列分块路径。
  - **设计期后果（实录）**：把上表 3-D 整平面单价（最高档 ≈128 B/元素）代入 2-D 行形态会高估容量需求 **16×**，使可行负载被误判为不可实现（或过度引入列分块/转置等结构性开销）；反之按 8 B/元素对 bf16 记账则低估 1.5×。⇒ 容量判据按 `bm·N ≤ E_cap`（**元素数**）记账，且**引用本表任一单价前先核对形态适用域与 dtype 档**。
  - **第七数据点（910B4 / 2026-10-18 / `4d2e413`：f32 在 16384 元素单 tile 档也落在 ≈12 B/元素，8 B/元素外推会漏判 overflow）**：同一条 `vcmp(eq) → vcast(i1→f32) → reduce_max(dim=1)` 链的 **f32** 形态在 `W=16384, bm=1`（16384 元素）编译失败 `ub overflow, requires 1590784 bits while 1572864 bits available` ⇒ **198,848 B ÷ 16384 = 12.14 B/元素**（超 196,608 B 预算 2,240 B = 1.14%）。即上表「fp16/fp32 8.00 B/元素 ⇒ 24576 元素上限」在**大行宽档不成立**：f32 的 16384 元素单 tile 实际**不可编译**（第五/六数据点已就 fp16/bf16 收窄 `(W,bm)` 依赖，本点补 f32 且落在 12.14 > 12.0）。把指示平面改 f16（省 2 B/元素）后同形态可编译并经运行期逐位校验（本任务交付形态按 path 分派：resident=f16、column-tiled/tail=f32）。判据：容量档必须在目标 `(dtype, W, bm)` 上直接编译探针，不得由其他档的 B/元素外推。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/all/all_kernel/perf_opt/{logs/round1/L0_rprime*.log, logs/round1/L0_tilecap*.log}`；工具链见 front-matter 第七数据点。
  - 形态与双路径落地（resident / 列分块 + 静态尾块）见 layout.md `PL-1.25-row-reduce-indicator-complement`；上游降级原因（ED-A）：探针为任务工作区 `history_version/design_probe_ub.py`，本形态最小化 repro 待同族任务回填（`repro-missing`）。


---
id: CONST-dual-output-whole-plane-reduce-ub
kind: constant
family: [pooling, reduction, general]
apis: [T.reduce_max, T.reduce_min, T.vreduce, T.vselect, T.vcmp, T.vbrc, T.serial]
dtype: [fp16, int32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z（Stage 4）
toolchain: tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23（commit 67a0441 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
source: TRAP-npuir-arange-pattern-with-whole-plane-reduce
repro: repro-missing
---

### 双输出整平面 argmax 的归约链 UB 单价 ≈ **237 B / 平面元素**（Ascend910B3）

- **口径**：`x: (N, C, 1, H*W)` fp16 整平面块 + 运行期构建的位置表 `(C,1,plane)` int32 + `vecmp(eq)→vselect→reduce_min`（first-hit）
  + `vselect→reduce_max`（last-NaN）+ `reduce_max`（值），**含 auto-multi-buffer ×2**。
  实测（`ub overflow, requires N bits while 1572864 bits available` 反推）：`plane=49` 时 **`ch=8` 编译通过**，
  **`ch=16` 需 1 618 432 bit（202 304 B）> 1 572 864 bit（196 608 B）**；声明 buffer 仅 ~21 B/元素 ⇒ 归约链 lowering 额外 ≈ **237 B/平面元素**。
- **与单输出同族对照**：同族 `adaptive_max_pool2d` 单输出 C4（无索引、原位平方 `reduce_sum` 代 NaN 探测、无位置表）
  实测 **32 B/平面元素**（`_UB_REDUCE_B_PER_ELEM`）⇒ **索引侧的位置表 + eq 掩码 + 候选平面缓冲把单价抬高 ~7x**，
  把 `ch` 从 64+ 压到 **8**；`ch=8` 在 `N=8,C=2048,plane=49` 下产生 `num_local = ceil(2048/40) = 52` 轮，
  实测 Task **262.8 µs**（远差于同 workload 的一般网格路径 144.4 µs）⇒ **该特化对双输出不可采纳**。
- **归约轴形态**：`(C, plane)` 二维 `reduce(dim=1)`（归约轴 = 最内轴）**被标量化**——实测 `ch=64` 时
  `scalar = 98.1 µs`、Task **237.0 µs**、`UB_to_GM = 96 KB/blk`（转置/布局往返）；
  合法形态是「**扁平空间轴**」`(C, 1, plane)` + `dim=2`（同 PL-1.23），或 `(C, plane)` 经 `(plane, C)` 转置后按行 `reduce(dim=1)`（但转置本身会中 TRAP-transpose-ub-gm-roundtrip）。
- **用法**：设计期做「整平面 argmax + 索引」特化的容量判据时用 `237 B/平面元素`（双输出）而非 `32`（单输出）；
  若 `ch` 被压到 ≤8 且 `num_cblk` 很大（`num_cblk >> 物理核数`），该特化在**时延**上不划算——须按 `num_local = ceil(N*num_cblk/vector_cores)` 一并核算。
- 数据来源（任务工作区，允许失效）：`perf_opt/probe_global_reduce.py`（ch=8 精确、ch=16 UB overflow）、
  `perf_opt/probe_global_2d.py`（二维行归约标量化）、`perf_opt/logs/g2d_ch64.log` / `pr_ch8.log`。

---

id: CONST-reduce-ub-price-axis-alignment
kind: constant
family: [attention, reduction, migration]
apis: [T.vcast, T.vmul, T.reduce_sum, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: multi_head_attention-_flashattn_bwd_preprocess_kernel-20261016T015430Z（Stage 3, first_impl attempt 1 + retry_impl attempt 2）
toolchain: tilelang 0.1.2+1bc8dbc3（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16
repro: repro/CONST-reduce-ub-price-axis-alignment.py
---

### 归约链 UB 单价按**归约轴长度是否为向量宽整数倍**分两带：对齐 ≈ 10–12 B/元素 vs 非对齐 **96 B/元素**（Ascend910B3 / CANN 9.0.0）

- **口径**：链形态 = `vcast ×2`（fp16→f32）+ 原位 `vmul`（`o ∘ do`）+ `reduce_sum(dim=2)` + 1-D store staging，三维扁平声明 `(blk_s, 1, D)` + `reduce(dim=2)`，含声明缓冲与 auto-multi-buffer 膨胀；单价由编译期 `ub overflow, requires N bits while 1572864 bits available` 报文反解（不需要 profiler）。
- **分带数据（同一 kernel 只变 `D`）**：
  - **对齐带**（`D % 8 == 0`）：`D = 56/64/72/96/120/192` 全部 COMPILE OK；`D=64, blk_s=384 → E=24576` 溢出报文 = **11.3 B/元素**（`D=64, blk_s=256 → E=16384` 通过）⇒ **≈10–12 B/元素**。
  - **非对齐带**（`D % 8 != 0`）：`D = 2/3/4/…/60/62/63/65/…/127` 全部 **96.0 B/元素**（`D=63, blk_s=64 → E=4032` 即溢出）。
  - **例外**：`reduce_axis_len == 1` 退化为 copy，走便宜带。
- **设计期后果（量级）**：manifest 域（`D ∈ {64,128}`，均 8 的倍数）的真实元素预算比「单一保守上界 42 B/元素」（`E ≤ 4681`）**宽 2–4 倍**——实测 `blk_elems = 8192` 编译通过且 `max_diff = 3.815e-06`，`D=64` 上 `E=16384` 仍可编译。反之域外防御档（如 `D=63` 的尾 lane 用例）在默认元素预算下**直接编译失败**，须由实现层按对齐带自动收敛有效元素预算（本任务收敛到 `blk_s=24`，`E=1512`）。
- **用法**：UB 预算不得用「声明量 × 固定膨胀」的单一口径推全形态；廉价带的存在使「提高元素预算以摊薄发射项」的杠杆在 manifest 域比纸面估算可达区间大一个档位（设计原估 `E ≤ 4681`，实测 ≥16384）。
- **族内位置**：本条是 `CONST-reduce-ub-per-element` 的**对齐带细分**（该条按归约**链形态/dtype** 分带：原位平方和 ≈32 / 非原位 ≈64 / fp16 掩码 ≈96 / bool 掩码 ≈128 B/元素；同库 2-D 行归约按 dtype 分档 fp16/fp32 8.00 / bf16 12.00）。本条给出与 dtype 正交的第二个自变量——**归约轴长度对向量宽的整除性**（非对齐档恰好落在该族「fp16 掩码 ≈96」同一档位上，说明非对齐形态退化为掩码级重型链）。两条目互为形态维度，不重复。
- 溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/{_flashattn_bwd_preprocess_kernel.py 的 _UB_REDUCE_B_PER_ELEM_{ALIGNED,UNALIGNED} / _UB_BUDGET_SAFETY 常量与注释, repro/CONST-reduce-ub-price-axis-alignment.py}`；sweep 原始记录 = 该任务 Stage 3 Skill Flow Issues 第 2 行（probe3.py sweep + L0-10 阶梯）。
- **证明**：`python repro/CONST-reduce-ub-price-axis-alignment.py`（断言：① 对齐 `D=64` 大 tile 在 <20 B/元素下溢出；② 非对齐 `D=63` 在 >60 B/元素下溢出；③ 非对齐/对齐单价比 >3；④ 阳性对照——非对齐 shape 在小 tile 下仍编译且数值正确；`--check-only` 只做编译）。

---

id: CONST-vcast-rint-downcast-bitexact
kind: constant
family: [elementwise, quantization, migration]
apis: [T.vcast, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: multi_head_attention-_flashattn_bwd_postprocess_kernel-20261016T015430Z（Stage 3, first_impl attempt 1）
toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 + Ascend910B3 / 2026-10-16
repro: repro/CONST-vcast-rint-downcast-bitexact.py
---

### `T.vcast(..., round_mode="rint")` 的 f32→f16/bf16 **下转换与 `torch.Tensor.to` 逐位相等**（含全部角点）⇒ 纯下转换算子的门禁可以合法采用逐位判据

- **口径与实测**：`T.vcast(src, dst, round_mode="rint")` 的下转换方向在纯 dtype 下转换算子上对 CPU oracle（`torch.Tensor.to(storage_dtype)`）**0 bit 差异**——逐元素承载算子 115 个用例（L0/L1/L2/Boundary，含次正规、`±0`、`±inf`、`NaN`、fp16 溢出阈值 65520、`1+2^-11` / `1+2^-8` tie、fp32 极值）全部 `bit_mismatch = 0` 且 `max_diff = 0.000e+00`；独立 repro 在 2048 元素（29 个角点铺满 + 混合随机域）上 `f32→f16` / `f32→bf16` 各 **0/2048** 差异，`f16→f32`（控制组）同样 **0/2048**。
- **对照（必须区分）**：同 dtype `f32→f32` 形态**不是**恒等搬运——实测 **2047/2048** 元素 ≠ 源（静默退化为 round-to-integer，见 `TRAP-vcast-same-dtype-rounds`）。因此同 dtype 搬运的载体是 `T.copy`（按 dtype 在 trace 期分派），`vcast` 只用于**真 dtype 变化**；下转换的源/目标 dtype **必须不同**。
- **用法**：纯下转换（cast-only）算子的门禁可采用**逐位相等**（`torch.equal` + NaN 等价）而非容差——前提是显式 `round_mode="rint"`。该条件同时给出「舍入语义可机器验证」的正向结论：`torch.Tensor.to` 与「精确有理数 RNE」在 12471 点 × 2 dtype 上逐位相等，而被否决的 `round`（half-away）/`trunc` 分别差 91/6091（fp16）、82/6278（bf16）——即该等价性检查有判别力，非恒真陈述。
- **边界/量纲**：判据为逐位相等（0 差异），无时延量纲；本任务四档 manifest 与 `blk_elems ∈ {4096, 8192, 16384}` 三档全部一次编译通过。
- **族内位置**：与 `TRAP-vcast-same-dtype-rounds`（同 dtype 退化）互为正反面；`PL-1.17-subfp32-fp32-transit` 是**上抛** fp32 中转模式，本条是**下压**到存储 dtype 的逐位等价结论（repro 头部以其为「最近邻条目」并登记于本条）。
- 溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_flashattn_bwd_postprocess_kernel/{_flashattn_bwd_postprocess_kernel.py L0-12/L0-13 角点用例, verify_equiv.py 的 D-1 结果表, repro/CONST-vcast-rint-downcast-bitexact.py}`。
- **证明**：`python repro/CONST-vcast-rint-downcast-bitexact.py`（断言：① `f32→f16` == `torch.to(float16)` 逐位；② `f32→bf16` == `torch.to(bfloat16)` 逐位（各在角点 + 混合域）；③ `f32→f32` 在 >半数非整数输入上 ≠ 源；④ `f16→f32`（控制组）逐位）。头部另记一条附属陷阱：独立探针/repro 在 `import tilelang` 前未设 `TILELANG_ASCEND_MODE` 时 `jit_npu` 走 Expert 选项，合法 Developer 形态 kernel 会在 bishengir-compile 报**误导性**的 `'hivm.hir.store' op only support store ub to gm currently!`（该附属陷阱无独立 repro，另立 Tier 1 条目跟进）。

---

id: CONST-device-peak-910B3
kind: constant
family: [pooling, elementwise, reduction, general, migration]
apis: [T.copy, msprof]
dtype: [fp16, bf16]
device: 910B3, 910B4
status: verified
origin_task: max_pool2d-_max_pool2d_kernel-20261017T043853Z（Stage 4 optimize，`peak.py` / `run_peak2.py` / `peak_recheck.jsonl` / `logs/peak_*.log`）/ count_nonzero-_logical_reduce_kernel-20261018T043104Z（第四数据点：910B4 自报屋顶 + 纯 Vector 行归约族 4/4 用例的目标不可达闭合；第五数据点：用户裁定的双轨验收口径 + round-4 达标结果）
toolchain: tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025（commit b3a7b21）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3（npu-smi 26.1.1；aicore 20 → AIV 40 / UB 196608 B）/ 2026-10-17；第四/第五数据点 tilelang 0.1.2+c316b0e6b8a1ddd13a6927b70f38a614e8b0474d（源树 417dae9）+ CANN 9.0.0 + Ascend910B4（npu-smi 26.1.1；AIV=40 / UB 196608 B）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 / 2026-10-18（第五数据点的达标时延为 round-4 final 重采，report `20261018_064405_334073`）
source: CONST-fabric-bw（聚合口径）/ CONST-mte2-degradation（总流量退化曲线）/ CONST-copy-floor-dma-granularity（小流量档含 L2 成分）
repro: repro/CONST-device-peak-910B3.py
---

### AIV 侧 GM 可达带宽三口径（910B3 / CANN 9.0.0）：纯连续读 ≈753–779 GB/s、纯写 ≈1060–1115 GB/s、读写混合聚合 ≈0.70–1.06 TB/s

- **口径**：`msprof op --kernel-name=main --launch-count=20 --warm-up=5 --dump=off --aic-metrics=PipeUtilization,Memory` 的 **Task Duration(us) 中位**；单 kernel persistent 探针（40 个 AIV、每核一段连续大块，chunk = 120 KB 或 60 KB，`T.Pipelined` 深度见命令），**唯一变量 = 方向**（纯读保活 / 纯写 / 读写各半）。
- **实测（同一任务内多 run 取中位）**：
  - **纯连续读**：读 256 MiB + 单元素 GM 写保活 355.5–357.2 µs ⇒ **753.3 GB/s**；64 MiB 档 85.6–86.4 µs ⇒ **778.7 GB/s**（`peak_recheck.jsonl` a1/a2）。
  - **纯写**：256 MiB / 247.1–249.0 µs ⇒ **1078–1087 GB/s**；同栅格 32 通道 × 8 行块写标定档 308 KB / 11.52 µs ⇒ **1115 GB/s**（`copy_floor_calibration.jsonl` `wr_c32_r8`）。
  - **读写混合**：256 MiB 读 + 256 MiB 写 / 763.8–773.7 µs ⇒ **694–703 GB/s 聚合**；同任务不同 chunk 档散落在 0.68–1.07 TB/s 区间（任务 `opt_log.md#3B` 汇总口径 682–1065 GB/s）。
- **推论（本条用途）**：任务书/设计常用的「`achieved_bw / 1.75 TB/s ≥ 70%`」= 1.225 TB/s 算法流量，**高于本机单向读上限（0.78 TB/s）约 57%**——属**口径不可达**而非结构未优化；改用实测混合上限（≈1.0 TB/s）评估时同一批算子可达 23%–49%。设计期 roofline 应写「实测三口径 + 引用的总流量档」，不写 1.75 TB/s 单值。
- **与既有条目互链（口径分野，非矛盾）**：① `CONST-fabric-bw` 的「fabric 聚合 ≥1.73 TB/s」是聚合/fabric 口径，不等于 AIV 侧单向 GM 流量可达值；② `CONST-mte2-degradation` 的「256M 3:1 R/W ≈1.26 TB/s 地板」是 910B2C / 3:1 读写比 / 48 core 口径，本条是 910B3 / 1:1 与单向上限；③ `CONST-copy-floor-dma-granularity` 的「16.78 MB 读 1.06 TB/s」是**小流量档**（含 L2 驻留成分），与本条 64–256 MiB 档不矛盾（同一总流量退化轴）。
- **适用域与边界**：AIV（Vector）侧 GM↔UB `T.copy` 流；**不覆盖** Cube/L1 端口（见 `CONST-L1-port-bw`）与 L2 驻留 workload（驻留档可显著更高，见 `CONST-mte2-degradation` 的 L2 口径）。数字随驱动/驻留状态漂移——只锁量级与相对关系（read < write、mixed < read 上限），绝对值仅作参考。
- **第四数据点（910B4 / 2026-10-18，纯 Vector 行归约族；与 70% 目标线的口径差 2.19×）**：本机 msprof 自报 GM 屋顶 = **0.80 TB/s**（`visualize_data.bin` 的 `GM/L2 : GM Read + Write` 条目，4 个 workload 的 baseline/final 全一致；同一份 report 在 910B3 档该字段为 1.75），AIV 单向读 22.4 GB/s/核 × 40 ≈ **0.90 TB/s**（16 KB tile 探针，8/16/32 KB 三档 26.0/22.4/21.5 GB/s/核）⇒ 落在本条三口径带内。**`achieved_bw / 1.75 TB/s ≥ 70%`（=1.225 TB/s）在该形态 4/4 用例上结构性不可达**，逐例的机械判据是「目标线 vs 本机地板」（不是「调优空间还剩多少」）：hidden fp16/bf16 目标 13.71 µs 既低于屋顶地板 20.99 µs 也低于 read-only 地板 23.83 µs（实测 27.59 / 31.17 µs = 屋顶的 76.1% / 67.4%）；seq fp16 目标 0.86 µs **低于空 kernel 地板 2.44 µs**（2.8×）；3d fp16 目标 3.42 µs 低于实测 copy 地板 7.42 µs（2.2×，128 行 / 40 核 = 3.2 波）。换算到自报 0.8 口径后 4 例 Ratio = **34.68–73.43%**（验收 report `examples/TileOPs/reports/tileops/20261018_053642_635995_CountNonzeroFwdOp/`，`Bandwidth` 列 0.8000 逐例一致）。**族内适用**：`count_nonzero` / `all` / `any` 及「纯 Vector + 行归约 + 大 N 读主导」族（同一 AIV 单引擎数据流 GM→UB→向量→UB→GM，结构差异只在补式方向与合并算子）。**不适用 Cube / 矩阵乘算子**——L1 端口带宽本身更高（148–154 GB/s/核，见 `CONST-L1-port-bw`）但只有 Cube 通路能用，纯搬运仍走 AIV MTE2，故把无 matmul 的归约算子放进 Cube 换不来该口径的提升。
- **第五数据点（验收口径双轨，用户裁定 2026-10-18；族验收线 = `0.7 × 设备自报 roof`）**：**台账记账线不变**（`achieved_bw / 1.75 TB/s`，fabric 聚合口径，`/workspace/全量13算子台账.md` 的数字不改）；**本算子族的验收线改为设备自报屋顶的 70%**——本机自报 **0.80 TB/s** ⇒ 验收流量 **560 GB/s**，换算 `(2048,4096)` hidden fp16/bf16 **29.99 µs**（16 793 600 B）、`(32,16384)` seq **1.873 µs**（1 048 832 B）、`(4,128,4096)` 3d **7.49 µs**（4 195 328 B）。**三处独立证据落在同一带（0.70–1.04 TB/s）**：① msprof `visualize_data.bin` 的 `GM/L2 : GM Read + Write` = **0.8 TB/s**（4 workload × baseline/final 全部一致）；② 端到端 report 的 `Bandwidth (TB/s)` 列本机自动写 **0.8000**；③ 独立探针反解——AIV 读 26.0 / 22.4 / 21.5 GB/s/核（8/16/32 KB tile）× 40 核 = **1.04 / 0.90 / 0.86 TB/s**，纯读地板 16.79 MB / 23.83 µs = **0.70 TB/s**。与 fabric 1.75 差 **2.19×**，量纲解释：纯 Vector 算子的唯一通路是**每核单个 AIV 的 MTE2 单向 GM→UB 流**（20–26 GB/s/核），**用 fabric 聚合带宽作分母等于要求单 AIV 流跑出 2.19× 物理上限 ⇒ 判为口径错误，不是调优不足**。**round-4 达标结果**（同一最终候选 sha `faae175e…`，report `reports/tileops/20261018_064405_334073_CountNonzeroFwdOp/`，correctness 37/37 逐位、`avg max abs error 0.00e+00`）：hidden fp16 **27.68 µs / 75.76%**、hidden bf16 **28.02 µs / 74.85%**（bf16 驻留 3 行档，同 run 交错 A/B **−10.55%**，3/3 同向）、seq **3.86 µs / 33.96%**、3d **9.78 µs / 53.61%**（后两例按空 kernel 地板 2.44 µs / copy 地板 7.42 µs 以 `[DESIGN_LIMIT]` 收束，不再投入）；Ratio 分母 = report 自报 0.8000。**族范围**：`count_nonzero` / `all` / `any` 及一切「纯 Vector + 行归约 + 大 N 读主导」算子（池化/norm 族的大 N 读主导 shape 同口径）；**不适用** Cube / 矩阵乘（L1 端口 148–154 GB/s/核 只有 Cube 通路能用，纯搬运仍走 AIV MTE2）。
- **量纲**：GB/s（聚合流量 ÷ Task Duration）。
- **证明**：`python repro/CONST-device-peak-910B3.py --mode read1 --mb 256 --span 61440 --stages 3`（可运行部分 = 构建 + 跑通 + 打印读字节与输出形状；时延按头部 msprof 命令读取，断言为量级/相对关系）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/perf_opt/{peak.py, run_peak2.py, peak_recheck.jsonl, logs/peak_read_256mb_s1.log, logs/peak_read_64mb_s3.log, logs/peak_write_256mb_s1.log, logs/peak_rw_256mb_s2.log, profiles/peak2/}`、`opt_log.md#3B`。

> 〔2026-10-17 蒸馏，D 类 Tier 0 合入（provenance + 版本戳 + repro 三件套）；repro 为 ED-C 机械转正（逐字节拷贝 `peak.py` + 头部登记块，代码零改动，登记后 `py_compile` PASS；103 行 / 4 918 B（其中 19 行 docstring 登记块）——行数超 ED-B 100 行预算，按 PL-1.10 先例不裁剪披露）。本任务 `[DESIGN_LIMIT]` **未触发**：该项是测量导出的硬件事实（非 2× 结构性机会），本条替代的是「目标线口径」而非性能结论。〕

---

id: CONST-groupnorm-smallshape-memfloor-910B4
kind: constant
family: [norm, reduction, elementwise]
apis: [msprof, T.Kernel, T.copy]
dtype: [fp16, bf16]
device: 910B4
status: verified
origin_task: TileOPs-GroupNormFwdOp-20261018T021206Z（GroupNormFwdOp Stage 4）/ all-_logical_reduce_kernel-20261018T070754Z（第二工具链栅格档补齐：8/16/32 块 1.455/1.845/2.511–2.675 µs）/ any-_logical_reduce_kernel-20261018T094101Z（第三数据点：行归约 4k 形态 16 块 1.7713 µs，跨结构 −4.0%）
toolchain: tilelang 0.1.2+2afbf3b17f419e489a958fc9ee611e1d68ec8294 + CANN 9.0.0 + Ascend910B4（AIV=40 / ub_cap 196608 B）；栅格档补齐 tilelang 0.1.2+4d2e41313438e8b79c25d3fe8020f26078400106（dev root build）+ CANN 9.0.0 + Ascend910B4 / 2026-10-18；第三数据点 tilelang 0.1.2+c316b0e6b8a1ddd13a6927b70f38a614e8b0474d（源树 417dae9）+ CANN 9.0.0 + Ascend910B4 / 2026-10-18
source: perf_opt/probe_floor.py + perf_opt/copy_floor_calibration.jsonl
repro: repro/CONST-groupnorm-smallshape-memfloor-910B4.py
---

### 小形状（1.8–4.2 MB）空 kernel 与纯 DMA 地板（910B4 / 40 Vector 核）

- **空 kernel 地板随栅格增长**（同 `T.Kernel(n)` persistent 形态、单元素保活写，`msprof op` median of 20）：**1 block 1.13 µs / 2 blocks 1.42 / 29 blocks 2.32 / 32 blocks 2.46 / 40 blocks 2.89–2.91**。即「固定发射/调度开销」= 1.1 µs 量级 + 每块约 0.045 µs 的栅格代价；给目标线做筛查时**必须用目标栅格的那一档**（用 b2 会低估、用 b40 会高估）。
- **纯 DMA 地板随流量档骤降**（同结构、同栅格、读+写全量、无计算）：
  - 4.10 MB / 29 blocks（bm=3 行块）= **4.39 µs（956 GB/s）**；同栅格 40 blocks/bm=1 = 6.14 µs（684 GB/s）
  - 3.14 MB / 32 blocks（bm=2）= **3.81 µs（844 GB/s）**
  - 1.81 MB / 32 blocks（bm=2）= **3.09 µs（598 GB/s）**
  - copy-floor（在其上 +2 个保活 v-op）= 4.55 / 3.97 / 3.27 µs ⇒ **2 个 pass 可被 DMA 吸收（+0.16–0.18 µs）**，但整行归约链不可（见 `CG-2026-0027`）。
- **用法**：给「小形状 + 整行归约」类算子设 roofline/latency 目标时，先算 `目标线 vs max(空 kernel 地板, 纯 DMA 地板)`；目标线低于地板的下界即口径不可达（本例 4 档目标低于纯 DMA 地板 1.81–3.42x，3 档低于空 kernel 地板）。**不要**用大流量同族常数外推：同族 16.78 MB 档纯拷贝 1.06 TB/s（`CONST-copy-floor-dma-granularity`），本档只有 0.60–0.96 TB/s。
- **第二工具链栅格档补齐（2026-10-18 追加；`4d2e413` / 910B4 / 40 AIV；同 persistent 形态 + 单元素保活写，`msprof op` `--launch-count=15 --warm-up=3`）**：**8 块 1.455 µs / 16 块 1.845 / 32 块 2.511–2.675**，补齐本条 1–2 块与 29–32 块之间的两档空档，并复现同一斜率——逐块代价 (1.845−1.455)/8 = **0.049 µs/块**（8→16 档）、(2.511…2.675−1.845)/16 = **0.042–0.052 µs/块**（16→32 档），与本条「≈0.045 µs/块」一致 ⇒ 该斜率跨工具链（`2afbf3b` → `4d2e413`）稳定。**应用正例**：同任务把 16 块的小 workload 改成 32 块（「填满核」候选）实测 **+0.55 µs（−18.5%）**，与地板预测 +0.67 µs（2.511−1.845）同量级 ⇒ 小 M 读主导形态下多开核的账面收益必须先扣该档差。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/all/all_kernel/perf_opt/{copy_floor_calibration.jsonl（M32_N4096_bm{1,2,4} 的 `empty` 行）, probe_floor.py, logs/round1/}`；工具链 tilelang 0.1.2+4d2e41313438e8b79c25d3fe8020f26078400106 + CANN 9.0.0 + Ascend910B4（AIV=40 / UB 196608 B）/ 2026-10-18。
- **第三数据点（2026-10-18，行归约 4k 形态，`c316b0e` / 910B4）：地板由栅格主导、跨结构近似可平移**：`(32,4096)` bool→f32 行归约（`bm=2`、**16 blocks**、单元素保活写）空 kernel **1.7713 µs**（30 launch mean）——与同栅格档 16 块 **1.845 µs**（`4d2e413` / group_norm persistent 形态）差 **−4.0%**，也与「1.13 + 15×0.045 ≈ 1.81 µs」斜率外推一致 ⇒ 空 kernel 地板可跨结构、跨工具链按**目标栅格**近似平移（跨设备仍禁止复用）。**应用**：该 workload（GM 524 320 B、baseline 2.9833 µs）地板占 **59%**，1.7713 µs 对应带宽上限 **296 GB/s**，低于族验收线 560 GB/s（`CONST-device-peak-910B3` 第五数据点）⇒ 4k 档**结构性不可达**、只能按 Task Duration 判据评估（A/B −4.73%）；同 workload 的 `ind_vmin` 探针 3.1507 µs（`vec` 0.45）⇒ 向量/延迟受限而非带宽受限。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/any/any_kernel/perf_opt/{copy_floor_calibration.jsonl（4k_empty / 4k_ind_vmin 行）, probe_floor.py}`；工具链同上第三数据点戳。
- **互链**：`CONST-msprof-task-floor-910B3`（3.37/4.25 µs，910B3 大 tile 形态；本条是 910B4 小形状多栅格扫描，两机常数不可互换）、`CONST-msprof-ratio-compute-roofline`（Ratio 目标口径）。
- **证明**：`python repro/CONST-groupnorm-smallshape-memfloor-910B4.py --mode both --case image`（可运行部分 = 两个探针构建 + 数值精确性断言；时延按头部 msprof 命令读取）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/norm/group_norm/group_norm_kernel/perf_opt/{probe_floor.py, copy_floor_calibration.jsonl, profiles/round0/, profiles/floor_v2/}`。

---

id: CONST-ub-price-grid-dependent
kind: constant
family: [norm, reduction, elementwise]
apis: [alloc_shared, reduce_sum, npuir.enable_auto_multi_buffer]
dtype: [fp16, bf16, fp32]
device: 910B4
status: verified
origin_task: TileOPs-GroupNormFwdOp-20261018T021206Z（GroupNormFwdOp Stage 4）
toolchain: tilelang 0.1.2+2afbf3b17f419e489a958fc9ee611e1d68ec8294 + CANN 9.0.0 + Ascend910B4（ub_cap 196608 B）
source: perf_opt/probe_ub.py（编译期 `ub overflow ... requires N bits while M bits available` 反解）
repro: repro/CONST-ub-price-grid-dependent.py
---

### 同一缓冲结构的 UB 单价随 block_m 与「persistent 栅格」变化（不可线性外推）

- **结构**：fp16 `stage(bm,D)` + 2 个 fp32 平面 `(bm,D)` + 2–5 个 `(bm,1)` 统计暂存 + 2 次 `T.reduce_sum`（GroupNorm/LayerNorm 族的标准驻留形态）。
- **实测单价（B/元素，用编译期溢出报文反解）**：
  - fp16/bf16 D=4096/6272/7200、**bm=3**（部署档，mb on）→ **12.000–12.002 B/elem**；fp32 D=4096 → **10.001–10.002**
  - fp16 D=4096、**bm=4**、mb **on**、**32-block 栅格** → **14.002 B/elem（229408 B）⇒ 不存在**
  - fp16 D=4096、**bm=4**、mb **on**、**40-block 栅格** → **12.002–12.004 B/elem（196640–196672 B）⇒ 仍超 64 B，不存在**
  - fp16 D=4096、**bm=4**、mb **off** → **COMPILES（≤12.0 B/elem）⇒ 存在（本轮 16.1% 收益的来源）**
  - fp16 D=6272、bm=3、mb off → 12.002 B/elem（225824 B）⇒ 不存在（wider 保持 bm=2）
- **两条反直觉结论**：① 同一结构同一 bm，**仅换栅格**（32 vs 40 blocks）就使单价 12.0 → 14.0 B/elem（+2.0 B/elem，多出一层 fp16 平面）；② 去掉 2 个 `(bm,1)` 暂存（原地统计变体）**并不降低** bm=4 的 14.0 B/elem —— 编译器的缓冲复用/多层判定不是「缓冲个数 × 元素数」的线性函数。
- **用法**：判「bm 能不能再大」必须**在目标 (dtype 路径 × bm × 栅格 × 缓冲形态) 上直接用编译探针读报文**；任何跨 shape/跨栅格/跨工具链的线性外推（含 DESIGN 期用同族前作 20 B/elem 做预算）都可能错 1.7x 以上。DESIGN §4.5 的 20 B/elem 在本 kernel 上高估 67%。
- **互链**：`TRAP-UB-multibuffer-inflation`（多缓冲记账膨胀的原始陷阱；本条补「栅格」这一维与「mb off 换 bm」的自由度）、`CONST-reduce-ub-per-element`（归约链形态分档 32/48/64 B/元素）。
- **证明**：`python repro/CONST-ub-price-grid-dependent.py`（编译探针：断言 `mb on + bm=4` 必须报 overflow、`mb off + bm=4` 必须编译；字节数与 B/elem 仅打印）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/norm/group_norm/group_norm_kernel/perf_opt/{probe_ub.py, copy_floor_calibration.jsonl（ub-* 行）, logs/round3/}`。

---
id: CONST-aiv-read-roof-and-tiny-dma-910B4
kind: constant
family: [reduction, pooling, elementwise, migration]
apis: [T.copy, T.reduce_sum, msprof]
dtype: [fp16, bf16, fp32]
device: 910B4
status: verified
origin_task: count_nonzero-_logical_reduce_kernel-20261018（Stage 4 optimize，scenario=optimize）/ all-_logical_reduce_kernel-20261018T070754Z（读单价第二自变量：每 tile 行数 × 并行读者数，5 点 0.129–1.508 µs / 43–171 GB/s/核）
toolchain: tilelang 0.1.2+c316b0e6b8a1ddd13a6927b70f38a614e8b0474d（dev root build；tilelang/ 源树 commit 417dae9）+ CANN 9.0.0 + Ascend910B4（npu-smi 26.1.1；AIV=40 / UB 196608 B）/ 2026-10-18；第二自变量数据点 tilelang 0.1.2+4d2e41313438e8b79c25d3fe8020f26078400106（dev root build）+ CANN 9.0.0 + Ascend910B4 / 2026-10-18
source: perf_opt/{probe_floor.py, copy_floor_calibration.jsonl, op_cost_calibration.jsonl, roofline_from_profile.py}
repro: repro-missing
---

### AIV 侧 GM 屋顶 0.8 TB/s + 逐任务小 DMA 单价 0.17 µs/op（910B4 / CANN 9.0.0）

- **屋顶口径（决定性）**：`visualize_data.bin` 的 `GM/L2 : GM Read + Write` 条目在本机报
  **`bw = 0.8 TB/s`**（同一份 report 在 910B3 档环境报 1.75 TB/s；两机同一字段相差 2.19×）。
  `Ratio` 实测恒等于 `achieved_bw / 0.8`（例：`(32,16384)` fp16 1.049 MB / 4.61 µs =
  0.227 TB/s ⇒ 28.4% ≈ 报告的 28.62%），故**引用 1.75 TB/s 作分母的 70% 目标（=1.225 TB/s）
  在本机是口径不可达**（超过本机屋顶 1.53×）。
- **AIV 读单价（per-core，`copyout - empty` 反解，hidden `(2048,4096)` fp16）**：
  8 KB tile → 26.0 GB/s/核；16 KB → 22.4；32 KB → 21.5（40 核聚合 0.86–1.04 TB/s）。
  **tile 越大单价越低效**——读侧是带宽受限，不是「每 op 固定开销」受限。
- **读单价第二自变量：每 tile 行数 × 并行读者数（2026-10-18 追加，`4d2e413` / 910B4；同构合成 tile、`copyout − empty` 反解，`--launch-count=15`）**：同一 tile 字节数下读单价仍差 2–4×——
  16 KB / 8 块（`bm=4, W=4096`）**0.129 µs（124 GB/s/核）**、32 KB / 16 块（`bm=2, W=4096`）
  **0.244（131）**、32 KB / 32 块（`bm=1, W=8192`）**0.192（171）**、64 KB / 8 块（`bm=4, W=4096`）
  **0.829（79）**、64 KB / 32 块（`bm=1, W=16384`）**1.508（43 GB/s/核，7.9× 最差档）**。
  **口径警示**：这五点同时变了 tile 字节、每 tile 行数、block 数与总活跃足迹（8–32 块 × 16–64 KB
  = 0.13–2 MB），**不能当纯「行宽 W」轴读**；最可能的主导项是 L2 驻留 / 并发读者压力（同一批 tile
  被反复读 15 次），即「同 tile 字节、单行更宽」不等于「更慢」——引用时须在目标栅格上直接探针。
  **对本族（小 M、读主导）的机械后果**：单 tile 超预算形态（R′）在 64 KB / W=16384 档每次读付
  1.508 µs，是列分块 2×0.192 µs 的 3.9×——该差值直接解释了 R′ 在 bool→f32 W=16384 上 **+9.6%**
  的回退（见 layout.md `PL-1.25` 第六数据点）。溯源（任务工作区，允许失效，ED-A）：`examples/TileOPs/tileops/kernels/reduction/all/all_kernel/perf_opt/{op_cost_calibration.jsonl（`mte2_read_tile` 5 行）, copy_floor_calibration.jsonl, probe_floor.py}`。
- **逐任务输出小 DMA（8–32 B `(block_m,)` int64）**：**0.16–0.20 µs/op**
  （`empty` 探针斜率：52/26/13 任务 → 12.741/6.672/4.495 µs，基座 ≈2.4 µs）。
  **但它与 vec/mte2 部分重叠**：把它合成一次大 DMA 后 `mte3` 从 11.54 → 0.28 µs，
  总 `Task Duration` 不变（26.96 vs 26.64）；在向量已饱和的 bf16 上反而 **−10%**
  （新增 1 个 UB→UB 拷贝压在向量 pipe 上）。
- **向量 pass 单价（指示链）**：`vcmp(eq)→vcast(i1→f32)→reduce_sum(dim=1)` 在
  `(2,4096)` tile 上 ≈ **0.158 µs/pass**（0.019 ns/元素），低于
  `CONST-vector-launch-overhead` 的 0.23–0.3 µs/pass 低档。
- **判据**：「删掉小 DMA / 增加 block 尺寸」类优化**必须先看 pipe ratio 是否已达 0.5 以上**；
  被盖住的小 DMA 删除零收益，而块变大在读侧可能**变差**。带宽受限 workload 的唯一有效方向
  是减少 GM 流量或提高重叠，不是减少 op 条数。
- **口径更正（2026-10-18 蒸馏核对）：`0.158 µs/pass` 是吞吐口径，不是发射单价**——该值
  = `aiv_vec_time 12.33 µs ÷ (26 任务 × 3 pass)`，每个 pass 覆盖 8192 元素 ⇒ **0.019 ns/元素**，
  是整 kernel 链在流水下的向量吞吐；与本库 `CONST-vector-launch-overhead` 的 12–23 ns/op
  （64–1024 lane 孤立 depend 链、发射主导）**不同量纲**，不可互推。源任务把它与
  「0.23–0.3 µs/pass 低档」相比属跨量纲比较，引用时不得据此下调发射单价常数。
  同源的屋顶与读单价数字已并入 `CONST-device-peak-910B3` 第四数据点（同一 `visualize_data.bin`
  口径），本条只对「本机 0.8 口径独立复核」与「小 DMA 单价 + 被 vec/mte2 覆盖」两轴负责。
- **互链**：`CONST-device-peak-910B3`（910B3 单向读 753–779 GB/s，同量级；本条补 910B4 的
  msprof 屋顶值与「小 DMA 单价 + 被重叠」两个反直觉量；**该条第五数据点另收用户裁定的双轨验收口径**——
  族验收线 = `0.7 × 自报 roof` = 560 GB/s，台账 1.75 口径不变，引用本条的 0.8 时须同时引用该裁定）、`CONST-msprof-task-floor-910B3`
  （空 kernel 地板；本形态单任务实测 2.44 µs 更低）、`CONST-fabric-bw`（fabric 聚合口径，
  **不等于** AIV 可达读带宽——本条是它被误用的直接反例）。
- **证明**：`repro-missing`（待补：从 `perf_opt/probe_floor.py` 裁出 empty/copyout × bm 三档，
  断言 bm=4 < bm=2 < bm=1 的总时延序与 16 KB 单价区间）。溯源（任务工作区，允许失效）：
  `examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero_kernel/perf_opt/`。
