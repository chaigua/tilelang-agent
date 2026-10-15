# Elementwise 多输入搬运效率与带宽曲线

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。

---
id: PL-1.6-copy-floor
kind: constant
family: [elementwise]
apis: [T.copy, T.vadd]
dtype: [fp16, bf16, fp32]
device: 910B2C, 910B3
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z（首证：copy-floor 标定法 + MTE2 带宽退化曲线）/ adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015（追加：global 类地板由「每核串行轮数」主导 + 轮数自变量扫描）/ adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z（追加：写侧生产者形态维度 + 假地板识别纪律）
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0；2026-10-15 追加 tilelang 0.1.2+489385e9b98806365b585b6e1f984a5dae337154（commit 489385e / dev root build）/ Ascend910B3 / CANN 9.0.0
repro: repro/CONST-copy-floor-rounds.py
---

- **〔2026-10-15 追加，双输出 + 写侧生产者形态；origin_task `adaptive_max_pool2d_with_indices-...-20261015T100705Z`，toolchain 67a0441〕标定须把「写侧生产者形态」当自变量，否则会产出假地板**：
  同一栅格 / 同 DMA 粒度下，把「值 + int64 索引」双输出的 store 源从**连续 op 链**（`vbrc`→`vcast`）换成**跨步 UB gather**
  （`flat[:,:,0]`，即从 `(C,plane)` 取列），Task 从 **4.73 µs → 109.29 µs**（`UB_to_GM` 0.8 → 96 KB/blk，真实写 4.5 KB/blk）；
  该 95–111 µs 的「dual 地板」会把 baseline 144.4 µs 判成已达 70% 线（156 µs），**方向性错误**。
  **纪律**：① 双输出/带索引写出的算子，标定必须至少给「连续生产者」与「跨步生产者」两档；② **当地板显著低于 baseline 时先怀疑标定假地板**；
  ③ 合法地板锚 = 1 元素保活读地板（本任务 global CH=64 → **6.86 µs**，70% 线 **9.80 µs**）与连续生产者双输出地板（**4.73 µs**）。
  数据（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/.../perf_opt/copy_floor_calibration.jsonl`
  （行 `global_ch64_read_floor_elem` / `global_ch64_dual_contig_store` / `global_ch64_dual_transpose_poison`）+ `opt_log.md` §2。
### 1.6 copy-floor 标定与 MTE2 带宽退化曲线（elementwise 多输入）✅ 已验证

- **copy-only 探针标定法**：与目标 kernel 同结构（同 persistent/分块/映射）的 3 载入 + 2 vadd 保活（DCE-proof）+ 1 写出探针，一步判定计算链暴露量。lerp_tensor（3 输入 1 输出，7-pass fp32-transit 链）：≥16M 全 dtype delta ≤1.1us（16M 仅 0.34us）——**auto multi-buffer 将全部向量 pass 隐藏于 MTE2 窗口内**，计算链优化在该区间 ROI≈0，直接进入搬运效率维度。
- **轮数自变量扫描（2026-10-15 adaptive_max_pool2d Stage 4 追加，整平面连续搬运域）**：copy-floor 必须**按将要落地的每核串行轮数**标定，不能只测一个粒度就外推。同流量（1.6 MB）、同 DMA 粒度（CH×49 fp16 元素/轮）下：CH=32 → 13 轮 **10.99 µs**（mte2 3.85）、CH=64 → 7 轮 **6.85 µs**（2.13）、CH=128 → 4 轮 **5.09 µs**（1.21）；即 mte2 只差 3.2x 而 Task 差 2.2x，差额是**每轮 DMA→落盘 的串行延迟（≈0.7–0.9 µs/轮）**。此时带宽不是地板（211–690 GB/s vs fabric ≥1.73 TB/s），**轮数才是**；70% 校准线（地板 ÷ 0.7）也随之从 15.70/9.79/7.27 µs 变化。纪律：① 标定报告须标注轮数；② 单粒度标定值不可外推到其它 CH；③ 探针落盘用「1 元素 store」保活，勿用 `(M,1)` 列切片（stride=plane 会触发列写放大：实测 52 KB/block、Task 93.95 µs 的假地板）。repro：`repro/CONST-copy-floor-rounds.py`（骨架 + 轮数表）。
- **MTE2 有效带宽随总流量退化**（4-set 轮换，fp16 bs4096，48 vector core）：1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；256M 的 3:1 R/W 混合全流量 ~1.26 TB/s 为该设备地板——**"~1.8 TB/s 峰值参考"对混合读写流量不可达**（不要据其估算大 N headroom）。1M 档读带宽含 L2 成分。
- **访问模式**：grid-stride（48 核聚集于 48×tile 移动窗口）实测优于每核连续分块 +3.2%（256M）——HBM 混合流下聚集窗口访问更优，"连续流 DRAM 局部性"直觉反向（详见 bottleneck-patterns BP 候选）。
- **标量削减判定式**：mte2_ratio ≥0.95（搬运饱和）时 per-tile 标量/分支开销已被隐藏——静态尾块（去 T.min）全档 tie/-0.6%，guard-free 预判 <0.3%；此类优化仅在 mte2_ratio <0.9 的流水浅区（如 1M 档 0.70–0.84）可能有收益。
- 溯源：lerp_tensor optimize 任务（2026-09-07），`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md`；tilelang 0.1.2+ed787bb / Ascend910B2C / CANN 8.5.0。

---
id: PL-1.10-loads-first-decoupling
kind: pattern
family: [elementwise, reduction, norm]
apis: [T.copy, T.vcast, T.serial, T.alloc_shared]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（2026-09-10 蒸馏溯源归位：原回写误标 145312Z，实际 optimize task_id 以 stage_state.json 为准）
toolchain: tilelang 0.1.2+a83118285a（2026-09-10 build）/ Ascend910B2C / CANN 8.5.0
repro: repro/PL-1.10-loads-first-decoupling.py
---

### 1.10 单 staging 复用链的 MTE2/VEC 零重叠 → per-input staging + 前置装载 ✅ 已验证

**现象信号**：多输入 kernel 的 msprof PipeUtilization 中**各 pipe 忙时之和 ≈ Task Duration**（如 prefill fp16 bm=1：vec 38.8 + scalar 13.2 + mte2 42.4 + mte3 6.5 ≈ 101µs vs wall 90.5µs）——MTE2 与 VEC 交替串行而非重叠。**根因**：单一 staging 缓冲被 x→scale→shift 顺序复用，数据流强制 MTE2(x)→VEC(vcast)→MTE2(scale)→VEC→MTE2(shift)，每次 GM 装载都等前一个 v-op 释放缓冲。

**优化（慢→快关键更改，delta 形态见 repro）**：① 每个输入一张专用 staging 缓冲；② 三输入 `T.copy` 全部前置到 persistent serial body 顶部（纯 MTE2 相）；③ 向量链不间断地跑完；④ 输出 staging 复用 x-in staging（活区间不相交：x-in 死于首个 vcast，out 始于末个 vcast）——不新增缓冲。auto-multi-buffer pass 会物化第 2 实例使跨迭代重叠合法（UB 预算按 TRAP-UB-multibuffer-inflation 平台值：transit ≈20 B/elem、fp32 ≈26 B/elem，**不是** resident × 系数）。

**实测**（ada_layer_norm，2048×4096 fp16，bm=2，msprof op median-of-20）：58.27 → 39.81µs（**-31.7%**）；pipe 忙时和 100.9→68.9µs vs wall 39.8（~29µs 跨 pipe 重叠）；kernel 转为 vector-bound（vec_ratio 0.914）。次级收益：dit-xl-2（1024×1152 fp16）-12.5%、decode（1×4096 bf16）-16.1%（少 2 个 mid-chain MTE2→V 同步点）。**适用条件**：persistent serial（num_local_tasks≥2，跨迭代重叠有深度）收益最大；num_local=1 时退化为 tie~小幅回退（smoke 64×1152 A/B tie——intra-iteration 无重叠可赢，实测勿盲改）。同族先例：PL-1.6 lerp 3 载入前置 + 向量链全隐藏（≥16M 档）。

**注意**：bm 选择与该结构正交但交互——bm 上限受 multi-buffer 平台 footprint 约束（见上）；且最优 bm 非单调（ada dit 1024×1152：bm 曲线 {5:9.75, 6:9.66, 7:8.76, 8:9.55}µs，bm=7 局部最优；同 N 下 4096×1152 则 bm=8 胜——**按 shape 实测扫描，勿从单点泛化 UB 松弛律**）。

- 溯源：ada_layer_norm Stage 4（2026-09-10），`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md`（Iteration 2 机制确认 + A/B 裁决记录）。

---
id: PL-1.14-mte3-strided-ws
kind: pattern
family: [elementwise, mixcv, mte]
apis: [T.copy]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑复刻 ws 块连续布局，全量门禁绿）
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### 跨引擎 ws 中继的 MTE3 布局铁律：块连续 ≫ band 化

- **形态**：Vector→Cube 因子中继（GM ws）的写侧布局两个候选——①块连续（每 [64,64] 块占连续 8KB，ws 形如 [..., s_blk, bl, bs]）；②band 化（每 l-tile 的因果带 [bl, l0+bs] 连续，行 stride = Q×2B=512B，换取 Cube 侧单条 band 读）。
- **实测（ssd_chunk_scan w2，msprof op）**：band 化使 **AIV MTE3 时间 3×**（106→220µs，ratio 0.17→0.35）——512B 行距的跨步写吞掉 Cube 侧全部搬运合并收益（w2 净 +2.3% 回退）。**写侧块连续 + 读侧逐块连续读入 L1 band 列偏移区**（`l1[0:bl, s0:s0+ts]`，**ts 必须尾块裁剪**——见 traps-runtime.md TRAP-L1-band-dst-tail-overrun）是两全形态：Vector 写快、Cube 仍可单 gemm（v5_xband，w2 −1.8%/w4 −4.3%）。
- 机制：MTE3 跨步写每行 128B 段 + 512B 行距 → 段级延迟不摊销（CONST-mte2-degradation 的指令/段维度）；块连续 8KB 单段流。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log R1 v2_band 行 + R2 v5_xband 行；profiles/round1|round2 对比）。

---
id: PL-1.17-subfp32-fp32-transit
kind: pattern
family: [elementwise, vector, norm, mamba]
apis: [T.vcast, T.vadd, T.vsub, T.vmul, T.vexp, T.copy]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z（首证，VP-2026-0002）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（第二证：bf16 触发编译实证 + 因子链形态）
toolchain: 首证 tilelang-mlir-dev dev root build（2026-09-07）/ 第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）
repro: repro-missing
---

### sub-fp32 逐元素算子 fp32 中转模式（bf16 支持缺失 + fp16 golden 对齐双触发）✅ 两任务实证

- **双触发条件**：① 契约 dtype 含 bf16——v-prefix 算术（vadd/vsub/vmul/vexp）dtype 矩阵不含 bf16（docs/Tilelang.language/数学操作/ 各 §2.2.1），vcast 升 fp32 是唯一计算路径（ssd 第二证：c_scaled 链 fp16 化变体 v7 在 bf16 workload **编译失败** rollback——`T.vmul` bf16 × 的运行时实证）；② fp16 需对齐 torch CPU golden——golden 经 fp32 opmath + 单次舍回，原生域逐步舍入差 2–3 ulp（TRAP-fp16-opmath-golden）。
- **链结构**：GM→UB(同 dtype) → `vcast(round_mode="rint")`→fp32 → fp32 域 v-prefix 原地链 → `vcast(rint)`→原 dtype → UB→GM（f16/bf16→f32 仅 rint，T.vcast.md §2.2.1）。ssd 因子链形态：cb/C/dt(dtype)→f32 因子域运算、末端回写（Stage 3 L0–Boundary 全过实证）。
- **UB 预算**：中转路径按 Σ(elem_bytes × buffer_count) 复算（bf16 中转 24B/elem、fp16 中转 20B/elem——VP-2026-0009 混合字节口径）。
- **实测**（lerp 首证）：max_diff ≤1 ulp fp16（抽样 binade 4.883e-04～1.953e-03）、全量 0 violations；NaN/Inf 角点 IEEE 传播与舍入路径无关（中转后 inf-corner 逐位不变）。
- repro：repro-missing（知识域最小 repro 待同族任务回填；两证任务内复现命令见 queue VP-2026-0002 证据链——provenance 允许失效）。

---
id: PL-1.19-scatter-col-write-amplification
kind: pattern
family: [reduction, elementwise]
apis: [T.copy]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_strided_colblock_read.py
---

### (M,1) 列切片 UB→GM 写 42× 放大 → 扁平 chunk-major 连续写

慢：`T.copy(ext_out[0:M,0:1], ws[0:M, c:c+1])`（(M,1) 目标列 × 行 stride）——每散布元素拉 128B 行：tn=256/nchunk=400 实测 UB_to_GM 400KB（真实 9.6KB，42×），mte3_ratio 0.78。快：workspace 取扁平 (nchunk*M,) chunk-major，每 chunk 一次 1D 连续写 `T.copy(ext_out[0:M], ws_flat[c*M:(c+1)*M])`（同 kernel mte3 降至 0.03–0.18）；消费端按 flat-view 重解释形状（narrow-path 先例）。写侧块连续律与 PL-1.14（跨引擎 ws）同族，此处为纯 Vector UB→GM。

---
id: PL-1.20-nsplit-perrow-partial
kind: pattern
family: [reduction]
apis: [T.Kernel, T.transpose, T.vbrc]
dtype: [fp16, bf16, fp32]
device: 910B2C, 910B3
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z / argmin-_argreduce_kernel-20261015T023310Z（第二实例：跨设备 910B3 + CANN 9.0.0 双 dtype 复现）
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C；第二实例 tilelang 0.1.2+417dae9（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro/repro_nsplit_pattern.py
---

### C6 N-split per-row partial + transpose merge：小 M 大 N 负载 2.2–2.6×

适用：M ≤ cores、N 超驻留（lm-head 类 (4,102400)）——单 kernel tiled 被 bm=1 陷阱钉死在 M 个核（4/48）。结构（慢→快关键 delta）：
- partial：`T.Kernel(min(nchunk,cores))` + `for s in T.serial: c=cid+s*K; if c<nchunk:` + **`for r in range(M)` 展开的 (1,tn) 行链**（copy/reduce/vbrc/fused-ite/reduce_min 全部 bm=1 已验证形态——关键修复是 (M,tn) 2D 融合 ite 踩 TRAP-parallel2d-bm-ge2-scalarize；跨步列块读本身不慢，见 CONST-gm-to-ub-bw-dilution 更正记录）+ 每 chunk 两条 1D 连续 ws 写（chunk-major 扁平，消 PL-1.19 写放大）。
- merge：单 block，ws 按 (nchunk,M) 视图读入 shared → `T.transpose(...,[1,0])` → (M,nchunk) dims=1 tie-break 链（E1 同构：vselect 命中 chunk 的已全局化 idx，min = 首现）。
- **tn 规则（扫描实测 tn∈{1280,2048,2560,5120} → 8.58µs@2560 最优）：最大单波 nchunk ≤ cores**（tn 升序 256 倍数扫首个除数）；nchunk>cores（双波 2:1）与 nchunk≪cores（核饥饿）均劣。
- 实测：lm-head fp16 25.90→11.78µs（2.20×）、bf16 30.61→11.75µs（2.60×）；工厂契约保持单 callable（组合层 torch.empty workspace + 双发射）。
- **跨设备/跨工具链复现（第二实例，2026-10-15，argmin 共享 ArgreduceKernel 的 Stage 4）**：910B3（aicore 20 / 40 Vector）+ CANN 9.0.0 + tilelang 417dae9 上，`(4,102400)` **双 dtype** 均得 **tn=2560 / nchunk=40（零尾）**——tn 规则逐条复现（tn 升序 256 倍数扫首个 `N % tn == 0` 且 `nchunk = N/tn ≤ min(vector_cores, N//256, MERGE_ELEMS/M)`）；端到端 **fp16 11.898 µs（26.247 → 0.453×）/ bf16 11.914 µs（31.370 → 0.380×）**，与 argmax 在 910B2C/CANN 8.5.0 的锚点（0.461× / 0.386×）同量级——**结构可迁移、参数按本机 40 核重导后恰好同值**。裁决谓词本轮补全为 `M ≤ 16 且 M < cores 且 N_padded ≥ 32768`。该形态的 msprof 口径须记**两段 Task Duration 之和**（partial + merge，分别按 op 名过滤捕获）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/opt_log.md#Iteration-4`。
