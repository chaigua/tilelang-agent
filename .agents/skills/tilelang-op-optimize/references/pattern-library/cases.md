# 案例索引与参考实现集（同类问题的正/反例参考）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）。
>
> **用途**：按"哪类问题该参考哪个算子目录"组织 `examples/` 的检索入口。**目标未上库的条目标注「任务工作区，未上库」并给出 durable 载体**（该经验的持久形态 = pattern-library 自包含条目 / repro）——工作区目录随时可能被清理，条目知识不随之失效（ED-A）。Stage 1 设计（强制步骤 0.5）与 optimize skill Phase 0 均会读取本文件——命中时优先精读对应算子目录，优先于盲目 Glob。**维护**：由 `tilelang-skill-evolver` 在任务终态追加（C 类，Tier 0）；条目 = 路径 + 一句话 + 适用触发条件，不复制内容；路径失效或长期零引用 → deprecate。
>
> **反例档案**（BLOCKED 终态任务的根因链 + 能力缺口互链）见本文件末节——capability-gaps（`.agents/evolution/capability-gaps.md`）的 open 条目实质是"该形态当前做不了"的反例，以指针行互链（只加指针不复制内容，边界按该簿 §与 pattern-library 的边界）。

---
id: CASE-pool-maxpool3d-tileops
kind: case
family: [pool]
mode: [developer]
dtype: []
status: verified
origin_task: max_pool3d-harness-migration-2026-08
toolchain: 2026-08
repro: none
---

### `examples/TileOPs/tileops/kernels/pool/max_pool3d/max_pool3d_kernel/`（任务工作区，未上库）

**durable 载体**：本文件无自包含池化条目时见 layout.md PL-1.1/1.2/1.4（模式与实测数字自包含）；集成包结构形态见已上库的 mish/logsumexp 同构目录。

TileOPs 集成包形态：kernel + `{func}_DESIGN.md` 快照 + 聚合 `__init__.py` + integration_log。适用触发条件：migration-harness 集成结构参考；Stage 5 集成问题排查。**（2026-10-13 复核，task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z：该工作区路径已不在库——`examples/TileOPs/tileops/kernels/pool/` 下现仅 `adaptive_avg_pool2d`；预注入命中该路径时以本条 durable 载体（layout.md PL-1.1/1.2/1.4）为准，工作区路径不作证据。）**

---
id: CASE-elementwise-mish-opt
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, fp32]
status: verified
origin_task: mish-optimize-2026-08
toolchain: 2026-08（详见其 opt_log 版本戳）
repro: none
---

### `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/`

elementwise cast 类完整调优档案（persistent 甜点、multi-buffer 教训、Expert 流水平台约束、run 双态协议）。适用触发条件：elementwise 调优前必读先例；<3% 结论多 run 协议。

---
id: CASE-reduction-logsumexp
kind: case
family: [reduction]
mode: [developer]
dtype: []
status: verified
origin_task: logsumexp-migration-2026-08
toolchain: 2026-08
repro: none
---

### `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`

规约类（logsumexp）独立算子目录：DESIGN.md + kernel + 分层测试。适用触发条件：规约类（水平归约/online 单遍）设计与 L0 测试计划参考。

---
id: CASE-elementwise-lerp-opt
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
repro: none
---

### `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/`

elementwise 3 输入 persistent 调优：copy-floor 探针标定法 + MTE2 带宽退化曲线 + 交错 A/B 采纳复核 + grid-stride vs 连续分块反例。适用触发条件：判定"已贴内存地板"时的方法学参考；多输入 elementwise 调优。

---
id: CASE-elementwise-lerp-precision
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, bf16]
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: tilelang-mlir-dev dev root build（2026-09-07）；golden torch 2.9.0+cpu
repro: none
---

### `examples/lerp_tensor/_make_lerp_tensor_kernel/`（任务工作区，未上库）

**durable 载体**：traps-runtime.md TRAP-fp16-opmath-golden（现象/对齐通解/数字自包含）。

fp16 精度失败定征与修复完整档案：attempt-1 原生域失败基线（`history_version/`）+ R2 fp32 中转修复（≤1 ulp，全量 62 PASS）+ kernel 头部 Implementation Notes 等价性论证。适用触发条件：fp16/bf16 逐元素迁移精度失败（golden opmath 域分歧）定征与修复参考；vcast 中转路径 UB 混合字节预算复算案例。

---
id: CASE-pool-maxpool3d-standalone
kind: case
family: [pool]
mode: [developer]
dtype: []
status: verified
origin_task: maxpool3d-2026-08
toolchain: 2026-08
repro: none
---

### `examples/maxpool3d/_max_pool3d_kernel/`（任务工作区，未上库）

**durable 载体**：layout.md PL-1.1/1.2/1.4（池化类模式与实测数字自包含）。

standalone 形态的池化算子目录（DESIGN.md + kernel）。适用触发条件：窗口/池化类设计参考（分核策略、边界处理）。

---
id: CASE-attention-gqa-expert-full
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: none
---

### `examples/multi_head_attention/_gqa_prefill_fwd_kernel/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.8（bf16 直连）、traps-compiler/runtime.md attention 族陷阱条目（正文自包含）。

Expert 模式 GQA/MHA prefill attention 完整档案：v0→v1→v2 设计修订链（K_A ceildiv→floordiv 差 1 反例 + 检视建议自我纠正全程 + per-lane 阈值符号错误）+ 3 轮 REVIEW + debug_log D1–D6（load_nd2nz 误读 / T.copy 语义 / 解析器规则 / tier-3 精度分类）+ bf16 直连 + E7 守卫不变式。适用触发条件：attention 族 Expert 迁移/重设计；跨引擎 flag 流水协议设计；bf16 Cube 直连决策；分界公式检视反例。〔2026-09-15 注：该目录 DESIGN.md/REVIEW.md 已被两相位 v3 重设计覆盖（见 CASE-attention-twophase-causal-regen）；v0→v2 修订链与 E1-E7 单遍设计存档于 history_version/（design_v0/v1/v2.md），debug_log.md（含 D1–D6）仍在本目录——引用本条修订链/检视反例时以 history_version/ 为准。origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z〕

---
id: CASE-attention-expert-stage4
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, f16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z + 20260909T033622Z + 20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08～09（三轮）
repro: none
---

### `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.9-blockwidth/hardlimits/twophase（三轮结论自包含）+ `repro/PATT-twophase-restructure.py` + constants.md 硬件常数表。

attention 族 expert Stage 4 完整调优档案：块宽摊减调优序（bn 64→512）、cbuf=512KB 实测、wide 单槽 L1 模式、clean trace 特化、S-f16 传输裁决、G=2 备选 A 裁决、H=1 任务平衡公式、工厂内 tuned 分派表（S4-5 模式）；第二轮硬上限测绘（L0C 128KB / L1 端口 ~148–154GB/s/核 / 向量发射 ~0.5µs/op / fabric ≥1.73TB/s）+ f16 softmax 链（终值 17.55/27.24/69.13/226.72µs）+ [DESIGN_LIMIT] 五段式反馈档案（perf_feedback.md，硬目标 2.09x/2.27x 缺口归因）；**第三轮两相位重构达标**（参考实现 flash_attn_npuir.py 结构迁移 + transpose-free lse，终值 12.94/18.11/30.88/98.05µs 四硬目标全达成、fa1024/2048/4096 优于参考、三轮累计 vs Stage-3 基线 -77.2%/-82.2%/-91.8%/-91.1%）+ [DESIGN_LIMIT] 负结论被同 API 结构推翻的完整闭环档案（perf_feedback.md 修正附录：局部地板 vs 绝对地板界定 + 两处根因 + morph 阶梯 14 点矩阵）。适用触发条件：attention/persistent 双 Scope 类调优前必读；跨引擎流水结构选型（块宽 vs 流水深度）；备选 A/B 裁决执行范例；设计层天花板判定与 [DESIGN_LIMIT] 归因/证据组织参考；**结构不可达类负结论的修正闭环参考**。

---
id: CASE-attention-developer-stage4
kind: case
family: [attention]
mode: [developer]
dtype: [fp16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
repro: none
---

### `examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.7（expert vs developer 同门对照结论自包含）。

developer 谱系（简单 tiling）同 4 case 调优档案：S4-1 mask 消除 + S4-2/3/4 死缓冲/in-place/退化 trace-time 特化 + S4-5 工厂内 shape 分派（终值 16.1/33.6/105.2/300.3µs）。适用触发条件：与 expert 谱系（attention.md PL-1.9 系列）做同门调优杠杆对照（mask 消除/config 扫描 vs 块宽摊减/L1 上限突破）；编程模式/谱系选型参考。

---
id: CASE-deepseek-v4-highperf
kind: case
family: [attention, expert]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: 2026-09-07
repro: none
---

### `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py`

Expert 模式 attention/跨引擎结构先例：双 Scope + GM workspace 多槽 + per-slot flag + staggered stream + 运行时 if + int16 索引矩阵 vcmp mask 链（L345–354）+ PIPE_V 内 T.arange strides（L611）。适用触发条件：Expert 模式 kernel 结构设计/先例检索（Stage 1 设计起步）；vcmp 标量规则与 flag 协议先例核对。

---
id: CASE-ref-flash-attn-npuir
kind: case
family: [attention]
mode: [developer]
dtype: [fp16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z（第三轮实证）
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: none
---

### 参考实现集：`examples/flash_attention/flash_attn_npuir.py`

**参考实现集首条**（T-1 参照锚定）。仓库内同 API 可达结构的参照锚点：两相位（pass-1 全 S / pass-2 全 O_partial）+ 非持久 grid + 单 L0C 顺序复用 + Q hoist + transpose-free lse——第三轮 [DESIGN_LIMIT] 负结论正是被该结构推翻（四硬目标全达成 12.94/18.11/30.88/98.05µs）。适用触发条件：attention 族调优 plateau / 拟产出 `[DESIGN_LIMIT]` 时的**强制参照对照**（perf-feedback.md §1 参照锚定门槛）；两相位结构迁移的母本。

---
id: CASE-norm-adalayern-migration
kind: case
family: [norm]
mode: [developer]
dtype: [fp32, fp16, bf16]
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
repro: none
---

### `examples/ada_layer_norm/_ada_layer_norm_kernel/`（任务工作区，未上库）

**durable 载体**：vrsqrt 陷阱与绕法自包含于 traps-runtime.md TRAP-vrsqrt-plain-precision（含知识域 repro）；迁移决策与双判据验证见该目录 DESIGN.md/verify_equiv.py（溯源，允许失效）。

norm 族首个迁移档案（Developer 模式单函数，零设计修订零重试一次过）：vrsqrt 近似指令精度链定位（TRAP-vrsqrt-plain-precision，分层探针单次运行定位）+ GPU 256-pad 方差校正整体舍弃（恒等式机器验证 0 违反，精确 N 式在小 N 大均值时数值更优）+ verify_equiv 45 案例等价性验证设计（矩式方差 rejection evidence）。适用触发条件：layernorm/rmsnorm/AdaLN-Zero gated 族迁移设计参考（wrapper 已保留 has_gate 分支，gated 变体复用同一集成路径）；Stage 3 精度失败含 rsqrt 段时优先对照。

---
id: CASE-norm-adalayern-stage4
kind: case
family: [norm, reduction]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
repro: none
---

### `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/`（任务工作区，未上库）

**durable 载体**：staging 解耦模式自包含于 elementwise.md PL-1.10 + repro/PL-1.10-loads-first-decoupling.py（知识域）；调优数字溯源 opt_log.md（允许失效）。

norm/row-reduction 族完整调优档案（6 轮，全组几何平均 2.14x）：wrapper 部署态 bm=1 → per-shape 调优表（第一杠杆）+ per-input staging 解耦（PL-1.10-loads-first-decoupling）+ bm 任务几何精搜（bm=7 非单调局部最优）+ 调优表条目 U 曲线闭合验证 + A/B 交错协议两次防误判 + 保护性泛化验证（「UB 85% 松弛律」单点否定）。适用触发条件：norm/规约族 Stage 4 调优；多输入 staging 复用链串行化诊断（pipe 忙时和 ≈ wall 信号）；调优表条目鲁棒性验证方法参考。

---
id: CASE-attention-twophase-causal-regen
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z（Stage 1→3 重生成）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4 调优）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4 调优）
toolchain: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967（Stage 1→3）+ tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD，第五轮）+ tilelang 0.1.2+a13585dc（第六轮），均 + CANN 8.5.0 / Ascend910B2C
repro: none
---

### `examples/multi_head_attention/_gqa_prefill_fwd_kernel/`（任务工作区，未上库）

**durable 载体**：两相位 causal 适配五机制待 queue VP-2026-0057（1/2）；config 依赖性数据已合入 attention.md §1.9-twophase causal 域 bullet（VP-2026-0063 两证，2026-09-15）；第五轮调优知识自包含于 attention.md PL-1.11 + traps-compiler.md TRAP-expert-v-operands/TRAP-UB 第四证 + constants.md CONST-capacity UB 系数；**第六轮调优知识自包含于 attention.md PL-1.12 + repro/PL-1.12-bn-clamp-bm-guard.py**（屏障审计 / 深度 2 / f32 载体 / Ratio 口径 / r9 分派守卫）。

两相位 S/P 物化 v3 重生成完整档案（manifest causal 多头域 Stage 1→3 重跑，用户判定上一版为 case_fa 特化后从设计层重新生成）：DESIGN.md v3（persistent 24 核双 Scope + per-n-block 同 id 三次握手 + E6 flag 预算-bn 耦合守卫）+ verify_equiv.py（13/13 EQUIV_PASS，双跑一致——等价性机器验证脚本结构参照：基线式忠实到常数与运算序 / fp64 参照 / 违反率双口径 / 探针行分离，torch CPU 无 NPU 依赖）+ REVIEW.md 9 维度通过 + Stage 3 `--level all` 29/29 + RETROSPECTIVE.md 2026-09-15 章节（gate 误报修复姿势 / T.ceildiv 空核钳制 / per-core ws 槽 row0 索引 / 装载不满宽掩码门控口径 / dump IR 新 cache 目录）。**第五轮 Stage 4 调优（2026-09-15，4 轮，14.2x）**：判别实验链教科书档案（因子分离 → bn/bm 扫描证伪 → TILELANG_DUMP_IR 编译警告定位 → 微探针定案）锁定 vcmp int16 标量化根因（占壁钟 45%）→ 算术惩罚掩码 + UB diet 宽块解锁 + zbuf l1_b 零初始化；探针族 probe_vcmp_forms/penalty_chain/nan_clamp/rt_slice/l2flake 等（微探针方法论参照）；perf_records.jsonl 36 行 + profiles/{phase1,round2,round3,final}/。**第六轮 Stage 4 调优（2026-09-16，roofline Ratio 硬目标 20%→35% 口径）**：TASKDONE 屏障冗余审计删除（全域 −7~−21%，纯程序顺序论证零实验成本）+ Cube 深度 2 任务流水（flag slot 双槽 2×nk≤15）+ per-shape bm 分派 + fp16 域 f32 ws_s 载体（8b-long 1191.63→989.26µs = −16.8%，vs 第五轮启动基线累计 17.1x；long 4 配置 Ratio 22.64–23.63%——20% 口径达标、35% 口径差 11.4–12.4pt）；r9 precision_fix 分派守卫收窄（bn 钳位域 × bm80 联合 UB 核算判据，S_kv≥3841 域修复，TileOPs pytest 6/6）+ 双镜像同步先例。适用触发条件：causal 多头域两相位/persistent 结构设计参考；causal mask 链标量化排查与判别实验设计参照；verify_equiv 脚本结构参照；per-n-block flag 协议与尾块门控设计；task 级流水与同步屏障审计参照；派发守卫组合的联合 UB 核算参照；msprof roofline Ratio 口径目标调优参照；前谱系（Sep-07 E1-E7 单遍）见 history_version/ 与 CASE-attention-gqa-expert-full，调优轮次史见 CASE-attention-expert-stage4。

---
id: CASE-attention-mha-config-unvalidated
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z
toolchain: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
repro: none
---

### 反例：设计默认 config 路径未经目标域编译验证即出厂（`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` Round 2；任务工作区，未上库——溯源允许失效，根因链要点已自包含于本条）

**durable 载体**：UB 预算偏差实证已合入 traps-compiler.md TRAP-UB-multibuffer-inflation 第四证（VP-2026-0059 两证，2026-09-15，含 ×1.10–1.12 系数 + UB diet 绕法）；能力缺口本体见 capability-gaps CG-2026-0008（recurring）。

反例档案：设计默认 config（E6 替换路径 bn_eff=256）在 dim=128 causal 域的首次编译发生在 Stage 5 bench 且直接 UB 硬溢出（8/10 失败）——根因三层：Stage 3 L1 门禁变体集不含 wrapper-default 派发路径（VP-2026-0065）+ DESIGN §4.5 手工预算低估实际分配 ~26KB（VP-2026-0059 / CG-2026-0008）+ pytest 域全 non-causal 掩盖（causal dim=128 traced 变体从未被 pytest 编译）。修复 = wrapper default_config num_stages 1→2 路由到门禁验证过的逐字路径（VP-2026-0074 config-契约范式），1 attempt 闭环、bench 10/10 复核。适用触发条件：带 config 替换语义 kernel 的集成期 bench 编译失败排查；「pytest 全绿 ≠ manifest 域全绿」的域覆盖核对；宽 config UB 预算校准。

---
id: CASE-ssd-chunkscan-migration
kind: case
family: [mamba, ssd, mixcv]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z / ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z（4515de8 重跑）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z（二轮调优 Stage 4 追记）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T120526Z（重建会话：断裂修复 + 三胜出 repro 骨架重推导 + 增量调优 +9.7%）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（npu-smi 26.0.rc1）/ 2026-09-17；重跑 tilelang 0.1.2+4515de8 / CANN 8.5.0 / Ascend910B2C / 2026-09-20；二轮调优 tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21；重建会话 tilelang 0.1.2+96f287eeaa（与 15ad002 delta 仅 .agents 文档、tilelang 源码零改动）/ CANN 8.5.0 / Ascend910B2C / 2026-09-21
repro: none
---

### `examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/` + `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/`（任务工作区 + 集成包）

**durable 载体**：Stage 4 调优知识自包含于 attention.md PL-1.12 update（消费侧前导 set）/ PL-1.13 + repro/PL-1.13-aiv-dup-subid-split.py（双 AIV 分片）/ elementwise.md PL-1.14（ws 块连续）/ layout.md PL-1.15（列广播性能税）/ traps-compiler.md TRAP-UB-dynsubview-dominance / traps-runtime.md TRAP-L1-band-dst-tail-overrun / TRAP-DEVMODE-PERSIST-GEMM + repro（Developer+persistent 崩溃）/ constants.md CONST-mte2 指令维度口径；PL-1.16（Expert 双 Scope 绕法第二证）/ PL-1.17（fp32 中转第二证）/ elementwise 均以本任务为第二证合入。

**〔2026-09-21 4515de8 重跑任务 Stage 4 追记，tilelang 0.1.2+15ad002b3d（与 4515de8 同源）/ CANN 8.5.0 / 910B2C〕**：二轮调优 7 轮 10 分支 plateau，几何 1.078×（w2 223.53→204.33 / w3 638.85→601.61 / w4 3902.26→3616.80µs，msprof op median-of-20 + ab_test）——三项新胜出（prev_states lt 循环冗余重读削除〔旧段数记账漏算 ×4〕/ L0C acc 乒乓配对循环 / vbrc hoist 干净形态）+ 五项新否决（L1 双缓冲二次实证 / 运行时 if 进 Cube 热循环调度毒 w4 +15.65% / x 流头局部最优 / 发射序变体 / acc 深度 4）。知识自包含于 attention.md PL-1.18 update + repro/PL-1.18-floor2-wins.py + constants.md CONST-mte2 段数更正注。stop_reason=plateau；[DESIGN_LIMIT] 双门槛核对立（实测 204µs 落在 DESIGN 估算区间 170–280µs 内，无 >2× 实证替代结构）。

**〔2026-09-21 重建会话追记（task 20260921T120526Z，perf_opt 被外部删除后的断裂修复 + 增量调优），tilelang 0.1.2+96f287eeaa / CANN 8.5.0 / 910B2C〕**：上一调优会话 DONE 后 `perf_opt/` 被外部删除（wrapper 指向 perf_opt 致算子断裂，14 实验分支 + 64 行 records 永久丢失——未提交产物不可恢复实证，queue VP-2026-0117）；本会话 mode=full 重建三步：① Stage 3 基准拷贝 drop-in 修复断裂（factory 签名一致，L0 首编即过 = PL-1.12/1.13/1.14/1.16/TRAP-L1-band/TRAP-DEVMODE 结构在 96f287eeaa〔与 15ad002 tilelang 源码零改动〕存活重验，probe-w2 锚点 217.58 vs 旧 217.65 −0.03% 无漂移）；② 从 `repro/PL-1.18-floor2-wins.py` 骨架重推导三胜出（prevhoist/l0c2x/vbrchoist——ED-A/ED-B 自包含性经工件全失场景检验）；③ 11 个 benchmark workload 全量调优（H=24/80 双族，分支全量测量制 + 双独立测量合并检查）：**model-scale 7 workload 几何 +9.7%**（全 11 +6.4%；serving-130m +16.1% / longctx-130m +15.2%；w2 probe 217.58→200.18µs −8.0%，复现并略超旧二轮 −8.6%）；跨引擎累积序调制实证（vbrchoist 单点 H=80 回退在 prevhoist 合入后消失——queue VP-2026-0115/0112）；probe_bn64.py 补配置路径覆盖缺口（queue VP-2026-0065 第三现）；stop_reason=blocked（候选穷尽：H-fold / 任务级行装载 / bl=128 邻域 / per-lt 细粒度 flag / vbrchoist-lite 五新否决 + 旧五否决 docs-only delta carry-over 论证）。适用触发条件补充：perf_opt 断裂/重建场景（repro 骨架重推导路径）；MixCV 调优的累积序检验、全量测量制与双独立测量合并门参照。

mamba/SSD 族首个 MixCV Expert 迁移完整档案（设计修订 1 轮——列因子广播错向 / FLOPs 2× 高估 / PL-1.11 误判 stale 三阻塞；Stage 3 Developer→Expert 模式切换〔模式级不兼容实证〕+ L0–Boundary 全过；Stage 4 六轮 2.91× 几何平均〔608.15→217.65µs@w2：深度 2 任务流水 −39.6% → Cube band 组装 → block_n=128 → AIV subid 蛇形分片 −34.4%；stop_reason=blocked——UB 容量/编译器 dominance/API/性能税清单〕；Stage 5 首次集成即全过 smoke 2/2 + full 4/4 + bench 11/11）。TileOPs bench_mamba 11 dispatch 基线（wrapper 默认 config 口径）：Perf 稳定 32–34 TOps/s 平台、Ratio 7.5–14.2%、小 dispatch（<40µs）启动开销主导——**口径注记**：bench 经 wrapper `default_config` 显式传参（block_n=64/num_stages=3），优先于 kernel 内嵌 TUNED_DEFAULT_CONFIG(block_n=128/num_stages=2)，与 Stage 4 自建 workload 数值不可同口径对比（VP-2026-0093）。适用触发条件：mamba/SSD/chunk-scan 族迁移设计；MixCV persistent 因子链（Vector 产因子 → ws 中继 → Cube 消费）结构参考；AIV 分片与任务级流水参照；band 分域掩码（band-free 惩罚 / band-carrying vselect）落地参照。

**〔2026-09-20 4515de8 重跑档案（task 20260920T122332Z，迁移重做——升级工具链上的先例知识重验）〕**：设计以「结构性结论继承（有 HEAD 生产代码佐证）/ 硬边界标待重验」两分法消费 stale 预注入条目（VP-2026-0101），3 轮检视收敛：v0 协议/形状缺陷（尾置 wait WAR 竞争〔VP-2026-0096〕+ P_tiles≥2 域形状矛盾）→ v1 写实占位伪代码自创 **L1→L1 cbuf→cbuf 拷贝硬编译失败**（CG-2026-0002 occurrences 2 / queue VP-2026-0095）→ v2 清零；Stage 3 直接复刻 HEAD 旧 perf_opt 终版 + 三处差异适配（pass_configs 按规格扩为双关闭〔PL-1.16 4515de8 重验〕、golden 内联仓内参考 torch 计算〔薄包装被 S3-GOLDEN-TORCH 拒绝，VP-2026-0107〕、测试套件按新 L0 计划扩 w2 用例）——**首编即过、零调试往返**（L0–Boundary 全绿，max_diff ≤1.1e-3 bf16 / 1.4e-4 fp16；Boundary B-n16 实测 gemm K=16 静默有效〔VP-2026-0098〕）；R3 表 ws 中继流量曾按 per-l-tile 口径低估 2.5×（87→216MB@w2，per-task 全维乘积口径修正——复杂度表复算教训）；Stage 5 集成 attempts=0 首过（工厂闭包自分配 ws + 输入 cast 的零胶水形态〔VP-2026-0099〕；report 11 条 "missing tileops candidate record" 告警≠失败〔VP-2026-0100〕；11/11 bench msprof 全有效，ratio 0.13–14.00%）。适用触发条件补充：迁移重做任务（旧工件被 Stage 0 删除、仅存于 git HEAD——检视须 `git show HEAD:<path>` 核对，VP-2026-0102）的设计/检视/实现全链参照。

---
id: CASE-CG-INDEX
kind: case
family: [general]
mode: [developer, expert]
dtype: []
status: verified
origin_task: conductor-improvement-2026-09-10
toolchain: 登记于 2026-09-10；各条目版本戳见 .agents/evolution/capability-gaps.md
repro: none
---

### 反例档案（能力缺口互链）

capability-gaps 登记簿（`.agents/evolution/capability-gaps.md`）open 条目指针——"该形态当前做不了"的反例，调优/设计遇同族负向结论时先查：

| gap_id | 层 | 一句话触发条件 |
|--------|-----|---------------|
| CG-2026-0001 | codegen (Developer mode) | Developer 形态 codegen 能力缺口 |
| CG-2026-0002 | BishengIR（cbuf→cbuf copy） | cbuf→cbuf 搬运不支持（vbrc 清零 lowering + Expert T.copy L1→L1 双形态；recurring） |
| CG-2026-0003 | TileLangIR pass (Expert fixpipe lowering) | Expert fixpipe lowering 与 BishengIR 版本配对缺口 |
| CG-2026-0004 | tladapter/codegen (load_nd2nz 区域描述符下推) | load_nd2nz 跨步区域静默误读（绕法见 traps-runtime.md TRAP-load-nd2nz-strided） |
| CG-2026-0005 | Frontend API / codegen | 前端 API 能力缺口 |
| CG-2026-0006 | TileLangIR pass / codegen（向量算子融合与 f16 打包发射） | 向量算子融合缺口（f16≈f32 发射速率的机制根源，见 attention.md PL-1.9-hardlimits） |
| CG-2026-0007 | runtime / codegen（跨引擎同步原语粒度） | 跨引擎同步原语粒度缺口 |
| CG-2026-0008 | BishengIR（UB 基础分配可见性） | UB 手工预算 vs 实际分配差（宽 config 不可编译形态） |
| CG-2026-0009 | runtime / codegen（MTE2/MTE3 引擎边界同 buffer WAR） | MTE2/MTE3 同 buffer WAR 无顺序原语（Vec 深度 2 blocked） |
| CG-2026-0010 | codegen（Developer mode persistent+gemm） | Developer 模式 persistent 分核 + gemm 混排运行时崩溃 |
| CG-2026-0011 | Frontend API（gemm dst L0C region 写 / acc 相加原语） | gemm dst 不支持 L0C region 写、两 L0C acc 无相加原语 |
| CG-2026-0012 | BishengIR（auto-multi-buffer 动态 subview 支配性） | 动态偏移 UB subview 进嵌套循环产出非支配 IR |

> BLOCKED 终态任务的反例根因链条目同入本节（由 evolver 追加，标注「反例」）。

---
id: CASE-argmax-argreduce-stage4
kind: case
family: [reduction]
apis: []
dtype: [fp16, bf16, fp32]
device: 910B2C, 910B3
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z / argmin-_argreduce_kernel-20261015T023310Z（同族第二实例：optimize 场景 Stage 4 + [DESIGN_LIMIT]）
toolchain: 首证 tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C；第二实例 tilelang 0.1.2+417dae9（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro-missing
---

### argmax/_argreduce_kernel Stage 4 完整档案（C6 预注册裁决翻转 + 证伪更正）

几何平均 11.5×（lm-head 2.20×/2.60× C6 N-split 翻转、hidden-state 5.8×/4.4× 链式+arange 修复、3d 130× 窄行 deinterleave）。要点：①DESIGN §1.6.4 实验裁决三件套（结构+阈值+回写路径）在续跑会话直接闭环——预注册的价值实证；②session-1 标量化归因被操作数变体对照证伪（TRAP-parallel2d-bm-ge2-scalarize）；③C6 双 kernel 的 msprof 口径 = 两段 Task Duration 之和（--kernel-name 分别捕获）。工件（工作区过程文件，任务级溯源）：examples/argmax/_argreduce_kernel/perf_opt/opt_log.md（[DESIGN_LIMIT] 双门槛达成：2.20×>2× + 设计层归因）；知识域证据见本库 repro/repro_nsplit_pattern.py 等六件。

#### 同族第二实例：argmin 共享 ArgreduceKernel 的 Stage 4（`scenario=optimize`，`stage_plan=[4]`，2026-10-15）

任务 `argmin-_argreduce_kernel-20261015T023310Z`（910B3 / CANN 9.0.0 / tilelang 417dae9）；4 轮 12 个实验分支，`stop_reason=budget_exhausted`（rounds 4/10、experiments 12/40，主要结构候选已充分验证），几何平均 1.30×（6 tune workload：2.21×/2.63×/1.004×/1.47×/1.011×/1.011×）。要点：① **C6 N-split 分派翻转**（lm-head (4,102400) 双 dtype 11.898/11.914 µs，0.453×/0.380×，tn=2560/nchunk=40；跨设备跨工具链与 argmax 锚点同量级 → PL-1.20 第二实例）；② **block_m 阶梯重标定**（用编译报错口径的真实 UB 足迹 ~16 B/elem 替换 Stage 3 的 64 KB 探针代理预算 → bf16 hidden-state 由 bm=1 提档 bm=2，54.010→36.865 µs = 1.47×）；③ **DMA 粒度是拷贝地板的真实杠杆**（16 KB → 64 KB：15.82 → 13.61 µs，CONST-copy-floor-dma-granularity）但 64 KB 档需 4× UB，与算法缓冲不可共存；④ **正确性证伪**：per-row 展开写共享 (bm,N) 缓冲静默错值（12.91–15.25 µs「更快」= 伪收益，→ TRAP-scf-for-row-offset-silent-miscompile）；⑤ **`[DESIGN_LIMIT]`**：hidden-state fp16 目标（≤13.7 µs / Ratio ≥70%）在本机同粒度纯拷贝地板（15.82 µs / Ratio 上限 60.7%）之下不可达，`perf_feedback.md` 给出量化归因 + 本机实测常数回填（附录 B），路由「附录补记」；⑥ **优化场景下的集成观察**：该 op 的 wrapper 在同一 report 上 `status=partial`（correctness 42/42 通过但 6 个 msprof case 中 2 个 `profiler_valid=false`、2 个「valid 实为欠归因」）——根因是「多 `@T.prim_func` 名 + wrapper 未上浮 `msprof_kernel_name`」，见 CASE-argmax-argreduce-integration 缺陷 ⑤ 与 queue VP-2026-0232。工件（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/{opt_log.md, perf_feedback.md, perf_records.jsonl, profiles/}`；知识域证据见本库 `repro/{CONST-copy-floor-dma-granularity,CONST-aiv-time-decomposition,TRAP-scf-for-row-offset-silent-miscompile}.py`。

---
id: CASE-argmax-argreduce-integration
kind: case
family: [reduction]
apis: []
dtype: [fp16, bf16, fp32]
device: 910B2C, 910B3
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z（首证，910B2C / CANN 8.5.0）+ argmin-_argreduce_kernel-20261014T151121Z（第二证，910B3 / CANN 9.0.0）+ argmin-_argreduce_kernel-20261015T023310Z（第三证：缺陷 ⑤ 多 prim_func 采数名歧义）+ count_nonzero-_logical_reduce_kernel-20261016T005019Z（第四数据点：wrapper config 继承 GPU 启发式的预防性重接线 + 缺陷 ⑤ 未复现）
toolchain: 首证 tilelang 0.1.2+ubuntu.22.4.npuir / Ascend910B2C / CANN 8.5.0 / git 013dbbf5；第二证 tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3（20 AICore / 40 Vector）/ 2026-10-14；第三证 tilelang 0.1.2+417dae9（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro-missing
---

### TileOPs reduce 族集成契约参考（argmax 首集成的胶水修复档案 + argmin 共享 kernel 第二证）

`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/`（任务级溯源，允许失效——集成包为任务交付件，未随知识库演进存活；integration_log.md + wrapper `argmax.py` + history_version/argmax_s5_attempt1.py 修复前备份）。触发条件：任一 reduce 族算子接入 TileOPs（ArgminFwdOp 等共享 ArgreduceKernel 的迁移单元前置阅读）。要点：① Op 层 `_ReduceOpBase._prepare_input` 在 `_kernel_handles_padding=False` 时 pad 到 align_up(N,256)——kernel jit 声明宽度须取 N_padded（raw-N 契约与 Op 层 F.pad 冲突的症状：行 0 正确、行 ≥1 全错且不报错，初次 report 17 failed 与 N_padded≠N 用例 100% 重合）；② wrapper `default_config` 转调 kernel 模块 `_select_config`（GPU 时代 smem 启发式选出的 block_m 在新 kernel 上直接 `ub overflow, requires 2097408 bits while 1572864 bits available`）；③ argmax 语义 -inf pad 与首现索引兼容（全 -inf 行 kernel 与 torch 同返 0），argmin 共享 kernel 时 `_pad_value` 须为 +inf；④ 修复后 42/42 + benchmark 5 case 全有效（run.json 20260928_030100_974855）。

#### 第二证（argmin 共享 kernel 迁移，910B3 / CANN 9.0.0，2026-10-14）——同 kernel 先例的胶水点复用与跨工具链复现

`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/`（任务工作区/集成包，未上库，允许失效——`ArgminFwdOp`，与 ArgmaxFwdOp 共享 `ArgreduceKernel`；integration_log.md + `history_version/argmin_s5_attempt1.py`）。**要点 ② 逐字节复现**：本任务 Stage 5 首跑 report `20261014_160039_216067_ArgminFwdOp` 失败 40/42（2 个失败用例 100% 落在 N_padded=4096，fp16 `ub overflow, requires 2359808 bits while 1572864 bits available` / bf16 `2361856 bits`），另有 lm-head `[4,102400]` 的硬 `ValueError`（single row exceeds 49152-byte budget，`49152 // 204800 == 0`）——根因同为首证要点 ②：Stage-0 wrapper 继承的 GPU 48 KiB SMEM 启发式 `SHARED_MEMORY_BUDGET_BYTES // (N_padded*elem)` 选出 `block_m=4` 并以 override 传入工厂，而 kernel 内 `_select_config` 校准阶梯给 fp16 `block_m=2` / bf16 `1`（含 `min(block_m,M)` 钳制与窄 N 预算减半，外部启发式均不掌握）。**修复形态（可直接复用的胶水代码）**：wrapper `argmin.py#default_config` 转调 `from .argmin_kernel._argreduce_kernel import _select_config` 后取 `_select_config(self.M, self.N, self.dtype_str)["block_m"]`；kernel / tests / manifest / op 零改动。修复后 42/42 正确性 + benchmark 9/9（6/6 report performance case 有效 msprof，`invalid_profile_count=0`/`profiler_fallback_count=0`），report run `20261014_160246_645105_ArgminFwdOp`，attempts=1。
- **诊断指纹（跨两族/两工具链复现）**：编译期 `ub overflow, requires N bits` 恰好命中**某一个** N_padded（此处 4096）而更小 shape 全过 → 先查「谁传的 block_m、是否经 kernel 校准」，而非 kernel 算法缺陷。
- **要点 ①/③ 的第二证与边界修正**：argmin 迁移的 Stage 1 做了双契约对照（`_kernel_handles_padding=False` + `_pad_value=+inf` 保留 vs 翻转为 raw-N），结论**保留 pad 契约、零 Op 层改动**——原因是 argmin manifest 无窄 N 负载（大 N 为主），pad 契约的流量代价不显著，而 raw-N 翻转触及 wrapper/Op 契约面更大；pad 方向反向（`-inf`）会让 pad 列在 argmin 中胜出并静默错值。pad 契约下 `N_padded` 恒为 256 倍数（`_pick_tile_n` 候选含 256 且偏好整除 ⇒ tiled 路径静态尾 tile `tail ≡ 0`）；且在 `N_padded` 全域计算 argmin 候选是**索引中性**的（实列索引恒小于 pad 列索引）——pad 契约下 kernel 无需掩码/尾 tile 特化，N 向控制流最简（`verify_equiv.py` E5 组 264 行含全 `+inf`/tie/`±inf` 混合，0 违反）。
- **跨工具链重验（新设备/新 CANN）**：设计 §9.1 的 argreduce 陷阱族 C-1..C-8（全部绑定 910B2C / CANN 8.5.0）在 910B3 / CANN 9.0.0 + 本任务规避形态下**未复现**（resident bm=1/2/8 × fp16/bf16/fp32 与 tiled bm=1 融合式首跑编译+精度 0 失败；L0 20 / L1 11 / L2 3 / Boundary 3 全过、`torch.equal` 精确 max_diff=0）；设备常数复验与设计逐值一致（aicore=20 / `aivector_core_num`=40 / `ub_cap`=196608）。**本 CASE 在 kb_stale_check 中被标 stale（工具链戳不同）但其缺陷 ② 在新工具链上逐字节复现**——「stale 标记」不构成对该类共享类知识的否定（消费口径见 queue VP-2026-0101 两分法）。
- **实测（910B3，msprof Task Duration，集成态）**：smoke `[32,256]` 1.80/1.86/1.88 µs（fp16/bf16/fp32，bm=8）；lm-head `[4,102400]` 26.08/31.68 µs（fp16/bf16，tiled bm=1 tn=5120）；hidden-state `[2048,4096]` 31.16/54.10 µs（fp16 bm=2 / bf16 bm=1）；non-aligned `[256,300]` 3.10/3.32 µs（fp16/bf16，bm=8）。3 条 smoke 记录仅存在于 `benchmark.json`（`tileops/reporting/analyzer._is_smoke_record` 把 label 含 "smoke" 的用例从 `run.json#performance.cases` 与 `report.md` 过滤，即使 msprof profile 有效）——数值引用走 `benchmark.json`。
- **durable 载体**：argreduce 边界语义（±0.0 全序 / NaN→哨兵 `2^30`）自包含于 traps-runtime.md TRAP-argreduce-edge-semantics；`T.reshape` 被 `npuir_reshape(src,dst)` 覆盖自包含于 traps-compiler.md TRAP-npuir-reshape-dst-buffer-shadow（含知识域 repro）；索引失配指纹见 error-fixer references/precision-patterns.md。
- 溯源（任务工作区，未上库，允许失效）：`examples/argmin/_argreduce_kernel/{DESIGN.md#§0.2/§5.6/§1.6.1-E5/§9.1, RETROSPECTIVE.md Stage 1/2/3}` + `examples/argmin/RETROSPECTIVE.md Stage 5` + 集成侧 integration_log.md。
- **缺陷 ⑤（第三证 / 新形态，2026-10-15，argmin 的 Stage 4 optimize 集成跑）：多 `@T.prim_func` 名 + wrapper 未上浮采数名 → report `status=partial`（correctness 42/42 全过）**。现象：perf_opt 版 kernel 文件（`perf_opt/_argreduce_kernel.py`）含 **4 个不同名的 `@T.prim_func`**（`main` / `argreduce_tiled` / `argreduce_partial` / `argreduce_merge`），而 baseline kernel 文件只有 `main` 一个 distinct 名。后果（6 个 msprof case，报告 `examples/TileOPs/reports/tileops/20261015_061354_481681_ArgminFwdOp/`，基线对照 `…/20261015_003723_597141_ArgminFwdOp/` `status=passed` 6/6 valid）：① 派生名歧义时取到**某一个** kernel——lm-head 两 dtype 派生 `argreduce_merge` → 只捕获 merge 段，记 **3.640/3.180 µs（`profiler_valid=true`）**，而 C6 真实 workload 时延 = partial + merge = **11.898/11.914 µs**（**看似 valid 实为欠归因**）；② 派生 `None` 时退回**无过滤全捕获**——non-aligned 两 dtype 捕获到 `MemSet_*` + `PadV3_*` + `main` 三 op → **`profiler_valid=false`**，记 5.420/5.400 µs（含 Op 层 pad/fill 算子），而 kernel-only 实测 3.192/3.317 µs；③ 唯一「恰好正确」的 hidden-state 两 dtype（派生 `None` 但窗口内仅 1 op）记 31.601 µs / 36.421 µs。机制（源码口径）：`tileops/benchmark/msprof.py::_derive_kernel_name_from_prim_func`（Pass 2 静态源码扫描）在候选模块内 distinct `@T.prim_func` 名 > 1 时返回 `None`/歧义；`msprof.py` 的解析链为「显式 `kernel_name` → `functor.msprof_kernel_name` → `functor.kernel.msprof_kernel_name` → `TILEOPS_MSPROF_KERNEL_NAME` → 派生名」。同族对照（tier-2 解析正例）：argmax wrapper 显式 `self.msprof_kernel_name = getattr(self.kernel, "msprof_kernel_name", None) or "main"` 把工厂的 `_func.msprof_kernel_name` 上浮到 Op 层，故其 C6 双发射可归因；argmin wrapper 缺这一行（Stage 5 integration_log §issues 曾以「report-layer cosmetic」记录该现象——本次实证其**非 cosmetic**）。判读口径：`run.json.summary.invalid_profile_count > 0` 或「case 时延显著低于 workload 流量下界」⇒ 先查派生采数名与捕获段数，再谈性能。**该缺陷属 integration 期接线缺口，`optimize` 场景下 wrapper 仅允许翻转 switch 块、本任务范围内不可修**（处置提案见 queue VP-2026-0232 / VP-2026-0234）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/{integration_log.md#issues, ../../argmin.py, perf_opt/perf_records.jsonl}`。
- **第四数据点（count_nonzero 计数类迁移，910B3 / CANN 9.0.0 / 2026-10-16）——同缺陷类的`预防性`命中 + 缺陷 ⑤ 未复现的两点事实**。① **wrapper 出厂 config 继承 GPU 期启发式**：脚手架产出的 `count_nonzero.py` 保留 48 KiB SMEM 启发式，manifest `(2048,4096)` ⇒ `block_m=8` ⇒ `8×4096 = 32768` 元素 > kernel 侧硬上限 **24576 元素**（8 B/元素探针口径）⇒ 首跑会以 `ub overflow` 编译失败，且该启发式无法表达 `N > E_cap` 的列分块路径。本任务在**第一次 report 之前**依 integrator 已知修复目录重新接线（`default_config` 委托 `_select_config(self.M, self.N, dtype_str)["block_m"]`，kernel/tests/manifest 零改动）⇒ 集成一次通过、report `status=passed`（correctness 37/37、benchmark 7/7）。**口径补充**：kernel 侧 `_resolve_block_m` 的防御性回落**不是** wrapper 接线的替代（回落只让编译不炸，出厂契约仍以 kernel tiling 单源为准）。② **缺陷 ⑤ 未复现**：本任务 kernel 同含多 `@T.prim_func` 变体（resident / 列分块 / 列分块带尾块），但 report 4/4 msprof case 有效（`invalid_profile_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`）——与缺陷 ⑤ 的 partial 形态形成对照；仅 `report.md` 的 `Kernel` 列显示 `main`（本任务判定为 reporting-layer 展示行为，与 argmin 的欠归因非同一现象），差异条件（wrapper 是否上浮 `msprof_kernel_name` / 派生名歧义）待后续同族任务定钉。溯源（任务工作区，未上库，允许失效）：`examples/TileOPs/tileops/kernels/reduction/count_nonzero/{count_nonzero.py, count_nonzero_kernel/{integration_log.md, integration_report.json}}`。

---

id: CASE-pool-adaptiveavgpool2d-migration
kind: case
family: [pool, pooling]
mode: [developer]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z（迁移）；adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（Stage 4 调优 round 1/2）；adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z（Stage 4 续调 round 3）
toolchain: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13；**Stage 4 round 1/2 段** tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14；**Stage 4 round 3** tilelang 0.1.2+31738b08cb8f44e9596aba32ffeda82040d2152c（commit 31738b0）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: none
---

### `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/` + `examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/`（任务工作区/集成包，未上库）

**durable 载体**：C 轴向量轴模式自包含于 layout.md PL-1.2/PL-1.4 + PL-1.21（纯 Vector persistent 分核模板）+ PL-1.5（runtime let-bound 标量倒数 / 3D reduce_sum 链）；`T.vcast` 与 `range` 两条陷阱自包含于 traps-compiler.md TRAP-vcast-shape-and-gm-source / TRAP-tir-range-loop-not-unrolled（含知识域 repro）。

pool 家族首个 TileOPs 迁移算子（AdaptiveAvgPool2dFwdOp，Developer 模式、纯 Vector、Stage 4 跳过）。要点：①**向量轴 = C 轴**（核内 NHWC 二轴 `T.transpose` 重排后 C 轴累加；W 轴 bin 宽 9–10 非 8 倍数 + 变长段归约不可向量化而弃选，§1.6.3 决策）；②**几何特化四路径**（一般网格 / C4 全局 `out=1×1` → 3D `T.reduce_sum` 链 / C4b 恒等 `out==in` → 纯拷贝 / persistent 分核），退化路径以编译期容量判据 `identity_fits`/`global_fits` 决定是否走特化、否则回退一般路径（纯优化，语义与精度不变）；③**C4b 恒等路径的 `num_logical` 口径**为整平面单元 `N*num_cblk`（`(1,8,9,11)`=1，非通用式 9——设计修订 2 轮的阻塞根因，见 queue VP-2026-0147）；④**设计修订链 3 轮**（Stage 2 检视两轮不通过：v0 漏盘 `acc` 零初始化 + 候选 #1 倒数乘采纳结论跨 5 节矛盾 → v1 C4b 恒等 `num_logical`/单元粒度/解码三要素不自洽 → v2 通过），Stage 3 首轮即 `[PRECISION_PASS]`（零精度重试）。**实测**：L0 10/10、L1 13/13、L2 7/7、Boundary 6/6 全过；worst `max_diff` fp16 = 1.221e-04（resnet-global，C1 倒数乘 ≤1 ulp）、bf16 = 0.0；TileOPs 集成 7/7 正确性（avg max abs err 0.00e+00），msprof 有效 bench 3 例——resnet-global `[8,2048,7,7]→(1,1)` fp16 169.26µs / Ratio 2.30%、spp-6x6 `[2,128,56,56]→(6,6)` fp16 16.60µs / 3.81%、nondiv-7x7 `[2,64,55,57]→(7,7)` bf16 14.50µs / 2.35%（smoke 3.70µs）。适用触发条件：adaptive/窗口池化类迁移设计（变长 bin 的整数边界差口径 + 向量轴选型）；退化路径容量判据与回退设计参考；纯 Vector 多单元 persistent 分核形态；单算子 TileOPs 集成（两层 wrapper 类继承 → gate `S5-ASCEND-MODE`，见 queue VP-2026-0150/0151）。

**Stage 4 续档（2026-10-14，`scenario=optimize`，task `adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z`）**：C4 全局路径 baseline 166.26 µs → **16.78 µs（-89.9% / 9.91x）**，中止原因 `plateau`（round 2 无 >3% 有效提升 + 参照结构 diff 检查完成）。要点：① **`T.transpose` 链在当前工具链失效**——`hivm.hir.vtranspose` 在池化族大 shape 上被降级为 UB→GM→UB 往返（`UB_to_GM_datas` = 48 KB/block、`aiv_mte3_time` 43–58 µs、单次 ≈8–19 µs/单元），而 `(64,56)→(56,64)` 的 2D f16 转置仍走 UB 内（8.5 KB/block）⇒ 转置代价强依赖 shape/dtype 形态（traps-compiler.md `TRAP-transpose-ub-gm-roundtrip`；本条同时把 layout.md **PL-1.1** 标为 stale 并收窄其适用域）；② **替代结构 = 扁平空间轴归约**（`(N,C,1,H*W)` 声明 + 内轴 49 宽单次 `reduce_sum`，**PL-1.23**，9.96x，逐位精确）；③ **单波 CH 分派**（`_pick_ch_general`：`units(base)·2 > vector_cores` 则用 base，否则折半且**仅当折半后仍单波**才采纳——**PL-1.4** 第五数据点；同规则解释 spp 上 CH=32 反而 +49%）；④ 其余候选全否决（`T.Pipelined(2)` 噪声内 -0.4%、`T.vcast` 子区域编译失败、`CH_GLOBAL∈{8,16,64}` 劣化、f16 直接归约仅 3.6% 且有精度风险）；⑤ **msprof roofline `Ratio` 口径更正**：本次结构改写**同时**删掉两级窄归约与转置，使 `total_fops` 16.86M→2.20M（7.66x），故 10.12x 时延下降只换来 Ratio 2.30%→2.90%（1.26x）——`Ratio = (total_fops/duration)/computility`，**不与 latency 成反比**；按此口径「池化类 Ratio→70%」结构不可达（resnet 时延地板 ≈4 µs ⇒ 上限 ≈12%）。集成侧：单算子 report `status=passed`、correctness 46/46、Avg Max Abs Error 0.00e+00、Ratio Range 2.90%–4.04%（`examples/TileOPs/reports/tileops/20261015_003854_809663_AdaptiveAvgPool2dFwdOp/`，provenance 允许失效）。durable 载体：PL-1.23 / PL-1.4（单波判据）/ PL-1.1（stale 收窄）/ traps-compiler.md 三条 TRAP + `repro/{TRAP-transpose-ub-gm-roundtrip,PL-1.23-flat-spatial-axis-reduce,PL-1.4-single-wave-ch}.py`。

**Stage 4 续档 round 3（2026-10-15，`scenario=optimize`，task `adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z`）**：本轮净提升 resnet **−2.9%**（ab 合并中位 −3.79%）、spp −4.3%～−5.27%、nondiv 噪声内/零回退；累计 vs round-0 baseline resnet **−90.2%（10.20x）**、spp −5.4%、nondiv −9.0%，中止原因 `budget/scope` 收束（剩余候选全阻塞或证伪）。要点：① **Step 0 地板标定**（口径 = `msprof op` Task Duration，median/20）：空转固定开销 `F_empty(40)` = **4.81 µs**（占优化后 16.29 µs 的 ~30%）、1 轮读地板 5.64 µs、13 轮读地板 9.64 µs；校准线 = 地板/0.7 ⇒ 三 workload 均未达（且 70% Ratio 在 compute-roofline 口径下结构不可达）；② **G11 fp16 精简归约链**（`reduce_sum(f16→f16)+单次 vcast`，删 f32 全平面中转；UB 单价 64→48 B/平面元素 → 解锁 `CH_GLOBAL=64`）——resnet `aiv_vec_time` 3.88→2.12 µs、Task ab −3.79%；代价 f16 累加精度（max_diff 1.2e-4→4.9e-4，仍在 1e-3 内）；**bf16 不适用**（无 bf16 累加形态）；③ **G14 CH 判据 v2**（单波集合内「核占用最大 → 尾块浪费最少 → 更宽 CH」，取代 round-1 折半判据）——spp CH 64→48（24→36 核）Task ab −5.27%，nondiv 保持 CH=32 零回退；④ **证伪/阻塞**：slab 结构（一次大 DMA + 核内分段归约）被 UB 记账封死（常驻 staging buffer 按归约单价计费，单次归约输入 ≤≈125 平面）、`auto_multi_buffer` 无重叠收益、均值乘在 fp16 做精度失败（max_diff 5.87e-1）；⑤ **Ratio 双口径分列**：resnet Ratio 45.40%（分母 `computility` 因删 fp32 op 从 4.53→0.28 TOps/s 塌缩所致，非利用率改善）；`CH_GLOBAL=120` 可达 Ratio 68.97% 但时延 +2.9% 回退 ⇒ 未采纳（Ratio 硬目标与 latency 目标冲突实例）。durable 载体：PL-1.23（fp16 同型链 dtype 变体，VP-2026-0248）/ PL-1.4（CH v2，VP-2026-0249）/ `CONST-reduce-ub-per-element`（单价 48 B/元素 + 常驻 buffer 计费，VP-2026-0246）/ `CONST-msprof-task-floor-910B3`（固定开销—Block Dim，VP-2026-0247）；能力缺口 `.agents/evolution/capability-gaps.md#CG-2026-0026`（recurring）/ `#CG-2026-0027`（recurring）/ `#CG-2026-0029`（add）。

---

id: CASE-pool-adaptivemaxpool2d-migration
kind: case
family: [pool, pooling]
mode: [developer]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z
toolchain: tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13
repro: none
---

### `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/` + `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/`（任务工作区/集成包，未上库）

**durable 载体**：max 归约的 NaN 探测陷阱与其合法绕法自包含于 traps-compiler.md TRAP-vcmp-ne-ordered-nan（含知识域 repro）；C 轴向量轴 / 纯 Vector persistent 分核模板见 layout.md PL-1.2/PL-1.4/PL-1.21；几何特化路径模式见 queue VP-2026-0143。本任务函数级 `RETROSPECTIVE.md` 与 `SESSION_TIMING_ANALYSIS.md` 为 provenance（允许失效）。

**Stage 4 续档（2026-10-15，optimize 场景；tilelang 0.1.2+489385e9 / CANN 9.0.0 / Ascend910B3）**：对该集成的四个 benchmark workload 做调优，产出
`perf_opt/_adaptive_max_pool2d_kernel.py`（基线与 wrapper 未改）。**结果（msprof op Task Duration）**：
`global-1x1` [8,2048,7,7] fp16 **320.83 → 18.86 µs（17.0x）**；`spp-6x6` [2,128,56,56] fp16 20.37 → **18.89 µs（-7.3%）**；
`nondiv-7x7` [2,64,55,57] bf16 18.32 → **15.46 µs（-15.6%）**；smoke 仅精度回归（bit-exact）。三 dtype 全 **max_diff = 0.000e+00**（atol=rtol=0）。
结构改动：① C4 全局路径改**扁平空间轴**（`x` 声明 `(N,C,1,H*W)`，两级窄归约 → 单次宽归约；`GM_to_UB` 336 → 48 KB/block）；
② NaN 追踪改 **`isnan(reduce_sum(x*x))`（原位平方）**，归约 UB 单价 96-128 → 32 B/平面元素（见 constants.md `CONST-reduce-ub-per-element`；layout.md PL-1.22 ⑧）；
③ general 路径 **最小可行 CH 单波分派**（spp CH 64→48、nondiv 64→32；layout.md PL-1.4 第六数据点）。
**负向结论（durable）**：PL-1.13 的 subid 分片**不适用于纯 Vector kernel**（`subid` 恒 0，分片会静默丢弃一半逻辑单元）——见 attention.md PL-1.13 适用域收窄 + `repro/PL-1.13-vector-kernel-subid-zero.py`。
**未达 70% 口径**（global 18.86 vs 本轮标定 70% 线 7.27–9.79 µs；余下缺口归因于归约 lowering 的 UB 记账限制 CH 与 reduce/DMA 不重叠，属工具链层）；
**未触发 `[DESIGN_LIMIT]`**（双条件未同时满足：天花板非算法/设计层，且结构加速已在 Stage 4 内消化）。
provenance（允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/perf_opt/{opt_log.md, perf_records.jsonl, workload_inventory.json, copy_floor_calibration.jsonl}`。

pool 家族第二个 TileOPs 迁移算子（AdaptiveMaxPool2dFwdOp，Developer 模式、纯 Vector、Stage 4 跳过；同族追加——结构骨架直接对照前作 `examples/adaptive_avg_pool2d/`（同族前作任务工作区，未上库，允许失效））。要点：①**向量轴 = C 轴**（核内 NHWC 二轴 `T.transpose` 重排后 C 轴 `T.vmax` 归约），与 avg 同构，差异仅在归约算子（max vs sum）与 NaN 追踪；②**max 无代数化简**（非线性 → 常数权重卷积/GEMM 不等价；无逆元 → 前缀差分不可用），"数学等价优化"只余结构特化（C4 全局整平面 `reduce_max` / C4b 恒等纯拷贝）与归约融合；③**NaN 语义显式追踪**（"窗口内任一 NaN → 输出 NaN" + `{+inf,-inf}` 同窗输出 `+inf` 非 NaN）——不能用求和类探测，改用 `eq` 掩码乘性累积（详情见 TRAP-vcmp-ne-ordered-nan）；max 累加器单位元为 `-inf`（`T.clear` 置 0 在全负窗口出错）；④**Stage 3 near-miss**：首版按 DESIGN 字面用 `vcmp(...,"ne")` 做 NaN 探测，L0-10 语义门禁**假阳性通过**（靠 `vmax`/`reduce_max` 的 NaN 传播兜底），5 组最小探针定位后改 `eq` 取反，全部用例复跑仍全过且 `max_diff=0`；⑤**Stage 1 gate-1 重试 1 次**（`S1-REF-PATHS`：ASCII `:行号`/`:函数名` 紧贴路径 + `…` 缩写 token——见 queue VP-2026-0020 证据链追加）；⑥**Stage 5 integrate 一次成功**（`integrate_kernel.py` 对生成 `__init__.py` 仍非幂等，单次运行规避——同 VP-2026-0150）。**实测**：L0 10/10、L1 17 项、L2 7/7、Boundary 8/8 全过；worst `max_diff` fp16 = bf16 = **0.0**（max 是选择、无累加，`atol=rtol=0` 判据成立；同族 avg 为 1.221e-04）；TileOPs 集成 correctness 5/5（max abs err 0.00e+00），msprof 有效 bench 3 例——global-1x1 `[8,2048,7,7]→(1,1)` fp16 322.11µs / Ratio 5.38%、spp-6x6 `[2,128,56,56]→(6,6)` fp16 20.42µs / 7.77%、nondiv-7x7 `[2,64,55,57]→(7,7)` bf16 17.98µs / 4.69%（smoke 4.02µs）。Session 墙钟 ≈55min：Stage 3 40.3%（含 NaN 陷阱探针）+ Stage 1 20.9%。适用触发条件：max/归约类算子迁移（NaN 语义追踪 + 单位元 + 无化简边界）；同族追加以对照前作降风险的设计路径；池化族集成（共享 `tests/ops/test_pool.py` → 单算子 report 计兄弟算子用例，见 queue VP-2026-0152）。

---
id: CASE-pool-adaptivemaxpool2dindices-migration
kind: case
family: [pool, pooling]
mode: [developer]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z（函数级 / 迁移）；adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z（Stage 4 调优）
toolchain: tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；**Stage 4 段** tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23（commit 67a0441 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: none
---

### `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/` + `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/`（任务工作区/集成包，未上库）

**durable 载体**：两条 API 陷阱与其合法绕法自包含于 traps-compiler.md TRAP-npuir-vbrc-region-dropped / TRAP-npuir-arange-pattern-with-whole-plane-reduce（均含知识域 repro）；索引语义判据与「比较-更新次序」见 queue VP-2026-0159/0160/0161；同族骨架与 NaN 追踪见 layout.md PL-1.2/PL-1.21 + traps-compiler.md TRAP-vcmp-ne-ordered-nan + CASE-pool-adaptivemaxpool2d-migration。本任务 `RETROSPECTIVE.md`（op 级/函数级）与 `SESSION_TIMING_ANALYSIS.md` 为 provenance（允许失效）。

pool 家族第三个 TileOPs 迁移算子（AdaptiveMaxPool2dIndicesFwdOp，Developer 模式、纯 Vector、Stage 4 跳过；同族追加）。要点：①**双输出（values + int64 扁平索引）**——索引 = H×W 平面行主序扁平下标 `ih*W_in+iw`，并列取**首个命中**，窗口含 NaN 时值 NaN + 索引取**最后命中的 NaN**，`{+inf,-inf}` 同窗 → 值 `+inf`（§0.1 12 行 torch 2.9.0 CPU 实测）；②**first-hit 的向量化极值方向**：把「严格 `>` 顺序覆写」改写为集合取极值时取**破纪录集合扁平最大值**（`vselect(gt, fv, -1)` + `vmax`）——误用 `vmin`（"第一个破纪录"）静默给出窗口内首个局部峰值的索引（v0 版本 N3 首轮 20426 mismatch，修正后 340 例全 `EQUIV_PASS`）；③**共享累加器的比较-更新次序是正确性字段**：`vmax(acc, v)` 先于 `v > acc` 时判据恒假、索引静默退化为窗口首元素（v0 §3.3 缺陷，公式级 D-1 6 项全 PASS 未覆盖；v1 新增 `N3-ord` 次序守卫 342 checks）；④**Stage 1 设计修订 1 轮**（Stage 2 检视 #1 不通过：first-hit 次序 + 退化路径 UB 预算低报 ≈46% → v1 `CH_deg=min(C,32)` + 逐 buffer 复算 + N3-ord；#2 通过）；⑤**Stage 3 两条 API 陷阱**（`T.vbrc` 标量→region 静默丢弃〔int32 形态向量核 fault 507034〕；`arange` 平面下标 pattern in-chain 读回损坏——values 全对而 14050/16384 索引错，`out=1×1` 交回一般路径）——均转正为 `repro/*.py` 并机械入知识域；⑥**Stage 5 integrate 一次成功**（`integrate_kernel.py` 聚合 `__init__.py` 非幂等第三现，单次运行规避——同 VP-2026-0150）；⑦int32 载体元素级链（transpose/vcast/vmax/vcmp/vselect/rank-2 slice 累加）实测逐位可用，「无 int32 转置先例」不构成能力缺口。

**实测**：Stage 3 内嵌分层测试 L0/L1/L2/Boundary 全过、`[PRECISION_PASS]`——values `max_diff = 0`（fp16/bf16；max 是选择、无累加，`atol=rtol=0` 判据成立），int64 索引 `torch.equal` 精确（0/104487 mismatch）；TileOPs 单算子 report `status=passed`、本算子 rollup **7/7**（`avg_max_abs_err = 0.00e+00`，文件级 `correctness_tests=17` 含兄弟算子 avg 6 + max 4）、benchmark 12/12 通过、9 msprof case 全有效（`invalid_profile_count=0`、`profiler_fallback_count=0`）；本算子 3 例 msprof——global-1x1 `[8,2048,7,7]→(1,1)` fp16 **151.60 µs** / Ratio 2.54% / BW 1.75 TB/s、spp-6x6 `[2,128,56,56]→(6,6)` fp16 **85.36 µs** / 4.57%、nondiv-7x7 `[2,64,55,57]→(7,7)` bf16 **81.60 µs** / 2.56%（smoke 5.92µs）。Session 墙钟 184.3 min（op 级 10963s elapsed）：Stage 3 开发 62.8%（6944.7s，含两陷阱探针）+ Stage 1 首次设计 12.0% + 设计修订 6.6%。**适用触发条件**：双输出/带索引归约类迁移（索引语义 + 极值方向 + 次序守卫）；`vbrc`/`arange` pattern 形态的落点与 in-chain 消费风险排查；池化族追加迁移的同族骨架复用（Stage 3 占墙钟 62.8% → 复用语料）；池化族集成（共享测试文件 → 读 per-operator rollup）。

---

id: CASE-pool-maxpool2d-migration
kind: case
family: [pool, pooling, window]
mode: [developer]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: max_pool2d-_max_pool2d_kernel-20261014T062320Z（函数级；op 级 max_pool2d-max_pool2d-20261014T061506Z）
toolchain: tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: none
---

### `examples/max_pool2d/_max_pool2d_kernel/` + `examples/TileOPs/tileops/kernels/pool/max_pool2d/`（任务工作区/集成包，未上库）

**durable 载体**：`T.vcast` 同 dtype 陷阱与合法分派形态自包含于 traps-compiler.md TRAP-vcast-same-dtype-rounds（含知识域 repro）；max 归约 NaN 追踪实现形态见 layout.md PL-1.22；C 轴向量轴 / tiling / 纯 Vector persistent 分核见 layout.md PL-1.2/PL-1.4/PL-1.21；**窗口池化退化特化条件与 adaptive 不可平移**见 queue VP-2026-0172；**NPU reference 后端（torch_npu）能力缺口 + 纯选择类 CPU golden 回退**见 queue VP-2026-0173。本任务 `RETROSPECTIVE.md`（op 级 / 函数级，四个 Stage 章节）与 `SESSION_TIMING_ANALYSIS.md` 为 provenance（允许失效）。

pool 家族**第四个** TileOPs 迁移算子、**同族追加**（AdaptiveAvgPool2dFwdOp → AdaptiveMaxPool2dFwdOp → AdaptiveMaxPool2dIndicesFwdOp → MaxPool2dFwdOp；Developer 模式、纯 Vector、Stage 4 跳过）。与前三个的族内差异：本条是**窗口池化**（窗口形状由 `k/s/p/d` 固定）而非自适应池化（变长 bin）。要点：

①**退化特化条件不可从 adaptive 家族平移**——`out==1×1` 只约束输出网格、**不**保证窗口覆盖整平面（`d>1` 时窗口仅取等差子集 `{k·d}`）；Stage 2 首轮独立推演 + torch 数值反例（`H=W=5,k=2,s=1,p=0,d=4`：正确 0.0 vs 特化 1.0）拦下 v0 的充分条件缺口，设计修订 v1 补 `d_H==d_W==1` 六处同步 + 加机器反例组（C4x，69 case-group 全 PASS 含负向控制守卫）；②**设计修订 1 轮**（Stage 2 首轮 1 阻塞 + 4 建议 → v1 闭合后复检通过；`retry_count=1`）；③**Stage 3 首轮即 `[PRECISION_PASS]`**（一次成稿、零精度轮次）——但首轮暴露 `T.vcast` 同 dtype 陷阱（C4 整平面 fp32 分支 8/8 元素错，相对差 0.194），改用按 dtype 的 `T.copy`/`T.vcast` 分派后逐位一致，并转正 repro；④**容量驱动 CH 选择**（`CH` = 满足 `ub_bytes(CH,W_in,out_W,dtype)×1.7 ≤ 192KB` 的最大 8 的倍数；`W_in=112` 的 resnet-stem 被压到 CH=32、fp32 更落到 CH=16，小 `W_in` 保持 CH=64）；⑤**Stage 5 集成 attempt 1 `[INTEGRATE_FAIL]`**（根因 = **NPU 参考后端能力缺口**：torch_npu 的 `F.max_pool2d` 经 `aclnnMaxPool2dWithMask` 仅支持 `dilation == 1`，`dilation=(2,1)` 用例在 op 执行前抛 `AclNN_Parameter_Error(EZ1001)`）→ attempt 2 受控闭环内把该用例 `ref_program` 回退 CPU golden（纯选择无累加 → 逐位一致、`atol=rtol=0` 保持、用例与 `dilation>1` 覆盖全保留）→ `INTEGRATE_COMPLETED`；⑥**轮次结构特征**：Stage 3 一次成稿（19.7% 墙钟）使 Stage 1（含修订，26.2%）与 Stage 5（含一轮失败重试，21.8%）合计反超——「公式等价验证 PASS ≠ 条件充分」与「精度一次过 ≠ 集成一次过」两条边界同时出现。

**实测**（Session 墙钟 76.7 min / 4603s；op 级 elapsed 4508s、函数级 2981s）：Stage 3 内嵌分层测试 **L0 31 / L1 23 / L2 7 / Boundary 8 全过**、`[PRECISION_PASS]`，fp16/bf16/fp32 三 dtype `max_diff = 0.000e+00`（max 是选择无累加，`atol=rtol=0` 成立；含 NaN / ±inf / 全 NaN 窗用例）；TileOPs 单算子 report `status=passed`、本算子 rollup **22/22**（`max_abs_err = 0.00e+00`，文件级 `correctness_tests=36` 含 adaptive 兄弟算子）、benchmark **24/24**、9 msprof case 全有效（`invalid_profile_count=0`）。本算子 3 workload × 3 dtype msprof（Ratio = roofline 利用率）：resnet-stem `[32,64,112,112]` k3s2p1 fp16 **1390.3 µs**（13.63%）/ bf16 1391.4 / **fp32 2646.5**；vgg-block `[16,128,56,56]` k2s2p0 fp16 **113.6 µs**（14.25%）/ bf16 116.4 / fp32 190.9；alexnet-ceil `[32,64,55,55]` k3s2p0 ceil fp16 **201.8 µs**（13.55%）/ bf16 202.1 / fp32 365.6——per-op Ratio 区间 12.11–14.25%、带宽 1.75 TB/s（≈ fabric 聚合上限，算子带宽受限）。**适用触发条件**：窗口池化（max/avg pool2d/3d）迁移设计（退化特化条件重推 + 容量驱动 CH）；fp32 比较域算子的同 dtype 通路分派（TRAP-vcast-same-dtype-rounds）；含编译期特化分支的 kernel 的「分支归属断言」设计；池化族 TileOPs 集成的 NPU golden 能力边界处置（reference 后端缺口 → CPU golden 回退，queue VP-2026-0173）。

---

id: CASE-pool-maxpool1d-migration
kind: case
family: [pool, pooling, window]
mode: [developer]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: max_pool1d-_max_pool1d_kernel-20261014T082309Z（函数级；op 级 max_pool1d-max_pool1d-20261014T081737Z）
toolchain: tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9，dev root build）+ torch_npu 2.9.0.post8 + torch 2.9.0 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: none
---

### `examples/max_pool1d/_max_pool1d_kernel/` + `examples/TileOPs/tileops/kernels/pool/max_pool1d/`（任务工作区/集成包，未上库）

**durable 载体**：max 归约 NaN 追踪形态（含 1D 第三数据点与单空间轴守卫前提）见 layout.md PL-1.22；容量驱动 CH 与 1D 容量不等式见 layout.md PL-1.4；窗口池化退化特化条件（含 1D `d==1` 同构复现）见 algorithm-candidates.md ALG-pooling；NPU 参考后端 CPU golden 回退见 traps-runtime.md TRAP-npuir-ref-backend-golden-capability；1D 长轴必需结构（输出块 BL + 连续 span 载入 + 起点钳制）与「容量回退不总可达」见 queue VP-2026-0180/0181。本任务 `RETROSPECTIVE.md`（op 级/函数级）与 `SESSION_TIMING_ANALYSIS.md` 为 provenance（允许失效）。

pool 家族**第五个** TileOPs 迁移算子、**1D 首例**（MaxPool1dFwdOp；Developer 模式、纯 Vector、Stage 4 跳过；`_max_pool1d_with_indices_kernel` 不在迁移范围）。**零返工**：`retry_count=0`、各 Stage `stage_retry_count=0`（Stage 1 首次设计一次通过检视、Stage 3 attempt 1 即 `[PRECISION_PASS]`、Stage 5 一次集成成功）。与前四个（自适应池化 / 2D 窗口池化）的族内差异为「**空间轴数 2 → 1**」，三处 2D 结论必须重新推导。要点：

①**长空间轴必需结构**：`L_in` 可达 4096，`(CH, L_in)` 整线驻留 UB 不可行 ⇒ 必需「输出位置块 `BL` + 连续 span 载入 + 起点钳制」——`span=(BL−1)·s+(k−1)·d+1`、`LSPAN=min(span, L_in)`、`s0=max(0, min(base, L_in−LSPAN))`（载入恒在界内、长度静态）、读出下标 `j=iw−s0` 与守卫 `0<=iw<L_in` 等价（机器断言逐 lane 验下标落界）；**2D 的 `W_in<=112` 整行驻留结论不可外推到 1D**；②**`BL` 方向与 2D 的 `BH` 相反**——1D 的 halo 重读随 `BL` 下降（`LSPAN/(BL·sW) → 1`），应取容量允许的**最大 `BL`**（ecg `k5s2p2d2`：BL=31 → ≈1.11× GM 读；BL=1 → ≈4.5×），而 2D 是 `BH=1` 最优；③**1D 无「第二轴判据外提」**（唯一判据 `0<=iw<L_in` 同时依赖 `(o_w,k_w)`；2D 的「H 判据每行一次」在 1D 无对应物）；④**1D 尾块有三类**——任务尾块（`uid < num_logical`）、通道尾块（`real_c = min(CH, C−cb·CH)`）、**输出块尾（`BL_real = min(BL, L_out−ow0)`，1D 新增；2D 因 `W_in` 整行驻留而无此类）**；⑤**`num_logical` 的单元粒度随路径改变**——一般路径 `N·ceil(L_out/BL)·ceil(C/CH)`；C4（`L_out==1`）/C4b（整线恒等）为 `N·ceil(C/CH)`（恒等路径沿用通式会越界，`out==in` 时 `ceil(L_out/BL)>1`）；⑥**退化特化 `d==1` 必要性跨维复现 + 1D manifest 恰好触发整线归约**（`textcnn-global` `[64,256,128]` k128s1p0 → `L_out=1` 触发 C4；2D 三档 manifest 均不触发任何特化）；⑦**容量回退不总可达**——C4 超容时一般路径的 span 亦等于 `L_in`（C4 条件要求核覆盖整线）且每元素 UB 成本更高（一般 `2·eb+9` > C4 `eb+9`）⇒ 该 shape **无 UB 可行路径**，实现如实拒绝（`ValueError: no UB-feasible plan`；`general > cap ∧ global > cap ∧ general > global` 机器断言）；C4b 是反例——`k==1∧s==1` 时一般路径 `LSPAN=BL` 与 `L_in` 无关，回退可达且 NPU 实跑逐位正确；⑧**CH 选择 dtype 感知**（同一 shape，fp32 折半：`textcnn` fp16 CH=64 → fp32 CH=32，`num_logical` 256 → 512）。

**实测**（Session 墙钟 3094.2 s ≈ 51.6 min；`retry_count=0`、各 `stage_retry_count=0`——设计一次通过检视、开发一次精度过、集成一次成功）：Stage 3 内嵌分层门禁 L0/L1/L2/Boundary 全过、`[PRECISION_PASS]`，fp16/bf16/fp32 三 dtype `max_diff = 0.000e+00`（`atol=rtol=0`；含 NaN / ±inf / subnormal / large / zeros 类用例）；设计期 `verify_equiv.py` **117 case-group 全 EQUIV_PASS（0 failures）**，含 C4x 守卫组与逐 lane 下标落界断言；**`_plan` 输出与 DESIGN §5.2 代表表逐值吻合（实现期实跑，非纸面）**——`smoke (1,8,32) k3s2p1`：CH=8 / BL=16 / LSPAN=32 / num_logical=1；`sincnet (32,80,4096) k3s3p0`：CH=64 / BL=26 / LSPAN=78 / 3 392（40 kernels × 85 local）；`textcnn (64,256,128) k128s1p0`：fp16 CH=64 / global / 256，fp32 CH=32 / global / 512；`ecg (16,128,2048) k5s2p2 d2 ceil`：CH=64 / BL=31 / LSPAN=69 / 1 056；`num_kernels = min(num_logical, 40)`（纯 Vector = 实查 20 核 × 2）、`num_local` 为 Python 编译期常量（静态界）。TileOPs 单算子 report：correctness **本算子 rollup 34/34**（`avg_max_abs_err = max_abs_err = 0.00e+00` 逐位；文件级 `correctness_tests=55` 含 adaptive/max_pool2d 兄弟算子——authoritative per-op view 是 rollup，queue VP-2026-0157 已生效知识正例）；`status=partial`，benchmark **6 例失败**（`kernel_size = tuple(op_params["kernel_size"])` 对 int 标量报 `TypeError: 'int' object is not iterable`——Stage 0 脚手架 bench 段缺陷，全部在 profiling 前 abort；**correctness 不受影响**，本算子无有效 msprof 记录 `performance_cases=0`，见 queue VP-2026-0182）。

**适用触发条件**：1D / 长空间轴窗口池化（pool1d、长序列滑窗类）迁移设计——空间轴分块 + 连续 span 钳制载入 + `BL` 取最大 + 三类尾块；**同族 2D→1D 迁移时的结论平移检查**（照搬前先问「该结论依赖几个空间轴 / 轴长上限多少」——整轴驻留 / 守卫外提 / `BH=1` 最优三处 2D 结论均不可平移）；纯 Vector persistent 分核与容量驱动 CH 的 1D 形态参照；容量回退可达性核算（拒绝路径 vs 回退路径）。集成侧经验：`integrate_kernel.py` 的 `--meta` 路径按 **CWD（`examples/TileOPs`）**解析而非仓库根，任务内须传绝对路径（或相对仓库根的路径）；benchmark 段缺陷与 correctness 判定须分离（`partial` report 由 out-of-scope bench 引起，本任务验收判据为 correctness）。

---
id: CASE-pool-maxpool1dindices-migration
kind: case
family: [pool, pooling, window, argmax]
mode: [developer]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z（函数级；op 级 max_pool1d_with_indices-max_pool1d_with_indices-20261014T100411Z）
toolchain: tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…）+ torch 2.9.0+cpu + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: none
---

### `examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/` + `examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/`（任务工作区/集成包，未上库）

**durable 载体**：双输出池化索引契约（2D 扁平 / 1D 单轴）见 traps-runtime.md TRAP-pool-index-contract-dual-output；索引归约极值方向 + 两遍法见 TRAP-index-reduce-extremum-direction；空窗口 regime「torch 不可作 oracle」见 TRAP-torch-oracle-maxpool1d-empty-window（含 repro）；`return_indices=True` 的 NPU 静默损坏 + CPU golden 回退见 TRAP-npuir-ref-backend-golden-capability；双输出 1D 容量闭式 `69+2·eb` + 薄裕度纪律见 layout.md PL-1.4；1D 长轴必需结构见 algorithm-candidates.md ALG-pooling。本任务 `RETROSPECTIVE.md`（op 级/函数级）、`SESSION_TIMING_ANALYSIS.md`、`history_version/design_v0.md` 为 provenance（允许失效）。

pool 家族**第六个**迁移算子、**双输出（value + int64 indices）1D 变体**（MaxPool1dIndicesFwdOp；Developer 模式、纯 Vector、Stage 4 跳过）。与同族前作的族内差异 =「输出个数 1 → 2（索引型）」+「空间轴 2 → 1」叠加。要点：

①**索引型算子的退化特化必须逐路输出独立可证**——单输出变体可采纳的「整线归约 C4」在双输出变体上不可照搬：整线 argmax 索引的自然实现依赖 `T.arange` 派生 pattern 在 `vcmp→vselect→reduce_min` 链内消费（TRAP-npuir-arange-pattern-with-whole-plane-reduce 的已验证断裂链），**值半成立不代表索引半成立** ⇒ 无法证明索引构造独立于该链时走一般路径（本任务设计记录为 C4/C4b 不可采纳）；②**两遍法消除次序依赖**（本任务相对同族前作的改进）：第一遍值/NaN/last-NaN、第二遍在 `acc` 定稿后由等式集合取 `min{iw : x[iw]==acc}`（`BIG` 哨兵），避免了「winner 判据 vs 累加器更新相对次序」陷阱（前作 v0 该次序错致整链失效；本任务 `N2N3-ord` 误形态 39/72 组检出）；③**空窗口契约对齐源**：`d≥2 ∧ p>0` 可产生无界内 lane 的窗口；源状态机给 `(-inf, 0)`，而 torch CPU 在该域返回**越界索引** ⇒ golden 只断言契约、显式记录 torch 越界，并复用 `(BL,CH)` 布尔缓冲实现 `max_idx==BIG ∧ ¬has_nan ⇒ idx=0`（不新增 UB）；④**容量闭式须复现设计值**：双输出 1D 记账 `G = CH·LSPAN·(2·eb+9) + BL·CH·(69+2·eb) + 256`（相对单输出只增单元类 buffer，`+44 B/元素`，`BL` 压 2–3 档：sincnet `BL=16` / ecg `BL=17`）；首版设计把系数写成 `53+2·eb`（与同行分量注释求和 69 自相矛盾）被 Stage 2 拦截，实现侧若改按「实际 buffer 精确记账」则 `BL` 静默选到 18/20、分核表与设计永久漂移（本任务 Stage 3 实测两版 plan 差：`num_logical` 4 864 vs 5 504）⇒ 实现 `_plan` 逐字复现冻结闭式；⑤**验证脚本的枚举成本纪律**：D-1 等价脚本若对全配置矩阵跑「全线公式」（`O(out_l × L_in)`）会 40 min 级卡死（本任务首轮设计因此被取消，占总墙钟 40.3%），改为按 `(k,s,p,d,ceil)` 分支采样 + 特殊化子句 value-only 少量论证后 ≈12–43 s / 1 074 项检查；⑥**集成期 reference 路径受控修复**：torch_npu 的 `F.max_pool1d(..., return_indices=True)` 在 NPU 上 indices **静默损坏**（int8 `(N,C,k,L_pad)` mask buffer，全 dilation；values 正常）⇒ 纯选择类算子改无条件 CPU golden，per-op 47/47、`max_abs_err=0.0`（原 40/47）。

**实测**（Session 墙钟 9 366.4 s ≈ 156.1 min，其中 Stage 1 首轮被取消的长跑占 40.3%）：Stage 3 attempt 1 即 `[PRECISION_PASS]`（L0+L1 阻塞全过、L2 8/8、Boundary 8/8；values `max_diff = 0.000e+00` 三 dtype，**int64 索引 0 / 12 207 226 失配**）；设计期 `verify_equiv.py` 1 074 项全 `EQUIV_PASS`；Stage 5 集成 per-op **47/47**（`max_abs_err=0.0`）、benchmark 36/36（`msprof_case_count=24`、`invalid_profile_count=0`）；非 smoke 性能 3 例——`sincnet` fp16 3 192.84 µs / `textcnn` fp16 138.52 µs / `ecg` bf16 1 429.55 µs，**Ratio 3.40–5.70%、带宽均 1.750 TB/s**（≈ fabric 上限，见 profile-collection.md 的纯选择算子 Ratio 判读口径）。

**适用触发条件**：双输出（value + index）池化/argmax 类算子迁移或设计——索引契约逐要素实测、退化特化逐路输出独立可证、空窗口契约与 torch-oracle 陷阱、索引型 UB 记账、`_plan` 逐字复现设计闭式、等价脚本分支采样；Stage 5 集成期 torch 多输出 API golden 可用性核对（静默损坏判据 = dtype/shape 与契约不符）。

---
id: CASE-norm-groupnorm-migration
kind: case
family: [norm, reduction]
mode: [developer]
dtype: [fp32, fp16, bf16]
device: 910B3
status: verified
origin_task: group_norm-group_norm-20261015T152158Z（op 级）/ group_norm-_group_norm_kernel-20261015T152755Z（函数级；Stage 1/2/3 各 1 attempt、Stage 5 attempts 2）
toolchain: tilelang 0.1.2+9307929320d + CANN 9.0.0 + Ascend910B3 / 2026-10-15～16；torch 2.9.0 + torch_npu 2.9.0.post8
repro: none
---

### `examples/group_norm/_group_norm_kernel/` + `examples/TileOPs/tileops/kernels/norm/group_norm/`（任务工作区/集成包，未上库）

**durable 载体**：fp16 分层 affine 二次舍入陷阱自包含于 traps-runtime.md TRAP-fp16-op-affine-double-rounding（含知识域 repro）；row-reduction 迁移基准形态见 algorithm-candidates.md ALG-layernorm R4；矩式方差定量边界见 ALG-layernorm R1（本任务为第二证）；UB 膨胀平台值第三结构数据点见 traps-compiler.md TRAP-UB-multibuffer-inflation 与 queue VP-2026-0254；rsqrt 近似指令绕法见 traps-runtime.md TRAP-vrsqrt-plain-precision。

norm 族 **GroupNorm 迁移档案**（同族追加于 ada_layer_norm 之后；Developer 单函数、Stage 1/2/3 各 1 attempt 零返工）。要点：① **affine 分层**——op 层经 `kernel(x_2d)` 只传 x，per-channel `(C,)` weight/bias 留在 op 层施加；kernel 的 `(D_padded,)` 行广播 affine 形参保留但恒为单位元（D-1 机器验证 max ulp = 0），单元测试的 affine 模式作为 kernel 参数自造（`_run_case(affine="op"/"unit")` + `_affine_for(dtype)`）；② **Stage 3 全绿 → Stage 5 fp16 marginal 超差**：kernel 级 fp16 门禁用单位 affine 遮蔽了 op 路径真实 per-channel fp16 affine 的二次舍入（7/21 例失败、每例 1–3 元素、max ~1.7e-3 vs 1e-3），逐位比较确证 kernel 无缺陷（0/4096）→ 用户显式授权 ops 层**单点**修复（fp32 affine + 单次 `.to(dtype)`，备份 `history_version/group_norm.py_s5_attempt2.bak`、diff 仅该一处）后 19/0/2（见 traps-runtime.md 同名条目、queue VP-2026-0256/0257）；③ **UB 平台值第三结构数据点**——fp32 直连双缓冲（`stage`+`x_f32`+`sq_f32`+5×(bm,1)）`bm=2, D=4096` 编译运行通过 ⇒ 该结构 ≤24 B/elem，证伪「fp32 直连恒为 26 B/elem」在该结构的普适性（与原始溢出报文 `245760 B = 20×3×4096` 的 20 B/elem 同向）；④ **矩式方差灾难性抵消第二证**（group_norm 形态：大均值 fp16 相对误差 6.018e3、违反 1536 元素；D=2 padding 抵消角点基线 0.995 vs 精确式 0.0094，见 ALG-layernorm R1）；⑤ **msprof bench 4 例**（`n=8,c=128,g=32,spatial=[32,32]` fp16 10.7802µs / bf16 11.0602µs；`n=4,c=256,g=32,spatial=[28,28]` fp16 8.1802µs；`n=4,c=128,g=16,spatial=[30,30]` fp16 6.0201µs；roofline util 10.5–15.4%，带宽 1.75 TB/s——与 constants.md CONST-fabric-bw「聚合 ≥1.73 TB/s」同量级、独立数据点）；⑥ **设计侧复用形态**：wrapper 重导出 `ALIGNMENT = 1` 使 host `F.pad` 分支与 `y[:, :D]` 截断分支均不触发（256 对齐属 CUDA SMEM 后端实现细节、NPU 无对应约束），`T.vsqrt`+`T.vdiv` 组合替代 `T.vrsqrt`，oracle conditioning 审计把「精确常量行（var=0）」与「大均值小方差」降为 Boundary（`F.group_norm` 在该位形自身返回 -2.459e-5 而精确值为 0）。适用触发条件：groupnorm/layernorm 类**分层 affine** 迁移（kernel 只出归一化张量、affine 留 op 层）；fp16 集成期 marginal 精度失败的分层定位；norm 族 UB 预算与矩式/两遍方差取舍设计参考。
---
id: CASE-quant-fp8-quant-migration
kind: case
family: [quantization, fp8, elementwise]
mode: [developer]
dtype: [fp8_e4m3, float32, float16, bfloat16]
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（函数级；op 级 fp8_quant-fp8_quant-20261014T152721Z）；Stage 4 续档 fp8_quant-_fp8_quant_kernel-optimize（2026-10-15）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ torch 2.9.0 / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；Stage 4 续档 tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 + CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: none
---

### `examples/fp8_quant/_fp8_quant_kernel/` + `examples/TileOPs/tileops/kernels/quantization/fp8_quant/`（任务工作区/集成包，未上库）

**durable 载体**：软件 e4m3 编码的舍入/符号/幂精确性三陷阱自包含于 traps-compiler.md `TRAP-parallel-f32-uint8-cast-rounding` / `TRAP-vexp2-not-exact-pow2` 与 constants.md `CONST-npuir-f32-division-not-ieee-rounded`（三条含知识域 repro）；出参 dtype 静默回落见 traps-runtime.md `TRAP-out-idx-uint8-dtype-fallback`（含 repro）；`uint8` 载体 + host `.view(torch.float8_e4m3fn)` 保留对外 dtype 契约的 runtime 事实与 npuir codegen float8 缺口本体见 `.agents/evolution/capability-gaps.md` CG-2026-0018（codegen，加宽至表达式）/ CG-2026-0023（runtime，新增）；等价性验证的覆盖缺口与独立重生成见 queue VP-2026-0212/0213；集成胶水层签名问题见 queue VP-2026-0215。本任务 `RETROSPECTIVE.md`（op 级/函数级）、`SESSION_TIMING_ANALYSIS.md`、`history_version/design_v0.md`、`verify_equiv.py` 为 provenance（允许失效）。

quantization 族**首个** TileOPs 迁移算子 / **首个字节载体（fp8）算子**（FP8QuantOp；Developer 模式、纯 Vector、Stage 4 跳过）。核心矛盾 =「对外 fp8 dtype 契约 vs 工具链任意 float8 构造不可 codegen」——解法是**软件 e4m3 编码 + uint8 载体 + host `.view`**，对外 shape/dtype/数值完全保持（非契约降级）。要点：

①**绕过的三段式**：(a) kernel 侧输出声明 `uint8`（不出现任何 fp8 类型）；(b) UB 内**纯算术**算 e4m3 字节（`bits = 8(ec+6) + rint(a·2^(3-ec))`，`ec = clamp(floor(log2 a), -6, 8)`），`byte = bits + 128·[v<0]`；(c) wrapper 端 `out.reshape(B,S,G,D).view(torch.float8_e4m3fn)`——`.view(dtype)` 是纯元数据操作、**设备侧可用**（不走 `aclnnInplaceCopy`），而 `.to(fp8)`（561103）与 fp8 `zeros_`（161002）不可用。codegen 缺口比登记文本更宽：fp8 作为 **cast 表达式**亦被 `DTypetoMLIRType`（CastNode 路径）拒绝——「fp8 cast + reinterpret」绕过路径不成立。
②**字节级等价需要三件事同时成立**（缺一即 ±1 LSB 或整片错）：(a) 字节出在 **signed i8** 空间（`bits - 128·[q<0]`，clamp `[-128,127]`；否则 ≥128 值饱和为 127 = e4m3 NaN 载荷）；(b) mantissa scale `2^(3-ec)` **精确**（走 `T.vshl(1,k)`；`T.vexp2` rel err ~6e-8 使所有 `n.5` tie 反向）；(c) 量化值用与参考**同一除法形态**（`x/scale`，勿 reciprocal-mul——设备 fp32 除法非 IEEE correctly rounded，乘法倒数在 tie 处不等价）。三者齐备后 fp32 随机数据 524288/524288 逐字节一致，fp16/bf16 仅剩 tie 处 ±1 LSB。
③**设计修订 1 轮（`retry_count=1`，路径 A 检视不通过）**：v0 的符号位用**有限乘数式** `128·min(max(−v·C,0),1)`（`C=1e30`）——对任意有限 C，`|v| < 1/C` 的负值使该式落入 `(0,1)`，字节取到任意值（实测 `q=-9.9e-31 → 0x7E` 解码 +448，参考 `0x80`）。改为**与幅值无关的比较式** `128·[q<|q|]`（⟺`128·[q<0]`，落为 `T.vcmp(q, av, cond, "lt")` + `T.vcast(cond, sbv, "rint")`，i1→f32 得 {0,1}）。缺陷是**非对称且微小幅值**（只在负侧、只在 `(0,1e-30)`），随机测试与典型角点系统性漏过。
④**「覆盖缺口下的假 PASS」实证**：v0 的 `verify_equiv.py` E4 角点集缺 `(0,1e-30)` 负值区间，机器判据 4/4 PASS 却漏掉真实编码错误；补入微小负值/次正规/`±0`/`±inf`/`NaN` 后 E5 **直接判定「v0 有限乘数式 12/13 字节错、v1 比较式 0/1 错」**——旧式回归守卫（E5 形态）是覆盖缺口的判别力自证。复检侧进一步由检视方**独立重生成** ~3.1M lane 验证集（设计脚本之外）才确认缺口闭合。
⑤**组合体编译 UB 溢出（设计期估算系统性偏低）**：DESIGN §4.5 手工预算 ×1.12 ≈ 165724B，但 `BLOCK_ELEMS=4096` 组合体实测编译器要求 **253952B > 192KB**（auto multi-buffer 使多 buffer 翻倍，实际 ≈ 手工估计的 ~1.5×）——设计期「逐 op COMPILE_OK」不蕴含组合体可编译（§9.1-R1 高风险项命中）；降至 `BLOCK_ELEMS=2048` 后通过。
⑥**集成胶水层签名差异**：Stage 3 kernel 采用**显式输出签名**（`(X(R,D), Scale(R,), OutU8(R,D))`，无 `out_idx`——因 uint8 out_idx 静默回落 float32）；`integrate_kernel.py` 只改写 wrapper 的 import 切换块、**不校验调用签名**，改写后的 wrapper 首调即 `ValueError: Expected 3 inputs, got 1`；修法 = 只改胶水层 `_fp8_quant_wrapped_kernel`（host 分配 `scale`/`out_u8`、经 `reshape + .view` 还原对外契约），未改 kernel/dtype/config/测试/manifest。集成脚本的 import 冒烟（`hasattr`）**不覆盖调用签名**。

**实测**（Session 墙钟 8054.4s ≈ 134.2 min，含 ~21.4 min 用户等待；op 级 elapsed 6674s / dispatches 2；函数级 elapsed 5571s——S1 attempts 3〔1492+603+73s，1 fail runtime = S1-PLACEHOLDER 机械误报〕、S2 attempts 2〔272s fail design_revision + 521s complete〕、S3 attempts 1〔2602s，首轮 `[PRECISION_PASS]` 19/19〕）：Stage 3 内嵌分层门禁全过（L0/L1/L2/Boundary 19/19），fp8 字节级判据「契约域内不一致 ⊆ {精确 `-0.0`}」。TileOPs 单算子 report `status=passed`：**正确性 8/8**（含 fp32/fp16/bf16）、benchmark **3/3** msprof 有效（`invalid_profile_count=0`）——kv-index-8k-d64-fp16 **238.42µs**（roofline 1.619%）/ bf16 238.62 / 4k-d128-fp32 238.44（1.569%）；**带宽均 1.75 TB/s**（≈ fabric 聚合上限，算子带宽受限）。设计期 `verify_equiv.py` 5/5 `EQUIV_PASS`（E4：契约域 397568 样本不一致 7 个全为精确 `-0.0`、非预期不一致 0、NaN 16 lane 属契约域外）。

**适用触发条件**：低比特/字节载体（fp8、int8、自定义编码）算子迁移或设计——对外 dtype 契约保持的「软件编码 + 载体 + `.view`」绕过模式；软件编码的舍入/符号/幂精确性三陷阱核对；「与参考逐字节一致」类断言的角点集全定义域覆盖 + 旧式回归守卫；设计期组合体编译探针（勿以逐 op COMPILE_OK 代替）；TileOPs 显式输出 kernel 的 wrapper 胶水适配。

**Stage 4 续档（2026-10-15，task fp8_quant-_fp8_quant_kernel optimize；工具链 tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 + CANN 9.0.0 + Ascend910B3，AIV=40；`perf_opt/opt_log.md` / `perf_records.jsonl` / `perf_feedback.md`）**：Stage 3 交付的 238.5 µs 是**纯标量化伪影**，与 shape/带宽无关（fp16/bf16/fp32 三 workload 恒 238.4–238.6 µs，而 fp32 流量多 64%）。

- **根因（减法探针定位）**：baseline `aiv_scalar_ratio = 0.966`（scalar 228.5 µs / vec 8.0 µs）。逐段 bisect（`probes/probe_bisect.py`）把跳变锁在 `c2→c3`（8.2 → 305.5 µs）；再单点隔离（`probes/probe_c3.py`）证明**元凶是单条 `T.vshl`**——`T.vshl(one_i (32,64) int32, qi, pw)` 被标量化，单独即 305 µs（≈20 ns/元素）。同时证伪了「2D `T.Parallel(block_m, dim)` 整体标量化」的假设：plain f32 2D 元素级 loop 正常向量化（`loop2d_f32` = 8.36 µs，与 base 同档），`T.vshl` 才是唯一瓶颈。
- **修复**：精确幂构造 `(2^9)/2^(ec+6)`（`vshl`）→ **向量化 `T.vsub(c3, ec_f, pwf)` + `T.vexp2(pwf, lv, tmp)` + `T.vmul(av, lv, num)`**（`2^(3-ec) = exp2(3-ec)`）。代价 = `vexp2` 非精确 → 精确 `n.5` tie 翻转，但落在 `_classify_mismatches` 的 `tie_1lsb` 接受类（`unexpected = 0`、cos ≈ 1.000，`--level all` 全过）。
- **第二收益**：`BLOCK_ELEMS` 2048 → **4096**（`block_m` 32→64，serial task 数减半、MTE2 跨度 2.16→1.44 µs），另 −5~6%。
- **实测**：238.485 / 238.505 / 238.265 → **11.26 / 11.30 / 10.84 µs（21.2× / 21.1× / 22.0×）**；Ratio 1.62/1.62/1.57% → **34.15/34.03/34.34%**（`total_fops` 实测保持 ≈17.83 M，**机制 B 未发生**——分子未随 op 削减而下移）。`aiv_vec_ratio` 0.034 → 0.86（转向量受限），`scalar_ratio` 0.966 → 0.10。
- **否决方向（均有实测）**：numpy-free 融合（把 cast+abs、量化 mul+clamp、byte+store 合并成大 2D loop）**无增益**（−2.4%，且 `reduce_mode="absmax"` 形态实测静默错值 `unexpected=262320`）；reciprocal-mul 替代 2D `vdiv` −0.7%；`T.vselect` 构造符号位**静默错值**（`unexpected=262320`）；`BLOCK_ELEMS=8192` 编译失败（UB）。成本为**元素吞吐受限**：4 次 ablation 给出 pass 单价 ~0.23–0.3 µs（`reduce` 1.06 / `vlog2` 0.52 / `vexp2` 0.34 / 2D `vdiv` 0.22 µs）。
- **口径不可达（`[DESIGN_LIMIT]`，见 `perf_opt/perf_feedback.md`）**：平台空 kernel `Task Duration` 地板 **3.37 µs**（固定开销 ≈2.4 µs）+ 必需 GM 流量后 **4.25 µs**（占 5.5 µs 目标预算 77%）；软件 e4m3 编码器 ~7 µs（≥14 pass）→ 70% Ratio（≤5.52 µs）需编码器 ≤1.27 µs，DQ 结构性不可达；§12.5 回落线（≤1.82/2.99 µs）甚至低于空 kernel 地板。结构性出路 = 补齐 CG-2026-0018 后回退硬件 fp8 窄化（预期 Task ≈4.8 µs / Ratio ≈80%）。
- **durable 载体**：`T.vshl` 标量化代价并入 traps-compiler.md `TRAP-vexp2-not-exact-pow2` 代价边注（+ repro `TRAP-vexp2-not-exact-pow2.py`；标量化代价侧的 provenance 为任务工作区 `perf_opt/probes/probe_c3.py`，允许失效——原文引用的 `TRAP-vshl-ub-scalarize-cost.py` 未落盘，2026-10-15 终态蒸馏 D2 机械核验纠正）；msprof op Task 固定地板/pass 单价新条 constants.md `CONST-msprof-task-floor-910B3`（+ repro）。



---

### CASE-pool-adaptivemaxpool2dindices-migration — Stage 4 续档（2026-10-15，toolchain 67a0441）

- **任务/产物**：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/perf_opt/`
  （`_adaptive_max_pool2d_with_indices_kernel.py` sha256 `8156c03c…`；`opt_log.md` / `perf_records.jsonl` / `copy_floor_calibration.jsonl`）。
- **结果（`msprof op` Task Duration，median/20）**：`global-1x1` 144.39 → **91.92 µs（-36.3%，1.57x）**；
  `spp-6x6` 85.57 → **79.86 µs（-6.7%）**；`nondiv-7x7` 81.41 → **72.60 µs（-10.8%）**；双输出在 L0/L1 全用例逐位精确
  （`index mismatches 0/104487`）。
- **采纳的改动**：① fp16 `out_H==out_W==1` 新增**扁平整平面**路径（`x: (N,C,1,H*W)` 一次连续 DMA + 单次 `(plane,C)` 转置 +
  沿平面 `(1,C)` 行记录扫描；扫描期**只用 dtype 宽度平面缓冲**，`CH_GLOBAL=104`）；② general 路径删除死代码 `h_flag` 乘积（8→7 op/元素）；
  ③ general 路径**单波最小 CH** 分派（spp 48 / nondiv 32）。
- **证伪/未采纳（避免重复劳动）**：
  - **C4 整平面 argmax（运行期位置表取代 arange）**：正确性可达（逐元素精确，16384/16384）但**被 UB 单价封顶**——
    `ch=8` 才编译通过、`num_local=52` 轮、实测 **262.8 µs**；`ch=16` UB overflow（需 202 304 B > 196 608 B）。见 `CONST-dual-output-whole-plane-reduce-ub`。
  - **二维 `(C,plane)` 行归约（免转置）**：`reduce(dim=1)` 标量化，Task **237.0 µs**（scalar 98.1 µs）。
  - **免转置按列扫描**（`(C,1)` 跨步列切片作向量操作数）：运行期 **core dump**。
  - **bool 切片直送 `vselect`**（8→6 op）：编译失败（`hivm.hir.vsel` 形态限制，见 TRAP-hivm-vsel-cond-slice）。
  - **`(C,plane)→(plane,C)` 平面转置**（`plane=49`）：必被降级为 UB→GM→UB 往返（`UB_to_GM` 104 KB/blk vs 真实写 4.5 KB），
    是 global 残余缺口（91.9 µs 中 ~60 µs）的主因——**本工具链硬上限**。
- **新增纪律**：写侧 **store 源必须是连续 op 链**（`vbrc`/`vcast`/`reduce_*`），禁止跨步 UB gather 与「一侧为 1」的 `transpose` 输出；
  写侧地板标定必须区分「连续 / 跨步」生产者形态（否则产出假地板，见 PL-1.6 更新）。
- **状态**：`plateau` 收束（`CH_GLOBAL` 邻域 96/104/112/128 无一优于 104；剩余结构候选全部 blocked）。
  **不产出 `perf_feedback.md`**（`perf-feedback.md` §1 双条件未同时满足：天花板归因于工具链 lowering 而非算法/设计层）。

---
id: CASE-reduction-countnonzero-migration
kind: case
family: [reduction]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: count_nonzero-_logical_reduce_kernel-20261016T005019Z（函数级）/ count_nonzero-count_nonzero-20261016T004109Z（op 级，Stage 5 集成）
toolchain: tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892 + CANN 9.0.0 + Ascend910B3（AIC=20 / AIV=40 / UB 196608 B）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 / 2026-10-16
repro: none
---

### `examples/count_nonzero/_logical_reduce_kernel/`（任务工作区，未上库；集成副本 `examples/TileOPs/tileops/kernels/reduction/count_nonzero/`，同任务工作区）

reduction 家族**首个 logical_reduce / 计数类** TileOPs 迁移（`CountNonzeroFwdOp`，`op_kind=count_nonzero` ⇒ 逐行 **int64** 计数输出，与共享工厂的兄弟 kind `any`/`all` 的 bool/int8 输出**契约不同**）。零返工：`retry_count=0`、各 `stage_retry_count=0`、Stage 0/1/2/3/5 各 1 attempt、Stage 3 attempt 1 即 `[PRECISION_PASS]`（Stage 4 按 harness 跳过）。

- **语义/契约（设计师第一件事）**：源共享工厂按 `op_kind` 分派**不同归约原语与不同输出 dtype**——错认 kind 契约（把 `count_nonzero` 当 bool 类）会让整条精度验证链失效。host/Op 层语义 = `movedim(dims→last) → contiguous → reshape(M,N)` + 非浮点 dtype（bool/int32/int64/complex）预转 float32；kernel 声明 **raw N**（`_kernel_handles_padding=True`，host 不 pad）。
- **设计主选（元素预算 + 双路径）**：容量判据从「48 KB SMEM 启发式」换成**元素预算**——编译期探针实测「`vcmp(eq) → vcast(i1→f32) → reduce_sum(dim=1)` 指示链」**UB 单价 8.00 B/元素**（与 dtype 无关）⇒ 单 block 硬上限 24576 元素、竞态安全档 `E_cap = 8192`。**路径 R**（`N ≤ E_cap`，全 N 驻留 + `reduce dim=1`）/**路径 C**（`N > E_cap`，`bm=1` 列分块在线累加 + 静态尾块），`_pick_tile_n` 取「整除 N 的 256 倍数优先」。零计数补式 `y = N − Σ[x == 0]`（有序 EQ，NaN 安全）落在 `(bm,)` 微缓冲。
- **陷阱（本任务新增知识条目）**：`@T.prim_func` 内解析期 `if` 定义 buffer → 分支名不逃逸（tail>0 报 `Undefined variable`，tail==0 同源可编译 ⇒ 缺陷随 shape 潜伏）——见 traps-compiler.md `TRAP-primfunc-compile-time-if-buffer-def`（含知识域 repro）；计数类合法判决形态见 `TRAP-vcmp-ne-ordered-nan` 第二数据点。
- **集成（Stage 5）**：脚手架生成的 wrapper 仍带 GPU 期 `default_config`（48 KiB SMEM 启发式 → manifest `(2048,4096)` 取 `bm=8` ⇒ `8×4096 = 32768 > 24576` 元素 ⇒ 编译期 `ub overflow`，且无法表达 `N > E_cap` 的列分块路径）——wrapper 胶水改为委托 kernel 的 tiling 单源 `_select_config(M, N, dtype)["block_m"]`（kernel/tests/manifest 零改动；同族缺陷档案见 `CASE-argmax-argreduce-integration`）。harness 集成/report/kernel main 的运行前置 = `PYTHONPATH=<tilelang repo root>`（本仓 `tilelang` 非 pip 安装）。
- **实测**：函数级 L0/L1/L2/Boundary 全过（`torch.equal`、int64 精确）；集成 report `status=passed`——正确性 **37/37**、benchmark **7/7**（4 例 msprof 有效）：`sparsity-hidden [2048,4096]` fp16 22.7405 µs（roofline 42.16%）/ bf16 29.2006 µs、`sparsity-seq [32,16384]` fp16 4.5001 µs（路径 C）、`3d-multidim-reduce [4,128,4096]` fp16 9.6602 µs；带宽口径 1.75 TB/s。
- **边界声明（迁移范围）**：源的 tiled 变体 `_logical_reduce_kernel_tiled` **明确不在迁移范围**（用户约束），但其**能力动机**（大 N 可处理）由 NPU 自身 UB 元素预算驱动的列分块承接——源工厂不迁移 ≠ 其能力动机不迁移。
- **durable 载体**：两条 trap 条目（`TRAP-primfunc-compile-time-if-buffer-def` 含 repro / `TRAP-vcmp-ne-ordered-nan` 计数类延伸）；元素预算单价与适用域限定见 constants.md `CONST-reduce-ub-per-element`（追加数据点入队 VP-2026-0263）；行归约「布尔指示平面→补式」形态集入队 VP-2026-0265。本任务 `RETROSPECTIVE.md`（op 级/函数级）/ `SESSION_TIMING_ANALYSIS.md` / 集成侧 `integration_log.md` 为 provenance（任务工作区，未上库，允许失效）。

---
id: CASE-reduction-all-migration
kind: case
family: [reduction]
mode: [developer]
dtype: [bool, int8, fp16, bf16, fp32, int32, int64, complex64, complex128]
status: verified
origin_task: all-_logical_reduce_kernel-20261016T024920Z（函数级）/ all-all-20261016T023951Z（op 级，Stage 5 集成）
toolchain: tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa（dev root build）+ CANN 9.0.0 + Ascend910B3（AIC=20 / AIV=40 / UB 196608 B）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 / 2026-10-16
repro: none
---

### `examples/all/_logical_reduce_kernel/`（任务工作区，未上库；集成副本 `examples/TileOPs/tileops/kernels/reduction/all/`，同任务工作区）

reduction 家族**第二个 logical_reduce / 谓词 AND 类** TileOPs 迁移（`AllFwdOp`，`op_kind=all` ⇒ 逐行 **bool** 输出、kernel 载体 int8，与同工厂兄弟 kind `count_nonzero` 的 **int64** 契约相反；`op signature 另有 keepdim` 参数，兄弟无）。Stage 0/5 各 1 attempt 首过；函数级有一次 `S2-MIG-DIM0` 机械误报返工与一次 `S3-GOLDEN-TORCH` 门禁修复（均非设计缺陷）。

- **语义/契约（设计师第一件事）**：输出契约取自 **manifest**（`bool`，kernel 载体 int8；兄弟 kind `count_nonzero` 为 int64，两者相反）；`ops/reduction/reduce.py` 里的 `"all"` 只是 **roofline 记账 kind、不是实现入口**。边界语义须以 **torch 实测** 为锚：`±0.0` → `all=False`、NaN 无零行 → `True`、NaN 混零行 → `False`、`N=0`（空规约维）→ `True`。**测试输入的判别性信号** = 约 30% 置零 + 强制角点（`x[0,0]=0`、`x[0,1]=1`、`-0.0`、`NaN`），并显式构造「NaN 无零行（True）」与「NaN 混零行（False）」两用例（`randn` 的零密度 ≈ 0 ⇒ 恒等实现亦通过）。
- **设计主选（补式 + 元素预算双路径）**：`vcmp("eq") → vcast(i1→f32, rint) → reduce_max(dim=1) → (bm,) micro-buffer vsub(1 − max)`——`vcmp("ne")` 不适用于本语义（ordered NE，NaN 恒假 ⇒ 含 NaN 行整行误判 False）；取反不可行（`T.vnot` dtype 表无 bool/int1）⇒ 补式落 `(bm,)` 微缓冲而非全宽平面。容量判据按元素预算（`bm·W ≤ E_cap`，本算子链最坏 dtype bf16 实测 12.00 B/元素 ⇒ `E_cap = 8192`；硬上限 24576）。**路径 R**（`N=8192` 单 tile 驻留）/**路径 C**（`N>E_cap` 列分块 + 补列置 AND 中性 1.0 + 静态宽度尾块），跨 tile 合并 = **`T.vmax(acc, mx, new_acc)` + `T.copy(new_acc, acc)`（no-alias 三件套；原位累加会命中 carried-state 陷阱）**——照抄兄弟计数类的 `vsum` 语义翻错。
- **实现纪律（可复用）**：`bm` 尾块守卫（`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice）；**调用方 `block_m` 越预算时 kernel 侧按元素预算回落**（迁移期 wrapper `default_config` 仍是 GPU 期 SMEM 启发式，`(256,4096)` 给 `bm=8` ⇒ 32768 元素 > 24576；kernel `_resolve_block_m` 回落即不溢出且结果不变，**无需改 wrapper**）；结构性条件一律提到 `@tilelang.jit` 工厂层、按变体返回独立 `prim_func`（`@T.prim_func` 内解析期 `if` 定义 buffer 会 `ParserError`）；IR 形态门禁走 `kernel.mlir_content`（`get_kernel_source()` 在 NPU 类上不可用）。
- **陷阱（本任务关联知识条目）**：`TRAP-npuir-lowered-mlir-access`（本任务新增，含 repro）/ `TRAP-vcmp-ne-ordered-nan` 第三数据点（谓词 AND 类补式方向）/ constants.md `CONST-reduce-ub-per-element` 形态适用域 + dtype 分档 / layout.md `PL-1.25-row-reduce-indicator-complement`。
- **实测**：函数级 L0 27 / L1 25 / L2 / Boundary 全过（`--level all` exit 0，bool 输出 `max_diff=0.00e+00`）；集成 report `status=passed`——正确性 **42/42**（bool/int32/int64/complex64/complex128/非整除 N/尾 M/1D-4D/多 dim/keepdim）、benchmark **6/6**、3 例 msprof 有效（`invalid_profile_count=0`）：`mask-validation-4k [32,4096]` bool dim=-1 2.8601 µs（roofline 10.48%）/ `mask-validation-32k [32,32768]` 6.7201 µs（35.67%）/ `3d-multidim-reduce [4,128,4096]` dim=[0,2] 10.9002 µs（43.98%）；带宽 1.75 TB/s。集成 attempt 1 首跑仅因 `PYTHONPATH` 缺仓库根在 import 冒烟报 `ModuleNotFoundError: No module named 'tilelang'`（环境路径，非代码缺陷，未计 attempt）。
- **门禁交互（非设计缺陷的两类返工，供后续同族任务预判）**：① `S2-MIG-DIM0` 裸子串 `"n/a"` 被源路径 `reduction/all` 命中（词边界修复见 queue VP-2026-0197）；② `S3-GOLDEN-TORCH` 要求 golden **函数体内联 `torch.*` 命名空间调用**——`x.bool().all(dim, keepdim)` 语义等价但被 AST 判失败，改 `torch.all(x, dim=dim, keepdim=keepdim)` 即过（VP-2026-0107 第三数据点；本例未产生精度语义变化，复数真值判据 `real != 0 or imag != 0` 与 Op 层契约一致）。
- **边界声明（迁移范围）**：源的 tiled 变体 `_logical_reduce_kernel_tiled` **明确不在迁移范围**（用户约束）；其「大 N 可处理」的能力动机由 NPU 侧元素预算驱动的列分块路径承接。同工厂兄弟档案见 `CASE-reduction-countnonzero-migration`（契约相反：int64 vs bool）。
