# 已知编译器/解析器陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。运行时/数值/语义类陷阱见 [traps-runtime.md](traps-runtime.md)。
>
> **证伪协议（强制，canonical 见 INDEX.md §2，两者同步演进）**：
>
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。实测教训：曾以非法 3-cycle permutation（`[1,2,0]`）与 Parallel 循环累加形式测出"编译失败"，误判"transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益。
> 2. **一切编译器约束结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

---
id: TRAP-C1-old-compiler-transpose
kind: trap
family: [general]
apis: [T.transpose]
dtype: []
device: 910B2C
status: overturned
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08-27 前旧 build
repro: repro-missing
---

### C1 旧编译器 transpose/C-slice 形态不可用

**已失效**（2026-08-27 tilelang 重编译后推翻；合法形态见 [layout.md](layout.md) PL-1.1/PL-1.2）。

---
id: TRAP-C10-strided-gather
kind: trap
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C10 跨步 gather 向量指令缺失

非 C 轴连续的跨步访问小心——跨步 gather 向量指令缺失（截至 2026-08-28 build 有效）。

---
id: TRAP-C11-parser-vartable
kind: trap
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C11 parser var-table 作用域问题

有效（截至 2026-08-28 build）。

---
id: TRAP-C12-copy-dtype-cast
kind: trap
family: [elementwise]
apis: [T.copy]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: 截至 2026-08-28 build；lerp 证伪 2026-09-07
repro: repro-missing
---

### C12 `T.copy` 静默跨 dtype 转换

有效（截至 2026-08-28 build）；Developer GM→UB lowering 为隐藏 hivm::VCastOp（`docs/Tilelang.language/内存操作/T.copy.md` §3），跨 dtype copy 不减少 vector pass 且舍入不可控——勿用作 cast 融合（lerp_tensor 2026-09-07 文档证伪）。

---
id: TRAP-UB-multibuffer-inflation
kind: trap
family: [elementwise, general, reduction]
apis: [T.Pipelined, T.serial]
dtype: []
device: 910B2C
status: verified
origin_task: mish-optimize-2026-08（首证）/ lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z（第二证）/ ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（第三证：结构依赖精测；2026-09-10 蒸馏溯源归位——原回写误标 145312Z，实际 optimize task_id 以 stage_state.json 为准）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z（第四证首源：基础分配差发现）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第四证重校：×1.10–1.12 4 点标定 + UB diet 绕法）/ fp8_quant-_fp8_quant_kernel-20261014T153942Z（第五数据点：纯 Vector 多 buffer 组合体，手工 ×1.12 估计实测 ~1.5×）
toolchain: 2026-08-24 mish 首证；2026-09-07 lerp_tensor 第二次独立证据；2026-09-10 ada_layer_norm 精测（tilelang 0.1.2+a83118285a + CANN 8.5.0）；2026-09-15 基础分配差（首源 tilelang 0.1.2+28783f45 / 重校 0.1.2+6797758，均 + CANN 8.5.0）；2026-10-14 第五数据点（tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760 + CANN 9.0.0 + Ascend910B3）
repro: repro-missing
---

### auto-multi-buffer UB 膨胀：footprint 与 staging 数量解耦（~20/26 B/elem 平台）+ num_local/静态 extents 依赖

大 block_size 探索须按膨胀后预算评估；削 buffer 不必然解锁（lerp fp32 3-buffer 12288/16384 仍 blocked）。有效（2026-08-24 mish 首证；2026-09-07 lerp_tensor 第二次独立证据：`ub overflow, requires 1966080 bits while 1572864 bits available`）。

**第三证精测（ada_layer_norm，2026-09-10，编译溢出报文逐配置读数）**——三条可操作规律：

1. **膨胀后 footprint 是"结构类平台值"，不随 staging 数量线性增长**：persistent serial（num_local_tasks≥2）激活时，fp16/bf16 中转循环（vcast 链）无论 resident 10 B/elem（3 缓冲）还是 14 B/elem（5 缓冲），multi-buffered 需求同为**精确 20.01 B/elem**；fp32 直连无论 8 B 还是 16 B resident，同为**精确 26.00 B/elem**。⇒ UB 预算 guard 从"resident × 膨胀系数"改为"按结构类查平台值"（transit≈20/21、fp32≈26/27 B/elem）。
2. **需求随 num_local_tasks 变化**：同配置（bm=8×N=1152 transit）nl=3 可编译（20.01 B/elem=184.3KB）、nl=4 溢出（22.00 B/elem=202,784B）——分核参数（num_kernels）会改变 UB 可行域，调 num_kernels 必须连带复验编译。
3. **静态 slice extents 也改变需求**：`M % block_m == 0` 特化（real_m 常量化、静态 extents）使 transit 需求 20.01→22.00 B/elem——静态化优化在 UB 贴限时可能直接翻车。

**第四证/姊妹现象（2026-09-15 两任务实证，VP-2026-0059 两证合入）——基础分配差（multi-buffer 关闭仍在）**：`--enable-auto-multi-buffer=false` 下 BishengIR per-AIV UB 实际需求 ≈ 真实手工预算 × **1.10–1.12**（首证 28783f45：178.2KB 预算实测 ~204.1KB +15%；第二证 6797758 4 点标定：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%）——手工逐 buffer 预算表**不能当可编译证明**，设计期按系数放大且勿漏计小 buffer（第五轮实证：DESIGN §4.5 漏计 ub_cond2 8.2KB）；宽 config 上靶前以编译探针校准（`ub overflow` 报文 requires bits 即精确实测需求）。系数已录入 constants.md CONST-capacity-910B2C；与 CG-2026-0008（能力缺口本体，recurring）互链。绕法实证：UB diet 削减手工预算 ≥12.4KB（N/D staging 合并 −24.6KB + rowmat 删除 −16.3KB）解锁 (64,256) 宽块。**反例边界（2026-09-16 第六轮 r9，a13585dc）**：显式 alloc 结构（全 UB buffer 首维 = half）实测 0.981×manual（actual 低于手工）——系数结构依赖、非单向上偏，该结构 ~12% 余量即足（constants.md CONST-capacity-910B2C / PL-1.12 r9 update）。

机制推测（未实证）：pass 按 buffer 角色/循环形态决定复制哪些实例，非全量 ×2。

**第五数据点（2026-10-14，fp8_quant 软件 e4m3 编码，tilelang 0.1.2+8b16bcc / CANN 9.0.0 / Ascend910B3）——纯 Vector 多 buffer 组合体的设计期手工估计系统性偏低 ~1.5×**：设计 §4.5 逐 buffer 表 14 行复算 = 147968B（fp16 输入）/ 156160B（fp32 最坏），按 ×1.12 得 **165724B / 174899B**（余量 ≈30.2KB / ≈21.2KB，设计判 PASS）；Stage 3 组合体实测 `BLOCK_ELEMS=4096` 编译失败——`ub overflow, requires 253952B while 192KB available`（**≈ 手工估计的 1.53×**），降至 `BLOCK_ELEMS=2048` 后通过。与第四证「基础分配差 ×1.10–1.12」互补：本条是 **auto multi-buffer 默认开启**下的多 buffer 组合体，膨胀来自多数 buffer 的实例复制（该算子 ~18 个向量 op/元素、重算术链、多中间 buffer）。纪律：设计期 ×1.12 校核**不得作为容量 PASS 的唯一依据**；「每 tile 元素数」类常量（如 `BLOCK_ELEMS`）须在设计期标注为 Stage 3 可调参数并给出减半降级路径（本任务 DESIGN 已含，Stage 3 现场据此降级一次）。溯源：`examples/fp8_quant/_fp8_quant_kernel/RETROSPECTIVE.md` Stage 3 Skill Flow Issues + `SESSION_TIMING_ANALYSIS.md` §3。
溯源：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md`（Iteration 2/3 探针记录 + probe_ub_v2.py）；第四证溯源：首证 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` Round 2 + 第二证 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/opt_log.md` Phase 1 编译探针（probe_ub.py，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z）。

---
id: TRAP-vbrc-scalar-shared
kind: trap
family: [expert]
apis: [T.vbrc, T.if_then_else]
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### vbrc 标量→shared 不可用；serial 变量条件 segfault

vbrc 标量→shared 不可用；serial 变量条件 if_then_else select segfault（有效，截至 2026-08-28 build）。

---
id: TRAP-threads-kwarg-noop
kind: trap
family: [general, migration]
apis: [T.Kernel]
dtype: []
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: tilelang-mlir-dev dev root build 2026-09-07 源码通读
repro: repro/TRAP-threads-kwarg-noop.py
---

### `T.Kernel(..., threads=)` 在 npuir 无效果

`src/ir.cc` L175–203 KernelLaunch NPU 分支仅消费 grid_size、block_size 不生成任何 threadIdx 绑定（Python 签名仍接受并透传该 kwarg）；GPU 源码迁移遇 `threads=` 直接移除即可，threads/npt 折叠为单一 block_size 由 wrapper 参数传入（复现：`python repro/TRAP-threads-kwarg-noop.py`——机械核对 src/ir.cc NPU 分支无 threadIdx 绑定）。有效（tilelang-mlir-dev dev root build 2026-09-07 源码通读；origin provenance：lerp_tensor 任务 REVIEW.md 附#28——任务工作区，未上库，允许失效）。

---
id: TRAP-tvm-parser-rules
kind: trap
family: [expert]
apis: [T.prim_func, T.rs, T.alloc]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；developer 谱系 v3_deadbuf 跨工具链二次一致（tilelang 67db6f3 + CANN 26.0.rc1）
repro: repro/TRAP-tvm-parser-rules.py
---

### TVM script 解析器四条硬规则

违反报 Undefined variable / trace error：`if` 语句一律生成 TIR runtime if（不折叠、赋值作用域困在 if 体内）→ trace-time 选择用标量三元式或 Python 层预计算；三元式**两臂都会求值**（buffer 切片臂不能用哨兵 tensor → 无条件全尺寸 alloc）；条件 `T.alloc` 非法（哨兵只可用于 shape 三元式且该 buffer 不得在运行 if 体内引用）；`with T.rs()` 块内局部量块外不可见（标量常量提到 Scope 顶层）。条件 alloc 规则与 developer 谱系 v3_deadbuf 教训跨工具链二次一致。复现：`python /tmp/opencode/probe_forms.py on`（四形态逐一验证，session-local；镜像于 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md` D4）。

**〔第六轮补充实证，2026-09-16，tilelang a13585dc + CANN 8.5.0，origin_task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T041000Z〕**精确折叠边界修正（上文「一律 runtime if」过粗）：**纯名 Python bool 的无-else `if X:` 在 trace 时折叠**（第五/六轮 `if band_free:` / `if use_pipe2:` 均产出单 trace，实证两次）；**else 分支的存在破坏折叠或作用域**——`if use_pipe2: slot = task_id % 2 else: slot = 0` 报 Undefined variable（即使条件是纯名常量），绕法 = factory 层 Python int 常量 + 单算术表达式 `slot = (task_id % 2) * pipe2_on`（r7f Vec slot 报错→算术消解通过；r8g 生成脚本同型失败三证）。runtime 条件 `if task_id + 1 < N:` 内的 T.serial/rs/变量定义合法（r7f Cube 段编译通过）。

---
id: TRAP-expert-v-operands
kind: trap
family: [expert, attention]
apis: [T.vcmp, T.vbrc, T.vtanh]
dtype: [fp32, int16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro-missing
---

### Expert v 算子操作数规则

`T.vcmp` 标量操作数拒绝 `tir.Cast`（codegen `NpuirOperand::FromExpr cannot handle the expr with type of "tir.Cast"`）；**〔2026-09-15 第五轮实测补充，工具链 6797758〕vcmp 在 int16 上对所有操作数形态（运行时标量 PrimExpr / 文档合法的同 shape int16 张量）一律降级为标量指令执行**（编译警告 `will execute by scalar instruction with low efficiency`，[32,256] 实测 **236µs/链**）——在 causal mask 链上占整 kernel 壁钟 ~45%（aiv_scalar 94–97% 的主成分，与 flag 自旋同形，须查编译警告区分）。**绕法 = 算术惩罚掩码**：`scr = clamp(tpl - thr, 0, 1)`（vmax/vmin 标量广播，文档合法）→ `S += scr·(-1e38)`（f32 大数吸收，masked 位精确变 -1e38，与 vselect 假支逐位一致；NaN 无法被算术掩埋——stale 列/band 垃圾须 vselect 选择语义或源头零初始化）。标量 PrimExpr B 操作数合法有先例（`examples/deepseek_v4/example_sparse_attn_kernel_highperf.py` L353）；`T.vbrc` 标量源须 let 绑定局部量（字面量触发 rank 断言）；`T.vtanh` 需 fp32 同 dtype 操作数（int16 scratch 不可复用，MLIR verify 报错）。**v-op 切片仅接受静态边界**（运行时 Var 触发 parser `int(Var)` 拒绝，P10 探针）；**脚本解析器只折叠纯名常量 if**（`not X`/`X and Y` 表达式生成 TIR runtime if，须预组合纯名）；**条件 T.alloc 永不折叠**（含常量条件，v3_deadbuf 的强化实证——用宽度切换 `w = X if cond else 16` 替代）。证据：debug_log.md D4/D5 + 本轮任务工作区 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/`（溯源：probe_vcmp_forms.py P6 / probe_forms.py P1-P5 / probe_rt_slice2.py P10 / opt_log.md Round 2）；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915（第五轮 Stage 4，工具链 6797758）。

**〔第六轮补充实证，2026-09-16，tilelang a13585dc，origin_task 同上〕**v-op 操作数拒绝**带 runtime 首维索引的多维 buffer 切片**：`T.vlog2(ub_ell_snap[st, 0:half, 0:1], ...)`（snap 为 [2,half,1]，st 是 runtime Var）在 codegen `getBroadcastDim` 报 `buffer_shape0.size() == buffer_shape1.size()` 检查失败（3 维 vs 2 维不折叠）；`T.vmul(ub_m_snap[st, ...], ...)` 同型。绕法：**先 `T.copy` 拷回单槽 buffer 再喂 v-op**（plain `T.copy` 的源/目标接受 runtime 索引切片——`ws_s[kernel_id, slot, ...]` 先例，r8g epi 修复实证）。与既有「v-op 切片仅静态边界」（P10）同族：静态边界 + runtime 首维索引亦不可用。

---
id: TRAP-int32-vcmp-scalarized
kind: trap
family: [pool, pooling, window, argmax, elementwise, general]
apis: [T.vcmp, T.vcast, T.vmax, T.vselect]
dtype: [int32, int16, fp32]
device: 910B4
status: verified
origin_task: max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261017T235029Z（Stage 4 optimize）
toolchain: tilelang 0.1.2+e8b5d51e3d098422c3970f4e714fb5dba1493609（commit e8b5d51 / dev root build）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B4（aicore 20 → AIV 40 / UB 196608 B）/ 2026-10-18
repro: repro/CONST-vector-launch-overhead.py
---

### `T.vcmp` 在 **int32 操作数**上被逐 lane 标量化（16 ns/lane，34× fp32；结果正确但代价灾难）

- **现象（实测单价，910B4 / AIV=40，512 深 depend 链，`msprof op` Task median/20，扣同 shape floor）**：64 lane `(1,64)` 的 int32 `T.vcmp` **1039 ns/op = 16.2 ns/lane**；同 shape fp32 `vcmp`（data-dependent 体）**30.5 ns/op = 0.48 ns/lane** ⇒ **34×**；f32 `vmax` **12.3 ns/op** ⇒ **≈85×**。指纹：`scalar_us` 占该档 Task 的 **98.2%**（525.70 / 535.17 µs）、`vec_us` 仅 12.78 —— 即 op 被降级为逐 lane 标量循环（编译期只有一条 `will execute by scalar instruction with low effiency` 告警，**无量级提示**）。载体同族：int16 `vcmp` 同样全形态标量化（[32,256] 236 µs/链，见本条 TRAP-expert-v-operands 2026-09-15；不同 dtype、同机制）。fp16/fp32 形态正常向量化。
- **后果（真实 kernel，双输出池化：values + int64 indices）**：每逻辑单元 2 条 int32 `vcmp`（`nan_idx >= 0` 选 NaN、`idx32 == BIG` 判空窗口）——sincnet `[32,80,4096]` fp16 baseline **Task 3480.77 µs**，`scalar 2796 µs`（占 `aiv` 的 **81%**），其中约 2400 µs 来自这两条判决；timing-only 消融（删这 2 条，结果不正确仅计时）**3481.69 → 1094.60 µs（−68.6%）**，`scalar` 2796 → 426 µs。
- **合法绕法（本工具链逐位正确）**：索引判决改到 **fp32 域**——`T.vcast(idx_i32, idx_f32, "rint")` + fp32 `T.vcmp`（`|idx| < 2^24` 时 int32→fp32 精确；索引上界远小于该界）。代价 = 2 个 `(BL,CH)` f32 平面（UB **+8 B/输出元素**，须计入容量闭式）换回 ~2400 µs：实测 sincnet **3480.77 → 1128.20 µs（−67.6%）**、ecg `[16,128,2048]` bf16 **1558.15 → 669.38 µs（−57.0%）**，int64 索引 0/5 633 552 失配、`values max_diff = 0.000e+00`。
- **判据**：`scalar` 占比 > 50% 时，**逐 op 按 dtype 核对**（同一 op 不同 dtype 的单价可差一个量级以上），不要只按「op 条数 × 单一单价」估算；整数域判决（`==`/`>=`/`!=` 哨兵与下标合法性）是最容易踩中的形态。同类但**语义**不同的陷阱见 TRAP-vcmp-ne-ordered-nan（比较模式的 NaN 语义）——本条只讲**代价**，不改语义。
- **证明**：`python repro/CONST-vector-launch-overhead.py --op vcmp --dtype int32 --m 1 --ch 64 --reps 512`（配 `--dtype float32` 对照档，msprof 命令见文件头）；相对关系断言见该文件登记块。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/perf_opt/{op_cost_calibration.jsonl（p2_vcmp_int_m1_ch64_r512 / p2_vcmp_gt_m1_ch64_r512）, _probe_decomp_nointcmp.py, opt_log.md#§3.1/§3.2}`；能力缺口 `.agents/evolution/capability-gaps.md#CG-2026-0035`。

---
id: TRAP-transpose-epilogue-poison
kind: trap
family: [attention, expert, general]
apis: [T.transpose]
dtype: [fp16, f16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 + Ascend910B2C，2026-09-09
repro: repro-missing
---

### 活跃源 `T.transpose([N,1]→[1,N])` epilogue 毒化整 kernel（2.6x）

epilogue 位置的 transpose 若其源为活跃生产者（被前序 v-op 写过且结果被消费），整个 kernel 管道重叠崩塌（实测 fa4096 98→255µs 级）；单 op 无罪（vlog2/vmul/vadd/死源 transpose 各自 97µs 级）、4 个平凡 vadd 无罪——**组合阈值效应，pass 层根因未定位（未文档化假设，依据 = 14 点 morph 实测矩阵）**。绕法：目标 GM 张量增维视图（如 [B,H,S,1]，与 [B,H,S] 同内存）+ UB→GM 自然 2D 区域拷贝（`ub[0:real_m,0:1] → dst[·,·,bx:bx+real_m,0]`）+ host 侧 reshape 还原契约形状——零 transpose、零拷贝。**全谱系存量税**：v9 级 per-block kernel 同样中招（226.72→181.46µs，-20%，perf-only 探针）。证据：`perf_opt/opt_log.md` round 11 morph 矩阵（14 点二分）+ perf_feedback.md 修正附录 + `profiles/round11/`、`profiles/final3/` raw；复现：morph 阶梯脚本形态与 14 点实测矩阵镜像于 opt_log round 11 Diagnosis（scratch `/tmp/opencode/ref_morph.py`，session-local），终值对账 `msprof op --kernel-name=_gqa_prefill_fwd_main_mix_aic --launch-count=20 --warm-up=5`（median of 20）跑 bench `--use-default-config`。

---
id: TRAP-UB-dynsubview-dominance
kind: trap
family: [compiler, ub]
apis: [T.copy, T.alloc_ub]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### task 级 UB 行 buffer + 嵌套循环动态偏移 subview → auto-multi-buffer 非支配 IR

- **形态**：UB buffer（如 [1,Q] 因子行）在 task 级装载、在内层循环中以动态偏移切片消费（`T.copy(ub_row[0:1, s0:s0+ts], blk[0:1, 0:ts])`，s0 为循环变量派生）——Q≥128（切片非全宽）时 BiSheng `--enable-auto-multi-buffer=true` 产出非支配 IR：`error: operand #2 does not dominate this use`（位于 task 级 GM→UB copy 行）；Q=64（单切片=全宽）不触发。
- **绕法**：①退回 per-block GM 直载（基线形态，多 ~12 次 tiny GM 读/任务）；②切片偏移静态化（trace 分派）；③尝试关 auto-multi-buffer（未在本任务验证——会全局改变 L1/UB 多缓冲行为，风险自担）。
- 关联：PL-1.9-hardlimits「v-op codegen 拒绝多维 UB 切片操作数」——本条是 copy 路径 + multi-buffer pass 的同族表现。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/logs/round1/v3_hoist_L0.log`（报错全文）；npuir 现场见 opt_log R1 v3 行分析。

---
id: TRAP-parallel2d-bm-ge2-scalarize
kind: trap
family: [reduction, elementwise]
apis: [T.Parallel, T.if_then_else]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z / argmin-_argreduce_kernel-20261015T023310Z（绕法收窄：per-row 展开仅独立缓冲成立）
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C；绕法收窄 tilelang 0.1.2+417dae9 + CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro/repro_parallel2d_bm2_scalarize.py
---

### 2D T.Parallel(bm,N) 融合 if_then_else 在 bm≥2 整体标量化（与操作数形态无关）

`for i,j in T.Parallel(bm,N): cand[i,j]=T.if_then_else(x[i,j]==ext[i,j], T.cast(j,"float32"), BIG)` 在 bm≥2 时 scalar ratio 0.84–0.95（(2,4096) 222µs）；bm=1 同形态向量化正常（tiled (1,5120) 25.9µs）。**操作数变体对照证伪旧归因**：(bm,1) 直接索引 m[i,0]（0.95）、ext_brc 物化 + x_work staging（0.948）——两者同样标量化，触发器是循环形态本身。绕法：bm≥2 用显式向量 op 链（vbrc/vcmp/vselect，r1a 形态 47µs）。**旧绕法「per-row (1,W) 展开循环」已在 2026-10-15 收窄**（argmin Stage 4，tilelang 417dae9 + CANN 9.0.0 + 910B3）：该写法只在「每行**独立**缓冲」时正确——若 per-row 展开写入**共享 (bm,N) 候选缓冲**（`for r in range(bm): for i,j in T.Parallel(1,N): cand[r+i, j] = ...`）则静默错值（(2048,4096) fp16 实测 y0 = −3 vs ref 3211，且时延 12.91–15.25 µs「更快」＝伪收益）；独立 per-row 缓冲形态正确但在本工具链下整体标量化（154.98 µs），无可采纳空间。专条见 **TRAP-scf-for-row-offset-silent-miscompile**。证伪更正留痕：argmax opt_log Iteration 4（session-1 嫌疑 (a)/(b) 均否）；argmin opt_log Iteration 2（per-row 展开形态的正确性证伪 + 证伪更正记录）。

**〔2026-10-15 适用域收窄（第三数据点，origin_task fp8_quant-_fp8_quant_kernel-20261015T064614Z，scenario=optimize Stage 4）〕**上文「触发器是循环形态本身」对**普通逐元素 2D loop 不成立**：`(block_m, dim)`（fp16 例为 `(32,64)`）的 2D `T.Parallel` 循环体内只放单一向量 op / 整型算术 / 纯 cast 时**正常向量化**——在基准态 `c2`（8.23 µs，`vec_ratio` 0.812、`scalar_ratio` 0.155）上单加一条 `loop2d_int`（`qi = ei+6`，2D int32 loop）/ `loop2d_f32` / `loop2d_vadd`（`T.vadd(ei, six_i, qi)`）/ `vcast_i2f`（`T.vcast(ei, f32, rint)`）分别测得 **8.47 / 8.36 / 8.53 / 8.19 µs**（与 base 同档、噪声内，`scalar_ratio` 均 0.15–0.16）。即本条的触发条件含**循环体内 `T.if_then_else` 融合条件选择形态**这一项，`bm≥2` 单独不构成触发（该 kernel 实测 96.6% 标量化的元凶经单点隔离为单条 `T.vshl`——见 TRAP-vexp2-not-exact-pow2 代价边注——与 2D `T.Parallel` 无关；本条标题的「与操作数形态无关」指**操作数**变体，与本次收窄的**循环体形态**维度不冲突）。**负面侧（plain 2D 逐元素 loop 正常向量化）的独立最小 repro：`repro-missing`（待补）**；既有 `repro/repro_parallel2d_bm2_scalarize.py` 覆盖正向侧。溯源（任务工作区，允许失效）：`perf_opt/opt_log.md` §3 单点隔离表 + `perf_opt/probes/probe_c3.py`；另见 cases.md `CASE-quant-fp8-quant-migration` Stage 4 续档；toolchain tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 + CANN 9.0.0 + Ascend910B3。

---
id: TRAP-vbrc-same-shape-empty-broadcast
kind: trap
family: [general]
apis: [T.vbrc]
dtype: []
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_vbrc_same_shape.py
---

### vbrc 同形 src/dst → 空广播维数组 MLIR verify fail

`T.vbrc(idx_row (1,N), idx_j (1,N))`（src 与 dst 完全同形，无尺寸差维）→ `'hivm.hir.vbrc' op have empty broadcast dims array` 编译期 verify 失败。绕法：同形场景直接用源缓冲（或 T.copy）；vbrc 仅在存在尺寸 1 的差维时使用。

---
id: TRAP-vselect-inplace-carried-state
kind: trap
family: [reduction, general]
apis: [T.vselect, T.copy, T.serial]
dtype: [fp16]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/TRAP-vselect-inplace-carried-state.py
---

### 原位 T.vselect(cond, A, B, B) 在多迭代 T.serial 循环丢失 loop-carried 状态

在线递推（running (max, idx) 类）跨 tile 级联时，`T.vselect(cond, chunk, running, running)`（false 操作数同时是输出）在多迭代 `T.serial` 循环中静默丢失状态——最终输出停留在 tile-0 局部值。触发边界：单迭代（num_full==2）不受影响，num_full ≥ 3（≥2 次更新迭代）才暴露。实测（argmax (4,102400) tn=5120、num_full=20，fp16）：inplace 得 `[4779,3248,2345,1833]`（全为 tile-0 局部索引，真值 `[31919,3248,44964,44710]`）；同 kernel 无别名形态 bit-exact 与 torch.argmax 一致。绕法 = select 到独立 scratch + `T.copy` 回写（多一对 (bm,1) 微缓冲的可忽略代价）。诊断指纹：输出全落 tile-0 局部索引 ⟹ loop-carried 状态未携带。

---
id: TRAP-tir-range-loop-not-unrolled
kind: trap
family: [general, migration]
apis: [T.prim_func, T.serial]
dtype: []
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z / argmin-_argreduce_kernel-20261015T023310Z（第二形态：行偏移静默错值）
toolchain: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13；第二形态 tilelang 0.1.2+417dae9 + CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro/TRAP-tir-range-loop-not-unrolled.py
---

### `@T.prim_func` 内的 `for i in range(n)` 是串行 TIR 循环，不是 Python 编译期展开

- **现象**：`@T.prim_func` 内的 `for o in range(n)` 被前端解析为 **serial TIR 循环**，循环变量是 `tir.Var`（不是 Python int）——trace 期即报 `list indices must be integers or slices, not Var`（对编译期 Python 列表/预计算表的索引、`int()`/`float()` 转换同样失败）；由循环变量构造的 slice 偏移保持为运行时表达式。实测（adaptive_avg_pool2d 迁移 Stage 3 首轮，2026-10-13，417dae99 + CANN 9.0.0 + 910B3）：按 DESIGN §1.6.2 #8/§6.1「Python `range` 编译期展开为向量 op 序列」的字面形态实现（`range` + 预计算 bin 表索引）在 trace 期失败。
- **本工具链可行形态**（与源 GPU kernel 同形）：静态上界 `T.serial(max_extent)` + 运行时 `if k < extent` 守卫 + 边界表达式内联计算（`adaptive_bin(o, ...)`）。repro 实测该形态编译通过且 `max_diff=0`（exit 0 证明 trap 与绕法两侧）。
- **适用**：迁移任务把 GPU 端「Python 预计算表 + `range` 索引」当作编译期展开的形状假设（设计文档声明某 `range` 循环「编译期展开」时，该结论依赖具体前端行为，本工具链下不成立）。与 TRAP-tvm-parser-rules（`if` 折叠/条件 alloc 等解析器规则）同族；前者管条件与作用域，本条管 `range` 循环的 trace 语义。
- **第二形态：静默错值（2026-10-15 补，argmin Stage 4）**——`range` trap 不只有「trace 期抛 `list indices ... not Var`」这一面：当循环变量仅**参与缓冲行偏移**（不索引 Python 列表）时，解析器不报错，循环体照常 lower 为运行时 `scf.for`，但**以该运行时常量为行偏移写共享/片段缓冲会失配向量化与 region 分析 → 静默产出错误数据**（无编译错误、无警告）。故本条与「per-row 展开」写法（TRAP-parallel2d-bm-ge2-scalarize 的旧绕法建议）强相关，专条见 **TRAP-scf-for-row-offset-silent-miscompile**（含 repro 与逐元素精确门的判据）。
- **证明**：`python repro/TRAP-tir-range-loop-not-unrolled.py`（exit 0：`range` trap 复现 + 静态界+守卫形态数值通过）；第二形态证明见 `repro/TRAP-scf-for-row-offset-silent-miscompile.py`。溯源（任务工作区，未上库）：`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/{DESIGN.md,repro/}`；第二形态溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/opt_log.md#Iteration-2`。

---
id: TRAP-vcast-shape-and-gm-source
kind: trap
family: [general, elementwise]
apis: [T.vcast, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z
toolchain: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13
repro: repro/TRAP-vcast-shape-and-gm-source.py
---

### `T.vcast` 的两条 lowering 约束：形状须一致（无 broadcast/reshape）+ 源须为 UB buffer

- **约束一（形状）**：`T.vcast` 的 src/dst 形状须一致（broadcast 兼容），**不承担 reshape**。把 `(out_W, CH)` fp32 累加器按 NCHW 顺序舍回输出 dtype 时，写成 `T.vcast(acc_(out_W,CH)_fp32, acc_t_(CH,out_W)_dtype)` 报 `The input and output shapes do not match for broadcast`；可行路径 = 先分配同形 buffer 完成 `T.vcast`（UB→UB），再以 `T.transpose([1,0])` 换布局。
- **约束二（源位置）**：`T.vcast` 的源**不能是 GM memref region**（`T.vcast(x[...], ub_fp32)`）——该形态降级为 `hivm.hir.vcast` + `tensor.empty` 目标，在 BiShengHIR 阶段报 `NPU IR opt failed: ... Failed to run BiShengHIR pipeline`（与后续 transpose/copy 无关）。可行路径 = 同 dtype `T.copy` GM→UB 之后再做 `T.vcast` UB→UB。
- **伴随行为（同 repro 验证）**：① `T.vmul` 的 `src2` 可以是 `T.Kernel` 内由运行时表达式 let 绑定出的 fp32 标量（文档「标量须在 `T.Kernel` 内、`T.Scope` 外定义」不等于「须为编译期常数」）；② 3D `T.reduce_sum` 链（`dim=2` → `dim=1`）可用，是 `out=1×1` 全局池化特化的原语，无需 `T.reshape`+transpose 链（后者有 mis-compile 风险，PL-1.1）。
- **证明**：`python repro/TRAP-vcast-shape-and-gm-source.py`（exit 0：两条 trap 报文匹配 + 合法形态 `max_diff<1e-3` + 3D reduce 链通过）。文档侧：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.3/§2.4 未覆盖这两条约束。与 TRAP-C12-copy-dtype-cast（`T.copy` 跨 dtype 静默 cast）构成 `T.copy`/`T.vcast` 的互补边界：copy 管搬运（跨 dtype 静默转换），vcast 管转换（形状/源位置受限）。溯源（任务工作区，未上库）：`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/{DESIGN.md,repro/}`。

---

id: TRAP-vcast-same-dtype-rounds
kind: trap
family: [general, elementwise, pooling, migration]
apis: [T.vcast, T.copy]
dtype: [fp32]
device: 910B3
status: verified
origin_task: max_pool2d-_max_pool2d_kernel-20261014T062320Z（第二证，含知识域 repro）/ argmax-_argreduce_kernel-20260924T031005Z 与 argreduce 前身任务 2026-09-23（首证，pyc 考古恢复——见 queue VP-2026-0127）
toolchain: 第二证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；首证 tilelang 0.1.2 dev build + CANN 8.5.0 + Ascend910B2C / 2026-09-23（跨工具链复现）
repro: repro/TRAP-vcast-same-dtype-rounds.py
---

### `T.vcast` 同 dtype（f32→f32）不是搬运，而是退化为「四舍五入到整数」

- **现象**：`T.vcast(src_f32, dst_f32, round_mode="rint")`（同 dtype、同 shape）逐元素给出 `round(src)` 并静默销毁小数部分（-1.1258→-1.0、0.8487→1.0、-0.4339→0.0）——既不是 no-op 也不报错；`round_mode` 操作数在两侧同为 f32 时仍按「整数转换」施加。合法的同 dtype 搬运形态 = `T.copy(src_f32, dst_f32)`（逐位一致）；真正的 dtype 变化（f16/bf16→f32）`T.vcast` 正常工作——陷阱**只**出现在同 dtype 形态，不是 `vcast` 的普遍性质。
- **后果（静默错值）**：凡「fp32 比较/累加域」的 kernel 需要把 fp32 输入行搬进 UB 时，最自然的写法就是 f32→f32 `vcast`，而它会静默量化数据。实测两任务：`_max_pool2d_kernel` C4 整平面分支 8/8 元素错（最大绝对差 0.488、相对差 0.194），换 `T.copy` 后逐位一致；argreduce 族同款 ~0.5 量级 diff（首证，独立任务、跨工具链）。识别信号：fp32 用例整片错、相对误差 ~0.1–0.2 量级且视觉上「像」被量化到整数。
- **合法形态（本工具链实测）**：按 dtype 在 trace 期分派——`is_f32 = (dtype == "float32")`，同 dtype 走 `T.copy`、真 dtype 变化走 `T.vcast`；`_max_pool2d_kernel` 对 fp32 路径三处采用该分派（该分派同时使 fp32 的 UB 预算与 fp16/bf16 不同档——CH 由 32 落到 16，实现形式一改容量结论须重算）。
- **族内位置**：与 TRAP-vcast-shape-and-gm-source（形状须一致 + 源须为 UB）同族互补——本条补第三条约束「同 dtype 不是恒等」；与 TRAP-C12-copy-dtype-cast（`T.copy` 跨 dtype 静默转换）构成镜像对：「copy 换 cast 会静默变值」vs「cast 当 copy 用会静默坏值」。
- **证明**：`python repro/TRAP-vcast-same-dtype-rounds.py`（exit 0：f32→f32 `vcast == round(src)` 且 `!= src` 复现；`T.copy` 逐位一致（control）；f16→f32 `vcast` 正确）。文档侧：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 未说明同 dtype 行为。
- 溯源（任务工作区，未上库，允许失效）：`examples/max_pool2d/_max_pool2d_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-vcast-same-dtype-rounds.py}`；首证见 queue VP-2026-0127（argmax `_argreduce_kernel` 2026-09-24 经 pyc 考古恢复的 Transferable Lessons）。

---
id: TRAP-vcmp-ne-ordered-nan
kind: trap
family: [general, pooling, reduction]
apis: [T.vcmp, T.vcast, T.vmul, T.vselect, T.reduce_min, T.vmax, T.reduce_max]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z（首证）/ count_nonzero-_logical_reduce_kernel-20261016T005019Z（第二数据点：计数类的合法形态延伸 + NPU 运行期实证）/ all-_logical_reduce_kernel-20261016T024920Z（第三数据点：谓词 AND 类补式方向 + 运行期三 dtype 实证）/ any-_logical_reduce_kernel-20261016T062400Z（第四数据点：谓词 OR 类补式方向与误配暴露集 + 空 OR 单位元方向 + 运行期三 dtype 实证）/ count_nonzero-_logical_reduce_kernel-20261018T043104Z（第五数据点：换路径 R′ 后计数类补式仍成立，910B4）
toolchain: tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13；第二数据点 tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892 + CANN 9.0.0 + Ascend910B3 / 2026-10-16；第三数据点 tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16；第四数据点 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16；第五数据点 tilelang 0.1.2+c316b0e6b8a1ddd13a6927b70f38a614e8b0474d（源树 417dae9）+ CANN 9.0.0 + Ascend910B4（AIV=40 / UB 196608 B）/ 2026-10-18（仅计数类补式 + NaN 语义重验）
repro: repro/TRAP-vcmp-ne-ordered-nan.py
---

### `T.vcmp(..., "ne")` 是 ordered 不等——NaN 操作数下恒假，不能用作 NaN 探测

- **现象**：`T.vcmp(x, x, "ne")` 在含 NaN 的行返回全 0（NaN 所在 lane 亦为 0）——即 ordered not-equal（等价 `lt | gt`）语义；IEEE 的 `!=`（NaN 操作数为真）在本工具链不可得。合法枚举 `eq/ne/lt/gt/le/ge` 的 NaN 行为实测：`eq` / `le` / `ge` 与 IEEE 一致（NaN lane 上 `eq` 为 0、`le`/`ge` 为 0，是 IEEE 的正确比较），`ne` 为 ordered-NE（NaN 恒假），`lt`/`gt` 与 IEEE 一致（NaN 恒假）。对不含 NaN 的操作数对 `ne` 行为正常（并非坏 op）——陷阱只在意图为 NaN 探测时触发（6 种 `cmp_mod` × 2 种构造逐模式实测表见 repro 头部 provenance）。
- **后果（静默失效加假阳性通过）**：以 `v != v` 作 NaN 掩码时掩码恒假（标志位永不触发），而端到端 NaN 用例仍可能"通过"——因为 `T.vmax` / `T.reduce_max` 在本工具链会传播 NaN，"窗口内任一 NaN 胜出"被该未文档化行为顺带修对。adaptive_max_pool2d 迁移 Stage 3 首版即此形态（`vcmp(...,"ne")` + `vmax` 累积），L0-10 NaN/±inf 语义门禁首轮假阳性通过；经 5 组最小探针逐层剥离定位，改用 `eq` 取反后语义判定变为确定性路径、全部用例复跑仍全过且 `max_diff=0`。
- **合法绕法（本工具链实测，逐位正确）**：NaN 探测走 IEEE 正确的 `"eq"`（或 `"le"`/`"ge"`）取反——行级 `cond = T.vcmp(x, x, "eq")` → `T.vcast(cond, f32)` 得 {0,1} 掩码 → 逐元素 **in-place `T.vmul(flag, mask, flag)`**（`flag` 初值 1：0 吸收、1 中性）→ 末尾 `vcmp(flag, 0, "eq")` 即 `has_nan`（整平面 C4 路径以 `T.reduce_min` 收尾）。全程只用 IEEE 正确的比较模式，不依赖 `T.vmax`/`T.reduce_max` 的 NaN 行为。文档侧：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3 只列合法枚举，未说明 NaN 行为。
- **证明**：`python repro/TRAP-vcmp-ne-ordered-nan.py`（exit 0：`ne` 自比 NaN 行全 0 复现 + `eq` 为 `isnan` 的 IEEE 补集断言 + eq-mask 乘性累积 `flag==0 ⟺ 列含 NaN` 断言 + NaN-free 操作数对 `ne` 正常）。与 TRAP-expert-v-operands（vcmp 操作数形态与 int16 标量化）互补：前者管操作数形态与执行代价，本条管比较模式的 NaN 语义。溯源（任务工作区，未上库，允许失效）：`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-vcmp-ne-ordered-nan.py}`。
- **〔2026-10-16 第二数据点：计数/谓词类的合法形态〕** 需要「非零 / 非等」判决的**计数类**算子（`count_nonzero` 形态）在本工具链的合法表达式是**有序 EQ 的补式**：`y = N − Σ_j [x_j == 0.0]`——① 数学恒等（行内 `#{x≠0} + #{x=0} = N`）；② `eq` 为 IEEE 正确 ⇒ NaN 不满足 `== 0` ⇒ NaN 计入非零，与 torch `count_nonzero` 及源侧 unordered-NE 判决一致（用 `ne` 判非零会把 NaN 误判为零）；③ 取反路线不可行——`T.vnot` 的 dtype 表不含 `bool/int1`（`docs/Tilelang.language/逻辑操作/T.vnot.md` dtype 表；bool 仅可作片上中间结果、不可 load/store GM，同文档注），故 `[x≠0]` 只能以补式表达；④ **补式的落点在行级微缓冲**（`N − Σ` 在 `(bm,)` 微元上做 1 次 `vsub`），不是在全宽平面做 `1 − ind`（后者 +1 个 4 B/元素平面与 +1 个全宽 pass）。**运行期实证（NPU，非仅文档）**：fp16/bf16/fp32 三路 `vcmp(eq)→vcast(i1→f32, rint)→reduce_sum(dim=1)` 链在 `count_nonzero` L0-12 / L1-4 上逐位精确——`[NaN,0,-0,1,-1]` 行计数 2（NaN 计入非零、±0 不计入）、最小亚正规行（fp16 2^-14 谱系 / fp32 2^-149 谱系）计数与 torch 一致（**亚正规不做 FTZ**）；纯算法侧判别力对照（有序-NE 路线在 NaN 行报出 42 处差异）见 `examples/count_nonzero/_logical_reduce_kernel/verify_equiv.py`（torch CPU，provenance 允许失效）。溯源：`examples/count_nonzero/_logical_reduce_kernel/{DESIGN.md#§1.6.1-E1, _logical_reduce_kernel.py#run_L0 L0-12 / run_L1 L1-4}`（任务工作区，未上库，允许失效）；工具链同 front-matter 第二数据点。
- **〔2026-10-16 第三数据点：逻辑 AND / 谓词类的合法形态——补式方向由目标语义决定，且两向后果不对称〕** 同一「有序 EQ 补式」纪律在 `torch.all` 语义（`all` / 谓词 AND 类）上的落地形态是 `y = 1 − max_j [x_j == 0]`：① **合并算子是 max 而非 sum**——补式表示下「行内命中零」= OR，跨 tile 在线合并须用 `T.vmax`（照抄计数类的 `T.vadd`/`T.vsum` 或源 tiled 的 `T.min(acc, tile_acc)` 都会语义翻错）；② **补式落点仍为 `(bm,)` 微缓冲**（在微元上做 1 次 `vsub`，不在全宽平面做 `1 − ind`）；③ **取反不可行**（`T.vnot` dtype 表无 bool/int1，同第二数据点）；④ **误用 ordered-NE 的后果比计数类更重**——`vcmp("ne")` 对含 NaN 的行把 `all` 判决**整行翻 False**（NaN 在 `all` 语义下应计为非零 ⇒ 不破坏 AND），而计数类只少计 1；⑤ **运行期实证（NPU，非仅文档/编译期）**：fp16 / bf16 / fp32 三路上 `vcmp("eq") → vcast(i1→f32, rint) → reduce_max(dim=1) → (bm,) vsub` 链与 `torch.all` **逐位一致**——11 行构造角点（`got=[F,T,F,F,F,T,T,F,T,T,T]`，其中「NaN 无零行 = True」与「NaN 混零行 = False」两条必须同时成立，任一相反即 ordered-NE 误用）、NaN 整列 × 3 dtype = 全 True、次正规/极值行 `FTTTFT`；⇒ 本机 `vcmp("eq")` 为 IEEE ordered EQ、`±0` 计入零、**亚正规不做 FTZ**；⑥ **路径 C（列分块）合并链同时被证实可用**：`T.vmax(acc, mx, new_acc)` + `T.copy(new_acc, acc)`（no-alias 三件套，勿原位累加）在 `(bm,1)` fragment 上编译通过且逐位精确（见 layout.md PL-1.25）。溯源：`examples/all/_logical_reduce_kernel/{RETROSPECTIVE.md Stage 3, _logical_reduce_kernel.py#run_L0 L0-2a/b/c / run_boundary / run_L1 L1-4, DESIGN.md#1.6.1-A1}`（任务工作区，未上库，允许失效）；工具链同 front-matter 第三数据点。
- **〔2026-10-18 第五数据点：换路径后计数类补式仍成立（910B4 / 新工具链）〕** `count_nonzero` Stage 4 把 tiling 换成「`E_CAP < N ≤ UB` 单 tile 驻留（R′）」，**比较路径故意未改**（仍 `vcmp("eq")` + 补式 `N − Σ[x==0]`）：`--level all` 的 L0-12 NaN 语义（三 dtype：NaN 计入非零、`-0` 不计入）与 L1 角点（非连续输入 / 高阶与末轴 dim / 次正规 / 尾块）全过、`torch.equal` 逐位，集成验收 37/37（`avg max abs error = 0.00e+00`）⇒ **该陷阱只随比较算子的选择触发，不随 tiling / 路径变化**；改结构时可把「比较链不动」当作安全前提，但仍须保留 NaN 角点用例（`randn` 输入的零密度 ≈0，判不出语义错）。工具链见 front-matter 第五数据点。
- **〔2026-10-16 第四数据点：逻辑 OR / 谓词 `any` 类的合法形态——同一 eq 谓词的下游对偶，四项全反转〕** `torch.any` 语义下的合法形态 = `vcmp("eq") → vcast(i1→f32, rint) → reduce_min(dim=1) → (bm,) vsub(1 − min)`：① 与第三数据点（`all`）共用同一谓词 `[x==0]`，但下游四项在 OR/AND 对偶下**全部反转**——归约算子 `reduce_max→reduce_min`、补式方向 `1−max→1−min`、跨 tile 合并算子 `vmax→vmin`、合并累加器单位元 `0.0→1.0`（用 `ne` 直判「存在非零」的语义失败即第三数据点 ④ 的对偶形态，但**暴露集方向相反**：`any` 侧仅「NaN 与零共存且无其他非零」与「all-NaN」两行暴露，其余行被非零「救回」True——详见本文件 `TRAP-any-predicate-reduction-pair`）；② **运行期实证（NPU，非仅文档/编译期）**：fp16 / bf16 / fp32 三路上该链与 `torch.any` **逐位一致**——13 行角点含 `NaN+零`（True）、`all-NaN`（True）、全零（False）、`±0`、亚正规、`±inf`，本任务 `--level all` 记 101 次精确比对 0 差异；③ **空 OR 单位元方向与 `all` 相反**：`N=0`（空归约维）时 `torch.any` 实测为 `False`（`torch.all` 为 `True`）——边界表的单位元方向以 torch 实测为锚（本任务 33/33 torch 边界实测）；④ **取反不可行**（`T.vnot` dtype 表无 bool/int1，同第二数据点）。溯源：`examples/any/_logical_reduce_kernel/{DESIGN.md#§0.1/§9.2/§11, _logical_reduce_kernel.py#run_L0 L0-2a/b/c / run_boundary, REVIEW.md#机械复核}`（任务工作区，未上库，允许失效）；工具链 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba + CANN 9.0.0 + Ascend910B3 / 2026-10-16。

---
id: TRAP-npuir-vbrc-region-dropped
kind: trap
family: [general, pooling, migration]
apis: [T.vbrc, T.copy]
dtype: [int32, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z
toolchain: tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-npuir-vbrc-region-dropped.py
---

### `T.vbrc` 的 destination 为 buffer region 时写入被静默丢弃（UB region 形态）

- **现象**：标量 `T.vbrc(scalar, buf[a:b, 0:ch])`（destination 是带**运行时偏移**的 UB region、且位于 `T.serial` 循环内）在本工具链**静默丢弃**——destination 保持旧值，既不报错也不崩溃；写整 buffer（或写 `(1, ch)` staging buffer + `T.copy(staging, region)`）则逐元素落位（ch = 8 / 32 / 64、int32 / fp32 全测）。
- **int32 形态的额外后果**：region 从未被写入 → 后续读命中**未初始化 UB** → 向量核 fault（acl 507034 "Vector core execution timed out"）——表现为进程级 core dump 而非错值，是本条陷阱最初被命中的形态（首次命中时整文件运行被 core dump 吞掉 stdout，须一 case 一子进程才定性）。
- **标量 operand 形态**：裸 Python 数（`1.0` / `H_in*W_in`）命中前端 rank 断言（`tilelang/language/customize_npuir.py::npuir_brc` 的 `if not isinstance(src, tir.PrimExpr): assert len(src_extent) == len(dst_extent)`，报 "The input vector and output vector must have same rank"）；typed `T.float32(1.0)` / `T.int32(-1)` 通过。
- **后果（静默错值）**：`_adaptive_max_pool2d_with_indices_kernel` 的 first-hit 索引累加器按输出列 seed 运行时值（`ih0*W_in + iw0(ow)`），region 形态下 seed 静默丢失——全 `-inf` 窗口返回索引 0 而不是首个被访问元素的扁平下标；**values 侧不受影响**，故值-only 门禁看不到该损坏。
- **合法等价形态（本工具链实测逐位正确）**：逐 `ow` 填不同标量 = 标量写整个 `(1, ch)` staging buffer → `T.copy(staging, buf[ow:ow+1, 0:ch])` 三件套（probe5/6/7/8 定性，probe9 全链 ch=8/32/64 OK）。
- **族内位置**：与 TRAP-vbrc-scalar-shared（标量→shared 不可用）、TRAP-vbrc-same-shape-empty-broadcast（同形 src/dst → 空广播维 verify fail）、TRAP-expert-v-operands（Expert v 算子标量源须 let 绑定局部量、字面量触发 rank 断言）共同构成 vbrc 落点/操作数形态族——本条管 **UB region destination**（前两条管 shared 与同形）。
- **证明**：`python repro/TRAP-npuir-vbrc-region-dropped.py`（exit 0；region 形态 seed 丢失、staging 形态逐 ch 生效、裸标量被前端拒绝三段断言，ch=8/32/64 全过）。
- 溯源（任务工作区，未上库，允许失效）：`examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-npuir-vbrc-region-dropped.py}`；设计伪代码 `DESIGN.md#3.3` 中 `T.vbrc(ih0*W_in+iw0, max_idx[ow:ow+1,:])` / `T.vbrc(1.0, h_flag)` / `T.vbrc(H_in*W_in, bigk)` 三条形态全部不可用（Stage 3 逐条改写）。

---
id: TRAP-npuir-arange-pattern-with-whole-plane-reduce
kind: trap
family: [general, pooling, reduction]
apis: [T.arange, T.reshape, T.vbrc, T.vselect, T.vcmp, T.reduce_min]
dtype: [int32, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z
toolchain: tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py
---

### `arange` 派生的平面下标 pattern 在「整平面 vselect + reduce」链内被读回损坏（standalone 精确 ≠ in-chain 精确）

- **现象**：`T.arange(1-D) → T.reshape → T.vbrc → T.reshape` 构造 `(CH, H, W)` 平面下标 `k = h*W + w`，**standalone**（pattern + `T.copy` 落 GM）逐位精确；同一 kernel 内再以 `T.vcmp(f32, gbrc, "eq") → T.vselect(eqb, kj, bigk, cand) → T.reduce_min(dim=2) → T.reduce_min(dim=1)` 消费该 pattern 时，pattern 被读回为「w 逐行重复」（H=3, W=4 得 `[0,1,2,3,0,1,2,3,0,1,2,3]`），派生的 first-hit 索引退化为列号。
- **无关变量（全部排除）**：pattern 构造方式（2D 链 / 4D 链 / `vbrc` 直写 3D）、载体 dtype（int32 / fp32）、reshape 方向、是否插入 `T.pipe_barrier("PIPE_ALL")`——损坏稳定复现。
- **后果（值-only 门禁不可见）**：**values 仍精确**，只有 index 静默错——本算子整平面（`out_H == out_W == 1`）argmax 特化路径的 first-hit / last-NaN 索引正由该 pattern 派生，`--level all` 时 values 全对而 14050/16384 条索引错；仅因 int64 索引 oracle 精确（`torch.equal`）才被发现。处置 = `out=1×1` 交回一般路径（`is_identity` 保留、整平面 reduce 链停用，语义不变的回退）。
- **退路形态**：改用与通用路径同构的「行扫描 + 无 pattern」机制；构造 argmax 型索引时对 pattern 做 **standalone 精确 + in-chain 精确**双向断言（单向不足）。
- **证明**：`python repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py`（exit 0；control 精确 + trap 损坏 + 派生 first-hit 与真值不等三段断言，int32 与 fp32 载体各一遍）。
- 溯源（任务工作区，未上库，允许失效）：`examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py}`；关联 CONST-arange-scalar-materialize（`T.arange` 物化代价）与 TRAP-vcast-shape-and-gm-source（同族 reshape/形状约束）。
- **〔2026-10-15 追加：适用域收窄为「pattern 生产者特有」+ 下一道墙；origin_task `adaptive_max_pool2d_with_indices-...-20261015T100705Z`（Stage 4），工具链 tilelang 0.1.2+67a0441 + CANN 9.0.0 + Ascend910B3〕** 被腐蚀的是 **`arange` 派生的 pattern 生产者**，不是「整平面 vselect + reduce」链本身：把下标改由**运行期构建**的位置表产生（`T.serial` + typed `T.vbrc` 标量写 `(plane,)` → `reshape` → `vbrc` 广播成 `(ch,1,plane)`，非 `T.arange`），同一 `vcmp(eq)→vselect→reduce_min/max` 链的 first-hit / last-NaN / 值全部**逐元素精确**（16384/16384，含 NaN / 并列 / ±inf / 全 -inf；`probe_global_reduce.py --ch 8 --check` 报告精确）⇒ 该正确性障碍**可绕过**，原条的「整类路径不可用」结论按此收窄。
- **〔2026-10-15 追加：绕过正确性后的下一道墙〕** 双输出整平面 argmax 的归约链 **UB 单价 ≈ 237 B/平面元素**（含 auto-multi-buffer ×2；对照同族单输出 C4 的 32 B/元素），把 `ch` 封顶在 **8** → `N=8,C=2048,plane=49` 下 `num_local = ceil(2048/40) = 52` 轮 → 实测 **262.8 µs**（远差于该 workload 的一般网格路径 144.4 µs；`ch=16` 报 `ub overflow`）；同时 `(C, plane)` 二维 `reduce(dim=1)`（归约轴 = 最内轴）被标量化（`ch=64`：scalar 98.1 µs、Task 237.0 µs、`UB_to_GM` 96 KB/blk），有效形态是「扁平空间轴」`(C,1,plane)+dim=2`（同 PL-1.23）。常数本体见 `constants.md` `CONST-dual-output-whole-plane-reduce-ub`，能力缺口本体见 `.agents/evolution/capability-gaps.md#CG-2026-0028`。
- 证据（本任务工作区，允许失效）：`perf_opt/probe_global_reduce.py`（`--ch 8 --check` 逐元素精确；`--ch 16` UB overflow）、`perf_opt/probe_global_2d.py`（二维行归约标量化）、`perf_opt/opt_log.md` §3.3 F3/F4/F5 与 §7「已知残留」。

---
id: TRAP-npuir-reshape-dst-buffer-shadow
kind: trap
family: [general, reduction]
apis: [T.reshape]
dtype: []
device: 910B3
status: verified
origin_task: argmin-_argreduce_kernel-20261014T151121Z
toolchain: tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-npuir-reshape-dst-buffer-shadow.py
---

### `T.reshape` 在 npuir 分支被 `npuir_reshape(src, dst)` 覆盖——第二参是目标 buffer，不是 shape list

- **现象**：`tilelang.language` 对 `reshape` 导出两次——`tilelang/language/__init__.py:68` 的通用 `customize.reshape(src, shape_list)` 与 `:196` 的 `npuir_reshape as reshape`（后者生效）。因此**生效的** `T.reshape` 是 `npuir_reshape(src, dst)`：第二参是**目标 buffer**（其形状即目标形状），函数签名实测 `(src, dst)`、`T.reshape.__name__ == 'npuir_reshape'`。按通用形态写的 `T.reshape(A, [N // M, M])`（仓内 `testing/python/language/test_tilelang_language_reshape.py` 演示的写法）在 npuir 上语义不同——把 Python list 当 `dst` 传会走 `_to_region` 失败。属两个同名符号共存的静默语义漂移面。
- **实测来源**：本算子（argmin 迁移，`_argreduce_kernel`）的实现按 npuir 形态书写并通过全部门禁——`T.reshape(first, first_flat)`（(bm,1)→(bm,)）、`T.reshape(idx_src, idx_row)`（(N,)→(1,N)），fp16/bf16/fp32 × resident(bm=1/2/8) × tiled 全部 L0 20 / L1 11 / L2 3 / Boundary 3 用例 PASS。
- **证明**：`python repro/TRAP-npuir-reshape-dst-buffer-shadow.py`（exit 0；三段断言：生效符号名 == `npuir_reshape`、签名参数 == `['src','dst']`、通用形 `customize.reshape(src, shape)` 仍以原名存在——未来工具链恢复通用签名或移除 alias 时断言触发，而非静默改变存量 kernel 代码的含义）。**纯前端符号检查，无设备依赖**（`T.reshape` 语义与是否编译/上机无关）。
- **族内位置**：与 TRAP-vcast-shape-and-gm-source（`T.vcast` 不承担 reshape）、TRAP-npuir-arange-pattern-with-whole-plane-reduce（`T.arange → T.reshape → T.vbrc → T.reshape` 链）同属 reshape/形状约束族。
- 溯源（任务工作区，未上库，允许失效）：`examples/argmin/_argreduce_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-npuir-reshape-dst-buffer-shadow.py, _argreduce_kernel.py:281,309}`；文档面缺口（`docs/Tilelang.language/shape操作/T.reshape.md` 未述 npuir 语义）已在源任务登记。

---
id: TRAP-parallel-f32-uint8-cast-rounding
kind: trap
family: [quantization, elementwise, general]
apis: [T.Parallel, T.cast, T.vcast]
dtype: [float32, uint8, int8]
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（op 级 fp8_quant-fp8_quant-20261014T152721Z）/ all-_logical_reduce_kernel-20261016T024920Z（现象 ④：bool/int8 载体出栈两步链）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14；现象 ④ tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16
repro: repro/TRAP-parallel-f32-uint8-cast-rounding.py
---

### `T.Parallel` 内直接 `T.cast(f32, "uint8")` 非最近舍入；且 uint8 字节实际落在 signed i8 空间

- **现象 ①（舍入）**：`T.Parallel` 体内对**分数** f32 表达式直接 `T.cast(x, "uint8")`（lowering 为 `hivm.hir.vcast` f32→i8）**不是最近舍入**：64 点探针中 16 个值偏离 `round()`，既非 trunc 也非 round-to-nearest-even。**先整数化再窄化**才是可靠形态：`T.vcast(src_f32, tmp_i32, round_mode="rint")` → `T.vcast(tmp_i32, tmp_f32, round_mode="rint")` → `T.cast(tmp_f32, "uint8")`（0/64 与 `round()` 一致）。
- **现象 ②（符号空间）**：codegen 把 `uint8` 降到 **i8**（`memref<...xi8>`）；f32→uint8 直接 cast 对 ≥128 的值**饱和到 127**（= e4m3 NaN 载荷），表现为输出大片 NaN。正确写法是让字节出在 **signed i8 空间**（`bits - 128·[q<0]`，clamp 到 `[-128,127]`）。
- **与前端 warning 的关系**：`src/transform/npu_loop_vectorize.cc:801` 的 warning「Parallel cast lowering uses fixed round mode 'rint'」在 f32→i8/ui8 **窄化**路径上**与实际行为不符**（该 warning 的截断/取整语义登记见 queue VP-2026-0205；本条以 repro 给出「窄化路径非 rint」的反例）。文档面：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 规格表无 `f32|ui8` 行（仅 `f16|ui8`）。
- **现象 ③（窄化源类型，2026-10-15 补充，origin_task fp8_quant-_fp8_quant_kernel-20261015T064614Z optimize Stage 4）**：`T.vcast` 的 **int32→int8** 形态在本工具链下不可用——探针里 `int32 buffer → u8` 的 store 路径两次触发 `vcast int32_t_to_int8_t_rintmode` 报错（探针侧缺陷，非 kernel 缺陷）。据此，写 uint8 字节载体的可靠窄化链存在于 **f32 载体**：`f32 --vcast(rint)--> i32 --vcast(rint)--> f32 --T.cast--> u8`（与现象 ① 的「先整数化再窄化」同一条链）；整型（i32）中间量直接窄化到 i8/u8 字节空间在本工具链无可采纳形态，字节落盘的载体是 f32。溯源（任务工作区，允许失效）：`perf_opt/` 探针侧两次触发记录；toolchain tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 + CANN 9.0.0 + Ascend910B3。**该形态的独立最小 repro：`repro-missing`（待补）。**
- **现象 ④（bool / int8 载体出栈的合法两步链，2026-10-16 补充，origin_task all-_logical_reduce_kernel-20261016T024920Z Stage 3/1）**：谓词类算子的 bool 载体（kernel 侧 int8）**出栈**须走 **`f32 → f16 → i8` 两步** `T.vcast`（两个表项均存在；0.0/1.0 值域下位精确，实测 A5 组 0 偏差、L0/L1 全过）。`T.vcast` 的**规格表无 `f32 → i8` 表项**（与本条现象 ① 的 `f32|ui8` 缺行同源）——**不得**据此误判「无法输出 bool / int8 载体」而改契约（如改用 int16 载体）。溯源（任务工作区，允许失效）：`examples/all/_logical_reduce_kernel/{verify_equiv.py#A5, DESIGN.md#3.2, _logical_reduce_kernel.py#run_L1 L1-5}`；toolchain tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa + CANN 9.0.0 + Ascend910B3。
- **证明**：`python repro/TRAP-parallel-f32-uint8-cast-rounding.py`（exit 0；断言直接 cast 与 `round()` 分歧、两步 rint 逐值一致；现象 ④ 的两步 bool 载体链证据在任务工作区 `verify_equiv.py` A5 组，独立最小 repro `repro-missing`）。
- 族内位置：与 TRAP-vcast-same-dtype-rounds（同 dtype 退化 rint）、TRAP-vcast-shape-and-gm-source（vcast 形状/源约束）同属 npuir cast/舍入族；能力缺口本体见 `.agents/evolution/capability-gaps.md#CG-2026-0020`。溯源（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-parallel-f32-uint8-cast-rounding.py}`。

---
id: TRAP-vexp2-not-exact-pow2
kind: trap
family: [quantization, elementwise, general]
apis: [T.vexp2, T.vshl, T.vcast]
dtype: [float32, int32]
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（op 级 fp8_quant-fp8_quant-20261014T152721Z）；代价边注量化 fp8_quant-_fp8_quant_kernel optimize Stage 4（2026-10-15）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14；代价边注量化 tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 + CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro/TRAP-vexp2-not-exact-pow2.py
---

### `T.vexp2` 不是精确 2 的幂——需要精确 `2^k` 时走 `T.vshl(1, k)`（整数域）

- **现象**：tile 级 `T.vexp2` 实现为 `exp(x·ln2)`，对整数指数返回非精确值（`2^-4 → 0.0624999962747097`，相对误差 ~6e-8，15 个整数指数中 1 个与 `2.0**k` 不等）。用它当 mantissa 缩放因子时，**所有精确 `n.5` tie 被推过错误一侧**（实测 fp16 4/16384、bf16 18/16384 字节错，全部为 `n.5`）。
- **绕法**：精确 `2^k` 走 **`T.vshl(1_int32, k)` → `T.vcast(..., "rint")`**（k = ec+6，实测 k=0..14 逐位精确）再以精确幂除得 `|q|·2^(3-ec)`；编码骨架的字段连续性（`rint` 达 16 时经 `(ec+7)·8+mantissa` 进位）使 ±1 指数误差自校正——但**scale 因子本身必须精确**。
- **证明**：`python repro/TRAP-vexp2-not-exact-pow2.py`（exit 0；断言 ①`vexp2` 路线与精确幂在每探针指数上不等、②`vshl` 路线对 k=0..14 逐位复现 `2.0**k`）。
- **代价边注（2026-10-15 Stage 4 实测量化，origin_task: fp8_quant-_fp8_quant_kernel optimize）**：`T.vshl` 在 UB 上被标量化（编译警告 `Op 'hivm.hir.vshl' will execute by scalar instruction with low effiency`）——**旧结论「每 tile 一次可接受」已被实测推翻**：把 `2^(ec+6)` 的精确整数幂构造（`T.vshl(one_i (32,64) int32, qi, pw)`）放进编码链后，**单单这一条 op 就占满整个 kernel**——fp8_quant 8k/fp16 实测 `Task Duration` 305.5 µs（`aiv_scalar_ratio` 0.979，`aiv_vec_time` 仅 5.3 µs），即 `(32,64)` int32 tile 的 vshl ≈ **20 ns/元素**（对比同一 tile 的普通逐元素 pass ≈ 0.23–0.3 µs / pass）。替换为**向量化的 `T.vsub` + `T.vexp2`**（`2^(3-ec) = exp2(3-ec)`，lowering 为 `VMul`+`VExp`）后整个 kernel 降到 11.3 µs（`aiv_vec_ratio` ~0.86，转为向量受限）——**同一算子 20× 提速的结构性单点**。判据：UB 上出现 `hivm.hir.vshl` 编译警告且 kernel 标量占比 >0.5 ⇒ 立即评估替换为向量化幂/移位替代（`vexp2` 或整数乘 2^k + `vbitcast`），不得按「每 tile 一次」放过。**代价**：`vexp2` 非精确（见上），用于 mantissa 缩放会翻转精确 `n.5` tie 的舍入方向（fp8_quant 中仍落在 `tie_1lsb` 接受类，`unexpected=0`）。repro：`repro/TRAP-vexp2-not-exact-pow2.py`（vshl 精确 vs vexp2 近似的确定性断言；代价侧由本任务 `perf_opt/probes/probe_c3.py` 的 vshl 单点隔离测得，provenance 为任务工作区、允许失效）。
- 族内位置：与 TRAP-vrsqrt-plain-precision（近似超越指令精度）同属「近似超越/幂运算陷阱」族；与 CONST-npuir-f32-division-not-ieee-rounded 同属软件编码的精确性链路。溯源（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-vexp2-not-exact-pow2.py}`。

---
id: TRAP-transpose-ub-gm-roundtrip
kind: trap
family: [pooling, layout, reduction, general]
apis: [T.transpose, T.copy, T.vcast]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（Stage 4）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-transpose-ub-gm-roundtrip.py
---

### UB 内 `T.transpose` 可被降级为 **UB→GM→UB 往返**（代价强依赖 shape/dtype 形态）

- **现象**：全局池化（`N=8,C=2048,H=W=7`，CH=64，40 blocks）里 `T.transpose(flat_(64,7,7) → (7,7,64), [2,1,0])` 使 kernel 从 **5.6 µs → 138.2 µs**；`UB_to_GM_datas` = **48 KB/block**（≈ 整个 tile × 单元数，即转置输出被写回 GM 再读回），`aiv_mte3_time` 43–58 µs。转置 + C 轴累加（`T.vadd`）的完整形态 148–174 µs，而同一 kernel 的向量忙时仅 5–8 µs（**不是向量慢，是访存往返**）。
- **非普适（重要）**：同工具链下 `(64,56)→(56,64)` 的 **2D f16** 转置（两边 ≥8 且为 8 的倍数）仍走 UB 内实现——`spp-6x6` general 路径每单元约 9 次该形态转置，`UB_to_GM_datas` 仅 **8.5 KB/block**、整体 16.3 µs。⇒ 转置是否被降级取决于**被转置维度的大小/对齐/ dtype**，不可用「转置很慢」一概而论，也不可用一次反例否定整类。
- **绕法**：① 避开转置——把「需要 C 内轴」的归约改成**扁平空间轴**（见 layout.md **PL-1.23**，同一 workload 166.3 → 16.7 µs）；② 若必须转置，选「两边 ≥ 向量宽且为向量宽整数倍」的 2D 形态。
- **与 PL-1.1 的关系（证伪留痕，工具链版本戳）**：layout.md `PL-1.1-transpose-chain`（origin_task `AvgPool2dFwdOp-optimize-2026-08`，toolchain 2026-08，`repro: repro-missing`）称「4 次核内融合转置合计 ~5.4 µs」。该结论在 **2026-10 工具链**（本 stamp）上对池化族大 shape 不成立 → PL-1.1 按 INDEX §2.2 降级为**待重验**（保留原文，见其正文标注）。
- **〔2026-10-15 追加，双输出 + 写侧生产者形态；origin_task `adaptive_max_pool2d_with_indices-...-20261015T100705Z`，工具链 tilelang 0.1.2+67a0441 + CANN 9.0.0 + Ascend910B3〕** 往返的**触发条件被进一步定到「写侧生产者形态」**，而非字节数：
  同一读侧（整平面 `ch*plane` fp16 连续 DMA，7 轮）、同一 `(real_c,1,1)` 目标 region 形态下，只改 store 源的生产者——
  | store 源生产者 | 写入内容 | Task(us) | mte3(us) | `UB_to_GM`(KB/blk) | 真实写出(B) |
  |---|---|---:|---:|---:|---:|
  | `T.vbrc` → `T.vcast`（连续） | fp16 值 | 4.73 | 0.82 | 0.8 | 896 |
  | 同上 + `vcast`int32→int64（连续） | fp16 值 + int64 索引 | 4.73 | 0.92 | 0.8 | 4480 |
  | 跨步 UB gather `flat[:,:,0]` | fp16 值 | 94.29 | 45.55 | 48.0 | 896 |
  | 跨步 UB gather `flat[:,:,0]` | int64 索引 | 44.16 | 25.43 | 48.0 | 3584 |
  | 跨步 UB gather（dual） | 值 + 索引 | 109.29 | 71.34 | 96.0 | 4480 |
  ⇒ **`(C,plane)→(plane,C)` 平面转置（`plane=49`，非向量宽整数倍）与跨步列切片的 store 源必中往返；连续 op 链的 store 源不中（且 int64 本身不是问题——p7 的索引+值仅 4.73 µs）。**
  另一独立触发点：**`GM_to_UB` 也被同时拉长**（同粒度纯拷贝 mte2 2.1 µs vs 含平面转置 31.4 µs），即转置同时污染读侧 DMA 的**时延**而不只是字节。
  可复现探针（本任务工作区）：`perf_opt/probe_write2.py` / `probe_write3.py`（写侧 1 变量 A/B）+ `perf_opt/copy_floor_calibration.jsonl`（行 `global_ch64_read_floor_elem` / `global_ch64_dual_contig_store` / `global_ch64_dual_transpose_poison`）。
  **纪律推论**：写侧 floor 标定必须把「生产者形态」当自变量；若标定地板低于 baseline（本轮 dual 标定 95–111 µs 会把 baseline 144.4 µs 判成「已达标」），先怀疑标定假地板。
- 证据：`repro/TRAP-transpose-ub-gm-roundtrip.py`（可运行部分：三形态正确性）；msprof 数字见 repro 头部与 `task: adaptive_avg_pool2d-...-20261014T172114Z` 的 `perf_opt/profiles/probe/round0/{p2_ct_c64,p6_f4,accum_a49_1}`（任务工作区，provenance 允许失效）。

---
id: TRAP-inner-axis-reduce-scalarize
kind: trap
family: [reduction, pooling, general]
apis: [T.reduce_sum]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（Stage 4）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-transpose-ub-gm-roundtrip.py
---

### 最内轴（窄、非向量宽整数倍）的 `T.reduce_sum` 逐元素标量化

- **现象**：在 `(ch,H,W)=(64,7,7)` 上追加**单个** `T.reduce_sum(f32, mid, dim=2)`（归约最内轴 W=7）把 Task Duration 从 **5.6 µs 推到 155.1 µs**（同结构仅差这一个 op，控制变量分离）；代价 ≈ 单元数 × 通道数 × 归约宽 × ~6 ns（标量速率），即 `≈ 元素数 × 6 ns`。归约中轴（`dim=1`，最内宽 =1）同样慢（165 µs）。
- **判据**：`T.reduce_sum` 只有在**归约轴不是最内轴、且最内轴足够宽（≥ 向量宽、宜为向量宽整数倍）**时才向量化；`(H,W,ch)` 形态的 `dim=1`/`dim=0` 归约实测向量忙时仅 8.45 µs（快）。
- **绕法**：① 把归约轴合并成一条**宽的最内轴**（H*W=49）后单次 `reduce_sum(dim=2)`（PL-1.23）；② 或把 C 变最内轴后用 `T.vadd` 累加——但后者需转置，在本工具链上被 GM 往返成本抵消（见 TRAP-transpose-ub-gm-roundtrip）。
- 伴随：该 op 同时触发 TRAP-copy-layout-propagation-dma-amplification。
- 证据：`repro/TRAP-transpose-ub-gm-roundtrip.py`（可运行部分断言最内轴 reduce 数值正确）；msprof 数字见 repro 头部与 `perf_opt/profiles/probe/round0/{bisect_s1,bisect_s2,bisect_s3}`（任务工作区）。

---
id: TRAP-copy-layout-propagation-dma-amplification
kind: trap
family: [general, elementwise, pooling, reduction]
apis: [T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（Stage 4）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-transpose-ub-gm-roundtrip.py
---

### `T.copy` 的 UB 目标一旦被推成非常规布局，GM→UB DMA 退化为 16 B/元素分散读

- **现象**：同一 `T.copy(x[nb, c0:c1, 0:H, 0:W], flat)`，消费者是**逐元素** `T.vcast` 时 `GM_to_UB_datas` = **36.75 KB/block**（真实输入 ≈44 KB/block，连续搬运）；把消费者换成 `T.reduce_sum` 或 `T.transpose` 后同一 copy 变成 **336 KB/block（≈9x 扇出，16 B/元素的 sector 读）**，`aiv_mte2_time` 从 2.2 µs 涨到 110 µs（**指令数不变**，只剩 19 条——是描述符/粒度退化，不是指令膨胀）。
- **机制**：布局推断把 UB 目标 buffer 指派为转置/归约所需的布局，使 UB 地址序与 GM 地址序不再一致 ⇒ 元素级分散访存。
- **绕法**：让 `T.copy` 的目标布局与 GM 自然序一致——把空间轴扁平化（PL-1.23）或显式声明与 GM 连续序同构的 UB 形状；**在 copy 与布局转换之间插一个 UB→UB 恒等 copy 并不能阻断传播**（实测 v3e 仍 146.9 µs）。
- **误判陷阱**：`aiv_gm_to_ub_bw`（如 2.35 GB/s）是**跨度/sector 稀释口径**，不能据此断言「HBM/MTE2 带宽不足」——须与 `GM_to_UB_datas` 的**扇出倍数**一起读，或做同结构减法探针（详见 layout.md PL-1.6 / constants.md CONST-gm-to-ub-bw-dilution）。
- 证据：`repro/TRAP-transpose-ub-gm-roundtrip.py`；msprof 数字见 `perf_opt/profiles/probe/round0/{bisect_s1,bisect_s2,p3_v3e}`（任务工作区）。

---

id: TRAP-scf-for-row-offset-silent-miscompile
kind: trap
family: [reduction, general]
apis: [T.serial, T.Parallel, T.if_then_else]
dtype: [fp16]
device: 910B3
status: verified
origin_task: argmin-_argreduce_kernel-20261015T023310Z
toolchain: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro/TRAP-scf-for-row-offset-silent-miscompile.py
---

### 运行时 `scf.for` 的循环变量作共享缓冲行偏移 → 静默错值（无报错）

- **现象**：`@T.prim_func` 内的 `for r in range(bm)` 是运行时 `scf.for`（见 TRAP-tir-range-loop-not-unrolled）。当 `r` 只参与**缓冲行偏移**（不索引 Python 列表）时解析器不报错，但 `for r in range(bm): for i,j in T.Parallel(1,N): cand[r + i, j] = T.if_then_else(x_ub[r+i,j] == ext_brc[r+i,j], j, BIG)` 这种「per-row 展开写**共享 (bm,N)** 候选缓冲」形态**静默产出错误数据**：(2048,4096) fp16 实测 `y0 = −3` vs `torch.argmin` ref `3211`（无编译错误、无 warning）。独立 per-row 缓冲（每行一个 `(1,N)` 缓冲）与显式向量链两种形态在同一数据上正确。
- **诊断指纹（本条的最可操作处）**：**错值档比正确档更「快」**——错值形态 12.91–15.25 µs vs 正确链式 31.10–31.46 µs、正确独立行缓冲 154.98 µs；凡「per-row 展开 + 加速显著」的分支，先过**逐元素精确门**（`torch.equal` vs CPU 参考）再进 msprof 采集，否则会把失配当收益写进 winner 流程。
- **合法形态（bm≥2 的每元素条件选择）**：显式向量 op 链 `vbrc → vcmp(eq) → vselect → reduce_min`（本机 31.1–31.5 µs，正确）；per-row 展开只对**独立** per-row 缓冲成立，而该形态在本工具链下整体标量化（154.98 µs），无采纳空间。与之并列的 `T.Parallel(bm,N)` 2D 融合形态在 bm≥2 亦标量化（TRAP-parallel2d-bm-ge2-scalarize）——三者构成「bm≥2 每元素条件选择」的完整边界。
- **证明**：`python repro/TRAP-scf-for-row-offset-silent-miscompile.py --level c2_perrow`（打印 MISMATCH，陷阱复现）与 `--level c4_chain`（打印 OK，合法形态通过）；`--level c1_2d` 为标量化对照（268.8 µs）。机制归因：运行时常量行偏移使向量化/region 分析失配 → 生成错误地址或错误 lane 映射；能力缺口本体见 `.agents/evolution/capability-gaps.md#CG-2026-0015`。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/{opt_log.md#Iteration-2, profiles/round1/, logs/round2/*.log}`。


---
id: TRAP-hivm-vsel-cond-slice
kind: trap
family: [general, pooling, reduction]
apis: [T.vselect, T.vcmp]
dtype: [fp16, bf16, fp32, int32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z（Stage 4）
toolchain: tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23（commit 67a0441 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
repro: repro-missing
---

### `hivm.hir.vsel` 的 cond **不接受从更大 UB buffer 切出的 bool 区域**（同 slice 作 `vcmp` 操作数合法）

- **现象**：`row_mask` 声明为 `(w_in, ch)` **bool**、逐元素用 `T.vselect(row_mask[iw0+kw : iw0+kw+1, 0:ch], neg1, fvec, nanc)` 消费时，BiShengHIR 报
  `error: 'hivm.hir.vsel' op operands' shape are inconsistent`（编译失败，非静默）。
  同一 slice 作为 **`vcmp` 的操作数**（`T.vcmp(eq_f32[iw0+kw : iw0+kw+1, 0:ch], zero_e, isn_b, "eq")`）**合法且向量化**。
- **后果（工程性）**：把「行级 `x==x` 掩码」直接送 `vselect` 以省掉每元素一次 `vcmp` 的 op 削减（8→6 op/元素）在本工具链**不可达**；
  可行形态是让专用 `(1,ch)` bool 缓冲由 `vcmp` 产生（每元素 1 条 op），或先 `T.copy` 该 slice 到 `(1,ch)` bool 缓冲（等代价）。
- **无关变量**：`ch` 大小（8..128 均复现）、掩码 dtype（bool）与 `vselect` 的 a/b dtype（int32/fp16 均复现）——
  差异只在**操作数形态**（whole-buffer/专用缓冲 vs 切片）。
- **族内位置**：与 TRAP-vcast-shape-and-gm-source（形状一致约束）、TRAP-vselect-inplace-carried-state（原位 select 丢状态）、
  TRAP-vbrc-region-dropped（region destination）共同构成「v 算子**操作数形态子集**」族——**否定任一 API 前必须穷举 whole-buffer / 切片 / bool-cond / rank 形态**。
- **证明（本任务工作区，未上库，允许失效）**：`perf_opt/_adaptive_max_pool2d_with_indices_kernel_round1_gen.py` 的 G6 分支编译报该错；
  改用 `vcmp` 产出的专用 bool 缓冲后同分支 L0 全过（`perf_opt/logs/` 内有 stderr）。
  溯源：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/perf_opt/opt_log.md` §3.3 F6。

---

id: TRAP-primfunc-compile-time-if-buffer-def
kind: trap
family: [general, reduction, migration]
apis: [T.prim_func, T.alloc_shared, T.alloc_fragment, T.Kernel]
dtype: [fp16]
device: 910B3
status: verified
origin_task: count_nonzero-_logical_reduce_kernel-20261016T005019Z（Stage 3）/ any-_logical_reduce_kernel-20261016T062400Z（Stage 3，第四数据点：适用域限定为「分支内定义 buffer」——纯语句分支合法）
toolchain: tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892 + CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16；第四数据点 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16
repro: repro/TRAP-primfunc-compile-time-if-buffer-def.py
---

### `@T.prim_func` 内解析期（Python 层）`if` 定义 buffer——分支内名字不逃逸，缺陷随 shape 潜伏

- **现象**：在 `@T.prim_func` 体内用 Python 层 `if <静态条件>:` 的分支内 `T.alloc_shared(...)` 定义缓冲、随后在 **sibling 分支**引用该名字时——① 条件为真 → 构建期抛 `ParserError: Undefined variable: <name>`（经 `DiagnosticError` 包裹，异常链文本可匹配）；② 条件为假（该分支体整体被跳过）→ **同源可正常编译**（repro 断言 B，隔离子进程 exit 0）。⇒ 缺陷**潜伏且随 shape 变**：可除尽宽度（无尾块）一路全绿，一旦「可选宽度」非零即编译失败。同一份更大形态（完整守卫尾块体 + 守卫 epilogue 分支）在条件为假时另观察到编译器进程 SIGSEGV（该最小触发未定钉，repro 只断言可靠的 parse-error 签名并如实标注）。
- **量纲/边界**：纯前端解析期行为，与 dtype、tile 规模、设备无关（本任务最小形态 M=8 / N=8192 / fp16 复现）；断言 A/B 为 compile-only，不需要 NPU 设备。
- **绕法（本工具链实测可用）**：把结构性条件提升到 `@tilelang.jit` **工厂层**，按变体返回**独立 prim_func**（一 variant 一 prim_func）——repro 断言 C：两形态均编译通过，且在有 NPU 设备时逐位精确（`torch.equal`，`x[:, ::3] = 1.0` 的行非零计数精确）。本任务三形态 = 路径 R（resident）/ 列分块无尾块 / 列分块带静态尾块。
- **族内位置**：本文件 §C11「parser var-table 作用域问题」（无细节的裸条目）在此获得可复现的具体形态与最小触发。与 `TRAP-tir-range-loop-not-unrolled`（Python `range` 非编译期展开）同属「Python 层直觉 ≠ 前端语义」族——两者的共同事实：Python 层的条件/循环不构成前端可用的结构分派，结构决策的有效表达位置是 `@tilelang.jit` 工厂层。
- **证明**：`python repro/TRAP-primfunc-compile-time-if-buffer-def.py`（exit 0，`ALL ASSERTIONS PASS`；A = tail>0 抛 `Undefined variable` / B = tail==0 干净编译（隔离进程）/ C = 工厂层两 prim_func 变体编译且（设备存在时）精确）。溯源（任务工作区，未上库，允许失效）：`examples/count_nonzero/_logical_reduce_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-primfunc-compile-time-if-buffer-def.py}`；同族决策记录见该任务 `_logical_reduce_kernel.py` docstring（"one dedicated prim_func per variant"）。
- **〔2026-10-16 适用域限定（第四数据点，谓词 OR 类迁移）〕** 本条的适用域是「**分支内定义 buffer**（名字不逃逸到 sibling 分支）」这一具体形态——分支内**只包含语句/调用**时本陷阱不触发：`if mode == "min": T.reduce_min(ind, red, dim=1, clear=True) else: T.reduce_max(...)`、`if complement: T.vsub(one, flat, dst) else: T.vsub(flat, zero, dst)` 在**同一个** `@T.prim_func` 体内**编译运行均正常**（本任务 repro 的 3 个编译期分支变体全过：`eq+min` 正确形态 / `ne+max` 错配 A / `eq+max` 错配 B，均同源编译并在设备上跑出预期差异集）。把本条读成「`prim_func` 内禁用任何解析期 `if`」会催出重复的整 kernel 拷贝；**结构性条件（含 buffer 定义、路径分派、`tail > 0`）仍归 `@tilelang.jit` 工厂层、按变体返回独立 `prim_func`**（第四数据点观测：本任务列分块路径的静态尾块仍以独立变体表达）。溯源：`examples/any/_logical_reduce_kernel/repro/TRAP-any-predicate-reduction-pair.py`（任务工作区，未上库，允许失效）；工具链 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16。

---

id: TRAP-npuir-lowered-mlir-access
kind: trap
family: [general, reduction, migration]
apis: [JITKernel.get_kernel_source, kernel.mlir_content, T.Kernel]
dtype: [fp16, int8]
device: 910B3
status: verified
origin_task: all-_logical_reduce_kernel-20261016T024920Z（Stage 3）
toolchain: tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16
repro: repro/TRAP-npuir-lowered-mlir-access.py
---

### 取降级后 kernel body 的文本通道是 `kernel.mlir_content`；`JITKernel.get_kernel_source()` 在 NPU 类上不是文本通道

- **现象**：编译后的 `target="npuir"` kernel 是 `JitKernel_NPU`（非通用 `JITKernel`）。公开 API `JITKernel.get_kernel_source()`（签名 `(self) -> str`）在该类上**不可用作文本通道**：同一会话内既观察到返回**链接 ELF 的 bytes**（实测 3736 B），也观察到抛 `TypeError: a bytes-like object is required, not 'str'`。降级后的 device function body 可用属性 **`kernel.mlir_content`（`str`**，TileLangIR/HIVM MLIR，含 `hivm.hir.*` op 与 `reduce_dims` 属性）。
- **代价（本任务实测）**：按签名直调 `get_kernel_source()` 的「向量化/IR 形态门禁」首版直接失败，须自查出 `mlir_content` 后重写门禁，白耗一轮实现。
- **适用场景**：凡需要在 kernel 编译后核对**是否真的向量化**（判决链有无退化为逐元素标量循环）的 Stage 3 L0 检查——典型回归通道是 `TRAP-parallel2d-bm-ge2-scalarize`（MLIR 中出现 `arith.cmpf` / `scf.for` / `if_then_else` 即说明判决链退化为标量形态）；grep ELF bytes 或生成的 host source 均取不到该信息。
- **可用断言形态（本 repro 的三段）**：A = `mlir_content` 为非空 `str` 且含本算子输出链 `hivm.hir.vcmp → vcast(i1→f32) → vreduce<max> → vsub → vcast(f32→f16) → vcast(f16→i8)`（含 `<max>` 归约标记与 `reduce_dims = [1]`）；B = 同一文本**不含** `arith.cmpf` / `scf.for` / `if_then_else`（bm = 8 下无标量退化）；C = `get_kernel_source()` 不返回文本源（抛 `TypeError` 或返回 `bytes`，两条分支都视为 PASS——断言只锁「文本源须来自 `mlir_content`」）。
- **边界**：断言 A/B 属**编译期**检查（只证明 IR 形态，不证明数值语义）；语义前提仍须 Stage 3 的运行期用例闭合（同族纪律见 develop skill Phase 4）。
- **证明**：`python repro/TRAP-npuir-lowered-mlir-access.py`（exit 0，`ALL ASSERTIONS HOLD`；需 NPU 环境，纯编译期检查）。溯源（任务工作区，允许失效）：`examples/all/_logical_reduce_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-npuir-lowered-mlir-access.py, _logical_reduce_kernel.py#_run_vectorised_ir_check}`。
- **〔2026-10-16 第二数据点：IR 文本匹配格式〕** `hivm.hir.vreduce` 的 op 名与 `<mode>` 之间**有空格**——lowering 输出形如 `hivm.hir.vreduce <min> ins(...)`（本任务 `any` 迁移 Stage 3 实测原文）。照字面写 `vreduce<min>` 的子串断言在本工具链**必然落空**（假性失败），故匹配时把 op 名与 `<mode>` 分开断言（`"hivm.hir.vreduce <min>"`）；`reduce_dims` 属性用正则（如 `reduce_dims\s*=\s*\[\s*1\b`）容忍 `= [1 : i32]` 之类的属性变体。溯源：`examples/any/_logical_reduce_kernel/{RETROSPECTIVE.md Stage 3 Skill Flow Issues 第二行, repro/TRAP-npuir-ir-gate-loop-scope.py}`（任务工作区，未上库，允许失效）；工具链 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16。

---

id: TRAP-any-predicate-reduction-pair
kind: trap
family: [reduction, migration]
apis: [T.vcmp, T.vcast, T.reduce_min, T.reduce_max, T.vsub]
dtype: [fp32, fp16, bf16]
device: 910B3
status: verified
origin_task: any-_logical_reduce_kernel-20261016T062400Z（Stage 3）
toolchain: tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16
repro: repro/TRAP-any-predicate-reduction-pair.py
---

### 谓词类归约（`any` / 逻辑 OR）的「比较模式 + 归约算子」是一对——两种错配的**暴露集**不同，最短写法最隐蔽

- **现象（NPU 运行期实测，8 行构造角点）**：`torch.any`（逻辑 OR）语义下，`vcmp(eq) → vcast(i1→f32, rint) → reduce_min(dim=1) → (bm,) vsub(1 − min)` 与 `torch.any` **逐位一致**（含 NaN 行）；而两个错配形态的差异集可精确刻画——① **错配 A（最短写法）**：`vcmp("ne") → reduce_max`（无补式，直接读「存在非零」）的差异**恰为**「行内持 NaN 且无其他非零」的行（实测差异集 `[4, 5]`：`NaN + 零` 行与 `all-NaN` 行）；② **错配 B（照抄姊妹 `all` 的方向）**：`vcmp("eq") → reduce_max → 1 − max`（实为「存在零」）的差异集为 `[2, 4, 6, 7]`（判决式非不动点的每行）。
- **机制**：`any` 与 `all` 共用同一谓词 `[x==0]`，但下游四项在 OR/AND 对偶下**全部反转**——归约算子 `reduce_max↔reduce_min`、补式方向 `1−max↔1−min`、跨 tile 合并算子 `vmax↔vmin`、合并累加器单位元 `0.0↔1.0`；照抄任一即语义翻错。错配 A 之所以隐蔽：`vcmp("ne")` 是 ordered-NE（NaN 恒假 ⇒ NaN lane 计为 0），但任何其他非零会把该行「救回」True——只有在「NaN 与零共存且无其他非零」时才分歧，故 `randn`（零密度 ≈ 0）或纯零/非零混合用例集给出**假阳性**；错配 B 则对多数行显性失败。
- **边界/量纲**：判别性输入的最小必检集 = 「NaN 且无其他非零」的行（`NaN + 零`、`all-NaN`）+ 全零行（False，反向）+ `±0` 行；对 `any`，测试输入的置零比例须**高**（约 70% 置零）而 `all` 相反——方向反转，见 `examples/TileOPs/tileops/workloads/reduction.py` 的 `_make_logical_input` 与 `verify_equiv.py`。
- **族内位置**：与 `TRAP-vcmp-ne-ordered-nan` 同源（该条给出 `ne` 的 ordered-NaN 语义与 `eq` 的 IEEE 正确性，本条给出「两向错配的暴露集」与「同一 eq 谓词在 OR/AND 对偶下的四项反转」）；下游落地形态（`any` 的补式 + `T.vmin` no-alias 合并链 + 单位元 1.0）见 layout.md `PL-1.25-row-reduce-indicator-complement`。
- **证明**：`python repro/TRAP-any-predicate-reduction-pair.py`（exit 0，需 NPU：`eq + min` 与 `torch.any` `torch.equal`；`ne + max misrows: [4, 5]` 与 `eq + max misrows: [2, 4, 6, 7]` 两断言锁定暴露集；同 repro 内 3 个编译期 `if` 变体通过亦构成本文件 `TRAP-primfunc-compile-time-if-buffer-def` 的适用域限定证据）。溯源（任务工作区，未上库，允许失效）：`examples/any/_logical_reduce_kernel/{DESIGN.md#§9.1-C-3/§11, _logical_reduce_kernel.py#run_L0 L0-2a/b/c, verify_equiv.py, repro/TRAP-any-predicate-reduction-pair.py}`。

---

id: TRAP-npuir-ir-gate-loop-scope
kind: trap
family: [general, reduction, migration]
apis: [kernel.mlir_content, T.serial, T.reduce_min, T.vmin, T.copy]
dtype: [fp32]
device: 910B3
status: verified
origin_task: any-_logical_reduce_kernel-20261016T062400Z（Stage 3）
toolchain: tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16
repro: repro/TRAP-npuir-ir-gate-loop-scope.py
---

### IR 形态门禁的 `scf.for` / `if_then_else` 禁令**按路径/变体限定**——驻留路径无循环，列分块 tile 循环合法地含 `scf.for`

- **现象（compile-only，两变体对照实测）**：`TRAP-parallel2d-bm-ge2-scalarize` 要求 IR 门禁对 `scf.for` / `if_then_else` 报失败（`T.Parallel` + `if_then_else` 判决退化为逐元素标量循环的形态）。但同一份 kernel 的**列分块变体**（静态 `T.serial(num_full)` tile 循环 + no-alias `vmin` 在线合并）合法 lower 出 `scf.for`——那个循环**就是 tile 循环**，不是被标量化的逐元素循环。实测：驻留路径（整行块驻留 UB、每 task 一次归约）`scf.for=False / if_then_else=False`；列分块路径（`bm=1` + 静态 tile 循环 + `vmin` 合并）`scf.for=True`。把禁令逐字套到列分块变体会对**完全向量化**的 kernel 报**假失败**。
- **正确门禁形态（对所有变体断言）**：`hivm.hir.vreduce <mode>`（**带空格**，见 `TRAP-npuir-lowered-mlir-access`）+ `reduce_dims = [1]` + 无 `arith.cmpf`；`scf.for` / `if_then_else` 的**禁令只对驻留（单 tile）变体**断言。
- **边界/量纲**：纯编译期检查（断言只证明 IR 形态，不证明数值语义）；两路径的数值正确性须运行期闭合（`torch.equal`，见 layout.md PL-1.25）。变体枚举：驻留路径 = `num_full == 1`，列分块路径 = `num_full > 1`。
- **族内位置**：本条是 `TRAP-parallel2d-bm-ge2-scalarize`（标量化判据）的**适用域限定**，两者构成同一 IR 门禁的两面——判据面（什么形态说明退化）+ 作用域面（该判据对哪些变体有效）。
- **证明**：`python repro/TRAP-npuir-ir-gate-loop-scope.py`（exit 0，compile-only、无设备依赖：两变体均断言 `hivm.hir.vcmp/vcast/vsub/vreduce <min>` + `reduce_dims=[1]` + 无 `arith.cmpf`，并断言 `scf.for` 仅驻留路径缺席）。溯源（任务工作区，未上库，允许失效）：`examples/any/_logical_reduce_kernel/{RETROSPECTIVE.md Stage 3 Value Point Proposals 第 4 行, repro/TRAP-npuir-ir-gate-loop-scope.py, _logical_reduce_kernel.py#_run_vectorised_ir_check}`。

---

id: TRAP-parallel-affine-cast-plane
kind: trap
family: [attention, general, migration]
apis: [T.Parallel, T.Cast, T.serial]
dtype: [fp32]
device: 910B3
status: verified
origin_task: multi_head_attention-_gqa_bwd_kernel-20261016T015430Z（Stage 3, first_impl attempt 1）
toolchain: tilelang 0.1.2 dev root build（/workspace/tilelang-mlir-ascend/build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16
repro: repro/TRAP-parallel-affine-cast-plane.py
---

### `T.Parallel` 二维 body 内的 affine 索引 cast 静默丢行项——唯一可靠形态是嵌套 `T.serial` 标量构造

- **现象（NPU 运行期实测，Developer 模式）**：2-D `T.Parallel(bn, bm)` 的 body 写逐操作数 cast 的 affine 平面
  ```python
  for i0, j0 in T.Parallel(bn, bm):
      plane[i0, j0] = T.Cast("float32", j0) - T.Cast("float32", i0)
  ```
  在 `(16, 32)` fp32 平面上**静默丢行项**：最后一行末 6 lane 读回 `j`（`(i=15, j=26) → 26`，正确值 `15 − 26 = −11` 对应项丢失）。既非编译失败也非 NaN，值-only 抽查极易漏过。
- **形态相关（关键澄清）**：把 cast 移到**差**上（`T.Cast("float32", j0 - i0)`）在**隔离探针里精确**，却在同一 repro 的另一上下文里同样出错 ⇒ `T.Parallel` 平面构造的精度是**上下文相关**的，「隔离探针通过」不蕴含「在本 kernel 通过」。本次实测**唯一**在四档几何 × 4 迭代下全 0 错的形态是**嵌套 `T.serial` 标量构造**：
  ```python
  for i0 in T.serial(bn):
      for j0 in T.serial(bm):
          plane[i0, j0] = T.Cast("float32", j0) - T.Cast("float32", i0)
  ```
- **边界/量纲**：`(16,32)` fp32 实测 mismatch **6 行内 lane**（Parallel 形态）；serial 嵌套形态在 4 档几何（含 `(bn,bm)=(64,64)`/`(32,32)`）全 0。代价 = 每任务一次 `bn×bm` 标量 store（causal mask 平面的构造量级；相对后续 5 个 GEMM + 9 个整 tile 向量 op 可忽略）。
- **族内位置**：与 `TRAP-parallel2d-bm-ge2-scalarize`（同 `T.Parallel` 2-D body，但表现为**标量化**而非错值）同族、失效模式不同；与 `TRAP-scf-for-row-offset-silent-miscompile`（循环变量作共享缓冲行偏移 ⇒ 静默错值）并列为「循环变量/affine 索引 → 静默错值」两形态。causal attention 族的 `(j−i)` 距离平面构造是本条的高发落点。
- **证明**：`python repro/TRAP-parallel-affine-cast-plane.py`（exit 0 = 陷阱复现 + 绕法成立：断言 A 逐操作数 cast 形态 mismatch > 0、断言 C 嵌套 serial 形态 mismatch == 0；差式 cast 形态仅打印记录，因其上下文相关）。溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_gqa_bwd_kernel/{_gqa_bwd_kernel.py 平面构造段注释, repro/TRAP-parallel-affine-cast-plane.py}`；同族任务侧证据 = `probe_plane_path` / `probe_mask_final`（(16,32) Parallel 6/24 错 vs serial-nest 四几何 0 错）。

---

id: TRAP-gemm-dynamic-size-cbuf
kind: trap
family: [attention, general, migration]
apis: [T.gemm, T.Pipelined, T.copy, T.clear]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: multi_head_attention-_gqa_bwd_kernel-20261016T015430Z（Stage 3, first_impl attempt 1；尾块 S % 64 != 0）
toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 + Ascend910B3 / 2026-10-16
repro: repro/TRAP-gemm-dynamic-size-cbuf.py
---

### `T.Pipelined` 内的 `T.gemm(size=<runtime extent>)` ⇒ **伪 `cbuf overflow`**（数字无意义），两条件缺一不可

- **现象**：Developer 模式下，`T.gemm` 的 `size=[m, k, n]` 显式覆盖携带**运行时 extent**（如尾块界 `rn = T.min(bn, S − c0)` 且 `S % bn != 0`）**且该 gemm 位于 `T.Pipelined` 循环体内**时，BishengIR pipeline 报
  `cbuf overflow, requires 12009693044328448 bits while 4194304 bits available`
  ——声明 tile 仅 ~100 KB（L1/cbuf = 512 KB）而报文里的需求位数是个无意义的巨大数（≈1.2e16 bit），报文不可作为容量反解依据。
- **触发条件（bisect 实测，两条件缺一不可）**：① `size=` 携带运行时 extent；② 位于 `T.Pipelined` 循环体内。**静态** `size` 位于 Pipelined 内编译通过；**运行时** `size` 位于 Pipelined 外编译通过；`T.copy` 的运行时 extent 不受影响（各形态均可编译）。首轮 smoke 实测 `(1,8,8,100,64)` / `(1,8,8,300,64)` 全失败，`T.Pipelined(0, n_q)`（静态 start）后 tier-1 通过。
- **可用形态（尾块 `S % tile != 0`）**：动态性只留在 **GM↔UB `T.copy` 的 slice** 与 **行原子循环界**（`T.serial(real_n)`）上；**GEMM 全部用全 tile extent**（不传 `size=`），并以**操作数预清零**（任务级 `T.clear(k_s/v_s)` + 每迭代级 `T.clear(q_s/do_s)`）保证 `[real_*, block)` 的 lane 精确为 0 ⇒ 全 tile GEMM 的额外 lane 贡献为 0（须同时保证消费侧对「0 lane」的其它操作数也是有限值：`0×finite = 0`，`0×NaN` 会污染）。实测 tail / causal+tail / GQA+tail 全部 tier-1（`max_diff ≤ 1.4e-3`）；未修正时同 shape 全 `cbuf overflow`。
- **边界/量纲**：报文字节数不可用（伪值）；可用判据是「编译通过/失败」二元 + 尾块精度。Pipelined start 位置也是同类症状源（运行时 start 直接编译失败，症状同样是伪 `cbuf overflow`）。
- **族内位置**：与 `TRAP-primfunc-compile-time-if-buffer-def`（形状相关的潜伏缺陷）同属「尾块/动态 extent 形态」族；L1 容量的正常记账口径见 `constants.md` 的 L1/cbuf 512 KB 行（`cbuf overflow, requires N bits while 4194304 bits available` 曾用于反解 L1 尺寸——本条说明该报文**只在静态 extent 形态**下可作反解依据）。
- **证明**：`python repro/TRAP-gemm-dynamic-size-cbuf.py`（exit 0 = 三态对照成立：A pipelined+runtime size 报 `cbuf overflow`、B pipelined+static size 编译通过、C pipeline 外 runtime size 编译通过）。溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_gqa_bwd_kernel/{_gqa_bwd_kernel.py 尾块段, repro/TRAP-gemm-dynamic-size-cbuf.py}`；同族设计期证据 = DESIGN §3.3 骨架形态 + `fix_tail.py` 六 case。

---

id: TRAP-vec-cube-ub-scratch-role-conflict
kind: trap
family: [attention, mixcv, migration]
apis: [T.gemm, T.vbrc, T.vsub, T.alloc_shared]
dtype: [fp32]
device: 910B3
status: verified
origin_task: multi_head_attention-_gqa_bwd_kernel-20261016T015430Z（Stage 3, first_impl attempt 1；causal 档 dsum 兼作 mask 暂存与 g2 输出）
toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0 + Ascend910B3 / 2026-10-16
repro: repro/TRAP-vec-cube-ub-scratch-role-conflict.py
---

### 同一 UB buffer 兼作 vector 暂存与 GEMM 输出 ⇒ 消费端**静默读到损坏值**（一个 buffer 只能有一个引擎写者角色）

- **现象（NPU 运行期实测，Developer 模式，块内）**：
  ```python
  T.vbrc(7.0, buf)                 # vector 写
  T.gemm(A, B, buf, initC=True)    # Cube 写同一 buffer
  T.vsub(buf, row, out)            # vector 读
  ```
  消费端读到的 `buf` **既非 GEMM 真值也非 GEMM 前的 vector 值**：2048 cell **全错**（`max_diff 1.33e2`），无编译错误、无告警、无 NaN。给 vector 暂存独立的 scratch buffer，或让该 buffer 只由 GEMM 写，即精确。
- **kernel 级后果（同源形态）**：causal 档的掩码惩罚平面与 `dsum`（g2 输出）复用同一 buffer ⇒ `ds = P·(惩罚 − δ)` 读到惩罚值而非 `dO·Vᵀ`，实测 `dk` 最大 **1.96e4**（换独立 `dsd` 缓冲后 **6.7e-4**）。即一个 buffer 角色冲突可把整个 kernel 的某个梯度输出毁掉，而精度门禁若只看总体元素占比会把它记成「量化噪声」。
- **纪律（可复用形态）**：一个 UB buffer 只允许一个**引擎写者角色**（vector 写 xor Cube/GEMM 写）；读侧可跨引擎。causal/掩码类算子最容易踩（掩码惩罚暂存 + GEMM 输出常被写成同一个 tile buffer）。
- **边界/量纲**：`(64,64)` fp32 平面 2048 cell 全错；三态对照 2048 / 0 / 2048 命中（A gemm-only 精确、B vector-then-gemm 全错、C 独立 scratch 精确）。
- **族内位置**：与 `TRAP-vselect-inplace-carried-state`（别名/原位导致静默状态丢失）同属「别名与角色」族；MixCV 的 buffer 角色纪律在 mixcv skill 侧另有纪律条目。
- **证明**：`python repro/TRAP-vec-cube-ub-scratch-role-conflict.py`（exit 0 = 三态对照成立）。溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_gqa_bwd_kernel/{_gqa_bwd_kernel.py 的 dsd 独立缓冲修正, repro/TRAP-vec-cube-ub-scratch-role-conflict.py}`。

---

id: TRAP-reduce-rowsum-store-shape
kind: trap
family: [attention, reduction, migration]
apis: [T.copy, T.reduce_sum, T.transpose]
dtype: [fp32]
device: 910B3
status: verified
origin_task: multi_head_attention-_flashattn_bwd_preprocess_kernel-20261016T015430Z（Stage 3, first_impl attempt 1）
toolchain: tilelang 0.1.2+1bc8dbc3（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16
repro: repro/TRAP-reduce-rowsum-store-shape.py
---

### 行归约结果写 S 连续段须经 1-D UB 中转——region 单位维**只能从左侧折叠**，payload 轴须在两侧尾部对齐

- **现象**：三维行归约输出 `dlta` 形状 `(blk_s, 1, 1)`，直接区域拷贝到 GM 最内轴连续段
  `T.copy(dlta[0:real_s, 0, 0], delta[bz, bx, s0:s0+real_s])`（两侧 region 分别是 `(real_s,1,1)` 与 `(1,1,real_s)`）
  被 npuir codegen 拒绝：`error: 'bufferization.materialize_in_destination' op source/destination shapes are incompatible`。文档 `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 只说明「静态大小为 1 的维度会被自动折叠（rank 可不同）」，未说明**折叠只从左侧进行**、payload 轴必须落在两侧相同的尾部位置。
- **可用形态**：1-D UB 中转 `T.copy(dlta[0:real_s, 0, 0], mid1[0:real_s])` + `T.copy(mid1[0:real_s], delta[bz, bx, s0:s0+real_s])`（代价 ≤ `real_s × 4 B`/task；实测 `max_diff = 3.3e-06`，精确）；单轴 `T.transpose(dlta, mid, permutation=[2,1,0])`（得 `(1,1,blk_s)`）同样正确，但引入 `vtranspose` op。
- **边界/量纲**：本形态是「最内轴连续段写出」的通用约束——同一 kernel 内 `real_s` 须同时约束 GM→UB 装载、UB→GM 写出两侧 slice（三方同源），否则尾块越界/漏写（对照 `TRAP-T-copy-region-semantics` / `TRAP-UB-dst-align`）。
- **族内位置**：行归约/逐行点积族的写出形态；下游同类落点 = `_flashattn_bwd_postprocess_kernel` 的 dq 回写、`_gqa_bwd_kernel` 的行原子写出（后者用逐行 1-D 行原子规避）。
- **证明**：`python repro/TRAP-reduce-rowsum-store-shape.py`（exit 0 = 断言 1 原形态被拒且报错文本含 `materialize_in_destination` + `shapes are incompatible`、断言 2 绕法编译运行且与 torch reference 在 1e-4 内一致；`--check-only` 只做编译）。溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/{_flashattn_bwd_preprocess_kernel.py 模块 docstring「Implementation note」, repro/TRAP-reduce-rowsum-store-shape.py}`；设计侧原形态见该任务 DESIGN §3.3（已由实现替换）。

---

id: TRAP-bishengir-crash-s32
kind: trap
family: [attention, general, migration]
apis: [bishengir-compile, T.Kernel, T.Pipelined, T.gemm, T.atomic_add]
dtype: [fp16, bf16]
device: 910B3
status: verified
origin_task: multi_head_attention-_gqa_bwd_kernel-20261016T015430Z（Stage 3，L2 warn-only 记录，形态①）/ multi_head_attention-MultiHeadAttentionBwdOp-20261016T082329Z（Stage 5 重试段单算子 report benchmark，形态②）
toolchain: tilelang 0.1.2 dev root build + CANN 9.0.0（/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile 1.1.0 / 428ab8fdab46 2026-04-30）+ Ascend910B3 / 2026-10-16
repro: repro/TRAP-bishengir-crash-s32.py
---

### `bishengir-compile` 的**形状/dtype 相关段错误**（CANN 9.0.0）——已知两形态（`S == 32` / `(B=1,H=64,D=128,S=2048,fp16)`），与 kernel 逻辑无关，须与精度告警分离

- **形态①（`S == 32`，Stage 3）**：同一 Developer 骨架（persistent `T.Kernel` + 核内 `T.serial` 任务循环 + `T.Pipelined` query 循环 + 运行时 extent `T.copy` 到预清零操作数缓冲 + `T.gemm` + 逐行 `T.atomic_add`）在 `S = 32` 上 `bishengir-compile` **段错误**：
  `NPU IR opt failed: PLEASE submit a bug report ... Stack dump:`；而 `S ∈ {16, 24, 40, 48, 56, 64, 96}` **全部编译通过**。崩溃与 tile 形状（`(bm,bn) = (64,64)` 或 `(32,32)`）、`heads`、`dim ∈ {64,128}`、`is_causal` 均无关（各变体实测）。
- **形态②（大 shape × fp16，Stage 5 重试段单算子 report，2026-10-16）**：`bench_multi_head_attention.py` 的 `llama-3.1-70b-long-float16` 档（`batch=1, heads=64, dim=128, seq_len=2048, fp16`）在 **fwd 与 bwd 两条 bench 上同时** `bishengir-compile` 段错误（`NPU IR opt failed: PLEASE submit a bug report ... Stack dump without symbol names`，栈顶为 `__kernel_rt_sigreturn`，无 MLIR 诊断）；**同一 shape 的 bf16 档与其余全部 shape 正常** profile + build（该报告 20 条 bench 中仅此 2 例失败，18 条产出记录）。即形态②同样是**形状×dtype 联合取值**触发的后端崩溃，而非 kernel 逻辑或精度问题。**两形态是否同源未证**——无 MLIR 诊断可比对、触发骨架不同（① = Developer persistent + 行原子；② = 同包 fwd Expert / bwd 两条 bench），本条只断言「同一后端版本上存在≥2 个形状/dtype 敏感段错误触发点」。该档的 op 级影响 = benchmark 降级 `partial`（正确性 12/12 不受影响）。
- **影响面**：manifest 档（S 为 64 倍数）不受影响；`S == 32` 这一档不可编译，在分层测试里表现为 L2/Boundary 的一条 WARN；形态②使对应 bench 档在报告里记为 failed 而**不进性能统计**，两形态均**与数值精度无关**。
- **记录纪律**：后端编译器崩溃（`bishengir-compile` 段错误 / `PLEASE submit a bug report`）与精度告警属两类事件，混在同一 WARN 通道会让读者把「工具链对某 shape 崩溃」误读成「本 kernel 数值可疑」。本条的工具链缺陷身份与升级路径见 `.agents/evolution/capability-gaps.md` 的对应条目（CG-2026-0030，两形态同条登记）。
- **重验触发**：若本 repro 报 `S=32: COMPILED`（形态①）或该 bench 档 `(batch1,heads64,dim128,seq2048,fp16): COMPILED`（形态②）说明对应触发点已被工具链升级修复——按证伪协议刷新版本戳并关闭/收窄对应 capability-gap 条目。
- **边界/量纲**：判据为二元（编译通过 / 段错误），无性能量纲。
- **证明**：形态① = `python repro/TRAP-bishengir-crash-s32.py`（三态对照：A `S=32` 抛崩溃、B `S=40` 通过、C `S=24` 通过）。形态② **`repro-missing`**（证据形态 = 单算子 report 的 benchmark 记录，需整包 fwd+bwd + TileOPs 报告链，暂无法最小化）。溯源（任务工作区，未上库，允许失效）：`examples/multi_head_attention/_gqa_bwd_kernel/{repro/TRAP-bishengir-crash-s32.py, history_version 的 d10.py S 扫描 16/24/32/40/48/56/64/96 与 d9.py tile/heads 变体}`；`examples/TileOPs/reports/tileops/20261016_082438_849619_MultiHeadAttentionBwdOp/{run.json,report.md}`（`summary.benchmark_failed=2`、report §Failed Cases 两行）。

---

id: TRAP-window-decimation-primitives
kind: trap
family: [pooling, window, layout, reduction]
apis: [T.gather, T.transpose, T.copy, T.deinterleave, tensor.extract_slice]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 optimize，`probe_dmg` / `probe_pm` 决定性实验）/ max_pool2d-_max_pool2d_kernel-20261017T043853Z（Stage 4 optimize，④ 可用原语 = 2 相 deinterleave 抽取路径）
toolchain: tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3（aicore 20 → 纯 Vector 40 / UB 196608 B）/ 2026-10-16；④ 数据 tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025（commit b3a7b21）+ CANN 9.0.0 + Ascend910B3 / 2026-10-17
repro: repro/TRAP-window-decimation-primitives.py
---

### 窗口类「折叠成宽归约」的三种抽取原语在本工具链实测全部不可用（region 行距不匹配 ⇒ UB→GM→UB 物化**且被标量化**）

- **背景**：窗口/重叠窗类 general 路径的自然瓶颈是「每输出位置 × 每 tap 一条 **CH 宽**的窄向量 op」（sincnet 实测 `2·k·BL = 156` 条窄 op/单元、全 kernel ≈570K 条、`aiv_vec` 占 `aiv_time` 73%）。结构性出路是把 tap 循环折叠成一个「整 span dilate-max（宽 op）+ 抽取（decimation）」结构。本条目记录该出路的**三条原语级实测否决**（同一任务、同一工具链、均与 baseline 302 µs 对比）。
- **① `T.gather`（vgather）是逐元素实现，≈1.8 ns/元素**：`(80,21) ← (80,61)` 两次 gather（每 unit 一次，52 单元）使 `aiv_vec` 增加 **≈320 µs**（320 µs / (2·80·21·52) ≈ **1.8 ns/元素**）——抽取本身比它要省的窄 op 更贵。与 `TRAP-C10-strided-gather`（跨步 gather 向量指令缺失）同一根因的**代价侧**数据。
- **② `tensor.extract_slice` 源/目标**行距不一致**时被 bufferization 降级为 UB→GM→UB 物化，且该物化被标量化**：`(80,61)` 取自 `(80,64)`（结果行 stride ≠ 源 buffer 行 stride）的形态在 sincnet 上使单分支 Task **301.886 → 23328 µs**，其中 `scalar ≈ 22878 µs`（`aiv_vec` 480 µs）——**22328 倍的时延放大**，且编译无任何报错/告警。判据：结构改造必须让**每个 region 操作数与其 buffer 行距一致**（整宽行区间），否则触发该降级。
- **③ 单点列布局改造（位置主序）不足以救回**：把每个切片都做成整宽行区间（位置主序 `(JLEN, ch)` + `T.transpose` 取回）后，`BL=13` 逐位精确、切片全部连续，Task 仍 **998.92 µs**（`aiv_vec` 771 µs）**> baseline 302 µs** ⇒ 该结构族被实测否决（不是「未穷举即否证」：两个独立结构分支各一实测）。
- **④ 可用原语（2026-10-17 追加，窗口池化 `s_w==2` 形态）：`T.deinterleave(channel_nums=2)` 是本族唯一实测可用的抽取路径**——结构 = 「整行滑窗 max（宽 op，行长 = 输入行）+ 偶/奇相位分离」，之后 `k_w` 个 tap 变成对两张相位 tile 的 `vmax`（每 tap 一条 CH 宽宽 op），替代原来的 `k_w·out_w` 条 128 B 窄 op。max_pool2d Stage 4 round-1 实测（910B3 / CANN 9.0.0 / commit b3a7b21，三 dtype 逐位 `max_diff = 0.0`）：resnet-stem `[32,64,112,112]` k3s2p1 1512.06→**230.42 µs（−84.8%）**、alexnet-ceil `[32,64,55,55]` k3s2p0 ceil 217.67→**66.65 µs（−69.4%）**、vgg-block `[16,128,56,56]` k2s2p0 124.55→**55.08 µs（−55.8%）**（fp16；bf16/fp32 同向）。**成立前提**：`s_w == 2`（相位 = 偶/奇）；`k_w` 的每个 tap 只落单一相位（taps 以 `(phase, base)` 表达）；`k_w` 有上限（本实现 `_PHASE_KW_MAX`）；右侧越界的 tap 须由结构守卫 `(out_w−1)·s_w + (k_w−1)·d_w − p_w ≤ w_in − 1` 挡回一般路径（否则末列读相位 tile 陈旧尾部，实测 5–9/960 元素错）；相位 tile 的行宽 `q_pad` 与行数 `pad_rows` 各有对齐/留块规则（见 queue **VP-2026-0306**）。**本条据此收窄**：「折叠成宽归约」不是整类不可行——被否的是 `T.gather` / `T.extract_slice` 行距 / 位置主序三条路径；2 相 deinterleave 在 `s_w==2` 窗口类成立（C 类档案见 `cases.md#CASE-pool-maxpool2d-migration` 的 Stage 4 续档）。
- **逃逸方向（同任务实测）**：不要碰「抽取 → 宽归约」，改从**通道分块形状**入手（`CH = min(C, cap)` 且 cap ≥ C，消除尾块 lane 浪费与 `num_logical` 翻倍 ⇒ sincnet −37.7%）与 **DMA 行长**（见 `CONST-dma-burst-rowstride-floor`）。C4/整平面域（`L_out==1`）不受本陷阱影响，其免抽取路径见 layout.md `PL-1.23-flat-spatial-axis-reduce`。
- **同族条目与缺口**：指令缺失根因见 `TRAP-C10-strided-gather`；物化往返机制族见 `TRAP-transpose-ub-gm-roundtrip`（同一「UB 内 region 操作被降级为经 GM 往返」家族，本条是 **extract_slice 行距** 触发形态且**额外被标量化**）；多相抽取原语见 `TRAP-deinterleave-two-phase-only`。能力缺口登记：`.agents/evolution/capability-gaps.md` **CG-2026-0031**（宽区域跨步抽取无向量原语）/ **CG-2026-0033**（bufferization 物化 + 标量化）。
- **量纲**：① `ns/元素`；② 倍率（Task 放大 77×，其中 `scalar` 占 98%）；③ 绝对时延对比（µs）。
- **证明**：`python repro/TRAP-window-decimation-primitives.py --case sincnet --check`（结构正确性）/ 去掉 `--check` 后按头部 msprof 命令测时延（测量型 repro）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/{probe_dmg.py, probe_pm.py, probe.py, logs/probe_dmg_*.log, logs/probe_pm_sincnetbl.log}`；opt_log 侧 `#§5 结构探针`。

> 〔2026-10-16 蒸馏，D 类 Tier 0 合入（provenance + 版本戳 + repro 三件套齐备）；repro 为 ED-C 机械转正（逐字节拷贝 `probe_dmg.py` + 头部登记块，代码与断言零改动，登记后 `py_compile` PASS；317 行 / 13.8 KB **超 ED-B 双预算**，按 PL-1.10 先例不裁剪披露——裁剪会破坏多形态可运行对照）。第一手材料同时是 `[DESIGN_LIMIT]` 归因（`perf_feedback.md`「不可移植论证」②③）的量化依据。〕

> 〔2026-10-17 蒸馏，④ 为 D 类 Tier 0 **update**（新增可用原语，收窄「整类不可行」的旧表述）：本条目自带 repro 覆盖原三形态（负面侧逐条断言）；④ 的**正面**代码形态由共享 repro `PATT-row-pingpong-pipeline.py`（同一 kernel 结构，逐位断言）与 C 类档案 `CASE-pool-maxpool2d-migration` 承载，本条目未新增独立 repro（④ 的完整 phase 路径依赖该算子的 `_phase_plan`，最小化会丢失 taps/守卫语义——如实披露，待同族任务补）。〕

---

id: TRAP-deinterleave-two-phase-only
kind: trap
family: [pooling, window, layout, general]
apis: [T.deinterleave]
dtype: [fp16]
device: 910B3
status: verified
origin_task: max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 optimize，`probe_deint` 可行性探针）/ max_pool2d-_max_pool2d_kernel-20261017T043853Z（Stage 4 optimize，`probe_pipe` 运行时分支形态）
toolchain: tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-16；运行时分支形态 tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025（commit b3a7b21）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3（aicore 20 → 纯 Vector 40 / UB 196608 B）/ 2026-10-17
repro: repro/TRAP-deinterleave-two-phase-only.py
---

### `T.deinterleave` 相位数上限实测 = **2**：`channel_nums=3` 使 `bishengir-compile` 段错误（非报错、非精度问题）

- **合法形态（case A）**：`T.deinterleave(src, d0, d1, channel_nums=2)` 语义正确——`d0 == x[:, 0::2]`、`d1 == x[:, 1::2]` 逐位相等；**region 目标也成立**（case C：`d0[0:ch, 0:T]` 写入 taller buffer 的切片，未触碰的尾行保持 seed 值 ⇒ 可复用已有缓冲、不额外占 UB）。
- **失效形态（case B）**：`T.deinterleave(src, d0, d1, d2, channel_nums=3)`（stride 3 的滑窗类需要 3 相）在 **NPU IR 已生成之后**由 `bishengir-compile` **段错误**（rc = −6 / SIGABRT，无 MLIR 诊断）⇒ 相位数 ≥3 在本工具链**不可用**（非「参数不当」）。
- **失效形态（case D，运行时控制流内，2026-10-17）**：`T.deinterleave` 落在**运行时 `if/else` 分支**内时被 npuir lowering **静默丢弃**——探针用 `r % 2` 运行时奇偶分支对两路 deinterleave 做 ping-pong 行选择：**编译通过、运行无报错，输出全错**；生成的 hivm IR 中**不存在** `vdeinterleave`（应被写入的缓冲被下游当成未初始化数据读）。判据：含 deinterleave 的 kernel 结果全错且降级 IR 无该 op ⇒ 先怀疑落在运行时控制流内（降级 IR 文本通道见 `TRAP-npuir-lowered-mlir-access`）。**绕法**：奇偶在 **trace 期展开**（pair loop 拆两、两条 deinterleave 都是直线代码）——同探针 448-unit 负载逐位精确（`max_diff = 0.0`）。
- **文档依据（负向断言）**：`docs/Tilelang.language/shape操作/T.deinterleave.md` §2.1 把 `channel_nums` 描述为 `int`（缺省 `2`）、§2.3「特殊限制说明：**无**」、§2.4 唯一示例为 2 相（`N % 2 == 0` 断言）——即**相位数上界未文档化**；本节结论按实测记录并绑版本戳（未文档化假设）。
- **绕法**：只用 2 相形态（`channel_nums=2`）且**必须是直线代码**（不得位于运行时 `if/else` 内——case D）；需要 3+ 相时改走逐相 `T.copy`/切片或 reshape 后的多路 2 相分解（代价另计）。**正向参照**：argmax Stage 4 的「窄行宽视图 + deinterleave 相位分解」130.3× 属合法 2 相形态（`pattern-library/cases.md#CASE-argmax-argreduce-stage4`）；窗口池化 `s_w==2` 的「整行滑窗 max + 2 相抽取」路线实测 −55.8%~−89.6%（见 `TRAP-window-decimation-primitives` ④）。
- **与后端崩溃族的关系**：同属 `bishengir-compile` 段错误家族但触发维度不同——`TRAP-bishengir-crash-s32` 是 **shape × dtype** 敏感，本条是 **API 参数（相位数）** 敏感；两形态是否同源未证（均无 MLIR 诊断可比对），分别登记。
- **重验触发**：若 repro 的 case B 打印三路 `d0/d1/d2 == x[:, i::3]` 全 `True`，说明该限制已解除——按证伪协议刷新版本戳并关闭/收窄 `.agents/evolution/capability-gaps.md` **CG-2026-0032**。
- **量纲**：判据为二元（语义正确 / 后端段错误），无性能量纲。
- **证明**：`python repro/TRAP-deinterleave-two-phase-only.py`（三 case 各自捕获异常并打印 PASS/FAILED，exit 0）；溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/probe_deint.py`。**注**：任务 `opt_log.md#§7.2` 引用的崩溃日志 `perf_opt/logs/probe_deint.log` 在工作区中**不存在**（工作区扫描 0 命中）——durable 证据是本 repro 脚本本身（重跑 case B 即复现），日志路径按 provenance（允许失效）对待。
- **case D 的证明形态（如实披露）**：负面现象（运行时分支内 op 消失）由来源任务实测记录承担（provenance，允许失效）；知识域侧只断言**合法形态**——`python repro/PATT-row-pingpong-pipeline.py --check` 断言 trace 期展开后的 ping-pong 结果与 serial 参考 `torch.equal` 逐位相等（448 unit），并打印两变体的 `dur/vec/mte2/mte3`。原探针未保留 `r % 2` 分支变体，故**负面侧未内联**（与 `TRAP-bishengir-crash-s32` 形态② 同处置形态）。

> 〔2026-10-16 蒸馏，D 类 Tier 0 合入（provenance + 版本戳 + repro 三件套齐备）；repro 为 ED-C 机械转正（逐字节拷贝 `probe_deint.py` + 头部登记块，代码与断言零改动，登记后 `py_compile` PASS；179 行超 100 行预算（<8 KB），按 PL-1.10 先例不裁剪披露）。〕

---
id: TRAP-region-atomic-4d-bshd
kind: trap
family: [attention]
apis: [T.atomic_add]
dtype: [fp32]
device: 910B4
status: verified
origin_task: TileOPs-_gqa_bwd_kernel-20261018T145922Z（DESIGN §0.6 R3 的 910B3 首验在 910B4 重验）
toolchain: tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4 / 2026-10-18
source: DESIGN §0.6 R3 / §9 R1
repro: repro/TRAP-region-atomic-4d-bshd.py
---

### rank-reduced 4-D BSHD 区域 `T.atomic_add` 静默写错（910B4 + CANN 9.0.0 重验，2026-10-18）

- 形态：`T.atomic_add(dst[bz, r0:r0+ROWS, h, 0:D], src_ub, [ROWS, D])`，`dst=(B,S,H,D)`、`H>1`、`D != H*D`、`h=5`（中间 head）。
- 静态 extent：目标区 `[0, 16:48, 5, :]` 只有 **256/2048** cell == 1.0，**区外 1792** cell 被写坏
  （被当作"从该偏移起的连续平铺块"）。
- 动态 extent（`T.min(ROWS, S-r0)`）直接前端报错：`int() argument must be a string, a bytes-like object or a real number, not 'Var'`。
- **可用替代（逐位正确，7× 快）**：把该张量声明为 `(B*S, H*D)` 视图，工厂返回薄闭包做
  `dq.view((B*S, H*D))`（对外签名不变），写 `T.atomic_add(dq2[flat:flat+bn, h*D:(h+1)*D], blk, [bn, D])`。
  该形态是 rank-2 显式切片跨步区域，实测目标区全中、区外 0（本任务 L0/L1 全绿）。
- 重验触发词：`python repro/TRAP-region-atomic-4d-bshd.py --mode static` → 打印 `TRAP reproduced`（若打印 `FIXED` 则后端已修）。
- 同族纪律：region 类 API 的「形态三要素」= 张量 rank / 索引方式（标量 vs 切片）/ 目标行 stride 关系，任一不同即须独立实测。
- 〔2026-10-18 蒸馏核验〕**本条承接 pending `VP-2026-0283` 的 ①**（rank-reduced 4-D 区域形态静默写错）并首次带**知识域 repro**（910B3 首验 + 910B4 重验，跨设备两证）；该提案残余价值 = ② `T.atomic_add` 是真 fetch-add（非零初值 `5.0+1.0 ⇒ 6.0`，零初始化探针无法区分累加与覆盖）、③ 20 block 并发同址精确无丢更新、边界 `H == 1` 或 `dim == H*D` 时缺陷被掩盖——仍待同族任务以 repro 补齐后合入，不与本条重复登记。
- 与 `constants.md` `CONST-atomic-writeback-forms` 的**单价**结论分账：本条是**语义/形态**（可用不可用），单价阶梯（逐行 224 ns/行 vs 区域 32 ns/行）属常数条目；二者互链。

---
id: TRAP-dev-subid-zero-mix
kind: trap
family: [mixcv, attention, expert]
apis: [T.Kernel, T.Scope]
dtype: [fp16, bf16]
device: 910B4
status: verified
origin_task: TileOPs-_gqa_bwd_kernel-20261018T145922Z（PL-1.13 适用域再收窄）
toolchain: tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4 / 2026-10-18
source: PL-1.13-aiv-dup-subid-split（2026-10-15 已收窄到 Mix kernel）
repro: repro/TRAP-dev-subid-zero-mix.py
---

### Developer 模式 Mix kernel 的 `subid` 恒为 0（Vector 程序在双 AIV 上复制执行）

- 探针：`with T.Kernel(4, is_npu=True) as (cid, subid): out[cid*2+subid] = 1 + subid`
  → `out = [1,0,1,0,1,0,1,0]`、`max = 1.0`（从无 2.0）⇒ **`subid` 恒为 0**。
- 含义：本工具链**Developer 模式**的 Mix kernel 无法用 `subid` 切分 Vector 工作；按 subid 切分会静默丢一半行
  （本任务 Stage 4 r4 实测：dq/dk/dv 全错，tier1 超差 12.5 万点，`(1,512,8,64)` fp16）。
  ⇒ `PL-1.13` 的 subid 分片**只适用于 Expert 双 Scope Mix kernel**（`T.Scope("C")`/`T.Scope("V")` + 显式同步）。
- 冗余判据（真实 kernel，非探针）：两个 AIV 子块的 `GM_to_UB_datas` 相同而 `UB_to_GM_datas` 只在 vector0 出现
  （本任务 `(4,512,32,128)` fp16：45045 vs 43698 KB / 16432 vs 0 KB）⇒ Vector 装载与计算各做两遍、原子写单份。
  该冗余是 Developer 模式 Mix kernel 的固有成本（约 AIV 侧 2×），只能靠切 Expert 双 Scope 消除。
- 重验触发词：`python repro/TRAP-dev-subid-zero-mix.py` → `max = 1.0`（若出现 2.0 说明后端已支持 subid 分片）。
- 〔2026-10-18 蒸馏核验〕`attention.md` `PL-1.13-aiv-dup-subid-split` 的适用域已同步做**第二级收窄**（只适用 Expert 双 Scope Mix kernel），并在 front-matter 记本任务溯源；能力缺口侧登记 `CG-2026-0039`（Developer 模式无可用 AIV 子块索引 ⇒ AIV 侧 2× 冗余为固有成本，消除需切 Expert 双 Scope，估计 ~20–30%）。
