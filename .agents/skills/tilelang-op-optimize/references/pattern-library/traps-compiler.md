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
origin_task: adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z
toolchain: tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-13
repro: repro/TRAP-vcmp-ne-ordered-nan.py
---

### `T.vcmp(..., "ne")` 是 ordered 不等——NaN 操作数下恒假，不能用作 NaN 探测

- **现象**：`T.vcmp(x, x, "ne")` 在含 NaN 的行返回全 0（NaN 所在 lane 亦为 0）——即 ordered not-equal（等价 `lt | gt`）语义；IEEE 的 `!=`（NaN 操作数为真）在本工具链不可得。合法枚举 `eq/ne/lt/gt/le/ge` 的 NaN 行为实测：`eq` / `le` / `ge` 与 IEEE 一致（NaN lane 上 `eq` 为 0、`le`/`ge` 为 0，是 IEEE 的正确比较），`ne` 为 ordered-NE（NaN 恒假），`lt`/`gt` 与 IEEE 一致（NaN 恒假）。对不含 NaN 的操作数对 `ne` 行为正常（并非坏 op）——陷阱只在意图为 NaN 探测时触发（6 种 `cmp_mod` × 2 种构造逐模式实测表见 repro 头部 provenance）。
- **后果（静默失效加假阳性通过）**：以 `v != v` 作 NaN 掩码时掩码恒假（标志位永不触发），而端到端 NaN 用例仍可能"通过"——因为 `T.vmax` / `T.reduce_max` 在本工具链会传播 NaN，"窗口内任一 NaN 胜出"被该未文档化行为顺带修对。adaptive_max_pool2d 迁移 Stage 3 首版即此形态（`vcmp(...,"ne")` + `vmax` 累积），L0-10 NaN/±inf 语义门禁首轮假阳性通过；经 5 组最小探针逐层剥离定位，改用 `eq` 取反后语义判定变为确定性路径、全部用例复跑仍全过且 `max_diff=0`。
- **合法绕法（本工具链实测，逐位正确）**：NaN 探测走 IEEE 正确的 `"eq"`（或 `"le"`/`"ge"`）取反——行级 `cond = T.vcmp(x, x, "eq")` → `T.vcast(cond, f32)` 得 {0,1} 掩码 → 逐元素 **in-place `T.vmul(flag, mask, flag)`**（`flag` 初值 1：0 吸收、1 中性）→ 末尾 `vcmp(flag, 0, "eq")` 即 `has_nan`（整平面 C4 路径以 `T.reduce_min` 收尾）。全程只用 IEEE 正确的比较模式，不依赖 `T.vmax`/`T.reduce_max` 的 NaN 行为。文档侧：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3 只列合法枚举，未说明 NaN 行为。
- **证明**：`python repro/TRAP-vcmp-ne-ordered-nan.py`（exit 0：`ne` 自比 NaN 行全 0 复现 + `eq` 为 `isnan` 的 IEEE 补集断言 + eq-mask 乘性累积 `flag==0 ⟺ 列含 NaN` 断言 + NaN-free 操作数对 `ne` 正常）。与 TRAP-expert-v-operands（vcmp 操作数形态与 int16 标量化）互补：前者管操作数形态与执行代价，本条管比较模式的 NaN 语义。溯源（任务工作区，未上库，允许失效）：`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-vcmp-ne-ordered-nan.py}`。

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
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（op 级 fp8_quant-fp8_quant-20261014T152721Z）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-parallel-f32-uint8-cast-rounding.py
---

### `T.Parallel` 内直接 `T.cast(f32, "uint8")` 非最近舍入；且 uint8 字节实际落在 signed i8 空间

- **现象 ①（舍入）**：`T.Parallel` 体内对**分数** f32 表达式直接 `T.cast(x, "uint8")`（lowering 为 `hivm.hir.vcast` f32→i8）**不是最近舍入**：64 点探针中 16 个值偏离 `round()`，既非 trunc 也非 round-to-nearest-even。**先整数化再窄化**才是可靠形态：`T.vcast(src_f32, tmp_i32, round_mode="rint")` → `T.vcast(tmp_i32, tmp_f32, round_mode="rint")` → `T.cast(tmp_f32, "uint8")`（0/64 与 `round()` 一致）。
- **现象 ②（符号空间）**：codegen 把 `uint8` 降到 **i8**（`memref<...xi8>`）；f32→uint8 直接 cast 对 ≥128 的值**饱和到 127**（= e4m3 NaN 载荷），表现为输出大片 NaN。正确写法是让字节出在 **signed i8 空间**（`bits - 128·[q<0]`，clamp 到 `[-128,127]`）。
- **与前端 warning 的关系**：`src/transform/npu_loop_vectorize.cc:801` 的 warning「Parallel cast lowering uses fixed round mode 'rint'」在 f32→i8/ui8 **窄化**路径上**与实际行为不符**（该 warning 的截断/取整语义登记见 queue VP-2026-0205；本条以 repro 给出「窄化路径非 rint」的反例）。文档面：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 规格表无 `f32|ui8` 行（仅 `f16|ui8`）。
- **证明**：`python repro/TRAP-parallel-f32-uint8-cast-rounding.py`（exit 0；断言直接 cast 与 `round()` 分歧、两步 rint 逐值一致）。
- 族内位置：与 TRAP-vcast-same-dtype-rounds（同 dtype 退化 rint）、TRAP-vcast-shape-and-gm-source（vcast 形状/源约束）同属 npuir cast/舍入族；能力缺口本体见 `.agents/evolution/capability-gaps.md#CG-2026-0020`。溯源（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-parallel-f32-uint8-cast-rounding.py}`。

---
id: TRAP-vexp2-not-exact-pow2
kind: trap
family: [quantization, elementwise, general]
apis: [T.vexp2, T.vshl, T.vcast]
dtype: [float32, int32]
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（op 级 fp8_quant-fp8_quant-20261014T152721Z）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-vexp2-not-exact-pow2.py
---

### `T.vexp2` 不是精确 2 的幂——需要精确 `2^k` 时走 `T.vshl(1, k)`（整数域）

- **现象**：tile 级 `T.vexp2` 实现为 `exp(x·ln2)`，对整数指数返回非精确值（`2^-4 → 0.0624999962747097`，相对误差 ~6e-8，15 个整数指数中 1 个与 `2.0**k` 不等）。用它当 mantissa 缩放因子时，**所有精确 `n.5` tie 被推过错误一侧**（实测 fp16 4/16384、bf16 18/16384 字节错，全部为 `n.5`）。
- **绕法**：精确 `2^k` 走 **`T.vshl(1_int32, k)` → `T.vcast(..., "rint")`**（k = ec+6，实测 k=0..14 逐位精确）再以精确幂除得 `|q|·2^(3-ec)`；编码骨架的字段连续性（`rint` 达 16 时经 `(ec+7)·8+mantissa` 进位）使 ±1 指数误差自校正——但**scale 因子本身必须精确**。
- **证明**：`python repro/TRAP-vexp2-not-exact-pow2.py`（exit 0；断言 ①`vexp2` 路线与精确幂在每探针指数上不等、②`vshl` 路线对 k=0..14 逐位复现 `2.0**k`）。
- **代价边注**：`T.vshl` 在 UB 上被标量化（编译警告 `Op 'hivm.hir.vshl' will execute by scalar instruction with low effiency`）——每 tile 一次可接受，放进逐元素热点则发射开销显著（见 queue VP-2026-0214）。
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
