# NPU 性能模式库索引（pattern-library/INDEX.md）

> **本目录是 `pattern-library` 的唯一入口**（K-2 渐进披露：INDEX 必读 + `kb_search` 命中条目精读，替代历史单文件全文阅读——历史单文件已收缩为上级目录 `pattern-library.md` 存根）。收录**已实测验证**的性能模式、代价数据、编译器/运行时陷阱与实验方法学规则（首版素材：AvgPool2dFwdOp optimize，2026-08，Ascend910B2C）。

## 1. 维护规则（由 optimizer 任务内回写 + `tilelang-skill-evolver` 终态蒸馏共同维护）

- 每次调优任务产出的新模式/代价数据/证伪更正，由调优 Agent 在任务结束前追加到对应**主题文件**（含任务溯源与工具链版本戳）；evolver 是第二个授权写入者（覆盖各主题文件的增量条目，五种 delta：add/update/consolidate/negate/deprecate；含其他 Stage 产出的 D/C 类价值点）；两次写入互补防双写（evolver 跳过已回写条目）。
- **代价类结论须含量纲（2026-10-15 起）**：条目正文中的性能代价陈述须给出**元素级或单价量纲**（如 `ns/元素`、`µs/pass`、`µs/tile`）并绑定 shape/dtype/workload 上下文；「可接受 / 不显著 / 需预期开销」等**无标度措辞**不构成结论（实测教训：`TRAP-vexp2-not-exact-pow2` 原写「`vshl` 每 tile 一次可接受」，被后续任务实测推翻为单条 op 305.5 µs / ≈20 ns/元素、占满整个 kernel；该条目属预注入知识，未量化的判定直接使该任务首轮未做替换评估）。量化形态参考同库 `CONST-msprof-task-floor-910B3`（地板 + pass 单价 + 拟合式）与 TRAP-vexp2 代价边注（`ns/元素` 与 `µs/pass` 对照）。历史条目不强制回改。
- **条目溯源（D2）**：新增条目须带 `origin_task`（来源任务 id）与工具链版本戳；**下游引用条目时须标注其 origin_task**（optimizer Phase 0 检索注入、designer 步骤 0.5、复盘引用），据此判别可信度与是否需按版本戳重验。
- **预算与生命周期**（由 evolver 维护，K-2 字节预算）：INDEX ≤12KB（承载全条目路由表，2026-09-10 从 8KB 上调，超限优先压缩"一句话"列而非删条目）；每主题文件 ≤16KB；`repro/` 每文件 ≤8KB/100 行。超限触发 consolidate（合并、压缩措辞；负面条目只可 deprecate 不可删除）；`update` 优先于 `add`（同主题已有条目时禁止新开）；**工具链变更后相关条目自动降级"待重验"**（`kb_stale_check.py` 机械检测）。
- **条目自包含（ED-A）**：关键数字、适用条件（shape/dtype/dispatch/workload 上下文）、结论必须内嵌条目正文——provenance（origin_task 与过程文件出处）允许失效，条目仍完整可用；**代码证据唯一合法形态是 `repro/` 内自包含脚本**（`repro: repro-missing` 表示待回填）。**经验需要代码时，用最少的代码表达**（ED-B）：优化点条目（kind: pattern）的代码证据 = 慢→快关键代码更改的最小形态（完整对照或 delta 骨架）+ 优化见解摘要（机制归因到 CONST-\*/TRAP-\* 条目）——经验不随过程文件或最终调优 kernel 的存亡而失效（它们不一定合入主干）。
- 机械校验：`python3 .agents/tools/kb_lint.py`（三件套齐备、front-matter 合法、字节预算、repro 存在性与解耦、状态枚举）；检索：`python3 .agents/tools/kb_search.py "<query>"`。

## 2. 证伪协议（canonical，否定一个 API/模式前必须遵守）

1. **必须用文档合法形态测试**——否定任何 API/模式前，先查 `docs/Tilelang.language/` 确认其合法参数/形式，穷举代表性写法后再下结论。实测教训：曾以非法 3-cycle permutation（`[1,2,0]`）与 Parallel 循环累加形式测出"编译失败"，误判"transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益。
2. **一切编译器/运行时结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
3. **证伪更正须留痕**：推翻旧结论时在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

## 3. 检索规则（调优 Agent 每轮必读）

1. **首轮必查项**：① 向量化轴/布局重估（对照 layout.md，即使上游设计已定——设计期结论可能基于旧工具链）；② 标量执行占比诊断（>50% 触发换轴）；③ 陷阱条目版本戳与 origin_task 核对（工具链变更则重验，`kb_stale_check.py`）。
2. 新模式/新代价数据/证伪更正 → 任务结束前追加进对应主题文件（含溯源）。
3. 本库模式是**起点不是终点**：每算子的最优 tiling（CH/BH/block）仍须实测扫描。

## 4. 条目索引（ID + 一句话 + 状态 + 所在文件）

> 完整 front-matter（family/apis/dtype/device/status/origin_task/toolchain/repro）见各条目；`kb_search.py` 按本索引与条目 facets 检索。

### layout.md — 向量化轴与布局模式

| ID | 一句话 | 状态 |
|----|--------|------|
| PL-1.1-transpose-chain | 核内融合转置链（二轴交换链，实测 ~5.4µs；仅支持二轴交换、禁 reshape→transpose）——**2026-10-14 重验收窄**：仅「两边 ≥ 向量宽且为向量宽整数倍」的 2D 形态成立；池化族大 shape 的 3D 二轴交换被降级为 UB→GM→UB 往返（见 traps-compiler.md TRAP-transpose-ub-gm-roundtrip） | stale |
| PL-1.2-caxis-accum | C 轴切片累加（i/j serial，实测 1.98–13.2x） | verified |
| PL-1.3-host-permute | host permute 路线通常净亏（106–147µs） | verified |
| PL-1.4-tiling-heuristic | BH=1 + 最宽 CH 最优；UB 192KB 容量驱动 CH 取档（2D/1D/双输出 1D 不等式，dtype 感知）；工厂层回退分发；**单波判据**（`num_local==1` 约束优先于「更宽向量」，第五数据点 2026-10-14）；**最小可行 CH**（第六数据点 2026-10-15：一维扫描 `min{CH: units≤cores}` 取代「折半」判据）；**第七数据点**（2026-10-15 双输出：同一规则给出 spp CH 64→48 / nondiv 64→32；进一步削 op 的形态边界见 TRAP-hivm-vsel-cond-slice） | verified |
| PL-1.5-quickref | 乘常数/运行时 let-bound 倒数 / 3D reduce_sum 链 / H-collapse / fp32 求和序匹配 / launch 开销判定 / host 编译期常量折叠 | verified |
| PL-1.15-broadcast-perf-tax | vsub/vmul 未文档化列广播可编译可算对但 +12~16% 性能税——合法化≠可用，列因子仍走 vbrc 展开 | verified |
| PL-1.17-subfp32-fp32-transit | sub-fp32 逐元素 fp32 中转模式（bf16 v-prefix 缺失 + fp16 golden 对齐双触发；lerp+ssd 两证） | verified |
| PL-1.21-vector-persistent-split | 纯 Vector persistent 分核模板（num_kernels=min(num_logical, aicore×2) + T.serial 静态界 + 尾任务守卫；lerp+adaptive-pool 两证） | verified |
| PL-1.22-maxreduce-nan-tracking | max 池化/归约向量实现形态：C 轴向量轴 + fp32 比较域 + eq 掩码乘性 NaN 追踪（`reduce_min+==0` 才是任一 NaN）+ `-inf` 单位元 + 无代数化简（adaptive_max + 窗口 max 2D/1D 四数据点；守卫外提需 ≥2 空间轴）；**C4/整平面域**等代价形态 `isnan(reduce_sum(x*x))`（原位平方，32 vs 掩码链 96-128 B/元素，2026-10-15） | verified |
| PL-1.23-flat-spatial-axis-reduce | 扁平空间轴归约（整平面/全局池化免转置）：UB tile `(CH,H,W)`→`(CH,1,H*W)` + GM 同步声明，归约轴换成宽最内轴——resnet-global 166.26→16.68µs（9.96x，逐位精确） | verified |
| PL-1.25-row-reduce-indicator-complement | 行归约「布尔指示平面 → 归约 → 补式」+ 元素预算双路径（resident / 列分块 + `tn` 整除优先 + 静态尾块免 pad）；补式方向由语义决定（计数 `N−Σ[x==0]` / 谓词 AND `1−max[x==0]` → **vmax no-alias 三件套** / 谓词 OR `1−min[x==0]` → **vmin no-alias 三件套 + 单位元 1.0**，四项随 OR/AND 对偶全反转）；三证（count_nonzero + all + **any**，含运行期逐位精确） | verified |

### elementwise.md — 多输入搬运效率

| ID | 一句话 | 状态 |
|----|--------|------|
| PL-1.6-copy-floor | copy-floor 标定法 + MTE2 带宽退化曲线 + 混合流量地板 + grid-stride 反直觉 + 标量削减判定式（数字见条目）；**轮数自变量扫描**（global 类地板由每核串行轮数主导，≈0.7-0.9 µs/轮，2026-10-15）；**写侧生产者形态是标定自变量**（连续 op 链 4.73µs vs 跨步 gather/`(C,plane)` 转置源 105.7µs；地板显著低于 baseline ⇒ 假地板嫌疑，2026-10-15） | verified |
| PL-1.10-loads-first-decoupling | 单 staging 复用链致 MTE2/VEC 零重叠 → per-input staging + 三输入前置装载（收益数字见条目） | verified |
| PL-1.14-mte3-strided-ws | 跨引擎 ws 中继写侧必须块连续（band 化 MTE3 3× 回退；读侧逐块入 L1 band 列偏移区两全） | verified |
| PL-1.19-scatter-col-write-amplification | (M,1) 列切片 UB→GM 写 42× 放大 → 扁平 chunk-major 连续写 | verified |
| PL-1.20-nsplit-perrow-partial | C6 N-split per-row partial + transpose merge（小 M 大 N 2.2–2.6×；tn=最大单波 nchunk≤cores）；**跨设备复现** 910B3/CANN 9.0.0 双 dtype（0.453×/0.380×，tn=2560） | verified |

### attention.md — Expert persistent / attention 族

| ID | 一句话 | 状态 |
|----|--------|------|
| PL-1.7-expert-persistent-boundary | Expert persistent vs 简单 tiling 收益分界：长 KV 1.57–1.63x 提速、短 KV ~1.31x 回退 | verified |
| PL-1.8-bf16-cube-direct | bf16 Cube 直连链实测可用且无延迟税（含两轮负向断言被推翻的证伪更正） | verified |
| PL-1.9-blockwidth | 块宽摊减律 + cbuf 硬上限 + wide 单槽 + 工厂分派 + H=1 任务平衡 | verified |
| PL-1.9-hardlimits | 第二轮硬上限测绘：L0C/L1 端口/发射/fabric 常数 + f16 softmax 链（数字见条目） | verified |
| PL-1.9-twophase | 两相位重构达标 + morph 阶梯方法 + causal 域画像与宽块解锁反转（第五轮） | verified |
| PL-1.11-causal-mask-scalartrap | 第五轮 causal 域 14.2x：vcmp int16 全形态标量化陷阱 + 算术惩罚掩码 + zbuf l1_b 零初始化（NaN 边界）+ UB ×1.10–1.12 开销 | verified |
| PL-1.12-task-pipeline-depth2 | TASKDONE 屏障冗余审计（全域 −7~−21%）+ Cube 深度 2 任务流水 + f32 S 载体；Vec 侧同构 blocked；r9 bn 钳位域×bm80 UB 守卫（判据/守卫式见条目） | verified |
| PL-1.13-aiv-dup-subid-split | Mix kernel 双 AIV 默认重复执行 Vector 程序；subid 边界表达式分片（蛇形均衡）实测 −28~−36%（判据：两 AIV 子块指标相同）；**2026-10-15 适用域收窄**：纯 Vector kernel `subid` 恒 0，本条仅适用 Mix kernel（否则静默丢一半工作，假收益破绽 = GM 流量精确减半） | verified |
| PL-1.18-ssd-steady-structure-floor | 稳态画像以最长任务串 workload 为准 + 段数记账更正（指令数互证法）+ 三胜出（几何 1.078×）+ 十方向否决 + 重建复证（+9.7%；明细见条目） | verified |
| PL-1.16-expert-dualscope-bypass | Developer 阻塞（标量化 / 条件构造崩溃 / persistent+gemm 崩溃）的结构级绕法：Expert 双 Scope + pass_configs 硬边界（attention+ssd 两证） | verified |

### traps-compiler.md — 编译器/解析器陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-C1-old-compiler-transpose | 旧编译器 transpose/C-slice 不可用（已失效，见 PL-1.1/1.2） | overturned |
| TRAP-C10-strided-gather | 跨步 gather 向量指令缺失 | verified |
| TRAP-C11-parser-vartable | parser var-table 作用域问题 | verified |
| TRAP-C12-copy-dtype-cast | T.copy 静默跨 dtype 转换（勿用作 cast 融合） | verified |
| TRAP-UB-multibuffer-inflation | auto-multi-buffer UB 膨胀 ~1.7x 超 192KB；基础分配差 ×1.10–1.12（多 buffer 组合体实测可达 ~1.5×，见第五数据点） | verified |
| TRAP-vbrc-scalar-shared | vbrc 标量→shared 不可用；serial 变量条件 segfault | verified |
| TRAP-threads-kwarg-noop | T.Kernel(threads=) 在 npuir 无效果 | verified |
| TRAP-tvm-parser-rules | TVM script 解析器四条硬规则（if/三元式/条件 alloc/T.rs 作用域） | verified |
| TRAP-UB-dynsubview-dominance | task 级 UB 行 + 嵌套循环动态偏移 subview → auto-multi-buffer 非支配 IR（Q≥128） | verified |
| TRAP-expert-v-operands | Expert v 算子操作数规则（vcmp 拒绝 tir.Cast 等） | verified |
| TRAP-transpose-epilogue-poison | 活跃源 transpose epilogue 毒化整 kernel（2.6x；绕法 = 增维视图） | verified |
| TRAP-parallel2d-bm-ge2-scalarize | 2D T.Parallel(bm,N) 融合 ite 在 bm≥2 标量化（与操作数形态无关；链式/per-row 绕法）；**2026-10-15 适用域收窄**：plain 逐元素 2D loop（单向量 op/算术/cast）在 bm≥2 正常向量化，触发项含循环体内 ite 形态 | verified |
| TRAP-vbrc-same-shape-empty-broadcast | vbrc 同形 src/dst → 空广播维数组 MLIR verify fail | verified |
| TRAP-vselect-inplace-carried-state | 原位 vselect(cond,A,B,B) 多迭代 T.serial 丢 loop-carried 状态（tile-0 局部值）；绕法 = noalias select + T.copy 回写 | verified |
| TRAP-tir-range-loop-not-unrolled | `@T.prim_func` 内 `for i in range(n)` 是串行 TIR 循环（loop var = tir.Var）非 Python 展开；静态 T.serial 界 + 运行时守卫为可行形态 | verified |
| TRAP-vcast-shape-and-gm-source | `T.vcast` 形状须一致（无 broadcast/reshape）+ 源须为 UB（GM 源 BiShengHIR pipeline 失败）；伴随：runtime let-bound 标量可作 vmul src2 | verified |
| TRAP-hivm-vsel-cond-slice | `hivm.hir.vsel` 的 cond **不接受从更大 UB buffer 切出的 bool 区域**（同 slice 作 `vcmp` 操作数合法）→ 双输出池化「行级掩码直送 vselect」的削 op 不可达；**否定 v 算子 API 前须穷举 whole-buffer/切片/bool-cond/rank 形态** | verified |
| TRAP-vcmp-ne-ordered-nan | `T.vcmp("ne")` 是 ordered 不等（NaN 恒假），不可作 NaN 探测；`eq`/`le`/`ge` 为 IEEE 正确（合法绕法 = eq 掩码乘性累积，含 repro）；**2026-10-16 计数类延伸**：`[x≠0]` 的合法形态 = 有序 EQ 补式 `N − Σ[x==0]`（NaN 计入非零，±0 不计入，亚正规不 FTZ，NPU 运行期三 dtype 实证；`T.vnot` dtype 表无 bool ⇒ 取反路线不可行）；**2026-10-16 谓词 AND 类延伸**：`all` = `1 − max[x==0]`（合并算子由语义决定，误用 ordered-NE 整行翻 False；运行期 11 行角点 + NaN 列 × 3 dtype 逐位实证）；**2026-10-16 谓词 OR 类延伸**：`any` = `1 − min[x==0]`（单位元 1.0，四项随 OR/AND 对偶全反转；误用 ordered-NE 的暴露集更窄 = 恰「NaN 与零共存且无其他非零」/`all-NaN` 两行；运行期三 dtype 逐位 + 空 OR 单位元 False） | verified |
| TRAP-npuir-ref-backend-golden-capability | NPU 参考后端（torch_npu）两形态能力缺口：`dilation!=1` 抛 `AclNN_Parameter_Error`（op 未执行）+ `return_indices=True` 的 indices **静默损坏**（int8 mask buffer，全 dilation）；纯选择类算子回退 CPU golden 逐位一致 | verified |
| TRAP-npuir-vbrc-region-dropped | 标量 `T.vbrc` 的 destination 为 UB region（运行时偏移 + `T.serial` 内）时写入静默丢弃（int32 形态读未初始化 → 向量核 fault 507034）；等价形态 = staging buffer + `T.copy`；裸 Python 标量触发 rank 断言 | verified |
| TRAP-npuir-arange-pattern-with-whole-plane-reduce | `arange` 派生平面下标 pattern standalone 精确、在整平面 `vcmp(eq)→vselect→reduce_min` 链内读回损坏（values 仍精确 → 值-only 门禁漏；int64 索引 oracle 才暴露）；**2026-10-15 收窄为「pattern 生产者特有」**——运行期构建的位置表（非 `arange`）可绕过（逐元素精确），但双输出下被归约链 UB 单价 237 B/元素与轮数封顶（见 CONST-dual-output-whole-plane-reduce-ub） | verified |
| TRAP-vcast-same-dtype-rounds | `T.vcast` 同 dtype（f32→f32）退化为 round-to-integer、静默坏值——同 dtype 搬运用 `T.copy`（按 dtype 在 trace 期分派；`vcast` 只用于真 dtype 变化） | verified |
| TRAP-npuir-reshape-dst-buffer-shadow | npuir 分支 `T.reshape` 被 `npuir_reshape(src,dst)` 覆盖（第二参 = 目标 buffer，非 shape list）；通用形 `T.reshape(A,[m,n])` 语义漂移（含 repro） | verified |
| TRAP-parallel-f32-uint8-cast-rounding | `T.Parallel` 内直接 `T.cast(f32,"uint8")` 非最近舍入（16/64 偏 `round()`）+ uint8 字节实际落 i8 空间（≥128 饱和 127 = fp8 NaN 载荷）；绕法 = 两步 `vcast rint` 后窄化 + signed 空间；**2026-10-15 补充**：`vcast` int32→int8 形态不可用，字节窄化链载体是 f32（含 repro）；**2026-10-16 现象 ④**：bool 载体出栈须两步 `vcast` `f32→f16→i8`（`f32→i8` 无表项；据此推断「无法输出 bool 载体」会得出错误结论） | verified |
| TRAP-vexp2-not-exact-pow2 | `T.vexp2` = `exp(x·ln2)` 非精确 2^k（rel err ~6e-8 使 `n.5` tie 全反向）；精确 `2^k` 走 `T.vshl(1,k)`（k=0..14 逐位精确；**`vshl` 被标量化 —— 实测 (32,64) tile 单独 305.5µs / ≈20ns/元素，须换 `vexp2` 向量化路线**）（含 repro） | verified |
| TRAP-transpose-ub-gm-roundtrip | UB 内 `T.transpose` 可被降级为 **UB→GM→UB 往返**（5.6→138.2µs；`UB_to_GM_datas` 48KB/block）——代价强依赖 shape/dtype 形态（`(64,56)→(56,64)` f16 仍走 UB 内）；绕法 = 免转置扁平归约（PL-1.23）。**2026-10-15 追加**：触发条件定到「**写侧生产者形态**」——连续 op 链 store 源 4.73µs vs 跨步 gather / `(C,plane)` 转置 store 源 105.7µs（96KB/blk）；且转置同时拉长 `GM_to_UB` 的时延（mte2 2.1→31.4µs），故**写侧 floor 标定必须区分生产者形态**（含 repro） | verified |
| TRAP-inner-axis-reduce-scalarize | 最内轴（窄、非向量宽整数倍）的 `T.reduce_sum` 逐元素标量化 ≈ 元素数 × ~6ns——单个 op 使 kernel 5.6→155.1µs；绕法 = 合并成宽最内轴（H*W=49）（含 repro） | verified |
| TRAP-copy-layout-propagation-dma-amplification | `T.copy` 的 UB 目标被推成非常规（转置/归约）布局后 GM→UB DMA 退化为 16B/元素分散读（36.75→336KB/block ≈9x 扇出，指令数不变）；UB→UB 恒等 copy 不能阻断传播（含 repro） | verified |
| TRAP-scf-for-row-offset-silent-miscompile | 运行时 `scf.for` 循环变量作**共享缓冲行偏移**（`for r in range(bm): … cand[r+i, j] = …`）→ **静默错值**（(2048,4096) fp16 y0=−3 vs ref 3211）；错值档「更快」= 伪收益，须先过 `torch.equal` 精确门（含 repro；与 TRAP-tir-range-loop-not-unrolled / TRAP-parallel2d-bm-ge2-scalarize 同族） | verified |
| TRAP-primfunc-compile-time-if-buffer-def | `@T.prim_func` 内**解析期（Python 层）`if` 定义 buffer** → 分支内名字不逃逸：条件真报 `ParserError: Undefined variable`、条件假同源可编译/运行（**缺陷随 shape 潜伏**，落尾块宽度即炸）；绕法 = 工厂层按变体返回独立 prim_func（含 repro；本文件 §C11 parser var-table 作用域的具体形态）；**2026-10-16 适用域限定**：仅限「分支内定义 buffer」——分支内只含语句/调用（`if mode=="min": T.reduce_min(...) else: T.reduce_max(...)`）编译运行正常，勿泛化为「禁任何解析期 if」 | verified |
| TRAP-npuir-lowered-mlir-access | 取**降级后 kernel body 的文本通道 = `kernel.mlir_content`**（str，TileLangIR/HIVM MLIR）；`JITKernel.get_kernel_source()` 在 `JitKernel_NPU` 上**不是**文本通道（返回链接 ELF 的 `bytes` 或抛 `TypeError`）——向量化/IR 形态门禁（判决链有无退化为 `arith.cmpf`/`scf.for`/`if_then_else`）的文本通道即 `mlir_content`（含 repro）；**2026-10-16 第二数据点**：`hivm.hir.vreduce` 的 op 名与 `<mode>` 之间**有空格**（`vreduce <min>`）——写 `vreduce<min>` 的子串断言必然落空，匹配时 op 与 `<mode>` 分开断言 | verified |
| TRAP-any-predicate-reduction-pair | 谓词类归约（`any`/逻辑 OR）的「比较模式 + 归约算子」是**一对**：合法形态 = `vcmp("eq")→vcast(i1→f32)→reduce_min→(bm,) vsub(1−min)`；两向错配暴露集相反——`ne+max`（最短写法）差异**恰为 NaN 行**（`[4,5]`，仅「NaN 与零共存且无其他非零」/`all-NaN` 暴露，其余行被非零救回）、`eq+max`（照抄 `all`）差异 `[2,4,6,7]`；`any`↔`all` 四项全反转（max↔min / `1−max`↔`1−min` / vmax↔vmin / 0.0↔1.0）（含 repro；与 TRAP-vcmp-ne-ordered-nan 同源） | verified |
| TRAP-parallel-affine-cast-plane | `T.Parallel` 二维 body 内的 affine 索引 cast **静默丢行项**（`(16,32)` fp32 平面末行末 6 lane 读回 `j`；差式 cast 形态上下文相关——隔离探针精确而本 repro 同错）——唯一可靠形态 = 嵌套 `T.serial` 标量构造（四档几何全 0 错，代价每任务一次 `bn×bm` 标量 store）；**2026-10-16 新增，含 repro** | verified |
| TRAP-gemm-dynamic-size-cbuf | `T.Pipelined` 内的 `T.gemm(size=<runtime extent>)` ⇒ **伪 `cbuf overflow`**（报文 `requires 12009693044328448 bits` 无意义，声明 tile 仅 ~100KB）；**两条件缺一不可**（运行时 extent ∧ Pipelined 内；静态 size 或 Pipelined 外均通过，`T.copy` 动态 extent 不受影响）；尾块可用形态 = 动态性只留 GM↔UB copy 与行原子、GEMM 全 tile + 操作数预清零；**2026-10-16 新增，含 repro** | verified |
| TRAP-vec-cube-ub-scratch-role-conflict | 同一 UB buffer 兼作 vector 暂存与 GEMM 输出 ⇒ 消费端**静默读到损坏值**（2048 cell 全错、`max_diff 1.33e2`、无报错无 NaN）——一个 buffer 只能有一个引擎写者角色；kernel 级后果 = causal 档 mask 惩罚与 g2 抢 `dsum` 致 `dk` 最大 1.96e4（独立 `dsd` 缓冲后 6.7e-4）；**2026-10-16 新增，含 repro** | verified |
| TRAP-reduce-rowsum-store-shape | 行归约输出 `(blk_s,1,1)` 直写 S 连续段被 codegen 拒（`bufferization.materialize_in_destination` shapes incompatible）——region 单位维**只能从左侧折叠**、payload 轴须两侧尾部对齐；绕法 = 1-D UB 中转（`max_diff 3.3e-06`）或单轴 `T.transpose`（引入 `vtranspose`）；**2026-10-16 新增，含 repro** | verified |
| TRAP-bishengir-crash-s32 | `bishengir-compile` **形状/dtype 敏感段错误**（CANN 9.0.0）**两形态**：① `S == 32`（Developer persistent+Pipelined+动态 extent copy+gemm+行原子骨架；`S ∈ {16,24,40,48,56,64,96}` 全编译、与 tile/heads/dim/is_causal 无关）② `(B=1,H=64,D=128,S=2048,fp16)`（fwd/bwd 两条 bench 同崩，同 shape bf16 与其余 shape 全正常）⇒ 后端缺陷、须与精度告警分通道记录；**2026-10-16 新增（形态①含 repro，形态② repro-missing）**（重验触发词：`S=32: COMPILED` / `(batch1,heads64,dim128,seq2048,fp16): COMPILED`） | verified |
| TRAP-npuir-ir-gate-loop-scope | IR 形态门禁的 `scf.for` / `if_then_else` **禁令按路径/变体限定**——驻留（单 tile）路径 lowering 无 `scf.for`，列分块路径的静态 `T.serial` tile 循环**合法地**含 `scf.for`；把 `TRAP-parallel2d-bm-ge2-scalarize` 的禁令逐字套到 tiled 变体会对**完全向量化**的 kernel 报假失败；正确门禁 = 对所有变体断言 `vreduce <mode>` + `reduce_dims=[1]` + 无 `arith.cmpf`，禁令只对驻留变体（含 repro，compile-only） | verified |

### traps-runtime.md — 运行时/数值/语义陷阱

| ID | 一句话 | 状态 |
|----|--------|------|
| TRAP-C9-taskqueue-async | TASKQUEUE=false 异步 launch 损坏 fp32 数据 | verified |
| TRAP-UB-dst-align | UB dst 非零起点切片 + 32B 倍宽度触发 VEC 对齐错误 | verified |
| TRAP-fp16-opmath-golden | torch CPU golden fp16 opmath 域分歧（对齐通解 = fp32 中转链） | verified |
| TRAP-load-nd2nz-strided | load_nd2nz / T.copy base+size 对跨步区域静默平坦误读（绕法 = slice 形态） | verified |
| TRAP-T-copy-region-semantics | T.copy 区域语义三规则（前向补 1 / 越界写坏相邻 GM / [N,1] 越界读） | verified |
| TRAP-zero-input-crash | 零输入 kernel 必崩 MTE DDR（同款崩溃形态易误诊） | verified |
| TRAP-vrsqrt-plain-precision | vrsqrt 近似指令 ~2.9e-3（绕法 vsqrt+vdiv 1.07e-7；dtype 指纹见条目） | verified |
| TRAP-L1-band-dst-tail-overrun | L1 band 组装 dst 列区间须尾块裁剪（越界写依 L1 布局触发——bf16 必现 fp16 靠运气；分支门禁须含 bf16×非整除 shape） | verified |
| TRAP-DEVMODE-PERSIST-GEMM | Developer+persistent+gemm 混排运行时崩溃（unaligned UUB）；Expert 同结构正常（双模式对照 repro） | verified |
| TRAP-BENCH-CONFIG-CALIBRATION | 采数 harness 默认配置 ≠ kernel TUNED 交付配置 → 续跑场景 +19~21% 假回退（先做双配置探针再谈设备漂移；harness 默认须从 TUNED 常量解析） | verified |
| TRAP-torch-oracle-maxpool1d-empty-window | 空窗口池化 regime（`d≥2 ∧ p>0`）torch 索引越界、源给 `(-inf,0)`、哨兵给 `L_in` 三方互异——该域 torch 不可作 oracle（含纯 torch repro） | verified |
| TRAP-pool-index-contract-dual-output | 带索引池化源语义契约：索引 = 空间域展平位置（2D 扁平 / 1D 单轴）、tie 取首个、NaN 取最后命中、padding 全 `-inf` 窗取首个有效元素 | verified |
| TRAP-index-reduce-extremum-direction | 索引归约极值方向由「写入语义 + 候选集」决定——单遍破纪录集取 max / 两遍等式集取 min，混搭即静默错值 | verified |
| TRAP-argreduce-edge-semantics | argreduce 边界语义三基准分野：±0.0 可区分全序（NPU）/ NaN 传播→掩码全失败输出哨兵 `2^30`（两证）/ ±inf 精确——golden 选择与容差设计据此分野 | verified |
| TRAP-out-idx-uint8-dtype-fallback | `@tilelang.jit(out_idx=...)` 对非白名单输出 dtype（uint8 字节载体）静默 fallback 到 `torch.float32`（shape 对、dtype 错、不报错）；绕法 = 显式传输出张量（含 repro） | verified |
| TRAP-fp16-op-affine-double-rounding | fp16 分层算子（kernel 舍回 dtype + op 层 dtype affine）二次舍入在 ≤0.3% 元素上超 atol=rtol=1e-3；kernel 逐位正确（0/4096）、op 层 fp32 affine 单次舍回后 0 失败（含 repro，纯 CPU） | verified |

### constants.md — 硬件常数表（D-2，设计期 roofline 口径）

| ID | 一句话 | 状态 |
|----|--------|------|
| CONST-capacity-910B2C | UB 192KB（×1.7 膨胀）/ L1 512KB / L0C 128KB + 双槽 bn 上限公式 | verified |
| CONST-L1-port-bw | L1 端口 r+w ≈ 148–154 GB/s/核（Cube 操作数流地板） | verified |
| CONST-vector-launch-overhead | 向量发射 ~0.5µs/op；f16≈f32 | verified |
| CONST-fabric-bw | fabric 聚合 ≥1.73 TB/s | verified |
| CONST-mte2-degradation | MTE2 带宽退化曲线 + 混合流量地板 + L2 驻留口径（数字见条目） | verified |
| CONST-copy-floor-method | copy-floor 标定法（方法学） | verified |
| CONST-flag-id-budget | flag id 预算 ≤15/核（n-block 下标可贴限） | verified |
| CONST-store-fixpipe-gm-only | store_fixpipe 仅 L0C→GM（跨引擎传输强制 GM 往返） | verified |
| CONST-aicore-910B2C | 物理核数 24（910B2C）/ **20（910B3 实查）** + persistent 任务平衡公式；跨型号不可复用本值 | verified |
| CONST-arange-scalar-materialize | T.arange (M,W) 物化标量 ~1ns/elem；(1,N)+首轴 vbrc 绕法（-18.2%） | verified |
| CONST-dual-output-whole-plane-reduce-ub | 双输出整平面 argmax 归约链 UB 单价 ≈237 B/平面元素（单输出 C4 仅 32）→ 封顶 ch=8 → 52 轮 → 262.8µs；`(C,plane)` 行归约标量化 237µs | verified |
| CONST-gm-to-ub-bw-dilution | gm_to_ub_bw 是总时长稀释口径（用 mte2_ratio×aiv_time 判搬运）；跨步列块 2D 读不慢（0.98× 对照 + 误读更正） | verified |
| CONST-reduce-dims0-skinny | dims=0 长条形 (K,4) 逐列串行（(400,4)=33µs）；transpose+dims=1 绕法 | verified |
| CONST-npuir-f32-division-not-ieee-rounded | npuir fp32 除法非 IEEE correctly rounded：`T.vdiv` 2-D ~6.1% lane 1 ulp 偏、标量除法式 135/256 值 1–2 ulp 偏、**列形态 (M,1)/(M,1) 逐位精确**（含 repro） | verified |
| CONST-copy-floor-dma-granularity | 同形纯拷贝地板随 **DMA 粒度**变化（(2048,4096) fp16 / 910B3：16 KB 15.82µs=1.06TB/s、64 KB 13.61µs=1.23TB/s；64 KB 档需 4× UB，与算法缓冲不可共存） | verified |
| CONST-aiv-time-decomposition | `Task Duration ≈ max(mte2, vec) + scalar` 分解口径（三处同档拟合）+ ratio 读法（vec/scalar 是 aiv_time 分量非总时长占比）+ 「UB 字节吞吐」误判更正（链式改 fp16 仅 −1.7%） | verified |
| CONST-msprof-task-floor-910B3 | msprof op Task 固定地板（空 kernel 3.37µs / 固定开销 ≈2.4µs）+ 必需流量地板 4.25µs + 逐元素 pass 单价 0.23–0.3µs + 拟合式 `Task ≈ aiv_time + 2.4–2.6µs` | verified |
| CONST-reduce-ub-per-element | BishengHIR 归约 lowering 的 UB 单价：按 `ch×H*W` 线性记账、单价由归约链形态决定（原位平方和 ≈32 / 非原位 ≈64 / fp16 掩码 ≈96 / bool 掩码 ≈128 B/元素），**绑定 3-D 整平面 `(CH,1,H*W)`+dim=2**；CH 上限 = 196608/(单价·H*W)；探针须 DCE 免疫；**2026-10-16 形态适用域追加**：2-D 行归约（`(bm,tn)`+dim=1）单价按 dtype 分档——fp16/fp32 精确 8.00 / **bf16 12.00**（`vcmp` 上抛 f32）B/元素，硬上限 24576（fp16/fp32 口径），竞态安全档 `E_cap=8192`；**第四数据点**（`any`，独立重标定）：`reduce_min` 链与 `sum`/`max` 链**同价**（归约算子不额外定价） | verified |

| CONST-vcast-rint-downcast-bitexact | `T.vcast(..., round_mode="rint")` 的 f32→f16/bf16 **下转换与 `torch.Tensor.to` 逐位相等**（承载算子 115 用例 0 bit 差异、`max_diff 0.000e+00`；独立 repro 角点 + 混合域各 0/2048，`f16→f32` 控制组 0/2048）⇒ 纯下转换算子可合法采用**逐位门禁**；对照 = 同 dtype `f32→f32` 2047/2048 ≠ 源（须走 `T.copy`）；**2026-10-16 新增，含 repro** | verified |
| CONST-reduce-ub-price-axis-alignment | 归约链 UB 单价按**归约轴长度是否向量宽整数倍**分带（与 dtype 正交）：对齐 `D % 8 == 0` **≈10–12 B/元素**（`D=64` 时 `E=16384` 通过、`E=24576` 溢出@11.3）vs 非对齐 **96.0 B/元素**（`E=4032` 即溢出）；`reduce_axis_len == 1` 走便宜带；manifest 域元素预算比「单一保守上界 42 B/元素」宽 2–4 倍，域外防御档须实现层按带收敛；**2026-10-16 新增，含 repro**（与 `CONST-reduce-ub-per-element` 正交维度） | verified |

### cases.md — 案例索引与参考实现集

| ID | 一句话 | 状态 |
|----|--------|------|
| CASE-pool-maxpool3d-tileops | TileOPs 集成包形态参考 | verified |
| CASE-elementwise-mish-opt | elementwise cast 类完整调优档案 | verified |
| CASE-reduction-logsumexp | 规约类独立算子目录参考 | verified |
| CASE-elementwise-lerp-opt | copy-floor 方法学 + 交错 A/B 复核先例 | verified |
| CASE-elementwise-lerp-precision | fp16 精度失败定征与修复完整档案 | verified |
| CASE-pool-maxpool3d-standalone | standalone 池化算子目录 | verified |
| CASE-attention-gqa-expert-full | Expert attention 迁移完整档案（设计修订链 + debug_log D1–D6） | verified |
| CASE-attention-expert-stage4 | attention expert Stage 4 三轮调优档案（含 [DESIGN_LIMIT] 修正闭环） | verified |
| CASE-argmax-argreduce-stage4 | argmax Stage 4 档案（C6 预注册裁决翻转 2.2×/2.6× + 标量化归因证伪更正；几何 11.5×）+ **同族第二实例** argmin Stage 4（几何 1.30× / DMA 粒度地板 / bm 阶梯重标定 / `[DESIGN_LIMIT]`） | verified |
| CASE-argmax-argreduce-integration | TileOPs reduce 族集成契约参考（N_padded 声明宽度 / wrapper config 转调 _select_config / argmin +inf pad；argmax 首证 + **argmin 第二证**〔910B3/CANN 9.0.0 逐字节复现 + 胶水代码 + bench 数值〕+ **第三证缺陷 ⑤**：多 `@T.prim_func` 名 + wrapper 未上浮采数名 → report `partial`/欠归因）+ **第四数据点**：count_nonzero（wrapper config 继承 GPU 启发式的预防性重接线 + 缺陷 ⑤ 未复现对照） | verified |
| CASE-attention-developer-stage4 | developer 谱系同门对照档案 | verified |
| CASE-deepseek-v4-highperf | Expert 跨引擎结构先例（vcmp/flag 协议） | verified |
| CASE-ref-flash-attn-npuir | **参考实现集首条**：两相位结构参照锚点（[DESIGN_LIMIT] 强制对照） | verified |
| CASE-CG-INDEX | 反例档案（capability-gaps open 条目互链） | verified |
| CASE-norm-adalayern-migration | norm 族首个迁移档案（vrsqrt 精度链 + pad 校正舍弃 + 双判据验证） | verified |
| CASE-norm-adalayern-stage4 | norm/row-reduction 调优档案（bm 第一杠杆 + loads-first + 2.14x 几何平均） | verified |
| CASE-attention-twophase-causal-regen | 两相位 causal 域重生成 + 第五轮（标量化判别链 14.2x）+ 第六轮（屏障审计/深度 2/Ratio 口径/r9 守卫）调优档案 | verified |
| CASE-attention-mha-config-unvalidated | 反例：设计默认 config 路径未编译验证即出厂（bench 期 UB 溢出） | verified |
| CASE-ssd-chunkscan-migration | mamba/SSD 族 MixCV Expert 迁移完整档案（模式切换 + 六轮 2.91× + 首过集成 + 4515de8 重跑 + 二轮 1.078× + 重建会话 repro 骨架重推导 +9.7%） | verified |
| CASE-pool-adaptiveavgpool2d-migration | pool 族首个迁移档案（C 轴向量轴 + 四几何特化路径 + 3 轮设计修订 + 首轮精度过 + 集成 bench 3 例）——**Stage 4 续档 round 1/2**（`T.transpose` 链在当前工具链失效 → 扁平归约替代 9.91x + 单波 CH 分派；含「msprof roofline Ratio 不与 latency 成反比」口径更正实例）；**round 3**（fp16 精简归约链 −3.79% + CH 判据 v2 −5.27% + 地板标定 4.81 µs 固定开销；累计 resnet 10.20x） | verified |
| CASE-pool-adaptivemaxpool2d-migration | pool 族同族追加档案（max 归约 NaN 追踪 + 无代数化简 + vcmp-ne 陷阱定位 + 首轮逐位 max_diff=0 + 集成 bench 3 例）——**Stage 4 续档**（C4 扁平宽归约 + 原位平方-sum NaN 追踪 + 单波 CH 判据；PL-1.13 在纯 Vector 上不适用；320.83→18.86 µs） | verified |
| CASE-pool-adaptivemaxpool2dindices-migration | pool 族第三个迁移档案（双输出值+int64 扁平索引：first-hit 极值方向 + 两 API 陷阱转正 + 整平面索引链回退 + 集成 7/7）；**Stage 4 续档**（2026-10-15：扁平整平面读 + dtype 宽度扫描 + 写侧生产者纪律 + 双输出单波 CH；global 144.4→91.9µs/1.57x；C4 双输出被归约链 UB 单价封顶） | verified |
| CASE-pool-maxpool2d-migration | pool 族第四个 / **窗口池化**首例（adaptive 特化条件不可平移 + 容量驱动 CH + `T.vcast` 同 dtype 陷阱 + NPU golden 缺口两轮集成；三 dtype 逐位 0） | verified |
| CASE-pool-maxpool1d-migration | pool 族第五个 / **1D 首例**（长轴分块 + 连续 span 钳制载入 + 三类尾块 + 容量回退不可达 + bench 脚手架缺陷；三 dtype 逐位 0，零返工） | verified |
| CASE-pool-maxpool1dindices-migration | pool 族第六个 / **双输出 1D 变体**（索引契约 + 两遍法 + 空窗口契约 + 索引型 UB 记账 + NPU golden 静默损坏 CPU 回退；集成 47/47、索引 0 失配） | verified |
| CASE-quant-fp8-quant-migration | quantization 族首个 / **字节载体（fp8）首例**（软件 e4m3 编码 + uint8 载体 + host `.view` 保契约；舍入/符号/幂三陷阱 + 覆盖缺口假 PASS；集成 8/8 + bench 3/3 BW 1.75 TB/s） | verified |
| CASE-norm-groupnorm-migration | norm 族 GroupNorm 迁移档案（affine 分层 + Stage 5 fp16 二次舍入逐位定位 + UB 平台值第三结构数据点 + 矩式方差第二证；集成 19/0/2、bench 4 例 6.0–11.1µs / 1.75 TB/s） | verified |
| CASE-reduction-countnonzero-migration | reduction 族**首个 logical_reduce / 计数类**迁移（count_nonzero kind → int64 契约 + 元素预算双路径 [8 B/元素探针 / E_cap=8192 / 硬上限 24576] + `@T.prim_func` 解析期 if 陷阱 + wrapper `default_config` 重接线；集成 37/37 + bench 4 例 msprof 有效） | verified |
| CASE-reduction-all-migration | reduction 族**第二个 logical_reduce / 谓词 AND 类**迁移（all kind → **bool 输出/int8 载体**（与计数类 int64 契约相反）+ keepdim 额外参数 + 补式 `1 − max[x==0]` 与 `vcmp("ne")` 方向陷阱 + `T.vmax` no-alias 跨 tile 合并 + kernel 侧 `block_m` 元素预算回落（无需改 wrapper）；集成 42/42 + bench 6/6 + msprof 3 例有效；含 `S2-MIG-DIM0`/`S3-GOLDEN-TORCH` 两类门禁交互） | verified |
| CASE-reduction-any-migration | reduction 族**第三个 logical_reduce / 谓词 OR 类**迁移（any kind → bool 输出/int8 载体 + 补式 `1 − min[x==0]` 与 `T.vmin` no-alias 三件套 + 单位元 1.0；与姊妹 `all` 四项全反转（max↔min/补式方向/vmax↔vmin/0.0↔1.0）+ 判别性输入方向相反（高置零 ~70%）+ 空 OR 单位元 False 与 all 相反；两新 trap（`TRAP-any-predicate-reduction-pair` / `TRAP-npuir-ir-gate-loop-scope`）+ `S2-MIG-DIM0` 第四次误报 + `integrate_kernel.py` 预置选择块非幂等（重复表头）；集成 42/42 + bench 6/6 + msprof 3 例有效） | verified |

| CASE-attention-mha-bwd-migration | attention 族**首个 attention-backward 迁移**（同文件追加进既有 fwd 包：3 个 bwd kernel 全部 `[PRECISION_PASS]` 且 conductor 二次校验；**两段终态** = 首轮 `BLOCKED_ENVIRONMENT`〔fwd Expert 核 `syn_instr_mode` 工具链错配〕→ 环境外部 shim 修复后 Stage 5 重试段 **correctness 12/12 / per-op 8/8**、benchmark partial、唯一残余 failure = `S5-ASCEND-MODE` 多 Kernel 共 wrapper 结构性冲突 ⇒ `BLOCKED_INTEGRATION`）——高密度反例档案：3 新 trap（affine cast 平面 / gemm 动态 size / UB buffer 角色冲突）+ `S==32`/大 shape fp16 后端段错误 + 逐行 1-D 行原子形态 + 源忠实 golden tier-3 判据 + gate 5 同文件追加悖论 + bwd 集成态 msprof 基线（`flash_bwd_prep` 35.2–79.7 µs / Ratio 24.1–28.9% / 1.75 TB/s）+ 环境类阻塞「只重跑受影响 Stage」820 s 实证；**2026-10-16 新增（2026-10-16 重试段续档）** | verified |

### repro/ — 最小可复现代码（ED-B）

规范与索引见 [repro/README.md](repro/README.md)；批量执行 `python3 .agents/tools/repro_runner.py`（知识回归测试，ED-D）。

## 5. front-matter schema（K-1）

`id`（稳定 ID，引用与命中统计锚点）/ `kind`（pattern / trap / case / constant）/ `family`（算子族）/ `apis` / `dtype` / `device` / `status`（verified / stale / overturned）/ `origin_task`（provenance，允许失效，不做存在性核验）/ `toolchain`（tilelang commit/build + CANN 版本）/ `repro`（`repro/<file>.py` 或 `repro-missing`；repro 必须存在且可执行，`kb_lint.py` 校验）。

- `update` delta 可对 front-matter 字段做机械更新（status 翻转、repro 登记等）；正文保持 markdown。
- **kind: pattern（优化点）的代码证据要求**：正文含优化见解摘要（机制归因），repro 为关键代码更改最小形态——完整 before/after 对照（效应可小规模复现）或 delta 骨架（效应只在完整 kernel 规模显现，py_compile）+ 见解摘要入 repro 头部（分级断言形态见 [repro/README.md](repro/README.md)）。
- 迁移期新旧规范并存：`repro: repro-missing` 即共存机制（新 D 类条目缺 repro 降级 Tier 1 入队，merge-policy §1）。
