# repro/ — 最小可复现代码（ED-B 规范）

> 位置：`.agents/skills/tilelang-op-optimize/references/pattern-library/repro/`——知识域内，随知识条目版本化管理。**知识与过程文件解耦**（ED-A）：条目的代码证据唯一合法形态是本目录内的自包含脚本；任务工作区路径（`examples/{project}/{op}/...`、`/tmp/...`）只是 provenance（允许失效）。**原则：经验如果需要代码，就用最少的代码表达这个经验**——过程文件甚至最终调优版本都不一定随 agent 合入主干，经验的可验证性不得依赖它们存活。

## 每个 repro 的要求（ED-B）

1. **可独立执行/可独立检查**：`python repro/<条目ID>.py` 直接跑或通过语法检查——输入自造（脚本内构造最小张量/shape）、不 import 任务工作区模块、不读 `examples/{project}/` 路径（kb_lint 的 `KB-REPRO-DECOUPLE` 机械校验此解耦）；
2. **自描述头**（docstring）：条目 ID 与所在文件、现象/优化点一句话、断言说明、首次验证版本戳、origin_task（provenance）、重验记录（追加式）；
3. **分级断言形态**（按条目 kind，见下节）；
4. **最小化预算**：≤100 行 / ≤8KB——超限即未完成最小化（削减输入规模、剥离与经验无关的结构、只保留慢→快的关键更改）；
5. **单一经验**：一个 repro 只表达一个条目的经验，组合现象拆多文件。

## 分级断言形态（按条目 kind）

**陷阱类（kind: trap；正向/反向经验中的"反向"）**——assert 现象存在或绕法通过：错误数据的 max_diff 阈值 / 编译失败文本匹配 / **绕法通过**（合法形态 bit-exact）。例：`TRAP-load-nd2nz-strided.py`。

**常数类（kind: constant）**——可运行部分 assert 正确性/结构自检；测量部分输出数字并 assert **量级与相对关系**（绝对值仅打印供人对照，环境漂移不锁绝对值）；需要 msprof 的测量步骤在头部写明命令（人工/工具链升级时按需执行，`repro_runner` 只验证可运行部分）。例：`CONST-copy-floor-method.py`（3 载入 + 2 vadd 保活探针，从 lerp `probe_copy.py` 抽取最小化）。

**优化点（kind: pattern；正向经验与优化点）**——**关键代码更改 + 优化见解摘要**，按效应的可复现规模二选一：

- **完整形态**（效应可在小规模独立复现时，如 C 轴切片累加、乘常数倒数）：最小 **before/after 可运行对照**——"慢形态"与"快形态"两个最小 kernel（或同 kernel 的两参数形态），断言**相对关系**（快形态正确且不慢于慢形态 / 收益方向与条目记载一致；收益绝对值仅打印）；
- **delta 形态**（效应只在完整 kernel 规模显现时，如两相位重构、块宽摊减）：**慢→快的关键代码更改**——足以让后续任务定位"改了什么"的最小 diff（慢结构骨架 vs 快结构骨架，≤100 行，`py_compile` 通过，不要求可运行）+ **优化见解摘要**（见下）写入 repro 头部与条目正文。例：`PATT-twophase-restructure.py`（两相位重构的相位分离骨架 + 机制归因）。

**优化见解摘要**（LLM 在回写/蒸馏时从任务上下文生成的一般化洞察）：为什么这个代码更改导致慢→快——机制归因到硬件常数（引用 CONST-\* 条目）或编译器行为（引用 TRAP-\* 条目），写成自包含的一段文字（后续任务不读原 opt_log 也能理解因果）；摘要必须同时落在条目正文（经验自包含）与 repro 头部（与代码同址）。条目正文已含机制说明的（如 PL-1.9 系列的块宽摊减机制解释）即达标，repro 承载代码部分。

## 命名与登记

- 文件名 = 条目 ID（`TRAP-*.py` / `CONST-*.py` / `PATT-*.py`）；
- 条目 front-matter 的 `repro:` 字段登记 `repro/<file>.py`；`repro-missing` 表示待回填（存量条目允许，新 D 类条目缺 repro 降级 Tier 1 入队，merge-policy §1）；
- **责任前移（ED-C）**：repro 的编写与首次运行发生在有环境、有上下文的任务内 Subagent（Stage 3 developer 探针转正 `examples/{op}/repro/`；Stage 4 optimizer 回写 D 类条目与优化模式条目时同步抽取关键代码并实际跑一遍）；evolver 只做**转正（机械拷贝）与校验**（头部规范 + 语法 + 登记），不编写、不裁剪代码。**抽取原则**：从任务工件（perf_opt 探针 / 实验分支 / 参考 kernel）裁出表达该经验所需的最小代码——经验不随过程文件或最终 kernel 的存亡而失效。

## 批量执行（ED-D：知识回归测试）

```bash
python3 .agents/tools/repro_runner.py                     # 全量（trap/const 可运行部分）
python3 .agents/tools/repro_runner.py --filter traps      # 按类别
python3 .agents/tools/repro_runner.py --filter TRAP-load-nd2nz-strided  # 按条目
python3 .agents/tools/repro_runner.py --filter stale      # 工具链升级后受影响条目（kb_stale_check 联动）
```

用途：① 工具链升级重验（`kb_stale_check` 列 stale → `repro_runner --filter stale` → FAIL 即条目推翻、PASS 即刷新版本戳）；② apply 预验证（E-4）与 Tier 1 第二次独立证据（另一上下文重跑通过）复用同一套件；③ delta 形态的 PATT-\* 只做语法级检查（全量 NPU 执行与 msprof 测量按需人工触发）。

## 现存 repro 索引

| repro | 条目 | 形态/断言 | 首次验证 |
|-------|------|---------|---------|
| `TRAP-load-nd2nz-strided.py` | traps-runtime.md TRAP-load-nd2nz-strided | 陷阱：绕法通过（slice 形态 bit-exact；base+size 误读形态见头部重验记录） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-T-copy-region-semantics.py` | traps-runtime.md TRAP-T-copy-region-semantics | 陷阱：合法形态通过（[H,D]ub → 4D **slice**；base+size 形态当前工具链 MTE fault 待重验，见头部重验记录） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-zero-input-crash.py` | traps-runtime.md TRAP-zero-input-crash | 陷阱：绕法通过（dummy input 后全过；崩溃形态见头部说明） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-tvm-parser-rules.py` | traps-compiler.md TRAP-tvm-parser-rules | 陷阱：合法形态通过（两种 FLAG 均正确） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-threads-kwarg-noop.py` | traps-compiler.md TRAP-threads-kwarg-noop | 陷阱：源码语义核对（src/ir.cc NPU 分支无 threadIdx 绑定） | 2026-09-07 源码通读（2026-09-10 转正重跑 PASS） |
| `CONST-copy-floor-method.py` | constants.md CONST-copy-floor-method | 常数：可运行部分探针数值精确（从 lerp `probe_copy.py` 抽取）；msprof 测量步骤见头部 | 2026-09-07 task（2026-09-10 抽取重跑 PASS） |
| `PATT-twophase-restructure.py` | attention.md PL-1.9-twophase | 优化点 **delta 形态**：两相位重构关键代码更改（相位分离骨架 + 单 L0C 复用 + flag=n-block 下标）+ 机制归因见解摘要；py_compile | 2026-09-09 task（2026-09-10 抽取，语法 PASS） |
| `PL-1.10-loads-first-decoupling.py` | elementwise.md PL-1.10-loads-first-decoupling | 优化点 **delta 形态**：单 staging 复用链（慢）vs per-input staging + 三输入前置装载（快）骨架 + 机制归因见解摘要；py_compile | 2026-09-10 ada_layer_norm Stage 4（任务内编写，语法 PASS） |
| `TRAP-vrsqrt-plain-precision.py` | traps-runtime.md TRAP-vrsqrt-plain-precision | 陷阱：现象存在（raw vrsqrt rel err > 1e-3）+ 绕法通过（vsqrt+vdiv < 1e-6）；NPU 可运行 | 2026-09-10 ada_layer_norm Stage 3（evolver 机械转正，py_compile PASS；任务内首次运行见 origin 工件） |
| `TRAP-tir-range-loop-not-unrolled.py` | traps-compiler.md TRAP-tir-range-loop-not-unrolled | 陷阱：`range` trap 复现（tir.Var 索引）+ 绕法通过（静态 T.serial 界 + 守卫，max_diff=0） | 2026-10-13 adaptive_avg_pool2d Stage 3（evolver 机械转正，逐字节拷贝 + 头部登记 + py_compile PASS） |
| `TRAP-vcast-shape-and-gm-source.py` | traps-compiler.md TRAP-vcast-shape-and-gm-source | 陷阱：形状不符 + GM 源两条 lowering 失败复现 + 合法形态/伴随行为通过 | 2026-10-13 adaptive_avg_pool2d Stage 3（evolver 机械转正，逐字节拷贝 + 头部登记 + py_compile PASS） |
| `TRAP-vcmp-ne-ordered-nan.py` | traps-compiler.md TRAP-vcmp-ne-ordered-nan | 陷阱：`ne` 自比 NaN 恒假复现 + `eq` 为 IEEE 补集断言 + eq-mask 乘性累积绕法通过（`flag==0 ⟺ 列含 NaN`）；NPU 可运行 | 2026-10-13 adaptive_max_pool2d Stage 3（evolver 机械转正，逐字节拷贝 + 头部条目登记 + py_compile PASS；133 行超 100 行预算按 PL-1.10 先例不裁剪披露） |
| `TRAP-npuir-vbrc-region-dropped.py` | traps-compiler.md TRAP-npuir-vbrc-region-dropped | 陷阱：标量 `T.vbrc` 的 destination 为 UB region（运行时偏移 + `T.serial` 内）时写入被丢弃复现（region 形态 seed 丢失）+ 绕法通过（staging buffer + `T.copy`，ch=8/32/64 逐 ch 生效）+ 裸 Python 标量被前端 rank 断言拒绝；NPU 可运行 | 2026-10-14 adaptive_max_pool2d_with_indices Stage 3（evolver 机械转正，逐字节拷贝〔md5 4b4e4d12…〕+ 头部核验 + py_compile PASS；132 行超 100 行预算按 PL-1.10 先例不裁剪披露） |
| `TRAP-npuir-arange-pattern-with-whole-plane-reduce.py` | traps-compiler.md TRAP-npuir-arange-pattern-with-whole-plane-reduce | 陷阱：`arange` 派生平面下标 pattern standalone 精确（control 断言）而在整平面 `vcmp(eq)→vselect→reduce_min` 链内读回损坏（trap 断言 + 派生 first-hit 索引与真值不等）；int32 与 fp32 载体各一遍；NPU 可运行 | 2026-10-14 adaptive_max_pool2d_with_indices Stage 3（evolver 机械转正，逐字节拷贝〔md5 31d162f2…〕+ 头部核验 + py_compile PASS；159 行超 100 行预算按 PL-1.10 先例不裁剪披露） |
| `TRAP-vcast-same-dtype-rounds.py` | traps-compiler.md TRAP-vcast-same-dtype-rounds | 陷阱：f32→f32 `T.vcast` 静默退化为 round-to-integer 复现（`vcast == round(src) != src`）+ `T.copy` 同 dtype 逐位一致（control）+ f16→f32 `vcast` 正确；NPU 可运行 | 2026-10-14 max_pool2d Stage 3（evolver 机械转正，逐字节拷贝〔md5 4c8f72e7…〕+ 头部核验〔条目 ID/版本戳/断言齐全，origin_task 为简写形态已披露〕+ py_compile + AST PASS；115 行超 100 行预算按 PL-1.10 先例不裁剪披露） |
| `TRAP-torch-oracle-maxpool1d-empty-window.py` | traps-runtime.md TRAP-torch-oracle-maxpool1d-empty-window | 陷阱（测试设计）：空窗口 regime（`d≥2 ∧ p>0`）premise + torch 索引越界 + 源对齐契约值 0 且 ≠ torch + 非空窗口对照组四段断言；**纯 torch CPU、无 NPU 依赖**（oracle 类陷阱的首例，头部已标 `First verified` 环境戳） | 2026-10-14 max_pool1d_with_indices Stage 3（evolver 机械转正，逐字节拷贝〔md5 46159ba6…〕+ 头部核验 + py_compile PASS + `repro_runner --filter` 实跑 PASS〔7.2s〕；96 行 / 4 705 B 在预算内） |
| `TRAP-npuir-reshape-dst-buffer-shadow.py` | traps-compiler.md TRAP-npuir-reshape-dst-buffer-shadow | 陷阱（前端符号层）：生效 `T.reshape.__name__ == 'npuir_reshape'`、签名 `(src,dst)`（第二参 = 目标 buffer）、通用 `customize.reshape(src, shape)` 仍以原名存在——三段断言防「alias 移除/签名恢复」静默改变存量 kernel 含义；**纯前端符号检查、无设备依赖** | 2026-10-14 argmin Stage 3（evolver 机械转正：源 `examples/argmin/_argreduce_kernel/repro/` md5 4983c9bb… → 逐字节拷贝 + **头部追加登记行**〔条目 ID / First verified / origin_task；代码与断言零改动〕→ md5 aa5cf057…（登记后变更已披露）+ py_compile PASS + `/workspace/venv/bin/python` 实跑 PASS；54 行/2 287 B 在预算内） |
| `CONST-npuir-f32-division-not-ieee-rounded.py` | constants.md CONST-npuir-f32-division-not-ieee-rounded | 常数：2-D `T.vdiv` 结果对正确舍入商（fp64→fp32）断言非零 1-ulp 比例并打印实测率（锁量级与相对关系，绝对值随环境漂移；同时断言偏差 ≤2 ulp、非 lowering 损坏）；NPU 可运行 | 2026-10-14 fp8_quant Stage 3（evolver 机械转正，逐字节拷贝〔md5 84149c83…〕+ 头部核验〔条目 ID / First verified 版本戳齐全〕+ py_compile PASS；80 行/3 855 B 在预算内） |
| `TRAP-out-idx-uint8-dtype-fallback.py` | traps-runtime.md TRAP-out-idx-uint8-dtype-fallback | 陷阱：`out_idx` 返回 dtype == float32 而 prim_func 声明 uint8（静默回落，shape 对 dtype 错）+ 绕法通过（显式传 uint8 张量往返 bit-exact）；NPU 可运行 | 2026-10-14 fp8_quant Stage 3（evolver 机械转正，逐字节拷贝〔md5 7f736ddd…〕+ 头部核验 + py_compile PASS；81 行/3 479 B 在预算内） |
| `TRAP-parallel-f32-uint8-cast-rounding.py` | traps-compiler.md TRAP-parallel-f32-uint8-cast-rounding | 陷阱：`T.Parallel` 内直接 `T.cast(f32→uint8)` 与 `round()` 分歧复现（16/64）+ 两步 `vcast rint` 后窄化逐值一致（0/64）；覆盖舍入空间与 signed i8 字节空间两陷阱；NPU 可运行 | 2026-10-14 fp8_quant Stage 3（evolver 机械转正，逐字节拷贝〔md5 e9db7ba0…〕+ 头部核验 + py_compile PASS；82 行/3 592 B 在预算内） |
| `TRAP-vexp2-not-exact-pow2.py` | traps-compiler.md TRAP-vexp2-not-exact-pow2 | 陷阱：`T.vexp2`（= `exp(x·ln2)`）对整数指数与 `2.0**k` 不等复现 + 绕法通过（`T.vshl(1,k)` 对 k=0..14 逐位精确）；NPU 可运行 | 2026-10-14 fp8_quant Stage 3（evolver 机械转正，逐字节拷贝〔md5 f2456ec6…〕+ 头部核验 + py_compile PASS；75 行/3 398 B 在预算内） |
| `TRAP-transpose-ub-gm-roundtrip.py` | traps-compiler.md TRAP-transpose-ub-gm-roundtrip / TRAP-inner-axis-reduce-scalarize / TRAP-copy-layout-propagation-dma-amplification（一条 repro 覆盖三形态） | 陷阱：control（连续 DMA 5.4µs）/ +`vcast`（5.6µs，逐元素近免费）/ +`reduce_sum(dim=2)` 最内轴窄（155.1µs，GM_to_UB 36.75→336KB/block）/ +`T.transpose([2,1,0])`（138.2µs，UB_to_GM 48KB/block）四形态对照；可运行部分 = 三形态数值正确性 + 性能断言（reduce/transpose ≥10x control） | 2026-10-14 adaptive_avg_pool2d Stage 4（optimizer 任务内回写，Pattern-library Phase 4 例外授权；evolver 复核：头部条目 ID/origin_task/版本戳/断言说明齐备）、`py_compile` PASS、md5 `16fba364…`；130 行超 100 行预算按 PL-1.10 先例不裁剪披露 |
| `PL-1.23-flat-spatial-axis-reduce.py` | layout.md PL-1.23-flat-spatial-axis-reduce | 优化点 **delta 形态**：整平面归约慢/快两形态关键代码更改（`(CH,H,W)`+`dim=2/1` 双窄归约 → `(CH,1,H*W)`+单次宽归约）+ 机制归因见解摘要（→ TRAP-inner-axis-reduce-scalarize / TRAP-copy-layout-propagation-dma-amplification）；可运行部分输出与 torch 整平面均值一致 | 2026-10-14 adaptive_avg_pool2d Stage 4（同上；evolver 复核头部规范齐备）、`py_compile` PASS、md5 `2990334c…`；137 行同上披露 |
| `PL-1.4-single-wave-ch.py` | layout.md PL-1.4-tiling-heuristic（单波判据第五数据点） | 优化点 **delta 形态**：`_pick_ch_general` 波次判据最小形态（`units = N·out_H·ceil(C/CH)`；`units(base)·2 > vector_cores` 则不折半）+ 两 shape 分派对照；纯 Python、断言分派结果 = 实测最优档（无需 NPU） | 2026-10-14 adaptive_avg_pool2d Stage 4（同上；evolver 复核头部规范齐备）、`py_compile` PASS、md5 `b270417e…`；81 行 / 3 889 B 在预算内 |
| `CONST-copy-floor-dma-granularity.py` | constants.md CONST-copy-floor-dma-granularity | 常数：三级 DMA 粒度（d1_bm8 = 64KB 行块 / d2_bm16 = 128KB / d3_flat32 = 64KB 扁平）纯拷贝地板 + 标量消费者保活（避免 DCE）；**测量型**——可运行部分为语法/结构检查，msprof 命令与断言说明在头部 | 2026-10-15 argmin Stage 4（`scenario=optimize`；evolver 机械转正：源 `perf_opt/_argreduce_kernel_round1_probe4.py` md5 `f269b1ea…` → 逐字节拷贝 + **头部追加登记块**〔条目 ID / First verified 版本戳 / origin_task / 断言说明 / 测量命令〕→ 代码零改动）+ `py_compile` PASS；88 行 / 3 400 B 在预算内。**披露**：本 repro 无内联 assert（源为测量脚本，驱动入口 `bench_driver.py` 属任务工作区过程文件），登记形态 = 测量脚本 + 头部断言说明 |
| `CONST-aiv-time-decomposition.py` | constants.md CONST-aiv-time-decomposition | 常数：同 (2048,4096)/bm=2/grid=40 的逐 op 类边际代价标定（q0_copy 地板档 → 1/2/4 个 fp16 elementwise pass → fp16 reduce_min → fp32 cast+reduce → 双 reduce）；**测量型**——msprof 命令与断言说明（`Task Duration ≈ max(mte2, vec) + scalar`）在头部 | 2026-10-15 argmin Stage 4（同上；evolver 机械转正：源 `perf_opt/_argreduce_kernel_round1_probe2.py` md5 `38ab0aea…` + 头部登记块 + 代码零改动）+ `py_compile` PASS；131 行 / 5 618 B **超 100 行预算**（<8KB）按 PL-1.10 先例不裁剪披露；无内联 assert（测量脚本） |
| `TRAP-scf-for-row-offset-silent-miscompile.py` | traps-compiler.md TRAP-scf-for-row-offset-silent-miscompile | 陷阱：四形态对照（c1_2d 2D 融合 = bm≥2 标量化 268.8µs / **c2_perrow `for r in range(bm)` 写共享 (bm,N) 缓冲 = 静默错值**〔y0=−3 vs ref 3211〕/ c3_rowbuf 独立行缓冲正确但标量化 154.98µs / c4_chain 显式链正确 31.46µs）+ `torch.equal` 正确性读；NPU 可运行（`--level <name>`） | 2026-10-15 argmin Stage 4（同上；evolver 机械转正：源 `perf_opt/_argreduce_kernel_round2_structures.py` md5 `64cb15f4…` + 头部登记块 + 代码与断言零改动）+ `py_compile` PASS；231 行 / 10 885 B **超 100 行与 8KB 双预算**的极端形态——按 PL-1.10 先例不裁剪披露（裁剪会破坏四态对照的可运行性） |
| `TRAP-fp16-op-affine-double-rounding.py` | traps-runtime.md TRAP-fp16-op-affine-double-rounding | 陷阱（测试设计）：fp16 dtype affine（乘-舍-加-舍两次舍入）违反数 > 0 + fp32 affine 单次舍回 ×10 < trap（对齐通解）+ kernel 级单位 affine == 0（逐位）；**纯 torch CPU、无 NPU 依赖**（oracle/接口分层类陷阱第二例） | 2026-10-15 group_norm Stage 3（evolver 机械转正：源 `examples/group_norm/_group_norm_kernel/repro/` md5 `ff7d058e…` → 逐字节拷贝〔md5 一致，代码与断言零改动〕+ 头部核验〔条目 ID / Home file / 断言 / First verified 版本戳 / origin_task / Re-verify 齐备〕+ `py_compile` PASS + `/workspace/venv/bin/python` 实跑 PASS〔314 / 6 / 0 与源记录一致〕；99 行 / 4 823 B 在预算内） |
