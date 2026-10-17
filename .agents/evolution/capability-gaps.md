# 编译器能力缺口登记簿（Capability Gaps）

> 编译器层面自进化机制的登记簿。与 skill 层自进化（`queue.md` / pattern-library）平行：skill 层沉淀「如何在现有能力下绕」，本簿登记「npuir 缺什么能力」并驱动补齐。
>
> - **用途**：任务过程中识别出「当前 npuir 能力不足以实现所选算法」（无法实现 / 被迫换算法 / 触及性能上限）时，登记具体缺失能力、证据与绕法，跟踪复发次数与补齐状态。
> - **写入者**：任务各阶段 Agent（design 调研与 API 映射 / develop / optimize / review）在得出能力不足结论时**直接追加条目或为既有条目计数**；`tilelang-skill-evolver` 在任务终态蒸馏时查重、冲突消解与升级裁决（与 queue.md 的维护分工同口径）。
> - **证据规则**：与 AGENTS.md「Docs Auto Routing / Negative claims」同口径——须引用具体 `docs/` 路径与限制条款、`testing/`/`examples/` 佐证或复现命令；无法佐证时显式标注「未文档化假设 + 估算依据」。禁止凭记忆或 GPU 先验断言能力缺失。
> - **版本戳**：一切「能力缺失」结论绑定工具链版本（tilelang commit + CANN/设备版本）；工具链升级后相关条目自动待重验，不得沿用旧结论否决新设计。
> - **升级阈值**：同一缺口被 **≥2 个不同任务**独立识别 → `recurring`，**必须推动补齐**：产出能力补齐提案（`proposal` 字段），并在任务终态报告中向用户显式列出（与 Tier 1 的 2 次独立证据阈值同口径）。

## 字段速查

```yaml
gap_id: CG-{YYYY}-{NNNN}           # 唯一 ID，追加时按现有最大号递增
layer: Frontend API / TileLangIR pass / BishengIR / tladapter / runtime / codegen
capability: <缺失能力的具体描述：缺什么 API / 优化 / dtype / 布局 / 同步原语 / 性能上限，越具体越好>
blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>
evidence:                          # 证据链，规则同负向断言
  - <docs 路径#限制条款 / 复现命令 / 源码位置 / pattern-library 条目 / 实测数据>
workaround: <当前任务的实际绕法及其代价量化（如"两遍算法替代单遍，访存 +1.8x"），或 none（被迫弃用）>
occurrences: {n}                   # 独立任务识别次数（同一任务反复命中只计 1）
tasks: [<task_id>, ...]
toolchain_stamp: <tilelang commit + CANN 版本 + 设备，或 latest-rebuild>
status: open / recurring / proposal / in-progress / fixed / closed / withdrawn
proposal: <达 recurring 后必填：目标层 + 建议改动（API 签名 / pass 行为）+ 收益量化（引用各任务实测）+ 受影响算子清单 + issue 草稿（[npuir] 前缀）>
created_by: <task_id + 日期>
last_seen: <最近一次识别的任务与日期>
```

## 生命周期

```
open ──(≥2 独立任务识别)──> recurring ──(提案产出)──> proposal ──> in-progress ──> fixed ──> closed
  │                                                                            │
  └── withdrawn（证据被证伪/复核不成立）            fixed 后曾绕行的任务回访评估去绕化 ┘
```

- **追加即计数**：新识别的缺口先查重（本簿 + pattern-library §2），命中既有条目则 `occurrences+1` 并更新 `last_seen`/`tasks`，不新建重复条目。
- **推动补齐的形式**：提案正文落在条目 `proposal` 字段；对外建 issue（标题带 `[npuir]` 前缀，流程见 tilelang-github-operations skill）**前须用户批准**。
- **fixed 判定**：新工具链实测确认能力可用（附版本戳与验证记录）；fixed 后记录过 workaround 的任务可回访评估去绕化收益。
- **与 pattern-library 的边界**：绕法经验（怎么绕、绕的代价、陷阱）按 D 类进 pattern-library §1/§2；本簿只登记缺口本体与补齐跟踪，两处以 `gap_id` / 条目路径互相引用，不复制内容。

---

## Open

### CG-2026-0004

```yaml
gap_id: CG-2026-0004
layer: tladapter / codegen (load_nd2nz 区域描述符下推)
capability: >-
  T.load_nd2nz 的 src 区域若跨步非连续（BSHD 布局 [B,S,H,D] 张量按固定头取 (S,D)
  tile：区域 [1, real_m, 1, dim]，dim1 步长 H*D），后端按"从基址起的平坦连续内存"
  读取并静默产出确定性错误数据——不报错、不告警，数值表现为整块 S 矩阵乱值。
  即 load_nd2nz 仅支持张量布局的尾二维连续 tile（docs/Tilelang.language/内存操作/
  T.load_nd2nz.md §2.2.2 "src支持2-4D tensor" 未标注连续性约束）。同形态的
  T.copy（slice 写法）对相同跨步区域 bit-exact 正确，可作为绕法。
blocked_algo: >-
  BSHD 布局 attention 的 Cube 侧直接分块装载（q/k/v 按 (seq, dim) tile per head）；
  设计层按文档字面 "2-4D tensor" 假定跨步 tile 可用（_gqa_prefill_fwd_kernel
  DESIGN.md E2/E7 直连装载形态），实现期实测证伪，被迫换 T.copy slice 形态承载
  全部 GM→L1 装载（结构不变、语义等价、性能路径相同 PIPE_MTE2）。
evidence:
  - 复现：pattern-library repro/TRAP-load-nd2nz-strided.py（知识域，slice 绕法
    bit-exact 断言）；原 session 探针 probe_l1slot.py 为 provenance（session-local，
    flat-read 假设数值逐点复现 got[i,j] == flat_q[i]·flat_k[j]）
  - docs/Tilelang.language/内存操作/T.load_nd2nz.md §2.4 与
    docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.4 全部示例均为尾二维
    连续 tile（(b,n,s,d) 布局的 (s, block_d)）
  - examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py L83 的 4D 基址形态
    实为 dims 2-3 连续 tile（[heads, full_dim]），非跨步
workaround: >-
  GM→L1 全部改用 slice 形态 T.copy（highperf C1L/C2L 先例形态），跨步区域
    bit-exact 正确；代价：无额外拷贝（同为 PIPE_MTE2 单次搬运），仅依赖
    T.copy 而非文档指定的 load_nd2nz 分形通路（NZ 转换由后端隐式处理，实测
    gemm 结果正确）。
occurrences: 1
tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z]
toolchain_stamp: tilelang dev root build 2026-09-07 + CANN 8.5.0 + Ascend910B2C
status: open
created_by: multi_head_attention/_gqa_prefill_fwd_kernel Stage 3 attempt 3, 2026-09-07
last_seen: 2026-09-07
```

### CG-2026-0003 → 已升级 recurring（2026-10-16 occurrences 2：Expert 显式 `T.sync_block_set` 的 `syn_instr_mode` 属性为**第二形态**独立识别；条目正文在原位更新，status: recurring）

```yaml
gap_id: CG-2026-0003
layer: TileLangIR pass / codegen（Expert 模式 lowering 产出的新 HIVM 属性）与 BishengIR 版本配对
capability: >-
  **前端产物的新 HIVM 方言特性与已装 bishengir-compile 的版本配对缺失**（docs/ 无配对说明），
  两个独立形态：
  〔形态①（2026-09-07）〕新版 tilelang 前端（机器默认激活构建 /home/tilelang/j00919838/tilelang-mlir-ascend，
  2026-09-02 期）在 Expert 模式下把 L0C→GM 的 T.copy lowering 为 hivm.hir.fixpipe 并
  携带 dma_mode = #hivm.dma_mode<nz2nd> 属性；本机全部 3 个可用 bishengir-compile 均
  无法编译该 IR：CANN 8.5.0 版（0.1.0，e4e2ba9841d1，2026-01-16，PATH 默认）与 upload
  仓库 3rdparty 版（19.1.7，31f690369d，2026-07-24）解析期报 "unknown attribute
  `dma_mode` in dialect `hivm`"；j00919838 配套 3rdparty 版（1.1.0，139bb6c7aef6，
  2026-09-02）解析通过但在 ConvertHFusionToHIVMPass 段错误
  （mlir::utils::getAnnotateOpWithAttr，HFusionToHIVM.cpp）。
  〔形态②（2026-10-16，第二任务独立识别）〕Expert 核内**显式 `T.sync_block_set`** 的 codegen
  产出携带 syn_instr_mode 属性
  （hivm.hir.sync_block_set[... flag = %N syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>]），
  系统 CANN 9.0.0 的 `bishengir-compile 1.1.0`（/usr1/BiSheng/AscendNPU-IR/bishengir
  428ab8fdab46 2026-04-30）**解析期**报
  loc(kernel.npuir:50:72): error: Dialect `' not found for custom op 'syn_instr_mode'
  (tried '.syn_instr_mode' as well) → [ERROR] Failed to parse input file；
  而仓内**自建** 3rdparty/AscendNPU-IR/build/install/bin/bishengir-compile 能解析同一
  .npuir（越过 parse 阶段），但其后续向 /usr/local/Ascend/cann-9.0.0/bin/hivmc 发出的
  flag（--enable-cpu-trace-intrinsic=false / --enable-lir-compile=true）被 CANN 9.0
  hivmc 拒绝（Unknown command line argument）⇒ **两侧编译器组合均不可用**（前端产物新特性
  ↔ 已装后端解析能力之间无可用配对）。两形态的共同缺口本体（配对缺失、跨工具链可用编译器
  组合均不可用）在两次识别中一致，属性名/前端构建/编译器版本不同，故计为同一缺口的第二形态。
blocked_algo: >-
  〔形态①〕Expert 模式示例（examples/flash_attention/flash_attn_npuir.py，T.rs("PIPE_*") +
  T.sync_block_set/wait 跨核同步结构）在机器默认激活的 tilelang 构建下无法编译，
  迫使三实现性能对比任务整体切换到 upload 仓库的旧版前端构建完成。
  〔形态②〕**同包既有 Expert 产物在新工具链上不可编译**：CANN 8.5.0 时代交付的 fwd
  Expert 核 `GQAPrefillFwdKernel`（`examples/TileOPs/tileops/kernels/attention/
  multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`，含显式
  `T.sync_block_set`）在 CANN 9.0.0 / Ascend910B3 上无法编译 ⇒ MultiHeadAttentionBwdOp
  的「同文件追加」集成任务整体阻塞（op 级 BLOCKED_ENVIRONMENT，8/12 report 失败）；且
  MhaBwdWorkload.gen_inputs 依赖 fwd 运行产 `o` ⇒ bwd 用例连带不可达。受影响面 = 一切
  依赖该 fwd 核的算子（含本 op 的 fwd/bwd）与一切 Expert 显式同步原语产物。
evidence:
  - 复现（形态①，默认工具链）：PYTHONPATH 默认（tilelang 解析到 j00919838 checkout）下
    python flash_attn_npuir.py --seq_len 512 → bishengir-compile 报
    loc(kernel.npuir:41:42): error: unknown attribute `dma_mode` in dialect `hivm`；
  - 三编译器交叉实测（形态①，同一份 .npuir）：CANN 8.5.0 与 upload 3rdparty（2026-07-24）
    报同一 parse 错误；j00919838 3rdparty 1.1.0 在 ConvertHFusionToHIVMPass 崩溃
    （LLVM stack dump，栈顶 getAnnotateOpWithAttr）；
  - 绕法实测（形态①）：PYTHONPATH 前置 upload 仓库构建后（其前端不产出 dma_mode），同一源码
    用 CANN 8.5.0 bishengir-compile 编译运行通过（含精度自检 All check passed，
    seq_len=512/4096）。
  - 〔形态② 复现（2026-10-16）〕task multi_head_attention-MultiHeadAttentionBwdOp-
    20261016T015430Z Stage 5：report run.json 的 fwd 用例报
    `RuntimeError: NPU IR opt failed: loc("/tmp/.../kernel.npuir":50:72): error: Dialect `'
    not found for custom op 'syn_instr_mode'`（4 例 test_mha_fwd[*] + 4 例 test_mha_bwd[*]
    下游同源）；出处
    examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/
    integration_log.md §Root cause；report 侧 provenance（任务工作区，未上库，允许失效）= examples/TileOPs/reports/tileops/20261016_065819_471539_MultiHeadAttentionBwdOp/run.json。
  - 〔形态② 独立性证据〕① 回退本次 Stage-5 编辑（wrapper `__init__.py` + wrapper 还原到
    集成前状态）后重跑 `test_mha_fwd[smoke-*]` 复现同一失败 ⇒ 非本次编辑引入；② 该 fwd 的
    DESIGN/opt_log 工具链戳为 CANN 8.5.0 / Ascend910B2C，`reports/tileops/` 下无其历史
    report ⇒ 从未在当前工具链验证过；③ 自建编译器对照：3rdparty/AscendNPU-IR@31f690369
    自建 bishengir-compile 能越过 parse（.td 已带 syn_instr_mode），失败点下移到 CANN 9.0
    hivmc 的 flag 拒绝（env.sh 已记录该版本错配）；④ 同任务 bwd 三核独立确认可用
    （`_gqa_bwd_kernel.py --level L0` 16/16 PASS，844.7 s）⇒ 阻塞面确在 fwd Expert 核与
    工具链配对，非 kernel 逻辑。
  - 〔形态② 缓解路径实证（2026-10-16 Stage 5 重试段，task …082329Z）〕外部 shim
    落地（`tilelang/jit/jit_npu.py` 的 `linalg.replace("syn_instr_mode","sync_instr_mode")`）
    之后：共享测试文件 `tests/ops/test_multi_head_attention.py` **4/12 → 12/12 passed
    （8.12 s）**；单算子 report（任务工作区，未上库，允许失效）`examples/TileOPs/reports/tileops/
    20261016_082438_849619_MultiHeadAttentionBwdOp/run.json` 的
    `summary.correctness_passed=true`、per-operator rollup `MultiHeadAttentionBwdOp`
    **8/8 passed**（`avg_max_abs_err=1.158e-3` / `max_abs_err=1.95e-3`）；benchmark 20 条
    请求 / 18 条记录 / 2 条失败（2 条失败为 shape 相关后端段错误，见 CG-2026-0030）。
    出处（任务工作区，未上库，允许失效）：`examples/multi_head_attention/{RETROSPECTIVE.md, SESSION_TIMING_ANALYSIS.md}` +
    集成副本 `integration_log.md` §Stage 5 Retry #1（任务工作区 / 集成副本，未上库，允许失效）。
    缓解路径代价量纲：一行字符串重写、无运行时开销；适用域 = 「前端产物拼写与已装后端接受的
    拼写不一致」方向（不覆盖新属性类形态①）。
  - 未文档化假设：docs/ 未找到 bishengir-compile 与前端产物的版本配对说明；估计依据
    为两任务的编译器交叉实测。
workaround: >-
  〔形态①〕PYTHONPATH 前置 upload 仓库 tilelang 构建（旧版前端，fixpipe 不带 dma_mode），
  三种实现（Expert npuir / Developer baseline / Developer gqa）统一在其下编译对比；
  代价：任务级工具链钉版本，机器默认 tilelang 构建的 Expert 路径完全不可用。
  〔形态②〕**2026-10-16 重试段更新：已找到廉价上游绕法（缓解路径）**——生成 IR 时做一次
  字符串重写 `syn_instr_mode` → `sync_instr_mode` 再交给 `bishengir-compile`
  （落点 `tilelang/jit/jit_npu.py::_npuir_to_bin_enable_npu_compile`，本任务由外部落地，
  不在 agent 编辑域内；成本 ≈0 / 一行 `linalg.replace`）。该 shim 落地后共享测试文件
  `tests/ops/test_multi_head_attention.py` 由 4/12 → **12/12 passed（8.12 s）**，单算子
  report correctness 12/12、per-operator rollup 8/8。**缓解路径的适用边界**：只在
  「前端产物多写一个拼写、已装后端只认修正拼写」这一方向上成立（即修的是**名字配对**），
  不覆盖形态①（`dma_mode` 属后端不认识的**新属性**，字符串重写无解）；3rdparty pin 更新到
  含该改名修复的线（`origin/master` 已含）之后应移除该 shim。**缺口本体不变**：版本配对
  仍未被文档化、无机械预检手段。
  〔首轮结论（历史）〕形态②曾记 `none`：fwd 核属「add-only 保留的既有交付件」、fwd 依赖
  属禁止编辑的 workload 接线（`tileops/workloads/attention.py`）、工具链配对属环境层 ⇒
  首轮按失败分类表记 `BLOCKED_ENVIRONMENT` 并终止（未消耗 debug attempt）；当时的备选正向
  路径 = 在 CANN 8.5.0 或与前端匹配的 `bishengir-compile` 上集成（与形态①同一类工具链钉
  版本），或待上游把 `syn_instr_mode` 支持回灌进 CANN 9.0 的 `bishengir-compile`。
occurrences: 2
tasks: [FA 三实现性能对比 baseline/npuir/gqa 2026-09-07, multi_head_attention-MultiHeadAttentionBwdOp-20261016T015430Z]
toolchain_stamp: >-
  形态①：tilelang 默认激活构建 j00919838 checkout（jit_npu.py dacebc3d 2026-08-05，3rdparty
  bishengir 1.1.0 2026-09-02）+ CANN 8.5.0 bishengir-compile 0.1.0（2026-01-16）+
  910B2C(IT21HMDB01-B2) + torch_npu 2.7.1。
  形态②：tilelang 0.1.2 dev root build /workspace/tilelang-mlir-ascend（3rdparty/AscendNPU-IR
  @31f690369）+ 系统 bishengir-compile 1.1.0（428ab8fdab46 2026-04-30）+ CANN 9.0.0
  （/usr/local/Ascend/cann-9.0.0，hivmc）+ Ascend910B3 / 2026-10-16。
status: recurring
proposal: >-
  目标层 BishengIR（bishengir-compile / hivmc 解析与 flag 兼容）+ 版本配对文档化。
  建议改动（三选一或组合）：① **把前端新产出的 HIVM 属性回灌进 CANN 发行版 bishengir-compile**
  （`syn_instr_mode` / `dma_mode` 及后续同名族），使「系统 PATH 编译器」与当前 tilelang 前端
  能配对；② 至少把**配对关系文档化**——在 `docs/developer/EnvironmentVariables.md` 或
  `env.sh` 的 TILELANG_NPU_COMPILER_PATH 段给出「tilelang commit ↔ 3rdparty AscendNPU-IR
  commit ↔ CANN bishengir-compile 版本 ↔ hivmc flag 集」的对应表与不匹配时的**可诊断报错**
  （现状是 parse 阶段 `Dialect '' not found` 或 hivmc `Unknown command line argument`，
  两处都不指出「版本错配」这一根因）；③ 修 hivmc 侧的 flag 兼容（自建 bishengir-compile
  发出的 `--enable-cpu-trace-intrinsic=false` / `--enable-lir-compile=true` 在 CANN 9.0
  hivmc 上被拒 ⇒ 目前「自建前端 + 系统后端」组合亦不可用）。
  收益量化（两任务实测）：形态①使整个三实现性能对比任务的 Expert 路径不可用、被迫全局切换
  前端构建；形态②使一个已交付 Expert 核在新工具链上不可编译，导致 bwd 迁移任务 op 级
  BLOCKED_ENVIRONMENT（report 12 例中 8 例失败，其中 4 例为其下游），**纯工具链侧成本、
  无 kernel 缺陷**；且「既有 CANN 8.5.0 时代 Expert 产物在新 CANN 上是否可用」此前无任何
  机械预检手段（本次靠 20 min 的 git 回溯 + 双编译器对照才定征）。
  受影响算子清单：attention 族 Expert 产物（`_gqa_prefill_fwd_kernel` 及其全部下游，含本
  op 的 fwd/bwd）、及一切使用显式 `T.sync_block_set/wait` 或 fixpipe `dma_mode` 的 Expert
  kernel（即跨 CANN 大版本搬迁时的存量 Expert 全集）。
  issue 草稿（[npuir] 前缀，**须用户批准后建**）：[npuir] Expert 前端产物与 bishengir-compile
  存在未文档化的版本配对约束（`dma_mode` / `syn_instr_mode` 两属性在 CANN 8.5/9.0 各自的
  已装编译器上均解析失败；自建编译器与 CANN 9.0 hivmc 亦 flag 不兼容）
created_by: FA 3-impl perf comparison 2026-09-07
last_seen: multi_head_attention-MultiHeadAttentionBwdOp-20261016T015430Z Stage 5 集成期（2026-10-16，第二形态独立识别 → 升级 recurring）；同上任务 Stage 5 重试段 task multi_head_attention-MultiHeadAttentionBwdOp-20261016T082329Z（2026-10-16，缓解路径实证 4/12 → 12/12，workaround 字段已回填）
```

> **升级依据（2026-10-16）**：task `FA 三实现性能对比 baseline/npuir/gqa`（2026-09-07，形态① `dma_mode`/fixpipe）+ task `multi_head_attention-MultiHeadAttentionBwdOp-20261016T015430Z`（2026-10-16，形态② `syn_instr_mode`/`sync_block_set`）——两个**不同任务、不同属性、不同前端构建与编译器版本**独立识别同一缺口本体（前端新 HIVM 特性 ↔ 已装 bishengir-compile 无可用配对，且两任务下「本机全部可用编译器组合」均不可用），达 ≥2 任务阈值。proposal 字段已产出，待向用户显式列出并推动（外部 issue 须用户批准后建）。
>
> **缓解路径回填（2026-10-16 Stage 5 重试段）**：形态② 获**廉价上游绕法**（emit 侧一行字符串重写，`tilelang/jit/jit_npu.py`，本任务外部落地）——落地后共享测试 4/12 → 12/12。**该进展不撤案**：缺口本体（版本配对未文档化 / 无机械预检）不变，且**形态①（新属性 `dma_mode`）无字符串绕法**，proposal 的 ①②③ 三条建议均仍成立；优先级排序调整 = 「② 配对关系文档化 + 机械预检」上升（对两类形态都有效）、「① 属性回灌」维持、「③ hivmc flag 兼容」维持（自建前端 + 系统后端组合仍不可用）。重验触发词：`syn_instr_mode` 直通编译成功（即系统 `bishengir-compile` 开始接受该拼写）时按证伪协议刷新本条目。

---

### CG-2026-0002 → 已升级 recurring，条目移入下方 Recurring 区（2026-09-20 occurrences 2：Expert T.copy L1→L1 第二形态独立识别）

### CG-2026-0001

```yaml
gap_id: CG-2026-0001
layer: codegen (target.build.tilelang_npuir_dev, Developer mode)
capability: >-
  条件构造在 T.Pipelined 循环上下文中使 Developer 模式 codegen 崩溃（SIGSEGV，无诊断输出）：
  (a) Pipelined 体内直接出现运行时（TIR 条件）if 语句——即使条件为平凡比较
  （if k_idx == 0:）也复现；(b) T.Parallel（位于 Pipelined 体内）的循环体中出现
  比较式的 `&` 合取或 T.if_then_else 表达式。简单单比较的语句 if/else 与嵌套
  单比较 if 在 T.Parallel 体内可正常编译运行。
blocked_algo: >-
  flash-attention 类 kernel 的块级快速路径（运行时条件跳过 mask 预填，DESIGN R6/R-3）
  与多条件 mask 的紧凑写法不可用——被迫回退为逐块无条件 mask 预填（源 GPU kernel
  同构路径，正确性无虞，mask 预填从约 6-25% 块升至 100% 块，性能损失记入 tradeoff）。
evidence:
  - 复现（本任务 Stage 3；examples/multi_head_attention/_gqa_prefill_fwd_kernel 任务工作区未上库，provenance 允许失效）：
    变体探针 /tmp 之外已固化于 history_version 与 debug_log.md 的探针矩阵：
    E1 简单单比较 if（编译通过）/ E2 `&` 合取（SIGSEGV exit 139）/ E3 局部变量+单比较 if
    （通过）/ E4 T.if_then_else（SIGSEGV）/ E7 嵌套单比较 if×3（通过+精度通过）/
    variant_D `if k_idx == 0:` 直接位于 Pipelined 体（SIGSEGV）。
  - 崩溃栈：faulthandler 显示 segfault 于 tilelang/engine/lower.py:180
    device_codegen -> target.build.tilelang_npuir_dev（C++ FFI）。
  - 过时先例：examples/deepseek_v4/example_sparse_attn_kernel.py L71 `if n == 0:`
    位于 T.Pipelined 体内（当前工具链不可编译，疑似未在 CI 中执行）。
  - 同族补充证据（同任务后续用例）：
    (c) T.Parallel 体内对 shared buffer 的谓词标量写（if j >= real_n: v_shared[j,d]=0）
    使 BishengIR root-alloc pass 报错（'tensor.insert_slice'/'hivm.hir.vcast' op
    Unsupported op for finding the root alloc）→ 编译失败（seq_len 16 探针）；
    (d) T.Parallel 体内条件读 fragment（if acc_s==−inf: … else: acc_s=tmp_s2[i,j]，
    softcap −inf 保持）编译通过但运行期 fixpipe 路径损坏 → aicore 设备崩溃
    （"The MPU address access is invalid", fixp_error，softcap 探针）。
  - 未文档化假设：docs/Tilelang.language/ 无 T.Pipelined/T.Parallel 条件构造限制条款
    （遍历 同步管道操作/数学操作 目录未见相关文档）；估计依据为本任务探针实测。
  - Stage 4 同任务补充（2026-09-07，tuning 阶段）：(e) 跨块计算重叠结构
    （gemm1(j+1) 与块 j 的 v-chain 重叠，需 2 个 acc_s fragment 按迭代奇偶交替，
    运行时奇偶条件选择 fragment / 变量下标缓冲）同属本崩溃类，未敢实施
    （运行时条件 fragment 访问 = (d) 的已知崩溃形态）；评估记录于
    perf_opt/opt_log.md Round 7 blocked 候选。
workaround: >-
  mask 用嵌套单比较运行时 if/else 语句（sparse_mla_fwd.py L119-123 模式）重写；
  块级快速路径按 DESIGN §9.2 R-3 预授权回退放弃（无条件 mask 预填，源同构）。
  (c) 的绕法：shared buffer 清零改用无条件整 buffer T.vbrc（sparse_attn L81 模式）；
  (d) 的绕法：softcap −inf 保持改为纯向量算术等价变换
  kept = capped + (acc_s − clamp(acc_s))（有限值加 0 精确、−inf 加 −inf 保持），
  且 tanh 前先 clamp 使输入恒有限（不依赖硬件 tanh(±inf)）。
  Stage 4 可用 vselect/vcmp 广播构造替代（R-4 优化路径，需先例确认）。
  Stage 4 实测更新（perf_opt/opt_log.md Round 1）：无需 vselect/vcmp——mask 消除的
  正确形态是「工厂期 trace-time 分支特化」（clean_path = not is_causal and
  Sq%bm==0 and Skv%bn==0 为纯 Python 常量条件，静态消解，不产生运行时 if），
  non-causal 整除 shape 的 mask 预填整体删除 + gemm initC=True，Task Duration
  -73~75%（123.7→16.1us @fa512 等 4 case），位级等价（0+QK == QK）。
  NK==1 退化（single_block）同理特化（alpha=e^-inf=0 恒等，bit-exact）。
  Expert 模式结构级绕法（2026-09-07 expert 重开发任务实证，
  multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z）：双 Scope
  （Cube/Vector）+ GM workspace 多槽 + per-slot flag + T.rs 流内运行时 if 为合法
  形态，(a)/(b)/(e) 类阻塞结构在 Expert 下全部可表达——expert 重写后长 KV
  1.57–1.63× vs developer 基线（pattern-library §1.7 / §4 expert attention
  案例行）；Developer 阻塞时先评估 Expert 形态（queue VP-2026-0013）。
occurrences: 1
tasks: [multi_head_attention/_gqa_prefill_fwd_kernel Stage 3 first_impl]
toolchain_stamp: tilelang 67db6f3 (2026-09-04) + CANN 26.0.rc1 + 910B2C(IT21HMDB01-B2) + torch_npu 2.7.1
status: open
created_by: multi_head_attention/_gqa_prefill_fwd_kernel s3 attempt1 2026-09-04
last_seen: multi_head_attention/_gqa_prefill_fwd_kernel s4 tune 2026-09-07
```

### CG-2026-0005

> **第三轮注记（2026-09-09，task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）**：不阻塞目标达成——两相位重构（DESIGN §11.3）已在该缺口存在下四硬目标全部达成（12.94/18.11/30.88/98.05µs）；本条目描述的是两相位结构自身的 ~99µs 量级边界，非达标障碍。

```yaml
gap_id: CG-2026-0005
layer: Frontend API / codegen
capability: >-
  L0C（Cube 侧 gemm 累加器）到 Vector 侧 UB/L1 无直接搬运通路：T.store_fixpipe 的
  dst 仅允许 GM（docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1「目标
  张量（必须来自 GM 地址空间）」），Cube→Vector 数据必须经 GM workspace 往返
  （fixpipe 写 GM + 对端 MTE2 读回）。跨引擎协作算子中该往返流量与算法中间量
  同阶（attention 的 S/P 矩阵 S²×dtype×2 方向），且占用 L1/L2 fabric 带宽。
blocked_algo: >-
  attention 族跨引擎流水（Cube 双 gemm + Vector softmax）的 S/P 传输：性能上限
  ——fa4096 (S=4096,D=128) S/P w+r = 128MB 固定流量（f16 双向），是 2.27x 硬目标
  缺口的第二大构成项（详见 examples/TileOPs/tileops/kernels/attention/ 任务工作区
  内 perf_opt/perf_feedback.md——未上库，provenance 允许失效；结论已自包含于本条）。
evidence:
  - docs/Tilelang.language/内存操作/T.store_fixpipe.md §2.1（dst 仅 GM 的文档限定）
  - perf_opt/profiles/round8/recheck_fa4096（Memory CSV：cube 读写 ws 流量实测）
  - perf_opt/opt_log.md 第二轮 round 8 Diagnosis 第 5 条（流量算术）
workaround: >-
  S/P 传输 dtype 减半（f16，round 5 裁决 + round 9 f16 softmax 链，合计 -3%~-14%
  wall）；无通路级绕法。
occurrences: 1
  tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z]
toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
status: open
created_by: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
```

### CG-2026-0006

> **第三轮注记（2026-09-09，task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）**：不阻塞目标达成——两相位重构（DESIGN §11.3）已在该缺口存在下四硬目标全部达成（12.94/18.11/30.88/98.05µs）；本条目描述的是两相位结构自身的 ~99µs 量级边界，非达标障碍。

```yaml
gap_id: CG-2026-0006
layer: TileLangIR pass / codegen（向量算子融合与 f16 打包发射）
capability: >-
  向量算子按独立指令发射、无相邻融合（vcast/vmul/vsub/vexp/reduce 链每 op ~0.5µs
  固定发射开销主导），且 f16 与 f32 逐元素速率相同（无 f16 打包/SIMD 加速）——
  向量链时间 ∝ op 数 × 元素数，与 dtype 字节宽无关。
blocked_algo: >-
  窄 tile 向量链（[22,256] 级 softmax pass）：同元素总量下块数翻倍使向量时间翻倍
  ——fa4096 深流水变体（bn=256）vec-active 90→180µs、wall +12% 回退（vs bn=512），
  封死"窄块深流水"结构方向。
evidence:
  - perf_opt/profiles/round10/v10dp_fa4096_bm44_bn256_ns2（aiv_vec_ratio 70.7% 实测）
    vs perf_opt/profiles/round9/v9_fa4096（39.5%，同元素量 bn=512）
  - perf_opt/opt_log.md 第二轮 round 10「回归根因」段（发射开销定律推导）
workaround: >-
  宽块 op 优先（bn=512 摊减发射次数）；删除纯 dtype 搬运 pass（f16 链消 vcast×2）。
occurrences: 1
  tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z]
toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
status: open
created_by: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
```

### CG-2026-0007

> **第三轮注记（2026-09-09，task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）**：不阻塞目标达成——两相位重构（DESIGN §11.3）已在该缺口存在下四硬目标全部达成（12.94/18.11/30.88/98.05µs）；本条目描述的是两相位结构自身的 ~99µs 量级边界，非达标障碍。

```yaml
gap_id: CG-2026-0007
layer: runtime / codegen（跨引擎同步原语粒度）
capability: >-
  T.sync_block_wait 表现为阻塞整条后续发射流（非窄管道门控）：流水化只有在 wait
  到达时必然已满足（深 deferral）才有效，而深 deferral 要求更多 L1 槽位（与
  cbuf 512KB 容量冲突）；且 per-block flag id 预算上限 15（DESIGN §7.2），多槽
  （≥16）workspace 协议无法表达。缺少「per-pipe 门控 wait」与「计数器/批量
  信号量」原语。
blocked_algo: >-
  跨引擎块级深流水（attention C1/V1/C2/V2 重叠）：实测 ns=1≈ns=2（±0.5%）、
  wide≈serial、defer-2 深流水在 bn=512 因 L1 槽容量（K/V 双槽 567KB>512KB）不可
  行、bn=256 因 CG-2026-0006 回退——busy 核 aic_scalar 40–58% flag 自旋成为
  结构性时延地板。
evidence:
  - perf_opt/opt_log.md 第一轮 round 3/4（ns 平区）+ 第二轮 round 10（v10_dp 回退）
  - perf_opt/profiles/round10/v10dp_*（aic_scalar 50.7→40.5%：深 deferral 部分
    解锁发射流的直接证据）
  - /tmp/opencode/probe_kvbw.py 探针四模式（session-local；结论镜像于 opt_log）
workaround: >-
  深流水仅在 wait 预满足深度可行（本轮 v10_dp 证实方向有效但被 L1/向量上限抵消）；
  宽块串行为实证最优折衷。
occurrences: 1
  tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z]
toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
status: open
created_by: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z expert 谱系第二轮续调 (2026-09-09)
```

### CG-2026-0009

```yaml
gap_id: CG-2026-0009
layer: runtime / codegen（MTE2/MTE3 引擎边界同 buffer 访问顺序原语）
capability: >-
  同一 UB buffer 跨 DMA 引擎复用时（MTE2 装载写 + MTE3 store 读），指令流内无原语
  表达「MTE3 读完成前不发射 MTE2 写」的 WAR 顺序约束：跨引擎（Cube↔Vector）的
  T.sync_block_set/wait flag 机制不覆盖该边界；自等自身 flag 在双 AIV gather 语义下
  有死锁风险；重复 buffer（第二份 ND staging）是唯一可靠绕法但受 UB 容量约束
  （attention 第六轮实证：(80,256) 贴限下 +20KB 即爆 192KB/AIV）。与 CG-2026-0007
  （跨引擎同步原语粒度）同属同步原语覆盖缺口族——本条在 MTE2↔MTE3 引擎边界，
  0007 在 Cube↔Vector flag wait 粒度。
blocked_algo: >-
  两相位 persistent 结构 Vector 侧深度 2 任务流水（p1(T+1) 提前后其 S load〔MTE2 写
  ub_f16_ND〕与 p1(T) 尾块 P store〔MTE3 读同 buffer〕背靠背）——flaky aicore
  timeout（AIVector+MTE error，L0 单跑 PASS / --level all 挂）。该形态是 35% roofline
  Ratio 目标的主要结构缺口（short 域 per-task 固定成本 8.05µs 中引擎 busy 仅 ~3.5µs，
  跨引擎 flag 依赖链 + 引擎边界顺序约束主导），第六轮判定 blocked。
evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：opt_log
    Round 8 r8g（精度失败两连：legacy 域单 flag-id 空间下 advance 消费 task T 的
    O-ready 计数读未写数据 L1 全挂 + pipe2 域 flaky timeout；根因归因与三条修复
    路径评估）+ perf_records.jsonl round 8 留档
  - pattern-library/attention.md PL-1.12「Vec 侧同构改造 blocked」段（引擎独占
    buffer 为流水深度 >1 的隐含前提；r7f 中该 WAR 被 p2/epi 时间距离掩盖的对照）
  - 未文档化假设：docs/Tilelang.language/ 无 MTE2/MTE3 引擎边界同 buffer 顺序约束
    条款；估计依据为本任务 flaky timeout 实测 + 排除法（时间距离掩盖 / 双缓冲
    UB 不可行 / flag 自等死锁风险）
workaround: >-
  保持单深度形态（r8h 终版：p2/epi 的时间距离天然掩盖该 WAR）；流水深度 >1 要求
  「引擎独占 buffer」（每引擎专属 staging，代价 = UB 需求翻倍）。
occurrences: 1
tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z]
toolchain_stamp: tilelang 0.1.2+a13585dc / CANN 8.5.0 / Ascend910B2C / 2026-09-16
status: open
created_by: multi_head_attention/_gqa_prefill_fwd_kernel 第六轮 Stage 4（distill 登记），2026-09-16
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 第六轮 Stage 4（2026-09-16）
```

### CG-2026-0013

```yaml
gap_id: CG-2026-0013
layer: Frontend API（TILELANG_ASCEND_MODE 作用域）
capability: >-
  编程模式（Expert / Developer）只能经进程级环境变量 TILELANG_ASCEND_MODE 设置，
  无 per-kernel / per-jit 作用域参数：三处实时读 os.environ——
  tilelang/language/customize_npuir.py:886（前端 trace，Scope/alloc 形态分派）、
  tilelang/engine/lower.py:170（device_codegen）、tilelang/jit/jit_npu.py:1789
  （--disable-hivm-tensor-compile 旗标）。docs/developer/EnvironmentVariables.md L31
  仅登记环境变量一种设置方式，docs/Tilelang.language/ 无 per-kernel 模式 API
  （全文无 ASCEND_MODE 条目）。后果：同一进程混编两种模式的 kernel 套件时，
  后编译者被先入者锁定的模式污染。
blocked_algo: >-
  单 pytest 进程跑混合模式算子套件（TileOPs tests/ops/：mha=Expert 两相位
  Cube/Vector 结构 vs ada_layer_norm/mish/logsumexp=Developer 向量链）——Developer
  赢则 Expert kernel 报 'hivm.hir.store' op only support copy gm to ub or copy ub
  to gm or copy ub to ub（mha full-fwd-bf16 变体更在 lower.py:180 device_codegen
  原生段错误）；Expert 赢则 Developer kernel 报同类 store 约束错（44 case 失败）。
evidence:
  - 复现（tilelang 0.1.2+1990aa9 / CANN 8.5.0 / Ascend910B2C，2026-09-18）：
    TILELANG_ASCEND_MODE=Developer pytest tests/ops/ → mha 失败+段错误（core dump）；
    TILELANG_ASCEND_MODE=Expert pytest tests/ops/ → 44 failed / 22 passed
    （ada/mish/logsumexp 全挂，mha 过）；不设环境变量 → import 顺序决定胜负
    （pytest 参数形式〔目录 vs 整文件 vs node ID〕改变 import 顺序，两次同参运行
    结果可反转——ada 先 import 则 mha 挂，mha 先 import 则 ada/mish 挂）。
  - tileops 侧 import 竞争现场：kernel 模块级 os.environ.setdefault
    （_gqa_prefill_fwd_kernel.py:92/126=Expert；_ada_layer_norm_kernel.py:86/130、
    mish perf_opt/mish.py:57、logsumexp×2=Developer），setdefault=先 import 者赢。
  - 源码三读取点（上文 layer 字段）；docs/developer/EnvironmentVariables.md L31。
workaround: >-
  tileops 级：kernel 模块 import 时 setdefault（现状，单算子进程正确、混合进程
  按 import 顺序随机胜负）；或每次 compile 前后显式 set/restore 环境变量（待实施，
  代价 = 每个 kernel 工厂包一层 env 作用域）。测试级绕法：按模式拆分 pytest 调用。
  根治需 tilelang 提供 per-kernel 模式参数（如 tilelang.jit(..., mode="Expert")）。
occurrences: 1
tasks: [tileops-tests-ops-mode-battle-20260918]
toolchain_stamp: tilelang 0.1.2+1990aa9(452f447 build) / CANN 8.5.0 / Ascend910B2C / 2026-09-18
status: open
created_by: tileops tests/ops/ 混合模式失败排查（conductor 会话），2026-09-18
last_seen: tileops-tests-ops-mode-battle-20260918（2026-09-18）
```

### CG-2026-0010

```yaml
gap_id: CG-2026-0010
layer: codegen (Developer mode, persistent + gemm 混排)
capability: >-
  Developer 模式（auto CV-split）对 persistent 分核结构（T.Kernel(N, is_npu=True)
  核内 T.serial 多任务）与 T.gemm + v-prefix 向量 op 的混排不支持：运行时崩溃
  "Illegal instruction, which is usually caused by unaligned UUB addresses"
  （vector core exception，retCode=0x31），无编译期诊断。Expert 模式（显式双
  Scope）同结构完全正常。docs 无 Developer/Expert 适用范围条款（未文档化假设：
  依据双模式 repro + 全仓 24 处 is_npu=True 用例扫描——Developer 仅用于非
  persistent 网格）。
blocked_algo: >-
  persistent 混合算子（Mamba-2 SSD chunk scan 类：Cube 双 gemm 路径 + Vector
  因子链）在 Developer 模式不可实现——本任务 user_requirement 指定 Developer
  模式，被迫切换 Expert（GQA/sparse_mla 先例）完成。
evidence:
  - 复现：pattern-library repro/TRAP-DEVMODE-PERSIST-GEMM.py（知识域，Expert
    数值断言 + Developer 崩溃双模式对照）；原件
    examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/repro/DEVMODE_PERSIST_CRASH.py（溯源，任务工作区允许失效）
  - 仓库实态：Developer 仅 non-persistent 先例（flash_attn_npuir_dev.py /
    fp8_lighting_indexer.py），persistent 混合算子全部 Expert（GQA /
    sparse_mla_fwd_exp.py）
  - 任务档案：examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/RETROSPECTIVE.md（溯源，允许失效）——Stage 3 章节（崩溃日志 + 模式切换链）
workaround: >-
  Expert 模式显式双 Scope（T.Scope("Cube")/("Vector") + alloc_L1/alloc_L0C/
  alloc_ub + sync_block_set/wait + pass_configs 关闭
  TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION）——完全绕过、无性能代价
  （pattern-library attention.md PL-1.16 / traps-runtime.md
  TRAP-DEVMODE-PERSIST-GEMM）。
occurrences: 1
tasks: [ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z]
toolchain_stamp: tilelang 0.1.2+1990aa9fe4 + CANN 8.5.0 + Ascend910B2C / 2026-09-17
status: open
created_by: ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel Stage 3（distill 登记），2026-09-17
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 3（2026-09-17）
```

### CG-2026-0011

```yaml
gap_id: CG-2026-0011
layer: Frontend API（Cube gemm dst 形态 / L0C 累加原语）
capability: >-
  T.gemm 的 dst 不支持 L0C region 写（单 [Q,bp] 大 acc 与 per-lt band gemm 的
  [64,64] 小 dst 不可共用同一 L0C 张量的不同区域），且无「两 L0C acc 相加」
  原语——跨 lt 的累加结果无法在 L0C 域合并，必须经 GM/UB 中转。（未文档化
  假设：docs/Tilelang.language/线性代数操作/T.gemm.md 无 region dst 条款，
  依据本任务优化候选的结构分析。）
blocked_algo: >-
  SSD chunk scan history 路径的 M=256 合并 gemm（1 读 1 gemm/任务，消除
  per-lt gemm 碎片）不可实现——保持 per-lt [64,64] band gemm 分块形态，
  gemm 指令数 4/任务无法进一步合并（该候选列为 blocked，opt_log §5）。
evidence:
  - examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/opt_log.md §5（溯源，允许失效）——blocked 候选清单（history M=256 合并 gemm 行：API 阻塞点 + dst 形态分析）
workaround: >-
  保持 per-lt band gemm 分块（每 lt 单条 gemm，ws 写侧块连续布局配合——
  elementwise.md PL-1.14 两全形态）；性能缺口由深度 2 任务流水与 AIV 分片
  在其他维度补偿（终局 2.91×）。
occurrences: 1
tasks: [ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z]
toolchain_stamp: tilelang 0.1.2+1990aa9fe4 + CANN 8.5.0 + Ascend910B2C / 2026-09-17
status: open
created_by: ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel Stage 4（distill 登记），2026-09-17
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 4（2026-09-17）
```

### CG-2026-0012

```yaml
gap_id: CG-2026-0012
layer: BishengIR（auto-multi-buffer pass，动态 subview 支配性）
capability: >-
  task 级装载的 UB 行 buffer（如 [1,Q] 因子行）在内层循环以动态偏移切片消费
  时，--enable-auto-multi-buffer=true 产出非支配 IR：error: operand #2 does
  not dominate this use（Q≥128 切片非全宽触发；Q=64 全宽不触发）——合法的
  UB→UB 动态 extent copy 形态被 pass 破坏支配性。
blocked_algo: >-
  「task 级 dA/dt 行装载」优化（省 12 次 tiny GM 读/任务）不可实现——被迫
  退回 per-block GM 直载（基线形态）；该候选与 UB 容量候选同列 blocked
  （opt_log §5，v3 分支双重 blocked 实证）。
evidence:
  - examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/logs/round1/v3_hoist_L0.log（溯源，允许失效——报错全文）
  - pattern-library traps-compiler.md TRAP-UB-dynsubview-dominance（形态与
    绕法三则 + 知识域 repro/PL-1.13 关联骨架）
workaround: >-
  ① per-block GM 直载；② 切片偏移静态化（trace 分派）；③ 关
  auto-multi-buffer 未验证（全局改变 L1/UB 多缓冲行为，风险自担）。
occurrences: 1
tasks: [ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z]
toolchain_stamp: tilelang 0.1.2+1990aa9fe4 + CANN 8.5.0 + Ascend910B2C / 2026-09-17
status: open
created_by: ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel Stage 4（distill 登记），2026-09-17
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 4（2026-09-17）
```

### CG-2026-0014

```yaml
gap_id: CG-2026-0014
layer: TileLangIR pass（向量化/分块）
capability: >-
  multitile（num_tiles>1 的 T.serial tile 循环）+ bm>1 组合触发编译器 SIGSEGV
  （rc 139，无报错文本）。触发矩阵：bm=1 任意 tile 数 ✓ / bm>1 单 tile ✓ /
  bm>1 多 tile ✗。本条目按 DESIGN.md §9.4-R9 补录（前次任务已用该 ID 登记
  但条目丢失；源 repro .py 已清理仅存 repro/__pycache__ pyc）。
blocked_algo: >-
  argmax/_argreduce_kernel tiled 路径被迫 bm=1（DESIGN §5.2 强制）——lm-head 类
  小 M 负载核利用率被钉死在 M/48（4/48），是 C6 N-split 结构（本库 PL-1.20）
  的存在动因；Stage 4 实测翻转后 lm-head 2.20x/2.60x。
evidence:
  - repro/__pycache__/TRAP-multitile-bm-gt1-segfault.cpython-311.pyc（前次任务，
    2026-09-23；触发矩阵见 DESIGN.md examples/argmax/_argreduce_kernel §9.1-C-5）。
  - DESIGN.md §9.1-C-5 / §9.2-R3（本任务引用其结论未复现崩溃本身）。
  - 〔2026-10-16 本工具链**不复现**（适用域收窄）〕count_nonzero-_logical_reduce_kernel-20261016T005019Z
    Stage 1 编译期探针：列分块形态 + `bm>1`（fp16 `2×4096`×4 tile / `8×512`×32 tile / fp32 `2×1024`×16 tile）
    在 tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892 + CANN 9.0.0 + Ascend910B3 上
    **全部编译通过**（结论落笔于该任务 DESIGN.md §9.4）。⇒ 本条目适用域限定为**原工具链**
    （dev root build 2026-09-24 / CANN 8.5.0 / Ascend910B2C）；另一工具链上的可编译性单独立论，
    旧结论不构成对新设计的否决（版本戳纪律）。**occurrences 不计**（本条非新识别，属复核记录）。
    provenance（任务工作区，未上库，允许失效）：`examples/count_nonzero/_logical_reduce_kernel/{DESIGN.md#§9.4, history_version/design_probe_tileloop.py}`
workaround: tiled 路径强制 bm=1（工厂期保证）；小 M 大 N 负载改 N-split 两段 kernel。
occurrences: 1  # 前次任务识别；本链 Stage 1（DESIGN §9.4-R9）发现登记丢失并补录、Stage 4 引用未复现崩溃本身
tasks: [argmax-_argreduce_kernel-stage3-前次任务, argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel DESIGN §9.4 补录（2026-09-28）
last_seen: count_nonzero-_logical_reduce_kernel-20261016T005019Z（2026-10-16，同组合在新工具链不复现——复核记录，非独立识别）
```

### CG-2026-0015

```yaml
gap_id: CG-2026-0015
layer: TileLangIR pass（npu_loop_vectorize）
capability: >-
  2D T.Parallel(bm,N) 融合 if_then_else 循环在 bm>=2 时整体标量化（scalar ratio
  0.84–0.95），与操作数形态无关（(bm,1) 直接索引 / ext_brc 物化 + fragment staging
  两变体同标量化）；bm=1 同形态正常向量化。argmax hidden-state 实测 222µs（标量化）
  vs 链式 47µs（4.7x 差距）。
blocked_algo: >-
  row-reduction 族的融合候选/条件选择类 2D 循环在 bm>=2 不可用——被迫退到显式
  向量 op 链（vbrc/vcmp/vselect，pass 数 +2）或 per-row (1,W) 展开循环（op 数
  x bm）。跨任务影响：任何「逐元素条件 + 多行 block」形态。
evidence:
  - pattern-library TRAP-parallel2d-bm-ge2-scalarize（三组操作数变体对照 + msprof
    数据 + repro/repro_parallel2d_bm2_scalarize.py，ratio 3.02 实测复现）。
  - argmax opt_log Iteration 4 证伪更正记录（session-1 嫌疑 (a)/(b) 均被变体对照否证）。
workaround: bm>=2 用显式向量 op 链（vbrc/vcmp/vselect + reduce）。**2026-10-15 收窄**：「或 per-row (1,W) 展开」（PL-1.20 形态）仅在「每行**独立**缓冲」时成立——per-row 展开写**共享 (bm,N)** 缓冲为静默错值（新缺口本体见 CG-2026-0025），独立行缓冲形态则在本工具链下同样标量化（154.98 µs）无可采纳空间。
occurrences: 1
tasks: [argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel Stage 4 round 4（2026-09-28）
last_seen: 2026-10-15 argmin-_argreduce_kernel-20261015T023310Z Stage 4 复核（同族 `_argreduce_kernel` 谱系再现 + 绕法收窄；**不计 occurrences**——同 op 谱系且知识已预注入〔kb_search 命中〕，非独立识别；与之并列的静默错值形态另立 CG-2026-0025）
```

### CG-2026-0016

```yaml
gap_id: CG-2026-0016
layer: TileLangIR pass（向量化/codegen）
capability: >-
  T.arange(buf (M,W), [0,1], 0) 的索引物化按标量执行（~1ns/elem；((2,4096) 实测
  8.56µs）。(1,N) 形态正常。另：核内重复 T.arange 写同 buffer 非幂等（放大探针
  误编译，见 repro_arange_scalar.py docstring）。
blocked_algo: >-
  需要 j 索引载体的融合/链式候选构造在 bm>=2 时被迫 hoist 一份 (bm,N) fp32 缓冲
  并付出标量物化税（hidden-state 47µs 中占 ~18%）。
evidence:
  - pattern-library CONST-arange-scalar-materialize（msprof 46.97→38.41µs，
  -18.2%；节省量 8.56µs 与 8192 elem × ~1.05ns 模型吻合）。
workaround: T.arange 于 (1,N) + vbrc 首轴广播 (1,N)→(M,N)（文档合法形态）；
  bm=1 直接用（同形 vbrc 触发 CG-2026-0017）。
occurrences: 1
tasks: [argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel Stage 4 round 4（2026-09-28）
last_seen: argmax-_argreduce_kernel-20260924T031005Z（2026-09-28）
```

### CG-2026-0017

```yaml
gap_id: CG-2026-0017
layer: BishengIR（vbrc verifier）
capability: >-
  T.vbrc 同形 src/dst（无尺寸差维）产出空 broadcast_dims 数组，MLIR verify 报
  "'hivm.hir.vbrc' op have empty broadcast dims array"——语义上应等价于逐元素
  拷贝或前端拒绝（更清晰的报错）。
blocked_algo: 轻微：同形广播场景需改用源缓冲或 T.copy。
evidence:
  - pattern-library TRAP-vbrc-same-shape-empty-broadcast（stderr 诊断原文 +
  repro/repro_vbrc_same_shape.py 断言复现）。
workaround: 同形场景直接用源缓冲 / T.copy。
occurrences: 1
tasks: [argmax-_argreduce_kernel-20260924T031005Z]
toolchain_stamp: tilelang 0.1.2 (dev root build 2026-09-24) / CANN 8.5.0 / Ascend910B2C
status: open
created_by: argmax/_argreduce_kernel Stage 4 round 4（2026-09-28）
last_seen: argmax-_argreduce_kernel-20260924T031005Z（2026-09-28）
```

### CG-2026-0018

```yaml
gap_id: CG-2026-0018
layer: codegen (src/target/codegen_npuir_api.cc / codegen_npuir_api_a5.cc DTypetoMLIRType)
capability: >-
  target="npuir" 的 MLIR API codegen 无法表示 float8 缓冲区类型（e4m3_float8 /
  e5m2_float8）：DTypetoMLIRType 仅覆盖 f16/f32/f64/bf16 与 int/uint，无 float8
  分支，任何 float8 Buffer（GM 入参/出参或 UB alloc）在 lower 阶段即
  LOG(FATAL) "Cannot convert type e4m3_float8"（codegen_npuir_api.cc:493-593）。
  前端 TIR 支持该 dtype（tvm dtype 名 e4m3_float8，npu_loop_vectorize 可对 fp8
  cast 表达式按 rint 舍入 lowering），故缺口位于 codegen 的类型映射层。
  **加宽（2026-10-14 本迁移设计期独立复核）：float8 作为 cast 表达式同样不可 codegen。**
  `DTypetoMLIRType` 在 `VisitExpr_(CastNode)` 路径（codegen_npuir_api.cc:730 同源调用链）
  亦被调用，故 `u[i,j] = T.cast(x[i,j], "e4m3_float8")`（赋值目标是 uint8 buffer）与
  嵌套 `T.cast(T.cast(x,"e4m3_float8"),"uint8")` 均在 lower 阶段报
  `Cannot convert type e4m3_float8`——「fp8 cast 表达式 + 软件 reinterpret」不构成绕过路径
  （登记文本「任何 float8 Buffer」比实际缺口窄）。
blocked_algo: >-
  FP8QuantOp（per-group fp8_e4m3 量化，输出 float8_e4m3fn + fp32 scale）：无法
  原样产出/读写 fp8 张量——无法实现。降级（软件模拟 e4m3 编码写入 uint8 字节缓冲 /
  替换输出 dtype / 阻塞）均改变算法或 dtype 契约，未自行采纳，待用户裁决。
evidence:
  - src/target/codegen_npuir_api.cc:493 DTypetoMLIRType：float 分支仅 16/32/64，
    bfloat16 单列，无 is_float8 处理；:593 default LOG(FATAL) "Cannot convert type"
  - src/target/codegen_npuir_api_a5.cc 全文 grep float8/e4m3 -> 无命中
  - 复现（Ascend910B3 / CANN 9.0.0 / tilelang 0.1.2+ac0522ed2）：
    tilelang.engine.lower(target="npuir") 对含 e4m3_float8 GM 出参的 prim_func 报
    TVMError: Cannot convert type e4m3_float8；仅含 UB fp8 buffer（出参 uint8）同样报错；
    去掉 fp8 后同构 kernel（reduce_absmax + clamp + fp32 scale，fp16 出参）lower 成功
  - docs/ 全目录 grep float8/e4m3/fp8 -> 无命中（无文档化支持声明）
  - 对照 src/target/codegen_ascend.cc 有 float8 分支（fp8_e4_*_t），但属
    target.build.tilelang_ascend（CATLASS 字符串 codegen）链路，与 target="npuir"
    的 MLIR API codegen 非同一路径
  - 加宽复现（本迁移设计期，tilelang 0.1.2+ac0522ed2 / CANN 9.0.0 / Ascend910B3）：
    （c）单层 `u[i,j] = T.cast(x[i,j], "e4m3_float8")`（目标 uint8 buffer）→ 同报错；
    （d）嵌套 `T.cast(T.cast(x,"e4m3_float8"),"uint8")` → 前向量化产出 fp8 临时 buffer
    后同报错。探针（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/history_version/design_probe_encoder_lowering.py` / `design_probe_tileops.py`
  - **Stage 4 复验 + 绕法代价实测量化（2026-10-15，tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 / CANN 9.0.0 / Ascend910B3）**：
    fp8 cast（`u[i,j] = T.cast(x[i,j], "e4m3_float8")`，目标 uint8 buffer）**仍**报
    `BackendInternalError: Cannot convert type e4m3_float8` —— 缺口跨 3 个工具链戳存活。
    绕法代价实测（msprof op，同任务 `perf_opt/opt_log.md`）：软件 e4m3 编码器 ~7.0 µs
    （20+ 个整 tile 逐元素 pass，pass 单价实测 ~0.23–0.3 µs），叠加平台「空 kernel」
    地板 3.37 µs + 必需 GM 流量后 Task=11.26 µs；70% Ratio（≤5.52 µs）需编码器 ≤1.27 µs
    （≤4 pass），软件 RNE e4m3 编码最少 ~14 pass → **目标结构性不可达**（详见
    `perf_opt/perf_feedback.md`）。硬件窄化路径（若本缺口补齐）预期 Task ≈4.8 µs（Ratio ≈80%）。
workaround: >-
  已回填（2026-10-14 本迁移交付）：软件 e4m3 编码写入 **uint8 字节载体** + host 侧
  `out.reshape(B,S,G,D).view(torch.float8_e4m3fn)` 还原对外 dtype 契约（`.view` 为纯元数据
  操作、设备侧实测可用，见 CG-2026-0023）。代价：约 11–13 个向量 op/元素的软件编码链；
  对外 shape/dtype/数值与源算子一致（非契约降级）。缺口补齐后应去绕化（回退硬件窄化）。
  **Stage 4 量化该代价（2026-10-15）**：编码链实测 ~7.0 µs（基线 238.5 → 11.26 µs，
  其中标量化元凶 `T.vshl` 单独贡献 305 µs，已由 `T.vexp2` 向量化路线替换）；
  该代价使 Design §12.4 的 70%-Ratio 目标（≤5.52 µs）在本平台结构性不可达。
occurrences: 3
tasks: [fp8_quant-FP8QuantOp-20261014T150047Z（preflight）, fp8_quant-_fp8_quant_kernel-20261014T153942Z（迁移，独立复核并加宽：表达式路径同样 blocked）, fp8_quant-_fp8_quant_kernel-optimize-20261015（Stage 4：复验存活 + 绕法代价实测量化 → 70% 目标不可达）]
toolchain_stamp: preflight tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce；迁移 tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（实测为准，DESIGN §11 记录的 ac0522ed 为设计期探针戳）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14；Stage 4 复验 tilelang 0.1.2+31064b17e98e14347379b0926ee59dfd58c57f96 + CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: recurring
proposal: >-
  目标层 codegen（`src/target/codegen_npuir_api.cc` / `codegen_npuir_api_a5.cc` 的
  `DTypetoMLIRType`）：为 `is_float8()`（e4m3_float8 / e5m2_float8 及其 x4/x8/x16 向量类型）
  增加 MLIR 类型映射分支（对齐 `codegen_ascend.cc` 的 fp8_e4_*_t 语义），使
  target="npuir" 的 MLIR API codegen 能表示 float8 buffer 与 float8 cast 表达式。
  收益量化：解锁 FP8QuantOp（首个 fp8 迁移算子，当前被迫软件编码，+11–13 向量 op/元素的
  计算代价 + ~18 个中间 buffer 的 UB 压力）；受影响算子范式 = 一切低比特/字节载体算子
  （fp8 量化、fp8 attention indexer、int8 量化）——当前均须软件模拟或改契约。
  验证：以本任务 DESIGN §0.5.1 的四条 lower 探针（(a)(b)(c)(d)）为负向控制，补齐后应全部
  转为 COMPILE_OK，且 FP8QuantOp 可去绕化为硬件窄化路径。issue 草稿（标题前缀 [npuir]）
  须经用户批准后对外提交。
created_by: fp8_quant migration preflight (2026-10-14)
last_seen: fp8_quant-_fp8_quant_kernel optimize Stage 4（2026-10-15）：缺口跨 3 个工具链戳存活；绕法代价实测 ~7.0 µs 编码器 → 70% Ratio 目标结构性不可达（见 perf_opt/perf_feedback.md）
```

### CG-2026-0008 → 已升级 recurring，条目移入下方 Recurring 区（2026-09-15 occurrences 2）

---

### CG-2026-0019

```yaml
gap_id: CG-2026-0019
layer: Frontend API / runtime (tilelang/jit/jit_npu.py 出参分配)
capability: >-
  @tilelang.jit(out_idx=[...]) 的出参 torch dtype 取自 jit_npu.py::_tvm_dtype_to_torch 的
  白名单（float16/32/64、bfloat16、int8/16/32/64、bool）——**无 uint8**，并以
  `mapping.get(dtype_str, torch.float32)` 兜底。prim_func 声明 uint8 输出 buffer 时被
  静默分配 shape 相同但 dtype=float32 的张量；kernel 仍按 i8 memref 写 GM，返回内容
  按 4 字节解释——不报错、不告警。
blocked_algo: >-
  fp8/低比特软件编码类算子的字节载体输出（uint8 出参 + host .view(fp8)）。若沿用
  out_idx 表述输出，返回值不可用（被迫改为显式传输出张量，或改用 float32 出参从而
  丧失 1 字节/元素的载体契约）。
evidence:
  - tilelang/jit/jit_npu.py:1696-1708 （_tvm_dtype_to_torch 无 uint8 分支，末尾 get 兜底 float32）
  - 复现（任务工作区，允许失效）examples/fp8_quant/_fp8_quant_kernel/repro/TRAP-out-idx-uint8-dtype-fallback.py（知识域 repro：pattern-library/repro/TRAP-out-idx-uint8-dtype-fallback.py）——out_idx 返回 torch.float32 (256,64)；显式传 uint8 出参则 bit-exact
workaround: >-
  字节载体一律显式传输出张量、不声明 out_idx；由调用方分配 torch.uint8 缓冲。
  代价：调用方多一次分配（实测逐位一致，无性能/精度代价）。
occurrences: 1
tasks: [fp8_quant/_fp8_quant_kernel Stage 3]
toolchain_stamp: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760 / CANN 9.0.0 / Ascend910B3
status: open
proposal: none（未达 recurring；建议补齐 _tvm_dtype_to_torch 的 uint8（及 int8 语义区分）映射）
created_by: fp8_quant/_fp8_quant_kernel Stage 3, 2026-10-14
last_seen: 2026-10-14 fp8_quant/_fp8_quant_kernel Stage 3
```

### CG-2026-0020

```yaml
gap_id: CG-2026-0020
layer: TileLangIR pass / codegen (T.Parallel 内 f32 -> (u)int8 窄化 cast)
capability: >-
  `T.Parallel` 体内对**分数** f32 表达式直接 `T.cast(x, "uint8")` 时，lowering 得到
  `hivm.hir.vcast`（f32 -> i8）且**不是最近舍入**：实测 x.51 及部分 x.49 值映射到非
  最近整数（64 点探针 16 个值偏离 `round()`），既非 trunc 也非 round-to-nearest-even。
  同一表达式所在 buffer 的 dtype 是 uint8，codegen 实际降到 `i8`（signed），
  ≥128 的字节值饱和为 127。
blocked_algo: >-
  软件类型编码器（fp8 e4m3 / int8 量化等）需要在 `T.Parallel` 内把算出的字节值
  窄化到 1 字节；直接 cast 会静默产生错误字节（本任务表现为整片 e4m3 NaN 载荷）。
evidence:
  - 复现（任务工作区，允许失效）examples/fp8_quant/_fp8_quant_kernel/repro/TRAP-parallel-f32-uint8-cast-rounding.py（知识域 repro：pattern-library/repro/TRAP-parallel-f32-uint8-cast-rounding.py）——direct 16/64 值偏离 round()；两步 rint 0/64
  - docs/Tilelang.language/数据类型转换操作/T.vcast.md 规格表无 f32|ui8 行（仅 f16|ui8），
    前端 warning "Parallel cast lowering uses fixed round mode 'rint'" 与实际行为不符
  - src/transform/npu_loop_vectorize.cc:801（该 warning 文本）
workaround: >-
  先整数化再窄化：`T.vcast(src_f32, tmp_i32, round_mode="rint")` →
  `T.vcast(tmp_i32, dst_f32, round_mode="rint")` → `T.cast(dst_f32, "uint8")`；
  且字节须出在 signed i8 空间（`bits - 128*sign`，clamp 到 [-128,127]），
  两步 rint 后与 `round()` 0/64 一致。代价：每元素 +2 个向量 op。
occurrences: 1
tasks: [fp8_quant/_fp8_quant_kernel Stage 3]
toolchain_stamp: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760 / CANN 9.0.0 / Ascend910B3
status: open
proposal: none（未达 recurring；建议修 npu_loop_vectorize 的 Parallel cast lowering 使 f32->i8/ui8 遵循 rint，或在 lowering 报错而非静默错值）
created_by: fp8_quant/_fp8_quant_kernel Stage 3, 2026-10-14
last_seen: 2026-10-14 fp8_quant/_fp8_quant_kernel Stage 3
```

### CG-2026-0021

```yaml
gap_id: CG-2026-0021
layer: codegen / BishengIR (fp32 除法 lowering)
capability: >-
  npuir 的 fp32 除法**不是 IEEE correctly rounded**（无正确舍入的除法原语）：
  - `T.Parallel` 内标量除法表达式（per-row scale）：256 值中 135 个偏离正确舍入商 1-2 ulp；
  - `T.vdiv` 二维同形/广播形态（(M,D)）：6.1% lane 偏 1 ulp（3200x64 f32 实测）；
  - 仅 `T.vdiv` 的 (M,1)/(M,1) 列形态在上述实测中逐位精确。
blocked_algo: >-
  任何与 CPU/torch 参考实现做**逐字节**比对的量化/编码算子：参考值精确落在
  round-to-nearest-even 的 tie 上时，设备的 1 ulp 商偏差会把量化值推向另一侧邻居，
  产生 ±1 最低有效位差异（本任务 fp16 2/524288、bf16 8/524288 字节差异，全部为
  mantissa `n.5` tie）。无法通过实现手段消除（除非软件模拟正确舍入除法，代价 ~5+ 向量 op/元素）。
evidence:
  - 复现（任务工作区，允许失效）examples/fp8_quant/_fp8_quant_kernel/repro/CONST-npuir-f32-division-not-ieee-rounded.py（知识域 repro：pattern-library/repro/CONST-npuir-f32-division-not-ieee-rounded.py）——T.vdiv 2-D 6.136% lane 1 ulp
  - /tmp/opencode 探针（session-local）：probe_div2.py（SC_VDIV 0/256 vs SC_PAR 135/256）、
    probe_run3.py（scale 经 T.vdiv 后 q 仍 1 ulp 偏）
  - 本任务 §3 级 e2e 记录（任务工作区，允许失效）examples/fp8_quant/_fp8_quant_kernel/RETROSPECTIVE.md Stage 3 章节
workaround: >-
  (a) 把 scale 的除法改为 `T.vdiv` 列形态（实测精确），量化值除法保留 `x/scale`
  形态（与参考一致）而非 reciprocal-mul；
  (b) 精度判据须允许「参考 mantissa 落在 tie 的≤1 device-ulp 邻域内时取另一侧邻居」
  （本任务 classifier 用 |frac(|q|*2^(3-ec)) - 0.5| <= 1e-5），不得放宽为随机容差。
  代价：仍无法做到 100% 逐字节一致（残差 ~1e-5 量级元素数）。
occurrences: 1
tasks: [fp8_quant/_fp8_quant_kernel Stage 3]
toolchain_stamp: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760 / CANN 9.0.0 / Ascend910B3
status: open
proposal: none（未达 recurring；若后续多个逐字节比对类算子命中，建议评估提供正确舍入除法的 lowering 或在 docs 标注该限制）
created_by: fp8_quant/_fp8_quant_kernel Stage 3, 2026-10-14
last_seen: 2026-10-14 fp8_quant/_fp8_quant_kernel Stage 3
```

### CG-2026-0022

```yaml
gap_id: CG-2026-0022
layer: Frontend API / codegen (位级操作)
capability: >-
  npuir 前端缺少可用的位级操作路径，三条路线均不可用：
  (a) `T.reinterpret(dtype, expr)` 写在 `T.Parallel` 体内 → **编译进程 core dump**（无诊断）；
  (b) 移位/按位表达式（`>>`、`&`）写在 `T.Parallel` 体内 → 同样 core dump；
  (c) buffer 级 `T.vbitcast(ub_buf, "float32")`（含 `T.copy` 后位重解释）→
      `InternalError`（memref 形态/断言不符）。
  唯一可用形态是 tile 级 `T.vshl` / `T.vshr`（int32/int16/int64）。
blocked_algo: >-
  需要位级精确的编码/解码配方：从 f32 位型直接提取指数/尾数（frexp 风格）、
  位掩码构造、字节重解释。本任务的原计划（bitcast + 指数域加法构造精确 2^k）被迫
  改为「`T.vshl(1,k)` + 精确幂除」的整数域路线。
evidence:
  - /tmp/opencode 探针（session-local）：probe_bits.py — reinterp/bitshift 均
    `timeout: the monitored command dumped core`；vbitcast 报 InternalError
  - 可用形态：probe_exactu.py（`T.vshl` 编译通过并逐位正确）
  - docs/Tilelang.language/数据类型转换操作/T.vbitcast.md §2.2.1（uint8/int8/... 均 ×，
    仅 fp16/fp32 互转）+ docs/Tilelang.language/逻辑操作/T.vshl.md（int16/32/64）
workaround: >-
  位级精确改走整数域：需要精确 2^k 时用 `T.vshl(1_int32, k)` 再 `T.vcast(...,"rint")`；
  需要从 f32 取指数时用 tile 级 `T.vlog2` + `T.vcast(...,"floor")`（本任务实测
  floor(log2 a) 在 16384 元素上 0 误差）。代价：无法做 frexp 级位提取，
  位操作类配方须重写为算术配方。
occurrences: 1
tasks: [fp8_quant/_fp8_quant_kernel Stage 3]
toolchain_stamp: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760 / CANN 9.0.0 / Ascend910B3
status: open
proposal: none（未达 recurring；建议先在 `T.Parallel` 位操作路径给出诊断而非 core dump，并让 T.vbitcast 支持 UB buffer）
created_by: fp8_quant/_fp8_quant_kernel Stage 3, 2026-10-14
last_seen: 2026-10-14 fp8_quant/_fp8_quant_kernel Stage 3
```

### CG-2026-0024

```yaml
gap_id: CG-2026-0024
layer: BishengIR（`hivm.hir.vtranspose` UB 内 layout 变换的 lowering 选择）
capability: >-
  UB 内 `T.transpose`（二轴交换）在**池化族大 shape** 上被降级为 **UB→GM→UB 往返**，
  而非 UB 内 layout 变换：`(CH,H,W)=(64,7,7)` 的 `T.transpose([2,1,0])` 使 kernel
  5.6 µs → 138.2 µs，`UB_to_GM_datas` = 48 KB/block（≈ tile × 单元数）、
  `aiv_mte3_time` 43–58 µs（向量忙时仅 5–8 µs，代价全在访存往返）。该降级**强依赖
  shape/dtype 形态**：同工具链 `(64,56)→(56,64)` 的 2D f16 转置（两边 ≥8 且为 8 的
  倍数）仍走 UB 内实现（`UB_to_GM_datas` 仅 8.5 KB/block）——即「可用但形态受限」，
  非「不可用」。
blocked_algo: >-
  「核内布局重排 + 高整除性轴」路线（layout.md PL-1.1：核内 NHWC 二轴重排后把 C 变
  最内轴）在全局池化/整平面归约场景不可实现——被迫换结构（改为「保持原生内存序 +
  扁平空间轴 (N,C,1,H*W) + 长最内轴归约」，见 layout.md PL-1.23）。
evidence:
  - 受控分离探针（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/probe2.py`（p2_ct_c64：仅加 `T.transpose` 即 138.24 µs / UB_to_GM 336 KB 级）、`probe6.py`（p6_f4/f5 扁平+转置经 GM 往返 80.77/81.23 µs）、`probe_accum.py`（a49_1/a14_1 148–174 µs 而 vec 仅 5.46 µs）
  - 知识域 repro（可运行部分 = 三形态正确性 + 性能断言）：`.agents/skills/tilelang-op-optimize/references/pattern-library/repro/TRAP-transpose-ub-gm-roundtrip.py`（org：TRAP-transpose-ub-gm-roundtrip）
  - docs/Tilelang.language/shape操作/T.transpose.md（未记载 UB 内/往返的实现选择条件）
  - 反面形态（同工具链）：同任务的 spp-6x6 general 路径 `(64,56)→(56,64)` 2D f16 转置走 UB 内（UB_to_GM 8.5 KB/block）
workaround: >-
  绕过转置：整平面归约改「扁平空间轴」布局（`x: (N,C,1,H*W)`，同一连续内存的 rank
  重解释，I/O 字节序契约不变）+ 单次宽最内轴 `reduce_sum`——实测同 workload
  166.26 → 16.78 µs（9.91x），逐位精确（layout.md PL-1.23）。若必须转置，选「两边
  ≥ 向量宽且为向量宽整数倍」的 2D 形态。
occurrences: 2
tasks: [adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z, adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z]
toolchain_stamp: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14；第二任务重验 tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23（commit 67a0441 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: recurring
proposal: >-
  **目标层**：BishengIR（`hivm.hir.vtranspose` 的 UB 内 lowering 选择）+ Frontend（诊断）。
  **建议改动**：① 对「二轴交换的**两端均 ≥ 向量宽且为向量宽整数倍**」以外形态（如本次 `plane=49`）给出**显式诊断**（warning/error）而非静默走 UB→GM→UB；
  ② 或对该形态实现真正的 UB 内 layout 变换；③ 至少在 docs/Tilelang.language/shape操作/T.transpose.md 记载降级判据与可观测指纹（`UB_to_GM_datas` ≈ tile×单元数、`GM_to_UB` 同步被拉长）。
  **收益量化（两任务实测）**：avg pool 166.26→16.78 µs（9.91x，绕法 = 免转置扁平归约）；本任务 `global-1x1` 残余 ~60 µs（91.92 µs 中；纯拷贝地板 4.73–6.86 µs）——若转置走 UB 内，该 workload 预期进入 10–20 µs 量级。
  **受影响算子清单**：adaptive max/avg pool 2d（整平面/退化网格）、adaptive_max_pool2d_with_indices（双输出，写侧 store 源同样中招）、任何「核内布局重排 + 高整除性轴」路线。
  **issue 草稿（[npuir]，须用户批准后提交）**：`[npuir] hivm.hir.vtranspose silently lowers UB two-axis swap to a UB→GM→UB round trip when a side is not a multiple of the vector width`。
created_by: evolver 复原登记，2026-10-15（**溯源**：task adaptive_avg_pool2d-...-20261014T172114Z 的 opt_log.md §11 声明已按 AGENTS.md「Compiler Capability Gap Rule」登记本缺口并递增 occurrences，但蒸馏核对时 `.agents/evolution/capability-gaps.md` 无对应条目、文件亦无本任务未提交改动——即该声明未落盘；evolver 按 2026-09-21 先例（optimizer 声称入队未落盘 → 核对发现并补齐）以一手工件复原登记）
last_seen: 2026-10-15 adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z Stage 4（second occurrence: 写侧 store 源为 `(C,plane)` 转置 / 跨步 gather 时必中；连续 op 链生产者不中——触发条件定到写侧生产者形态，`UB_to_GM` 0.8 KB/blk vs 104 KB/blk）
```

### CG-2026-0025

```yaml
gap_id: CG-2026-0025
layer: TileLangIR pass / codegen（向量化与 region 分析 × 前端 `range` 语义）
capability: >-
  `@T.prim_func` 内的 `for r in range(bm)` 被解析为**运行时 `scf.for`**（非 trace-time
  展开，另一面见 traps-compiler.md TRAP-tir-range-loop-not-unrolled）。当循环变量仅
  参与**缓冲行偏移**（不索引 Python 列表）时解析器不报错，但该形态**静默产出错误数据**
  —— (2048,4096) fp16 上 `for r in range(bm): for i,j in T.Parallel(1,N):
  cand[r + i, j] = T.if_then_else(x_ub[r+i,j] == ext_brc[r+i,j], j, BIG)` 实测
  y0 = −3 vs `torch.argmin` ref 3211，无编译错误、无 warning。即「逐行展开 + 共享块
  缓冲写」这一在 GPU 后端常见的写法在本工具链上既非展开、又静默错值（不是标量化那种
  性能问题，而是**正确性缺失且无诊断**）。
blocked_algo: >-
  resident 归约族 bm≥2 的 fused 候选形态（单环 do-it-all 候选，省掉 cmp/idx/sent 三个
  全宽缓冲）不可用——被迫退到显式向量 op 链（`vbrc→vcmp→vselect→reduce`，每元素 ~5 pass）；
  「每行独立缓冲」形态正确但在本工具链下整体标量化（154.98 µs），无可采纳空间。
  跨任务影响：任何「bm≥2 的逐行向量化 + 共享块缓冲」形态（row-wise softmax/归约/池化候选）。
evidence:
  - pattern-library traps-compiler.md TRAP-scf-for-row-offset-silent-miscompile（四形态对照 + 断言说明）
  - 知识域 repro（可运行，`--level c2_perrow` 打印 MISMATCH / `--level c4_chain` 打印 OK）：
    .agents/skills/tilelang-op-optimize/references/pattern-library/repro/TRAP-scf-for-row-offset-silent-miscompile.py
  - 任务内证伪更正记录（任务工作区，允许失效）：examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/perf_opt/opt_log.md#Iteration-2（12.91–15.25 µs「更快」实为错值；c1_2d 268.82 µs 标量化 / c3_rowbuf 154.98 µs 正确但标量化 / f1_bm1_fused 284.72 µs）
  - docs/Tilelang.language/ 全目录未见「range 循环的 trace 语义」或「共享缓冲写与循环变量的限制」条款（未文档化，估量依据 = 上述本机实测）
workaround: >-
  bm≥2 的每元素条件选择走显式向量 op 链（vbrc/vcmp/vselect + reduce_min/max）；per-row 展开
  仅对**每行独立缓冲**成立（且本工具链下会标量化）。形态裁决前必过逐元素精确门
  （`torch.equal` vs CPU 参考）——错值档的典型指纹是「比正确档更快」（本例 12.9–15.3 µs vs 31.1–31.5 µs）。
occurrences: 1
tasks: [argmin-_argreduce_kernel-20261015T023310Z]
toolchain_stamp: tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: open
proposal: none（未达 recurring；建议方向 = ① 对「运行时循环变量作共享/片段缓冲行偏移」的写给出诊断或正确 lowering（当前静默错值，属正确性级缺陷）；② 或在 docs/Tilelang.language 记载 `range` 循环的 trace 语义与共享缓冲写的限制条款；issue 须用户批准）
created_by: evolver 复原登记，2026-10-15（**溯源**：task argmin-_argreduce_kernel-20261015T023310Z 的 perf_opt/opt_log.md 遗留问题 2 声明「如需该形态需编译器侧支持（能力缺口登记见 §Skill Retrospective）」，但该任务 §Skill Retrospective **无** CG 登记、本簿亦无对应条目——属「声称未落盘」；按 2026-09-21 / 2026-10-15 先例以一手工件复原登记）
last_seen: 2026-10-15 argmin-_argreduce_kernel-20261015T023310Z Stage 4（Iteration 2 正确性证伪）
```

### CG-2026-0026

```yaml
gap_id: CG-2026-0026
layer: BishengIR（`T.reduce_*` 等归约 lowering 的 UB 记账模型）
capability: >-
  BishengHIR 的归约 lowering 在 UB 上**按输入 tile 元素数线性记账**，单价由**归约链形态**
  决定，与「实际 dtype 缓冲清单」无关（前端无预算模型/分配清单可查，只能 compile-time
  反解）。整平面 max 归约的 NaN 追踪链在 (CH,1,H*W=49) fp16 上实测单价：`reduce_max →
  vcmp(平面自比较,"eq") → vcast → reduce_min` ≈ **96–128 B/平面元素**（把可用 CH 压到
  ≤32）；等价的原位平方-sum 形态（`vmul(flat,flat,flat) → reduce_sum`）≈ **32 B/元素**
  （解锁 CH≤120）；非原位平方（多一个全平面中间量）翻倍到 ≈64 B/元素。单价在 ch=64 与
  ch=128 **恒定**（严格线性、无固定项），CH 上限 = 196608 / (单价·H*W)。编译期
  `ub overflow, requires N bits while M bits available` 是唯一可见信号。
  **第二任务扩展（adaptive_avg_pool2d round 3，2026-10-15）**：① fp16 **同型精简链**
  `reduce_sum(f16→f16) + 单次 vcast` ≈ **48 B/平面元素**（介于 32 与 64 之间；plane=49
  允许 CH=64、拒绝 CH=128）；② **常驻 staging buffer 本身也按同一归约单价计费**——
  「一次大 DMA（P 平面）+ 核内分段归约」的 slab 结构在编译期即 `ub overflow`
  （`f16same P128 SUB32` → 1608192 bit vs 1572864 bit = 201 024 B ≈ 128×49×32；
  `f16same P256 SUB64` → 402 048 B；`f32 P128 SUB32` → 251 136 B）⇒ 单次归约输入被
  锁在 ≈125 平面（6144 fp16 元素）内，DMA 粒度与归约粒度**不可解耦**。
blocked_algo: >-
  整平面（全局池化）归约路径的 **CH 上限被 lowering 的 UB 记账隐式限制**——「用更宽 CH
  摊大 DMA 粒度 / 减少每核串行轮数」的空间被压缩（同工具链 UB 内 `T.transpose` 被降级为
  UB→GM→UB 往返同属该路径的 UB 容量约束，见 CG-2026-0024）。第二任务补充：「**核内 slab
  结构**（每核一次大 DMA + UB→UB 子块归约，以解耦 DMA 粒度与归约粒度）」被常驻 staging
  buffer 的同单价计费封死（G16，编译期 blocked）——即「1 轮读地板 5.64 µs vs 13 轮 9.64 µs」
  的地板收益无法通过 slab 结构兑现。
evidence:
  - pattern-library constants.md `CONST-reduce-ub-per-element`（单价表 + CH 上限式；探针须 DCE 免疫）
  - 知识域 repro（compile-time sweep，反解 N bits）：.agents/skills/tilelang-op-optimize/references/pattern-library/repro/TRAP-reduce-ub-per-element.py
  - 任务工作区探针（溯源，允许失效）：examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/perf_opt/probe_ub2.py（无死代码精确镜像：201216 B @ ch=128 fp16 原位平方；401792 B @ ch=64 bool 掩码链）
  - 第二任务探针（溯源，允许失效）：examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/probe_r3_slab.py（slab 编译期 sweep：`ub overflow, requires 1608192 bits while 1572864 bits available`）+ `perf_opt/probe_r3_gf.py`（f16 同型链单价 48 B/元素，ch64 允许 / ch128 拒绝）+ `perf_opt/opt_log.md` §13.4 P8 / §13.5 G16
  - docs/Tilelang.language/规约操作/ 未记载归约的 UB 记账模型或形态相关单价，也未记载常驻 buffer 的计费方式（未文档化，估量依据 = 上述本机 compile-time sweep）
workaround: >-
  NaN 追踪改用**原位平方-sum 等价形**（`isnan(reduce_sum(x*x))`，单价 32 vs 96–128 B/元素）
  解锁更宽 CH；配合**扁平空间轴** `(N,C,1,H*W)` 单次宽归约（layout.md PL-1.23）把「成本 ∝
  平面元素」的归约链限制在单次。第二任务补充：整平面均值归约改用 **fp16 同型精简链**
  （`reduce_sum(f16→f16) + 单次 vcast`，单价 64→48 B/元素，解锁 CH=64）。代价：仍无法逼近
  同栅格纯拷贝地板（见 CG-2026-0027）；slab（大 DMA + 分段归约）无绕法，被迫接受
  「DMA 粒度 = 归约粒度」的耦合。
occurrences: 2
tasks: [adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015T081354Z, adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z]
toolchain_stamp: tilelang 0.1.2+489385e9b98806365b585b6e1f984a5dae337154（commit 489385e / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15；第二任务重验 tilelang 0.1.2+31738b08cb8f44e9596aba32ffeda82040d2152c（commit 31738b0 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: recurring
proposal: >-
  **目标层**：BishengIR（归约 lowering 的 UB 预算分配）。
  **建议改动**：① 公布/输出归约 lowering 的 UB 预算模型与逐 buffer 分配明细（含形态相关单价，并按 dtype 与归约链形态分档），使前端可在设计期核算 CH 上限而非 compile-time 反解；② 分离「常驻 staging buffer」与「归约链中间量」的计费口径——常驻 buffer 应按实际声明形状计费，使「一次大 DMA + 核内分段归约」的 slab 结构可用（当前同单价计费使其编译期即溢出）；③ 或在 `ub overflow` 诊断中列出逐 buffer 明细（当前仅给出总 required/available bits）。
  **收益量化（两任务实测）**：avg pool round 3 slab 结构被阻塞——若可用，单次大 DMA + 分段归约可把地板从 13 轮 9.64 µs 逼近 1 轮 5.64 µs（≈ −4 µs / −40%）；max pool 侧单价分档使 CH 从 ≤32 提到 ≤120（layout.md PL-1.22 ④）。
  **受影响算子清单**：adaptive avg/max pool 2d（整平面/退化网格）、adaptive_max_pool2d_with_indices（双输出）、任何整平面归约 + 大 DMA 粒度诉求的池化/归约算子。
  **issue 草稿（[npuir]，须用户批准后提交）**：`[npuir] vreduce UB accounting charges resident staging buffers at the per-element reduce unit price, blocking slab (single big DMA + in-core segmented reduce) structures`。
created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015T081354Z Stage 4（2026-10-15，opt_log.md §10 按 AGENTS.md「Compiler Capability Gap Rule」登记）
last_seen: 2026-10-15 adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z Stage 4（round 3：fp16 同型链 48 B/元素 + slab 常驻 staging buffer 同单价计费 → 编译期 blocked）
```

### CG-2026-0027

```yaml
gap_id: CG-2026-0027
layer: BishengIR（归约与 GM→UB 搬运的流水调度 / 跨 pipe 重叠）
capability: >-
  同一 kernel 内 `T.reduce_*` 与 GM→UB 的 `T.copy` **无法重叠**：整平面归约 kernel 的
  `aiv_mte2_time` 实测 **211 GB/s**，而**同栅格同 DMA 粒度的纯拷贝标定**达 **690 GB/s**
  （[8,2048,7,7] fp16、CH×49 fp16 元素/轮、40 block、910B3）——搬运窗口未被归约计算填充、
  也不与下一轮 DMA 重叠（`Task ≈ max(mte2,vec)+scalar` 的 max 退化为近似 sum）。循环构造
  `T.Pipelined(2/3)` 于单元循环实测无增益（未能形成归约 consumer 与 DMA 的双缓冲重叠）。
  **第二任务重验（adaptive_avg_pool2d round 3，2026-10-15）**：同栅格 DMA-only kernel
  （`dma_gm4d_ch32_r13`）Task 仅 **8.75 µs**（含 4.81 µs 空转固定开销 ⇒ 真实搬运 ≈3.9 µs），
  而真实 C4 kernel 的 `aiv_mte2_time` 跨度 = 整条串行链 11.62 µs —— 即 mte2 跨度被
  「DMA 等归约 consumer 释放 UB」拉长（复现同一不重叠）；`npuir.enable_auto_multi_buffer=True`
  （G17）实测并无重叠收益（`config_no_gain`）。
blocked_algo: >-
  global 类池化/归约逼近「同栅格纯拷贝地板」的通路受阻：final 18.86 µs vs 标定地板
  5.09–6.85 µs（**2.8–3.7x 缺口**），其中一部分即该不重叠（另一部分见 CG-2026-0026）。
  第二任务补充：round-3 resnet 的剩余缺口（16.29 vs 1 轮读地板 5.64 µs）同样主要由此构成
  （`mte2` 11.08 µs = 整条串行链）。
evidence:
  - 任务工作区 opt_log（溯源，允许失效）：examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/perf_opt/opt_log.md §3（baseline）/§4 R3（`mte2=9.10 µs` vs 拷贝标定 5.09–6.85 µs）/G5·G9（`T.Pipelined(2/3)` config_no_gain）
  - 任务工作区标定（溯源，允许失效）：examples/TileOPs/.../perf_opt/copy_floor_calibration.jsonl（同栅格纯拷贝地板 6.85/5.09 µs；mte2 2.13/1.21 µs）
  - 第二任务探针/标定（溯源，允许失效）：examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/{profiles/probe/round3/dma_gm4d_ch32_r13, probe_r3_gf.py} + `perf_opt/opt_log.md` §13.4 P6 / §13.5 G17（`npuir.enable_auto_multi_buffer=True` config_no_gain）
  - pattern-library elementwise.md `PL-1.6-copy-floor`（轮数主导地板）+ constants.md `CONST-aiv-time-decomposition`（`Task ≈ max(mte2,vec)+scalar` 分解口径）
  - docs/Tilelang.language/同步管道操作/ 与 线性代数操作/ 未记载归约与搬运的自动流水重叠条件（未文档化，估量依据 = 上述本机实测）
workaround: >-
  无有效重叠手段（`T.Pipelined(2/3)` 于单元循环实测无增益；`npuir.enable_auto_multi_buffer=True` 同）。
  当前只能靠**减少每核串行轮数**（更宽 CH，受 CG-2026-0026 的 UB 记账约束）压低地板逼近缺口。
occurrences: 2
tasks: [adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015T081354Z, adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z]
toolchain_stamp: tilelang 0.1.2+489385e9b98806365b585b6e1f984a5dae337154（commit 489385e / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15；第二任务重验 tilelang 0.1.2+31738b08cb8f44e9596aba32ffeda82040d2152c（commit 31738b0 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: recurring
proposal: >-
  **目标层**：BishengIR（归约 kernel 的流水调度）。
  **建议改动**：在「GM→UB `T.copy` → reduce consumer」链上实现自动双缓冲重叠，或提供显式 overlap 原语（如按 pipelined stage 声明 reduce 依赖）；至少在 docs 记载归约与搬运的自动流水重叠条件（含 `auto_multi_buffer` 是否覆盖该链）。
  **收益量化（两任务实测）**：max pool mte2 211 GB/s vs 同栅格纯拷贝 690 GB/s（3.3x 带宽缺口，2.8–3.7x 时延缺口）；avg pool round 3 mte2 跨度 = 整条串行链 11.62 µs vs 同栅格 DMA-only 8.75 µs（含 4.81 µs 固定开销）。
  **受影响算子清单**：adaptive avg/max pool 2d、argreduce/row-reduction、任何「搬运 + 核内归约」串行链算子。
  **issue 草稿（[npuir]，须用户批准后提交）**：`[npuir] T.reduce_* does not overlap with GM->UB T.copy in the same kernel (serialized chain; auto multi-buffer default / Pipelined have no effect)`。
created_by: task adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015T081354Z Stage 4（2026-10-15，opt_log.md §10 按 AGENTS.md「Compiler Capability Gap Rule」登记）
last_seen: 2026-10-15 adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z Stage 4（round 3：DMA-only 8.75 µs vs 真实链 11.62 µs；auto_multi_buffer config_no_gain）
```

### CG-2026-0029

```yaml
gap_id: CG-2026-0029
layer: Frontend API / BishengIR（归约 dtype 形态 + v 算子 UB 子区域操作数形态）
capability: >-
  (a) **`T.reduce_sum` 无 bf16 累加形态**——整平面均值归约在 bf16 输入下不能走「输入 dtype 内部累加」
  的精简链（`reduce_sum(bf16→bf16) + 单次 vcast`），必须保留 **fp32 全平面中转**
  （`vcast → f32 平面 → reduce_sum(f32)`），使 BishengHIR UB 单价从 fp16 同型链的 ≈48 B/平面元素
  抬到 ≈64 B/平面元素（plane=49 时 CH 上限降至 62 档，且多一份全平面缓冲 + 一次全平面 vcast）。
  (b) **`T.vcast` 不接受 UB 子区域形状**——对 4-D 常驻 buffer `big[K,SUB,1,HW]` 的前导轴切片
  做 `vcast` 报 `The input/init operands and result have the same rank`（G9/G16 `direct` 变体），
  即「一次大 DMA + 核内子块 cast」路径不可表达（与 CG-2026-0028(c) 的 `hivm.hir.vsel` cond
  不接受 UB 子区域同族——v 算子的 region 操作数形态限制）。
blocked_algo: >-
  (a) bf16 全局池化（nondiv-7x7-bfloat16 等）无法采用 fp16 侧的「同型精简归约链」优化
  （该优化在 fp16 路径贡献 resnet-global `aiv_vec_time` 3.88→2.12 µs / Task ab −3.79%，见 layout.md
  PL-1.23 dtype 变体）——bf16 被迫保留 fp32 中转与更高 UB 单价，性能路径与 fp16 不对称。
  (b) 核内 slab/sub-region 归约的 `direct` 变体（大 buffer 前导轴切片上直接 cast）不可表达，
  与 CG-2026-0026 的常驻 staging buffer 计费共同封死「大 DMA + 分段处理」结构。
evidence:
  - 任务工作区（溯源，允许失效）：examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/opt_log.md §13.4 P8 / §13.5 G16（bf16 单价 ≈64 vs fp16 同型 ≈48 B/平面元素；`T.vcast` 子区域报 `The input/init operands and result have the same rank`）+ `perf_opt/probe_r3_slab.py`（direct 变体编译报错）
  - 对照常数：pattern-library constants.md `CONST-reduce-ub-per-element`（fp16 链单价表）；同族 region 形态缺口 CG-2026-0028(c)
  - docs/Tilelang.language/规约操作/T.reduce_sum.md 与 数据类型转换操作/T.vcast.md 未记载跨 dtype 累加形态与 UB 子区域操作数限制（未文档化，估量依据 = 上述本机 compile-time 实测）；round 1/2 的 opt_log §11 曾声称「`T.vcast` 不接受 UB 子区域」已登记并递增 occurrences，蒸馏核对 capability-gaps.md **无对应条目**（声称未落盘），按先例以一手工件复原登记
workaround: >-
  (a) bf16 路径保留 fp32 全平面中转（无法用 fp16 侧的同型精简链）；(b) 子区域 cast 改走
  独立 staging buffer 形态（受 CG-2026-0026 常驻 buffer 单价约束，slab 仍不可行）。代价：
  bf16 路径多一份 f32 全平面缓冲 + 一次全平面 vcast，且 CH 档位低于 fp16。
occurrences: 1
tasks: [adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z]
toolchain_stamp: tilelang 0.1.2+31738b08cb8f44e9596aba32ffeda82040d2152c（commit 31738b0 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: open
proposal: none（未达 recurring；建议方向 = ① `T.reduce_sum` 支持 bf16 输入的内部累加形态（或文档化「bf16 必须 fp32 中转」及其 UB 代价）；② 放宽 `T.vcast` 的操作数区域形态（支持 UB 子区域/前导轴切片，与 `vcmp` 的合法 slice 形态对齐）；issue 须用户批准）
created_by: evolver 复原登记，2026-10-15（**溯源**：task adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z 的 opt_log.md §13.14-3 声明两类新实证未落盘（该 dispatch 硬约束「只写 perf_opt/」），evolver 按 AGENTS.md 缺口规则与 2026-10-15 先例登记）
last_seen: 2026-10-15 adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261015T124025Z Stage 4（round 3 G9/G16 探针）
```

### CG-2026-0031

```yaml
gap_id: CG-2026-0031
layer: Frontend API（`tilelang/language/customize_npuir.py` 的 `npuir_gather` / `npuir_deinterleave`——「宽区域跨步抽取」原语）
capability: >-
  宽区域「跨步抽取」（decimation：从一条整宽 span 中按固定步长取出子集）没有可用的向量原语：
  ① `T.gather`（vgather）是**逐元素**实现，实测 ≈**1.8 ns/元素**（`(80,21)←(80,61)` 两次 gather 使
  `aiv_vec` 增加 ≈320 µs / 52 单元）——抽取本身比它要省的窄 op 更贵；
  ② `T.deinterleave` 相位数上限 = 2（`=3` 触发 `bishengir-compile` 段错误，见 CG-2026-0032），而
  stride 3 的窗口类需要 3 相；
  ③ `T.transpose` 抽取需 3D 视图（`vtranspose` 仅 2 轴），且 `extract_slice` 行距不匹配即被物化
  （见 CG-2026-0033）。三者组合使「把窗口类 tap 循环折叠成 O(k) 条宽 op」的路线不可实现。
blocked_algo: >-
  窗口/重叠窗类 general 路径的「折叠 owl 循环 → 宽 dilate-max + 抽取」结构族：本任务以两个独立
  结构分支实测否决（通道主序 DMG 23328 µs、位置主序 PM 998.92 µs，均劣于 baseline 302 µs）⇒
  被迫放弃该结构族，转从通道分块形状与 DMA 行长入手（收益 −37.6%）。性能上限：窄 op 结构
  （每输出位置 × 每 tap 一条 CH 宽向量 op，实测 ≈14.4 ns/op/核）在冻结设计族内无法消除。
evidence:
  - 实测（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/{probe_dmg.py, probe_pm.py, probe_deint.py, logs/probe_dmg_*.log, logs/probe_pm_sincnetbl.log}`
  - 知识域 repro：`pattern-library/repro/TRAP-window-decimation-primitives.py`（gather + 物化形态）与 `pattern-library/repro/TRAP-deinterleave-two-phase-only.py`（相位数上限 2；重验触发词 = case B 三路全 True）
  - docs/Tilelang.language/shape操作/T.deinterleave.md §2.1（`channel_nums: int`，缺省 2）/ §2.3「特殊限制说明：无」（未标注相位数上界）；docs/ 无「宽 gather / decimation 原语」条目（未文档化假设）
  - 知识域条目：`pattern-library/traps-compiler.md` TRAP-window-decimation-primitives / TRAP-deinterleave-two-phase-only / TRAP-C10-strided-gather
workaround: >-
  放弃宽抽取路线，改从形状与搬运入手：① `CH = min(C, cap)` 且 cap ≥ C，消除通道分块尾块在窗口 op
  上的 padding lane 浪费并把 `num_logical` 3392→2080（sincnet −37.7% / ecg −26.9%）；② 把 `LSPAN`
  当 DMA 粒度（行长）看（`CONST-dma-burst-rowstride-floor`：行长 128–156 B ⇒ ≈0.38 TB/s）；
  ③ 2 相抽取可用（argmax 案例 130.3×，`CASE-argmax-argreduce-stage4`）。
occurrences: 1
tasks: [max_pool1d-_max_pool1d_kernel-20261016T090104Z]
toolchain_stamp: tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ CANN 9.0.0 + Ascend910B3（aicore 20 → 纯 Vector 40 / UB 196608 B）/ 2026-10-16
status: open
proposal: none（未达 recurring；建议方向 = ① 提供硬件级宽 gather（带 stride 的向量抽取）或把 `T.deinterleave` 扩展到任意相位数；② 至少在 docs 记载 `channel_nums` 上界与 decimation 原语的能力边界）
created_by: max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 `opt_log.md#§7.3`，evolver 终态蒸馏核验归档）2026-10-16
last_seen: 2026-10-16 max_pool1d-_max_pool1d_kernel-20261016T090104Z Stage 4（首次识别）
```

### CG-2026-0032

```yaml
gap_id: CG-2026-0032
layer: BishengIR（`bishengir-compile` 后端，`hivm.hir.vdeinterleave` lowering）
capability: >-
  `T.deinterleave(..., channel_nums=3)`（3 相解交织，stride-3 滑窗需要）使 `bishengir-compile`
  **段错误**（rc = −6 / SIGABRT）：NPU IR 已正常生成、崩溃发生在后端，无 MLIR 诊断可比对；
  `channel_nums=2` 同形态语义正确（含 region 目标）⇒ **相位数 >2 不可用**。
blocked_algo: >-
  stride-3 滑窗（sincnet k3s3p0）的「相位分解」抽取路线无法用单条原语实现，只能退化为逐相
  slice/`T.copy`（成本另计）；本任务据此放弃该抽取路线（结构侧后果见 CG-2026-0031）。
evidence:
  - 知识域 repro：`pattern-library/repro/TRAP-deinterleave-two-phase-only.py` case B（**注**：任务 `opt_log.md#§7.2` 引用的崩溃日志 `perf_opt/logs/probe_deint.log` 在工作区中不存在〔扫描 0 命中〕，durable 证据 = 重跑该 repro）
  - 任务工作区（允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/probe_deint.py`
  - docs/Tilelang.language/shape操作/T.deinterleave.md §2.1（`channel_nums: int`，缺省 2）/ §2.3「特殊限制说明：无」——**未标注相位数上界**（未文档化假设）
  - 关联：`pattern-library/traps-compiler.md` TRAP-bishengir-crash-s32（同后端段错误家族的 **shape×dtype** 形态；本条是 **API 参数** 维度，两形态是否同源**未证**）+ `.agents/evolution/capability-gaps.md` CG-2026-0030
workaround: >-
  仅使用 2 相形态（`channel_nums=2`，含 region 目标）；多相需求改走逐相切片/`T.copy` 或 reshape 后
  的多路 2 相分解。正向参照：argmax Stage 4 的「窄行宽视图 + 2 相 deinterleave」130.3×。
occurrences: 1
tasks: [max_pool1d-_max_pool1d_kernel-20261016T090104Z]
toolchain_stamp: tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ CANN 9.0.0（bishengir-compile）+ Ascend910B3 / 2026-10-16
status: open
proposal: none（未达 recurring；建议方向 = 修复 `vdeinterleave` 相位数 >2 的后端崩溃，或在前端对 `channel_nums > 2` 给出显式诊断而非静默进入后端崩溃）
created_by: max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 `opt_log.md#§7.3`，evolver 终态蒸馏核验归档）2026-10-16
last_seen: 2026-10-16 max_pool1d-_max_pool1d_kernel-20261016T090104Z Stage 4（首次识别）
```

### CG-2026-0033

```yaml
gap_id: CG-2026-0033
layer: TileLangIR pass / bufferization（`tensor.extract_slice` 的「行距不匹配」降级路径）
capability: >-
  `tensor.extract_slice` 的结果 tensor 行 stride ≠ 源 buffer 行 stride 时（如 `(80,61)` 取自
  `(80,64)`），bufferization 把它降级为 **UB→GM→UB 物化**，且该物化**被标量化**（无向量化兜底、
  无编译期告警）：sincnet 单分支 Task 301.886 → **23328 µs**（`scalar ≈ 22878 µs`、`aiv_vec 480 µs`）
  ⇒ **77× 时延放大**，正确性不受影响（不报错），只有时延与标量占比能暴露。
blocked_algo: >-
  「region 抽取 + 宽归约」结构（窗口类折叠 + 单点列布局改造）不可用——除非每个 region 操作数都与
  其 buffer **行距一致**（整宽行区间）；本任务位置主序变体（切片全部连续、BL=13 逐位精确）实测仍
  998.92 µs > baseline 302 µs ⇒ 结构族整体否决。
evidence:
  - 知识域 repro：`pattern-library/repro/TRAP-window-decimation-primitives.py`（`--mode` 选择 extract_slice 形态；sincnet `scalar` ≈ 22.9 ms）
  - 任务工作区（允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/{probe_dmg.py, probe_pm.py, logs/probe_dmg_sincnetblmodestatic.log}`
  - 机制同族但**不同层 / 不同 op**（故分列不合并，互链）：`pattern-library/traps-compiler.md` TRAP-transpose-ub-gm-roundtrip + 本簿 CG-2026-0024（BishengIR 层 `hivm.hir.vtranspose` → UB→GM→UB）；本条 = TileLangIR/bufferization 层 + `extract_slice` **行距**触发，且额外被**标量化**
  - 标量化兜底缺失证据：`scalar` 占 98%（22 878 / 23 328）
workaround: >-
  结构改造必须保证每个 region 操作数与其 buffer 行距一致（整宽行区间）；免抽取/免物化的替代路径：
  整平面域走 `pattern-library/layout.md` PL-1.23-flat-spatial-axis-reduce（扁平声明 + 宽最内轴归约，
  9.96×）；窗口类改从 CH 形状与 DMA 行长入手（`constants.md` CONST-dma-burst-rowstride-floor）。
occurrences: 1
tasks: [max_pool1d-_max_pool1d_kernel-20261016T090104Z]
toolchain_stamp: tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（commit c3c39ba / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-16
status: open
proposal: none（未达 recurring；建议方向 = 对「源/目标行距不一致的 `extract_slice`」给出编译期诊断（warning），或提供向量化的物化兜底路径）
created_by: max_pool1d-_max_pool1d_kernel-20261016T090104Z（Stage 4 `opt_log.md#§7.3`，evolver 终态蒸馏核验归档）2026-10-16
last_seen: 2026-10-16 max_pool1d-_max_pool1d_kernel-20261016T090104Z Stage 4（首次识别）
```

### CG-2026-0034

```yaml
gap_id: CG-2026-0034
layer: TileLangIR pass / lowering（条件分支内的 `vdeinterleave` 被丢弃）
capability: >-
  `T.deinterleave` 落在**运行时 `if/else` 分支**内时，lowering 把该 op **静默丢弃**：编译通过、
  运行无报错、输出全错；生成的 hivm IR 中**不存在** `vdeinterleave`，应被写入的目标缓冲被下游当作
  未初始化数据读。无编译告警、无 MLIR 诊断 —— 属**正确性级**缺口（静默错值），且只在「结果全错 +
  降级 IR 无该 op」时才可归因。
blocked_algo: >-
  「两路 deinterleave 按运行时奇偶（`r % 2`）做 ping-pong 行选择」这一自然写法不可用——本任务
  ping-pong 行预取的首个版本即因此产出垃圾输出（编译/运行均正常）；被迫改为**编译期（trace 期）
  奇偶展开**（pair loop 拆两、两条 deinterleave 均为直线代码），语义等价但失去运行时单路径的代码量
  优势。副作用：任何「按运行时索引选择抽取相位」的通用写法（如可变 `channel_nums` 路由）同样不可信。
evidence:
  - 知识域条目：`pattern-library/traps-compiler.md` TRAP-deinterleave-two-phase-only（case D 形态 + 绕法）
  - 知识域 repro（绕法侧）：`pattern-library/repro/PATT-row-pingpong-pipeline.py`（trace 期展开后 448 unit 逐位相等、`max_diff = 0.0`）
  - 任务工作区（允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool2d/max_pool2d_kernel/perf_opt/{probe_pipe.py, opt_log.md#Round-3, PROGRESS.md}`（hivm IR 中 op 缺失 + 输出全错的实测记录）
  - 同族但**不同触发维度**（不合并、互链）：CG-2026-0032（`channel_nums > 2` → `bishengir-compile` SIGABRT，参数维度）；本条是**控制流位置**维度（参数合法、op 被丢）
workaround: >-
  奇偶/相位选择在 **trace 期展开**（pair loop 拆两或 Python 层分支），保证每条 `T.deinterleave` 都是
  直线代码；改完后用 `--check` 对参考实现做逐位（`torch.equal`）验证，并 grep `kernel.mlir_content`
  确认 `vdeinterleave` 仍在。代价：行级 ping-pong 的代码量翻倍（两条 row body 路径），实测收益不变
  （vgg/alexnet 六例 −7.0%~−25.0%）。
occurrences: 1
tasks: [max_pool2d-_max_pool2d_kernel-20261017T043853Z]
toolchain_stamp: tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025（commit b3a7b21）+ torch 2.9.0+cpu / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3（aicore 20 → AIV 40 / UB 196608 B）/ 2026-10-17
status: open
proposal: none（未达 recurring；建议方向 = ① 条件分支内遇到无法下沉的 DMA/向量 op 时给出编译期 warning（而非静默丢弃）；② 在 `T.deinterleave` 文档补「必须位于直线代码」的限制条款）
created_by: max_pool2d-_max_pool2d_kernel-20261017T043853Z（Stage 4 round-3 falsification，evolver 终态蒸馏核验归档）2026-10-17
last_seen: 2026-10-17 max_pool2d-_max_pool2d_kernel-20261017T043853Z Stage 4（首次识别）
```

## Recurring（达升级阈值，待推动补齐）

> **条目位置说明（2026-10-16）**：`CG-2026-0003` 本次由 `open` 升级为 `recurring`（第二形态 = Expert 显式 `T.sync_block_set` 的 `syn_instr_mode` 属性与 CANN 9.0.0 bishengir-compile 的版本错配），为便于回溯两形态的完整证据链，其**正文保留在下方 `## Open` 区原位更新**（标题已标注 recurring），proposal 字段已产出，与本节其余 recurring 条目同等待推动状态。

### CG-2026-0023

```yaml
gap_id: CG-2026-0023
layer: runtime (torch_npu / aclnn 设备侧算子覆盖)
capability: >-
  torch_npu 设备侧不支持 float8_e4m3fn 的 **copy / 初始化**类算子：
  `x_fp32.to(torch.float8_e4m3fn)`（aclnnInplaceCopy）报 `error code 561103`（任意
  shape）；`torch.zeros(..., dtype=fp8, device="npu")` / 任意 fp8 `zeros_`
  （aclnnInplaceZero）报 `error code 161002`。但 **`.view(torch.float8_e4m3fn)` 字节
  重解释可用**（纯元数据，不走 aclnn copy），`torch.empty(fp8, device="npu")` 亦可用。
  即作用范围是 copy/init 类算子，不是「fp8 dtype 整体不可用」。
blocked_algo: >-
  fp8 算子的 host 侧构造/比较路径：无法在设备侧把 fp32/fp16 张量 cast 成 fp8（golden、
  reference、比较器须绕行），也无法用 `torch.zeros` 造 fp8 张量。对 FP8QuantOp 的对外
  dtype 契约**不构成阻塞**（kernel 输出 uint8 载体 + host `.view` 还原即可，见 CG-2026-0018
  workaround）；但对「设备侧 fp8 cast」直写路径与 fp8 golden 构造构成阻塞。
evidence:
  - 复现（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/history_version/design_probe_runtime_front8.py`
    （Ascend910B3 / torch 2.9.0+cpu / torch_npu 2.9.0.post8 / CANN 9.0.0，逐操作实测表）
  - 交叉印证（任务工作区，允许失效）：`examples/TileOPs/tests/ops/test_fp8_quant.py` 的 S7 注释
    （fp8 cast 走 CPU 执行后搬回设备 = golden-availability 适配）
  - 独立体现（集成侧）：`examples/TileOPs/tileops/testing/test_base.py` 的
    `_to_comparison_dtype()` 对 fp8 走 CPU 上抛（该改动为脚手架作者所加、自标未授权，见报告 issues）
  - docs/ 全目录 grep float8/e4m3/fp8 -> 无命中（无文档化支持声明）
workaround: >-
  (a) 设备侧 fp8 构造走 `torch.empty(fp8)` 或 `uint8` 缓冲 `.view(fp8)`（勿用 `zeros`/`.to`）；
  (b) host 侧 fp8 值转换走 CPU（`.cpu().to(fp32)`）+ H2D/D2H，或直接以 uint8 为载体比对；
  (c) golden/reference 的 fp8 cast 一律 CPU 路由后搬回设备（位级等价）。
occurrences: 1
tasks: [fp8_quant-_fp8_quant_kernel-20261014T153942Z]
toolchain_stamp: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760 + torch 2.9.0 / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
status: open
proposal: none（未达 recurring；建议补齐 aclnn 侧 fp8 copy/init 算子覆盖，或在 torch_npu 文档标注「设备侧 fp8 cast / zeros 不可用、.view 可用」）
created_by: fp8_quant/_fp8_quant_kernel 设计期独立复核（2026-10-14，脚手架实测）
last_seen: 2026-10-14 fp8_quant/_fp8_quant_kernel（设计+集成两阶段体现）
```

### CG-2026-0008

```yaml
gap_id: CG-2026-0008
layer: BishengIR（per-AIV UB 分配可见性）
capability: >-
  BiShengIR 的 per-AIV UB 实际分配与前端逐 buffer 显式预算之间存在未记载的基础差值
  （三次实测、双向偏差：28783f45 手工预算 178.2KB 实际 ~204.1KB +15%；6797758
  4 点标定 = 手工预算 × 1.10–1.12；a13585dc 显式 alloc 结构反例 0.981×〔actual 低于
  手工〕——系数结构依赖），且无逐 buffer 分配清单查询手段——编译报文只给
  总 requires bits，设计期手工预算不可静态核验，差值来源（疑 extra/sync buffers）
  无文档记载。与 CG-2026-0002（auto-multi-buffer 膨胀）不同：本差值在
  `bishengir-compile --enable-auto-multi-buffer=false` 下仍然存在（基础分配差）。
blocked_algo: >-
  宽 config 设计意图的性能上限形态：two-phase v3 的 bn_eff=256 宽块路径在
  dim=128 causal manifest 域（4 workload × fp16/bf16）不可编译（ub overflow 硬失败，
  首次编译发生在 Stage 5 bench，8/10 失败）——两相位 ws 物化流量的宽块摊销收益
  无法实现，被迫窄 config（bn_eff∈{64,144}）交付：短 KV 1.30–1.32x 提速 / 长 KV
  0.78x 回退（vs Sep-07 单遍基线，msprof kernel-only）。第五轮以 UB diet 削减
  −40.9KB 手工预算（设计妥协：N/D staging 合并 + rowmat 删除）绕开后，宽块收益
  才得以实现（长 KV 反转 11.1x 领先、全域几何 8.53x）——绕法本身证明该差值
  直接决定宽块设计可达性与结构选型。
evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Design-layer finding（溯源，任务工作区允许失效：DESIGN §4.5 预算 178.2KB ≤ 192KB "✓" vs 实际 ~204.1KB；报文 `ub overflow, requires 1666048–1672192 bits while 1572864 bits available`）
  - 双路径最小复现（工厂直调 (4,512,32,128) causal fp16）：(64,64,1)→E6 替换 (64,256) FAIL vs (64,64,2)→逐字 bn_eff∈{64,144} OK
  - dim=64 smoke ~161.7KB 可过（预算-实际差在窄变体不越界）
  - 第二证（task 20260915T080600Z 第五轮 Stage 4 Phase 1）：probe_ub.py 编译探针 4 点标定（6797758）——(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%（auto-multi-buffer=false 下仍在）+ 漏计小 buffer 实证（DESIGN §4.5 漏计 ub_cond2 8.2KB）
  - 第三数据点（task 20260916T033847Z 第六轮 r9 precision_fix，a13585dc，**不计 occurrences**——本轮阻塞项为分派守卫组合而非分配差，且实测方向相反系反例非阻塞）：显式 alloc 结构（全 UB buffer 首维 = half 的逐 buffer 显式分配）手工核算 214080B vs BishengIR 实际 210080B（ratio 0.981，actual 低于手工）——系数结构依赖的反例数据点，正负两向偏差均无文档，强化「逐 buffer 分配可见性」诉求；核算镜像 + pre-fix 溢出断言：pattern-library/repro/PL-1.12-bn-clamp-bm-guard.py（系数边界同步记入 constants.md CONST-capacity-910B2C / attention.md PL-1.12 r9 update）
  - 第四数据点（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z Stage 4，1990aa9fe4，**不计 occurrences**——无分配差阻塞，属显式 alloc 结构依赖的跨族确认）：Expert 直差分因子链（mamba 族 MixCV）手工峰值 ~179KB ≈ 192KB 上限 93%；v3 变体 +16KB 手工 → 实测 195.1KB 报错（偏差 <2%，显式 alloc ≈ 手工核算跨工具链跨族再证）；bl=128 编译实测需 582KB/357KB vs 192KB 容量否决（constants.md CONST-capacity 第三数据点）
workaround: >-
  ① 窄 config 逐字路径交付（Stage 3 L1 门禁验证过的 traced 变体）；② 设计期预算按
  ×1.10–1.12 系数放大（constants.md CONST-capacity-910B2C 已收录，勿漏计小 buffer）；
  ③ 预算表不作为可编译证明——宽 config 上靶前编译探针校准（报文 requires bits 反推
  实际需求，与 VP-2026-0046 探针步骤同口径）；④ 贴限设计走 UB diet 主动削减
  （第五轮实证 −40.9KB 解锁宽块）。互链 pattern-library TRAP-UB-multibuffer-inflation
  第四证（VP-2026-0059，2026-09-15 两证合入）。
occurrences: 2
tasks: [multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z, multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z]
toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD），均 + CANN 8.5.0 + Ascend910B2C / 2026-09-15（跨 commit 复现）
status: recurring
proposal: >-
  目标层 BishengIR（per-AIV UB 分配可见性）。建议改动（二选一或组合）：① 编译期
  逐 buffer UB 分配清单输出（如 `bishengir-compile --dump-ub-allocation`：每 buffer
  名/尺寸/归属 AIV/额外分配项），使设计期手工预算可静态核验；② 至少将基础分配差
  文档化（差值构成〔疑 extra/sync buffers〕+ 推荐设计期补偿系数 ×1.10–1.12）。
  收益量化（两任务实测）：首证宽 config 不可编译致长 KV 0.78x 回退、bench 8/10
  失败被迫窄 config 交付；第二证需 −40.9KB 手工 diet（两次结构妥协）才解锁宽块，
  解锁后长 KV 11.1x / 几何 8.53x——分配差直接决定宽块设计可达性，可静态核验可
  消除「预算表 ≠ 可编译证明」盲区并避免 diet 类设计妥协。受影响算子：attention
  族宽块设计（两相位 causal 域实证）+ 一切 UB 贴限设计（ada_layer_norm nl 依赖
  形态同族，TRAP-UB 平台律条目）。issue 草稿（[npuir] 前缀，**须用户批准后建**）：
  [npuir] BishengIR per-AIV UB 分配差无文档、无逐 buffer 查询手段（≈手工预算
  ×1.10–1.12，auto-multi-buffer=false 仍在）。
created_by: multi_head_attention/_gqa_prefill_fwd_kernel Stage 5 集成期发现（distill 登记），2026-09-15
last_seen: multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 第六轮 r9 反例数据点精化（2026-09-16，不计 occurrences——本轮流变系守卫组合而非分配差阻塞，见 evidence 末行）
```

> **升级依据**：task 20260915T025507Z（Stage 5 集成期发现，28783f45）+ task 20260915T080600Z（第五轮 Stage 4 编译探针重校，6797758）——两个不同任务跨 commit 独立识别并定量标定，达 ≥2 任务阈值。capability 补齐提案已产出（见条目 proposal 字段：逐 buffer 分配清单输出 / 差值文档化二选一），待向用户显式列出并推动（外部 issue 须用户批准后建）。

---

### CG-2026-0002

```yaml
gap_id: CG-2026-0002
layer: BishengIR (bishengir-compile, cbuf→cbuf copy)
capability: >-
  L1（cbuf）域内搬运方向整体不支持：'hivm.hir.copy' op Unsupported copy from
  cbuf to cbuf! → 编译失败。两种触发形态：① 首证（Developer）——Pipelined 循环
  （NK >= 2，触发 auto-multi-buffer 多缓冲）之前对 L1(shared) buffer 的整 buffer
  T.vbrc 广播清零被错误 lowering 为 cbuf-to-cbuf 拷贝（NK == 1 时同一构造可正常
  编译）；② 第二证（Expert，2026-09-20）——显式 T.copy(l1_a, l1_b) 的 L1→L1
  方向直接硬失败（最小 Expert 探针实证，非 lowering 路径）——缺口本体为
  cbuf→cbuf 搬运方向整体不支持，不限 vbrc lowering。
blocked_algo: >-
  ① 尾块防护类构造受限（首证）：多块 KV 流水 kernel 无法在循环前对 GEMM 输入
  L1 buffer 做一次性清零；被迫把清零守卫收缩到"仅单块且部分块"（seq_len_kv
  < block_n）场景，多块场景依赖有限残留值安全性论证绕过。② L1 域内重排/组装
  类构造受限（第二证）：band 组装「先读入小 buffer 再组装到大 buffer」两跳形态
  不可实现——被迫单 buffer GM→L1 列偏移直载形态（每 band 块一次 GM 读，无 L1
  域内中转）或 GM 中转。
evidence:
  - 复现（首证任务 Stage 3）：seq_len=520（NK=9，has_kv_tail=True，循环前
    T.vbrc(0, v_shared)）→ bishengir-compile 报 'hivm.hir.copy' op Unsupported
    copy from cbuf to cbuf!（npuir 162:30）+ 'hivm.hir.load' root-alloc 连锁报错；
    同一 kernel 在 seq_len=16（NK=1）下编译运行通过；去掉该 vbrc 后 520 编译运行
    精度全通过（原 session 探针 /tmp/opencode/probe，provenance session-local 允许失效）。
  - 未文档化假设（首证）：docs/Tilelang.language/广播类文档（vbrc/brc）无多缓冲
    交互限制条款；估计依据为本任务探针实测。
  - 复现（第二证，task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z
    Stage 2 复检轮）：最小 Expert kernel（T.Kernel(1, is_npu=True) + T.Scope("Cube")
    + 两个 T.alloc_L1([64,64],"float16") + T.copy(l1_a, l1_b)）→ bishengir-compile
    err code 1 报同款 'Unsupported copy from cbuf to cbuf!'；三证合一——docs
    T.copy.md §2.3 搬运方向表无 L1→L1 方向（GM→UB/UB→GM/UB→UB/GM→L1/L0C→GM）、
    全仓 examples 零先例、探针实证（REVIEW.md 复检轮机械复核段；探针原件
    /tmp/opencode/probe transient，provenance 允许失效）。
workaround: >-
  ① 首证：清零守卫收缩为仅 seq_len_kv < block_n（单部分块）时执行（NK=1 不触发
  多缓冲）；多块尾块场景依据有限残留值安全性免清零。② 第二证：L1 域内「重排/
  组装」意图落为单 buffer GM→L1 列偏移 dst 直载（T.copy(ws[...,0:bl,0:ts],
  l1_band[0:bl,s0:s0+ts])，src/dst 双侧尾块裁剪）+ 单条 band gemm——探针 2 在
  4515de8 编译通过（与 TRAP-L1-band-dst-tail-overrun 联防）；或经 GM 中转。
  绕法知识条目化提案见 queue VP-2026-0095（pattern-library traps-compiler.md）。
occurrences: 2
tasks: [multi_head_attention/_gqa_prefill_fwd_kernel Stage 3 first_impl 2026-09-04, ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 2026-09-20]
toolchain_stamp: 首证 tilelang 67db6f3 (2026-09-04) + CANN 26.0.rc1 + 910B2C(IT21HMDB01-B2) + torch_npu 2.7.1；第二证 tilelang 0.1.2+4515de8 + CANN 8.5.0 + Ascend910B2C / 2026-09-20
status: recurring
proposal: >-
  目标层 BishengIR / Frontend API。建议改动（二选一或组合）：① 支持 cbuf→cbuf
  copy lowering（L1 域内搬运原语，消除「域内重排必经 GM」约束）；② 至少将该
  方向缺口文档化——T.copy.md §2.3 搬运方向表显式标注 L1→L1 不支持 + 报错信息
  指向推荐形态（单 buffer GM→L1 列偏移直载）。收益量化（两任务实测）：首证使
  尾块清零守卫收缩（多块场景裸奔依赖残留值论证）；第二证使 band 组装类结构
  被 forced 单 buffer 直载形态（每 band 块一次 GM 读）——L1 域内中转可消该类
  GM 重读。受影响算子：L1 域内重排/组装类（MixCV band 组装、多块流水清零守卫）。
  issue 草稿（[npuir] 前缀，须用户批准后建）：[npuir] cbuf→cbuf copy 不支持且
  无文档（L1 域内重排/清零构造不可表达，vbrc lowering 与显式 T.copy 双路径同报错）。
created_by: multi_head_attention/_gqa_prefill_fwd_kernel s3 attempt1 2026-09-04
last_seen: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2 复检轮（2026-09-20）
```

> **升级依据**：task multi_head_attention/_gqa_prefill_fwd_kernel Stage 3（2026-09-04，Developer vbrc lowering 路径）+ task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z Stage 2（2026-09-20，Expert 显式 T.copy 路径）——两个不同任务、双触发形态、跨工具链（67db6f3/CANN 26.0.rc1 → 4515de8/CANN 8.5.0）独立识别同一缺口（cbuf→cbuf 搬运不支持），达 ≥2 任务阈值。proposal 字段已产出，待向用户显式列出并推动（外部 issue 须用户批准后建）。

---

## Resolved（proposal / in-progress / fixed / closed / withdrawn 归档）

（暂无条目）

### CG-2026-0028

```yaml
gap_id: CG-2026-0028
layer: BishengIR（`hivm.hir.vreduce` / reduce 链的 UB 预算分配）+ Frontend（vsel 操作数形态）
capability: >-
  (a) **双输出整平面 argmax 的归约链 UB 单价 ≈237 B/平面元素**（含 auto-multi-buffer ×2），
  而单输出同族 C4（`reduce_max` + 原位平方 `reduce_sum`）仅 **32 B/元素**——索引侧的位置表 +
  eq 掩码 + 候选平面缓冲把单价抬高 ~7x，把 `ch` 从 64+ 压到 **8**；`ch=16` 需 1 618 432 bit
  （202 304 B）> 1 572 864 bit（196 608 B，`ub overflow, requires N bits while 1572864 bits available`）。
  `plane=49, N=8, C=2048` 下 `ch=8` 产生 `num_local = ceil(2048/40) = 52` 轮，实测 Task **262.8 µs**
  （远差于一般网格路径 144.4 µs）⇒ 该特化在双输出上不可采纳。
  (b) `(C, plane)` 二维 `T.reduce(dim=1)`（归约轴 = 最内轴）被**标量化**：`ch=64` 实测 scalar 98.1 µs /
  Task 237.0 µs / `UB_to_GM` 96 KB/blk（转置-布局往返）——同日历的 `(C,1,plane)+dim=2` 形态正常。
  (c) `hivm.hir.vsel` 的 **cond 不接受从更大 UB buffer 切出的 bool 区域**（`vsel op operands' shape are
  inconsistent`），同一 slice 作 `vcmp` 操作数合法 ⇒ 双输出池化「行级掩码直送 vselect」的 8→6 削 op 不可达。
blocked_algo: >-
  「整平面 reduce + 位置表 argmax」的 C4/双输出特化（DESIGN candidate #9 在索引型上的推广）——
  被迫换算法（改为「扁平整平面 + 沿平面 (1,C) 行记录扫描」；正确性可达但残余受转置往返所累，见 CG-2026-0024）。
  同时「免转置二维行归约」路线被标量化否决，构成双输出极分解的完整边界。
evidence:
  - 任务工作区（允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/perf_opt/probe_global_reduce.py`（`--ch 8 --check` 逐元素精确 16384/16384；`--ch 16` 报 `ub overflow, requires 1618432 bits while 1572864 bits available`；ch=8 msprof Task 262.795 µs）、`probe_global_2d.py`（ch=64 Task 236.985 µs / scalar 98.139 µs / `UB_to_GM` 96 KB/blk）、`logs/g2d_ch64.log`、`logs/pr_ch8.log`
  - 知识域：`pattern-library/constants.md` `CONST-dual-output-whole-plane-reduce-ub`（Tier 0，已回写）；`traps-compiler.md` `TRAP-hivm-vsel-cond-slice`（新条目，已回写）
  - 对照常数：同族 `adaptive_max_pool2d` 单输出 C4 的 `_UB_REDUCE_B_PER_ELEM = 32`（`TRAP-reduce-ub-per-element`）
  - docs/Tilelang.language/规约操作/T.reduce_max.md（未记载 UB 预算模型与归约轴形态的向量化判据）
workaround: >-
  双输出 1x1 用「扁平整平面 + (plane,C) 转置 + 沿平面 (1,C) 行记录扫描」替代整平面 reduce
  （实测 global-1x1 144.39 → 91.92 µs，-36.3%，逐位精确）；`ch` 由 UB 预算反推（`CH_GLOBAL=104`）。
  general 路径保留 C 轴向量轴 + 破纪录集合取 max 的三路累加器（每元素 7 条窄向量 op）。
occurrences: 1
tasks: [adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z]
toolchain_stamp: tilelang 0.1.2+67a04413fb61385611796f21803cabc68d316e23（commit 67a0441 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-15
status: open
proposal: none（未达 recurring；建议方向 = ① 公布 reduce 链 UB 预算模型（按归约链形态分档），或在 `ub overflow` 诊断中列出逐 buffer 明细；② 对 `(C,plane)` + `reduce(dim=1)` 给出标量化诊断；③ 放宽 `hivm.hir.vsel` 的 cond 操作数形态（支持 UB 子区域）；issue 须用户批准）
created_by: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z Stage 4（optimizer 自登记，2026-10-15）
last_seen: 2026-10-15 adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z Stage 4
```

### CG-2026-0030（2026-10-16 Stage 5 重试段补**形态②**：大 shape × fp16 触发同一后端段错误）

```yaml
gap_id: CG-2026-0030
layer: BishengIR（bishengir-compile 后端崩溃 — ConvertHFusionToHIVMPass 后继阶段 / IR opt）
capability: >-
  `bishengir-compile`（CANN 9.0.0）对特定 **shape（×dtype）取值组合**段错误，两个形态：
  〔形态①（2026-10-16 首轮 Stage 3）〕同一 Developer 骨架
  （persistent `T.Kernel(min(num_logical, cores))` + 核内 `T.serial` 任务循环 +
  `T.Pipelined` query 循环 + 运行时 extent 的 GM↔UB `T.copy`（目标操作数预清零）+
  `T.gemm`（全 tile extent）+ 逐行 1-D `T.atomic_add`）在 attention 形状
  `(B,S,H,H_kv,D) = (1, 32, 4, 4, 64)` 上编译时报
  `NPU IR opt failed: PLEASE submit a bug report ... Stack dump:`（段错误，无 MLIR 诊断），
  而 `S ∈ {16, 24, 40, 48, 56, 64, 96}` 全部编译通过。崩溃与 `(bm,bn) = (64,64) / (32,32)`、
  `heads`、`dim ∈ {64,128}`、`is_causal` 均无关（逐变体实测）；S 为 64 倍数的 manifest 档
  不受影响。
  〔形态②（2026-10-16 Stage 5 重试段，task …082329Z）〕同一后端对**大 shape × fp16** 段错误：
  `bench_multi_head_attention.py` 的 `llama-3.1-70b-long-float16` 档
  `(batch=1, heads=64, dim=128, seq_len=2048, fp16)` 在 fwd 与 bwd **两条 bench 上同时**
  报同一 `NPU IR opt failed: PLEASE submit a bug report ... Stack dump without symbol names`
  （栈顶 `__kernel_rt_sigreturn`），而同 shape 的 **bf16** 档与其余全部 shape 正常 profile+build
  ⇒ 触发维度是 **shape × dtype 的联合取值**。两形态的**共同本体** = 后端在特定 shape/dtype
  取值上崩溃且不产出 MLIR 诊断；**是否同一触发点未证**（无诊断可比对、触发骨架不同），本条
  按「同一后端版本上 ≥2 个形状/dtype 敏感触发点」登记。
blocked_algo: >-
  〔形态①〕受影响的是「小 S（含 32 的非 8 倍数域）」的 attention-backward 形态：该 shape 档
  在本工具链**无法产出可运行产物**（编译即崩），分层测试只能把它记为 warn-only（不构成数值
  证据）。对 manifest 交付域（S 为 64 倍数）无影响；对 `S = 32` 的边界/小 shape 用例是硬阻塞
  （被迫从断言档降级为记录档）。
  〔形态②〕受影响的是大 shape × fp16 档的 attention fwd/bwd bench：该档在报告里记为
  benchmark failed（不进性能统计），使单算子 report 降级 `partial`——正确性（12/12）与其余
  shape 的性能结论不受影响。潜在波及：小 S（含 32 的非 8 倍数域）+ 大 S fp16 档两侧；两次
  观察都在 attention 类 kernel 上，故适用域按「shape/dtype 敏感」表述，未泛化为全体算子。
evidence:
  - 复现（任务工作区，允许失效）：`examples/multi_head_attention/_gqa_bwd_kernel/repro/
    TRAP-bishengir-crash-s32.py`（知识域 repro，三态对照：A `S=32` 抛崩溃 / B `S=40` 通过 /
    C `S=24` 通过；`ALL ASSERTIONS PASS`）
  - 变体扫描：同任务 `history_version` 的 S 扫描（16/24/32/40/48/56/64/96）与 tile/heads
    变体（`(bm,bn)`、heads、dim、is_causal 四维），仅 `S=32` 崩溃
  - 运行日志：`python _gqa_bwd_kernel.py --level all` 的 L2 段
    `[L2] WARN (record only): (B,S,H,Hkv,D)=(1,32,4,4,64)`
  - 知识域条目：`pattern-library/traps-compiler.md` `TRAP-bishengir-crash-s32`（含 repro，
    重验触发词 `S=32: COMPILED` / `(batch1,heads64,dim128,seq2048,fp16): COMPILED`）
  - 〔形态② 复现（2026-10-16 Stage 5 重试段，task …082329Z）〕
    `examples/TileOPs/reports/tileops/20261016_082438_849619_MultiHeadAttentionBwdOp/{run.json,report.md}`（任务工作区，未上库，允许失效）
    （`summary.benchmark_tests=20` / `summary.benchmark_failed=2`；report §Failed Cases 两行
    ——`bench_multi_head_attention::test_mha_fwd_bench[llama-3.1-70b-long-float16]` 与
    `test_mha_bwd_bench[...]`，均为 `bishengir-compile` 段错误栈）；同报告 `benchmark.json`
    的 18 条记录中该 shape 仅 bf16 档存在（fp16 档缺失）；`profiler_fallback_count=2` 属另一
    现象（fwd msprof 降级 events），与本条无关。
    provenance 形态 = 单算子 report（需整包 fwd+bwd 与 TileOPs 报告链），暂无法最小化为
    知识域 repro ⇒ 形态② 在 pattern-library 侧标 `repro-missing`。
  - 未文档化假设：docs/ 无 bishengir-compile 形状相关缺陷清单；估计依据为两形态的实测记录
workaround: >-
  〔形态①〕该 shape 档按「工具链崩溃（非精度问题）」单独记录为 warn-only，不做精度断言；
  分层测试的 WARN 通道须区分 `[toolchain-crash]` 与 `[precision-record]` 两类（同一通道混记
  会让读者把后端崩溃误读为数值可疑）。设计侧无可用替代形态（崩溃在编译期，与 kernel 写法
  无关）。
  〔形态②〕该 bench 档按 task 策略**只记录不修复**（`report.md` §Failed Cases 留痕；
  `summary.benchmark_failed=2`），报告整体降级 `partial` 但 correctness 独立判定为 pass；
  可选的可用形态未验证（换 dtype 档 bf16 可正常 profile+build ⇒ bf16 是该 shape 下当前唯一
  可用档）。
occurrences: 1
tasks: [multi_head_attention-MultiHeadAttentionBwdOp-20261016T015430Z（函数级 _gqa_bwd_kernel Stage 3，形态①）, multi_head_attention-MultiHeadAttentionBwdOp-20261016T082329Z（Stage 5 重试段单算子 report benchmark，形态②——同 op / 同环境 / 同一任务谱系的 Stage 5 重试段，按同 op 谱系**不计独立 occurrences**，故 occurrences 维持 1）]
toolchain_stamp: >-
  tilelang 0.1.2 dev root build（/workspace/tilelang-mlir-ascend/build）+ CANN 9.0.0
  （/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile）+ Ascend910B3（AIC=20 / AIV=40 /
  UB 196608 B）/ 2026-10-16
status: open
proposal: none（未达 recurring；建议方向 = ① 上游定位两触发点（提供最小 .npuir 与
  crash stack，形态① 已有可最小化 repro）；② 在 `bishengir-compile` 崩溃时输出「已进入
  pass 阶段 + 触发 shape/dtype」的最小上下文，避免 `PLEASE submit a bug report` 无信息量
  ——形态② 连「哪个 IR 构造触发」都无从判断，只能记录；issue 须用户批准）
created_by: multi_head_attention-MultiHeadAttentionBwdOp-20261016T015430Z（_gqa_bwd_kernel Stage 3 自登记，2026-10-16）
last_seen: 2026-10-16 multi_head_attention-MultiHeadAttentionBwdOp-20261016T082329Z Stage 5 重试段单算子 report benchmark（形态② = `(B=1,H=64,D=128,S=2048,fp16)` 大 shape × fp16 段错误；形态① 见 Stage 3 L2 warn-only）
```
