# 已知运行时/数值/语义陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。编译器/解析器类陷阱见 [traps-compiler.md](traps-compiler.md)。
>
> **证伪协议（强制，canonical 见 INDEX.md §2，两者同步演进）**：
>
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。
> 2. **一切运行时/数值结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

---
id: TRAP-C9-taskqueue-async
kind: trap
family: [general]
apis: []
dtype: [fp32]
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C9 `TILELANG_ENABLE_TASKQUEUE=false` 异步 launch 损坏 fp32 数据

有效（截至 2026-08-28 build）。

---
id: TRAP-UB-dst-align
kind: trap
family: [general]
apis: [T.copy]
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### UB dst 非零起点切片 + 32B 倍宽度触发 VEC 对齐错误

有效（截至 2026-08-28 build；host pad 或 0 起点拷贝绕开）。

---
id: TRAP-fp16-opmath-golden
kind: trap
family: [elementwise, migration]
apis: [vcast, vadd, vsub, vmul]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: torch 2.9.0+cpu + tilelang-mlir-dev dev root build 2026-09-07
repro: repro-missing
---

### torch CPU golden 的 fp16 opmath 域分歧

torch.lerp（torch 2.9.0+cpu）对 fp16 输入经**fp32 opmath + 单次舍回**计算，NPU fp16 原生域三步链（逐步 fp16 舍入）与之差 ~2–3 ulp fp16，N=2^24、atol=rtol=1e-3 下违反率 ~0.125% 且与 shape/block_size/核数无关（纯舍入路径统计性质，非 tiling/同步缺陷）；bf16 golden 与 fp32 中转 kernel 逐位一致。对齐通解：`vcast(rint)` 升 fp32 → fp32 域 v-prefix 链 → `vcast(rint)` 单次舍回，差 ≤1 ulp fp16（rtol≥5e-4 即覆盖）；NaN/Inf 角点 IEEE 传播不受中转影响。复现（provenance，允许失效）：`python examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py --level L0`（fp16 违反）vs `python examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py --level all`（全过）。

---

id: TRAP-npuir-ref-backend-golden-capability
kind: trap
family: [migration, pool, window, golden]
apis: [F.max_pool2d, F.max_pool1d]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: max_pool2d-_max_pool2d_kernel-20261014T062320Z（首证，2D 直接观测）/ max_pool1d-_max_pool1d_kernel-20261014T082309Z（第二证，1D 泛化应用）/ max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z（第三形态：`return_indices=True` 静默损坏）
toolchain: 首证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c + torch_npu 2.9.0.post8 + torch 2.9.0 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；第二证 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；第三证 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313 + torch 2.9.0+cpu + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro-missing
---

### NPU 参考后端（torch_npu）自身的能力缺口——`F.max_pool*` 只支持 `dilation == 1`、`return_indices=True` 的 indices 静默损坏；纯选择类算子的 CPU golden 回退逐位一致

**缺口本体**（与 npuir 前端 / TileLangIR pass 缺口不同类）：`F.max_pool2d` 在 NPU 上经 `aclnnMaxPool2dWithMask` 实现，**只支持 `dilation == 1`**；`dilation != 1`（int 或 tuple 任一非 1）时在 `ref_program` 内即抛 `AclNN_Parameter_Error(EZ1001): dilation only support 1`，**被测算子未执行**（堆栈落在 `torch.nn.functional`）。判别信号 = 异常出现在 `ref_program` 且算子尚未执行 + `AclNN_*_Error` 前缀 → 归因为参考后端能力缺口，而非 kernel/迁移缺陷。首证实测：[INTEGRATE_FAIL]（`run.json` `correctness_failed=1`，`dilation=(2,1)` 用例；stamp 见 front-matter）。

**处置形态**：纯**选择**类算子（max/min/argmax/排序，无累加）把该用例的 reference 计算回退 CPU（`F.x(input.cpu(), **kwargs).to(input.device)`）——结果与设备输出逐位一致，`atol=rtol=0` 与用例集（含 `dilation>1` 覆盖）均保持原样（放宽容差 / skip / xfail / 删例均不构成处置）。首证 2D attempt 2 修复后本算子 rollup 22/22、`max_abs_err = 0.00e+00`。**含累加算子不可照搬**该回退（sum/avg/conv/gemm 的 CPU/设备浮点序不同）；回退前提是「输出 = 输入元素的纯选择，无算术合并」。

**第二证（1D 泛化，不同任务/不同算子/不同 stamp）**：max_pool1d 迁移的 Stage 0 脚手架按 max_pool2d 教训**预置**了同款回退（`examples/TileOPs/tests/ops/test_pool.py` 的 `MaxPool1dTest.ref_program`；任务工作区/集成包，未上库，provenance 允许失效），Stage 5 单算子 report 本算子 rollup **34/34**、`max_abs_err = 0.00e+00`，`ecg-cnn-dilated`（`dilation=2`）用例实跑走该回退且无 `AclNN_*_Error`。限定：1D 侧实测的是「补救路径（CPU golden 回退）跨算子/跨 stamp 泛化成立」，未再触发 1D 的 `dilation>1` 后端报错本身（回退已前置）；缺口本体为 2D 侧直接观测。

**第三形态（2026-10-14，1D **双输出**变体——**静默损坏、不抛异常**）**：`F.max_pool1d(..., return_indices=True)` 在 NPU 上 **indices 输出损坏且无任何异常**——`dilation == 1` 时返回 `torch.int8 (N, C, kernel_size, L_pad)` 的**内部 mask buffer**（设备直探实测 `(1,8,3,64)`，含 0/1 噪声）而非契约要求的 `int64 (N, C, L_out)`；`dilation != 1` 另抛 `AclNN_Parameter_Error(EZ1001)`（= 首证形态）。**values 输出正常**，故「跑得通」不构成 golden 可用判据——错误直到测试 harness 的 tensor 比较处才以 `RuntimeError: The size of tensor a (16) must match the size of tensor b (64)` 浮出（形似 shape/runtime 失败）。识别信号因此须扩为「异常 `AclNN_*_Error` **或** 后端返回 tensor 的 dtype/shape 与算子契约不符（静默损坏）」；两形态同属「NPU reference 后端能力缺口」子类，处置同为纯选择类算子走 CPU golden（该子类已落盘 integrator 失败分类表，见 queue VP-2026-0179）。本形态的**覆盖面超出按参数预埋的回退**：损坏对**所有** dilation 成立，脚手架按「`dilation != 1` 受限参数」预埋的 CPU golden 回退覆盖不足，须对**纯选择类算子的多输出/`return_indices` 后端支持**先实测再定回退范围。实测与修复：探针 `i.shape/i.dtype` → `(1,8,3,64) torch.int8`；迁移算子对 CPU golden 逐位一致（`val_max_diff = 0.000e+00`、`torch.equal(idx_cpu, idx_ref) == True`）⇒ kernel 正确、仅 NPU reference 不可用；reference 计算路径改为无条件 CPU golden 后 per-op **47/47**（原 40/47，`max_abs_err=0.0`、int64 索引精确）、benchmark 36/36（24 有效 msprof）。溯源：`examples/TileOPs/reports/tileops/20261014_122210_526209_MaxPool1dIndicesFwdOp/run.json`（失败轮）与 `.../20261014_122557_915676_MaxPool1dIndicesFwdOp/run.json`（通过轮）+ 集成包 `integration_log.md#调试历史`（任务工作区，未上库，provenance 允许失效）。

**配套**：Stage 0 脚手架期的 golden 可用性自检见 queue VP-2026-0178（集成期失败回交/扩权成本 ≈17min 实测）；同族另一类「设计层 vs 后端」边界（窗口池化退化特化条件）见 algorithm-candidates.md ALG-pooling。**不登记 `capability-gaps.md`**——AGENTS.md 的缺口层枚举（Frontend API / TileLangIR pass / BishengIR / tladapter / runtime）不含 PyTorch-NPU 参考后端，故按 pattern-library 事实条目 + 流程提案双落点处置。

---
id: TRAP-load-nd2nz-strided
kind: trap
family: [attention, expert, cube]
apis: [T.load_nd2nz, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-load-nd2nz-strided.py
---

### `T.load_nd2nz` 对跨步（非尾二维连续）src 区域静默平坦误读

BSHD [B,S,H,D] 按固定头取 (S,D) tile（区域 [1,real_m,1,dim]，dim1 步长 H·D）被按「基址起平坦连续内存」读取——无告警无报错，数值表现为整块乱值（ws_s 与任何 head 的 QK^T 均不匹配；flat-read 假设逐点复现 got[i,j]==flat_q[i]·flat_k[j]）；`T.copy` **base+size 形态**对相同跨步区域同样错误（diff 3.4），**slice 形态** `T.copy(q[bz, s_lo:s_lo+real_m, by, 0:dim], l1[0:real_m, 0:dim])` bit-exact 正确且同为 PIPE_MTE2 单次搬运——跨步 GM 区域装载的实测正确形态为 slice 形态 T.copy（文档「src 支持 2-4D tensor」未标注连续性约束；能力缺口登记 CG-2026-0004）。复现：`repro/TRAP-load-nd2nz-strided.py`（知识域——slice 绕法 bit-exact 断言）；原 session 探针（provenance，session-local 允许失效）：probe_stridecopy.py（slice vs base+size 对照）、probe_l1slot.py——关键发现镜像于任务 debug_log D1/D2。

---
id: TRAP-T-copy-region-semantics
kind: trap
family: [attention, expert, general]
apis: [T.copy, T.transpose]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-T-copy-region-semantics.py
---

### `T.copy` 区域语义三规则

① 标量基址 + size 的 extents **前向补 1**（size=[1,N] 落末维、[N,1] 落倒数第二维；`tilelang/language/copy.py` L42-95 源码语义）——lse 写出方向由它决定；② **src/dst 区域元素数不匹配时引擎按 src 搬运并越界写坏相邻 GM**（`[0:half]` 源写 `[0:real_m_half]` 目的区砸坏相邻 tensor，症状酷似跨 pipe 竞态——仅部分核数据坏、set_flag/wait_flag 无效；判据 = 溢出值恰为源 buffer 内容；只在尾块位形 Sq%bm≠0 出现，整除 shape 全量测试发现不了；正确形态 = src 切片 `[0:real_rows]`）；③ [N,1] UB 源 + size=[1,N] 按步长越界读（MTE DDR fault）——行向量写 GM 尾维连续区先 `T.transpose` 到 [1,N]（Developer shared 源不需要，UB 源必须）。**推荐通用形态：src/dst 双显式 slice**。复现：`repro/TRAP-T-copy-region-semantics.py`（知识域——合法 slice 形态断言）；原 session 探针（provenance，session-local 允许失效）：probe_lsemulti2.py、probe_copyforms.py——关键发现镜像于任务 debug_log D2/D3。**〔2026-09-10 重验注记〕**repro 转正重跑（repro/TRAP-T-copy-region-semantics.py）：slice 形态 PASS 存活；历史 base+size=[HALF,D] 标量基址形态（原会话通过）在当前工具链复现 MTE DDR fault——按版本戳规则该形态**待重验**，条目结论以 slice 形态为准。

---
id: TRAP-zero-input-crash
kind: trap
family: [expert, general]
apis: [T.prim_func]
dtype: [bf16, fp16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-zero-input-crash.py
---

### 零输入 kernel（out_idx-only）运行期必崩 "MTE DDR address out of range"

启动参数错乱——崩溃形态与真实越界写完全同款，曾误诊为拷贝形态 bug 消耗 3 轮探针；探针 kernel 保留 ≥1 个输入 tensor 后同形态全过。诊断「部分核/部分行数据坏」先做元素级 fp64 对照判定哪侧偏离真值（本例 golden 才是偏离方：|kernel−true|=4.3e-4 vs |golden−true|=5.8e-3，bf16 P 量化噪声放大），避免在错误层面（跨 pipe 竞态假设）空转。复现：`repro/TRAP-zero-input-crash.py`（知识域——dummy input 绕法断言）；原 session 探针（provenance，session-local 允许失效）：probe_copymin.py 加 dummy input 前后对照——关键发现镜像于任务 debug_log D2/D6。

---
id: TRAP-vrsqrt-plain-precision
kind: trap
family: [norm, reduction, elementwise]
apis: [T.vrsqrt, T.vsqrt, T.vdiv]
dtype: [fp32, fp16, bf16]
device: 910B2C
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
repro: repro/TRAP-vrsqrt-plain-precision.py
---

### `T.vrsqrt` plain 模式为近似指令（max rel err ~2.9e-3）

npuir 上 `T.vrsqrt` lower 为 plain 近似 Vector 指令：输入几何扫描 [1e-6, 1e6] 实测 max rel err 2.87e-3（返回值量化到 ~10–11 位有效数字，如 1010/1024）。传播效应（ada_layer_norm Stage 3 实测）：fp32 1e-5 门禁下 87% 元素违反、fp16 1e-3 门禁下 6%、bf16 1.6e-2 宽容差掩盖——「fp32 大面积违反而 bf16 全过」是常数相对误差（近似指令）的指纹，属舍入路径性质而非数据依赖/同步缺陷。`examples/norm/layer_norm.py` 通过的 1e-2 容差会掩盖该问题，不构成 vrsqrt 精度佐证。

**绕法**（ada_layer_norm 交付形态）：`T.vsqrt` + `T.vdiv` 组合（sqrt(v)/v ≡ 1/sqrt(v) 实数恒等，两 op 全精度）实测 max rel err 1.07e-7；修正形式 Newton×2 迭代亦可达 6.3e-8（代价 12 个微型 op）。含 rsqrt 且容差 <1e-2 的算子（layernorm/rmsnorm/softmax 归一化族）在本工具链上以组合形态达标。定位手法：中间量分层导出（`out_idx` 多输出落 GM + fp64 精确值对照）单次运行把误差定位到具体指令（ada 案例：mean/d 精确 1.1e-7 而 rstd 偏 1.57e-3 ⇒ 误差独占于 vrsqrt 段）。配套事实：文档文件名与 API 导出名存在偏差是常态（T.rsqrt.md → `T.vrsqrt`；T.vLn.md → `T.vln`），`examples/` 实调代码是 API 名核对入口。

复现：`python repro/TRAP-vrsqrt-plain-precision.py`（断言量级：raw > 1e-3 现象存在 + bypass < 1e-6 绕法通过；工具链修复后首断言翻转即条目推翻信号，届时刷新版本戳）。溯源（provenance，允许失效）：`examples/ada_layer_norm/_ada_layer_norm_kernel/`（Stage 3 分层探针 + Implementation Notes attempt-1 precision fix 段；repro 原件 `examples/ada_layer_norm/_ada_layer_norm_kernel/repro/TRAP-vrsqrt-plain-precision.py`）。

---
id: TRAP-L1-band-dst-tail-overrun
kind: trap
family: [runtime, l1, mixcv]
apis: [T.copy, T.alloc_L1]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑：单跳直载 + ts 双侧裁剪形态编译探针通过，bf16×非整除边界用例全过）
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### L1 band 组装 dst 列区间未按尾块裁剪 → 越界写污染相邻 L1（布局敏感潜伏缺陷）

- **形态**：向 `[bl, Q]` L1 band buffer 逐块搬运时写 `l1[0:bl, s0:s0+bs]`——末 s-block 的 `s0+bs > Q`（Q%bs≠0）越出缓冲，越界写破坏相邻 L1 分配。**触发依 L1 布局而变**：fp16 同 shape 靠布局运气通过（L0 PASS 假象），bf16 布局命中关键 operand（max_diff 7.8e-3）——单 dtype 门禁放行双 dtype 契约的潜伏缺陷。
- **绕法**：src/dst 列宽同步裁剪 `ts = T.min(bs, Q−s0)`（`ws[..., 0:bl, 0:ts] → l1[0:bl, s0:s0+ts]`）；行维可保留 [0:bl]（gemm size 的 tmc 排除 stale 行，输出裁剪丢弃）。
- **教训**：分支精度回归须含 **bf16 × 非整除 shape** 组合（本缺陷潜伏 4 轮，仅 Boundary 用例拦截）；另注：tilelang 磁盘缓存会在源变更后命中旧二进制（同 max_diff 复现"修复无效"假象）——kernel 源变更后清 `~/.tilelang/cache`。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log §6.1；logs/final_level_all.log 修复前后）。

---
id: TRAP-DEVMODE-PERSIST-GEMM
kind: trap
family: [mixcv, expert, persistent]
apis: [T.Kernel, T.gemm, T.Scope, T.vmul]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 3）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（npu-smi 26.0.rc1）/ 2026-09-17
repro: repro/TRAP-DEVMODE-PERSIST-GEMM.py
---

### Developer 模式 + persistent 分核 + gemm 混排运行时崩溃（"unaligned UUB addresses"）

- **现象**：persistent `T.Kernel(24, is_npu=True)` 核内 `T.serial` 多任务 + `T.gemm` + v-prefix 向量 op 混排的 kernel 在 Developer 模式（auto CV-split）**运行时崩溃**——"Illegal instruction, which is usually caused by unaligned UUB addresses"（vector core exception，retCode=0x31）；Expert 模式（显式 `T.Scope("Cube")`/`T.Scope("Vector")` + alloc_L1/alloc_L0C/alloc_ub + sync_block_set/wait）同结构正常。
- **判定依据（仓库实态扫描）**：全仓 24 处 `is_npu=True` 中 Developer 模式仅用于非 persistent 网格（`examples/flash_attention/flash_attn_npuir_dev.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`），persistent 混合算子（先例：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/`〔任务工作区，溯源〕、`examples/deepseek_v32/sparse_mla_fwd_exp.py`）全部 Expert——「persistent 分核 + gemm」落在 Developer 支持域之外（docs 无 Developer/Expert 适用范围条款；未文档化假设，依据本任务双模式 repro + 全仓用例扫描）。
- **模式切换实证**：本任务 user_requirement 指定 Developer 模式，Stage 3 首实现即崩；切换 Expert（GQA/sparse_mla 先例 + `pass_configs={TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False}`，不关闭时多 UB buffer 被 pass 重排/重作用域 → codegen "cannot find variable"）后 L0/L1/L2/Boundary 全过——Expert 双 Scope 是该结构类的可用形态（结构级绕法模式见 attention.md PL-1.16；能力缺口登记 CG-2026-0010）。
- 复现：`python repro/TRAP-DEVMODE-PERSIST-GEMM.py`（Expert 路径数值断言通过 + Developer 路径崩溃观察——设备故障 out-of-band 即现象本体，工具链修复后 Developer 路径存活即条目推翻信号，届时刷新版本戳）。溯源（provenance，允许失效）：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/repro/DEVMODE_PERSIST_CRASH.py`（双模式对照原始探针）+ 同目录 RETROSPECTIVE.md Stage 3 章节。

---
id: TRAP-BENCH-CONFIG-CALIBRATION
kind: trap
family: [benchmark, harness, stage4]
apis: []
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4 第二轮 R7——续跑口径断裂）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro-missing
---

### 采数 harness 默认配置与交付配置断裂（续跑/翻转场景的 +19~21% 假回退）

- **现象**：调优产物的 kernel 内嵌 TUNED_DEFAULT_CONFIG（如 block_n=128），但采数 harness（bench/runner）的 argparse 默认回落到自己的 fallback（如 `min(64,N)`）——第一轮靠**显式传参**保持一致（而 opt_log 采集命令模板漏记该参数）；第二轮续跑者按模板走 runner 默认 → 所有分支数据系统性偏慢 +19~21%（w2 258.86 vs 217.65µs），初判为「设备漂移 ±20%」，实际是配置口径断裂。**双配置同 session A/B 探针（bn=64/128：261.55 vs 226.26µs，+15.6%）先于「设备漂移」假设**。
- **绕法**：① bench harness 的默认配置从**被测 kernel 模块的 TUNED 常量**解析（`getattr(mod, "TUNED_DEFAULT_CONFIG", {}).get(...)`），baseline 无常量时 fallback 兼容——使「交付配置=测量配置」成为机械保证；② 续跑/翻转轮次第一步 = current best 的同 session 重测校准（同口径基准行）；③ 调优日志的采集命令模板必须与实际执行完全一致（含全部显式参数）。
- **教训**：跨 session 的 perf_records 对比，「设备状态漂移」与「配置口径断裂」症状相同（系统性偏移）——配置探针成本 ~2 分钟，应优先排除；跨 run 双态（BP_run_state_bimodality ±3~5%）解释不了 ±20% 量级的偏移。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log R7 口径断裂段；profiles/round7_bn_check/；bench.py R7 修复）。

---

id: TRAP-torch-oracle-maxpool1d-empty-window
kind: trap
family: [pool, window, golden, migration]
apis: [F.max_pool1d]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z（Stage 3 repro 转正；Stage 2 问题 2 首记该 regime）
toolchain: tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…）+ torch 2.9.0+cpu + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-torch-oracle-maxpool1d-empty-window.py
---

### 空窗口（无界内 lane 的池化窗口）regime——torch 不可作 oracle，源/torch/哨兵三方行为互异

- **regime 与触发条件**：1D 窗口池化的窗口输入覆盖集合是等差子集 `{ow·s − p + kw·d}`；`d ≥ 2 ∧ p > 0` 时窗口可全落 `[0, L_in)` 之外（manifest-legal；最小例 `L_in=1, k=2, s=1, p=1, d=2` → lane `{−1, 1}`；ceil 例 `L_in=1, k=2, s=3, p=1, d=3` → lane `{−1, 2}`）。「`p ≤ ⌊k/2⌋` ⇒ 窗口非空」只在 `d == 1` 成立——非空窗口的真实充分条件是 `d == 1 ∧ p ≤ ⌊k/2⌋ ∧ s > 0`；`d ≥ 2` 下不成立（枚举 `L_in ≤ 13, k ≤ 6, s ≤ 5, p ≤ ⌊k/2⌋, d ≤ 6, ceil ∈ {F,T}` 得 89 组 `(config, ow)` 反例）。
- **三方行为实测**：① torch 2.9.0 CPU `F.max_pool1d(..., return_indices=True)` 值 = `-inf` 而索引**越出 `[0, L_in)`**（实测 `1`/`2`，其 CPU kernel 的 loop-artifact）；② 源状态机（`max_pool1d.py` 第 250/252/257 行：`max_val=-inf` 初值、`max_idx=0` 初值、越界 lane 跳过）输出 `(val, idx) = (-inf, 0)`；③ 设计侧 `BIG=L_in` 哨兵给出 `L_in`。**排除空窗口后的 3 195 组随机配置上 torch 与源/设计逐位一致（0 失配）**⇒ 该 regime 是唯一分歧域。
- **测试设计陷阱**：照字面 `torch.equal(indices, ref)` 会把一个**正确** kernel 在空窗口用例上判 fail；此类用例的可用判据只有「与源对齐的文档化契约」+ 显式记录 torch 越界作为 premise 证据（本算子 L0-15、`verify_equiv.py#empty_window_evidence()`、转正 repro 三段一致）。非空窗口下 torch 仍是合法 oracle（对照组 `idx=[0,2,2,4,5,5]` 全在界内）。
- **契约对齐的低成本实现形态**（本算子实测）：`idx32` epilogue 追加 `max_idx==BIG ∧ ¬has_nan ⇒ idx=0`，复用已存在的 `(BL,CH)` 布尔缓冲、不新增 UB 驻留（本族容量裕度薄，见 layout.md PL-1.4）——使「与源等价」在全部 manifest-legal 配置下成立（`EW` 分支 3 配置 × 3 dtype × 3 数据全 `EQUIV_PASS`）。
- **证明**：`python repro/TRAP-torch-oracle-maxpool1d-empty-window.py`（exit 0；premise + torch 越界 + 契约值 0 且 ≠ torch + 非空窗口对照组四段断言；纯 torch CPU、无 NPU、不 import 任务模块）。溯源：`examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/`（DESIGN.md §0.1/§8.3 L0-15、RETROSPECTIVE.md Stage 2 问题 2、REVIEW.md 问题 2——任务工作区，未上库，provenance 允许失效）。

---

id: TRAP-pool-index-contract-dual-output
kind: trap
family: [pool, window, argmax, golden]
apis: [F.max_pool2d, F.max_pool1d, F.adaptive_max_pool2d]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z（首证，2D 扁平索引）/ max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z（第二证，1D 单轴索引 + padding/空窗口退化域）
toolchain: 首证 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + torch 2.9.0+cpu + CANN 9.0.0 + Ascend910B3 / 2026-10-14；第二证 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…）+ torch 2.9.0+cpu + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro-missing
---

### 带索引池化（`return_indices=True`）的源语义契约——索引是「空间域展平位置」，非坐标对

- **索引编码**：2D —— `indices` = `H×W` 平面**行主序扁平下标** `ih*W_in + iw`（int64）；1D —— `indices` = 沿 L 轴的输入位置 `iw`（int64）。两者都是「输入张量在空间域内的展平位置」，**非** `(ih, iw)` 坐标对。
- **并列（tie）**：取**首个**命中（源为严格 `>` 更新的首写生效语义）。
- **NaN**：窗口含 NaN 时值 NaN，索引取**最后命中的 NaN**（非首 NaN、非哨兵）。
- **边界/退化域**：① 前 padding 下全 `-inf` 窗口的索引 = 窗口内**首个有效元素**的 `iw`（可能 `> 0`，源 `first_valid` 首次 seed，`max_pool1d.py` 第 266–269 行）；② 空窗口（无界内 lane，`d≥2 ∧ p>0`）契约 = `(val, idx) = (-inf, 0)`（`max_idx` 初值 0 + 越界 lane 跳过；torch 在此域返回越界索引、不可作 oracle——见 TRAP-torch-oracle-maxpool1d-empty-window）；③ `{+inf, -inf}` 同窗 → 值 `+inf`（非 NaN）。
- **与设备 golden 的三方分野**：torch-CPU（IEEE 相等取首现 + 本节编码）、torch-NPU 设备 argmax（`±0.0` 用可区分全序）、NPU 硬件 reduce（NaN 传播 → 哨兵 `2^30`）三方各不相同（见 traps-runtime.md TRAP-argreduce-edge-semantics）——跨设备 golden 对齐逐个实测记录。
- **证据**：首证 2D 12 行 torch 2.9.0 CPU 实测表 + 集成侧 NaN 窗用例（rollup 7/7、`avg_max_abs_err=0.0`、int64 索引 `torch.equal` 精确）；第二证 1D `DESIGN.md#§0.1` 11 行实测（`semantics_evidence()` 逐行确认）+ Stage 5 集成 per-op 47/47（`max_abs_err=0.0`、索引精确）。溯源：两任务工作区，未上库，provenance 允许失效。

---

id: TRAP-index-reduce-extremum-direction
kind: trap
family: [pool, argmax, reduction, vectorize]
apis: [T.vmax, T.vmin, T.vcmp, T.vselect, T.reduce_min]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261014T010230Z（首证：v0 `vmin` → v1 破纪录集 `vmax`）/ max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z（第二证：1D 两遍法 + 候选集口径澄清）
toolchain: 首证 tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + CANN 9.0.0 + Ascend910B3 / 2026-10-14；第二证 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro-missing
---

### 索引归约的极值方向判据——方向由「源写入语义 + 候选集」共同决定

- **改写前提**：把源的「顺序条件覆写 / 记录」改写为「集合上取可交换极值」前，先判源写入语义——**首写生效**（严格 `>`，并列取首）vs **末写生效**（无条件覆写，如 NaN 分支覆盖）。
- **两种等价形态（候选集不同 ⇒ 方向名不同，不可混搭）**：
  - 单遍（累加器未定稿）：候选集 = **破纪录集合** `{i : x[i] > max(x[<i])}`，取该集合的**位置最大值**（`vselect(gt, flat_idx, -1)` + `vmax`）。误取 `min` 得「窗口内首个局部峰值」索引——首证实测：v0 `vmin` 首轮 **20 426 mismatch**，`rand` 组 123/144 索引错而 `all_equal`/`zeros`/`all_neg_inf` 退化形态全过；改 `vmax` 后 340 例全 `EQUIV_PASS`。
  - 两遍（累加器定稿后）：候选集 = **等式集合** `{iw : x[iw] == acc}`，取**位置最小值**（`vselect(eq, iw, BIG)` + `reduce_min`，`BIG` 哨兵）。第二证实测：两遍法（第一遍值/NaN/last-NaN，第二遍 first-hit）消除了「winner 判据与累加器更新的相对次序」依赖，8 配置 × 6 kind × 3 dtype 全 `EQUIV_PASS`，而次序颠倒的误形态在 39/72 组被检出。
  - 两形态数值等价（`max(破纪录集) = min{x[i]==acc}` ⇒ first-hit），但「候选集 A + 取 min」与「候选集 B + 取 max」的混搭即错——判据表述须绑定候选集，不能只看算子名（`vmin`/`vmax`）。
- **两遍法的前提**：访问序 `iw` 单调递增（`d_W > 0`）；访问序非单调时 last-record 须按访问序计算，不能直接取 `min`/`max`。
- **last-NaN 方向**：末写生效（无条件覆写）= NaN 位置集合的**最大值** `max{iw : isnan(x[iw])}`（两证一致）。
- **D-1 用例形态要求**：须含「多个严格递增破纪录元素」的窗口（2D `H*W ≫ 1` 的 `rand`；1D 长 `k`）；`all_equal`/`zeros`/`all_neg_inf` 形态在错误方向下也可能全过。
- 溯源：首证 `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/{DESIGN.md#§1.6.1 N2/N3, REVIEW.md#问题 1}`；第二证 `examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/{DESIGN.md#§0.6 R3, §1.6.1 N2/N3(-ord), verify_equiv.py N2N3/N2N3-ord}`（均任务工作区，未上库，provenance 允许失效）。

---

id: TRAP-argreduce-edge-semantics
kind: trap
family: [reduction, argmax, argmin, pool]
apis: [T.reduce_min, T.max, T.vcmp, T.vselect]
dtype: [fp16, bf16, fp32]
device: 910B2C, 910B3
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z（首证，910B2C / CANN 8.5.0）+ argmin-_argreduce_kernel-20261014T151121Z（第二证，910B3 / CANN 9.0.0；覆盖 ②）
toolchain: 首证 tilelang 0.1.2 dev root build 2026-09-24 + torch 2.x（npu 与 CPU 对照）+ CANN 8.5.0 + Ascend910B2C / 2026-09-24；第二证 tilelang 0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce（dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro-missing
---

### argreduce 边界语义三基准分野——±0.0 全序、NaN 传播→哨兵 `2^30`、±inf 精确

- **① ±0.0（单源，第二证未复测）**：torch-NPU 设备 argmax/argmin 对 `±0.0` 用**可区分全序**（argmax 偏好 `+0.0`、argmin 偏好 `-0.0`），与 torch-CPU（IEEE 相等取首现）不一致——cross-device golden 对齐时 `±0.0` 混合行是已知分歧点。首证实测（三 dtype 一致）：`[0,-0,0]` → npu argmax=0 / argmin=1；`[-0,0,-0]` → argmax=1 / argmin=0；CPU 全 0。
- **② NaN 传播 → 哨兵 `2^30`（两证）**：NPU 硬件 reduce **传播 NaN**（非 fmin/fmax 的 NaN-忽略语义），掩码匹配式 argreduce（`reduce_min`/`vmax` + `vcmp` 链）在 NaN 输入行上全匹配失败 → 输出**恰为哨兵 `BIG = 2**30 = 1073741824`**；torch 返首 NaN 索引（如 100）。首证为 4 位形 `2^30`（910B2C / CANN 8.5.0）；第二证为两次独立实测（910B3 / CANN 9.0.0）：`kernel=[1073741824, 63, 33, 152]` vs `torch=[100, 63, 33, 152]` 与 `kernel=[1073741824, 234, 105, 186]` vs `torch=[100, 234, 105, 186]`——同一实测行同时证明「掩码链 vs torch 首-NaN 索引」结构可分（即该行可作结构区分器）。
- **③ ±inf**：精确（无舍入分歧）。
- **判读与处置**：`2^30` 恒值 / 全哨兵列是「匹配链全失败」的指纹（与竞态的非确定行数、步长错位的固定行错并列，见 precision-patterns.md 索引失配指纹族）；含 NaN 的负载在 L2/边界用例以 **INFO 记录**行为差（记录不断言崩溃，golden 选择与容差设计据此分野——torch-CPU golden / torch-NPU 设备实现 / 硬件 reduce 语义三方各不相同，设计期逐个实测记录比统一假设更诚实）。哨兵恒值行不可与「匹配链断裂」的缺陷指纹混判：本行是**语义分野**而非 kernel 缺陷。
- 溯源（任务工作区，未上库，允许失效）：首证 `examples/argmax/_argreduce_kernel/DESIGN.md#§9.2-R1/R2`；第二证 `examples/argmin/_argreduce_kernel/{DESIGN.md#§9.2-R1, _argreduce_kernel.py:927-946（L2 NaN record）}`。与 TRAP-index-reduce-extremum-direction（方向由候选集决定）、TRAP-pool-index-contract-dual-output（池化索引契约）同属索引/语义分野族。

---
id: TRAP-out-idx-uint8-dtype-fallback
kind: trap
family: [quantization, elementwise, general]
apis: [out_idx]
dtype: [uint8, int8]
device: 910B3
status: verified
origin_task: fp8_quant-_fp8_quant_kernel-20261014T153942Z（op 级 fp8_quant-fp8_quant-20261014T152721Z）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（dev root build）+ torch 2.9.0 / torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/TRAP-out-idx-uint8-dtype-fallback.py
---

### `@tilelang.jit(out_idx=...)` 对非白名单输出 dtype（uint8 字节载体）静默 fallback 到 `torch.float32`

- **现象**：出参 torch dtype 由 `tilelang/jit/jit_npu.py::_tvm_dtype_to_torch` 的白名单映射（float16/32/64、bfloat16、int8/16/32/64、bool）决定，**无 uint8**，末尾 `mapping.get(dtype_str, torch.float32)` 兜底。prim_func 声明 `uint8` 输出 buffer 时，kernel 仍按 i8 memref 写 GM，而 `out_idx` 分配的返回张量是**同逻辑 shape 的 float32**——shape 对、dtype 错、内容按 4 字节解释，**不报错、不告警**。
- **影响面**：fp8/低比特软件编码类算子的**字节载体输出**（uint8 出参 + host `.view(fp8)`）。照设计写 `out_idx=[...]` 会得到不可用返回值；改用 float32 出参则丧失 1 字节/元素载体契约。
- **绕法**：字节载体**一律显式传输出张量、不声明 `out_idx`**（由调用方分配 `torch.uint8` 缓冲），实测逐位一致、无性能/精度代价。
- **证明**：`python repro/TRAP-out-idx-uint8-dtype-fallback.py`（exit 0；断言 ①`out_idx` 返回 dtype == float32 而 prim_func 声明 uint8、②显式传 uint8 张量往返 bit-exact）。
- 族内位置：与 TRAP-zero-input-crash（out_idx-only 形态）同属出参分配族但缺陷面不同；能力缺口本体见 `.agents/evolution/capability-gaps.md#CG-2026-0019`。溯源（任务工作区，允许失效）：`examples/fp8_quant/_fp8_quant_kernel/{RETROSPECTIVE.md Stage 3, repro/TRAP-out-idx-uint8-dtype-fallback.py, _fp8_quant_kernel.py}`。

---
id: TRAP-fp16-op-affine-double-rounding
kind: trap
family: [norm, migration, golden, test-design]
apis: [F.group_norm]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: group_norm-_group_norm_kernel-20261015T152755Z（Stage 3 纯 CPU 探针转正）+ group_norm-group_norm-20261015T152158Z（Stage 5 NPU 集成 report 级确证）
toolchain: tilelang 0.1.2+9307929320d + CANN 9.0.0 + Ascend910B3 / 2026-10-15～16；torch 2.9.0+cpu + torch_npu 2.9.0.post8
repro: repro/TRAP-fp16-op-affine-double-rounding.py
---

### fp16「kernel 舍回 dtype + op 层 dtype affine」分层产生的二次舍入——真实 op 复刻超 atol=rtol=1e-3，而 kernel 本体逐位正确

- **形态与触发条件**：算子契约把 affine 分成两层——kernel 输出**已舍回输入 dtype** 的归一化张量，op 层再施加 per-channel `(C,)` `weight/bias`。op 层若在**输入 dtype（fp16）**里算 `y = y·w + b`（乘-舍-加-舍两次舍入），则归一化张量的量化误差 `e1 ≤ 0.5 ulp(n)` 被 `|w|` 放大为 `e1·|w|`；当 `n·w + b ≈ 0` 时绝对误差上界是 `ulp(|n·w|)` 而非 `ulp(|result|)`，`atol` 覆盖不到——表现为**少量元素**（本例 ≤0.3%）超差而非整体偏移。golden（`F.group_norm(x.float(), …, w.float(), b.float()).to(dtype)`）走 fp32 opmath + 单次舍回，与该路径不等价。
- **Stage 3 纯 CPU 定量（本轮 repro 实跑复现）**：`GroupNorm` 最小复刻（n=16/c=32/d=256/g=8）8 seeds —— dtype affine 违反 **314** 个元素、fp32 affine + 单次 `.to(fp16)` **6**、kernel 级**单位 affine** 对纯归一化 golden **0**（逐位）。
- **Stage 5 NPU / report 级确证（同一算子）**：`GroupNormFwdOp` 集成 report attempt 1 correctness 12 passed / 7 failed / 2 skipped；7 例全为 fp16、每例仅 1–3 个元素超差（max ~1.0–1.7e-3 vs 门限 1e-3，报告 max abs err 3.91e-03），fp32/bf16 用例全过。集成 kernel 的 fp16 归一化输出与 golden 归一化张量**逐位一致**（`k_norm16 != g_norm16 : 0 / 4096`，另两例 0/1024、0/960）⇒ kernel 无缺陷、残差全部落在 op 层 affine 的 dtype 算术。op 层改 fp32 affine + 单次 `.to(dtype)` 后同 21 用例 **19 passed / 0 failed / 2 skipped**（2 例为单 NPU 环境的多设备 smoke skip，非失败）、benchmark 7/7、4 例 msprof 有效。
- **门禁遮蔽机制（本陷阱最易漏的部分）**：kernel 级 L0/L1 门禁对 fp16 若采用**单位 affine**（该 kernel 的 `_affine_for` 对 fp16 返回 `"unit"`；DESIGN §8.1 口径「kernel 级输出不含 affine，故对照用单位 weight/bias」），则真实 op 路径的 per-channel affine 从未成为阻塞门禁——Stage 3 全绿（L0 17/17、L1 46/46）直到 Stage 5 以真实 op 调用路径跑全量 report 才暴露。分层算子的 fp16 门禁覆盖因此需要**真实 per-channel affine 的 op 级用例**；同任务 6-seed 实测：fp32 与 bf16 的真实 affine 复刻各 0 违反（可作阻塞判据），fp16 不可（其超差是接口伪影而非 kernel 缺陷）。
- **定位手法（本任务实测有效，零设备依赖）**：先取「归一化张量」这一中间量在 kernel 输出与 golden 之间做**逐位**比较（`!= : 0/N` 即 kernel 无缺陷），再把残差归因到 op 层 affine 的 dtype 算术（`ideal = normalized.fp16 → fp32 affine → 单次舍回` 给出 `bad = 0`）——「改哪个文件」由此有可执行证据而非猜测。伴生事实：device affine 与 cpu affine 在相同数据上违反集完全相同（排除「NPU 端 fp16 affine 精度异常」方向，归因收敛到接口分层本身）。
- **与同族条目的分工**：TRAP-fp16-opmath-golden 描述 kernel 侧逐步 fp16 舍入与 golden fp32 opmath 的分歧（对齐通解 = kernel 内 fp32 中转 + 单次舍回）；本条描述 **op 层** affine 的二次舍入（kernel 已单次舍回、对齐通解 = op 层 fp32 affine 单次舍回）——分别对应「kernel 侧舍入路径」与「接口/op 层舍入路径」，互补不重复。
- **证明**：`python repro/TRAP-fp16-op-affine-double-rounding.py`（exit 0；三段断言：trap 违反数 > 0、fp32-affine 违反数 ×10 < trap、单位 affine == 0；纯 torch CPU，无 NPU、不 import 任务工作区模块）。溯源（任务工作区，未上库，允许失效）：`examples/group_norm/_group_norm_kernel/{RETROSPECTIVE.md, repro/TRAP-fp16-op-affine-double-rounding.py}`、`examples/group_norm/RETROSPECTIVE.md`（Stage 5 复盘）、集成侧 `integration_log.md`（attempt 1 定位段 / attempt 2 授权修复段）。
