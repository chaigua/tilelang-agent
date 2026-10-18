# 向量化轴与布局模式（优先级最高的检索项）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（id/kind/family/apis/status/origin_task/toolchain/repro——schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码（存量条目允许，新 D 类条目缺 repro 降级 Tier 1 入队）。

**诊断触发器**：msprof 显示算子热点段**标量执行占比 > 50%** 时，强制触发"换向量化轴/换布局"分析——不要在原轴上微调（实测案例：窗口累加 98.3% 标量源于 `j·sW`（sW≠1）跨步系数阻碍向量化，原轴微调收益为 0，换轴后 5.6x）。

---
id: PL-1.1-transpose-chain
kind: pattern
family: [pooling, window, layout]
apis: [T.transpose, T.copy]
dtype: [fp16, fp32, bf16]
device: 910B2C
status: stale
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）；**2026-10-14 重验：仅对「两边 ≥ 向量宽且为向量宽整数倍」的 2D 形态成立，池化族大 shape 的 3D 二轴交换被降级为 UB→GM→UB（见正文末尾与 traps-compiler.md TRAP-transpose-ub-gm-roundtrip）**
repro: repro-missing
---

### 1.1 核内融合转置链（布局重排的正确形态）✅ 已验证

- **模式**：I/O 保持原生契约（如 NCHW），核内用 **T.transpose 二轴交换链**重排到目标布局（如 NHWC）。3D 置换 = 两次二轴交换：
  ```
  (CH, Hi, Wi) -[1,0,2]-> (Hi, CH, Wi) -[0,2,1]-> (Hi, Wi, CH)   # 输入侧
  (BH, WO, CH) -[0,2,1]-> (BH, CH, WO) -[1,0,2]-> (CH, BH, WO)   # 输出侧
  ```
- **实测代价**：4 次融合转置合计 **~5.4µs**（远低于算子本体）——"转置很慢"是错误印象，勿因直觉弃用。
- **硬约束**：① `T.transpose` 的 permutation **仅支持二轴交换**（如 `[1,0,2]`/`[0,2,1]`），**不支持 3-cycle**（`[1,2,0]`）——3D 置换必须拆成两次交换链；② **reshape→transpose 链会 mis-compile（数据错乱）**——必须始终在自然形状 buffer 之间做 transpose；③ 转置须在**自然形状** shared/UB buffer 间进行。
- **佐证**：`testing/npuir/broken/test_slice_transpose_dev.py`；AvgPool2d perf_opt v3a（`examples/TileOPs/tileops/kernels/pool/avg_pool2d/avg_pool2d_kernel/perf_opt/`——任务工作区，未上库，provenance 允许失效；模式与实测数字已自包含于本条目）。
- **⚠️ 2026-10-14 重验更正（工具链变更，status -> stale）**：在 tilelang `0.1.2+8b16bccacfd9fa17`（commit 417dae9）+ CANN 9.0.0 + Ascend910B3 上重测，
  「4 次转置 ~5.4µs」**对池化族大 shape 的 3D 二轴交换不成立**——`(64,7,7)↔(7,7,64)` 单次转置使 kernel 从 5.6 µs 涨到 138.2 µs，
  `UB_to_GM_datas` = 48 KB/block（UB→GM→UB 往返）；转置 + C 轴 `T.vadd` 累加完整形态 148–174 µs。
  **但 `(64,56)→(56,64)` 的 2D f16 形态仍走 UB 内实现（`UB_to_GM_datas` 8.5 KB/block）**，故本条不是整类推翻而是**适用域收窄**：
  仅「两边 ≥ 向量宽且为向量宽整数倍」的 2D 形态可用。细分证据与绕法见 traps-compiler.md `TRAP-transpose-ub-gm-roundtrip`；
  池化大 shape 的替代结构见本文件 **PL-1.23-flat-spatial-axis-reduce**（免转置，166.3 → 16.7 µs）。
  溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/profiles/probe/round0/{p2_ct_c64,p6_f4,accum_a49_1}`。

---
id: PL-1.2-caxis-accum
kind: pattern
family: [pooling, window, elementwise]
apis: [T.vadd, T.serial]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）
repro: repro-missing
---

### 1.2 C 轴切片累加（channel-batched vector accumulation）✅ 已验证

- **模式**：`T.vadd(acc[i, j, :], in_f32[i*SH+ki, j*SW+kj, :], acc[i, j, :])`——整 C 轴切片作为向量操作数，**i/j 用 T.serial**（不是 Parallel），ki/kj serial 外层。
- **适用**：C 恒为向量宽度整数倍的 NCHW 类算子（C=64/96/128；96 用 CH=48/32/16 分块）。
- **实测收益**：AvgPool2d kernel 级 1.98–13.2x（vs 同代 H-collapse 版本）。

---
id: PL-1.22-maxreduce-nan-tracking
kind: pattern
family: [pooling, window, reduction, general]
apis: [T.transpose, T.vmax, T.vcmp, T.vcast, T.vmul, T.vbrc, T.reduce_min, T.reduce_max, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261013T104045Z（首证）/ max_pool2d-_max_pool2d_kernel-20261014T062320Z（第二证，行级掩码细化 + 窗口池化域）/ max_pool1d-_max_pool1d_kernel-20261014T082309Z（第三数据点，1D 域：块级掩码 + 单空间轴守卫前提）/ adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015（第四数据点：**C4/整平面域**的等代价 NaN 追踪形态 `isnan(reduce_sum(x*x))`，UB 单价 32 vs 掩码链 96-128 B/元素）
toolchain: 首证 tilelang 0.1.2+db1783d948b43d3b5eed38c542e158c08cde3df5；第二证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c；第三数据点 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）；第四数据点 tilelang 0.1.2+489385e9b98806365b585b6e1f984a5dae337154（commit 489385e）；均 dev root build + CANN 9.0.0 + Ascend910B3
repro: repro/PL-1.22-max-nan-squaresum.py
---

### 1.22 max 池化/归约的 NPU 向量实现形态（NaN 追踪 + 无代数化简）✅ 已验证

> 由 queue VP-2026-0152 在两任务独立证据后合入（adaptive_max_pool2d 迁移 2026-10-13 / max_pool2d 窗口池化迁移 2026-10-14）。两任务为**不同任务、不同算子**（AdaptiveMaxPool2dFwdOp / MaxPool2dFwdOp），各自独立执行 Stage 3 分层门禁（前者全用例过且 worst max_diff=0.0；后者 L0 31 / L1 23 / L2 7 / Boundary 8 全过、fp16/bf16/fp32 三 dtype `max_diff=0.000e+00`、`atol=rtol=0` 成立）。保守限定同 queue VP-2026-0143 先例（max 为纯选择无累加，逐位判据可成立）。**第三数据点（2026-10-14，1D 域）**：max_pool1d 窗口池化迁移在本条 ①②③''④⑤ 上独立复现（见 ⑦）——同一形态集跨维（2D → 1D）成立，按 `update` 追加数据点（Tier 0，front-matter 版本戳同步）。**repro**：本条按 ED-B「无法当场最小化时显式 `repro-missing`」处置（P 类允许）——③ 的 NaN 追踪合法式已有可执行载体 `repro/TRAP-vcmp-ne-ordered-nan.py`（属 traps-compiler.md 条目，两侧互相引用，故不重复登记为本条 repro）；②/③'/③''/④/⑤/⑥ 部分的专属最小 repro 待同族任务回填。

- **① 向量轴 = C 轴**：核内 NHWC 二轴 `T.transpose` 重排后沿 C 轴 `T.vmax` 归约，与 avg 池化同构（PL-1.2），差异仅在归约算子与 NaN 追踪。窗口池化的窗口形状由 `k/s/p/d` 固定（非 adaptive 的变长 bin），但向量轴结论相同。
- **② 比较与掩码在 fp32 域**：bf16 先 `T.vcast` 到 fp32（`vmax`/`reduce_max`/`vsub` 不支持 bf16）。**注意源本身是 fp32 时不能**用 `T.vcast` 做同 dtype 搬运（退化为 round-to-integer，见 traps-compiler.md TRAP-vcast-same-dtype-rounds）——按 dtype 在 trace 期分派 `T.copy` / `T.vcast`，且该分派使 fp32 的 UB 预算与 fp16/bf16 不同档。
- **③ NaN 追踪用 IEEE 正确的 `eq` 掩码乘性累积**：`cond = T.vcmp(x, x, "eq")` → `T.vcast(cond, f32)` 得 {0,1} 掩码 → in-place `T.vmul(flag, mask, flag)`（`flag` 初值 1：0 吸收 / 1 中性）→ 末尾 `T.vcmp(flag, 0, "eq")` 或整平面 `reduce_min` 判决。**不用 `T.vcmp(...,"ne")`**（ordered 不等，NaN 恒假——见 traps-compiler.md TRAP-vcmp-ne-ordered-nan）。
- **③' 掩码按行级而非逐元素**（第二证细化）：每输入行一次 `T.vcmp(row,row,"eq") + T.vcast` 得 `(W_in, CH)` 掩码，窗口内按列切片 `T.vmul` 乘性累积；逐元素重算掩码会把 op 数放大 `out_W·k_W` 倍（第二证实测 `out_W` 达 56，是该设计自登记的头号发射风险）。行级形态下三 dtype 端到端逐位精确。
- **③'' 掩码归约算子 = 判决的集合语义**：0/1 掩码上「`reduce_min` 与 0 比较」=「存在 0」，「`reduce_max` 与 0 比较」=「全部为 0」；`x==x` 掩码上 0 意味着 NaN，故「任一 NaN」必须用 `reduce_min + ==0`——写成 `reduce_max + ==0` 只在全 NaN 平面成立（第二证 Stage 3 首版即 `reduce_max` 配注释「1.0 iff 窗口无 NaN」的自相矛盾形态；改 `reduce_min` 后部分含 NaN 的平面正确返回 NaN）。自然语言注释与算式不符时以数学推演为准并回写设计。
- **④ 累加器单位元为 `-inf`**：`T.vbrc(-T.infinity("float32"), acc)`；`T.clear` 置 0 在全负窗口产生错误输出。`T.vbrc` 的 destination 须为整 buffer、标量 operand 须为 typed PrimExpr（region/裸标量形态见 TRAP-npuir-vbrc-region-dropped）。
- **⑤ max 无代数化简**：非线性 → 常数权重卷积/GEMM 不等价；无逆元 → 前缀最大值差分对多条递增记录不可行。可分离 2D→1D 对窗口池化**亦不降运算量**（第二证 manifest 三档复算比 1.00 / 1.50 / 1.01 ≥ 基线；adaptive 侧见 queue VP-2026-0144）——「数学等价优化」对本形态只余**结构特化**（全局池化整平面 `reduce_max` / 恒等纯拷贝，见 queue VP-2026-0143）与**归约融合**（max + NaN 同轮扫描）。
- **⑥ 窗口池化的边界守卫可外提**（第二证）：`ih` 只依赖 `(o_h, k_h)`，H 维判据可提升为每行一次（`k_H` 次/单元而非每元素）；`iw` 依赖 `(o_w, k_w)` 无法再提。零风险 CSE 类优化，可直接写进通用路径。**前提 = 存在 ≥2 个空间轴且某方向判据只依赖该方向循环变量**——第三数据点在 1D 上复现该前提的缺失（唯一判据 `0<=iw<L_in` 同时依赖 `(o_w,k_w)`，无第二轴可外提；1D 的对应优化是连续 span 的钳制载入，见 queue VP-2026-0180）。
- **⑦ 1D 域第三数据点（max_pool1d 窗口池化迁移 2026-10-14，本 stamp 实测）**：①②③''④⑤ 在 **1D 同构成立**——`eq` 掩码 → `vcast` 到 f32 → in-place `vmul`（`flag` 初值 1）乘性累积、末尾 `reduce_min + ==0` 判决、`T.vbrc(-inf, acc)` 整 buffer 单位元（`T.vbrc` 的 destination 须为整 buffer 这一伴随约束同样复现，见 TRAP-npuir-vbrc-region-dropped）；**掩码在 1D 天然为块级一次**——每个逻辑单元每载入块只读一次（`(LSPAN, CH)` 块），`vcmp(row,row,"eq") + vcast` 每块一次、窗口内仅 `vmul`（每元素 1 条向量 op），不存在 2D 那样的「逐元素 vs 行级」选择面；三 dtype 端到端 `max_diff=0.000e+00`（`atol=rtol=0`），设计期机器等价脚本 117 case-group 全 PASS（含 nan / ±inf / subnormal / large / zeros 六类数据 × 3 dtype，逐 lane 断言下标落界）。
- **⑧ C4/整平面域第四数据点（adaptive_max_pool2d Stage 4，2026-10-15，本 stamp 实测）——等代价 NaN 追踪形态**：当窗口 = **整平面**（`out_H==out_W==1` 全局池化、扁平 `(CH,1,H*W)` 单次宽归约）时，③ 的掩码链（`vcmp(平面,平面,"eq") → vcast → reduce_min`）**退化为整平面量级的第二级归约**，其 BishengHIR UB 单价达 **96-128 B/平面元素**（把可用 CH 压到 ≤32）。等代价替代形：**`has_nan <=> isnan(reduce_sum(x*x))`**，其中平方必须**原位**（`T.vmul(flat, flat, flat)`；`flat` 在 max 归约后已死）——单价降到 **32 B/平面元素**，解锁 CH ≤ 120（H*W=49 时）。数学等价性：`x*x` 仅在该元素为 NaN 时为 NaN、其余值 ≥ 0 ⇒ 求和**不会**因 `+inf - inf` 产生假 NaN（这正是**不能**直接 `reduce_sum(x)` 的原因）；±inf 平方为 +inf、求和仍 +inf ⇒ 正确判为「无 NaN」。取「非原位平方」单价立刻翻倍到 ≈64 B/元素（多一个全平面中间量），**故「原位」是本形态的必要条件**。实测：`global-1x1`（[8,2048,7,7] fp16）320.83 → 19.16 µs（17x，逐位精确），其中掩码链→原位平方和一步贡献 36.77 → 19.16 µs。单价表与编译边界见 `constants.md` `CONST-reduce-ub-per-element`；数值等价 repro 见 `repro/PL-1.22-max-nan-squaresum.py`（含「原始值求和」的假阳性反例断言）。
- **负向依赖**：结论不依赖 `T.vmax`/`T.reduce_max` 的 NaN 传播行为（未文档化）——实现须显式追踪 NaN 以保证正确，不得依赖该行为兜底（首证 Stage 3 near-miss：靠 `vmax` 传播「顺带修对」致语义门禁假阳性通过）。
- **provenance（允许失效）**：首证 `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/{DESIGN.md#0.6-R2, RETROSPECTIVE.md}`；第二证 `examples/max_pool2d/_max_pool2d_kernel/{DESIGN.md#3.3, _max_pool2d_kernel.py, RETROSPECTIVE.md Stage 3}`；第三数据点 `examples/max_pool1d/_max_pool1d_kernel/{DESIGN.md#0.6-R2/§1.6.1, _max_pool1d_kernel.py, RETROSPECTIVE.md Stage 1/3}`（任务工作区，未上库）。

---
id: PL-1.3-host-permute
kind: pattern
family: [layout]
apis: []
dtype: [fp16, fp32, bf16]
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）
repro: repro-missing
---

### 1.3 host permute 路线 ❌ 通常净亏（已量化证伪）

- 两次 host permute（NCHW↔NHWC）实测 **106–147µs** > 多数中等算子本体耗时；仅当算子本体远大于此（如 >500µs）且无法核内重排时才值得复测。

---
id: PL-1.4-tiling-heuristic
kind: pattern
family: [pooling, window]
apis: []
dtype: [fp16, bf16, fp32]
device: 910B2C, 910B3, 910B4
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08（首证）/ max_pool2d-_max_pool2d_kernel-20261014T062320Z（第二证：容量驱动 CH 逐 buffer 带字节数口径）/ max_pool1d-_max_pool1d_kernel-20261014T082309Z（第三数据点：1D 容量不等式 + CH 的 dtype 感知折半）/ max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z（第四数据点：双输出（索引型）1D 记账式 `69+2·eb` + 薄裕度纪律 + 闭式自洽）/ adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（第五数据点：**单波判据**——纯 Vector general 路径的 CH 由 `num_local==1` 约束而非 C 大小）/ adaptive_max_pool2d-_adaptive_max_pool2d_kernel-20261015（第六数据点：**最小可行 CH**——第五数据点的「折半」规则细化为一维扫描 `min{CH : units ≤ cores}`）/ adaptive_max_pool2d_with_indices-_adaptive_max_pool2d_with_indices_kernel-20261015T100705Z（第七数据点：**双输出**下同一规则成立——spp CH 64→48 / nondiv CH 64→32；且**同族「bool 切片直送 vselect」的进一步削 op 不可达**，见 TRAP-hivm-vsel-cond-slice）/ max_pool1d-_max_pool1d_kernel-20261016T090104Z + max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261017T235029Z（第八数据点，两证：容量上限钉住 `CH<C` ⇒ 尾块 padding lane + `num_logical` 翻倍；逐 buffer 精确足迹 + import 期自检解锁 `CH=C`）
toolchain: 首证 2026-08 build（Ascend910B2C）；第二证 tilelang 0.1.2+0d3950aabfec7b3aa6a67a5cf21bfe4945a9655c；第三数据点 tilelang 0.1.2+ddf4255a4ae5214c27f9d731ce65c9107b748388（commit 417dae9）；第四数据点 tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313（dev root build ba60e99c…）；第五数据点 tilelang 0.1.2+8b16bccacfd9fa17（commit 417dae9）；第六数据点 tilelang 0.1.2+489385e9b98806365b585b6e1f984a5dae337154（commit 489385e）；第八数据点第一证 tilelang 0.1.2+c3c39ba6929f77c131a29c7cde8126ac245e104d（910B3）/ 第二证 tilelang 0.1.2+e8b5d51e3d098422c3970f4e714fb5dba1493609（910B4，AIV=40）；全部 dev root build + CANN 9.0.0
repro: repro/PL-1.4-single-wave-ch.py
---

- **〔2026-10-15 第七数据点，双输出（值 + int64 索引）；toolchain 67a0441，Ascend910B3〕** 「最小可行 CH」（`min{CH : N·out_H·ceil(C/CH) ≤ vector_cores}`）在**双输出**形态同样成立且收益更大（每元素 7 条向量 op，核占用更重要）：
  - `spp-6x6`（C=128,N=2,out 6×6）：CH 64 → 24 units / BD 24（16 核空）→ CH 48 → 36 units / BD 36，**85.57 → 79.74 µs（-6.8%）**；
  - `nondiv-7x7`（C=64,N=2,out 7×7）：CH 64 → 14 units / BD 14（26 核空）→ CH 32 → 28 units / BD 28，**81.41 → 72.64 µs（-10.8%）**。
  - **削 op 的形态边界（重要）**：双输出每元素 8 条向量 op 中只有 **1 条是死代码**（`h_flag` 乘积无消费者，NaN 判决由 `nan_idx>=0` 给出）⇒ 8→7 是可达的；
    进一步把「行级 `x==x` bool 掩码」直接送 `vselect`（想省掉每元素一次 `vcmp`，8→6）在本工具链**不可达**（`hivm.hir.vsel` 拒绝切片 cond，见 TRAP-hivm-vsel-cond-slice）。
    设计期把「8→6」写进收益表的须复核该形态限制。
  数据（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/.../perf_opt/{perf_records.jsonl（round1 gen 分支）, opt_log.md §4}`。
### 1.4 tiling 启发式（C 轴向量化形态）✅ 已验证

- **BH=1 + 最宽 CH 最优**：CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs——更宽 C 向量 + 更少 block 胜过更高空间并行度。
- **UB 约束**：CH×空间 tile 驻留超 192KB 时收缩 CH（实测 vis-3x3 的 CH=64 需 252.8KB 被排除，CH=32 落地）；buffer 生命周期不重叠时可探索 aliasing 复用（预估再省 10–20%，未验证）。
- **回退分发**：C%16≠0 或 shape 越界时回退到非重排路径（工厂层静态判定，无运行时开销）——保持全 shape 兼容。
- **容量驱动 CH 选择**（2026-10-14 两证合入，queue VP-2026-0174 达 2/2）：`CH` = 满足「实现将要落地的 buffer 清单」逐项带字节数求和后 `× 膨胀系数 1.7 ≤ 192 KiB` 的最大 8 的倍数。2D 形态按 `ub_bytes(CH, W_in, out_W, dtype)` 记账；1D（单空间轴，需先沿该轴分块）形态按 `G = CH·LSPAN·(2·eb+9) + BL·CH·(25+2·eb)` 记账（`eb` = 元素字节、`LSPAN = min((BL-1)·s + (k-1)·d + 1, L_in)`；`2·eb+9` = row_src + row_dst + row_f32 + eq_mask + eq_f32，`25+2·eb` = acc/flag/zeros/infbuf/nanbuf/out32 六个 f32 + cfinal(bool) + acc16 + acc_t）——**`BL` 由 `min(L_out, BL_MAX)` 逐级下调试到首个可行值，CH 在 `BL=1` 仍不可行时折半（下限 8）**，故两档是同一条不等式上的联合取档。实测压档：2D `W_in=112` 窗口池化 fp16 CH 64 → 32（CH=64 需 ≈244.5 KiB ✗）、fp32 → 16；1D `textcnn`（`C=256`、`L_in=128`、k128s1p0）fp16 CH=64 → **fp32 折半为 32**（`eb=4` 使行类缓冲翻倍，`num_logical` 256 → 512）——CH 选择必须 dtype 感知，按 fp16 档硬编码到 fp32 会 UB 溢出。两条纪律：① 不得用代表 shape 的头部大 buffer 结论外推（低报）——须给「逐 buffer 原始字节 + 膨胀后字节 + 统一 KiB」；② **实现形式改变（如掩码由逐元素 `(1,CH)` 改行级/块级）后容量结论必须重算**，不得沿用设计档位（第二证实证：fp32 resnet-stem CH 32→16）。
- **双输出（argmax 索引型）1D 记账式 = `G = CH·LSPAN·(2·eb+9) + BL·CH·(69+2·eb) + 256`**（2026-10-14 第四数据点，queue VP-2026-0174 同族续证）：相对单输出**只增加单元类 buffer**（行类 `CH·LSPAN` 项不变）——`69 = 25 + 44`，即单输出基式 `25+2·eb` 之上再加 `7×i32`（`max_idx`/`nan_idx`/`cand`/`nanc`/`bigv`/`neg1v`/`zero_i32` = 28 B）+ `idx32`/`idx_t` 转置缓冲 8 B + `idx64` 落盘缓冲 8 B；`+256` = `fv`（`(1,CH)` 上界）。实测档位（`INFLATE=1.7`，判据 192 KiB）：`sincnet` fp16 `CH64/BL16/LSPAN48` = `114 944 B`（×1.7 = 190.8 KiB ✓）、`ecg` bf16 `CH64/BL17/LSPAN41` = `113 792 B`（×1.7 = 188.9 KiB ✓）——**同 shape 单输出档位为 `BL=18/20`，索引型的额外 44 B/元素把 `BL` 压掉 2–3 档**。
  - **薄裕度纪律**（该族实测裕度 ≈0.6%）：`sincnet` 原始 114 944 B 距判据仅 1 203 B，**再加一个 `(BL,CH)=16×64` bool（1 024 B）即 `(115 968)×1.7 = 197 146 B > 196 608 B` 翻转**⇒ 薄裕度算子的 §4.3 buffer 清单须与分析表逐名一致（「为清晰而拆缓冲」的实现会越界），且须给出「再加一个最小缓冲是否翻转判据」的边界核算。
  - **闭式自洽自检**：闭式括号内分量的求和须等于系数（本条目首个设计版本记 `53+2·eb` 而同行分量注释求和为 `69`——纯文本即可发现的漏计；见 §4.5 的更正链）；逐 buffer 求和须等于闭式值；档位（`BL`/`CH`）变更后所有派生量（`LSPAN`→`num_oblk`→`num_logical`→`num_local`）与备选档连带重算。
- **单波判据（第五数据点，2026-10-14，纯 Vector general 路径）**：`CH` 的第三个约束面不是容量而是**波次**——逻辑单元 `units = N·out_H·ceil(C/CH)`，`num_local = ceil(units / min(units, vector_cores))`；`num_local > 1` 表示有核要串行跑第二波，其代价**大于**「更宽向量」的收益。实测（msprof op，vector_cores = 40，Ascend910B3）：`spp-6x6`（C=128, N=2, out 6x6）CH=64 → 24 units → 单波 **16.31 µs**，CH=32 → 48 units → 两波 24.32 µs（**+49%**）；`nondiv-7x7`（C=64, N=2, out 7x7）CH=64 → 14 units 单波 14.45 µs，CH=32 → 28 units 单波 **13.14 µs（-9%）**，CH=16 → 56 units 两波 22.17 µs（+53%）。**采纳规则**：`base = round_up8(min(C,64))`；若 `units(base)·2 > vector_cores` 直接用 base（已用掉过半核）；否则折半，**仅当折半后仍单波**才采纳（min 8）。该规则同时解释「CH32 在 spp 上反而更慢」（多一波）与「CH32 在 nondiv 上更快」（仍单波但把单元摊到更多空闲核）。repro：`repro/PL-1.4-single-wave-ch.py`（纯 Python，断言两 shape 的分派 = 实测最优档）。溯源：`examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/{opt_log.md#6, perf_records.jsonl}`（任务工作区）。
- **最小可行 CH（第六数据点，2026-10-15，纯 Vector general 路径）**：第五数据点的「折半且仍单波」规则只能二选一，遇到「折半后仍单波但更长块」的情形会选错。正确判据是**一维扫描取最小可行 CH**：`min{ CH = 8,16,…,cap : units(CH) ≤ vector_cores }`——它同时最大化占用核数并最小化最长块（单波时 wall ∝ 最长块）；连最窄块都装不下一波时退回 `base`。实测（msprof op，vector_cores=40，Ascend910B3）：`spp-6x6`（C=128,N=2,out 6×6）CH=64 → 24 units / 最长块 64 → **20.42 µs**；CH=48 → 36 units / 最长块 48 → **18.97 µs（-7%）**；CH=40 → 48 units（两波）→ 29.47 µs（**+44%**）。`nondiv-7x7`（C=64,N=2,out 7×7）CH=64 → 14 units → 18.32 µs；CH=32 → 28 units / 最长块 32 → **15.45 µs（-16%）**；CH=48 → 28 units 但块失衡 48/16 → 17.07 µs（**+10%**，证明「最长块」而非「units 数」才是第二判据）。
- **容量上限（而非容量不等式）把 `CH` 钉在 `C` 之下时，通道尾块同时付两份代价（第八数据点，2026-10-16/18 两证，纯 Vector 1D 窗口池化 general 路径）**：① 窗口/归约 op 仍按 **CH 全宽**执行（尾块只把 `real_c = min(CH, C−cb·CH)` 段写回，padding lane 照样占向量位）——`C=80, CH=64` ⇒ 第 2 块 `real_c=16`、**37.5% 的 lane 是 padding**；② 单元数随 `ceil(C/CH)` 翻倍（`num_logical` 3392 → 2080 / 5504 → 2752）。把上限抬到 ≥ `C` 使 `CH = min(C, cap) = 80`（**容量不等式不变**、`BL` 由同一不等式自动收缩）后实测：第一证（910B3，`c3c39ba`）sincnet `301.936 → 187.964 µs（−37.7%）`、ecg（`C=128`）`154.653 → 113.002 µs（−26.9%）`；第二证（910B4，`e8b5d51e`，双输出 indices 变体）sincnet `1005.71 → 538.91 µs（−46.4%）`、ecg `599.61 → 367.98 µs（−38.6%）`，三 dtype 逐位精确。**适用域**：`units ≫ vector_cores` 的多波 regime（此时第五/六数据点的单波判据集合为空，但排序键「尾块浪费最少」仍成立）；`C=256` 的 textcnn 因 span 平面在 `CH=128` 时需 138.9 KB > 预算而保持 `CH=64`（噪声内不回退）——**上限抬升不保证每个 workload 都能吃满 `CH=C`**。
  - **可达性机制（第二证的关键步骤）**：`_general_bytes` 原按 Stage 3 闭式 `69+2·eb` 记账，其中含已删死的 `flag`/`eq_mask`/`eq_f32`（死代码移除后释放 9 B/元素）⇒ 旧式把 `CH` 上限锁死在 64。改为**逐 buffer 精确和 + import 期自检**（`_general_bytes_selfcheck()` 断言逐 buffer 求和 == 闭式值）后 `CH` 才可到 `C`、`BL` 解到 10–16。
  - **与第四数据点（实现须逐字复现冻结闭式）的边界**：Stage 3 交付期以设计闭式为契约（防分核表/`BL`/`CH` 与 DESIGN 永久漂移）；**Stage 4 在实测证明某类 alloc 已死后才重推**，并在日志记录新旧闭式差异与派生量重算（本条实证 CH 64→80、`num_logical` 5504→2752）。删除 alloc 却不重推不等式 ⇒ 档位被历史公式锁死。
  - 数据（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/max_pool1d/.../perf_opt/{perf_records.jsonl round1 chfull, opt_log.md §3.1/§4}`；`examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/.../perf_opt/{perf_records.jsonl round2 chfull, opt_log.md §5 Round 2, _max_pool1d_with_indices_kernel.py::_general_bytes_selfcheck}`。

---
id: PL-1.5-quickref
kind: pattern
family: [elementwise, pooling, general]
apis: [T.vmul, T.ceildiv, T.Pipelined, T.serial, T.reduce_sum]
dtype: [fp16, fp32, bf16]
device: 910B2C, 910B3
status: verified
origin_task: mixed（AvgPool2d 2026-08 / lerp_tensor 2026-09-07 / adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z）
toolchain: 2026-08 build；lerp 行为 tilelang-mlir-dev dev root build 2026-09-07；adaptive_avg_pool2d tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b + CANN 9.0.0 + Ascend910B3 / 2026-10-13
repro: repro/TRAP-vcast-shape-and-gm-source.py
---

### 1.5 其他已验证模式速查

| 模式                  | 一句话                                                                                                                                                                                               | 实测参考                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 乘编译期常数倒数      | divisor 恒定时`T.vmul(acc, 1/d, acc)`，省逐元素除法                                                                                                                                                                                                                | AvgPool2d fast path                                                                                                |
| 乘运行时 let-bound 标量倒数 | divisor 在 `T.Kernel` 内由运行时表达式 let 绑定（逐 bin 计算，非编译期常数）时 `T.vmul(acc, inv, acc)` 同样合法且数值正确——文档「标量须在 `T.Kernel` 内、`T.Scope` 外定义」≠「须为编译期常数」；fp32 倒数+乘对 `sum/d` 为容差内等价（adaptive_avg_pool2d 实测 ≤1 ulp of target dtype，spp fp16 10/9216 元素差 1 ulp；量化数据见 queue VP-2026-0142） | adaptive_avg_pool2d 迁移 2026-10-13（repro: `repro/TRAP-vcast-shape-and-gm-source.py` FIX 段；origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z） |
| 3D `T.reduce_sum` 链 | `T.reduce_sum(x, mid, dim=2)` → `T.reduce_sum(mid, acc, dim=1)` 两级归约 = 整平面均值，是 `out=1×1` 全局池化特化的原语；免 `T.reshape`+transpose 链（后者有 mis-compile 风险，PL-1.1） | adaptive_avg_pool2d 迁移 2026-10-13（resnet-global C4 max_diff 1.221e-04；同上 repro FIX 段） |
| H-collapse            | 把 kH 折叠进向量管道，消除一层 serial 累加                                                                                                                                                                                                                           | AvgPool2d v2（对 sW 跨步问题的 W 轴解法）                                                                          |
| fp32 求和序匹配       | kernel 累加序与 golden（F.avg_pool2d 等）一致时解锁高精度快路径                                                                                                                                                                                                      | AvgPool2d fp32 13.2x                                                                                               |
| launch 开销主导判定   | 小 shape 与大 shape 耗时同量级 → 固定开销主导；拆 pad-only/padded-in 定位                                                                                                                                                                                           | bench_runner`--decompose`                                                                                        |
| host 层编译期常量折叠 | `@T.prim_func` 外的 `T.ceildiv(int,int)`/`min(IntImm,int)` 对 Python int 常量折叠为 IntImm（复现：`T.ceildiv(4097,4096)`→`2 <IntImm>`）——persistent 设计的 num_logical/num_kernels/num_local_tasks 可在工厂层直接求编译期常量，无需改写 Python 整数算术 | lerp_tensor 迁移 2026-09-07（REVIEW.md 附#29；origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z） |

---

id: PL-1.21-vector-persistent-split
kind: pattern
family: [general, pooling, elementwise]
apis: [T.Kernel, T.serial]
dtype: []
device: 910B2C, 910B3
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z（首证，第 1 证据）/ adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261013T084721Z（第二证，第 2 证据）
toolchain: 首证 tilelang-mlir-dev dev root build 2026-09-07（纯 Vector 核数翻倍 48）；第二证 tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b + CANN 9.0.0 + Ascend910B3（实查 20×2=40）/ 2026-10-13
repro: repro-missing
---

### 1.21 纯 Vector persistent 分核模板（静态界 + 尾任务守卫）

> 由 queue VP-2026-0003 在两任务独立证据（lerp_tensor 一维逐元素 2026-09-07 / adaptive_avg_pool2d 迁移 2026-10-13）后合入。知识域最小 repro 待同族任务回填（原任务内复现命令见下方 provenance 项）。

- **模板**：`vector_cores = NPUUtils.get().get_aicore_num() * 2`（纯 Vector 核数翻倍）；`num_kernels = min(num_logical, vector_cores)`；核内 `for t in T.serial(num_local)`（`num_local = ceil(num_logical / num_kernels)`，Python int **编译期常量** → 静态界成立）；`uid = t * num_kernels + cid`；`if uid < num_logical` 守卫尾任务。
- **逻辑单元（`num_logical`）粒度**：一维逐元素时 `num_logical = ceil(N/block_size)`；多维 unit 形态（如池化 `(n, o_h, c_block)`）时取该 unit 总数——同一模板，unit 粒度由算子决定（adaptive_avg_pool2d 第二证）。
- **不产生空核**：因 `num_kernels ≤ num_logical`，每核首任务 `uid = cid < num_logical` 恒有效——与 TRAP `T.ceildiv` 负数域 empty-core 越界（queue VP-2026-0058）的形态不同（此处 `num_local` 为 Python int 常量、核数不超过逻辑核数）。
- **尾块（一维 elementwise 形态，首证）**：`tail = T.min(block_size, N - t0)` + UB 零起点切片 `T.copy(x[t0:t0+tail], x_ub[0:tail])`——v-prefix 整 buffer 操作对 UB 陈旧段一并计算、只拷出 `[0:tail]`，正确性无碍。
- **提取期注意**：TileOPs wrapper 的 E3/E4 适配注释（如 `threads=` 取代）是提取期笼统预估，核数/grid 取值以 DESIGN.md 分核三要素为准（冲突时设计文档优先）。
- **provenance（允许失效）**：首证见 queue VP-2026-0003（`examples/lerp_tensor/_make_lerp_tensor_kernel/DESIGN.md#5.5`）；第二证见 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md#5.5` 与 `_adaptive_avg_pool2d_kernel.py` L211-234（任务工作区，未上库）。

---
id: PL-1.15-broadcast-perf-tax
kind: pattern
family: [vector, layout]
apis: [T.vsub, T.vmul, T.vbrc]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### 未文档化列广播形态「可编译可算对但有性能税」（vsub [M,N]−[1,bs] 实测 +12~16%；vsub dst=src2 alias 形态 +17~20%——同族第二实证）

- **形态一（列广播）**：`T.vsub` 的列广播 `[M,N] − [1,N]`（T.vsub.md §2.2.2 未列——文档仅同 shape/行广播 [M,N]−[M,1]/退化 [1,N]−[1,1]/标量四形）在当前工具链**可编译、L0 精度正确**，但实测 **w2 +12.2% / w3 +16.0%**（msprof op，vs `vbrc([1,N]→[M,N]) + 同 shape vsub` 的文档化两步形态）。
- **形态二（dst=src2 alias，2026-09-17 第二轮 R7 v11 第二实证）**：`T.vsub(a, b, b)`（diff 落到第二操作数原地址，元素级原地安全）同样可编译可算对，但实测同口径 **w2 +19.6% / w3 +17.0% / w4 +2.0%**——原语境为 vbrc hoist（把循环不变的 [M,1]→[M,N] 广播提出来需 diff 重定向到 alias 目标）。**AIV 接近关键路径的 workload（w2 瞬态/w3）受害最大，AIV 远离临界的稳态（w4）仅 +2%**——税的大小与该引擎的空闲余量负相关。
- **结论**：广播/alias 等「非典型形态」的「合法化」（编译通过 + 数值正确）不等于「可用」（无性能税）——证伪协议须含性能步；列因子在 vsub/vmul 上仍应走 vbrc 展开 + 同 shape 运算；循环不变量 hoist 若被迫引入 alias 形态，宁可保留循环内重算（与 Stage-2 review 的广播形态速查口径互补：形态合法性看文档 §2.2.2，可用性看实测）。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（形态一 opt_log R2 v4_p2 行 / perf_records round 2；形态二 opt_log R7 v11 行 / perf_records round 7 同口径对比 v9_recheck）。

---
id: PL-1.23-flat-spatial-axis-reduce
kind: pattern
family: [pooling, reduction, layout, general]
apis: [T.reduce_sum, T.vcast, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: adaptive_avg_pool2d-_adaptive_avg_pool2d_kernel-20261014T172114Z（Stage 4）
toolchain: tilelang 0.1.2+8b16bccacfd9fa17b09ebd4313bd4501fee03760（commit 417dae9 / dev root build）+ CANN 9.0.0 + Ascend910B3 / 2026-10-14
repro: repro/PL-1.23-flat-spatial-axis-reduce.py
---

### 1.23 扁平空间轴归约（整平面 / 全局池化免转置）✅ 已验证

- **模式（慢→快关键改动）**：整平面归约（`out_H==out_W==1` 的全局池化、整平面 sum/mean）把核内 UB tile 从**自然三维 `(CH,H,W)`** 改为**扁平 `(CH,1,H*W)`**，并把 GM tensor 同步声明为 `(N,C,1,H*W)`（同一块连续内存的形状重解释，**不改变 I/O 字节序契约**）：
  ```
  慢: x3:(N,C,H,W)   tile=(CH,H,W)   reduce_sum(dim=2)   # 归约轴 = 最内轴, 宽 W=7
                                            reduce_sum(dim=1)   # 归约轴 = 中轴, 最内宽 1
  快: x4:(N,C,1,H*W) tile=(CH,1,H*W) reduce_sum(dim=2)   # 归约轴 = 最内轴, 宽 H*W=49
  ```
- **实测收益**：`resnet-global`（`[8,2048,7,7]→(1,1)` fp16）**166.26 µs → 16.68 µs（9.96x）**，逐位精确（`max_diff=0.000e+00`）；`CH=32` 为该形态最优档（`CH=64` 在归约 lowering 触发 UB overflow：BishengIR 内部临时量随归约宽度放大；`CH=16` 因波数翻倍退化到 22.76 µs）。
- **机制归因**：① 最内轴窄（7，非向量宽整数倍）的 `T.reduce_sum` 被逐元素标量化（≈ 元素数 × 6 ns，见 traps-compiler.md `TRAP-inner-axis-reduce-scalarize`）；② 三维形态还触发布局传播，使 GM→UB DMA 退化为 16 B/元素分散读（36.75 → 336 KB/block，见 `TRAP-copy-layout-propagation-dma-amplification`）；③ 扁平声明让 DMA 保持整块连续，且**完全不需要 `T.transpose`**——该工具链上三维二轴转置被降级为 UB→GM→UB 往返（见 `TRAP-transpose-ub-gm-roundtrip`）。缓冲区数量同时下降（去掉 mid/acc_t 等中间层）。
- **适用条件与边界**：要求被合并的空间轴在内存中**连续**（NCHW 的 H、W 天然连续；C 在外层）且 `H*W` 为编译期常量。**不适用于 general grid**（每个输出 bin 是矩形子区域，扁平化后行不连续）——general 路径仍走逐行二轴转置 + C 轴累加（PL-1.2）。
- **替代关系**：本条是 PL-1.1「核内转置把 C 变最内轴」路线的**免转置替代**；当转置可用（两边 ≥ 向量宽且为向量宽整数倍）时两者皆可，转置被降级时本条是唯一可行路径。
- repro：`repro/PL-1.23-flat-spatial-axis-reduce.py`（slow/fast 两形态可运行、输出与 torch 整平面均值一致；≥2x 的性能断言需 msprof，命令族见 repro 头部）。溯源（任务工作区，允许失效）：`examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/perf_opt/{opt_log.md#4-6, perf_records.jsonl}` 与 `profiles/probe/round0/{p7_r49_c32, p6_f6}`。

---
id: PL-1.25-row-reduce-indicator-complement
kind: pattern
family: [reduction]
apis: [T.vcmp, T.vcast, T.reduce_sum, T.reduce_max, T.reduce_min, T.vsub, T.vmax, T.vmin, T.copy, T.min]
dtype: [fp16, bf16, fp32]
device: 910B3
status: verified
origin_task: count_nonzero-_logical_reduce_kernel-20261016T005019Z（首证，计数类）/ all-_logical_reduce_kernel-20261016T024920Z（第二证，谓词 AND 类：合并算子方向 + `T.vmax` no-alias 链 + dtype 分档）/ any-_logical_reduce_kernel-20261016T062400Z（第三形态，谓词 OR 类：`T.vmin` 合并 + 单位元 1.0 + 四项反转）/ count_nonzero-_logical_reduce_kernel-20261018T043104Z（第四形态，Stage 4：超预算单 tile 驻留 R′）
toolchain: tilelang 0.1.2+b39cc30679662a13dd911768a51c7eede45d2892 + CANN 9.0.0 + Ascend910B3 / 2026-10-16；第二证 tilelang 0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16；第三形态 tilelang 0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba（dev root build）+ CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608 B）/ 2026-10-16；第四形态 tilelang 0.1.2+c316b0e6b8a1ddd13a6927b70f38a614e8b0474d（源树 417dae9）+ CANN 9.0.0 + Ascend910B4（AIV=40 / UB 196608 B）/ 2026-10-18（仅第四形态重验——`--level all` L0 21 + L1 24 + L2 + Boundary 全过、`torch.equal` 逐位、集成 37/37；前三条形态的历史数字未在新戳重测）
repro: repro-missing
---

### 1.25 行归约的「布尔指示平面 → 归约 → 补式」形态与元素预算双路径（`(bm,tn)` + `dim=1`）

**优化见解摘要**：行归约（`(bm,tn)` tile，`reduce dim=1`）的 UB 记账按**元素数**线性、单价由链形态与 dtype 决定（`vcmp(eq) → vcast(i1→f32) → reduce` 指示链：fp16/fp32 = 8.00 B/元素；bf16 = 12.00，因 `vcmp` 操作数被 lowering 上抛 f32——见 constants.md `CONST-reduce-ub-per-element` 形态适用域），故容量判据写成 `bm·tn ≤ E_cap`（竞态安全档 8192；单 block 硬上限 24576 按最坏 dtype 记账）而非字节数。据此得到**双路径**：① **resident**（`N ≤ E_cap`）单 tile 全 N 驻留 + 行归约，`bm` 由元素预算阶梯给出；② **column-tiled**（`N > E_cap`）`bm=1`、`tn` 取「≤ min(E_cap,N) 的最大 256 倍数且整除 N」，跨 tile 用 `(bm,1)` 微元在线累加/合并（GM 流量仍 1 遍，代价 = 每 tile 数个微元 op + tile 循环控制）。判决量是**元素预算**，不是 GPU 期的 SMEM/线程启发式。

- **补式（判决方向由目标语义决定，两向后果不对称）**：需要「非零 / 非等」判决时，本工具链的合法形态是**有序 EQ 的补式**（`ne` 为 ordered-NE，NaN 恒假——见 traps-compiler.md `TRAP-vcmp-ne-ordered-nan`）：
  - 计数类（`count_nonzero`）：`y = N − Σ_j [x_j == 0]`（合并 = sum）；
  - 谓词 AND 类（`all`）：`y = 1 − max_j [x_j == 0]`（合并 = **max**，命中零即 OR）；
  - 谓词 OR 类（`any`）：`y = 1 − min_j [x_j == 0]`（合并 = **min**，累加器单位元 **1.0**——「全零行 → False、存在非零 → True」）；与 AND 类共用同一谓词 `[x_j == 0]`，但**归约算子 / 补式方向 / 跨 tile 合并算子 / 累加器单位元四项在 OR/AND 对偶下逐一反转**（`reduce_max↔reduce_min`、`1−max↔1−min`、`vmax↔vmin`、`0.0↔1.0`），照抄 `all` 一侧即语义翻错——机器证据（同一 repro 内三变体 + 暴露集 `ne+max=[4,5]` / `eq+max=[2,4,6,7]`）见 traps-compiler.md `TRAP-any-predicate-reduction-pair`；NPU 运行期 fp16/bf16/fp32 与 `torch.any` 逐位精确（101 次比对 0 差异）；
  - 共同纪律：补的落点 = **`(bm,)` 微缓冲**（在全宽平面做 `1 − ind` 会多出 1 个 4 B/元素平面与 1 个全宽 pass）；取反路线不存在（`T.vnot` dtype 表无 `bool/int1`，bool 仅可作片上中间结果）；误用 ordered-NE 的代价在 `all` 上更重（含 NaN 行整行翻 False，计数类只少计 1），而 `any` 侧的 ordered-NE 误用**暴露集更窄**——仅「NaN 与零共存且无其他非零」与「all-NaN」两行暴露，其余行被非零「救回」True，故 `any` 的判别性用例须**显式**含该行（`randn` 零密度 ≈ 0 会给出假阳性）。
- **跨 tile 合并算子是语义决定的实现正确性部件**：`all` 的补式表示下合并的合法形态是 **`T.vmax(acc, mx, new_acc)` + `T.copy(new_acc, acc)`（no-alias 三件套）**——照抄计数类的 `T.vadd`/`vsum` 或源 tiled 的 `T.min(acc, tile_acc)` 都语义翻错（实测路径 C 逐位精确）；原位累加命中 carried-state 陷阱（`TRAP-vselect-inplace-carried-state` 同族）。
- **静态尾块用「独立缓冲组 + 固定宽度 + 独立 prim_func」**：`tail` / `num_full` / 路径全为工厂期 Python 常量，每变体返回独立 `prim_func`（`@T.prim_func` 内解析期 `if` 定义 buffer 不可用，见 `TRAP-primfunc-compile-time-if-buffer-def`；运行时常量偏移的行展开另有静默错值先例 `TRAP-scf-for-row-offset-silent-miscompile`）。`tn` 选择 = **优先取整除 N 的最大 256 倍数**（`N=8448 → tn=2816/tail=0`），无整除因子时取最大 256 倍数 + 静态尾块（`N=9000 → tn=8192/tail=808`、`N=10000 → tail=1808`、`N=8193 → tail=1`）；**尾块宽度为 raw（零 pad）**——raw 非 256 倍数宽度可编译（实测 N=2/4/64/300；见 design `ascend-constraints.md` §1）。
- **实测（首证，count_nonzero，resident/column-tiled 两路径集成态）**：resident `[2048,4096]` fp16 22.7405 µs（roofline 42.16%）；column-tiled `[32,16384]` fp16 4.5001 µs；`[4,128,4096]` fp16 9.6602 µs；正确性 37/37、benchmark 7/7。
- **实测（第二证，all，逐位精确 + 集成态）**：路径 R（`N=8192`）与路径 C（`N=16384`，补列置 AND 中性 1.0）结果 `torch.equal`；整除/非整除分派实测一致（`N=16384 → tn=8192/num_full=2/tail=0`、`N=9000 → tail=808`、`N=10000 → tail=1808`、`N=8448 → tn=2816/tail=0`、`N=8193 → tail=1`）；`bm·W ≤ E_cap` 最紧档位 bf16 `(256,4096)`、`bm=2`（96 KB = UB/2）设备运行通过且逐位精确；集成 report 正确性 42/42、benchmark 6/6、msprof 3 例有效（roofline 10.48%–43.98%，带宽 1.75 TB/s）。
- **实测（第三形态，any，逐位精确 + 集成态）**：路径 R 与路径 C（补列置 **OR 中性 0.0**）在「同一有效数据 + 补中性元素」下逐位一致（`torch.equal`）；`tn` 整除档（`num_full=2/4`）与非整除静态尾块档（`tail=808 / 1808`、`tn=2816` 整除因子档）全部 `torch.equal`；函数级 `--level all` 记 101 次精确比对 0 差异（13 行角点含 `NaN+零`=True / `all-NaN`=True / 全零=False / `±0` / 亚正规 / `±inf`）；集成 report 正确性 42/42（`avg max abs error = 0.00e+00`，bool 输出逐位）、benchmark 6/6、msprof 3 例有效（roofline 10.94%–43.18%、带宽 1.75 TB/s）。第三形态与第二形态的差异仅在**补式方向与合并算子对偶翻转**（`1−min` + `T.vmin` no-alias 三件套 + 单位元 1.0），双路径结构与元素预算判据同构。
- **调用方分块参数须在 kernel 侧做元素预算复核**：迁移期 wrapper 的 `default_config` 可能是 GPU 期 SMEM 启发式（`(256,4096)` 下给 `bm=8` ⇒ 32768 元素 > 24576 硬上限），kernel 侧按 `bm·W ≤ E_cap` 回落到阶梯值即可（`block_m` 只决定行分块，不影响结果），无需改写 wrapper 即可零 report 失败。
- **实测（第四形态，Stage 4：超预算单 tile 驻留 R′，2026-10-18）**：`E_CAP < N ≤ UB 元素上限`
  时把 `bm` 固定 1、**取消 tile 循环与跨 tile 微元累加**（单缓冲，auto-multi-buffer 必然放弃），
  实测优于列分块主选：`(32,16384)` fp16 `4.61 → 3.89 µs`（−15.6%）、`(128,16384)` fp16
  `10.26 → 8.96 µs`（−12.7%），逐位精确；**单消费者形态无竞态**（30 发射 × 多 run 恒定，
  与同族多消费者竞态样本形态不同）。翻转比 0.87×，未达设计期 0.8× 阈值但仍采纳
  （Stage 4 规则：实测更快且无回退即 winner）。R′ 的容量判据按 dtype 分档：
  fp16/fp32 `N ≤ 23552`（实测通过点 188416 B）、bf16 `N ≤ 15701`（12 B/元素，8 KB reserve）。
  配套常数见 `CONST-aiv-read-roof-and-tiny-dma-910B4`。该数据点的工具链戳 =
  `tilelang 0.1.2+c316b0e（源树 417dae9）+ CANN 9.0.0 + 910B4`（本条目 front-matter 的
  旧戳未重验，故整条仍标 stale；本条第四形态仅对该新戳负责）。
- **第四形态的实现证据与第二证路径（2026-10-18 蒸馏补记）**：该形态的代码证据为 `repro-missing`
  （ED-B：R′ 分支的慢→快 delta 骨架待同族任务补——判据形态 = `_select_config` 在
  `E_CAP < N ≤ _ub_elems(dtype)` 时返回 `block_m=1`，工厂层走无 tile 循环的单 tile 变体；
  机制归因 = 省 1 个 tile pass 与跨 tile 微元累加，且单缓冲不触发 `TRAP-UB-multibuffer-inflation`
  的 ×1.7 膨胀）。第二证路径（E-3 快速通道）：任一 `E_CAP < N ≤ _ub_elems` 的 logical-reduce /
  行归约任务重跑同结构，逐位精确且单 run 时延 ≤ 0.9× 列分块即计第二证；第二证须来自不同任务
  （不同算子或不同 dtype 契约）。
- **溯源（任务工作区，允许失效）**：首证 `examples/count_nonzero/_logical_reduce_kernel/{DESIGN.md#§5.4/§5.6, _logical_reduce_kernel.py#_build_resident/_build_tiled/_build_tiled_tail}`；第二证 `examples/all/_logical_reduce_kernel/{DESIGN.md#4.5/§8.2, _logical_reduce_kernel.py#_resolve_block_m/run_L0 L0-3a..e/run_L1 L1-1/L1-7a..c/L1-9}`。代码骨架待最小化（ED-B delta 形态），两证任务均标 `repro-missing`。
