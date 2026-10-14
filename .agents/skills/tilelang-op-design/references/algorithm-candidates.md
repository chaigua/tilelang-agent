# 算法候选库（algorithm-candidates）— 常见算子族替代算法参考表

> **版本化结构化候选库**（D-4）：由 `algorithm-research.md` §5 参考表升级而来——该文件保留调研方法论，**候选表本体以本文件为唯一事实源**。R1/R2 的命中行是**必查清单**（下界不是上界）：命中行的候选必须评估，但调研不得止步于本表——结构判据（algorithm-research.md §3 R2）与外部已知算法仍须过一遍。表内条目涉及 API 时仍须本地佐证（`examples/` / `docs/`）——**算法思路与 API 存在性是两回事**。
>
> **维护**：`tilelang-skill-evolver` 蒸馏 Stage 4 发现的"算法族 X 的候选 Y 实测收益 Z"时同步更新本表（P 类，Tier 1，target_doc 指向本文件）；任务内 optimizer 回写 pattern-library 的算法级发现同样在复盘表标注本表候选行。条目带 front-matter：`known_impl`（本仓已验证实现指针）/ `kb_links`（pattern-library 条目 ID——候选的实测代价/反例证据）。检索：`kb_search.py`（K-3 统一入口）。

---
id: ALG-softmax
family: [softmax, log-softmax, logsumexp]
known_impl: []
kb_links: [PL-1.9-hardlimits（f16 softmax 链实测）, PL-1.9-twophase（两遍式 S/P 物化先例）]
---

R1 等价化简候选：减 max 稳定化（必选）；log-softmax 直接由 logsumexp 表示；exp → exp2·log2e 缩放。R2 在线变体：online softmax（分块 running max/sum + 重缩放）。R3 复杂度要点：三遍（max/sum/normalize）vs 两遍 vs 单遍 online；行缓冲 O(row) vs O(tile)。R4 亲和要点：行内水平归约 vs 垂直扫描（与 §1.6.3 交互）；整行 UB 驻留；attention 场景 MixCV 融合。

---
id: ALG-layernorm
family: [layer_norm, batch_norm]
known_impl: [examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/_ada_layer_norm_kernel.py（persistent 全 N 驻留 + loads-first，2.14x）]
kb_links: [PL-1.5-quickref（fp32 求和序匹配）, TRAP-vrsqrt-plain-precision（rsqrt 近似指令 2.9e-3，vsqrt+vdiv 绕法）, PL-1.10-loads-first-decoupling（staging 解耦）, CASE-norm-adalayern-stage4]
---

R1：`x/sqrt(var+ε)` → `x·rsqrt(...)`；单遍式 `var = E[x²]−(E[x])²`（须评估 fp32 累加与 catastrophic cancellation——ada_layer_norm 2026-09-10 实测定量证据入 queue VP-2026-0043，repro-missing 待补后合入）。R2：Welford 增量 mean/var；分块 running 统计。R3：两遍扫描 vs 单遍；统计量缓冲。R4：规约轴 lane 映射（见 §1.6.3）；fp32 累加路径。**row-reduction 迁移基准形态（两任务实测：logsumexp 2026-08 grid 形态 / ada_layer_norm 2026-09-10 persistent 形态）**：全 N 驻留 UB（流量 4MN 最小）+ 按 M 行块分核（persistent `num_kernels = min(ceil(M/bm), aicore×2)`，纯 Vector 核数翻倍实查）+ 尾块 `T.min` + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限（fp16 约 N>11520，20 B/elem UB 律见 TRAP-UB-multibuffer-inflation）才需 N-tile 流式（+25% 流量，ada 设计期估算）。

---
id: ALG-rmsnorm
family: [rmsnorm]
known_impl: []
kb_links: [TRAP-vrsqrt-plain-precision（rsqrt 近似指令——rmsnorm 的 rsqrt 直用同样受影响）]
---

R1：无 mean 减法（相对 layer_norm 少一遍扫描）；rsqrt。R2：平方和单遍天然在线。R3：平方和单遍；无均值遍。R4：同 layer_norm（row-reduction 迁移基准形态见 ALG-layernorm R4：全 N 驻留 + M 行块分核 + 尾块三件，两任务实测）。

---
id: ALG-attention
family: [attention]
known_impl: [examples/flash_attention/flash_attn_npuir.py（两相位 flash 结构，第三轮 [DESIGN_LIMIT] 推翻者）]
kb_links: [PL-1.7-expert-persistent-boundary, PL-1.9-twophase, CASE-ref-flash-attn-npuir]
---

R1：scale 融入 Q/K；softmax 分母倒数乘法化。R2：flash 结构——online softmax + 分块 KV，O(N²) 中间矩阵 → O(N)。R3：materialize O(N²) 访存/缓冲 vs flash O(N) 缓冲（FLOPs 同阶）。R4：Cube GEMM 分块 + Vector online softmax 流水（MixCV）；L0C/UB 容量定 tile。**已知实测**：两相位结构（pass-1 全 S / pass-2 全 O_partial + Q hoist）四硬目标全达成且优于 per-block persistent 链（PL-1.9-twophase）；短 KV 分档 expert 形态回退 1.31x（PL-1.7，shape 分派权衡见 T-6 多 dispatch 汇总）。

---
id: ALG-conv
family: [conv]
known_impl: []
kb_links: [PL-1.2-caxis-accum, PL-1.1-transpose-chain]
---

R1：im2col + GEMM；implicit GEMM；Winograd（小核乘法次数↓加法次数↑，须按单元吞吐比评估）。R2：滑窗天然流式。R3：direct vs im2col——FLOPs 同阶、访存与单元利用率不同；Winograd 乘法按窗口比例下降。R4：Cube 路径分形对齐；Vector 直接窗口的 C 轴整除性；im2col 展开缓冲。

---
id: ALG-pooling
family: [pooling]
known_impl: [examples/TileOPs/tileops/kernels/pool/avg_pool2d/avg_pool2d_kernel/perf_opt/]
kb_links: [PL-1.2-caxis-accum, PL-1.3-host-permute, PL-1.4-tiling-heuristic]
---

R1：avg pool = 常数权重卷积；sum pool × 常数 = avg pool。R2：滑窗天然流式。R3：窗口重叠数据的重复读。R4：C 轴向量轴；跨步系数（见 §1.6.3）。**已知实测**：核内融合转置链（NCHW→NHWC）~5.4µs 可忽略；host permute 106–147µs 通常净亏（PL-1.1/1.3）。**R3 补充实测（2026-10-13 两任务 2/2 合入）**：可分离 2D→1D 与前缀和差分**均不降低运算量**——自适应池化一侧 bin 宽为 `floor`/`ceil(in/out)` 两值且相邻 bin 重叠 ≤1，spp(56→6) 每平面可分离 3720 / 前缀和差分 3892 vs 直接法 3600，nondiv(55→7, 57→7) 3892/3918 vs 3843（四口径复算，两候选均否决）；窗口池化一侧（max_pool2d，manifest 三档 k3s2/k2s2/k3s2）可分离比 1.00 / 1.50 / 1.01 ≥ 基线。⇒ R3 的「窗口重叠数据的重复读」类结构优化对**两类**池化均不成立，直接 bin/窗口累加（C 轴向量化）为默认候选。

**几何特化路径模式集（2026-10-13 两任务实测合入——adaptive_avg_pool2d + adaptive_max_pool2d 迁移，Tier 1 2/2）**：自适应池化在退化几何下可走结构特化（纯优化，语义与精度不变）：① `out=1×1` 全局自适应池化 = 整平面单次归约（avg 用 3D `T.reduce_sum` 链 `dim=2`→`dim=1`；max 用整平面 `reduce_max`——整平面 flatten，比一般 bin 双循环少 ≈H·W 倍向量 op；免 `T.reshape`+transpose 链，后者有 mis-compile 风险见 PL-1.1）；② `out==in` 恒等 = 纯拷贝（bins 全为单元）；③ 退化路径的驻留缓冲随 `H·W` 线性放大 → 以**编译期容量判据**决定是否走特化，否则回退一般路径（`(1,8,32,32)→1×1` 的 C4 flat 缓冲 ≈393KB ≫ 192KB UB → 两任务均回退、回退路径 max_diff=0/1.22e-04 内）；④ 与一般网格的区分以编译期常量分支实现，特化路径的单元粒度须为整平面（`NUM_LOGICAL` 单一定义源；avg 的 C4b 恒等 `num_logical=N*num_cblk` 而非通用式 `N*out_H*num_cblk`——设计修订 2 轮阻塞根因）。证据：avg `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/{DESIGN.md#1.6.1, verify_equiv.py}`（任务工作区，未上库）／max `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/{DESIGN.md#0.6-R4, _adaptive_max_pool2d_kernel.py（`identity_bytes`/`global_bytes` 判据）}`（同，未上库）。max 侧无代数化简（非线性 + 无逆元），仅结构特化 + 归约融合可用（见 queue VP-2026-0152）。**适用域限定（2026-10-14）**：本节模式集与充分条件均为 **adaptive（自适应）域**——窗口池化（max_pool2d / max_pool3d 类）的退化特化条件不同（窗口形状由 `k/s/p/d` 固定，`out` 网格退化**不**蕴含窗口覆盖整平面），迁移/设计遇窗口池化须逐变量重新推导覆盖集合而**不得平移本节条件**；修正版条件见下方「窗口池化域退化特化条件」段（两证合入）。

**窗口池化域退化特化条件（2026-10-14 两证合入——2D `max_pool2d` 首证 + 1D `max_pool1d` 第二证；queue VP-2026-0172 达 2/2）**：
① **整线/整平面归约（`out==1`）的充分条件** = `out==1 ∧ p==0 ∧ d==1 ∧ (k−1)·d+1 ≥ L_in`——窗口的**输入覆盖集合**是 `{k·d}`（`p` 平移、`s` 取窗），`d>1` 时只覆盖等差子集，整线归约静默错值。跨维反例各自独立复现：2D `H=W=5, k=2, s=1, p=0, d=4`（窗口仅 `{(0,0),(0,4),(4,0),(4,4)}`，整平面归约 1.0 vs torch 正确 0.0）；1D `L=5, k=2, s=1, p=0, d=4`（窗口仅 `{0,4}`，整线归约 1.0 vs 正确 0.0）。`d==1` 且该轴 `in==1` 时该项自动成立。
② **恒等拷贝特化**在窗口池化下需 `k==1 ∧ d==1 ∧ p==0 ∧ s==1`（`out==in` 只是「窗口重叠的多次比较」，不蕴含恒等）。
③ **条件型特化回归组**（把充分条件变成可执行断言）：除「条件成立用例」外补「边界不成立用例」（`assert not cond(...)` + 一般路径对照 golden + 负向控制守卫），并要求**逐子句隔离负例**——至少一条用例仅违反单个子句（其余成立），否则「移除任一子句即失败」为空真（对称 `d_H==d_W` 只锁一轴的形态即此缺口）。
④ **触发形态随 workload 集变化**（1D 第二证新增）：2D manifest 三档均不触发任何特化，而 1D manifest 的 `textcnn-global`（`[64,256,128]` k128s1p0 → `L_out=1`）恰好触发整线归约——同一充分条件在不同 workload 集上是否被触发不同，回归组必须覆盖两侧（触发成立 / 边界不成立）而不能只按「代表 shape 不触发」下结论。

证据：首证 task `max_pool2d-_max_pool2d_kernel-20261014T062320Z`（`verify_equiv.py` 57→69 case-group 含 C4x 反例组 + 负向控制守卫，全 PASS；Stage 1 v0 → v1 六处同步）；第二证 task `max_pool1d-_max_pool1d_kernel-20261014T082309Z`（1D 同构复现 `d==1` 必要性 + `verify_equiv.py` 117 case-group 全 PASS 含 `[C4x-GUARD]` 段；kernel L0-11b/d4-guard 与 L0-11b3/d4-bit-exact 实跑走一般路径返回 0.0）。provenance 允许失效（两任务工作区，均未上库）。

**窗口池化域的必需空间分块形态（长轴 / 1D）（2026-10-14 两证合入——1D `max_pool1d` 首证 + 双输出 1D `max_pool1d_with_indices` 第二证；queue VP-2026-0180 达 2/2）**：单空间轴长超 UB 驻留预算时，沿该轴分块是**必需维度**而非可选优化（2D 的 `W_in ≤ 112` 整行驻留结论不可外推；1D `L_in` 可达 4096）。
① **判定式**：空间轴长 × `CH` × 元素宽度 × 膨胀系数 > UB 容量（192 KiB 档）⇒ 必须沿该轴分块，分块后读的是**连续 span**（非逐元素 gather）。
② **钳制载入形态**（恒在界内 + 长度静态）：`span = (BL−1)·s + (k−1)·d + 1`、`LSPAN = min(span, L_in)`、`s0 = max(0, min(base, L_in−LSPAN))`（`base = ow0·s − p`）、读出下标 `j = iw − s0`；守卫 `0 ≤ iw < L_in` 与「缓冲下标 ∈ [0, LSPAN)」是等价式，可写成可执行断言（逐 lane 验下标落界）。
③ **`BL` 取容量允许的最大值**：halo 重读随 `BL` 下降（`LSPAN/(BL·s) → 1`；ecg `k5s2p2d2` 实测 BL=31 ≈1.11× GM 读 vs BL=1 ≈4.5×），方向与 2D 的 `BH=1` 最优**相反**。
④ **访存削减的正确口径**：span 载入的收益 = 「窗口重叠复用 + 连续 span 合并访存」，**不是**「读次数降 `k_W` 倍」（后者只对高度重叠的窗口且以窗口/输入长度比计）——GM 读次数比 = `(num_oblk·LSPAN)/(L_out·k_W)`；`k/s > 1` 的负载降为 ecg ≈ `61·41/(1023·5)` = 0.48×，而 `k = s` 的负载（sincnet `86·48/4096` ≈ 1.01×）读次数≈不变、收益只在**访存模式**（连续 span 取代跨步 gather）。第二证（双输出 1D）复现同口径：`sincnet 86·48/4096` ≈ 1.008×、`ecg 61·41/2048` ≈ 1.22×。
⑤ **轴数前提清单**：整轴驻留 / 守卫外提 / `BH=1` 最优三处 2D 结论均要求 ≥2 个空间轴——同族迁移逐项核对「该结论依赖几个空间轴 / 轴长上限多少」。
⑥ **尾块三类**：任务尾块（`uid < num_logical`）、通道尾块（`real_c = min(CH, C−cb·CH)`）、输出块尾（`BL_real = min(BL, L_out−ow0)`；1D 独有，2D 因整行驻留而无）。
证据：首证 task `max_pool1d-_max_pool1d_kernel-20261014T082309Z`（`DESIGN.md#§0.6 R1/R3`、`#§1.6.1 C1`、`#§4.5`、`#§5.2`；`_span/_choose_bl/_choose_ch_bl/_plan` + `verify_equiv.py` C1 组 117 case-group 全 PASS）；第二证 task `max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261014T100907Z`（同形态在**双输出（argmax 索引）**变体上复现：`DESIGN.md#§0.6 R1/R3`、`#§1.6.1 C1`、`#§4.5`、`#§5.2` 同一 `span/LSPAN/s0/j` 形态 + 三类尾块；`verify_equiv.py` C1 组 90 (config, BL) 结构检查逐 lane 验下标落界；集成 per-op 47/47）——两证为**不同任务、不同算子（输出个数不同）**，该 1D 结构形态跨输出变体成立。provenance 允许失效（两任务工作区，均未上库）。

---
id: ALG-reduction
family: [reduction, statistics]
known_impl: [examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/]
kb_links: [CASE-reduction-logsumexp, CASE-norm-adalayern-stage4（persistent 形态调优档案）, CASE-argmax-argreduce-integration（TileOPs 集成 pad 契约档案）]
---

R1：max + sum 合并扫描；和与平方和一次扫描。R2：分块归约 + 合并（两阶段 / 树形）；Welford。R3：扫描遍数；跨核归并代价。R4：水平 vs 垂直 lane 映射；跨核 sync 代价。row-reduction 族迁移基准形态（全 N 驻留 UB + M 行块分核 + 尾块 T.min 双显式 slice——logsumexp grid 形态 / ada_layer_norm persistent 形态两任务实测，详见 ALG-layernorm R4）。

**R4 附注：TileOPs reduce 族的 Op 层 pad 契约（两证：argmax 2026-09-28 + argmin 2026-10-14，不同算子/不同设备）**

- **契约与声明宽度**：TileOPs reduce 族 Op 层默认把输入 pad 到 `align_up(N,256)`（`_ReduceOpBase._prepare_input`，`_kernel_handles_padding=False` 时）；kernel jit 声明宽度须等于 `N_padded`。raw-N 契约与 Op 层 `F.pad` 冲突的症状：**行 0 正确、行 ≥1 全错且不报错**（步长错位），初次 report 的失败用例与 `N_padded≠N` 用例 100% 重合。
- **pad 单位元方向**：`op_kind` 涉及 min/max 的共享 kernel 须核对 `_pad_value`——argmax 侧 `-inf` 安全（永不赢且全 `-inf` 行与 torch 同返 0），argmin 侧必须 `+inf`（反向会让 pad 列在 argmin 中胜出 → 静默错值）。argmin 第二证：`examples/argmin/_argreduce_kernel/DESIGN.md#§0.2` 双契约对照表 + `examples/TileOPs/tileops/ops/reduction/argmin.py:76-78`（`_pad_value() → +inf`）实读。
- **pad 契约下的索引中性（第二证新增，机器验证）**：在 `N_padded` **全域**计算 argmin 候选 ≡ 只在实列范围计算——实列索引恒 `< N ≤` pad 列索引，故 pad 契约下 kernel **无需掩码/尾 tile 特化**，N 向控制流最简（`verify_equiv.py` E5 组 264 行含全 `+inf` / tie / `±inf` 混合，0 违反；torch CPU fp64 参照 + dtype 舍入仿真）。同源推论：`N_padded` 恒为 256 倍数且 `_pick_tile_n` 候选含 256 并偏好整除 ⇒ tiled 路径静态尾 tile `tail ≡ 0`（该路径属防御性保留，不构成需覆盖的回归面；要真尾块须在工厂期强制非整除 `tile_n`）。
- **高杠杆翻转的边界（第二证修正）**：`_kernel_handles_padding=True` + kernel 接原始 N（规约单位元 padding 语义上不改变结果）在小 N/非对齐 N 的 dim=0/中间维负载最高 64× 流量改善（harness `reduce.py` 注释明示为仓库既定方向）；但**负载画像决定收益**——argmin manifest 以大 N 负载为主、无窄 N 用例时，保留 pad 契约（零 Op 层/wrapper 契约改动）优于 raw-N 翻转（第二证即按此裁决）。翻转前先核 manifest 的 N 分布。
- kb_links 补 CASE-argmax-argreduce-integration（集成契约与胶水修复档案）+ CONST-vector-launch-overhead（小 N 大 M 的发射主导预算另见 queue VP-2026-0137 设计规则集）。
- 溯源（任务工作区，上库部分见 kb_links；集成契约胶水修复形态见 CASE）：首证 `examples/argmax/_argreduce_kernel` + `integration_log.md#attempt-1`；第二证 `examples/argmin/_argreduce_kernel/{DESIGN.md#§0.2/§1.6.1-E5/§5.6, verify_equiv.py E5 组}` + `examples/TileOPs/tileops/kernels/reduction/argmin/argmin_kernel/integration_log.md`（允许失效）。

---
id: ALG-topk
family: [top-k, sort]
known_impl: []
kb_links: []
---

R1：部分选择 vs 全排序；分块 top-k + 归并。R2：分块 + 归并。R3：O(N·k) vs O(N log N)。R4：归并链的核内结构与缓冲。

---
id: ALG-gemv
family: [gemv, matvec]
known_impl: []
kb_links: [TRAP-load-nd2nz-strided（Cube 路径装载形态约束）]
---

R1：GEMM 分形路径（Cube，load_nd2nz）vs Vector 规约累加。R2：split-K 分块累加。R3：Cube 利用率 vs 归约代价。R4：K 维分块与 L0 容量。

---
id: ALG-triangular
family: [triangular-solve, inverse]
known_impl: []
kb_links: []
---

R1：只需解不需逆时——直接求解替代显式求逆（同阶但常数更小、数值更稳）；对角/三角特例降阶。R2：—。R3：求逆 + 乘法 vs 单次求解；特例 O(n²)/O(n)。R4：Cube 三角 GEMM 支持。

---
id: ALG-elementwise
family: [elementwise, activation]
known_impl: [examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/]
kb_links: [TRAP-fp16-opmath-golden, PL-1.6-copy-floor, PL-1.5-quickref]
---

R1：公共子表达式消除；除法乘倒数；cast 链合并。R2：逐元素映射天然单遍。R3：融合消除中间 GM 往返（访存收益主导）。R4：单遍 Vector 流；dtype 路径。**已知实测**：sub-fp32 逐元素算子 fp32 中转链对齐 torch golden（≤1 ulp，TRAP-fp16-opmath-golden）；≥16M 档计算链被 MTE2 窗口完全隐藏（PL-1.6-copy-floor，计算链优化 ROI≈0）。

---
id: ALG-transpose
family: [transpose, repack]
known_impl: []
kb_links: [PL-1.1-transpose-chain, PL-1.3-host-permute, TRAP-transpose-epilogue-poison]
---

R1：核内融合转置链（UB 级，实测 ~µs 级）vs host permute（~百 µs 级）。R2：分块转置。R3：GM 流量不变，代价在向量管线开销。R4：T.transpose dtype 矩阵（fp16/fp32 ✓，bf16/整型 ×）；UB 容量。**已知反例**：epilogue 活跃源 transpose 毒化整 kernel 2.6x（TRAP-transpose-epilogue-poison，绕法 = 增维视图）。
