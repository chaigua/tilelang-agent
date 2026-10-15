# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-reduce-ub-per-element.py -- BishengHIR 归约 lowering 的 UB 单价（delta 形态）

条目: pattern-library/constants.md CONST-reduce-ub-per-element
关联: layout.md PL-1.22-maxreduce-nan-tracking（NaN 追踪形态的选型依据）

现象/优化点（慢→快关键代码更改）:
  整平面 max 归约的 NaN 追踪有两种数学等价写法，BishengHIR 归约 lowering 在 UB 上
  **按输入 tile 元素数线性记账，单价取决于归约链形态**（与真实 dtype 缓冲清单无关）：

    慢（bool/f16 掩码链）: reduce_max -> vcmp(平面自比较) -> vcast -> reduce_min
    快（原位平方和）    : reduce_max -> vmul(flat,flat,flat) -> reduce_sum
    中间（非原位平方）  : reduce_max -> vmul(flat,flat,sq) -> reduce_sum

  实测单价（ch=64 与 ch=128 恒定，严格线性、无固定项；ch×H*W 元素）：
    | 形态 | B/元素 | 说明 |
    |---|---|---|
    | 仅 reduce_max（无 NaN 追踪） | < 24 | 下限对照 |
    | 原位平方 + reduce_sum | **≈ 32** | 落档形态（ch ≤ 120 可编译） |
    | 非原位平方 + reduce_sum | ≈ 64 | 多一个平面缓冲 ⇒ 单价翻倍 |
    | fp16 掩码 + reduce_min | ≈ 96 | |
    | bool 掩码 + vcast + reduce_min | ≈ 128 | 单价最高（ch 被压到 ≤32） |
  可用 CH 上限 = 196608 B / (单价 × H*W)。H*W=49 时：128 B/元素 ⇒ CH≤31；
  32 B/元素 ⇒ CH≤120。global-1x1 落档 CH=64（32 × 64 × 49 = 100 KB，编译通过）。

优化见解摘要（机制归因）:
  BishengHIR 的 `hivm.hir.vreduce` lowering 会为归约链物化多份中间张量
  （见 IR: `memref.alloc` + `tensor.insert_slice` + `hivm.hir.vreduce`），其规模随
  输入 tile 元素数增长；每多一个「全平面中间量」（bool 掩码平面 / 非原位平方平面）
  就多一份等价记账。因此**减少全平面中间量的个数与位宽**（原位操作、用元素级算术
  替代掩码物化）是整平面归约路径解锁更大 CH 的唯一杠杆，与 dtype 缓冲清单无关。

断言: delta 形态（效应只在完整 kernel 规模显现）——本文件只做 py_compile 语法校验；
  关键代码更改以骨架表达（慢/快两形态并排）。
首次验证: 2026-10-15 adaptive_max_pool2d-_adaptive_max_pool2d_kernel Stage 4
  (tilelang 0.1.2+489385e9 / CANN 9.0.0 / Ascend910B3)
  测量命令（需 NPU 编译环境；失败信息含
  「ub overflow, requires N bits while M bits available」）:
    python - <<'PY'   # 把下面 SLOW/FAST 骨架分别包进 T.prim_func 后 build(ch)
      # 记录 required bytes / (ch * H * W) 即单价
    PY
重验记录: (追加式)
"""

# ---- 慢形态骨架（bool 掩码链：平面自比较 + 掩码 reduce_min）----
SLOW_SKELETON = """
import tilelang.language as T

def c4_slow(x, out, ch, PLAIN, nblk, nk, nloc, C):
    with T.Kernel(nk, is_npu=True) as (cid, _):
        flat = T.alloc_ub((ch, 1, PLAIN), "float16")
        nmask = T.alloc_ub((ch, 1, PLAIN), "bool")        # <- full-plane mask
        m32 = T.alloc_ub((ch, 1, PLAIN), "float32")       # <- full-plane mask f32
        g16 = T.alloc_ub((ch, 1, 1), "float16")
        g32 = T.alloc_ub((ch, 1, 1), "float32")
        nmin = T.alloc_ub((ch, 1, 1), "float32")
        for t in T.serial(nloc):
            uid = t * nk + cid
            if uid < nLOCALGUARD:
                T.copy(x[uid // nblk, (uid % nblk) * ch, 0:1, 0:PLAIN],
                       flat[0:ch, 0:1, 0:PLAIN])
                T.reduce_max(flat, g16, dim=2, clear=True)
                T.vcmp(flat, flat, nmask, "eq")           # NaN probe
                T.vcast(nmask, m32, round_mode="rint")    # bool -> f32 plane
                T.reduce_min(m32, nmin, dim=2, clear=True)
                # ... select(ok, gmax, nan) -> store
"""

# ---- 快形态骨架（原位平方 + reduce_sum；无额外全平面中间量）----
FAST_SKELETON = """
import tilelang.language as T

def c4_fast(x, out, ch, PLAIN, nblk, nk, nloc, C):
    with T.Kernel(nk, is_npu=True) as (cid, _):
        flat = T.alloc_ub((ch, 1, PLAIN), "float16")
        g16 = T.alloc_ub((ch, 1, 1), "float16")
        g32 = T.alloc_ub((ch, 1, 1), "float32")
        nsum = T.alloc_ub((ch, 1, 1), "float32")
        for t in T.serial(nloc):
            uid = t * nk + cid
            if uid < nLOCALGUARD:
                T.copy(x[uid // nblk, (uid % nblk) * ch, 0:1, 0:PLAIN],
                       flat[0:ch, 0:1, 0:PLAIN])
                T.reduce_max(flat, g16, dim=2, clear=True)
                T.vmul(flat, flat, flat)                  # in place: flat is dead
                T.reduce_sum(flat, nsum, dim=2, clear=True)   # isnan(sum) <=> has NaN
                # ... select(ok=(sum==sum), gmax, nan) -> store
"""

if __name__ == "__main__":
    import os
    import py_compile
    import tempfile

    for name, skeleton in (("slow", SLOW_SKELETON), ("fast", FAST_SKELETON)):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("import tilelang.language as T\n" if name == "fast" else "")
            f.write(skeleton.replace("nLOCALGUARD", "nl"))
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
            print(f"[CONST-reduce-ub-per-element] {name} skeleton: py_compile PASS")
        except py_compile.PyCompileError as exc:  # pragma: no cover
            raise SystemExit(f"{name} skeleton FAILED: {exc}") from exc
        finally:
            os.unlink(path)
    print("NOTE: per-element UB numbers are compile-time measurements (see header).")
