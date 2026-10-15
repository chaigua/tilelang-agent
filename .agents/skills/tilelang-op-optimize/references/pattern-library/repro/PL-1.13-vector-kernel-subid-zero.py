# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.13-vector-kernel-subid-zero.py -- PL-1.13 适用域收窄：纯 Vector kernel 的 subid 恒为 0

条目: pattern-library/attention.md PL-1.13-aiv-dup-subid-split（update：适用域收窄）
关联: layout.md PL-1.21-vector-persistent-split（纯 Vector persistent 分核模板）

现象（反向陷阱）:
  PL-1.13 的「双 AIV 重复执行 + subid 边界分片」形态只适用于 **Mix** kernel
  （`T.Scope("Vector")` 与 Cube 共块）。对 `T.Kernel(n, is_npu=True)` 的**纯
  Vector** kernel，块内只有 1 个 Vector 子块，`subid` 恒为 0 —— 此时按 subid
  分片会把「偶 index 的逻辑单元」分给 subid=0、把奇 index 分给永不执行的
  subid=1，**静默丢弃一半工作**（profile 判据：GM 流量精确减半）。这种「加速」
  是假的（实测 -20%~-41% 的 Task Duration 下降 = 只做了一半）。

断言:
  对多个 grid 尺寸，kernel 把 `1 + subid` 写入 `out[cid*2 + subid]`；
  若 subid 真的取 0/1，必有槽位 = 2.0。断言 `max(out) == 1.0`（恒无 2.0）
  ⇒ subid 恒 0。需要 NPU 执行。

首次验证: 2026-10-15 adaptive_max_pool2d-_adaptive_max_pool2d_kernel Stage 4
  (tilelang 0.1.2+489385e9 / CANN 9.0.0 / Ascend910B3)
  标定侧同证据: global 拷贝 kernel 默认 vs subid 分片，GM_to_UB 36.8 -> 18.4
  KB/block（精确半量）；本探针给出决定性判据。
重验记录: (追加式)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def build(nblk: int):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((nblk * 2,), "float32"),
            out: T.Tensor((nblk * 2,), "float32"),
        ):
            with T.Kernel(nblk, is_npu=True) as (cid, subid):
                val = T.alloc_ub((1,), "float32")
                T.vbrc(T.cast(1 + subid, "float32"), val)
                T.copy(val[0:1], out[cid * 2 + subid])

        return main

    return _func()


def main():
    for nblk in (4, 6):
        kernel = build(nblk)
        x = torch.zeros(nblk * 2, dtype=torch.float32).npu()
        got = kernel(x).cpu()
        even = got[0::2].tolist()
        odd = got[1::2].tolist()
        print(f"nblk={nblk} out={got.tolist()}")
        print(f"  even slots (1.0 == subid 0): {even}")
        print(f"  odd  slots (2.0 iff a 2nd AIV runs): {odd}")
        assert float(got.max()) == 1.0, (
            "a slot holds 2.0 => subid actually varies; PL-1.13 split applies")
    print("\n[PL-1.13] PASS: subid is always 0 for a pure-Vector npuir kernel -- "
          "a subid-based partition would silently drop half the logical units.")


if __name__ == "__main__":
    main()
