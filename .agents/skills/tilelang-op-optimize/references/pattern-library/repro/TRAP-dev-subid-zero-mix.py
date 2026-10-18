#!/usr/bin/env python3
"""TRAP-dev-subid-zero-mix: Developer 模式 Mix kernel 的 `subid` 恒为 0。

`with T.Kernel(n, is_npu=True) as (cid, subid): out[cid*2+subid] = 1 + subid`
输出恒为 [1, 0, 1, 0, ...]（max = 1.0，从无 2.0）⇒ 两个 AIV 子块执行同一份代码，
按 subid 切分会静默丢一半工作（本任务 r4 精度全错）。

判据（真实 kernel）：两个 AIV 子块的 `GM_to_UB_datas` 相同而 `UB_to_GM_datas` 只在
vector0 出现（本任务 45045 vs 43698 KB / 16432 vs 0 KB）⇒ Vector 程序复制执行。

origin_task: TileOPs-_gqa_bwd_kernel-20261018T145922Z
toolchain:   tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4 / 2026-10-18
First verified: 2026-10-18, tilelang 0.1.2+2fb23ee + CANN 9.0.0 + Ascend910B4
"""
"""Probe: is the AIV sub-block index (`subid`) 0/1 in this Developer-mode
`T.Kernel(..., is_npu=True)` kernel, or always 0?

Writes ``1 + subid`` into ``out[cid * 2 + subid]`` (one element per AIV) and
prints the result.  A correct Mix-kernel split shows both 1.0 and 2.0; a
constant-zero subid shows only 1.0 (the PL-1.13 silent-half-work signature).

Usage: python probe_subid.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


@tilelang.jit(target="npuir")
def _probe(n_blocks: int):
    @T.prim_func
    def _main(out: T.Tensor((n_blocks * 2,), "float32")) -> None:
        with T.Kernel(n_blocks, is_npu=True) as (cid, subid):
            v = T.alloc_var("float32")
            v = T.Cast("float32", subid + 1)
            out[cid * 2 + subid] = v

    return _main


def main():
    n = 4
    out = torch.zeros(n * 2, dtype=torch.float32, device="npu")
    _probe(n)(out)
    torch.npu.synchronize()
    vals = out.cpu().tolist()
    print("out =", vals)
    print("max =", max(vals), "(2.0 -> subid is 0/1; 1.0 -> subid always 0)")


if __name__ == "__main__":
    main()
