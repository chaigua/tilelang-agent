#!/usr/bin/env python3
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
