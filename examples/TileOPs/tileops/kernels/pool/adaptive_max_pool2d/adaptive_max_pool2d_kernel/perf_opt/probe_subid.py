"""Diagnose whether `subid` is meaningful for a pure-Vector npuir kernel.

For a 1D `T.Kernel(n, is_npu=True)` Vector kernel, does the block consist of
two AIVs that both execute the body (with subid = 0/1), or a single Vector
sub-block (subid always 0)?

The probe writes `subid` into a distinct output slot per (block, subid):

    out[cid * 2 + subid] = subid      # only if subid actually varies

and also writes a sentinel (-1) into the odd slots beforehand, so the check
`out[1::2] == 0` distinguishes 'subid always 0' from 'subid = 1 on the second
AIV'.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def build(nblk=4):
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
    nblk = 4
    for nblk in (4, 6):
        kernel = build(nblk)
        x = torch.zeros(nblk * 2, dtype=torch.float32).npu()
        got = kernel(x).cpu()
        print(f"nblk={nblk} out =", got.tolist())
        print(f"  even slots (1.0 if only subid=0): {got[0::2].tolist()}")
        print(f"  odd  slots (2.0 if a 2nd AIV runs): {got[1::2].tolist()}")


if __name__ == "__main__":
    main()
