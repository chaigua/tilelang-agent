#!/usr/bin/env python3
"""Micro-probe: cost of the dq write-back forms on this toolchain.

Modes:
  row     : one `T.atomic_add(dst[i, 0:dim], src[r, 0:dim], [1, dim])` per row
            (the form the Stage 3 kernel uses on the strided BSHD destination)
  row_nosize: same without the explicit `size=`
  region  : ONE `T.atomic_add(dst[i0:i0+rows, 0:dim], src, [rows, dim])` per
            chunk (contiguous 2-D destination)
  store   : plain region store of the same bytes (floor reference)

Each block repeats the write ring `rep` times over its own destination slice,
so the bytes are identical across modes.

Usage:
    msprof op --kernel-name=atomic_probe --launch-count=10 --warm-up=3 \
        --dump=off --aic-metrics=BasicInfo,PipeUtilization,Memory \
        python probe_atomic.py --mode row
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

ROWS = 32
DIM = 128
REP = 64
BLOCKS = 20


@tilelang.jit(target="npuir")
def _probe(mode: str):
    n = BLOCKS * ROWS * REP

    @T.prim_func
    def atomic_probe(
        src: T.Tensor((ROWS, DIM), "float32"),
        dst: T.Tensor((n, DIM), "float32"),
    ) -> None:
        with T.Kernel(BLOCKS, is_npu=True) as (cid, _):
            s = T.alloc_shared((ROWS, DIM), "float32")
            T.copy(src, s)
            base = cid * ROWS * REP
            for it in T.serial(REP):
                if mode == "row":
                    for r in T.serial(ROWS):
                        T.atomic_add(
                            dst[base + it * ROWS + r, 0:DIM],
                            s[r, 0:DIM],
                            [1, DIM],
                        )
                elif mode == "row_nosize":
                    for r in T.serial(ROWS):
                        T.atomic_add(
                            dst[base + it * ROWS + r, 0:DIM],
                            s[r, 0:DIM],
                        )
                elif mode == "region":
                    T.atomic_add(
                        dst[base + it * ROWS : base + (it + 1) * ROWS, 0:DIM],
                        s,
                        [ROWS, DIM],
                    )
                else:  # store
                    T.copy(
                        s,
                        dst[base + it * ROWS : base + (it + 1) * ROWS, 0:DIM],
                    )

    return atomic_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="row",
                    choices=["row", "row_nosize", "region", "store"])
    ap.add_argument("--launch-count", type=int, default=10)
    args = ap.parse_args()
    src = torch.ones(ROWS, DIM, dtype=torch.float32, device="npu")
    dst = torch.zeros(BLOCKS * ROWS * REP, DIM, dtype=torch.float32, device="npu")
    k = _probe(args.mode)
    for _ in range(args.launch_count):
        k(src, dst)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
