"""Compile-only UB sweep: which C4 op inflates the UB requirement?

Four cumulative builder levels at a given CH; each prints COMPILE OK or the
BishengHIR `ub overflow, requires N bits` value.

    python probe_ub.py --ch 64
"""

import argparse
import os
import re

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

N, C, H, W = 8, 2048, 7, 7
PLANE = H * W


def _grid(ch):
    vc = NPUUtils.get().get_aicore_num() * 2
    nblk = (C + ch - 1) // ch
    nl = N * nblk
    return nblk, nl, min(nl, vc), (nl + min(nl, vc) - 1) // min(nl, vc)


def lvl1(ch, dtype="float16"):
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, PLANE), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), dtype)
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:PLANE],
                               flat[0:real_c, 0:1, 0:PLANE])
                        T.copy(flat[0:1, 0:1, 0:1], out[nb, cb * ch: cb * ch + 1, 0:1, 0:1])

        return main

    return _func()


def lvl2(ch, dtype="float16"):
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, PLANE), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), dtype)
                g16 = T.alloc_ub((ch, 1, 1), dtype)
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:PLANE],
                               flat[0:real_c, 0:1, 0:PLANE])
                        T.reduce_max(flat, g16, dim=2, clear=True)
                        T.copy(g16[0:1, 0:1, 0:1], out[nb, cb * ch: cb * ch + 1, 0:1, 0:1])

        return main

    return _func()


def lvl3(ch, dtype="float16"):
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, PLANE), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), dtype)
                g16 = T.alloc_ub((ch, 1, 1), dtype)
                nmsk = T.alloc_ub((ch, 1, PLANE), "bool")
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:PLANE],
                               flat[0:real_c, 0:1, 0:PLANE])
                        T.reduce_max(flat, g16, dim=2, clear=True)
                        T.vcmp(flat, flat, nmsk, "eq")
                        T.copy(g16[0:1, 0:1, 0:1], out[nb, cb * ch: cb * ch + 1, 0:1, 0:1])

        return main

    return _func()


def lvl4(ch, dtype="float16"):
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, PLANE), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), dtype)
                g16 = T.alloc_ub((ch, 1, 1), dtype)
                nmsk = T.alloc_ub((ch, 1, PLANE), "bool")
                nmin = T.alloc_ub((ch, 1, 1), "float32")
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:PLANE],
                               flat[0:real_c, 0:1, 0:PLANE])
                        T.reduce_max(flat, g16, dim=2, clear=True)
                        T.vcmp(flat, flat, nmsk, "eq")
                        T.reduce_min(nmsk, nmin, dim=2, clear=True)
                        T.copy(g16[0:1, 0:1, 0:1], out[nb, cb * ch: cb * ch + 1, 0:1, 0:1])

        return main

    return _func()


LEVELS = {1: lvl1, 2: lvl2, 3: lvl3, 4: lvl4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--levels", default="1,2,3,4")
    args = ap.parse_args()
    for level in [int(v) for v in args.levels.split(",")]:
        try:
            LEVELS[level](args.ch, args.dtype)
            print(f"ch={args.ch} dtype={args.dtype} level={level}: COMPILE OK")
        except Exception as exc:  # noqa: BLE001
            m = re.search(r"requires (\d+) bits while (\d+) bits", str(exc))
            if m:
                print(f"ch={args.ch} dtype={args.dtype} level={level}: OVERFLOW requires "
                      f"{int(m.group(1))//8} B (avail {int(m.group(2))//8} B)")
            else:
                print(f"ch={args.ch} dtype={args.dtype} level={level}: FAIL {str(exc)[:180]}")


if __name__ == "__main__":
    main()
