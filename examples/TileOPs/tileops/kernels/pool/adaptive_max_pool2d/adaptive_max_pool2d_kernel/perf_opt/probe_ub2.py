"""Compile-only sweep of the *exact* lean fp16 C4 (no dead code).

Mirrors the tuned C4 body (all buffers consumed: the NaN mask reduction feeds the
select, which feeds the store) so BishengHIR cannot eliminate anything.

    python probe_ub2.py --ch 32 --dtype float16
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


def lean(ch, dtype="float16", mb=True, plane=PLANE):
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": mb})
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, plane), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, plane), dtype)
                nmask = T.alloc_ub((ch, 1, plane), "bool")
                gmax16 = T.alloc_ub((ch, 1, 1), dtype)
                gmax = T.alloc_ub((ch, 1, 1), "float32")
                nmin = T.alloc_ub((ch, 1, 1), "float32")
                zeros = T.alloc_ub((ch, 1, 1), "float32")
                infbuf = T.alloc_ub((ch, 1, 1), "float32")
                nanbuf = T.alloc_ub((ch, 1, 1), "float32")
                cond = T.alloc_ub((ch, 1, 1), "bool")
                out32 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), dtype)
                pos_inf = T.infinity("float32")
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:plane],
                               flat[0:real_c, 0:1, 0:plane])
                        T.reduce_max(flat, gmax16, dim=2, clear=True)
                        T.vcmp(flat, flat, nmask, "eq")
                        T.reduce_min(nmask, nmin, dim=2, clear=True)
                        T.vcast(gmax16, gmax, round_mode="rint")
                        T.clear(zeros)
                        T.vcmp(nmin, zeros, cond, "eq")
                        T.vbrc(pos_inf, infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vselect(cond, nanbuf, gmax, out32)
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1, 0:1],
                               out[nb, cb * ch: cb * ch + real_c, 0:1, 0:1])

        return main

    return _func()


def maxonly(ch, dtype="float16", mb=True, plane=PLANE):
    """Copy + reduce_max only (one reduce) -- isolates the per-reduce UB cost."""
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": mb})
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, plane), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, plane), dtype)
                gmax16 = T.alloc_ub((ch, 1, 1), dtype)
                gout = T.alloc_ub((ch, 1, 1), dtype)
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:plane],
                               flat[0:real_c, 0:1, 0:plane])
                        T.reduce_max(flat, gmax16, dim=2, clear=True)
                        T.copy(gmax16[0:real_c, 0:1, 0:1],
                               out[nb, cb * ch: cb * ch + real_c, 0:1, 0:1])

        return main

    return _func()


VARIANTS = {"lean": lean, "maxonly": maxonly}


def lean16m(ch, dtype="float16", mb=True, plane=PLANE):
    """Lean C4 but the NaN mask is reduced from an fp16 plane (not bool)."""
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": mb})
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, plane), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, plane), dtype)
                nmask = T.alloc_ub((ch, 1, plane), "bool")
                m16 = T.alloc_ub((ch, 1, plane), dtype)
                gmax16 = T.alloc_ub((ch, 1, 1), dtype)
                nmin16 = T.alloc_ub((ch, 1, 1), dtype)
                gmax = T.alloc_ub((ch, 1, 1), "float32")
                nmin = T.alloc_ub((ch, 1, 1), "float32")
                zeros = T.alloc_ub((ch, 1, 1), "float32")
                infbuf = T.alloc_ub((ch, 1, 1), "float32")
                nanbuf = T.alloc_ub((ch, 1, 1), "float32")
                cond = T.alloc_ub((ch, 1, 1), "bool")
                out32 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), dtype)
                pos_inf = T.infinity("float32")
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:plane],
                               flat[0:real_c, 0:1, 0:plane])
                        T.reduce_max(flat, gmax16, dim=2, clear=True)
                        T.vcmp(flat, flat, nmask, "eq")
                        T.vcast(nmask, m16, round_mode="rint")
                        T.reduce_min(m16, nmin16, dim=2, clear=True)
                        T.vcast(gmax16, gmax, round_mode="rint")
                        T.vcast(nmin16, nmin, round_mode="rint")
                        T.clear(zeros)
                        T.vcmp(nmin, zeros, cond, "eq")
                        T.vbrc(pos_inf, infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vselect(cond, nanbuf, gmax, out32)
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1, 0:1],
                               out[nb, cb * ch: cb * ch + real_c, 0:1, 0:1])

        return main

    return _func()


def square2r(ch, dtype="float16", mb=True, plane=PLANE):
    """C4 with NaN detection folded into a second fp16-plane reduce.

    has_nan  <=>  isnan(reduce_sum(x*x)).  x*x is NaN exactly where x is NaN and
    >= 0 everywhere else, so the sum can never become NaN through +inf - inf
    (the failure mode of summing the raw values); it is NaN iff a NaN is in the
    plane.  Two fp16-plane reduces are far cheaper in UB than the bool-mask
    chain, which unlocks CH up to ~96.
    """
    nblk, nl, nk, nloc = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir",
                  pass_configs={"npuir.enable_auto_multi_buffer": mb})
    def _func():
        @T.prim_func
        def main(x: T.Tensor((N, C, 1, plane), dtype), out: T.Tensor((N, C, 1, 1), dtype)):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, plane), dtype)
                # (in-place square: no extra plane buffer)
                gmax16 = T.alloc_ub((ch, 1, 1), dtype)
                nsum = T.alloc_ub((ch, 1, 1), dtype)
                gmax = T.alloc_ub((ch, 1, 1), "float32")
                ok = T.alloc_ub((ch, 1, 1), "bool")
                infbuf = T.alloc_ub((ch, 1, 1), "float32")
                nanbuf = T.alloc_ub((ch, 1, 1), "float32")
                out32 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), dtype)
                pos_inf = T.infinity("float32")
                for t in T.serial(nloc):
                    uid = t * nk + cid
                    if uid < nl:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:plane],
                               flat[0:real_c, 0:1, 0:plane])
                        T.reduce_max(flat, gmax16, dim=2, clear=True)
                        T.vmul(flat, flat, flat)
                        T.reduce_sum(flat, nsum, dim=2, clear=True)
                        T.vcast(gmax16, gmax, round_mode="rint")
                        T.vcmp(nsum, nsum, ok, "eq")
                        T.vbrc(pos_inf, infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vselect(ok, gmax, nanbuf, out32)
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1, 0:1],
                               out[nb, cb * ch: cb * ch + real_c, 0:1, 0:1])

        return main

    return _func()


VARIANTS = {"lean": lean, "maxonly": maxonly, "lean16m": lean16m, "square2r": square2r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--variant", default="lean", choices=tuple(VARIANTS))
    ap.add_argument("--plane", type=int, default=PLANE)
    ap.add_argument("--no-mb", action="store_true")
    args = ap.parse_args()
    mb = not args.no_mb
    tag = f"ch={args.ch} plane={args.plane} dtype={args.dtype} mb={mb} variant={args.variant}"
    try:
        VARIANTS[args.variant](args.ch, args.dtype, mb, args.plane)
        print(f"{tag}: COMPILE OK")
    except Exception as exc:  # noqa: BLE001
        m = re.search(r"requires (\d+) bits while (\d+) bits", str(exc))
        if m:
            req = int(m.group(1)) // 8
            print(f"{tag}: OVERFLOW requires {req} B "
                  f"({req / (args.ch * args.plane):.1f} B/elem)")
        else:
            print(f"{tag}: FAIL {str(exc)[:200]}")


if __name__ == "__main__":
    main()
