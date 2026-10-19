"""Probe: can the C4 (global) path be done with a smaller UB footprint so that
CH can grow past 32 (the copy floor at CH>=64 is much lower)?

Variants (all reduce the flat (CH,1,H*W) tile over dim=2):
  v_f32 : control -- fp16->fp32 plane, reduce_max fp32, vcmp fp32, reduce_min fp32
  v_f16 : reduce_max on the fp16 plane; vcmp fp16 -> bool -> fp32; reduce_min fp32
  v_f16b: same as v_f16 (kept for A/B of the epilogue shape)

Correctness is checked against torch.nn.functional.adaptive_max_pool2d
(bit-exact, equal_nan=True) on the global-1x1 shape.

    python probe_c4_ch.py --variant v_f16 --ch 64
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

N, C, H, W = 8, 2048, 7, 7
PLANE = H * W


def _grid(ch):
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    nblk = (C + ch - 1) // ch
    num_logical = N * nblk
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    return nblk, num_logical, num_kernels, num_local


def build_f32(ch, dtype="float16"):
    nblk, num_logical, num_kernels, num_local = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, PLANE), dtype),
            out: T.Tensor((N, C, 1, 1), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), dtype)
                flat32 = T.alloc_ub((ch, 1, PLANE), "float32")
                nmask = T.alloc_ub((ch, 1, PLANE), "bool")
                nmask32 = T.alloc_ub((ch, 1, PLANE), "float32")
                gmax = T.alloc_ub((ch, 1, 1), "float32")
                nmin = T.alloc_ub((ch, 1, 1), "float32")
                zeros = T.alloc_ub((ch, 1, 1), "float32")
                infbuf = T.alloc_ub((ch, 1, 1), "float32")
                nanbuf = T.alloc_ub((ch, 1, 1), "float32")
                cond = T.alloc_ub((ch, 1, 1), "bool")
                out32 = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), dtype)
                pos_inf = T.infinity("float32")
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:PLANE],
                               flat[0:real_c, 0:1, 0:PLANE])
                        T.vcast(flat, flat32, round_mode="rint")
                        T.reduce_max(flat32, gmax, dim=2, clear=True)
                        T.vcmp(flat32, flat32, nmask, "eq")
                        T.vcast(nmask, nmask32, round_mode="rint")
                        T.reduce_min(nmask32, nmin, dim=2, clear=True)
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


def build_f16(ch, dtype="float16"):
    nblk, num_logical, num_kernels, num_local = _grid(ch)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, PLANE), dtype),
            out: T.Tensor((N, C, 1, 1), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, PLANE), dtype)
                nmask = T.alloc_ub((ch, 1, PLANE), "bool")
                nmask32 = T.alloc_ub((ch, 1, PLANE), "float32")
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
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // nblk
                        cb = uid % nblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(x[nb, cb * ch: cb * ch + real_c, 0:1, 0:PLANE],
                               flat[0:real_c, 0:1, 0:PLANE])
                        T.reduce_max(flat, gmax16, dim=2, clear=True)
                        T.vcmp(flat, flat, nmask, "eq")
                        T.vcast(nmask, nmask32, round_mode="rint")
                        T.reduce_min(nmask32, nmin, dim=2, clear=True)
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


BUILDERS = {"v_f32": build_f32, "v_f16": build_f16, "v_f16b": build_f16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=tuple(BUILDERS))
    ap.add_argument("--ch", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16)
    x[0, 0, 0, 0] = float("nan")
    x[1, 5, 3, 4] = float("inf")
    x[2, 7, 6, 6] = float("-inf")
    x[3, 9, :, :] = float("nan")

    kernel = BUILDERS[args.variant](args.ch)
    got = kernel(x.npu()).cpu()
    ref = F.adaptive_max_pool2d(x.float(), (1, 1)).to(torch.float16)
    ok = bool(torch.equal(got, ref)) or bool(
        torch.allclose(got.float(), ref.float(), atol=0.0, rtol=0.0, equal_nan=True)
    )
    diff = (got.float() - ref.float()).abs().max().item()
    print(f"[c4] variant={args.variant} ch={args.ch}: exact={ok} max_diff={diff:.3e}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
