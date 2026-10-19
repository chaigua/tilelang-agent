"""Step-0 copy-floor calibration harness for `_adaptive_max_pool2d_kernel`.

Pure-copy kernels with the *same grid / DMA granularity* as the two structural
paths of the tuned kernel, used as the IO floor and Ratio ceiling anchor of this
operator (DESIGN.md section 11.1):

  * ``global``  -- global-1x1 raster ``[8,2048,7,7]`` fp16, whole-plane
    contiguous DMA (one channel block of CH*1*49 fp16 per logical unit);
  * ``general`` -- spp-6x6 raster ``[2,128,56,56]`` fp16, per-bin row DMA
    (the rows a bin touches, width W_in).

Both kernels read the full input and write one element per output position so
the measured Task Duration is the IO floor of the real kernel (no compute).

Invoked through msprof op as:
    msprof op --kernel-name=main ... python calib_bench.py --case global
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

# global-1x1 raster (CH=32, mirroring the C4 path candidate block)
GLOBAL = dict(n=8, c_in=2048, h_in=7, w_in=7, ch=32, dtype="float16")
# spp-6x6 raster (CH=32, general grid path)
GENERAL = dict(n=2, c_in=128, h_in=56, w_in=56, out_h=6, out_w=6, ch=32,
               dtype="float16")


def _round_up(v, m):
    return ((v + m - 1) // m) * m


def _bin(o, size_in, size_out):
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def _max_extent(size_in, size_out):
    return max(e - s for s, e in (_bin(o, size_in, size_out) for o in range(size_out)))


def build_global(cfg, split=False):
    n, c_in, h_in, w_in, ch = cfg["n"], cfg["c_in"], cfg["h_in"], cfg["w_in"], cfg["ch"]
    dtype = cfg["dtype"]
    plane = h_in * w_in
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * num_cblk
    num_kernels = min(num_logical, vector_cores)
    if split:
        # PL-1.13: both AIVs of a block run the same instruction stream by
        # default; split the logical units across the two AIVs via subid.
        num_local = (num_logical + num_kernels * 2 - 1) // (num_kernels * 2)
    else:
        num_local = (num_logical + num_kernels - 1) // num_kernels
    # Closed-over compile-time integers (no traced `if` -> parser-safe).
    grid_stride = num_kernels * 2 if split else num_kernels
    blk_stride = 2 if split else 1
    sub_mul = 1 if split else 0

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, 1, plane), dtype),
            out: T.Tensor((n, c_in, 1, 1), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, subid):
                tile = T.alloc_ub((ch, 1, plane), dtype)
                for t in T.serial(num_local):
                    uid = t * grid_stride + cid * blk_stride + subid * sub_mul
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        real_c = T.min(ch, c_in - cb * ch)
                        T.copy(
                            x[nb, cb * ch: cb * ch + real_c, 0:1, 0:plane],
                            tile[0:real_c, 0:1, 0:plane],
                        )
                        # Floor probe: a single element store keeps the input copy
                        # alive (data dependency) without adding the strided
                        # column-write amplification of the real epilogue.  This
                        # matches the sibling's proven `probe_c4 copy` method.
                        T.copy(
                            tile[0:1, 0:1, 0:1],
                            out[nb, cb * ch: cb * ch + 1, 0:1, 0:1],
                        )

        return main

    return _func()


def build_general(cfg):
    n, c_in, h_in, w_in = cfg["n"], cfg["c_in"], cfg["h_in"], cfg["w_in"]
    out_h, out_w, ch = cfg["out_h"], cfg["out_w"], cfg["ch"]
    dtype = cfg["dtype"]
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * out_h * num_cblk
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    max_bin_h = _max_extent(h_in, out_h)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),
            out: T.Tensor((n, c_in, out_h, out_w), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                row = T.alloc_ub((ch, w_in), dtype)
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // (out_h * num_cblk)
                        rem = uid % (out_h * num_cblk)
                        oh = rem // num_cblk
                        cb = rem % num_cblk
                        real_c = T.min(ch, c_in - cb * ch)
                        ih0, ih1 = _bin(oh, h_in, out_h)
                        bh = ih1 - ih0
                        for kh in T.serial(max_bin_h):
                            if kh < bh:
                                T.copy(
                                    x[nb, cb * ch: cb * ch + real_c, ih0 + kh, 0:w_in],
                                    row[0:real_c, 0:w_in],
                                )
                        T.copy(
                            row[0:1, 0:1],
                            out[nb, cb * ch: cb * ch + 1, oh, 0:1],
                        )

        return main

    return _func()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=("global", "general"))
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--ch", type=int, default=None, help="override channel block")
    ap.add_argument("--split", action="store_true",
                    help="PL-1.13: split logical units across the two AIVs via subid")
    args = ap.parse_args()

    cfg = dict(GLOBAL if args.case == "global" else GENERAL)
    if args.ch is not None:
        cfg["ch"] = args.ch
    if args.case == "global":
        kernel = build_global(cfg, split=args.split)
    else:
        kernel = build_general(cfg)
    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16

    torch.manual_seed(0)
    x = torch.randn(cfg["n"], cfg["c_in"], cfg["h_in"], cfg["w_in"]).to(dtype).npu()

    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[calib] {args.case}: ran {args.iters} iterations, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
