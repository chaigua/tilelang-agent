"""Step-0 copy-floor calibration harness for
`_adaptive_max_pool2d_with_indices_kernel` (Stage 4).

Pure-copy kernels with the *same grid / DMA granularity* as the two structural
paths of this operator, used as the IO floor and 70%-line anchor:

  * ``global``  -- global-1x1 raster ``[8,2048,7,7]`` fp16, whole-plane
    contiguous DMA (one channel block of ``ch*1*plane`` fp16 per logical unit);
  * ``general`` -- spp-6x6 raster ``[2,128,56,56]`` fp16, per-bin row DMA (the
    rows a bin touches, width ``w_in``).

**Dual-output byte accounting** (this operator's real IO, DESIGN.md section
1.6.0): each unit READs ``ch*plane`` fp16 and WRITEs both the fp16 value output
and the ``int64`` index output, so

    bytes = in_bytes + out_bytes + idx_bytes
          = n_in*2 + n_out*2 + n_out*8         (8 B per index output element)

The write side is reproduced explicitly (value fp16 staging + int64 index
staging, both derived from the copied plane so the read stays live) instead of
the single-element "keep-alive" store -- a single-element store would zero the
write term and produce an optimistically low floor.

The round-count self-variable sweep (PL-1.6-copy-floor discipline) comes from
varying the channel block, which changes ``num_local`` (per-core serial rounds)
at constant total flow.

Invoked through msprof op as:
    msprof op --kernel-name=main ... python calib_bench.py --case global --ch 64
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

# global-1x1 raster
GLOBAL = dict(n=8, c_in=2048, h_in=7, w_in=7, dtype="float16")
# spp-6x6 raster (general grid path)
GENERAL = dict(n=2, c_in=128, h_in=56, w_in=56, out_h=6, out_w=6, dtype="float16")


def _round_up(v, m):
    return ((v + m - 1) // m) * m


def _bin(o, size_in, size_out):
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def _max_extent(size_in, size_out):
    return max(e - s for s, e in (_bin(o, size_in, size_out) for o in range(size_out)))


def _grid(num_logical, vector_cores):
    num_kernels = min(num_logical, vector_cores)
    return num_kernels, (num_logical + num_kernels - 1) // num_kernels


def build_global(cfg, ch, write="dual"):
    n, c_in, h_in, w_in = cfg["n"], cfg["c_in"], cfg["h_in"], cfg["w_in"]
    dtype = cfg["dtype"]
    plane = h_in * w_in
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * num_cblk
    num_kernels, num_local = _grid(num_logical, vector_cores)
    # write modes:
    #   elem     1-element fp16 keep-alive store only (read floor)
    #   dual     runtime-extent (real_c,1,1) regions (step-0 form)
    #   stat3d   static-extent full-buffer 3D copies (value + index)
    #   stat1d   static-extent full-buffer 1-D copies (value + index)
    #   value / index  = stat3d restricted to one output
    wv = write in ("dual", "stat3d", "stat1d", "value")
    wi = write in ("dual", "stat3d", "stat1d", "index")
    static = write in ("stat3d", "stat1d", "value", "index")
    one_d = write == "stat1d"

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, 1, plane), dtype),
            out: T.Tensor((n, c_in, 1, 1), dtype),
            indices: T.Tensor((n, c_in, 1, 1), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, 1, plane), dtype)
                vstg = T.alloc_ub((ch, 1, 1), dtype)
                vstg1d = T.alloc_ub((ch,), dtype)
                istg32 = T.alloc_ub((ch, 1, 1), "int32")
                istg = T.alloc_ub((ch, 1, 1), "int64")
                istg1d = T.alloc_ub((ch,), "int64")
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        real_c = T.min(ch, c_in - cb * ch)
                        # READ: contiguous whole-plane DMA (ch*plane fp16).
                        T.copy(
                            x[nb, cb * ch : cb * ch + real_c, 0:1, 0:plane],
                            flat[0:real_c, 0:1, 0:plane],
                        )
                        T.copy(
                            flat[0:real_c, 0:1, 0:1], vstg[0:real_c, 0:1, 0:1]
                        )
                        T.reshape(vstg, vstg1d)
                        T.vcast(vstg, istg32, round_mode="rint")
                        T.vcast(istg32, istg, round_mode="rint")
                        T.reshape(istg, istg1d)
                        if write == "elem":
                            # max-pool-style single-element keep-alive store
                            T.copy(
                                flat[0:1, 0:1, 0:1],
                                out[nb, cb * ch : cb * ch + 1, 0:1, 0:1],
                            )
                        if wv and write != "elem":
                            if static and one_d:
                                T.copy(
                                    vstg1d[0:ch],
                                    out[nb, cb * ch : cb * ch + ch, 0, 0],
                                )
                            elif static:
                                T.copy(
                                    vstg[0:ch, 0:1, 0:1],
                                    out[nb, cb * ch : cb * ch + ch, 0:1, 0:1],
                                )
                            else:
                                T.copy(
                                    vstg[0:real_c, 0:1, 0:1],
                                    out[nb, cb * ch : cb * ch + real_c, 0:1, 0:1],
                                )
                        if wi and write != "elem":
                            if static and one_d:
                                T.copy(
                                    istg1d[0:ch],
                                    indices[nb, cb * ch : cb * ch + ch, 0, 0],
                                )
                            elif static:
                                T.copy(
                                    istg[0:ch, 0:1, 0:1],
                                    indices[nb, cb * ch : cb * ch + ch, 0:1, 0:1],
                                )
                            else:
                                T.copy(
                                    istg[0:real_c, 0:1, 0:1],
                                    indices[
                                        nb, cb * ch : cb * ch + real_c, 0:1, 0:1
                                    ],
                                )

        return main

    return _func()


def build_general(cfg, ch):
    n, c_in, h_in, w_in = cfg["n"], cfg["c_in"], cfg["h_in"], cfg["w_in"]
    out_h, out_w = cfg["out_h"], cfg["out_w"]
    dtype = cfg["dtype"]
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * out_h * num_cblk
    num_kernels, num_local = _grid(num_logical, vector_cores)
    max_bin_h = _max_extent(h_in, out_h)

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),
            out: T.Tensor((n, c_in, out_h, out_w), dtype),
            indices: T.Tensor((n, c_in, out_h, out_w), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                row = T.alloc_ub((ch, w_in), dtype)
                vstg = T.alloc_ub((ch, out_w), dtype)
                istg32 = T.alloc_ub((ch, out_w), "int32")
                istg = T.alloc_ub((ch, out_w), "int64")
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
                        # READ: per-bin row DMA, width w_in (same granularity
                        # as the real general path).
                        for kh in T.serial(max_bin_h):
                            if kh < bh:
                                T.copy(
                                    x[
                                        nb,
                                        cb * ch : cb * ch + real_c,
                                        ih0 + kh,
                                        0:w_in,
                                    ],
                                    row[0:real_c, 0:w_in],
                                )
                        # READ-derived dual-output staging, then the real
                        # strided write pattern (c-stride = out_h*out_w).
                        T.copy(
                            row[0:real_c, 0:out_w], vstg[0:real_c, 0:out_w]
                        )
                        T.vcast(vstg, istg32, round_mode="rint")
                        T.vcast(istg32, istg, round_mode="rint")
                        T.copy(
                            vstg[0:real_c, 0:out_w],
                            out[nb, cb * ch : cb * ch + real_c, oh, 0:out_w],
                        )
                        T.copy(
                            istg[0:real_c, 0:out_w],
                            indices[nb, cb * ch : cb * ch + real_c, oh, 0:out_w],
                        )

        return main

    return _func()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=("global", "general"))
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--ch", type=int, required=True, help="channel block")
    ap.add_argument("--n", type=int, default=None, help="override batch (round sweep)")
    ap.add_argument(
        "--write",
        default="dual",
        choices=("dual", "stat3d", "stat1d", "value", "index", "elem"),
        help="global write-side variant (default dual = fp16 value + int64 index)",
    )
    args = ap.parse_args()

    cfg = dict(GLOBAL if args.case == "global" else GENERAL)
    if args.n is not None:
        cfg["n"] = args.n
    kernel = (
        build_global(cfg, args.ch, write=args.write)
        if args.case == "global"
        else build_general(cfg, args.ch)
    )
    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16

    torch.manual_seed(0)
    if args.case == "global":
        xshape = (cfg["n"], cfg["c_in"], 1, cfg["h_in"] * cfg["w_in"])
    else:
        xshape = (cfg["n"], cfg["c_in"], cfg["h_in"], cfg["w_in"])
    x = torch.randn(*xshape).to(dtype).npu()

    with torch.no_grad():
        for _ in range(args.iters):
            out, idx = kernel(x)
    print(
        f"[calib] {args.case} ch={args.ch} n={cfg['n']} write={args.write}: "
        f"ran {args.iters} iterations, out={tuple(out.shape)} idx={tuple(idx.shape)}"
    )


if __name__ == "__main__":
    main()
