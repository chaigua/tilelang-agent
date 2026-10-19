"""Step-0 DMA-ceiling probes (2D GM declarations) for `_max_pool2d_kernel`.

The first calibration round (`copy_floor.py`) showed the per-burst cost, not the
burst *length*, dominating: 224 B and 1792 B bursts both landed at ~500 GB/s
aggregate.  This file measures the ceiling as a function of the contiguous span
per channel (the only DMA knob that matters) and of the direction (read /
write), plus the realistic read+write mix of a multi-output-row unit.

GM tensors are declared 2D/3D (`x: (n, c, h*w)`) so a multi-row copy is a 2D
region with a long contiguous per-channel span -- the JIT accepts the original
4D torch tensor (verified), so the kernel ABI is unchanged.

Variants: see VARIANTS.  `--compile-only` checks UB feasibility.
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils

_DT = "float16"
_EB = 2

# shape key -> (n, c, h_in, w_in, out_w, s_h, k_h)
SHAPES = {
    "resnet": (32, 64, 112, 112, 56, 2, 3),
    "vgg": (16, 128, 56, 56, 28, 2, 2),
    "alexnet": (32, 64, 55, 55, 27, 2, 3),
}

# variant -> (shape, ch, read_rows, write_rows, n_units, kind)
#   rows are counted in *input* rows for read and *output* rows for write
VARIANTS = {
    # ---- read ceiling vs contiguous span per channel -----------------------
    "rd_c1_r1":   ("resnet", 1, 1, 1, 112, "read"),     # 224 B
    "rd_c1_r8":   ("resnet", 1, 8, 1, 14, "read"),      # 1792 B
    "rd_c1_r28":  ("resnet", 1, 28, 1, 4, "read"),      # 6272 B
    "rd_c1_plane": ("resnet", 1, 112, 1, 1, "read"),    # 25088 B (whole plane)
    "rd_c4_r8":   ("resnet", 4, 8, 1, 14, "read"),
    "rd_c16_r8":  ("resnet", 16, 8, 1, 14, "read"),
    "rd_c32_r1":  ("resnet", 32, 1, 1, 112, "read"),    # baseline pattern
    "rd_c32_r8":  ("resnet", 32, 8, 1, 14, "read"),
    "rd_c32_r17": ("resnet", 32, 17, 1, 7, "read"),     # BH=8 real unit
    # ---- write ceiling vs contiguous span per channel ----------------------
    "wr_c32_r1":  ("resnet", 32, 1, 1, 56, "write"),    # 112 B (current store)
    "wr_c32_r8":  ("resnet", 32, 1, 8, 7, "write"),     # 1792 B
    "wr_c32_r56": ("resnet", 32, 1, 56, 1, "write"),    # 6272 B (whole plane)
    "wr_c1_r56":  ("resnet", 1, 1, 56, 1, "write"),
    # ---- read+write mix of a BH=8 unit (resnet) ---------------------------
    "rw_c32_bh8": ("resnet", 32, 17, 8, 7, "rw"),
    "rw_c32_bh4": ("resnet", 32, 9, 4, 14, "rw"),
    "rw_c32_bh1": ("resnet", 32, 3, 1, 56, "rw"),
    # ---- vgg / alexnet realistic units ------------------------------------
    "rw_vgg_c64_bh8": ("vgg", 64, 17, 8, 4, "rw"),
    "rw_vgg_c64_bh1": ("vgg", 64, 3, 1, 28, "rw"),
    "rw_alx_c64_bh8": ("alexnet", 64, 17, 8, 4, "rw"),
    "rw_alx_c64_bh1": ("alexnet", 64, 3, 1, 27, "rw"),
    "rw_c32_bh2": ("resnet", 32, 5, 2, 28, "rw"),
    "rd_vgg_c1_plane": ("vgg", 1, 56, 1, 1, "read"),
    "rd_vgg_c1_r8": ("vgg", 1, 8, 1, 7, "read"),
    "rd_alx_c1_plane": ("alexnet", 1, 55, 1, 1, "read"),
    "wr_c1_r1": ("resnet", 1, 1, 1, 56, "write"),
}

_VECTOR_CORES = None


def _cores():
    global _VECTOR_CORES
    if _VECTOR_CORES is None:
        _VECTOR_CORES = NPUUtils.get().get_aicore_num() * 2
    return _VECTOR_CORES


@tilelang.jit(out_idx=[1], target="npuir")
def _k(n, c_in, hw_in, hw_out, ch, read_span, write_span, n_units, kind):
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * n_units * num_cblk
    num_kernels = min(num_logical, _cores())
    num_local = (num_logical + num_kernels - 1) // num_kernels
    mid_span = min(read_span, write_span) if kind == "read" else write_span

    @T.prim_func
    def main(
        x: T.Tensor((n, c_in, hw_in), _DT),
        out: T.Tensor((n, c_in, hw_out), _DT),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            buf = T.alloc_ub((ch, read_span), _DT)
            st = T.alloc_ub((ch, write_span), _DT)
            if kind == "write":
                T.clear(st)
            for t in T.serial(num_local):
                uid = t * num_kernels + cid
                if uid < num_logical:
                    nb = uid // (n_units * num_cblk)
                    rem = uid % (n_units * num_cblk)
                    u = rem // num_cblk
                    cb = rem % num_cblk
                    real_c = T.min(ch, c_in - cb * ch)
                    if kind != "write":
                        T.copy(
                            x[nb, cb * ch:cb * ch + real_c,
                              u * read_span:u * read_span + read_span],
                            buf[0:real_c, 0:read_span],
                        )
                        T.copy(buf[0:real_c, 0:mid_span],
                               st[0:real_c, 0:mid_span])
                    T.copy(
                        st[0:real_c, 0:write_span],
                        out[nb, cb * ch:cb * ch + real_c,
                            u * write_span:u * write_span + write_span],
                    )

    return main


def build(case):
    shape, ch, rr, wr, n_units, kind = VARIANTS[case]
    n, c, h, w, out_w, s_h, k_h = SHAPES[shape]
    read_span = rr * (w if shape != "x" else w)
    write_span = wr * out_w
    if kind == "write":
        read_span = write_span
    return _k(n, c, h * w, h * out_w, ch, read_span, write_span, n_units, kind), \
        (n, c, h, w, out_w, read_span, write_span, n_units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    kernel, meta = build(args.case)
    if args.compile_only:
        print(f"[compile-ok] {args.case} {meta}")
        return
    n, c, h, w, out_w, _, _, _ = meta
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, dtype=torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[dma] {args.case}: ran {args.iters} iterations, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
