"""Step-0 copy-floor calibration for `_max_pool2d_kernel` (Stage 4).

Pure-copy kernels on the *same grids and the same DMA granularity* as the
candidate pooling kernel: no window compute, no NaN tracking, just GM->UB
`T.copy` + UB->GM `T.copy` of the shapes each workload's unit actually moves.
Used to establish the IO floor of the operator family and to answer whether the
70% bandwidth line (achieved_bw / 1.75 TB/s) sits above the DMA burst/row-stride
floor (`CONST-dma-burst-rowstride-floor`, `CONST-copy-floor-dma-granularity`).

Variants (all NCHW, `_max_pool2d_kernel`-shaped grids):

  r112x1   resnet grid: unit = (n, oh, cblk); 3 separate (CH,112) row copies
  r112x3   same unit, one contiguous (CH, 3*112) copy (3 rows per burst chain)
  r112x8   unit = (n, oh8, cblk) covering 8 output rows: (CH, 8*112) copy
  r112x16  same with 16 output rows: (CH, 16*112) copy
  plane112 unit = (n, cblk): whole (CH, 112*112) plane, 25 KB per channel burst
  r56x1    vgg grid: unit = (n, oh, cblk); 2 separate (CH,56) row copies
  r56x2    same unit, one contiguous (CH, 2*56) copy
  r56x8    unit = (n, oh8, cblk): (CH, 8*56) copy
  r55x1    alexnet grid: unit = (n, oh, cblk); 3 separate (CH,55) row copies
  r55x3    same unit, one contiguous (CH, 3*55) copy
  w112     write-only floor: store (CH,56) per (n,oh,cblk) -- resnet output grid
  w56      write-only floor: store (CH,28) per (n,oh,cblk) -- vgg output grid
  w55      write-only floor: store (CH,27) per (n,oh,cblk) -- alexnet output grid

Each variant reports through `msprof op` only; `--iters` launches the kernel.

Run (msprof path, one variant at a time):
  msprof op --kernel-name=main ... python copy_floor.py --case r112x1 --iters 25
"""

import argparse

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils

# (n, c, h, w) of each manifest workload
SHAPES = {
    "resnet": (32, 64, 112, 112),   # out 56x56, k3s2p1
    "vgg": (16, 128, 56, 56),       # out 28x28, k2s2p0
    "alexnet": (32, 64, 55, 55),    # out 27x27, k3s3 -> k3s2p0 ceil
}

# variant -> (workload key, ch, rows-per-unit, (in_rows_per_out, rows_loaded),
#             out_w, n_rows_grid, kind)
VARIANTS = {
    "r112x1": ("resnet", 32, 1, (3, 1), 56, 56, "read"),
    "r112x3": ("resnet", 32, 1, (3, 3), 56, 56, "read"),
    "r112x8": ("resnet", 32, 8, (3, 8), 56, 7, "read"),
    "r112x16": ("resnet", 32, 16, (3, 16), 56, 4, "read"),
    "plane112": ("resnet", 32, 56, (3, 56), 56, 1, "read"),
    "r56x1": ("vgg", 64, 1, (2, 1), 28, 28, "read"),
    "r56x2": ("vgg", 64, 1, (2, 2), 28, 28, "read"),
    "r56x8": ("vgg", 64, 8, (2, 8), 28, 4, "read"),
    "r55x1": ("alexnet", 64, 1, (3, 1), 27, 27, "read"),
    "r55x3": ("alexnet", 64, 1, (3, 3), 27, 27, "read"),
    "w112": ("resnet", 32, 1, (0, 0), 56, 56, "write"),
    "w56": ("vgg", 64, 1, (0, 0), 28, 28, "write"),
    "w55": ("alexnet", 64, 1, (0, 0), 27, 27, "write"),
}

_DT = "float16"
_EB = 2
_VECTOR_CORES = None


def _cores():
    global _VECTOR_CORES
    if _VECTOR_CORES is None:
        _VECTOR_CORES = NPUUtils.get().get_aicore_num() * 2
    return _VECTOR_CORES


@tilelang.jit(out_idx=[1], target="npuir")
def _copy_kernel(n, c_in, h_in, w_in, ch, rows, in_rows, out_w, n_row_units,
                 kind):
    """Pure-copy kernel over the pooling kernel's unit grid."""
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * n_row_units * num_cblk
    num_kernels = min(num_logical, _cores())
    num_local = (num_logical + num_kernels - 1) // num_kernels
    nr = max(in_rows, 1)

    @T.prim_func
    def main(
        x: T.Tensor((n, c_in, h_in, w_in), _DT),
        out: T.Tensor((n, c_in, h_in, out_w), _DT),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            buf = T.alloc_ub((ch, nr, w_in), _DT)
            st = T.alloc_ub((ch, out_w), _DT)
            if kind == "write":
                T.clear(buf)
                T.clear(st)
            for t in T.serial(num_local):
                uid = t * num_kernels + cid
                if uid < num_logical:
                    nb = uid // (n_row_units * num_cblk)
                    rem = uid % (n_row_units * num_cblk)
                    ru = rem // num_cblk
                    cb = rem % num_cblk
                    real_c = T.min(ch, c_in - cb * ch)
                    if kind == "read":
                        if rows == 1 and in_rows > 1:
                            # one row copy per input row inside the unit
                            for r in T.serial(in_rows):
                                T.copy(
                                    x[nb, cb * ch:cb * ch + real_c,
                                      ru * rows * in_rows + r, 0:w_in],
                                    buf[0:real_c, r, 0:w_in],
                                )
                        else:
                            T.copy(
                                x[nb, cb * ch:cb * ch + real_c,
                                  ru * rows * in_rows:
                                  ru * rows * in_rows + in_rows, 0:w_in],
                                buf[0:real_c, 0:nr, 0:w_in],
                            )
                        T.copy(buf[0:real_c, 0, 0:out_w], st[0:real_c, 0:out_w])
                    T.copy(st[0:real_c, 0:out_w],
                           out[nb, cb * ch:cb * ch + real_c,
                               ru * rows:ru * rows + 1, 0:out_w])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    wkey, ch, rows, (in_per_out, in_rows), out_w, n_rows, kind = VARIANTS[args.case]
    n, c, h, w = SHAPES[wkey]
    kernel = _copy_kernel(n, c, h, w, ch, rows, in_rows, out_w, n_rows, kind)

    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[copy-floor] {args.case}: ran {args.iters} iterations, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
