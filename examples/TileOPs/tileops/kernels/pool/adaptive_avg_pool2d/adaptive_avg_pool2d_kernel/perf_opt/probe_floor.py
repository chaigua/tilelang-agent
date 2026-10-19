"""Round-3 Step 0: floor calibration for `_adaptive_avg_pool2d_kernel`.

Calibrates, with `msprof op` Task Duration (the only latency metric):

  * ``empty``  -- a `T.Kernel(NK)` doing essentially nothing: the fixed
                  launch/scheduling floor of the harness grid.
  * ``read``   -- pure GM->UB DMA of the whole input, split over NK blocks and
                  ``--rounds`` serial rounds per block, keeping one element
                  alive (writes 1 element back so nothing is dead-code
                  eliminated).  This is the "1-element keep-alive read floor"
                  of pattern-library PL-1.6.

Both are parameterised by the *same* grid/DMA granularity candidates as the real
kernel so the floor is comparable per workload.

Usage:
  msprof op --kernel-name=main ... python probe_floor.py --kind read \
      --case resnet-global-float16 --rounds 13
  python probe_floor.py --kind read --case resnet-global-float16 --rounds 1
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

# workload_id -> (n, c_in, h_in, w_in, out_h, out_w, dtype_str)
CASES = {
    "smoke-1x8x4x4-2x2-float16": (1, 8, 4, 4, 2, 2, "float16"),
    "resnet-global-float16": (8, 2048, 7, 7, 1, 1, "float16"),
    "spp-6x6-float16": (2, 128, 56, 56, 6, 6, "float16"),
    "nondiv-7x7-bfloat16": (2, 64, 55, 57, 7, 7, "bfloat16"),
}
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def build(kind: str, total_elems: int, blocks: int, rounds: int, chunk_elems: int, dtype: str):
    """kind='empty' -> do nothing; kind='read' -> rounds*chunk_elems DMA per block."""
    vcores = NPUUtils.get().get_aicore_num() * 2
    nk = min(blocks, vcores)

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_p: int):
        @T.prim_func
        def main(
            x: T.Tensor((total_elems,), dtype),
            out: T.Tensor((nk,), dtype),
        ):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                if kind == "empty":
                    out[cid] = x[cid]
                else:
                    buf = T.alloc_ub((chunk_elems,), dtype)
                    # keep one live element from each round (guards against DCE)
                    live = T.alloc_ub((1,), dtype)
                    for t in T.serial(rounds):
                        base = (t * nk + cid) * chunk_elems
                        if base + chunk_elems <= total_elems:
                            T.copy(x[base : base + chunk_elems], buf)
                            T.copy(buf[0:1], live)
                    out[cid] = live[0]

        return main

    return _func(chunk_elems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("empty", "read"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    n, c_in, h_in, w_in, out_h, out_w, dtype_str = CASES[args.case]
    tdtype = _TORCH_DTYPE[dtype_str]
    total = n * c_in * h_in * w_in

    if args.kind == "empty":
        blocks, rounds, chunk = args.blocks, 1, 1
    else:
        blocks = args.blocks
        rounds = args.rounds
        chunk = (total + blocks * rounds - 1) // (blocks * rounds)

    kernel = build(args.kind, total, blocks, rounds, chunk, dtype_str)

    torch.manual_seed(0)
    x = torch.randn(total, dtype=torch.float32).to(tdtype).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(
        f"[floor {args.kind} {args.case}] blocks={blocks} rounds={rounds} "
        f"chunk={chunk} out={tuple(out.shape)}"
    )


if __name__ == "__main__":
    main()
