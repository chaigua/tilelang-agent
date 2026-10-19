"""Bench driver for the C4 mte2-isolation probes (msprof-able)."""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from probe_ub2 import VARIANTS, N, C, H, W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    kernel = VARIANTS[args.variant](args.ch)
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[c4bench] {args.variant} ch={args.ch}: {args.iters} iters, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
