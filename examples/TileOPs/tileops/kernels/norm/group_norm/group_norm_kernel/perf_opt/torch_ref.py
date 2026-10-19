#!/usr/bin/env python3
"""Same-shape vendor reference: ``torch.nn.functional.group_norm`` on NPU.

Reference anchoring (T-1) for the GroupNorm Stage 4 ``[DESIGN_LIMIT]``
judgement: this runs the *vendor fused* GroupNorm (aclnn kernel, per-channel
affine included) on exactly the four tune workloads, so the tuned TileLang
kernel can be compared against a same-shape, same-semantics implementation
that is not part of the frozen design family.

Each workload is launched ``--launch-count + --warm-up + extra`` times with
``sets=1`` tensors (the benchmark harness access pattern).  The latency is
read from ``msprof op`` ``Task Duration(us)`` (never from wall time).

Usage:
    msprof op --output=<dir> --launch-count=20 --warm-up=5 --dump=off \
        python torch_ref.py --case image-g32/fp16
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = {
    "image-g32/fp16": (8, 128, (32, 32), 32, torch.float16),
    "image-g32/bf16": (8, 128, (32, 32), 32, torch.bfloat16),
    "wider-channel-g32/fp16": (4, 256, (28, 28), 32, torch.float16),
    "tail-spatial-g16/fp16": (4, 128, (30, 30), 16, torch.float16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--extra", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    n, c, spatial, g, dtype = CASES[args.case]
    torch.manual_seed(0)
    x = torch.randn((n, c) + spatial, dtype=dtype).npu()
    w = torch.randn(c, dtype=dtype).npu()
    b = torch.randn(c, dtype=dtype).npu()

    if args.check:
        y = F.group_norm(x, g, weight=w, bias=b, eps=1e-5)
        ref = F.group_norm(x.float().cpu(), g, weight=w.float().cpu(),
                           bias=b.float().cpu(), eps=1e-5)
        print(f"[torch-ref] {args.case} max_diff="
              f"{(y.float().cpu() - ref).abs().max().item():.3e}")
        del y, ref

    F.group_norm(x, g, weight=w, bias=b, eps=1e-5)  # compile/warm
    for _ in range(args.warm_up + args.launch_count + args.extra):
        F.group_norm(x, g, weight=w, bias=b, eps=1e-5)


if __name__ == "__main__":
    main()
