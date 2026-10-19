#!/usr/bin/env python3
"""Stage 4 measurement harness for GroupNormFwdOp (kernel-only, msprof op).

Runs the *kernel* (not the op layer) for one benchmark workload, repeatedly,
with the **same** input tensors across launches (`sets=1`) -- the access
pattern of the TileOPs `ManifestBenchmark` harness (`tileops/benchmark/msprof.py`),
so the numbers here are directly comparable with the integration baseline in
`examples/TileOPs/reports/tileops/20261016_003328_665600_GroupNormFwdOp/run.json`.

This script only *launches*; the wall measurement is `msprof op`'s
`Task Duration(us)` (single latency metric for Stage 4).

Usage:
    msprof op --kernel-name=main --output=<dir> --launch-count=20 --warm-up=5 \
        --dump=off --aic-metrics=... \
        python bench.py --impl <kernel.py> --case image-g32/fp16 --bm 2

    # precision gate for parameter-only branches (exact golden compare)
    python bench.py --impl <kernel.py> --case image-g32/fp16 --bm 2 --check
"""

import argparse
import hashlib
import importlib.util
import math
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_IMPL = os.path.abspath(
    os.path.join(HERE, "..", "_group_norm_kernel.py")
)

# Manifest `GroupNormFwdOp` workloads (benchmarks/ops/bench_group_norm.py
# expansion of load_workloads("GroupNormFwdOp") x dtypes).
CASES = {
    # label / dtype
    "smoke-g8/fp32": dict(N=2, C=32, spatial=(8, 8), g=8, dtype="float32"),
    "smoke-g8/fp16": dict(N=2, C=32, spatial=(8, 8), g=8, dtype="float16"),
    "smoke-g8/bf16": dict(N=2, C=32, spatial=(8, 8), g=8, dtype="bfloat16"),
    "image-g32/fp16": dict(N=8, C=128, spatial=(32, 32), g=32, dtype="float16"),
    "image-g32/bf16": dict(N=8, C=128, spatial=(32, 32), g=32, dtype="bfloat16"),
    "wider-channel-g32/fp16": dict(
        N=4, C=256, spatial=(28, 28), g=32, dtype="float16"
    ),
    "tail-spatial-g16/fp16": dict(
        N=4, C=128, spatial=(30, 30), g=16, dtype="float16"
    ),
}

_TOL = {
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
    "float32": (1e-5, 1e-5),
}


def spec_of(case):
    c = CASES[case]
    n, ch, sp, g = c["N"], c["C"], c["spatial"], c["g"]
    spatial_size = int(math.prod(sp)) if sp else 1
    cpg = ch // g
    d = cpg * spatial_size
    m = n * g
    return m, d, g, c["dtype"]


def load_module(path):
    """Import a kernel implementation file as a fresh module."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tag = hashlib.md5(path.encode()).hexdigest()[:10]
    name = f"gn_impl_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=BASELINE_IMPL)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--bm", type=int, default=0,
                    help="block_m; 0 = the implementation's own default")
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--extra", type=int, default=5,
                    help="extra launches so msprof's own warm-up skip has slack")
    ap.add_argument("--check", action="store_true",
                    help="exact golden compare (parameter-branch precision gate)")
    ap.add_argument("--print-bm", action="store_true")
    ap.add_argument("--mb", type=int, default=-1,
                    help="-1 = implementation policy; 0/1 = force auto "
                         "multi-buffer off/on (Stage 4 A/B knob)")
    ap.add_argument("--nkc", type=int, default=0,
                    help="0 = implementation policy; else cap num_kernels "
                         "(grid-balance knob)")
    args = ap.parse_args()

    m, d, g, dtype_str = spec_of(args.case)
    torch_dtype = getattr(torch, dtype_str)
    mod = load_module(args.impl)
    if args.mb >= 0 and hasattr(mod, "_MB_OVERRIDE"):
        mod._MB_OVERRIDE = bool(args.mb)
    if args.nkc > 0 and hasattr(mod, "_NUM_KERNELS_CAP"):
        mod._NUM_KERNELS_CAP = args.nkc
    factory = mod._group_norm_kernel

    bm = args.bm
    if bm <= 0:
        try:
            bm = mod._default_block_m(d, m, dtype_str)
        except TypeError:
            bm = mod._default_block_m(d)
    if args.print_bm:
        print(f"[bench] case={args.case} M={m} D={d} dtype={dtype_str} bm={bm}")

    torch.manual_seed(0)
    x = torch.randn(m, d, dtype=torch.float32).to(torch_dtype).npu()
    kw = torch.ones(d, dtype=torch_dtype).npu()
    kb = torch.zeros(d, dtype=torch_dtype).npu()

    kernel = factory(m, d, 1e-5, dtype_str)(bm)

    if args.check:
        import torch.nn.functional as F

        y = kernel(x, kw, kb)
        ref = F.group_norm(
            x.reshape(args_case_shape(args.case)).float(),
            g, weight=None, bias=None, eps=1e-5,
        ).to(torch_dtype)
        y_view = y.reshape(args_case_shape(args.case))
        atol, rtol = _TOL[dtype_str]
        diff = (y_view.float() - ref.float()).abs()
        viol = int((diff > (atol + rtol * ref.float().abs())).sum().item())
        max_diff = float(diff.max().item())
        status = "PASS" if viol == 0 else "FAIL"
        print(f"[check] {args.case} bm={bm} {status} max_diff={max_diff:.3e} "
              f"violations={viol}/{ref.numel()}")
        if viol:
            raise SystemExit(1)
        del ref, y_view, diff

    total = args.warm_up + args.launch_count + args.extra
    for _ in range(total):
        kernel(x, kw, kb)


def args_case_shape(case):
    c = CASES[case]
    return (c["N"], c["C"]) + tuple(c["spatial"])


if __name__ == "__main__":
    main()
