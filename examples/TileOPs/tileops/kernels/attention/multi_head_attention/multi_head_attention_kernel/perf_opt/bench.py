#!/usr/bin/env python3
"""Stage 4 measurement harness for `_gqa_bwd_kernel` (kernel-only, msprof op).

Launches **only** the main backward kernel (`_gqa_bwd_main`) for one benchmark
workload, repeatedly, with the same tensors across launches (the access
pattern of the TileOPs ``ManifestBenchmark`` harness).  The wall measurement
is taken outside this script by ``msprof op`` (``Task Duration(us)``), the
single latency metric of Stage 4.

Workload set = ``tileops/manifest/attention.yaml`` ->
``MultiHeadAttentionBwdOp.workloads`` x ``{float16, bfloat16}``, expanded by
``benchmarks/ops/bench_multi_head_attention.py::test_mha_bwd_bench``.  The
label carrying ``smoke`` is precision-only (never measured here).

Usage (inside msprof):
    msprof op --kernel-name=_gqa_bwd_main --output=<dir> --launch-count=20 \
        --warm-up=5 --dump=off python bench.py --impl <kernel.py> \
        --case llama-3.1-8b-short/fp16 [--bm 64 --bn 64 --ns 2]

Precision spot check (optional, CPU golden, small/medium shapes only):
    python bench.py --impl <kernel.py> --case llama-3.1-8b-short/fp16 --check
"""

import argparse
import hashlib
import importlib.util
import os
import sys
import time

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_IMPL = os.path.abspath(os.path.join(HERE, "..", "_gqa_bwd_kernel.py"))

# manifest workloads (all causal on NPU; heads_kv == heads in the manifest)
WORKLOADS = {
    "mha-bwd-smoke-s512-h8-d64": dict(B=1, S=512, H=8, D=64),
    "llama-3.1-8b-short": dict(B=4, S=512, H=32, D=128),
    "llama-3.1-8b-long": dict(B=2, S=2048, H=32, D=128),
    "llama-3.1-70b-short": dict(B=2, S=512, H=64, D=128),
    "llama-3.1-70b-long": dict(B=1, S=2048, H=64, D=128),
}
DTYPES = ("fp16", "bf16")
_DTYPE_MAP = {
    "fp16": ("float16", torch.float16),
    "bf16": ("bfloat16", torch.bfloat16),
}


def load_module(path):
    """Import a kernel implementation file as a fresh module."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tag = hashlib.md5(path.encode()).hexdigest()[:10]
    name = f"gqa_bwd_impl_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_attr(text):
    """Parse ``name=value`` for --mod-attr (int / float / bool / str)."""
    key, _, val = text.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    low = val.lower()
    if low in ("true", "false"):
        return key, low == "true"
    try:
        return key, int(val)
    except ValueError:
        pass
    try:
        return key, float(val)
    except ValueError:
        return key, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=BASELINE_IMPL)
    ap.add_argument("--case", required=True)
    ap.add_argument("--bm", type=int, default=0, help="0 = impl default")
    ap.add_argument("--bn", type=int, default=0, help="0 = impl default")
    ap.add_argument("--ns", type=int, default=0, help="0 = impl default")
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--launch-count", type=int, default=20)
    ap.add_argument("--extra", type=int, default=5,
                    help="extra launches so msprof's own warm-up skip has slack")
    ap.add_argument("--check", action="store_true",
                    help="tier-1 golden compare on this workload (CPU reference)")
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--mod-attr", action="append", default=[],
                    type=_parse_attr, metavar="NAME=VALUE",
                    help="override a module-level knob of the impl (repeatable)")
    args = ap.parse_args()

    label, _, dtype_key = args.case.partition("/")
    if label not in WORKLOADS:
        raise SystemExit(f"unknown workload label {label!r}")
    if dtype_key not in DTYPES:
        raise SystemExit(f"unknown dtype {dtype_key!r}")
    spec = WORKLOADS[label]
    batch, seq_len, heads, dim = spec["B"], spec["S"], spec["H"], spec["D"]
    dtype_str, torch_dtype = _DTYPE_MAP[dtype_key]

    mod = load_module(args.impl)
    for key, val in args.mod_attr:
        if not hasattr(mod, key):
            raise SystemExit(f"impl has no module attribute {key!r}")
        setattr(mod, key, val)

    cfg = mod._default_config(dim)
    if args.bm:
        cfg["block_m"] = args.bm
    if args.bn:
        cfg["block_n"] = args.bn
    if args.ns:
        cfg["num_stages"] = args.ns

    factory = mod._gqa_bwd_kernel(
        batch, heads, heads, seq_len, dim, True, dtype_str
    )
    kernel = factory(**cfg)
    if args.print_config:
        print(f"[bench] case={args.case} impl={os.path.basename(args.impl)} "
              f"cfg={cfg}")

    gen = torch.Generator().manual_seed(0)
    shape = (batch, seq_len, heads, dim)
    q = torch.randn(*shape, generator=gen, dtype=torch.float32).to(torch_dtype).npu()
    k = torch.randn(*shape, generator=gen, dtype=torch.float32).to(torch_dtype).npu()
    v = torch.randn(*shape, generator=gen, dtype=torch.float32).to(torch_dtype).npu()
    do = torch.randn(*shape, generator=gen, dtype=torch.float32).to(torch_dtype).npu()
    # Synthetic forward statistics: the kernel only reads them and the values do
    # not change the instruction mix (base-2 lse, rowsum(o*do) domain).
    lse = (torch.mul(torch.randn(batch, heads, seq_len, generator=gen), 2.0) + 5.0)
    delta = torch.mul(torch.randn(batch, heads, seq_len, generator=gen), 3.0)
    lse_npu = lse.contiguous().npu()
    delta_npu = delta.contiguous().npu()

    dq = torch.zeros(*shape, dtype=torch.float32, device="npu")
    dk = torch.zeros(*shape, dtype=torch.float32, device="npu")
    dv = torch.zeros(*shape, dtype=torch.float32, device="npu")

    if args.check:
        # Validated reference path: the module's own golden produces the base-2
        # lse / delta pair from q,k,v,do (the perf path above uses synthetic
        # values, which are only used for timing).
        lse_cpu, delta_cpu = mod.golden_gqa_lse_delta(
            q.cpu(), k.cpu(), v.cpu(), do.cpu(), True
        )
        refs = mod.golden_gqa_bwd(
            q.cpu(), k.cpu(), v.cpu(), do.cpu(), lse_cpu, delta_cpu, True
        )
        lse_npu = lse_cpu.contiguous().npu()
        delta_npu = delta_cpu.contiguous().npu()
        dq.zero_()
        dk.zero_()
        dv.zero_()
        kernel(q, k, v, do, lse_npu, delta_npu, dq, dk, dv)
        torch.npu.synchronize()
        atol, rtol = mod._TOLERANCE[dtype_str][1], mod._TOLERANCE[dtype_str][2]
        tier2 = mod._TWO_ULP_REL[dtype_str]
        ok = True
        for name, out, ref in zip(("dq", "dk", "dv"), (dq, dk, dv), refs):
            max_diff, v1, v2 = mod._compare(out.cpu(), ref, atol, rtol, tier2)
            print(f"[check] {args.case} cfg={cfg} {name} max_diff={max_diff:.3e} "
                  f"tier1_viol={v1} tier2_viol={v2}")
            ok = ok and v2 == 0
        print(f"[check] {args.case} {'PASS' if ok else 'FAIL'}")

    total = args.warm_up + args.launch_count + args.extra
    t0 = time.time()
    for _ in range(total):
        kernel(q, k, v, do, lse_npu, delta_npu, dq, dk, dv)
    torch.npu.synchronize()
    print(f"[bench] launched {total}x in {time.time() - t0:.2f}s wall")


if __name__ == "__main__":
    main()
