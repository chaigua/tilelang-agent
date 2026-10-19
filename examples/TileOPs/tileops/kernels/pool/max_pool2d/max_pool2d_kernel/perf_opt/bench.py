"""Direct single-workload harness for msprof op profiling (Stage 4).

Profiles the *kernel-only* path of `_max_pool2d_kernel` outside the Stage 5
wrapper.  Invoked by `run_experiments.py` (and ad-hoc probes) as:

    msprof op --kernel-name=main ... python bench.py --impl {kernel.py} --case {workload_id}

The workload ids match the pytest parameter ids of the migrated benchmark
(`benchmarks/ops/bench_pool.py::test_max_pool2d_bench`, manifest-driven; id =
f"{label}-{dtype_str}") so that `perf_records.jsonl` / `workload_inventory.json`
use the same identifiers.

`--check` additionally compares against the PyTorch CPU golden (used by probes,
not by the msprof path itself).
"""

import argparse
import importlib.util
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


# workload_id -> (n, c, h, w, k, s, p, d, ceil, dtype_str)
# k/s/p/d are (H, W) pairs, mirroring the manifest entries.
CASES = {
    # manifest smoke (label smoke-1x8x16x16-k3s2p1, 3 dtypes) -- precision only
    "smoke-1x8x16x16-k3s2p1-float16": (1, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False, "float16"),
    "smoke-1x8x16x16-k3s2p1-bfloat16": (1, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False, "bfloat16"),
    "smoke-1x8x16x16-k3s2p1-float32": (1, 8, 16, 16, (3, 3), (2, 2), (1, 1), (1, 1), False, "float32"),
    # manifest full workloads (labels from tileops/manifest/pool.yaml)
    "resnet-stem-float16": (32, 64, 112, 112, (3, 3), (2, 2), (1, 1), (1, 1), False, "float16"),
    "resnet-stem-bfloat16": (32, 64, 112, 112, (3, 3), (2, 2), (1, 1), (1, 1), False, "bfloat16"),
    "resnet-stem-float32": (32, 64, 112, 112, (3, 3), (2, 2), (1, 1), (1, 1), False, "float32"),
    "vgg-block-float16": (16, 128, 56, 56, (2, 2), (2, 2), (0, 0), (1, 1), False, "float16"),
    "vgg-block-bfloat16": (16, 128, 56, 56, (2, 2), (2, 2), (0, 0), (1, 1), False, "bfloat16"),
    "vgg-block-float32": (16, 128, 56, 56, (2, 2), (2, 2), (0, 0), (1, 1), False, "float32"),
    "alexnet-ceil-float16": (32, 64, 55, 55, (3, 3), (2, 2), (0, 0), (1, 1), True, "float16"),
    "alexnet-ceil-bfloat16": (32, 64, 55, 55, (3, 3), (2, 2), (0, 0), (1, 1), True, "bfloat16"),
    "alexnet-ceil-float32": (32, 64, 55, 55, (3, 3), (2, 2), (0, 0), (1, 1), True, "float32"),
}

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_kernel_module(path):
    path = os.path.abspath(path)
    name = "_bench_kernel_mod_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(mod, case):
    n, c, h, w, k, s, p, d, ceil_mode, dtype_str = case
    return mod._max_pool2d_kernel(
        n, c, h, w, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1],
        ceil_mode, dtype_str,
    )(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, help="path to the kernel .py file")
    ap.add_argument("--case", required=True, help="workload id (see CASES)")
    ap.add_argument("--iters", type=int, default=20, help="kernel launches")
    ap.add_argument("--check", action="store_true", help="compare vs torch golden")
    args = ap.parse_args()

    if args.case not in CASES:
        raise SystemExit(f"unknown case {args.case!r}; known: {sorted(CASES)}")
    case = CASES[args.case]
    n, c, h, w = case[0], case[1], case[2], case[3]
    k, s, p, d, ceil_mode, dtype_str = case[4:]
    tdtype = _TORCH_DTYPE[dtype_str]

    mod = load_kernel_module(args.impl)
    kernel = build(mod, case)

    torch.manual_seed(0)
    # Data generated on CPU then moved to NPU (device-independent, matches the
    # hierarchical test suite); the NPU transfer kernels are filtered out by
    # `--kernel-name` on the msprof side.
    x = torch.randn(n, c, h, w, dtype=torch.float32).to(tdtype).npu()

    with torch.no_grad():
        out = None
        for _ in range(args.iters):
            out = kernel(x)

    if args.check:
        ref = mod.golden_max_pool2d(x.cpu(), k, s, p, d, ceil_mode)
        got = out.cpu()
        exact = bool(torch.equal(got, ref))
        finite = torch.isfinite(got) & torch.isfinite(ref)
        diff = (got.float() - ref.float()).abs().masked_fill(~finite, 0.0).max().item()
        nan_ok = bool((torch.isnan(got) == torch.isnan(ref)).all())
        print(f"[check] {args.case}: max_diff={diff:.3e} exact={exact} nan_ok={nan_ok}")
        if not (exact or (diff == 0.0 and nan_ok)):
            raise SystemExit(1)
    else:
        print(f"[bench] {args.case}: ran {args.iters} iterations, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
