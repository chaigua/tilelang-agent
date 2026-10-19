"""Direct single-workload harness for msprof op profiling (Stage 4).

Profiles the *kernel-only* path of `_adaptive_avg_pool2d_kernel` outside the
Stage 5 wrapper.  Invoked by `run_experiments.py` (and ad-hoc probes) as:

    msprof op --kernel-name=main ... python bench.py --impl {kernel.py} --case {workload_id}

The workload ids match the pytest parameter ids of the migrated benchmark
(`bench_pool.py::test_adaptive_avg_pool2d_bench`, manifest-driven) so that
`perf_records.jsonl` / `workload_inventory.json` use the same identifiers.

`--check` additionally compares against the PyTorch golden (used by probes,
not by the msprof path itself).
"""

import argparse
import importlib.util
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


# workload_id -> (n, c_in, h_in, w_in, out_h, out_w, dtype_str)
CASES = {
    # manifest smoke (label smoke-1x8x4x4-2x2)
    "smoke-1x8x4x4-2x2-float16": (1, 8, 4, 4, 2, 2, "float16"),
    # manifest full workloads
    "resnet-global-float16": (8, 2048, 7, 7, 1, 1, "float16"),
    "spp-6x6-float16": (2, 128, 56, 56, 6, 6, "float16"),
    "nondiv-7x7-bfloat16": (2, 64, 55, 57, 7, 7, "bfloat16"),
}

_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def load_kernel_module(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("_bench_kernel_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, help="path to the kernel .py file")
    ap.add_argument("--case", required=True, help="workload id (see CASES)")
    ap.add_argument("--iters", type=int, default=20, help="kernel launches")
    ap.add_argument("--check", action="store_true", help="compare vs torch golden")
    args = ap.parse_args()

    if args.case not in CASES:
        raise SystemExit(f"unknown case {args.case!r}; known: {sorted(CASES)}")
    n, c_in, h_in, w_in, out_h, out_w, dtype_str = CASES[args.case]
    tdtype = _TORCH_DTYPE[dtype_str]

    mod = load_kernel_module(args.impl)
    factory = mod._adaptive_avg_pool2d_kernel(
        n, c_in, h_in, w_in, out_h, out_w, dtype_str
    )
    kernel = factory(None)

    torch.manual_seed(0)
    # Data generated on CPU then moved to NPU (device-independent, matches the
    # hierarchical test suite); the NPU transfer kernels are filtered out by
    # `--kernel-name` on the msprof side.
    x = torch.randn(n, c_in, h_in, w_in, dtype=torch.float32).to(tdtype).npu()

    with torch.no_grad():
        out = None
        for _ in range(args.iters):
            out = kernel(x)

    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().detach().float(), (out_h, out_w)).to(tdtype)
        got = out.cpu()
        ok = torch.allclose(got.float(), ref.float(), atol=1e-2, rtol=1e-2)
        diff = (got.float() - ref.float()).abs().max().item()
        print(f"[check] {args.case}: max_diff={diff:.3e} allclose={ok}")
        if not ok:
            raise SystemExit(1)
    else:
        print(f"[bench] {args.case}: ran {args.iters} iterations, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
