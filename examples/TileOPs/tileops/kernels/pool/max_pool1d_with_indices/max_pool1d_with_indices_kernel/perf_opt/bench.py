"""Direct single-workload harness for `msprof op` profiling (Stage 4).

Profiles the *kernel-only* path of `_max_pool1d_with_indices_kernel` outside the
Stage 5 wrapper.  Invoked by `run_experiments.py` (and ad-hoc probes) as:

    msprof op --kernel-name=main ... python bench.py --impl {kernel.py} --case {workload_id}

The workload ids match the pytest parameter ids of the migrated benchmark
(`bench_pool.py::test_max_pool1d_indices_bench`, manifest-driven; id =
f"{label}-{dtype_str}") so that `perf_records.jsonl` / `workload_inventory.json`
use the same identifiers.

`--check` additionally compares *both* outputs (values + int64 indices) against
the PyTorch CPU golden (used by the runner for smoke precision, not by the
msprof path itself).
"""

import argparse
import importlib.util
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


# workload_id -> (n, c_in, l_in, kernel_size, stride, padding, dilation, ceil, dtype_str)
CASES = {
    # manifest smoke (label smoke-1x8x32-k3s2p1, 3 dtypes) -- precision only
    "smoke-1x8x32-k3s2p1-float16": (1, 8, 32, 3, 2, 1, 1, False, "float16"),
    "smoke-1x8x32-k3s2p1-bfloat16": (1, 8, 32, 3, 2, 1, 1, False, "bfloat16"),
    "smoke-1x8x32-k3s2p1-float32": (1, 8, 32, 3, 2, 1, 1, False, "float32"),
    # manifest full workloads (labels from tileops/manifest/pool.yaml)
    "sincnet-speaker-local-float16": (32, 80, 4096, 3, 3, 0, 1, False, "float16"),
    "textcnn-global-float16": (64, 256, 128, 128, 1, 0, 1, False, "float16"),
    "ecg-cnn-dilated-bfloat16": (16, 128, 2048, 5, 2, 2, 2, True, "bfloat16"),
}

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


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
    ap.add_argument("--block-m", type=int, default=None,
                    help="override the wrapper's default block_m knob")
    args = ap.parse_args()

    if args.case not in CASES:
        raise SystemExit(f"unknown case {args.case!r}; known: {sorted(CASES)}")
    n, c_in, l_in, k, s, p, d, ceil_mode, dtype_str = CASES[args.case]
    tdtype = _TORCH_DTYPE[dtype_str]

    mod = load_kernel_module(args.impl)
    factory = mod._max_pool1d_with_indices_kernel(
        n, c_in, l_in, k, s, p, d, ceil_mode, dtype_str)
    kernel = factory(args.block_m)

    torch.manual_seed(0)
    # Data generated on CPU then moved to NPU (device-independent, matches the
    # hierarchical test suite); the NPU transfer kernels are filtered out by
    # `--kernel-name` on the msprof side.
    x = torch.randn(n, c_in, l_in, dtype=torch.float32).to(tdtype).npu()

    with torch.no_grad():
        out = idx = None
        for _ in range(args.iters):
            out, idx = kernel(x)

    if args.check:
        ref, ref_idx = mod.golden_max_pool1d_with_indices(
            x.cpu().detach(), k, s, p, d, ceil_mode)
        got = out.cpu()
        gidx = idx.cpu()
        assert gidx.dtype == torch.int64, f"indices dtype {gidx.dtype}"
        exact = bool(torch.equal(got, ref))
        diff = (got.float() - ref.float()).abs().masked_fill(
            ~(torch.isfinite(got) & torch.isfinite(ref)), 0.0).max().item()
        nan_ok = bool((torch.isnan(got) == torch.isnan(ref)).all())
        idx_ok = bool(torch.equal(gidx, ref_idx))
        print(f"[check] {args.case}: max_diff={diff:.3e} values_exact={exact} "
              f"nan_ok={nan_ok} idx_exact={idx_ok} idx_dtype={gidx.dtype}")
        if not ((exact or (diff == 0.0 and nan_ok)) and idx_ok):
            raise SystemExit(1)
    else:
        print(f"[bench] {args.case}: ran {args.iters} iterations, "
              f"out={tuple(out.shape)} idx={tuple(idx.shape)}")


if __name__ == "__main__":
    main()
