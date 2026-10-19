"""Standalone bench driver for the Stage 4 ``_logical_reduce_kernel`` workspace.

Loaded under ``msprof op`` (one workload per invocation).  The driver:

  1. imports the candidate kernel module from an explicit file path
     (``--impl``), so baseline / every experiment branch / the final file are
     all measured through exactly the same code path;
  2. builds the kernel through the frozen harness contract
     ``_logical_reduce_kernel(M, N, "count_nonzero", dtype)(block_m)(x)`` with
     ``block_m`` taken from the module's own ``_select_config`` (the wrapper
     contract: tiling single source of truth);
  3. runs ``--launches`` kernel invocations with a synchronize each.

Kernel-only latency then comes from ``msprof op`` ``Task Duration(us)`` --
the single latency metric of Stage 4 (no NPU events, no host timing).

Workload ids are exactly the benchmark-parameterised ids from
``tileops.benchmark.benchmark_base.workloads_to_params('CountNonzeroFwdOp',
include_extra=True)`` (see ``workload_inventory.json``).

Probe modes (``--probe``) reuse the same shapes to calibrate the structural
floor; they are NOT tune worklaods and never enter ``perf_records.jsonl``:

  * ``empty``   -- grid + output store only (launch / task floor)
  * ``read``    -- GM -> UB copy + ``reduce_sum`` on x itself (no vcmp/vcast)
  * ``read2``   -- same bytes read twice (two copies, second reduce) to probe
                   the read-bandwidth scaling
  * ``copy``    -- GM -> UB copy + UB -> GM copy of the same tile (1:1 R/W)

Print the compiled prim_func names to stdout (``IMPL_PRIM_FUNCS: ...``) so the
runner can pick the right ``--kernel-name`` filter.
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the "npu" device)

_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

# workload_id -> (shape, torch dtype, dim)  (benchmark parameterisation)
CASES = {
    "smoke-count-nonzero-float16": ((32, 256), torch.float16, -1),
    "smoke-count-nonzero-bfloat16": ((32, 256), torch.bfloat16, -1),
    "smoke-count-nonzero-float32": ((32, 256), torch.float32, -1),
    "sparsity-hidden-float16": ((2048, 4096), torch.float16, -1),
    "sparsity-hidden-bfloat16": ((2048, 4096), torch.bfloat16, -1),
    "sparsity-seq-float16": ((32, 16384), torch.float16, -1),
    "3d-multidim-reduce-float16": ((4, 128, 4096), torch.float16, [0, 2]),
}


def load_module(path):
    path = os.path.abspath(path)
    name = "perf_opt_candidate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_input(shape, dtype):
    """~30% zeros mixed-density input (same generator as the kernel L0 suite)."""
    gen = torch.Generator().manual_seed(0)
    numel = 1
    for s in shape:
        numel *= s
    x = torch.randn(numel, generator=gen)
    keep = (torch.rand(numel, generator=gen) > 0.3).to(torch.float32)
    x = (x * keep).reshape(shape)
    return x.to(dtype).to("npu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", required=True)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--launches", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--block-m", type=int, default=None,
                        help="override block_m (calibration probes only)")
    parser.add_argument(
        "--probe",
        default=None,
        choices=("empty", "copyout", "read", "read2", "copy"),
        help="calibration probe mode (not a tune workload)",
    )
    args, _ = parser.parse_known_args()

    shape, torch_dtype, dim = CASES[args.case]
    dtype_str = _STR[torch_dtype]
    mod = load_module(args.impl)

    x = make_input(shape, torch_dtype)
    x2, out_shape = mod._prepare_2d(x, dim)
    M, N = x2.shape

    if args.probe:
        kernel = mod.build_probe(args.probe, M, N, dtype_str, block_m=args.block_m)
        print("IMPL_PROBE: {}".format(args.probe), flush=True)
    else:
        cfg = mod._select_config(M, N, dtype_str)
        block_m = cfg["block_m"]
        factory = mod._logical_reduce_kernel(M, N, "count_nonzero", dtype_str)
        kernel = factory(block_m)
        print("IMPL_CFG: {}".format(cfg), flush=True)

    prim_names = []
    try:
        pf = getattr(kernel, "prim_func", None)
        if pf is not None and getattr(pf, "name", None):
            prim_names.append(str(pf.name))
    except Exception:  # noqa: BLE001
        pass
    for attr in ("msprof_kernel_name", "kernel_name"):
        v = getattr(kernel, attr, None)
        if isinstance(v, str) and v and v not in prim_names:
            prim_names.append(v)
    print("IMPL_PRIM_FUNCS: {}".format(",".join(prim_names)), flush=True)

    for _ in range(3):
        y = kernel(x2)
    torch.npu.synchronize()

    for _ in range(args.launches):
        y = kernel(x2)
        torch.npu.synchronize()

    if args.check and not args.probe:
        ref = torch.count_nonzero(x.cpu(), dim=dim).to(torch.int64)
        got = y.reshape(out_shape).cpu()
        ok = torch.equal(got, ref)
        print("IMPL_CHECK: {}".format("PASS" if ok else "FAIL"), flush=True)
        if not ok:
            bad = (got != ref).nonzero().flatten()
            print(
                "IMPL_CHECK_DETAIL: {}/{} rows differ; first at {} got={} want={}".format(
                    int(bad.numel()),
                    ref.numel(),
                    int(bad[0]),
                    int(got.flatten()[bad[0]]),
                    int(ref.flatten()[bad[0]]),
                ),
                flush=True,
            )
            raise SystemExit(1)

    print(
        "IMPL_OK case={} shape=({},{}) dtype={} out_dtype={} out0={}".format(
            args.case, M, N, dtype_str, y.dtype, int(y.reshape(-1)[0])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
