"""Standalone bench driver for the AnyFwdOp Stage 4 ``_logical_reduce_kernel``.

Loaded under ``msprof op`` (one workload per invocation).  The driver:

  1. imports the candidate kernel module from an explicit file path
     (``--impl``), so baseline / every experiment branch / the final file are
     all measured through exactly the same code path;
  2. builds the kernel through the frozen harness contract
     ``_logical_reduce_kernel(M, N, "all", dtype)(block_m)(x)`` with
     ``block_m`` taken from the module's own ``_select_config`` (the wrapper
     contract: tiling single source of truth);
  3. runs ``--launches`` kernel invocations with a synchronize each.

Kernel-only latency then comes from ``msprof op`` ``Task Duration(us)`` -- the
single latency metric of Stage 4 (no NPU events, no host timing).

Workload ids are exactly the benchmark-parameterised ids from
``pytest benchmarks/ops/bench_any.py --collect-only`` =
``workloads_to_params("AnyFwdOp", include_extra=True)`` (see
``workload_inventory.json``).

Accounting note: the Op layer pre-converts ``bool`` -> ``float32``
(``to_logical_float32``) before the kernel, so the *kernel* reads/writes
4 bytes/element even for the ``bool`` manifest rows.  Every byte figure in
this workspace uses those f32 kernel bytes.

Probe modes (``--probe``) reuse the same shapes to calibrate the structural
floor; they are NOT tune workloads and never enter ``perf_records.jsonl``:

  * ``empty``   -- grid + output store only (launch / task floor)
  * ``copyout`` -- GM -> UB copy + per-task output store
  * ``read``    -- GM -> UB copy + one f32 plane reduction (no vcmp/vcast)
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
    "smoke-any-bool": ((32, 256), torch.bool, -1),
    "smoke-any-float16": ((32, 256), torch.float16, -1),
    "smoke-any-float32": ((32, 256), torch.float32, -1),
    "mask-validation-4k-bool": ((32, 4096), torch.bool, -1),
    "mask-validation-32k-bool": ((32, 32768), torch.bool, -1),
    "3d-multidim-reduce-bool": ((4, 128, 4096), torch.bool, [0, 2]),
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
    """Boolean mask input (mirrors ``tileops.workloads.reduction`` generator).

    ``bool`` inputs are the manifest reality for every tune workload; row 0 is
    forced all-False and row 1 all-True so the verdicts are discriminative.
    """
    gen = torch.Generator().manual_seed(0)
    if dtype is torch.bool:
        x = torch.randint(0, 2, shape, generator=gen, dtype=torch.bool)
        if shape[0] > 4:
            x[0] = False
            x[1] = True
        return x.to("npu")
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
    parser.add_argument("--case", default=None)
    parser.add_argument("--launches", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--block-m", type=int, default=None,
                        help="override block_m (calibration probes only)")
    parser.add_argument("--probe-shape", default=None,
                        help="M,N for a probe (synthetic tile-shaped case)")
    parser.add_argument(
        "--probe",
        default=None,
        choices=("empty", "copyout", "read", "read_f16", "copy"),
        help="calibration probe mode (not a tune workload)",
    )
    args, _ = parser.parse_known_args()

    mod = load_module(args.impl)

    if args.probe:
        if not args.probe_shape:
            parser.error("--probe requires --probe-shape M,N")
        M, N = (int(v) for v in args.probe_shape.split(","))
        dtype_str = "float32"
        x2 = torch.randn(M, N, dtype=torch.float32, device="npu")
        out_shape = (M,)
        x = None
        kernel = mod.build_probe(args.probe, M, N, dtype_str, block_m=args.block_m)
        print("IMPL_PROBE: {} shape=({},{})".format(args.probe, M, N), flush=True)
    else:
        if not args.case:
            parser.error("--case is required unless --probe is used")
        shape, torch_dtype, dim = CASES[args.case]
        x = make_input(shape, torch_dtype)
        xf = mod._to_logical_float32(x) if torch_dtype is torch.bool else x
        x2, out_shape = mod._prepare_2d(xf, dim)
        M, N = x2.shape
        dtype_str = "float32" if x2.dtype == torch.float32 else _STR[x2.dtype]
        cfg = mod._select_config(M, N, dtype_str)
        block_m = cfg["block_m"]
        factory = mod._logical_reduce_kernel(M, N, "any", dtype_str)
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
        ref = torch.any(x.cpu(), dim=dim)
        got = y.reshape(out_shape).cpu().bool()
        ok = torch.equal(got, ref)
        print("IMPL_CHECK: {}".format("PASS" if ok else "FAIL"), flush=True)
        if not ok:
            bad = (got != ref).nonzero().flatten()
            print(
                "IMPL_CHECK_DETAIL: {}/{} entries differ; first at {} got={} want={}".format(
                    int(bad.numel()),
                    ref.numel(),
                    int(bad[0]),
                    bool(got.flatten()[bad[0]]),
                    bool(ref.flatten()[bad[0]]),
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
