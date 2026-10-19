"""Standalone bench driver for the Stage 4 perf_opt kernel.

Loaded under ``msprof op`` (one workload per invocation).  The driver:

  1. imports the candidate kernel module from an explicit file path
     (``--impl``), so baseline / every experiment branch / the final file are
     all measured through exactly the same code path;
  2. builds the kernel through the frozen harness contract
     ``_argreduce_kernel(M, N, "argmin", dtype)(block_m)(x)`` with
     ``block_m`` taken from the module's own ``_select_config`` (the wrapper
     contract);
  3. runs ``--launches`` kernel invocations with a synchronize each.

Kernel-only latency then comes from ``msprof op`` ``Task Duration(us)`` --
the single latency metric of Stage 4 (no NPU events, no host timing).

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


# workload_id -> (M, N, torch dtype)
CASES = {
    "smoke-argmin-float16": (32, 256, torch.float16),
    "smoke-argmin-bfloat16": (32, 256, torch.bfloat16),
    "smoke-argmin-float32": (32, 256, torch.float32),
    "lm-head-argmin-float16": (4, 102400, torch.float16),
    "lm-head-argmin-bfloat16": (4, 102400, torch.bfloat16),
    "hidden-state-argmin-float16": (2048, 4096, torch.float16),
    "hidden-state-argmin-bfloat16": (2048, 4096, torch.bfloat16),
    "non-aligned-argmin-float16": (256, 300, torch.float16),
    "non-aligned-argmin-bfloat16": (256, 300, torch.bfloat16),
}

_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def load_module(path):
    path = os.path.abspath(path)
    name = "perf_opt_candidate_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", required=True)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--launches", type=int, default=3)
    parser.add_argument("--check", action="store_true", help="verify torch.equal vs CPU golden")
    args, _ = parser.parse_known_args()

    M, N, torch_dtype = CASES[args.case]
    dtype_str = _STR[torch_dtype]
    mod = load_module(args.impl)

    cfg = mod._select_config(M, N, dtype_str)
    block_m = cfg["block_m"]
    factory = mod._argreduce_kernel(M, N, "argmin", dtype_str)
    kernel = factory(block_m)

    # prim_func names visible on the compiled kernel (for the msprof filter)
    prim_names = []
    try:
        pf = getattr(kernel, "prim_func", None)
        if pf is not None and getattr(pf, "name", None):
            prim_names.append(str(pf.name))
    except Exception:
        pass
    for attr in ("msprof_kernel_name", "kernel_name"):
        v = getattr(kernel, attr, None)
        if isinstance(v, str) and v and v not in prim_names:
            prim_names.append(v)
    print("IMPL_CFG: {}".format(cfg), flush=True)
    print("IMPL_PRIM_FUNCS: {}".format(",".join(prim_names)), flush=True)

    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    # The Op layer pads with +inf to align_up(N, 256); replicate it here.
    from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if N_padded != N:
        xp = torch.full((M, N_padded), float("inf"), dtype=torch_dtype, device="npu")
        xp[:, 0:N] = x
    else:
        xp = x

    # Warm-up (compilation already happened at factory/first-call time).
    for _ in range(3):
        y = kernel(xp)
    torch.npu.synchronize()

    for _ in range(args.launches):
        y = kernel(xp)
        torch.npu.synchronize()

    if args.check:
        ref = torch.argmin(x, dim=-1)
        ok = torch.equal(y.cpu(), ref)
        print("IMPL_CHECK: {}".format("PASS" if ok else "FAIL"), flush=True)
        if not ok:
            bad = (y.cpu() != ref).nonzero().flatten()
            print(
                "IMPL_CHECK_DETAIL: {}/{} rows differ; first row {} got={} want={}".format(
                    int(bad.numel()), M, int(bad[0]),
                    int(y.cpu()[bad[0]]), int(ref[bad[0]]),
                ),
                flush=True,
            )
            raise SystemExit(1)

    print(
        "IMPL_OK case={} shape=({},{}) dtype={} block_m={} out_dtype={} out0={}".format(
            args.case, M, N, dtype_str, block_m, y.dtype, int(y[0])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
