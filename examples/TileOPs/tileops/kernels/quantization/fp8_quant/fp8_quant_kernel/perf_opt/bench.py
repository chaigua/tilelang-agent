"""Direct single-workload harness for msprof op profiling (Stage 4).

Profiles the *kernel-only* path of `_fp8_quant_kernel` outside the Stage 5
wrapper.  Invoked by `run_experiments.py` (and ad-hoc probes) as:

    msprof op --kernel-name=_fp8_quant_fwd_main ... \
        python bench.py --impl {kernel.py} --case {workload_id}

The workload ids match the pytest parameter ids of the migrated benchmark
(`bench_fp8_quant.py::test_fp8_quant_bench`, manifest-driven) so that
`perf_records.jsonl` / `workload_inventory.json` use the same identifiers.

`--check` additionally compares against the module's own torch CPU golden
(used by probes / the elementwise exactness gate, not by the msprof path).
"""

import argparse
import importlib.util
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


# workload_id -> (batch, seq_len_kv, kv_group, index_dim, dtype_str)
CASES = {
    "smoke-kv-index-256-d64-fp16": (1, 256, 1, 64, "float16"),
    "kv-index-8k-d64-fp16": (1, 8192, 1, 64, "float16"),
    "kv-index-8k-d64-bf16": (1, 8192, 1, 64, "bfloat16"),
    "kv-index-4k-d128-fp32": (1, 4096, 1, 128, "float32"),
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
    ap.add_argument("--num-stages", type=int, default=0)
    ap.add_argument("--block-m", type=int, default=None, help="override block_m (else factory default)")
    ap.add_argument("--check", action="store_true", help="compare vs torch golden")
    args = ap.parse_args()

    if args.case not in CASES:
        raise SystemExit(f"unknown case {args.case!r}; known: {sorted(CASES)}")
    batch, seq_len_kv, kv_group, index_dim, dtype_str = CASES[args.case]
    tdtype = _TORCH_DTYPE[dtype_str]

    mod = load_kernel_module(args.impl)
    factory = mod._fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, dtype_str)
    cfg = {"num_stages": args.num_stages, "block_m": args.block_m if args.block_m else 32}
    try:
        kernel = factory(args.num_stages, cfg["block_m"])
    except TypeError:
        kernel = factory(args.num_stages, cfg["block_m"])

    torch.manual_seed(20261014)
    rows = batch * seq_len_kv * kv_group
    x = torch.randn(batch, seq_len_kv, kv_group, index_dim, dtype=torch.float32).to(tdtype)
    x_dev = x.reshape(rows, index_dim).contiguous().npu()
    scale_dev = torch.empty(rows, dtype=torch.float32, device="npu")
    out_dev = torch.empty(rows, index_dim, dtype=torch.uint8, device="npu")

    with torch.no_grad():
        for _ in range(args.iters):
            kernel(x_dev, scale_dev, out_dev)

    if args.check:
        scale_g, out_g = mod.golden_fp8_quant(x)
        ref_u8 = out_g.view(torch.uint8).reshape(rows, index_dim)
        scale_k = scale_dev.cpu().reshape(batch, seq_len_kv, kv_group)
        finite = torch.isfinite(scale_g)
        torch.testing.assert_close(scale_k[finite], scale_g[finite], rtol=1e-4, atol=1e-4)
        rep = mod._classify_mismatches(x, out_dev, ref_u8)
        cos = mod._cosine_similarity(out_dev, ref_u8)
        print(f"[check] {args.case}: cos={cos:.6f} mismatch={rep['mismatch']} "
              f"zero_ok={rep['zero_ok']} tie={rep['tie_1lsb']} nan={rep['nan_lane']} "
              f"unexpected={rep['unexpected']}")
        if cos < 0.99 or rep["unexpected"] != 0:
            raise SystemExit(1)
    else:
        print(f"[bench] {args.case}: ran {args.iters} iterations, block_m={cfg['block_m']}")


if __name__ == "__main__":
    main()
