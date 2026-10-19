#!/usr/bin/env python3
"""Compile-only probe: does a given impl compile one (shape, config) case?

Used to bisect the L2 UB overflow of the causal dim=128 *tail* shape
(2,300,4,4,128) across the Stage 4 candidates.

Usage: python probe_compile.py --impl <kernel.py> [--case tail-causal]
"""

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    # label: (B, S, H, Hkv, D, causal, dtype, bm, bn, ns)
    "tail-causal": (2, 300, 4, 4, 128, True, "float16", 64, 32, 1),
    "tail-noncausal": (1, 33, 3, 3, 128, False, "float16", 64, 32, 1),
}


def load_module(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(f"impl_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--case", default="tail-causal", choices=sorted(CASES))
    ap.add_argument("--bn", type=int, default=0)
    args = ap.parse_args()
    b, s, h, hkv, d, causal, dt, bm, bn, ns = CASES[args.case]
    if args.bn:
        bn = args.bn
    mod = load_module(args.impl)
    if hasattr(mod, "_USE_TUNED_DEFAULT"):
        mod._USE_TUNED_DEFAULT = False
    try:
        mod._gqa_bwd_kernel(b, h, hkv, s, d, causal, dt)(
            block_m=bm, block_n=bn, num_stages=ns
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        keep = "ub overflow" if "ub overflow" in msg else msg[:120]
        print(f"[compile] {os.path.basename(args.impl)} {args.case} bn={bn}: FAIL ({keep})")
        return 1
    print(f"[compile] {os.path.basename(args.impl)} {args.case} bn={bn}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
