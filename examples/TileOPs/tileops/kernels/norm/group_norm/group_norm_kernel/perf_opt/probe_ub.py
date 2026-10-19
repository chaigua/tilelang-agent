#!/usr/bin/env python3
"""UB-budget calibration probe (compile-only).

For a given (M, D, dtype, dtype-path) it compiles the kernel at a list of
`block_m` values with auto multi-buffer on/off and records the compiler's
`ub overflow ... requires N bits while M bits available` message, so the real
per-element UB price of *this* buffer structure can be read off (no linear
extrapolation from other tasks -- TRAP-UB-multibuffer-inflation).

Usage:
    python probe_ub.py --impl <kernel.py> --dtype float16 --D 4096 \
        --M 256 --bms 1,2,3,4,6,8 --mb 1
"""

import argparse
import importlib.util
import hashlib
import os
import re
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def load_module(path):
    path = os.path.abspath(path)
    tag = hashlib.md5(path.encode()).hexdigest()[:10]
    name = f"gn_ub_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--bms", default="1,2,3,4,6,8")
    ap.add_argument("--mb", type=int, default=1, help="1 = auto multi-buffer on")
    args = ap.parse_args()

    mod = load_module(args.impl)
    # Bypass the host UB guard: the guard's own constants are what we calibrate.
    # Disable the host budget guard: the guard's own constants are what we
    # calibrate (its threshold would mask the compiler's overflow message).
    if hasattr(mod, "_UB_PRICE_SCALE"):
        mod._UB_PRICE_SCALE = 1e-6
    for name in ("_UB_MAX_ELEMS",):
        if hasattr(mod, name):
            setattr(mod, name, 10 ** 9)
    mod._MB_OVERRIDE = (not bool(args.mb))

    for bm in [int(x) for x in args.bms.split(",") if x]:
        try:
            mod._group_norm_kernel(args.M, args.D, 1e-5, args.dtype)(bm)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).replace("\n", " ")
            m = re.search(
                r"requires\s+(\d+)\s*bits while\s+(\d+)\s*bits available", msg
            )
            if m:
                need_bits = int(m.group(1))
                avail_bits = int(m.group(2))
                elems = bm * args.D
                price = need_bits / 8.0 / elems
                print(
                    f"[ub] D={args.D} dtype={args.dtype} mb={args.mb} bm={bm}: "
                    f"OVERFLOW need={need_bits} bits ({need_bits/8:.0f} B) "
                    f"avail={avail_bits} bits -> {price:.3f} B/elem"
                )
            else:
                print(f"[ub] D={args.D} dtype={args.dtype} mb={args.mb} bm={bm}: "
                      f"ERR {msg[:300]}")
        else:
            print(f"[ub] D={args.D} dtype={args.dtype} mb={args.mb} bm={bm}: COMPILED")


if __name__ == "__main__":
    main()
