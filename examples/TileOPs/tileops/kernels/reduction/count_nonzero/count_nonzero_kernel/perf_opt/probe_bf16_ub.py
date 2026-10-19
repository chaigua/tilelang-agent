"""Round-4 UB-price probe (bf16 vs fp16 resident path, 2-D row reduce).

Not a tune workload: this only compiles the resident builder at an explicit
``block_m`` (bypassing the ``_select_config`` ladder / ``_resolve_block_m``
clamp) to read the compiler-reported UB requirement, i.e. B/element per dtype.

CLI: python probe_bf16_ub.py <impl> <dtype> <block_m> [N]
Prints PASS / UB_OVERFLOW <required bits> <B/elem>.
"""

import importlib.util
import os
import re
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")


def load_module(path):
    path = os.path.abspath(path)
    name = "ub_probe_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    impl, dtype, bm = sys.argv[1], sys.argv[2], int(sys.argv[3])
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    M = 2048
    mod = load_module(impl)
    builder = mod._build_resident(M, N, dtype, bm)
    try:
        kernel = builder(bm)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        bits = re.findall(r"requires\s+(\d+)\s*bits", text)
        print(f"COMPILE_FAIL dtype={dtype} bm={bm} N={N}")
        if bits:
            need = int(bits[0]) // 8
            elems = bm * N
            print(
                f"  UB_REQUIRED bytes={need} elems={elems} "
                f"B_per_elem={need / elems:.3f}"
            )
        else:
            print("  " + text[-1200:].replace("\n", " | "))
        return 1
    print(f"COMPILE_OK dtype={dtype} bm={bm} N={N} elems={bm * N} "
          f"eq_8192_elems={bm * N <= 8192}")
    del kernel
    return 0


if __name__ == "__main__":
    sys.exit(main())
