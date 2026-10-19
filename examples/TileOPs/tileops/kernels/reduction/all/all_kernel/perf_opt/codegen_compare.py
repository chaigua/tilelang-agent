"""Compare generated kernel source sha between two implementation files.

Byte-identical codegen between two artifacts means any measured delta is
run-state, not a code change (BP_run_state_bimodality protocol).
"""
import hashlib, importlib.util, os, sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")


def load(path):
    name = "cg_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


CASES = [((32, 4096), "float32"), ((32, 32768), "float32"), ((128, 16384), "float32")]


def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    for (M, N), dtype in CASES:
        hashes = []
        for mod in (a, b):
            cfg = mod._select_config(M, N, dtype)
            k = mod._logical_reduce_kernel(M, N, "all", dtype)(cfg["block_m"])
            src = k.get_kernel_source()
            if isinstance(src, str):
                src = src.encode()
            hashes.append((hashlib.sha256(src).hexdigest()[:16], cfg))
        same = hashes[0][0] == hashes[1][0]
        print(f"({M},{N}) {dtype}: A={hashes[0][0]} cfg={hashes[0][1]}")
        print(f"                B={hashes[1][0]} cfg={hashes[1][1]}  identical={same}")
        if not same:
            print('                -> differs')


if __name__ == "__main__":
    main()
