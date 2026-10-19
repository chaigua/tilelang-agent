"""Compare the lowered MLIR of two kernel files for the same shape.

Used to prove that the delivered file leaves the column-tiled path (32k / 3d)
byte-identical to the Stage-3 baseline, i.e. those workloads cannot have
regressed through the tuning change.
"""

import hashlib
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def load(path):
    name = "cmp_" + hashlib.sha256(path.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mlir_of(module, M, N, dtype):
    cfg = module._select_config(M, N, dtype)
    kernel = module._logical_reduce_kernel(M, N, "any", dtype)(cfg["block_m"])
    src = getattr(kernel, "mlir_content", None) or ""
    return cfg, hashlib.sha256(src.encode()).hexdigest(), len(src)


def main():
    base = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_logical_reduce_kernel.py")
    )
    final = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_logical_reduce_kernel.py")
    )
    a, b = load(base), load(final)
    for M, N, dt in ((32, 32768, "float32"), (128, 16384, "float32"), (32, 4096, "float32")):
        ca, ha, la = mlir_of(a, M, N, dt)
        cb, hb, lb = mlir_of(b, M, N, dt)
        same = ha == hb and la == lb
        print(
            f"({M},{N}) {dt}: cfg_base={ca['path']}/{ca['block_m']}/{ca['tile_n']} "
            f"cfg_final={cb['path']}/{cb['block_m']}/{cb['tile_n']} "
            f"mlir_sha_equal={same} len {la} vs {lb}"
        )


if __name__ == "__main__":
    main()
