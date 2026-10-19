"""End-to-end check through the *wrapper's* block_m (same call shape as all.py).

The wrapper (read-only here) passes ``default_config["block_m"]`` to the factory;
this script reproduces exactly that call for the delivered perf_opt file and
compares against ``torch.all`` bit-exactly.
"""
import importlib.util, os, sys
import torch, torch_npu
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
from tileops.kernels.reduction.all import LogicalReduceKernel

def load(p):
    s = importlib.util.spec_from_file_location("perf_opt_mod", os.path.abspath(p))
    m = importlib.util.module_from_spec(s); sys.modules["perf_opt_mod"] = m; s.loader.exec_module(m); return m

mod = load("_logical_reduce_kernel.py")
CASES = [((32, 4096), -1), ((32, 32768), -1), ((4, 128, 4096), [0, 2])]
for shape, dim in CASES:
    k = LogicalReduceKernel(*mod._prepare_2d(torch.empty(shape, dtype=torch.bool, device="npu"), dim)[0].shape,
                            "all", torch.bool)
    bm = k.default_config["block_m"]
    x = torch.randint(0, 2, shape, dtype=torch.bool, device="npu")
    x[0] = False
    x2, out_shape = mod._prepare_2d(mod._to_logical_float32(x), dim)
    M, N = x2.shape
    cfg = mod._select_config(M, N, "float32")
    resolved = mod._resolve_block_m(bm, cfg["block_m"], cfg["tile_n"], M)
    y = mod._logical_reduce_kernel(M, N, "all", "float32")(bm)(x2).bool().cpu()
    ok = torch.equal(y.reshape(out_shape), torch.all(x.cpu(), dim=dim))
    print(f"{str(shape):16s} wrapper_bm={bm} cfg_bm={cfg['block_m']} resolved_bm={resolved} exact={ok}")
    if not ok:
        raise SystemExit(1)
