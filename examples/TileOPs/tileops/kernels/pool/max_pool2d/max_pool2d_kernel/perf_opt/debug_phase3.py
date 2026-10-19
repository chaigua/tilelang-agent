"""Locate the source of a wrong phase-path output with a RANDOM input.

For a bad output position the golden window is known; searching the whole input
plane for the value the kernel produced tells which (ih, iw) it actually picked
(random fp32 values are effectively unique).  Also prints whether the same
mismatch pattern reproduces, i.e. race vs geometry.

  python3 debug_phase3.py n c h w k s p dtype [bh]
"""
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def load(path):
    spec = importlib.util.spec_from_file_location("kmod", os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    a = sys.argv[1:]
    n, c, h, w, k, s, p = (int(x) for x in a[:7])
    dt = a[7]
    if len(a) > 8:
        os.environ["MAXPOOL2D_PHASE_BH"] = a[8]
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dt]
    mod = load("_max_pool2d_kernel.py")
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, dtype=torch.float32).to(td)
    kern = mod._max_pool2d_kernel(n, c, h, w, k, k, s, s, p, p, 1, 1,
                                  False, dt)(None)
    out = kern(x.npu()).cpu()
    ref = mod.golden_max_pool2d(x, (k, k), (s, s), (p, p), (1, 1), False)
    plan = mod._plan(n, c, h, w, k, k, s, s, p, p, 1, 1, False, dt)
    print(f"=== n{n} c{c} {h}x{w} k{k}s{s}p{p} {dt} d={td} path={plan['path']} "
          f"CH={plan['ch']} out={plan['out_h']}x{plan['out_w']} phase={plan['phase']}")
    bad = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
    print("mismatch", int(bad.sum()), "/", out.numel(),
          "nan_out", int(torch.isnan(out).sum()))
    per_c = bad.any(dim=3).any(dim=2).sum(dim=(0,))
    print("bad per channel:", [int(v) for v in per_c][:16])
    per_row = bad.sum(dim=(0, 1, 3))
    print("bad per oh:", [(i, int(v)) for i, v in enumerate(per_row) if v][:12])
    per_col = bad.sum(dim=(0, 1, 2))
    print("bad per ow:", [(i, int(v)) for i, v in enumerate(per_col) if v][:12])
    xp = x[0, 0].float()
    idx = bad[0, 0].nonzero()
    print("n bad in plane", idx.shape[0])
    for oh, ow in idx[:10].tolist():
        g = out[0, 0, oh, ow].float()
        r = ref[0, 0, oh, ow].float()
        hit = (xp == g).nonzero()
        hs = [tuple(t.tolist()) for t in hit[:6]]
        print(f"  oh={oh:3d} ow={ow:3d} got={g:12.6f} ref={r:12.6f} "
              f"got_at={hs}  (window h={oh*s-p}..{oh*s-p+k-1} w={ow*s-p}..{ow*s-p+k-1})")


if __name__ == "__main__":
    main()
