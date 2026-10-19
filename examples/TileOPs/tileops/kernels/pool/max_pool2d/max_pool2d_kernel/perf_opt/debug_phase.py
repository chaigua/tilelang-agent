"""Locate the phase-path mismatch with a monotone (decodable) input.

x[n, c, h, w] = h*1000 + w, so for a k3s2p1 window the golden output is
min(H-1, 2oh+1)*1000 + min(W-1, 2ow+1): the decoded value names the (h, w) corner
the kernel actually took.

  python3 debug_phase.py
"""
import importlib.util
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def load(path):
    spec = importlib.util.spec_from_file_location("kmod", os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def show(tag, got, ref):
    print(f"--- {tag}  got/ref (channel 0)")
    for r in range(min(8, ref.shape[0])):
        g = [f"{int(v):5d}" for v in got[r]]
        e = [f"{int(v):5d}" for v in ref[r]]
        print(f"  row{r} got {g}")
        print(f"  row{r} ref {e}")


def main():
    mod = load("_max_pool2d_kernel.py")
    cases = [(1, 16, 12, 12, 3, 2, 1, "float32"),
             (1, 8, 16, 16, 3, 2, 1, "float32"),
             (2, 64, 55, 57, 3, 2, 1, "bfloat16"),
             (32, 64, 112, 112, 3, 2, 1, "float16")]
    for (n, c, h, w, k, s, p, dt) in cases:
        td = torch.float32 if dt == "float32" else torch.float16
        torch.manual_seed(0)
        if h * w <= 400:
            x = (torch.arange(h).view(h, 1) * 1000 +
                 torch.arange(w).view(1, w)).float().repeat(n, c, 1, 1).to(td)
        else:
            x = torch.randn(n, c, h, w, dtype=torch.float32).to(td)
        kern = mod._max_pool2d_kernel(n, c, h, w, k, k, s, s, p, p, 1, 1,
                                     False, dt)(None)
        out = kern(x.npu()).cpu()
        ref = mod.golden_max_pool2d(x, (k, k), (s, s), (p, p), (1, 1), False)
        plan = mod._plan(n, c, h, w, k, k, s, s, p, p, 1, 1, False, dt)
        print(f"=== n{n} c{c} {h}x{w} k{k}s{s}p{p} {dt} path={plan['path']} "
              f"CH={plan['ch']} phase={plan['phase']}")
        show("out", out[0, 0], ref[0, 0])
        bad = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
        print("  mismatch count", int(bad.sum()), "/", out.numel(),
              " nan_out", int(torch.isnan(out).sum()))
        if bad.any():
            per_row = bad[0, 0].sum(dim=1)
            per_col = bad[0, 0].sum(dim=0)
            print("  rows with mismatch:", [(i, int(v)) for i, v in
                                            enumerate(per_row) if v][:12])
            print("  cols with mismatch:", [(i, int(v)) for i, v in
                                            enumerate(per_col) if v][:12])
            idx = bad[0, 0].nonzero()[:6]
            for oh, ow in idx.tolist():
                print(f"    oh={oh} ow={ow} got={out[0,0,oh,ow].item()} "
                      f"ref={ref[0,0,oh,ow].item()}")


if __name__ == "__main__":
    main()
