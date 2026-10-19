"""Decoded-input probe for the phase path (monotone x = h*1000 + w).

Prints, for each mismatch, the (h, w) corner the kernel actually took, plus the
bad-row / bad-col histogram, so a stale-buffer hazard (h from another block) can
be told apart from a transpose slot offset (w off by a constant).

  python3 debug_phase2.py [n c h w k s p dtype] [bh]
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
    n, c, h, w, k, s, p = (int(x) for x in a[:7]) if len(a) >= 7 else (1, 64, 112, 112, 3, 2, 1)
    dt = a[7] if len(a) > 7 else "float16"
    if len(a) > 8:
        os.environ["MAXPOOL2D_PHASE_BH"] = a[8]
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dt]
    mod = load("_max_pool2d_kernel.py")
    x = (torch.arange(h).view(h, 1) * 1000 +
         torch.arange(w).view(1, w)).float().repeat(n, c, 1, 1).to(td)
    kern = mod._max_pool2d_kernel(n, c, h, w, k, k, s, s, p, p, 1, 1,
                                 False, dt)(None)
    out = kern(x.npu()).cpu()
    ref = mod.golden_max_pool2d(x, (k, k), (s, s), (p, p), (1, 1), False)
    plan = mod._plan(n, c, h, w, k, k, s, s, p, p, 1, 1, False, dt)
    print(f"=== n{n} c{c} {h}x{w} k{k}s{s}p{p} {dt} path={plan['path']} "
          f"CH={plan['ch']} out={plan['out_h']}x{plan['out_w']} phase={plan['phase']}")
    bad = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
    print("mismatch", int(bad.sum()), "/", out.numel(),
          "nan_out", int(torch.isnan(out).sum()))
    if not bad.any():
        return
    # histogram over the first (n, c) plane
    per_row = bad[0, 0].sum(dim=1)
    print("bad rows:", [(i, int(v)) for i, v in enumerate(per_row) if v][:20])
    print("bad cols:", [(i, int(v)) for i, v in enumerate(bad[0, 0].sum(0)) if v][:20])
    ph = plan["phase"]
    print("bh", ph["bh"], "rows", ph["rows"], "pad_rows", ph["pad_rows"],
          "q_pad", ph["q_pad"], "w_pad", ph["w_pad"], "taps", ph["taps"])
    idx = bad[0, 0].nonzero()
    print("n bad in plane", idx.shape[0])
    for oh, ow in idx[:24].tolist():
        g = out[0, 0, oh, ow].item()
        r = ref[0, 0, oh, ow].item()
        print(f"  oh={oh:3d} ow={ow:3d} got={g:9.1f} -> (h={int(g)//1000},w={int(g)%1000:4d})"
              f"  ref={r:9.1f} -> (h={int(r)//1000},w={int(r)%1000:4d})")
    # per-channel-plane spread: same bug for all channels?
    bad_c = bad.any(dim=3).any(dim=2)[0]
    print("channels with any bad:", int(bad_c.sum()), "/", c)


if __name__ == "__main__":
    main()
