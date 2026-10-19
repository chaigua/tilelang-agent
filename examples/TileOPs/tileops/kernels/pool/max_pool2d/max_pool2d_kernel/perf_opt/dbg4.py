"""Full decode of the failing phase shapes with a monotone input x = h*1000 + w.

Small shape (1,8,16,16) is printed completely so the wrong slots can be read
off directly: value v decodes to (h = v//1000, w = v%1000).

  python3 dbg4.py [name]
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


def dec(v):
    v = float(v)
    if v != v:
        return "nan"
    if v < -1e30:
        return "-inf"
    if v > 1e30:
        return "+inf"
    return f"{int(v)//1000},{int(v)%1000}"


def run(mod, n, c, h, w, k, s, p, dt):
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dt]
    x = (torch.arange(h).view(h, 1) * 1000 +
         torch.arange(w).view(1, w)).float().repeat(n, c, 1, 1).to(td)
    kern = mod._max_pool2d_kernel(n, c, h, w, k, k, s, s, p, p, 1, 1,
                                  False, dt)(None)
    out = kern(x.npu()).cpu()
    ref = mod.golden_max_pool2d(x, (k, k), (s, s), (p, p), (1, 1), False)
    plan = mod._plan(n, c, h, w, k, k, s, s, p, p, 1, 1, False, dt)
    print(f"=== n{n} c{c} {h}x{w} k{k}s{s}p{p} {dt} path={plan['path']} "
          f"CH={plan['ch']} out={plan['out_h']}x{plan['out_w']} "
          f"phase={plan['phase']}")
    bad = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
    print("mismatch", int(bad.sum()), "/", out.numel(),
          "nan_out", int(torch.isnan(out).sum()))
    per_col = bad.sum(dim=(0, 1, 2))
    print("bad per ow:", [(i, int(v)) for i, v in enumerate(per_col) if v][:20])
    per_row = bad.sum(dim=(0, 1, 3))
    print("bad per oh:", [(i, int(v)) for i, v in enumerate(per_row) if v][:20])
    # which (n,c) planes are bad?
    pl = bad.any(dim=3).any(dim=2)
    print("bad planes:", int(pl.sum()), "/", pl.numel())
    out_h, out_w = plan["out_h"], plan["out_w"]
    for oh in range(out_h):
        if out_h > 12 and oh not in (0, 1, 2, out_h // 2, out_h - 1):
            continue
        print(f"-- oh={oh}  got: " + " ".join(
            f"{ow}:{dec(out[0, 0, oh, ow])}" for ow in range(min(out_w, 12))))
        print(f"-- oh={oh}  ref: " + " ".join(
            f"{ow}:{dec(ref[0, 0, oh, ow])}" for ow in range(min(out_w, 12))))
    return out, ref, plan


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    mod = load("_max_pool2d_kernel.py")
    if which in ("all", "small"):
        for dt in ("fp32", "fp16"):
            d = {"fp32": "float32", "fp16": "float16"}[dt]
            run(mod, 1, 8, 16, 16, 3, 2, 1, d)
    if which in ("all", "big"):
        run(mod, 1, 64, 112, 112, 3, 2, 1, "float16")


if __name__ == "__main__":
    main()
