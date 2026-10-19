"""Test the "no UB slack at the end of et/ot" hypothesis without editing the
kernel: monkeypatch `_phase_plan` to inflate `pad_rows` and re-run the compare.

Trigger correlation found by dbg5: the slot math is exactly right (torch
emulation == golden), but the kernel output is wrong only for output columns
0..6 = the first 8 rows of et/ot, and ONLY for shapes where `q_pad == out_w`
(i.e. `pad_rows == 1 + q_pad`, zero spare rows).  Every passing L0 phase shape
has at least one spare row.

  python3 dbg6.py <slack_rows>
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
    slack = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mod = load("_max_pool2d_kernel.py")
    orig = mod._phase_plan

    def patched(*a, **k):
        p = orig(*a, **k)
        if p is not None:
            p = dict(p)
            p["pad_rows"] += slack
        return p

    mod._phase_plan = patched
    cases = [
        (1, 8, 16, 16, 3, 2, 1, "float32"),
        (1, 64, 112, 112, 3, 2, 1, "float16"),
    ]
    for n, c, h, w, kk, s, p, dt in cases:
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dt]
        plan = mod._plan(n, c, h, w, kk, kk, s, s, p, p, 1, 1, False, dt)
        ph = plan["phase"]
        torch.manual_seed(0)
        x = torch.randn(n, c, h, w, dtype=torch.float32).to(td)
        kern = mod._max_pool2d_kernel(n, c, h, w, kk, kk, s, s, p, p, 1, 1,
                                      False, dt)(None)
        out = kern(x.npu()).cpu()
        ref = mod.golden_max_pool2d(x, (kk, kk), (s, s), (p, p), (1, 1), False)
        bad = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
        per_col = bad.sum(dim=(0, 1, 2))
        print(f"[slack={slack}] n{n}c{c} {h}x{w} {dt} path={plan['path']} "
              f"bh={ph['bh']} ch={ph['ch']} q_pad={ph['q_pad']} out_w={plan['out_w']} "
              f"pad_rows={ph['pad_rows']} -> mismatch {int(bad.sum())}/{out.numel()} "
              f"cols={[(i, int(v)) for i, v in enumerate(per_col) if v][:10]}")


if __name__ == "__main__":
    main()
