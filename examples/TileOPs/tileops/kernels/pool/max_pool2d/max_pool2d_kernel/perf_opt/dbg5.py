"""Separate a slot-math (logic) error from a buffer/race error in the phase path.

Emulates, in torch on CPU, exactly what `_phase_plan`'s taps say the kernel
computes (per-row DMA -> deinterleave -> transpose into et/ot -> wide vmax ->
transpose -> store).  Then compares three tensors:

  kernel  (NPU run)   emu  (torch model of the intended algorithm)   golden

* kernel != emu  -> the kernel's buffers do not hold what the algorithm
                    intends (race / stale / codegen) for those positions.
* emu    != golden -> the tap/slot math itself is wrong, and which slot is
                    misread is visible from the emu-vs-golden diff pattern.

  python3 dbg5.py [n c h w k s p dtype]
"""
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

NINF = float("-inf")


def load(path):
    spec = importlib.util.spec_from_file_location("kmod", os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def emulate(mod, plan, x, k, s, p, d_h=1, d_w=1):
    ph = plan["phase"]
    bh, rows = ph["bh"], ph["rows"]
    pad_rows, q_pad, w_pad, taps = (ph["pad_rows"], ph["q_pad"], ph["w_pad"],
                                    ph["taps"])
    ch = ph["ch"]
    n, c_in, h_in, w_in = x.shape
    out_h, out_w = plan["out_h"], plan["out_w"]
    n_ohb = out_h // bh
    num_cblk = (c_in + ch - 1) // ch
    res = torch.full((n, c_in, out_h, out_w), NINF, dtype=torch.float32)
    xf = x.float()
    for nb in range(n):
        for cb in range(num_cblk):
            real_c = min(ch, c_in - cb * ch)
            for ohb in range(n_ohb):
                oh0 = ohb * bh
                ih0 = oh0 * s - p
                acc = torch.full((bh * out_w, ch), NINF)
                for r in range(rows):
                    ih = ih0 + r
                    if not (0 <= ih < h_in):
                        continue
                    src = torch.full((ch, w_pad), NINF)
                    src[:real_c, :w_in] = xf[nb, cb * ch:cb * ch + real_c, ih, :]
                    eo = src[:, 0:w_pad:2]
                    oo = src[:, 1:w_pad:2]
                    et = torch.full((pad_rows, ch), NINF)
                    ot = torch.full((pad_rows, ch), NINF)
                    et[1:1 + q_pad] = eo.t()
                    ot[1:1 + q_pad] = oo.t()
                    for kh in range(k):
                        num = r - kh * d_h
                        if num < 0 or num % s:
                            continue
                        ob = num // s
                        if ob >= bh:
                            continue
                        for odd, base in taps:
                            buf = ot if odd else et
                            dst = acc[ob * out_w:(ob + 1) * out_w]
                            acc[ob * out_w:(ob + 1) * out_w] = torch.maximum(
                                dst, buf[base:base + out_w])
                # store acc[(ob,ow), c] -> out[c, oh0+ob, ow]
                a = acc.view(bh, out_w, ch)
                res[nb, cb * ch:cb * ch + real_c, oh0:oh0 + bh, :] = \
                    a[:, :, :real_c].permute(2, 0, 1)
    return res


def cmp(a, b, name):
    bad = (a != b) & ~(torch.isnan(a) & torch.isnan(b))
    print(f"{name}: mismatch {int(bad.sum())}/{a.numel()}")
    per_col = bad.sum(dim=(0, 1, 2))
    per_row = bad.sum(dim=(0, 1, 3))
    print("  per ow:", [(i, int(v)) for i, v in enumerate(per_col) if v][:14])
    print("  per oh:", [(i, int(v)) for i, v in enumerate(per_row) if v][:14])
    return bad


def main():
    a = sys.argv[1:]
    n, c, h, w, k, s, p = (int(v) for v in a[:7]) if len(a) >= 7 \
        else (1, 8, 16, 16, 3, 2, 1)
    dt = a[7] if len(a) > 7 else "float32"
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dt]
    mod = load("_max_pool2d_kernel.py")
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, dtype=torch.float32).to(td)
    kern = mod._max_pool2d_kernel(n, c, h, w, k, k, s, s, p, p, 1, 1,
                                  False, dt)(None)
    out = kern(x.npu()).cpu().float()
    ref = mod.golden_max_pool2d(x, (k, k), (s, s), (p, p), (1, 1), False).float()
    plan = mod._plan(n, c, h, w, k, k, s, s, p, p, 1, 1, False, dt)
    print(f"=== n{n} c{c} {h}x{w} k{k}s{s}p{p} {dt} path={plan['path']} "
          f"CH={plan['ch']} out={plan['out_h']}x{plan['out_w']} phase={plan['phase']}")
    if plan["path"] != "phase":
        print("not the phase path -- nothing to model")
        return
    emu = emulate(mod, plan, x, k, s, p)
    cmp(ref, out, "golden vs kernel")
    cmp(ref, emu, "golden vs emu   ")
    bad = cmp(emu, out, "emu    vs kernel")
    idx = bad[0, 0].nonzero()
    print("first bad (plane 0,0):")
    for oh, ow in idx[:12].tolist():
        print(f"  oh={oh:3d} ow={ow:3d} kernel={out[0, 0, oh, ow]:12.6f} "
              f"emu={emu[0, 0, oh, ow]:12.6f} ref={ref[0, 0, oh, ow]:12.6f}")
    # is the emu wrong on the SAME positions as the kernel (=> pure logic bug)?
    bk = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
    be = (emu != ref) & ~(torch.isnan(emu) & torch.isnan(ref))
    print("kernel-bad", int(bk.sum()), "emu-bad", int(be.sum()),
          "both", int((bk & be).sum()), "kernel-only", int((bk & ~be).sum()),
          "emu-only", int((be & ~bk).sum()))


if __name__ == "__main__":
    main()
