"""Probe: does `T.deinterleave(channel_nums=3)` work on this toolchain?

Sibling task (tilelang 0.1.2+67a0441) reported SIGABRT in bishengir-compile for
channel_nums=3; here (0.1.2+e8b5d51e) it is re-verified before being used as a
phase extractor.  Also checks the (ch, bl) plane layout used by the batched
kernel variant (deinterleave along the last axis + full-plane ops).

    python probe_deint.py --ch 80 --bl 12 --phase 3
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def build(ch, bl, phase, dtype="float16"):
    lspan = bl * phase

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((ch, lspan), dtype),
            y: T.Tensor((ch, bl), dtype),
            z: T.Tensor((ch, bl), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                src = T.alloc_ub((ch, lspan), dtype)
                p0 = T.alloc_ub((ch, bl), dtype)
                p1 = T.alloc_ub((ch, bl), dtype)
                p2 = T.alloc_ub((ch, bl), dtype)
                T.copy(x, src)
                T.deinterleave(src, p0, p1, p2, channel_nums=phase)
                T.copy(p0, y)
                T.vmax(p1, p2, p2)
                T.copy(p2, z)

        return main

    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=80)
    ap.add_argument("--bl", type=int, default=12)
    ap.add_argument("--phase", type=int, default=3)
    args = ap.parse_args()
    k = build(args.ch, args.bl, args.phase)
    x = torch.arange(args.ch * args.bl * args.phase, dtype=torch.float32).reshape(
        args.ch, args.bl * args.phase)
    xn = x.to(torch.float16).npu()
    with torch.no_grad():
        y, z = k(xn)
    y = y.cpu()
    xc = x.to(torch.float16)
    ref0 = xc[:, 0::args.phase]
    ok0 = bool(torch.equal(y, ref0))
    ref1 = xc[:, 1::args.phase]
    ref2 = xc[:, 2::args.phase] if args.phase > 2 else xc[:, 1::args.phase]
    okz = bool(torch.equal(z.cpu(), torch.maximum(ref1, ref2)))
    print(f"[deint] phase={args.phase} p0_exact={ok0} p1p2_max_exact={okz}")
    if not (ok0 and okz):
        print("y[0,:4]", y[0, :4].tolist(), "ref", ref0[0, :4].tolist())


if __name__ == "__main__":
    main()
