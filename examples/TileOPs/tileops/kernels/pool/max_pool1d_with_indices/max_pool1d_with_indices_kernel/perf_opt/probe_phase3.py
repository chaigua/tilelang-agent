"""Probe the three primitives the phase-batched layout needs.

  (1) `T.arange` on a (ch, bl) int32 plane with strides [0, s] -> column ramp
  (2) `T.reshape` of a (ch, lspan) plane into a (ch, bl, P) view
  (3) `T.transpose` with permutation [2, 0, 1] (3-D phase extraction)

All three are checked for bit-exactness:

    python probe_phase3.py --ch 80 --bl 16
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


def build(ch, bl, p=3, s=3, dtype="float16"):
    lspan = bl * p

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((ch, lspan), dtype),
            y: T.Tensor((ch, bl), "int32"),
            z: T.Tensor((ch, bl), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                src = T.alloc_ub((ch, lspan), dtype)
                src3 = T.alloc_ub((ch, bl, p), dtype)
                ph = T.alloc_ub((p, ch, bl), dtype)
                ramp = T.alloc_ub((ch, bl), "int32")
                T.copy(x, src)
                T.reshape(src, src3)
                T.transpose(src3, ph, permutation=[2, 0, 1])
                T.arange(ramp, [0, s], 0)
                T.copy(ramp, y)
                T.copy(ph[2, 0:ch, 0:bl], z)

        return main

    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=80)
    ap.add_argument("--bl", type=int, default=16)
    args = ap.parse_args()
    ch, bl = args.ch, args.bl
    k = build(ch, bl)
    x = torch.arange(ch * bl * 3, dtype=torch.float32).reshape(ch, bl * 3)
    xn = x.to(torch.float16).npu()
    with torch.no_grad():
        y, z = k(xn)
    y, z = y.cpu(), z.cpu()
    xc = x.to(torch.float16)
    ramp_ref = torch.arange(bl, dtype=torch.int32).repeat(ch, 1) * 3
    p2_ref = xc[:, 2::3]
    print(f"[phase3] arange_ramp_exact={bool(torch.equal(y, ramp_ref))} "
          f"phase2_exact={bool(torch.equal(z, p2_ref))}")
    if not torch.equal(y, ramp_ref):
        print("ramp got", y[0, :6].tolist(), "ref", ramp_ref[0, :6].tolist())
    if not torch.equal(z, p2_ref):
        print("ph2  got", z[0, :6].tolist(), "ref", p2_ref[0, :6].tolist())


if __name__ == "__main__":
    main()
