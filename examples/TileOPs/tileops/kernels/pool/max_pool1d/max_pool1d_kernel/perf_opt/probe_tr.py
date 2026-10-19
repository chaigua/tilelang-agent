"""probe_tr.py -- isolate the cost of an in-UB `T.transpose` on npuir.

Two variants with identical GM traffic (load ch*L fp16, store ch*L fp16):
  copy : load (CH,L) -> same-shape copy -> store (CH,L) region
  trans: load (CH,L) -> T.transpose [1,0] -> (L,CH) -> store (L,CH) region
The Delta Task Duration is the transpose cost for one (CH,L) -> (L,CH) swap.

Usage: bash probe_run_tr.sh copy|trans
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

N = 32
CH = 80
L_IN = 4096
L = 64
UNITS = 52
CORES = 40
DT = "float16"


def build(variant):
    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _f(_tag):
        @T.prim_func
        def main(
            x: T.Tensor((N, CH, L_IN), DT),
            o1: T.Tensor((N, CH, L_IN), DT),
            o2: T.Tensor((N, L, CH), DT),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                a = T.alloc_ub((CH, L), DT)
                b = T.alloc_ub((L, CH), DT)
                c = T.alloc_ub((CH, L), DT)
                for t in T.serial(UNITS):
                    uid = t * CORES + cid
                    nb = uid % N
                    s0 = (uid * 7) % (L_IN - L)
                    T.copy(x[nb, 0:CH, s0:s0 + L], a)
                    if variant == "copy":
                        T.copy(a, c)
                        T.copy(c, o1[nb, 0:CH, 0:L])
                    else:
                        T.transpose(a, b, permutation=[1, 0])
                        T.copy(b, o2[nb, 0:L, 0:CH])

        return main

    return _f(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="copy", choices=["copy", "trans"])
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()
    kernel = build(args.variant)
    x = torch.randn(N, CH, L_IN, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    print(f"[probe_tr] variant={args.variant} shapes={[tuple(o.shape) for o in out]}")


if __name__ == "__main__":
    main()
