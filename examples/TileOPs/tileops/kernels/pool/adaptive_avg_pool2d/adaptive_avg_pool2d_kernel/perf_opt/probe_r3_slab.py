"""Round-3 probe: decouple DMA granularity from the reduce UB-price cap.

Diagnosis (round 3):
  * GM->UB DMA cost is dominated by the *number of rounds*: 13 x 6 KB -> 4.2 us
    above the empty floor, but one 40 KB transfer -> 0.83 us (probe_floor).
  * The reduce UB price (CONST-reduce-ub-per-element) caps a single
    `T.reduce_sum` input at ~6144 fp16 elements == ~125 planes of 49, so CH
    cannot exceed ~120 -> >=4 serial rounds per core -> DMA stays small-grained.
  * Naive wider CH (probe_r3_gf f16same ch120) does NOT help.

New structure: give every core **one big contiguous DMA** (a slab of P planes)
then loop the reduction over fixed sub-chunks copied UB->UB into a small
reduce buffer, so DMA granularity and reduce granularity become independent.

  NK = N * (C / P) cores, each owns channels [cb*P, +P) of sample nb
  big = alloc_ub (P, 1, HW) f16   -- ONE DMA of P*HW*2 bytes
  for k in serial(P // SUB):      -- static extents
      T.copy(big[k*SUB:(k+1)*SUB], sub)      # UB->UB
      vcast -> f32, reduce_sum(dim=2), vmul, vcast

Variants: `f32` (vcast->f32 plane->reduce) and `f16same` (reduce f16->f16).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

N, C, H, W = 8, 2048, 7, 7
HW = H * W
DTYPE = "float16"


def build(chain: str, P: int = 512, SUB: int = 64):
    assert C % P == 0 and P % SUB == 0
    NK = N * (C // P)
    K = P // SUB
    inv = 1.0 / float(HW)
    CBLK = C // P

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_s: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C // P, P, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(NK, is_npu=True) as (cid, _):
                nb = cid // CBLK
                cb = cid % CBLK
                big = T.alloc_ub((P, 1, HW), DTYPE)
                sub = T.alloc_ub((SUB, 1, HW), DTYPE)
                t32 = T.alloc_ub((SUB, 1, HW), "float32")
                s16 = T.alloc_ub((SUB, 1, 1), DTYPE)
                sred = T.alloc_ub((SUB, 1, 1), "float32")
                gout = T.alloc_ub((SUB, 1, 1), DTYPE)
                T.copy(x[nb, cb, 0:P, 0:1, 0:HW], big)
                for k in T.serial(K):
                    T.copy(big[k * SUB : (k + 1) * SUB, 0:1, 0:HW], sub)
                    if chain == "f16same":
                        T.reduce_sum(sub, s16, dim=2)
                        T.vcast(s16, sred, round_mode="rint")
                    else:
                        T.vcast(sub, t32, round_mode="rint")
                        T.reduce_sum(t32, sred, dim=2)
                    T.vmul(sred, inv, sred)
                    T.vcast(sred, gout, round_mode="rint")
                    T.copy(
                        gout,
                        out[nb, cb * P + k * SUB : cb * P + (k + 1) * SUB, 0:1, 0:1],
                    )

        return main

    return _func(P), NK, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, choices=("f32", "f16same"))
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--SUB", type=int, default=64)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    kernel, nk, k = build(args.chain, args.P, args.SUB)
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    tag = f"[slab {args.chain} P{args.P} SUB{args.SUB}] nk={nk} K={k}"
    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        d = (got - ref).abs()
        print(f"{tag} max_diff={d.max().item():.3e} exact={bool((got == ref.float()).all())}")
    else:
        print(f"{tag} ok")


if __name__ == "__main__":
    main()
