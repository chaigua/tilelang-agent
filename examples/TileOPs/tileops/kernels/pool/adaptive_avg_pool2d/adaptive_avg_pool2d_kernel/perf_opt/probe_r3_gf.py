"""Round-3 probe: C4 global path -- reduce-chain variants and CH sweep.

Motivation (CG-2026-0026 / CONST-reduce-ub-per-element): BishengIR's reduction
lowering books UB *linearly in input-tile elements*, with a unit price that
depends on the chain form.  The current `perf_opt` chain
(`vcast f16->f32 full plane` + `reduce_sum`) sits at the expensive tier, which
caps CH at <=62 and therefore pins the per-core serial round count at 13.

Chains (x declares (N, C, 1, H*W); reduce the innermost H*W axis):

  f32      : tile f16 -> vcast -> f32 plane -> reduce_sum(dim=2)   (= current best)
  copycast : T.copy(f16 GM -> f32 UB) -> reduce_sum(dim=2)
  f16same  : tile f16 -> reduce_sum(f16 -> f16) -> vcast -> f32 -> vmul
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

N, C, H, W = 8, 2048, 7, 7
HW = H * W
DTYPE = "float16"


def build(chain: str, ch: int, mb: bool = False):
    ch = min(C, ch)
    nblk = (C + ch - 1) // ch
    nlogical = N * nblk
    nk = min(nlogical, NPUUtils.get().get_aicore_num() * 2)
    nl = (nlogical + nk - 1) // nk
    inv = 1.0 / float(HW)
    pc = {"npuir.enable_auto_multi_buffer": mb}

    @tilelang.jit(out_idx=[1], target="npuir", pass_configs=pc)
    def _func(_c: int):
        @T.prim_func
        def main(
            x: T.Tensor((N, C, 1, HW), DTYPE),
            out: T.Tensor((N, C, 1, 1), DTYPE),
        ):
            with T.Kernel(nk, is_npu=True) as (cid, _):
                tile = T.alloc_ub((ch, 1, HW), DTYPE)
                t32 = T.alloc_ub((ch, 1, HW), "float32")
                s16 = T.alloc_ub((ch, 1, 1), DTYPE)
                sred = T.alloc_ub((ch, 1, 1), "float32")
                gout = T.alloc_ub((ch, 1, 1), DTYPE)
                for t in T.serial(nl):
                    uid = t * nk + cid
                    if uid < nlogical:
                        nb = uid // nblk
                        cb = uid % nblk
                        if chain == "copycast":
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], t32)
                            T.reduce_sum(t32, sred, dim=2)
                        elif chain == "f16same":
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], tile)
                            T.reduce_sum(tile, s16, dim=2)
                            T.vcast(s16, sred, round_mode="rint")
                        elif chain == "f16mul":
                            # no fp32 vector op at all: reduce and scale in fp16
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], tile)
                            T.reduce_sum(tile, s16, dim=2)
                            T.vmul(s16, inv, gout)
                        else:
                            T.copy(x[nb, cb * ch : cb * ch + ch, 0:1, 0:HW], tile)
                            T.vcast(tile, t32, round_mode="rint")
                            T.reduce_sum(t32, sred, dim=2)
                        T.vmul(sred, inv, sred)
                        T.vcast(sred, gout, round_mode="rint")
                        T.copy(gout, out[nb, cb * ch : cb * ch + ch, 0:1, 0:1])

        return main

    return _func(ch), nk, nl, nblk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, choices=("f32", "copycast", "f16same", "f16mul"))
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--mb", type=int, default=0)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    kernel, nk, nl, nblk = build(args.chain, args.ch, bool(args.mb))
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float32).to(torch.float16).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            out = kernel(x)
    tag = f"[r3gf {args.chain} ch{args.ch} mb{args.mb}] nk={nk} nl={nl}"
    if args.check:
        import torch.nn.functional as F

        ref = F.adaptive_avg_pool2d(x.cpu().float(), (1, 1)).to(torch.float16)
        got = out.cpu().float()
        d = (got - ref).abs()
        print(
            f"{tag} max_diff={d.max().item():.3e} exact={bool((got == ref.float()).all())}"
        )
    else:
        print(f"{tag} ok")


if __name__ == "__main__":
    main()
