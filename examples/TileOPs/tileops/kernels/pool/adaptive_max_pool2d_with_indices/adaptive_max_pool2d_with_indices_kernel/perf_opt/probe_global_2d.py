"""Global-1x1 whole-plane reduce probe, 2-D layout (no transpose).

Hypothesis chain established so far (msprof op):
  * p1/p7: a dual store whose staging is produced by a *contiguous* op chain is
    cheap (Task 4.73 us, mte3 0.92 us) -- the store form is NOT the problem;
  * the flat-scan form (r1_flat/flat1d) needs a `(C, plane) -> (plane, C)`
    transpose / a strided `flat[:, :, 0]` gather, and probe #1 + the profile
    show UB_to_GM == GM_to_UB == whole tile => TRAP-transpose-ub-gm-roundtrip
    (96 KB/block, mte3 ~70 us).
  * probe_global_reduce (runtime pattern instead of arange) proved the
    whole-plane argmax is EXACT with a non-arange pattern -- the arange trap is
    producer-specific, not chain-specific.

This probe removes the transpose entirely: the plane is declared as the
innermost axis of a 2-D tile `(ch, plane)` and every reduction runs on `dim=1`.
All value/index staging is then produced by `reduce_*` / `vcast` (contiguous
producers), so the epilogue store stays on the cheap path.

    python probe_global_2d.py --ch 64 --check
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: F401,E402
from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

N, C, H, W = 8, 2048, 7, 7
PLANE = H * W


def build(ch: int):
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    num_cblk = (C + ch - 1) // ch
    num_logical = N * num_cblk
    num_kernels = min(num_logical, vector_cores)
    num_local = (num_logical + num_kernels - 1) // num_kernels

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((N, C, PLANE), "float16"),
            out: T.Tensor((N, C, 1, 1), "float16"),
            indices: T.Tensor((N, C, 1, 1), "int64"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                flat = T.alloc_ub((ch, PLANE), "float16")
                gbrc = T.alloc_ub((ch, PLANE), "float16")
                eqb = T.alloc_ub((ch, PLANE), "bool")
                kw2d = T.alloc_ub((ch, PLANE), "int32")
                cand = T.alloc_ub((ch, PLANE), "int32")
                sent = T.alloc_ub((ch, PLANE), "int32")
                gmax16 = T.alloc_ub((ch, 1), "float16")
                gmax = T.alloc_ub((ch, 1), "float32")
                out32 = T.alloc_ub((ch, 1), "float32")
                gout = T.alloc_ub((ch, 1), "float16")
                infbuf = T.alloc_ub((ch, 1), "float32")
                nanbuf = T.alloc_ub((ch, 1), "float32")
                firstk = T.alloc_ub((ch, 1), "int32")
                nank = T.alloc_ub((ch, 1), "int32")
                zeros_i = T.alloc_ub((ch, 1), "int32")
                cfinal = T.alloc_ub((ch, 1), "bool")
                idx32 = T.alloc_ub((ch, 1), "int32")
                idx64 = T.alloc_ub((ch, 1), "int64")
                k1d = T.alloc_ub((PLANE,), "int32")
                k1r = T.alloc_ub((1, PLANE), "int32")
                stg = T.alloc_ub((1,), "int32")

                # runtime-built position table k = 0..PLANE-1 (NOT arange)
                for k in T.serial(PLANE):
                    T.vbrc(T.int32(k), stg)
                    T.copy(stg, k1d[k : k + 1])
                T.reshape(k1d, k1r)
                T.vbrc(k1r, kw2d)
                T.vbrc(T.int32(0), zeros_i)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // num_cblk
                        cb = uid % num_cblk
                        real_c = T.min(ch, C - cb * ch)
                        T.copy(
                            x[nb, cb * ch : cb * ch + real_c, 0:PLANE],
                            flat[0:real_c, 0:PLANE],
                        )
                        # whole-plane value max (fp16 max is exact)
                        T.reduce_max(flat, gmax16, dim=1, clear=True)
                        T.vcast(gmax16, gmax, round_mode="rint")
                        # first-hit = min flat index over the max-equality set
                        T.vbrc(gmax16, gbrc)
                        T.vcmp(flat, gbrc, eqb, "eq")
                        T.vbrc(T.int32(PLANE), sent)
                        T.vselect(eqb, kw2d, sent, cand)
                        T.reduce_min(cand, firstk, dim=1, clear=True)
                        # last-NaN = max flat index over the NaN set
                        T.vcmp(flat, flat, eqb, "eq")  # number mask
                        T.vbrc(T.int32(-1), sent)
                        T.vselect(eqb, sent, kw2d, cand)
                        T.reduce_max(cand, nank, dim=1, clear=True)
                        # cfinal = (nank >= 0) <=> the plane holds a NaN
                        T.vcmp(nank, zeros_i, cfinal, "ge")
                        T.vselect(cfinal, nank, firstk, idx32)
                        T.vcast(idx32, idx64, round_mode="rint")
                        T.copy(
                            idx64[0:real_c, 0:1],
                            indices[nb, cb * ch : cb * ch + real_c, 0, 0:1],
                        )
                        T.vbrc(T.infinity("float32"), infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vselect(cfinal, nanbuf, gmax, out32)
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(
                            gout[0:real_c, 0:1],
                            out[nb, cb * ch : cb * ch + real_c, 0, 0:1],
                        )

        return main

    return _func()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    kernel = build(args.ch)
    torch.manual_seed(0)
    x = torch.randn(N, C, PLANE, dtype=torch.float32)
    if args.check:
        x[0, 0, 0] = float("nan")
        x[0, 1, 5] = float("nan")
        x[0, 1, 40] = float("nan")
        x[0, 2, 3] = float("inf")
        x[0, 3, 7] = float("inf")
        x[0, 3, 11] = float("-inf")
        x[0, 4, :] = float("-inf")
        x[0, 5, :] = 5.0
        x[0, 6, 44] = 9.0
        x[0, 6, 4] = 9.0
        x[0, 7, 2] = 9.0
        x[0, 7, 46] = 9.0
    xh = x.to(torch.float16).npu()
    with torch.no_grad():
        out = idx = None
        for _ in range(args.iters):
            out, idx = kernel(xh)
    if args.check:
        xc = x.to(torch.float16)
        ref, ref_idx = F.adaptive_max_pool2d(
            xc.reshape(N, C, 1, PLANE).float(), (1, 1), return_indices=True
        )
        got, gi = out.cpu(), idx.cpu()
        val_ok = bool(
            torch.equal(got, ref.to(torch.float16))
            or torch.allclose(got.float(), ref.float(), atol=0.0, rtol=0.0, equal_nan=True)
        )
        per_ch = (gi.reshape(N * C) == ref_idx.reshape(N * C))
        bad = (~per_ch).nonzero().flatten().tolist()
        print(f"[check] ch={args.ch} val_ok={val_ok} idx_exact={bool(per_ch.all())} "
              f"idx_mismatch={bad[:8]} (n={len(bad)})")
        for c0 in (0, 1, 5, 6, 7):
            print(f"   c{c0}: got={int(gi[0, c0, 0, 0])} ref={int(ref_idx[0, c0, 0, 0])}")
        if not (val_ok and bool(per_ch.all())):
            raise SystemExit(1)
    else:
        print(f"[probe2d-run] ch={args.ch} out={tuple(out.shape)} idx={tuple(idx.shape)}")


if __name__ == "__main__":
    main()
