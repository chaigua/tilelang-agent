"""Primitive probes for the phase-decimated candidate (correctness only, no msprof).

Q1 `vmax` NaN propagation: `vmax(NaN, v)` and `vmax(v, NaN)` (both operand
   orders, accumulated over a serial chain) must yield NaN in f16/f32.  If yes,
   a window NaN is detected with one final `vcmp(acc, acc)` instead of a
   per-element eq-mask product (the design's largest op-count term).
Q2 deinterleave with `channel_nums=2` on a (CH, W) buffer:
   dst0 == src[..., 0::2], dst1 == src[..., 1::2].
Q3 static compile-time element-offset slices as vmax operands (1 element = 2 B /
   4 B misalignment) stay exact: `max(even[ow], odd[ow], even[ow+1])`.
Q4 the documented alternative: eq-mask `vcmp(row,row,"eq")` -> phase split ->
   window `vmin` == 0 iff the window holds a NaN; `vselect` picks NaN.
Q5 +-inf semantics on the same chain: {+inf,-inf} -> +inf (never NaN).

Run: python probe_prims.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401,E402

CH, L, OW = 32, 114, 56
NAN = float("nan")
INF = float("inf")


@tilelang.jit(out_idx=[1], target="npuir")
def _probe(dtype: str):
    @T.prim_func
    def main(
        x: T.Tensor((1, CH, L), dtype),
        out: T.Tensor((6, CH, OW), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            row = T.alloc_ub((CH, L), dtype)
            even = T.alloc_ub((CH, L // 2), dtype)
            odd = T.alloc_ub((CH, L // 2), dtype)
            acc = T.alloc_ub((CH, OW), dtype)
            accf = T.alloc_ub((CH, OW), "float32")
            mmask = T.alloc_ub((CH, L), "bool")
            mrow = T.alloc_ub((CH, L), "float32")
            meven = T.alloc_ub((CH, L // 2), "float32")
            modd = T.alloc_ub((CH, L // 2), "float32")
            macc = T.alloc_ub((CH, OW), "float32")
            mf = T.alloc_ub((CH, OW), "float32")
            cond = T.alloc_ub((CH, OW), "bool")
            zeros = T.alloc_ub((CH, OW), "float32")
            nanb = T.alloc_ub((CH, OW), "float32")
            infb = T.alloc_ub((CH, OW), "float32")
            pos_inf = T.infinity("float32")

            T.copy(x[0, 0:CH, 0:L], row)
            T.deinterleave(row, even, odd, channel_nums=2)
            # Q3/Q1: three static-offset taps, NaN folded in by vmax itself
            T.copy(even[0:CH, 0:OW], acc)
            T.vmax(odd[0:CH, 0:OW], acc, acc)          # NaN in operand 0
            T.vmax(acc, even[0:CH, 1:1 + OW], acc)     # NaN in operand 1
            T.copy(acc, accf)
            T.copy(accf, out[0, 0:CH, 0:OW])
            # Q2 phase dump
            T.copy(even[0:CH, 0:OW], out[1, 0:CH, 0:OW])
            T.copy(odd[0:CH, 0:OW], out[2, 0:CH, 0:OW])
            # Q4 eq-mask window min
            T.vcmp(row, row, mmask, "eq")
            T.vcast(mmask, mrow, round_mode="rint")
            T.deinterleave(mrow, meven, modd, channel_nums=2)
            T.copy(meven[0:CH, 0:OW], macc)
            T.vmin(modd[0:CH, 0:OW], macc, macc)
            T.vmin(macc, meven[0:CH, 1:1 + OW], macc)
            T.copy(macc, mf)
            T.copy(mf, out[3, 0:CH, 0:OW])
            # Q4b final pick: NaN where the mask hit 0
            T.clear(zeros)
            T.vcmp(macc, zeros, cond, "eq")
            T.vbrc(pos_inf, infb)
            T.vsub(infb, infb, nanb)
            T.vselect(cond, nanb, accf, accf)
            T.copy(accf, out[4, 0:CH, 0:OW])
            # Q5 the row itself (for the +-inf reference)
            T.copy(row[0:CH, 0:OW], out[5, 0:CH, 0:OW])

    return main


def run(dtype_str):
    td = torch.float16 if dtype_str == "float16" else torch.float32
    x = torch.randn(1, CH, L, dtype=torch.float32)
    x[0, 0, 5] = NAN            # interior NaN
    x[0, 1, 0] = NAN            # first element NaN
    x[0, 2, :] = NAN            # all-NaN channel
    x[0, 3, 7] = INF            # +inf
    x[0, 3, 8] = -INF           # -inf next to it -> +inf, never NaN
    x[0, 4, :] = -INF           # all -inf channel
    x = x.to(td)
    out = _probe(dtype_str)(x.npu()).cpu()
    got, even, odd, macc, picked, rowv = out[0], out[1], out[2], out[3], out[4], out[5]
    xf = x[0].float()
    ok = True

    def cmp(name, a, b):
        nonlocal ok
        d = (a != b) & ~(torch.isnan(a) & torch.isnan(b))
        good = not d.any()
        print(f"[{name}] {'PASS' if good else 'FAIL'} ({int(d.sum())} mism)")
        ok &= good

    cmp("Q2 even=src[0::2]", even, xf[:, 0::2][:, :OW])
    cmp("Q2 odd=src[1::2]", odd, xf[:, 1::2][:, :OW])
    # window of output ow = {2ow, 2ow+1, 2ow+2}
    win = torch.stack([xf[:, 0:OW], xf[:, 1:OW + 1], xf[:, 2:OW + 2]], dim=0)
    win_max = win.max(dim=0).values
    win_nan = torch.isnan(win).any(dim=0)
    # torch max propagates NaN; use the explicit rule instead
    ref = torch.where(win_nan, torch.full_like(win_max, NAN),
                      win.masked_fill(torch.isnan(win), -INF).max(dim=0).values)
    cmp("Q3 vmax static-offset taps", got, ref)
    cmp("Q1 vmax NaN propagation", torch.isnan(got), win_nan)
    cmp("Q4 eq-mask window min", macc, torch.where(win_nan, torch.zeros_like(macc),
                                                   torch.ones_like(macc)))
    cmp("Q4b vselect NaN pick", picked, ref)
    # +-inf: {+inf,-inf} -> +inf, all -inf -> -inf (no NaN anywhere else)
    ch = 3
    fin = ref[ch][~torch.isnan(ref[ch])]
    print(f"[Q5] ch3 (+inf/-inf mix) finite-no-NaN={bool(not torch.isnan(ref[ch]).any())} "
          f"has_inf={bool(torch.isinf(fin).any())}")
    ok &= bool(not torch.isnan(ref[ch]).any())
    return ok


if __name__ == "__main__":
    allok = True
    for dt in ("float16", "float32"):
        print(f"--- dtype {dt}")
        allok &= run(dt)
    print("ALL PASS" if allok else "SOME FAILED")
