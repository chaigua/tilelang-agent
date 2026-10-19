"""probe_dmg.py -- "dilate-max + gather decimation" (DMG) structural probe.

Problem solved: the general windowed path's inner loop
    for kw: for owl: vmax(acc[owl], row[j], acc[owl])
issues `k*bl` *narrow* (CH-wide) vector ops plus `k*bl` scalar index
computations.  DMG replaces it with

  1. a contiguous span load (as before),
  2. a full-span dilation max  M[j] = max_kw src[j + kw*d]   -- (k-1) *wide* ops,
     and the matching mask min  Mn[j] = min_kw eq[j + kw*d],
  3. **one** gather per accumulation lane group: out[owl] = M[(m+owl)*s], which
     decimates the dilated max to the output grid (`T.gather`, vgather),
  4. a direct (ch, bl) store -- no output transpose at all.

`m = (base - s0)/s` is a per-unit *runtime* scalar (nonzero only for blocks whose
span is clamped at the end of the line); it is folded into the gather source
region start.

This is a probe: correctness is asserted with --check, timing with msprof.
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: F401,E402

CASES = {
    "sincnet": (32, 80, 4096, 3, 3, 0, 1, False, "float16"),
    "ecg": (16, 128, 2048, 5, 2, 2, 2, True, "bfloat16"),
    "smoke": (1, 8, 32, 3, 2, 1, 1, False, "float16"),
    "textcnn": (64, 256, 128, 128, 1, 0, 1, False, "float16"),
}

_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}


def elem_bytes(dtype):
    return 4 if dtype == "float32" else 2


def pool_output_dim(l_in, k, s, p, ceil_mode, d=1):
    eff = d * (k - 1) + 1
    if ceil_mode:
        out = (l_in + 2 * p - eff + s - 1) // s + 1
    else:
        out = (l_in + 2 * p - eff) // s + 1
    if ceil_mode and out > 0 and (out - 1) * s >= l_in + p:
        out -= 1
    return max(out, 0)


def plan_dmg(c_in, l_in, k, s, p, d, ceil_mode, ch, bl):
    """Host-side feasibility check for one (ch, bl) of the DMG structure."""
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    if p % s != 0:
        return None
    # left pad: columns reserved for out-of-range taps left of the span start.
    # needs LEAD >= p, LEAD % 16 == 0 (32 B alignment for fp16/fp32 rows) and
    # LEAD % s == p % s (so that (delta + LEAD) stays a multiple of s).
    LEAD = 0
    if p > 0:
        LEAD = None
        for cand in range(16, 16 * (s + 4), 16):
            if cand >= p and cand % s == p % s:
                LEAD = cand
                break
        if LEAD is None:
            return None
    span = (bl - 1) * s + (k - 1) * d + 1
    LSPAN_LOAD = span + ((l_in - span) % s)          # >= span, keeps delta % s == 0
    if LSPAN_LOAD > l_in:
        return None
    num_oblk = (out_l + bl - 1) // bl
    m_max = None
    for b in range(num_oblk):
        base = b * bl * s - p
        s0 = max(0, min(base, l_in - LSPAN_LOAD))
        delta = base - s0
        if delta < 0 and delta != -p:
            return None
        if delta % s != 0:
            return None
        m = delta // s
        m_max = m if m_max is None else max(m_max, m)
        bl_real = min(bl, out_l - b * bl)
        need_hi = min(l_in, base + (bl_real - 1) * s + (k - 1) * d + 1)
        lo = max(0, base)
        if s0 > lo or s0 + LSPAN_LOAD < need_hi:
            return None
    jmax = (m_max + LEAD // s + bl - 1) * s + 1
    ne = jmax + (k - 1) * d
    lspan_alloc = max(ne, LEAD + LSPAN_LOAD)
    return dict(out_l=out_l, span=span, LSPAN_LOAD=LSPAN_LOAD, LEAD=LEAD,
                num_oblk=num_oblk, m_max=m_max, JMAX=jmax, NE=ne,
                LSPAN_alloc=lspan_alloc, jwin=(bl - 1) * s + 1,
                off_max=(m_max + LEAD // s) * s)


def pick_bl(c_in, l_in, k, s, p, d, ceil_mode, ch, bl_max, budget_bytes=115_000):
    bl = bl_max
    while bl >= 1:
        pl = plan_dmg(c_in, l_in, k, s, p, d, ceil_mode, ch, bl)
        if pl is not None:
            return bl, pl
        bl -= 1
    return 0, None


def build(case, ch=None, bl=None, bl_max=32, veccores=40, mode="dynregion",
          fill=True):
    n, c_in, l_in, k, s, p, d, ceil_mode, dtype = CASES[case]
    eb = elem_bytes(dtype)
    is_f32 = dtype == "float32"
    if ch is None:
        ch = c_in
    if bl is None:
        bl, pl = pick_bl(c_in, l_in, k, s, p, d, ceil_mode, ch, bl_max)
    else:
        pl = plan_dmg(c_in, l_in, k, s, p, d, ceil_mode, ch, bl)
    if pl is None:
        raise SystemExit(f"infeasible DMG plan for {case} ch={ch} bl={bl}")

    out_l = pl["out_l"]
    LEAD = pl["LEAD"]
    LSPAN_LOAD = pl["LSPAN_LOAD"]
    LSPA = pl["LSPAN_alloc"]
    JMAX = pl["JMAX"]
    JWIN = pl["jwin"]
    num_oblk = pl["num_oblk"]
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * num_oblk * num_cblk
    num_kernels = min(num_logical, veccores)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    taps = range(1, k)
    print(f"[plan] {case}: out_l={out_l} CH={ch} BL={bl} LEAD={LEAD} "
          f"LSPAN_LOAD={LSPAN_LOAD} LSPAN_alloc={LSPA} JMAX={JMAX} JWIN={JWIN} "
          f"m_max={pl['m_max']} num_cblk={num_cblk} num_oblk={num_oblk} "
          f"num_logical={num_logical} num_local={num_local}")

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_ch):
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, l_in), dtype),
            out: T.Tensor((n, c_in, out_l), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                src = T.alloc_ub((ch, LSPA), dtype)
                src32 = T.alloc_ub((ch, LSPA), "float32")
                eqm = T.alloc_ub((ch, LSPA), "bool")
                eq32 = T.alloc_ub((ch, LSPA), "float32")
                M = T.alloc_ub((ch, JMAX), "float32")
                Mn = T.alloc_ub((ch, JMAX), "float32")
                J = T.alloc_ub((ch, JWIN), "float32")
                Jn = T.alloc_ub((ch, JWIN), "float32")
                acc = T.alloc_ub((ch, bl), "float32")
                nacc = T.alloc_ub((ch, bl), "float32")
                zeros = T.alloc_ub((ch, bl), "float32")
                infbuf = T.alloc_ub((ch, bl), "float32")
                nanbuf = T.alloc_ub((ch, bl), "float32")
                out32 = T.alloc_ub((ch, bl), "float32")
                cfinal = T.alloc_ub((ch, bl), "bool")
                outd = T.alloc_ub((ch, bl), dtype)
                idxrow = T.alloc_ub((1, bl), "int32")
                idxtab = T.alloc_ub((ch, bl), "int32")
                offbuf = T.alloc_ub((ch, bl), "int32")
                idxabs = T.alloc_ub((ch, bl), "int32")

                neg_inf = -T.infinity("float32")
                neg_inf_d = -T.infinity(dtype)
                pos_inf = T.infinity("float32")
                one_val = T.float32(1.0)
                zero_i = T.int32(0)

                # Static decimation index table: idx[owl] = owl * s (built once).
                for o in T.serial(bl):
                    idxrow[0, o] = T.Cast("int32", o * s)
                T.vbrc(idxrow, idxtab)
                T.clear(zeros)

                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < num_logical:
                        nb = uid // (num_oblk * num_cblk)
                        rem = uid % (num_oblk * num_cblk)
                        oblk = rem // num_cblk
                        cb = rem % num_cblk
                        real_c = T.min(ch, c_in - cb * ch)
                        ow0 = oblk * bl
                        base = ow0 * s - p
                        s0 = T.max(0, T.min(base, l_in - LSPAN_LOAD))
                        off = base - s0                       # multiple of s
                        # neutral fill: everything outside the loaded span is
                        # -inf (== out-of-range tap for a max reduction).
                        if fill:
                            T.vbrc(neg_inf_d, src)
                        T.copy(
                            x[nb, cb * ch:cb * ch + real_c, s0:s0 + LSPAN_LOAD],
                            src[0:real_c, LEAD:LEAD + LSPAN_LOAD],
                        )
                        if is_f32:
                            T.copy(src, src32)
                        else:
                            T.vcast(src, src32, round_mode="rint")
                        T.vcmp(src32, src32, eqm, "eq")
                        T.vcast(eqm, eq32, round_mode="rint")
                        T.copy(src32[0:ch, 0:JMAX], M)
                        T.copy(eq32[0:ch, 0:JMAX], Mn)
                        for kw in range(1, k):
                            T.vmax(M, src32[0:ch, kw * d:kw * d + JMAX], M)
                            T.vmin(Mn, eq32[0:ch, kw * d:kw * d + JMAX], Mn)
                        if mode == "dynregion":
                            T.gather(M[0:ch, off:off + JWIN], acc, idxtab)
                            T.gather(Mn[0:ch, off:off + JWIN], nacc, idxtab)
                        elif mode == "dynidx":
                            T.vbrc(T.Cast("int32", off), offbuf)
                            T.vadd(idxtab, offbuf, idxabs)
                            T.gather(M, acc, idxabs)
                            T.gather(Mn, nacc, idxabs)
                        elif mode == "static":
                            T.gather(M[0:ch, 0:JWIN], acc, idxtab)
                            T.gather(Mn[0:ch, 0:JWIN], nacc, idxtab)
                        elif mode == "nogather":
                            T.copy(M[0:ch, 0:bl], acc)
                            T.copy(Mn[0:ch, 0:bl], nacc)
                        else:
                            raise ValueError(mode)
                        T.vcmp(nacc, zeros, cfinal, "eq")
                        T.vbrc(pos_inf, infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vselect(cfinal, nanbuf, acc, out32)
                        if is_f32:
                            T.copy(out32, outd)
                        else:
                            T.vcast(out32, outd, round_mode="rint")
                        bl_real = T.min(bl, out_l - ow0)
                        T.copy(
                            outd[0:real_c, 0:bl_real],
                            out[nb, cb * ch:cb * ch + real_c, ow0:ow0 + bl_real],
                        )

        return main

    return _func(ch), pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="sincnet", choices=sorted(CASES))
    ap.add_argument("--ch", type=int, default=None)
    ap.add_argument("--bl", type=int, default=None)
    ap.add_argument("--bl-max", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mode", default="dynregion",
                    choices=["dynregion", "dynidx", "static", "nogather"])
    ap.add_argument("--no-fill", action="store_true")
    args = ap.parse_args()

    kernel, pl = build(args.case, ch=args.ch, bl=args.bl, bl_max=args.bl_max,
                       mode=args.mode, fill=not args.no_fill)
    n, c_in, l_in, k, s, p, d, ceil_mode, dtype = CASES[args.case]
    tdtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    x = torch.randn(n, c_in, l_in, dtype=torch.float32).to(tdtype).npu()
    with torch.no_grad():
        out = None
        for _ in range(args.iters):
            out = kernel(x)
    if args.check:
        ref = F.max_pool1d(x.cpu().detach().float(), k, s, p, d,
                           ceil_mode).to(tdtype)
        got = out.cpu()
        exact = bool(torch.equal(got, ref))
        print(f"[check] {args.case} CH={pl['num_oblk'] and args.ch} BL={args.bl}: "
              f"exact={exact} "
              f"max_diff={(got.float() - ref.float()).abs().max().item():.3e}")
        if not exact:
            bad = (got != ref)
            idx = bad.nonzero()
            print(f"  mismatches={int(bad.sum())}/{bad.numel()} first={idx[:5].tolist()}")
    else:
        print(f"[probe] {args.case}: {args.iters} iters, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
