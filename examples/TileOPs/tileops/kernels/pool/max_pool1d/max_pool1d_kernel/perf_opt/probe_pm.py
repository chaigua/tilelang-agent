"""probe_pm.py -- position-major "dilate-max + gather decimation" (PM-DMG) probe.

Root cause addressed (measured): the channel-major DMG variant built
`tensor.extract_slice` ops whose result tensor had a row stride different from
the source buffer (e.g. `(80,61)` out of `(80,64)`); bufferization then lowered
those into UB->GM->UB materialisations -> 22.9 ms of scalar time on sincnet.

PM-DMG keeps every tensor-region operand *contiguous* by working in a
position-major layout:

  src(ch, LSPAN_LOAD)  --contiguous GM load-->
  pos(LSPA, ch)        --T.transpose [1,0]-->   (both dims multiples of 8)
  pos32                --vcast-->
  M(JLEN, ch)          --max over kw of pos32[kw*d : kw*d+JLEN, :]--
                       (every slice is a full-width row range == contiguous)
  M2(ch, JLEN)         --T.transpose [1,0]-->
  acc(ch, bl)          --T.gather(M2, idx[owl] = (m+owl)*s)-->
  out store            --direct 2D copy, no output transpose--

NaN tracking: same dilate-min over the eq mask, gathered identically.
`m = (base - s0)/s` is 0 for unclamped blocks and a per-unit runtime scalar for
blocks whose span is clamped at the end of the line (cell/overhang cases).
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


def ru8(v):
    return ((v + 7) // 8) * 8


def plan_pm(c_in, l_in, k, s, p, d, ceil_mode, ch, bl, ch_limit=None):
    """Feasibility of PM-DMG for one (ch, bl); returns a plan dict or None."""
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    if p % s != 0:
        return None
    # loaded width: >= span, == l_in + p (mod s) so delta stays a multiple of s,
    # multiple of 8 so the UB transpose keeps its fast in-core form.
    span = (bl - 1) * s + (k - 1) * d + 1
    lo = span
    LSPAN_LOAD = None
    for cand in range(ru8(lo) if ru8(lo) >= lo else ru8(lo) + 8,
                       ru8(lo) + 8 * (s + 1) + 8, 8):
        if cand >= span and (l_in - cand) % s == 0:
            LSPAN_LOAD = cand
            break
    if LSPAN_LOAD is None or LSPAN_LOAD > l_in:
        return None
    num_oblk = (out_l + bl - 1) // bl
    dmin = 0
    for b in range(num_oblk):
        base = b * bl * s - p
        s0 = max(0, min(base, l_in - LSPAN_LOAD))
        delta = base - s0
        if delta % s != 0 or delta < 0:
            if delta < 0:
                return None            # needs a left extension; unsupported here
        bl_real = min(bl, out_l - b * bl)
        need_hi = min(l_in, base + (bl_real - 1) * s + (k - 1) * d + 1)
        lo_need = max(0, base)
        if s0 > lo_need or s0 + LSPAN_LOAD < need_hi:
            return None
        dmin = min(dmin, delta)
    if dmin < 0:
        return None
    m_max = 0
    for b in range(num_oblk):
        base = b * bl * s - p
        s0 = max(0, min(base, l_in - LSPAN_LOAD))
        m_max = max(m_max, (base - s0) // s)
    jmax = (m_max + bl - 1) * s + 1
    jlen = ru8(jmax)
    lspa = ru8(LSPAN_LOAD + (k - 1) * d)
    if lspa < jlen + (k - 1) * d:
        lspa = ru8(jlen + (k - 1) * d)
    if LSPAN_LOAD > lspa:
        return None
    return dict(out_l=out_l, span=span, LSPAN_LOAD=LSPAN_LOAD, LSPA=lspa,
                JMAX=jmax, JLEN=jlen, m_max=m_max, num_oblk=num_oblk,
                jwin=(bl - 1) * s + 1)


def pick_bl(c_in, l_in, k, s, p, d, ceil_mode, ch, bl_max=32, eb=2,
            is_f32=False):
    bl = min(bl_max, pool_output_dim(l_in, k, s, p, ceil_mode, d))
    while bl >= 1:
        pl = plan_pm(c_in, l_in, k, s, p, d, ceil_mode, ch, bl)
        if pl is not None and ub_bytes(pl, ch, bl, eb, is_f32) * 1.7 <= 196_608:
            return bl, pl
        bl -= 1
    return 0, None


def ub_bytes(pl, ch, bl, eb, is_f32):
    L = pl["LSPA"]
    J = pl["JLEN"]
    tot = 0
    tot += ch * pl["LSPAN_LOAD"] * eb          # sload
    tot += L * ch * eb                          # pos
    tot += L * ch * 4                           # pos32 (skipped when is_f32? still allocated)
    tot += L * ch                               # eqm (bool)
    tot += L * ch * 4                           # eq32
    tot += J * ch * 4 * 2                       # M, Mn
    tot += ch * J * 4 * 2                       # M2, Mn2
    tot += 6 * ch * bl * 4                      # acc..out32
    tot += ch * bl                              # cfinal bool
    tot += ch * bl * eb                         # outd
    tot += 2 * ch * bl * 4                      # idxtab, idxabs (int32)
    return tot


def build(case, ch=None, bl=None, bl_max=32, veccores=40, mode="dynidx",
          off_mode="dynidx"):
    n, c_in, l_in, k, s, p, d, ceil_mode, dtype = CASES[case]
    eb = elem_bytes(dtype)
    is_f32 = dtype == "float32"
    if ch is None:
        ch = ru8(c_in)
    if bl is None:
        bl, pl = pick_bl(c_in, l_in, k, s, p, d, ceil_mode, ch, bl_max, eb, is_f32)
    else:
        pl = plan_pm(c_in, l_in, k, s, p, d, ceil_mode, ch, bl)
    if pl is None:
        raise SystemExit(f"infeasible PM plan for {case} ch={ch} bl={bl}")

    out_l = pl["out_l"]
    LSPAN_LOAD = pl["LSPAN_LOAD"]
    LSPA = pl["LSPA"]
    JMAX = pl["JMAX"]
    JLEN = pl["JLEN"]
    num_oblk = pl["num_oblk"]
    num_cblk = (c_in + ch - 1) // ch
    num_logical = n * num_oblk * num_cblk
    num_kernels = min(num_logical, veccores)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    print(f"[plan] {case}: out_l={out_l} CH={ch} BL={bl} LSPAN_LOAD={LSPAN_LOAD} "
          f"LSPA={LSPA} JMAX={JMAX} JLEN={JLEN} m_max={pl['m_max']} "
          f"num_cblk={num_cblk} num_oblk={num_oblk} num_logical={num_logical} "
          f"num_local={num_local} UB={ub_bytes(pl, ch, bl, eb, is_f32)/1024:.1f}KiB"
          f" x1.7={ub_bytes(pl, ch, bl, eb, is_f32)*1.7/1024:.1f}KiB")

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(_ch):
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, l_in), dtype),
            out: T.Tensor((n, c_in, out_l), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                sload = T.alloc_ub((ch, LSPAN_LOAD), dtype)
                pos = T.alloc_ub((LSPA, ch), dtype)
                pos32 = T.alloc_ub((LSPA, ch), "float32")
                eqm = T.alloc_ub((LSPA, ch), "bool")
                eq32 = T.alloc_ub((LSPA, ch), "float32")
                M = T.alloc_ub((JLEN, ch), "float32")
                Mn = T.alloc_ub((JLEN, ch), "float32")
                M2 = T.alloc_ub((ch, JLEN), "float32")
                Mn2 = T.alloc_ub((ch, JLEN), "float32")
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
                idx = T.alloc_ub((ch, bl), "int32")

                neg_inf_d = -T.infinity(dtype)
                pos_inf = T.infinity("float32")
                one_val = T.float32(1.0)

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
                        m = (base - s0) // s
                        T.vbrc(neg_inf_d, pos)
                        T.copy(
                            x[nb, cb * ch:cb * ch + real_c, s0:s0 + LSPAN_LOAD],
                            sload[0:real_c, 0:LSPAN_LOAD],
                        )
                        T.transpose(sload, pos[0:LSPAN_LOAD, 0:ch],
                                    permutation=[1, 0])
                        if is_f32:
                            T.copy(pos, pos32)
                        else:
                            T.vcast(pos, pos32, round_mode="rint")
                        T.vcmp(pos32, pos32, eqm, "eq")
                        T.vcast(eqm, eq32, round_mode="rint")
                        T.copy(pos32[0:JLEN, 0:ch], M)
                        T.copy(eq32[0:JLEN, 0:ch], Mn)
                        for kw in range(1, k):
                            T.vmax(M, pos32[kw * d:kw * d + JLEN, 0:ch], M)
                            T.vmin(Mn, eq32[kw * d:kw * d + JLEN, 0:ch], Mn)
                        T.transpose(M, M2, permutation=[1, 0])
                        T.transpose(Mn, Mn2, permutation=[1, 0])
                        if pl["m_max"] == 0:
                            T.gather(M2, acc, idxtab)
                            T.gather(Mn2, nacc, idxtab)
                        else:
                            T.vbrc(T.Cast("int32", m * s), offbuf)
                            T.vadd(idxtab, offbuf, idx)
                            T.gather(M2, acc, idx)
                            T.gather(Mn2, nacc, idx)
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
    args = ap.parse_args()

    kernel, pl = build(args.case, ch=args.ch, bl=args.bl, bl_max=args.bl_max)
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
        print(f"[check] {args.case} BL={args.bl}: exact={exact} "
              f"max_diff={(got.float() - ref.float()).abs().max().item():.3e}")
        if not exact:
            bad = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref))
            idx_ = bad.nonzero()
            print(f"  mismatches={int(bad.sum())}/{bad.numel()} "
                  f"first={idx_[:5].tolist()}")
    else:
        print(f"[probe] {args.case}: {args.iters} iters, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
