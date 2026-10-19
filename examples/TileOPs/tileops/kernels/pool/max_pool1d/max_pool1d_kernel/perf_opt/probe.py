"""probe.py -- structural A/B probes for the max_pool1d general path (Stage 4).

Not a deliverable candidate: this file exists to *measure* structure costs with
`msprof op` on the real manifest shapes.  Variants flagged `timing-only` produce
**wrong results on purpose** (they drop a layout transform) and are used only to
quantify that transform's cost; `--check` is therefore opt-in.

Usage:
  python probe.py --case sincnet --variant base
  python probe.py --case sincnet --variant ch_max128 --check
  python probe.py --case sincnet --variant drop_out_transpose      # timing only

Cases mirror perf_opt/bench.py CASES.
"""

import argparse
import functools
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: F401,E402

from tilelang.utils.npu_utils import NPUUtils  # noqa: E402

_UB_CAPACITY_BYTES = 196_608
_UB_MULTIBUFFER_FACTOR = 1.7
_VECTOR_WIDTH = 8

CASES = {
    "sincnet": (32, 80, 4096, 3, 3, 0, 1, False, "float16"),
    "ecg": (16, 128, 2048, 5, 2, 2, 2, True, "bfloat16"),
    "textcnn": (64, 256, 128, 128, 1, 0, 1, False, "float16"),
    "smoke": (1, 8, 32, 3, 2, 1, 1, False, "float16"),
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


def round_up(v, m):
    return ((v + m - 1) // m) * m


def span_of(bl, k, s, d, l_in):
    return min((bl - 1) * s + (k - 1) * d + 1, l_in)


def general_bytes(ch, lspan, bl, eb):
    return (2 * ch * lspan * eb + 2 * lspan * ch * 4 + lspan * ch
            + 6 * bl * ch * 4 + bl * ch + 2 * bl * ch * eb)


def choose_bl(ch, l_in, out_l, k, s, d, eb, bl_max, bl_align):
    bl = min(out_l, bl_max)
    if bl_align > 1:
        bl = (bl // bl_align) * bl_align
    while bl >= 1:
        lspan = span_of(bl, k, s, d, l_in)
        if general_bytes(ch, lspan, bl, eb) * _UB_MULTIBUFFER_FACTOR <= _UB_CAPACITY_BYTES:
            return bl
        step = bl_align if (bl_align > 1 and bl > bl_align) else 1
        bl -= step
    return 0


def choose_ch_bl(c_in, l_in, out_l, k, s, d, eb, ch_max, bl_max, bl_align):
    ch = max(_VECTOR_WIDTH, round_up(min(c_in, ch_max), _VECTOR_WIDTH))
    while True:
        bl = choose_bl(ch, l_in, out_l, k, s, d, eb, bl_max, bl_align)
        if bl >= 1:
            return ch, bl
        if ch == _VECTOR_WIDTH:
            return ch, 0
        ch = max(_VECTOR_WIDTH, round_up(ch // 2, _VECTOR_WIDTH))


def build(case, ch_max, bl_max, bl_align, drop_row_transpose,
          drop_out_transpose, num_cores_cap, out_idx=True, veccores=40):
    n, c_in, l_in, k, s, p, d, ceil_mode, dtype = CASES[case]
    eb = elem_bytes(dtype)
    out_l = pool_output_dim(l_in, k, s, p, ceil_mode, d)
    ch, bl = choose_ch_bl(c_in, l_in, out_l, k, s, d, eb, ch_max, bl_max, bl_align)
    if bl < 1:
        raise SystemExit("no feasible plan")
    lspan = span_of(bl, k, s, d, l_in)
    num_cblk = (c_in + ch - 1) // ch
    num_oblk = (out_l + bl - 1) // bl
    num_logical = n * num_oblk * num_cblk
    num_kernels = min(num_logical, veccores if num_cores_cap is None else num_cores_cap)
    num_local = (num_logical + num_kernels - 1) // num_kernels
    is_f32 = dtype == "float32"
    print(f"[plan] {case}: out_l={out_l} CH={ch} BL={bl} LSPAN={lspan} "
          f"num_cblk={num_cblk} num_oblk={num_oblk} num_logical={num_logical} "
          f"num_kernels={num_kernels} num_local={num_local} "
          f"UB={general_bytes(ch, lspan, bl, eb)}B x1.7="
          f"{general_bytes(ch, lspan, bl, eb) * 1.7:.0f}B")

    @tilelang.jit(out_idx=[1] if out_idx else [], target="npuir")
    def _func(_ch):
        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, l_in), dtype),
            out: T.Tensor((n, c_in, out_l), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                acc = T.alloc_ub((bl, ch), "float32")
                flag = T.alloc_ub((bl, ch), "float32")
                zeros = T.alloc_ub((bl, ch), "float32")
                infbuf = T.alloc_ub((bl, ch), "float32")
                nanbuf = T.alloc_ub((bl, ch), "float32")
                out32 = T.alloc_ub((bl, ch), "float32")
                cfinal = T.alloc_ub((bl, ch), "bool")
                row_src = T.alloc_ub((ch, lspan), dtype)
                row_dst = T.alloc_ub((lspan, ch), dtype)
                row_f32 = T.alloc_ub((lspan, ch), "float32")
                eq_mask = T.alloc_ub((lspan, ch), "bool")
                eq_f32 = T.alloc_ub((lspan, ch), "float32")
                acc16 = T.alloc_ub((bl, ch), dtype)
                acc_t = T.alloc_ub((ch, bl), dtype)

                neg_inf = -T.infinity("float32")
                pos_inf = T.infinity("float32")
                one_val = T.float32(1.0)

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
                        s0 = T.max(0, T.min(base, l_in - lspan))
                        T.vbrc(neg_inf, acc)
                        T.vbrc(one_val, flag)
                        T.clear(zeros)
                        T.copy(
                            x[nb, cb * ch:cb * ch + real_c, s0:s0 + lspan],
                            row_src[0:real_c, 0:lspan],
                        )
                        if drop_row_transpose:
                            T.copy(row_src, row_dst)
                        else:
                            T.transpose(row_src, row_dst, permutation=[1, 0])
                        if is_f32:
                            T.copy(row_dst, row_f32)
                        else:
                            T.vcast(row_dst, row_f32, round_mode="rint")
                        T.vcmp(row_f32, row_f32, eq_mask, "eq")
                        T.vcast(eq_mask, eq_f32, round_mode="rint")
                        for kw in T.serial(k):
                            for owl in T.serial(bl):
                                iw = base + owl * s + kw * d
                                j = iw - s0
                                if 0 <= iw and iw < l_in:
                                    T.vmax(
                                        acc[owl:owl + 1, 0:ch],
                                        row_f32[j:j + 1, 0:ch],
                                        acc[owl:owl + 1, 0:ch],
                                    )
                                    T.vmul(
                                        flag[owl:owl + 1, 0:ch],
                                        eq_f32[j:j + 1, 0:ch],
                                        flag[owl:owl + 1, 0:ch],
                                    )
                        T.vcmp(flag, zeros, cfinal, "eq")
                        T.vbrc(pos_inf, infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)
                        T.vselect(cfinal, nanbuf, acc, out32)
                        if is_f32:
                            T.copy(out32, acc16)
                        else:
                            T.vcast(out32, acc16, round_mode="rint")
                        bl_real = T.min(bl, out_l - ow0)
                        if drop_out_transpose:
                            T.copy(acc16[0:1, 0:ch], acc_t[0:1, 0:ch])
                        else:
                            T.transpose(acc16, acc_t, permutation=[1, 0])
                        T.copy(
                            acc_t[0:real_c, 0:bl_real],
                            out[nb, cb * ch:cb * ch + real_c, ow0:ow0 + bl_real],
                        )

        return main

    return _func(ch), dict(out_l=out_l, ch=ch, bl=bl, lspan=lspan,
                           num_cblk=num_cblk, num_oblk=num_oblk,
                           num_logical=num_logical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="sincnet", choices=sorted(CASES))
    ap.add_argument("--variant", default="base")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    kw = dict(ch_max=64, bl_max=32, bl_align=1, drop_row_transpose=False,
              drop_out_transpose=False, num_cores_cap=None)
    if args.variant == "base":
        pass
    elif args.variant == "ch_max128":
        kw["ch_max"] = 128
    elif args.variant == "ch_max256":
        kw["ch_max"] = 256
    elif args.variant == "bl_align8":
        kw["bl_align"] = 8
    elif args.variant == "bl_max8":
        kw["bl_max"] = 8
    elif args.variant == "bl_max64":
        kw["bl_max"] = 64
    elif args.variant == "ch_max128_bl_max64":
        kw["ch_max"], kw["bl_max"] = 128, 64
    elif args.variant == "drop_row_transpose":
        kw["drop_row_transpose"] = True
    elif args.variant == "drop_out_transpose":
        kw["drop_out_transpose"] = True
    elif args.variant == "drop_both_transpose":
        kw["drop_row_transpose"] = True
        kw["drop_out_transpose"] = True
    elif args.variant == "cores20":
        kw["num_cores_cap"] = 20
    else:
        raise SystemExit(f"unknown variant {args.variant!r}")

    kernel, plan = build(args.case, **kw)
    n, c_in, l_in, k, s, p, d, ceil_mode, dtype = CASES[args.case]
    tdtype = _TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    x = torch.randn(n, c_in, l_in, dtype=torch.float32).to(tdtype).npu()
    with torch.no_grad():
        out = None
        for _ in range(args.iters):
            out = kernel(x)
    if args.check:
        ref = F.max_pool1d(x.cpu().detach().float(), k, s, p, d, ceil_mode).to(tdtype)
        got = out.cpu()
        exact = bool(torch.equal(got, ref))
        print(f"[check] {args.case}/{args.variant}: exact={exact} "
              f"max_diff={(got.float() - ref.float()).abs().max().item():.3e}")
    else:
        print(f"[probe] {args.case}/{args.variant}: {args.iters} iters, "
              f"out={tuple(out.shape)} plan={plan}")


if __name__ == "__main__":
    main()
