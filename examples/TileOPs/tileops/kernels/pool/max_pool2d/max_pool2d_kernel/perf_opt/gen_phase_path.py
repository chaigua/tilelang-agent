"""Generate the phase-decimated path body for perf_opt/_max_pool2d_kernel.py.

The taps must be literal statements inside ``@T.prim_func`` (a Python ``for``
becomes a TIR loop there, so per-tap compile-time constants cannot be produced
by loops), therefore the (k_h x k_w) tap block is emitted as source text with
the constants baked in.

Usage: python gen_phase_path.py  (rewrites the marked section in place)
"""

import re
import sys

HEADER = '''        if path == "phase":
            ph = phase
            bh = ph["bh"]
            rows = ph["rows"]
            row_elems = ph["row_elems"]
            data_elems = ph["data_elems"]
            half = row_elems // 2
            base_a = ph["base_a"]
            n_bad = len(ph["bad_cols"])
            tap_lo = ph["tap_lo"]
            tap_hi = ph["tap_hi"]
            n_ohb = out_h // bh
            static_bounds = ph["static_bounds"]
            assert not ph["bad_cols"], "phase path has no row-crossing column"
            assert bh * out_w <= row_elems, "phase store staging does not fit"

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_phase(ch_):
                @T.prim_func
                def main(
                    x: T.Tensor((n, c_in, h_in * w_in), dtype),
                    out: T.Tensor((n, c_in, out_h * out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        rbuf = T.alloc_ub((ch, ncol), dtype)
                        acc = T.alloc_ub((ch, bh * out_w), dtype)
                        isok = T.alloc_ub((ch, bh * out_w), "bool")
                        nanb = T.alloc_ub((ch, bh * out_w), dtype)
                        keep = T.alloc_ub((ch, bh * out_w), dtype)
                        neg_inf = _ninf
                        pos_inf = _pinf
                        for t in T.serial(num_local):
                            uid = t * num_kernels + cid
                            if uid < num_logical:
                                nb = uid // (n_ohb * num_cblk)
                                rem = uid % (n_ohb * num_cblk)
                                ohb = rem // num_cblk
                                cb = rem % num_cblk
                                real_c = T.min(ch, c_in - cb * ch)
                                oh0 = ohb * bh
                                ih0 = oh0 * s_h - p_h
                                # a no-op clamp when the span is statically in
                                # bounds; it keeps the DMA range valid when the
                                # window span crosses the top/bottom border
                                start = T.max(0, T.min(ih0, h_in - rows))
                                # the -inf end padding is re-armed per unit
                                T.vbrc(neg_inf, rbuf)
                                T.copy(
                                    x[nb, cb * ch:cb * ch + real_c,
                                      start * w_in:start * w_in + data_elems],
                                    rbuf[0:real_c, 0:data_elems],
                                )
                                # one deinterleave decimates the row tile by
                                # s_w == 2: even half = iw 0::2, odd = 1::2
                                T.deinterleave(rbuf[0:ch, 0:row_elems],
                                               rbuf[0:ch, 0:half],
                                               rbuf[0:ch, half:2 * half],
                                               channel_nums=2)
                                T.vbrc(neg_inf, acc)
                                for o_r in T.serial(bh):
'''

FOOTER = '''                                # NaN wins: vmax propagates NaN on this
                                # toolchain (probe_prims.py Q1, f16/bf16/f32)
                                T.vcmp(acc, acc, isok, "eq")
                                T.vbrc(pos_inf, nanb)
                                T.vsub(nanb, nanb, nanb)
                                T.vselect(isok, acc, nanb, keep)
                                T.copy(
                                    keep[0:real_c, 0:bh * out_w],
                                    out[nb, cb * ch:cb * ch + real_c,
                                        oh0 * out_w:oh0 * out_w + bh * out_w],
                                )

                return main

            return _func_phase(ch)'''


def check_applicable(k_h, k_w, s_h, s_w, p_w, d_w, w_in, out_w):
    """Same predicate as _phase_plan; returns the (phase, c0, c1, lo, hi) taps."""
    assert s_w == 2 and d_w == 1 and s_h % 2 == 0
    q_w = w_in // 2
    taps = {}
    for kh in range(k_h):
        for kw in range(k_w):
            c = kw * d_w - p_w
            lo = max(0, -((-c) // s_w))
            hi = min(out_w, (w_in - 1 - c) // s_w + 1)
            if hi <= lo:
                continue
            if w_in % 2 == 0:
                phase = c % 2
                c0 = c // 2
                c1 = 0
            else:
                phase = (kh * d_h_global + c) % 2
                c0 = (kh * d_h_global + c - phase) // 2 + q_w * kh * d_h_global
                c1 = s_h // 2 + q_w * s_h
            taps[(kh, kw)] = (phase, c0, c1, lo, hi)
    return taps


def emit(k_h, k_w, s_h, w_in, out_w, d_h, p_w):
    q_w = w_in // 2
    odd_w = w_in % 2 == 1
    out = []
    for kh in range(k_h):
        ind = "                                    "
        out.append(f"{ind}if k_h >= {kh + 1}:")
        ind2 = ind + "    "
        out.append(f"{ind2}ih = ih0 + o_r * s_h"
                   + (f" + {kh * d_h}" if kh else "") + " * 1"
                   if False else
                   f"{ind2}ih = ih0 + o_r * s_h"
                   + (f" + {kh * d_h}" if kh else ""))
        out.append(f"{ind2}if 0 <= ih and ih < h_in:")
        ind3 = ind2 + "    "
        for kw in range(k_w):
            c = kw * d_w_global - p_w
            lo = max(0, -((-c) // 2))
            hi = min(out_w, (w_in - 1 - c) // 2 + 1)
            if hi <= lo:
                continue
            if not odd_w:
                phase = c % 2
                base = f"(ih - start) * {q_w} + ({c // 2})"
            else:
                phase = (kh * d_h_global + c) % 2
                c0 = ((kh * d_h_global + c - phase) // 2
                      + q_w * kh * d_h_global)
                c1 = s_h // 2 + q_w * s_h
                base = f"({c0}) + o_r * ({c1})"
            poff = f"{phase} * half + {base}"
            ind4 = ind3
            out.append(
                f"{ind4}T.vmax(\n"
                f"{ind4}    acc[0:ch, o_r * out_w + {lo}:o_r * out_w + {hi}],\n"
                f"{ind4}    rbuf[0:ch, {poff} + {lo}:{poff} + {hi}],\n"
                f"{ind4}    acc[0:ch, o_r * out_w + {lo}:o_r * out_w + {hi}],\n"
                f"{ind4})"
            )
    return "\n".join(out) + "\n"


def main():
    path = "_max_pool2d_kernel.py"
    src = open(path).read()
    start_mark = '        if path == "phase":\n'
    end_mark = "            return _func_phase(ch)"
    i = src.index(start_mark)
    j = src.index(end_mark, i) + len(end_mark)
    body = HEADER + emit(k_h_global, k_w_global, s_h_global, w_in_global,
                         out_w_global, d_h_global, p_w_global) + FOOTER
    src = src[:i] + body + src[j:]
    open(path, "w").write(src)
    print("phase path regenerated")


# the shape of the emitted section is generic: every tap is a literal statement
# whose constants come from these module-level knobs (set from the CLI)
k_h_global = 3
k_w_global = 3
s_h_global = 2
w_in_global = 112
out_w_global = 56
d_h_global = 1
d_w_global = 1
p_w_global = 1

if __name__ == "__main__":
    main()
