"""Second vector-op cost probe: loop-dependent (non-hoistable) bodies.

The first probe (`probe_ops.py`) used loop-invariant bodies, so `vselect`/`vcmp`
were partly constant-folded/hoisted.  This probe makes every body depend on the
loop variable so LICM/DCE cannot fire:

    msprof op ... python probe_ops2.py --op vbrc_i --m 1 --ch 64 --reps 512

Measured quantities (see perf_opt/op_cost_calibration.jsonl):
  * `vbrc_i`    : broadcast of a *runtime* scalar (loop var) into a (m,ch) buffer
  * `vcmp_gt`   : vcmp against a buffer that changes every iteration
  * `vsel_gt`   : vselect whose cond comes from that vcmp
  * `vmax_chain`: reference dependent vmax chain
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

CORES = 40


def build(op, m, ch, reps):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((CORES * m, ch), "float32"),
            y: T.Tensor((CORES * m, ch), "float32"),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                a = T.alloc_ub((m, ch), "float32")
                b = T.alloc_ub((m, ch), "float32")
                c = T.alloc_ub((m, ch), "float32")
                d = T.alloc_ub((m, ch), "float32")
                iv = T.alloc_ub((m, ch), "int32")
                iv2 = T.alloc_ub((m, ch), "int32")
                msk = T.alloc_ub((m, ch), "bool")

                T.copy(x[cid * m:(cid + 1) * m, 0:ch], a)
                T.copy(a, b)
                T.copy(a, c)
                T.copy(a, d)
                zero = T.float32(0.0)

                for i in T.serial(reps):
                    if op == "vbrc_i":
                        # runtime scalar broadcast (loop-dependent)
                        T.vbrc(T.int32(i), iv)
                        T.vbrc(T.float32(1.0), c)
                        T.vmax(c, a, d)
                    elif op == "vbrc_f":
                        T.vbrc(T.float32(1.0), c)
                        T.vmax(c, a, c)
                    elif op == "vcmp_gt":
                        # data-dependent compare: b changes with i
                        T.vadd(b, a, b)          # keep b varying
                        T.vcmp(a, b, msk, "gt")
                        T.vmax(c, a, c)
                    elif op == "vsel_gt":
                        T.vadd(b, a, b)
                        T.vcmp(a, b, msk, "gt")
                        T.vselect(msk, a, c, c)  # cond depends on data
                    elif op == "vsel_int":
                        T.vadd(iv2, iv, iv2)
                        T.vcmp(iv2, iv, msk, "gt")
                        T.vselect(msk, iv, iv2, iv2)
                    elif op == "vcmp_int":
                        T.vadd(iv2, iv, iv2)
                        T.vcmp(iv2, iv, msk, "gt")
                    elif op == "vmax_chain":
                        T.vmax(c, a, c)
                    elif op == "vsub_chain":
                        T.vsub(c, a, c)
                    else:
                        raise SystemExit(f"unknown op {op}")
                T.vselect(msk, a, c, d)
                T.copy(d, y[cid * m:(cid + 1) * m, 0:ch])

        return main

    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    k = build(args.op, args.m, args.ch, args.reps)
    x = torch.randn(CORES * args.m, args.ch, dtype=torch.float32).npu()
    with torch.no_grad():
        for _ in range(args.iters):
            y = k(x)
    print(f"[probe2] op={args.op} m={args.m} ch={args.ch} reps={args.reps}")


if __name__ == "__main__":
    main()
