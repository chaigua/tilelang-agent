"""Vector-op cost calibration probe (Stage 4, non-tune data).

Measures the per-op cost of the npuir vector ops used by the
max_pool1d_with_indices kernel on (m, ch) UB planes, with a dependent
accumulator chain (realistic for a pooling window scan).  Results land in
`op_cost_calibration.jsonl` (NOT perf_records.jsonl).

    msprof op --kernel-name=main ... python probe_ops.py --op vmax  --m 1 --ch 64 --reps 512
    msprof op ... python probe_ops.py --op floor --m 1 --ch 64 --reps 16

Cost model: cost_per_op_us = (T(reps) - T(floor)) / (reps * 40 cores).
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

CORES = 40


def build(op, m, ch, reps, dtype="float32"):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((CORES * m, ch), dtype),
            y: T.Tensor((CORES * m, ch), dtype),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                a = T.alloc_ub((m, ch), dtype)
                b = T.alloc_ub((m, ch), dtype)
                c = T.alloc_ub((m, ch), dtype)
                d = T.alloc_ub((m, ch), dtype)
                msk = T.alloc_ub((m, ch), "bool")
                msk2 = T.alloc_ub((m, ch), "bool")
                f16 = T.alloc_ub((m, ch), "float16")

                T.copy(x[cid * m:(cid + 1) * m, 0:ch], a)
                T.copy(a, b)
                T.copy(a, c)
                T.copy(a, d)
                T.vcmp(a, b, msk, "gt")
                T.vcmp(a, b, msk2, "ge")
                zero = T.float32(-1.0e30)
                one = T.float32(1.0e30)

                for r in T.serial(1):
                    for i in T.serial(reps):
                        if op == "vmax":
                            T.vmax(c, b, c)
                        elif op == "vmin":
                            T.vmin(c, b, c)
                        elif op == "vsub":
                            T.vsub(c, b, c)
                        elif op == "vmul":
                            T.vmul(c, b, c)
                        elif op == "vcmp":
                            T.vcmp(a, b, msk, "gt")
                        elif op == "vselect":
                            T.vselect(msk, a, c, c)
                        elif op == "vclamp":
                            T.vclamp(a, c, zero, one)
                        elif op == "vabs":
                            T.vabs(a, c)
                        elif op == "vand":
                            T.vand(msk, msk2, msk)
                        elif op == "vor":
                            T.vor(msk, msk2, msk)
                        elif op == "vxor":
                            T.vxor(a, b, c)
                        elif op == "vshr":
                            T.vshr(a, T.int32(31), c)
                        elif op == "vshl":
                            T.vshl(a, T.int32(1), c)
                        elif op == "vmul_i":
                            T.vmul(c, a, c)
                        elif op == "vcast":
                            T.vcast(a, f16, round_mode="rint")
                        elif op == "vbrc":
                            T.vbrc(T.float32(3.0), c)
                        elif op == "vrec":
                            T.vrec(a, c)
                        elif op == "floor":
                            T.vmax(c, b, c)
                        else:
                            raise SystemExit(f"unknown op {op}")
                # keep every buffer live
                T.vselect(msk, a, c, d)
                T.vselect(msk2, a, d, d)
                T.vcast(f16, c, round_mode="rint")
                T.copy(d, y[cid * m:(cid + 1) * m, 0:ch])

        return main

    return _f()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=512)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    k = build(args.op, args.m, args.ch, args.reps, args.dtype)
    x = torch.randn(CORES * args.m, args.ch, dtype=torch.float32)
    if args.dtype in ("float16", "bfloat16"):
        x = x.to(getattr(torch, args.dtype))
    xn = x.npu()
    with torch.no_grad():
        for _ in range(args.iters):
            y = k(xn)
    print(f"[probe] op={args.op} m={args.m} ch={args.ch} reps={args.reps} "
          f"dtype={args.dtype} ops_per_core={args.reps}")


if __name__ == "__main__":
    main()
