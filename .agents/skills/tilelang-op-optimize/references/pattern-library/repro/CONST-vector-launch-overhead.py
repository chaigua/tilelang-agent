"""Vector-op cost calibration probe (Stage 4, non-tune data).

Entry ID: CONST-vector-launch-overhead   (Home file:
.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md)
Second entry sharing this script: TRAP-int32-vcmp-scalarized (traps-compiler.md)
Phenomenon: narrow vector-op unit price by lane width -- (1,64) 12.3 ns/op,
(8,64) 21.1 ns/op, (8,128) 23.1 ns/op (8-16x lanes cost only 1.7-1.9x); and
int32 `T.vcmp` is scalarised (64 lane = 1039 ns/op = 16.2 ns/lane, 34x the
fp32 vcmp, ~85x a f32 vmax).
First verified: tilelang 0.1.2+e8b5d51e3d098422c3970f4e714fb5dba1493609
(commit e8b5d51 / dev root build) + torch 2.9.0+cpu / torch_npu 2.9.0.post8 +
CANN 9.0.0 + Ascend910B4 (aicore 20 -> AIV 40 / UB 196608 B) / 2026-10-18.
origin_task: max_pool1d_with_indices-_max_pool1d_with_indices_kernel-20261017T235029Z
Assertion note (measurement-type repro, no inline assert -- source is a
calibration script, per CONST-copy-floor-dma-granularity precedent):
  * relative relations (read off the printed Task durations): (1,64) vmax
    unit price must land in 10-25 ns/op; (8,64)/(8,128) per-op price only
    <=2.5x the (1,64) one (batching lever); `--op vcmp --dtype int32` must be
    >=10x the same-shape `--dtype float32` and its scalar_us share >95%
    (scalarisation fingerprint).
  * unit-price formula (this task's convention): per_op_ns =
    (Task(reps) - Task(floor)) * 1000 / reps, with Task = msprof
    `OpBasicInfo.csv` Task Duration(us), median of 20 launches.
    NOTE: the `/ (reps * 40 cores)` form below does not apply under this
    convention (the dependent chain runs on 40 cores in parallel, so the
    delta is the single-core chain length); code kept verbatim as copied.
Re-verify: on toolchain change, re-run the widths above; if the relative
relations no longer hold, negate/deprecate the entry.

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
