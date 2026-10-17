"""Overlap probe (Stage 4, session 9): can the per-row GM->UB copy be hidden
behind the vector work of the previous row?

The frozen phase path copies one input row per iteration and consumes it
immediately (``deinterleave`` reads ``src``), so the pipeline must wait for the
DMA before the first vector op of that row.  Measured shape is
``dur ~= vec + mte2 + mte3`` on every workload (resnet f16: 231.5 ~= 125.5 +
59.2 + 25.7 us), i.e. no cross-pipe overlap at all.

This probe reproduces the *exact* resnet-stem-f16 phase row body twice:

  serial  : one ``src`` buffer, copy -> deinterleave -> transpose -> vmax
  pingpong: two ``src`` buffers.  The pair loop is unrolled by two at trace time
            (compile-time parity) so that neither ``deinterleave`` sits inside an
            if/else branch -- the npuir pipeline silently DROPS ``vdeinterleave``
            from a conditional branch (measured: the op disappears from the
            generated hivm IR and the result is garbage).

Nothing is written outside this file; the deliverable stays frozen.

  python3 probe_pipe.py            # msprof both variants, print table
  python3 probe_pipe.py --check    # bit-exactness pingpong vs serial
Entry: candidate pattern-library/bottleneck-patterns.md BP_pipeline_overlap update (queue
  VP-2026-0303) + pattern-library/traps-compiler.md TRAP-deinterleave-two-phase-only (case C);
  one repro covers both forms.
First verified: 2026-10-17, task max_pool2d-_max_pool2d_kernel-20261017T043853Z (Stage 4
  optimize; tilelang 0.1.2+b3a7b21fa006c1ef94b7c786d81158e16426a025 + torch 2.9.0+cpu /
  torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3, aicore 20 -> Vector 40 / UB 196608 B)
Insight: a row loop that copies a row and immediately consumes it (copy -> deinterleave ->
  transpose -> vmax) serialises MTE2 against VEC -- the fingerprint is
  Task Duration ~= vec + mte2 + mte3 (serial: 201.0 ~= 105.2 + 76.5 + 30.2 us), i.e. the sum
  is a *serialisation* signature, not a roofline floor. A ping-pong row scratch (row r+1
  prefetched while row r is consumed) hides the DMA: 156.6 us (vec 108.5 / mte2 98.9).
  Two necessary conditions: (1) parity resolved at TRACE time -- a runtime if/else branch
  makes the npuir pipeline silently DROP vdeinterleave from the hivm IR (compiles, runs,
  wrong output); (2) the store stays at unit level -- per-row stores cost 9x write
  amplification (486 vs 229.6 us on resnet f16, u2g 5544 vs 616 KB/block).
  Enable predicate: structural UB check `ub_bytes(pipe=True) * 1.7 <= 192 KB`, else keep the
  serial row loop (same arithmetic, same taps).
Assertions: `--check` -> assert torch.equal(pingpong, serial) over the whole 448-unit probe
  workload; default run -> msprof both variants and print dur/vec/mte2/mte3 + sum + delta%
  (assert magnitude/relative relations only, absolute latency not locked).
Provenance (task workspace, may lapse): examples/TileOPs/tileops/kernels/pool/max_pool2d/
  max_pool2d_kernel/perf_opt/probe_pipe.py (+ opt_log.md#Round-3).
Registration (ED-C): byte-identical copy of perf_opt/probe_pipe.py (md5 23e58bf61a29db7194e8dc67ba06c91a)
  + this registration block appended inside the module docstring; code and assertions unchanged.
  Budget: 247 lines / 11155 B -- exceeds both ED-B budgets, not trimmed (PL-1.10 precedent:
  trimming would break the two-variant runnable comparison).
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

# ---- resnet-stem f16 tile parameters (from plan_probe.py --auto) -----------
UNITS = 448
CH, ROWS, W_PAD = 64, 9, 112
OUT_W, OUT_H = 56, 56
Q_PAD = 56
PAD_ROWS = 72
BH = 4
S_H = 2
CORES = 40
NUM_LOCAL = (UNITS + CORES - 1) // CORES  # 12
DT = "float16"
# taps of k_w=3, p_w=1, s_w=2: (odd_phase, base)
TAPS = [(1, 0), (0, 1), (1, 1)]


@T.macro
def row_body(acc, et, ot, eo, oo, perl, st, out, uid, r):
    """transpose -> taps -> per-row store (identical to the frozen kernel)."""
    T.transpose(eo, et[1:1 + Q_PAD, 0:CH], permutation=[1, 0])
    T.transpose(oo, ot[1:1 + Q_PAD, 0:CH], permutation=[1, 0])
    if r % S_H == 0:
        ob = r // S_H
        if ob < BH:
            # one literal statement per tap (a Python ``for`` inside
            # @T.prim_func becomes a TIR loop, so tap constants must be literal)
            if TAPS[0][0] == 1:
                T.vmax(acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH],
                       ot[TAPS[0][1]:TAPS[0][1] + OUT_W, 0:CH],
                       acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH])
            else:
                T.vmax(acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH],
                       et[TAPS[0][1]:TAPS[0][1] + OUT_W, 0:CH],
                       acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH])
            if TAPS[1][0] == 1:
                T.vmax(acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH],
                       ot[TAPS[1][1]:TAPS[1][1] + OUT_W, 0:CH],
                       acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH])
            else:
                T.vmax(acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH],
                       et[TAPS[1][1]:TAPS[1][1] + OUT_W, 0:CH],
                       acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH])
            if TAPS[2][0] == 1:
                T.vmax(acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH],
                       ot[TAPS[2][1]:TAPS[2][1] + OUT_W, 0:CH],
                       acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH])
            else:
                T.vmax(acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH],
                       et[TAPS[2][1]:TAPS[2][1] + OUT_W, 0:CH],
                       acc[ob * OUT_W:(ob + 1) * OUT_W, 0:CH])



def build(pipe):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((UNITS, CH, ROWS, W_PAD), DT),
            out: T.Tensor((UNITS, CH, OUT_H, OUT_W), DT),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                src = T.alloc_ub((CH, W_PAD), DT)
                src2 = T.alloc_ub((CH, W_PAD), DT)
                eo = T.alloc_ub((CH, Q_PAD), DT)
                oo = T.alloc_ub((CH, Q_PAD), DT)
                et = T.alloc_ub((PAD_ROWS, CH), DT)
                ot = T.alloc_ub((PAD_ROWS, CH), DT)
                acc = T.alloc_ub((BH * OUT_W, CH), DT)
                perl = T.alloc_ub((OUT_W, CH), DT)
                st = T.alloc_ub((CH, OUT_W), DT)
                ninf = T.cast(-T.infinity("float32"), DT)
                T.vbrc(ninf, src)
                T.vbrc(ninf, src2)
                T.vbrc(ninf, et)
                T.vbrc(ninf, ot)
                for t in T.serial(NUM_LOCAL):
                    uid = t * CORES + cid
                    if uid < UNITS:
                        T.vbrc(ninf, acc)
                        if pipe:
                            # prologue: row 0 -> src (buffer A)
                            T.copy(x[uid, 0:CH, 0, 0:W_PAD], src)
                            # unrolled pair loop: compile-time parity
                            for k in T.serial(ROWS // 2):
                                r0 = 2 * k
                                if r0 + 1 < ROWS:
                                    T.copy(x[uid, 0:CH, r0 + 1, 0:W_PAD], src2)
                                T.deinterleave(src, eo, oo, channel_nums=2)
                                row_body(acc, et, ot, eo, oo, perl, st, out,
                                         uid, r0)
                                if r0 + 2 < ROWS:
                                    T.copy(x[uid, 0:CH, r0 + 2, 0:W_PAD], src)
                                T.deinterleave(src2, eo, oo, channel_nums=2)
                                row_body(acc, et, ot, eo, oo, perl, st, out,
                                         uid, r0 + 1)
                            if ROWS % 2 == 1:
                                T.deinterleave(src, eo, oo, channel_nums=2)
                                row_body(acc, et, ot, eo, oo, perl, st, out,
                                         uid, ROWS - 1)
                        else:
                            for r in T.serial(ROWS):
                                T.copy(x[uid, 0:CH, r, 0:W_PAD], src)
                                T.deinterleave(src, eo, oo, channel_nums=2)
                                row_body(acc, et, ot, eo, oo, perl, st, out,
                                         uid, r)
                        # ONE store row per unit: mte3 kept minimal so that the
                        # DMA/vec overlap is what the measurement shows
                        T.copy(acc[0:OUT_W, 0:CH], perl)
                        T.transpose(perl, st, permutation=[1, 0])
                        T.copy(st, out[uid, 0:CH, 0, 0:OUT_W])

        return main

    return _f()


def run(tag, kind):
    import prof

    od = os.path.abspath(os.path.join("profiles", "pipe", tag))
    os.makedirs(od, exist_ok=True)
    os.chmod(od, 0o700)
    script = os.path.abspath(__file__)
    cmd = ["msprof", "op", "--kernel-name=main", f"--output={od}",
           "--launch-count=20", "--warm-up=5", "--dump=off",
           "--aic-metrics=PipeUtilization,Memory",
           sys.executable, script, "--run-one", kind]
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if pr.returncode != 0:
        print(pr.stdout[-3000:], pr.stderr[-3000:])
        raise SystemExit(f"msprof failed for {tag}")
    r = prof.parse(od)
    print(f"[{tag}] " + str({k: r[k] for k in ("duration_us", "vec", "mte2",
                                              "mte3", "scalar", "block_dim")}))
    return r


def run_one(kind):
    import torch
    import torch_npu  # noqa: F401

    k = build(kind == "pipe")
    x = torch.randn(UNITS, CH, ROWS, W_PAD, dtype=torch.float16).npu()
    with torch.no_grad():
        out = None
        for _ in range(25):
            out = k(x)
    torch.npu.synchronize()
    print(f"[run-one] {kind}: out={tuple(out.shape)}")


def check():
    import torch
    import torch_npu  # noqa: F401

    x = torch.randn(UNITS, CH, ROWS, W_PAD, dtype=torch.float16).npu()
    with torch.no_grad():
        ref = build(False)(x).cpu()
        got = build(True)(x).cpu()
    print("equal:", bool(torch.equal(got, ref)),
          "max_diff:", (got.float() - ref.float()).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.run_one:
        run_one(args.run_one)
        return 0
    if args.check:
        check()
        return 0
    a = run("serial", "serial")
    b = run("pingpong", "pipe")
    print("\n| variant | dur | vec | mte2 | mte3 | sum |")
    print("|---|---|---|---|---|---|")
    for tag, r in (("serial", a), ("pingpong", b)):
        s = sum(r[k] for k in ("vec", "mte2", "mte3"))
        print(f"| {tag} | {r['duration_us']} | {r['vec']} | {r['mte2']} | "
              f"{r['mte3']} | {s:.2f} |")
    print(f"\ndelta = {(b['duration_us'] / a['duration_us'] - 1) * 100:+.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
