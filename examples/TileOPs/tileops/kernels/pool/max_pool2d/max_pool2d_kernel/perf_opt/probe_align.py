"""Alignment + primitive semantics probe for the phase/transposed layouts.

Each test runs in its own process (a UB misalignment aborts the whole process),
so the runner loops over `--test`.

T1  base-offset alignment granularity of a 2-D VEC op (rows = CH, C or W
    innermost): identity `vmax(src[0:CH, off:off+OW], itself)` must return
    `off + ow`.  off in {0,1,2,4,8,16,32}.
T2  vmax NaN propagation with *aligned* operands (both operand orders).
T3  deinterleave into a dst region whose row stride is padded (row_elems > q).
T4  transpose of a (CH, q) region into a (q+pad, CH) buffer with a lead-in.
T5  deinterleave on an odd-width row (last dim even after padding).

  python probe_align.py --test t1
"""
import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401,E402

CH = 64
W = 112
OW = 56
Q = W // 2  # 56


def _jit(fn):
    return tilelang.jit(out_idx=[1], target="npuir")(fn)


# ---------------------------------------------------------------- T1
@_jit
def _t1(dtype):
    offs = (0, 1, 2, 4, 8, 16, 32)

    @T.prim_func
    def main(x: T.Tensor((1, CH, W), dtype), out: T.Tensor((len(offs), CH, OW), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), dtype)
            dst = T.alloc_ub((CH, OW), dtype)
            T.copy(x[0, 0:CH, 0:W], src)
            T.vmax(src[0:CH, 0:56], src[0:CH, 0:56], dst)
            T.copy(dst, out[0, 0:CH, 0:OW])
            T.vmax(src[0:CH, 1:57], src[0:CH, 1:57], dst)
            T.copy(dst, out[1, 0:CH, 0:OW])
            T.vmax(src[0:CH, 2:58], src[0:CH, 2:58], dst)
            T.copy(dst, out[2, 0:CH, 0:OW])
            T.vmax(src[0:CH, 4:60], src[0:CH, 4:60], dst)
            T.copy(dst, out[3, 0:CH, 0:OW])
            T.vmax(src[0:CH, 8:64], src[0:CH, 8:64], dst)
            T.copy(dst, out[4, 0:CH, 0:OW])
            T.vmax(src[0:CH, 16:72], src[0:CH, 16:72], dst)
            T.copy(dst, out[5, 0:CH, 0:OW])
            T.vmax(src[0:CH, 32:88], src[0:CH, 32:88], dst)
            T.copy(dst, out[6, 0:CH, 0:OW])

    return main


# ---------------------------------------------------------------- T2
@_jit
def _t2(dtype):
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), dtype), out: T.Tensor((4, CH, OW), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), dtype)
            a = T.alloc_ub((CH, OW), dtype)
            b = T.alloc_ub((CH, OW), dtype)
            T.copy(x[0, 0:CH, 0:W], src)
            # out0: vmax(a, b) with NaN in a
            T.vmax(src[0:CH, 0:OW], src[0:CH, OW:2 * OW], a)
            T.copy(a, out[0, 0:CH, 0:OW])
            # out1: vmax(b, a) with NaN in the *second* operand
            T.vmax(src[0:CH, OW:2 * OW], src[0:CH, 0:OW], b)
            T.copy(b, out[1, 0:CH, 0:OW])
            # out2: chain of three (accumulate)
            T.vmax(src[0:CH, 0:OW], src[0:CH, OW:2 * OW], a)
            T.vmax(a, src[0:CH, 0:OW], a)
            T.copy(a, out[2, 0:CH, 0:OW])
            # out3: inf / -inf pair -> +inf ?
            T.copy(src[0:CH, 0:OW], out[3, 0:CH, 0:OW])

    return main


# ---------------------------------------------------------------- T3
@_jit
def _t3(dtype):
    row_elems = 64  # padded stride (q = 56)

    @T.prim_func
    def main(x: T.Tensor((1, CH, W), dtype), out: T.Tensor((2, CH, Q), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), dtype)
            ev = T.alloc_ub((CH, row_elems), dtype)
            od = T.alloc_ub((CH, row_elems), dtype)
            T.copy(x[0, 0:CH, 0:W], src)
            T.deinterleave(src[0:CH, 0:W], ev[0:CH, 0:Q], od[0:CH, 0:Q],
                           channel_nums=2)
            T.copy(ev[0:CH, 0:Q], out[0, 0:CH, 0:Q])
            T.copy(od[0:CH, 0:Q], out[1, 0:CH, 0:Q])

    return main


# ---------------------------------------------------------------- T4
@_jit
def _t4(dtype):
    rows = 58  # q + 2 lead/tail pads

    @T.prim_func
    def main(x: T.Tensor((1, CH, W), dtype), out: T.Tensor((1, rows, CH), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), dtype)
            ev = T.alloc_ub((CH, Q), dtype)
            od = T.alloc_ub((CH, Q), dtype)
            et = T.alloc_ub((rows, CH), dtype)
            T.copy(x[0, 0:CH, 0:W], src)
            T.deinterleave(src[0:CH, 0:W], ev, od, channel_nums=2)
            T.transpose(ev, et[1:1 + Q, 0:CH], permutation=[1, 0])
            T.copy(et[0:rows, 0:CH], out[0, 0:rows, 0:CH])

    return main


# ---------------------------------------------------------------- T5
@_jit
def _t5(dtype):
    w_odd = 55
    w_pad = 56
    q = 28
    row_elems = 32

    @T.prim_func
    def main(x: T.Tensor((1, CH, w_odd), dtype), out: T.Tensor((2, CH, q), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, w_pad), dtype)
            ev = T.alloc_ub((CH, row_elems), dtype)
            od = T.alloc_ub((CH, row_elems), dtype)
            T.copy(x[0, 0:CH, 0:w_odd], src[0:CH, 0:w_odd])
            T.deinterleave(src[0:CH, 0:w_pad], ev[0:CH, 0:q], od[0:CH, 0:q],
                           channel_nums=2)
            T.copy(ev[0:CH, 0:q], out[0, 0:CH, 0:q])
            T.copy(od[0:CH, 0:q], out[1, 0:CH, 0:q])

    return main


# ---------------------------------------------------------------- T6/T7
# T6: vmax with a DYNAMIC row offset on a 2-D ep buffer (multi-row region).
# T7: transpose into a DYNAMIC row offset region of a big ep buffer.
@_jit
def _t6(dtype):
    slots = 64
    q = 56

    @T.prim_func
    def main(x: T.Tensor((1, CH, W), dtype), out: T.Tensor((4, q, CH), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, q), dtype)
            ep = T.alloc_ub((8 * slots, CH), dtype)
            dst = T.alloc_ub((q, CH), dtype)
            T.vbrc(T.cast(-T.infinity("float32"), dtype), ep)
            T.copy(x[0, 0:CH, 0:q], src)
            # (a) static dst region
            T.transpose(src, ep[1:1 + q, 0:CH], permutation=[1, 0])
            T.vmax(ep[1:1 + q, 0:CH], ep[1:1 + q, 0:CH], dst)
            T.copy(dst, out[0, 0:q, 0:CH])
            # (b) dynamic dst region from a serial loop
            T.copy(x[0, 0:CH, q:2 * q], src)
            for r in T.serial(2):
                T.transpose(src, ep[(r + 1) * slots + 1:(r + 1) * slots + 1 + q, 0:CH],
                            permutation=[1, 0])
            T.vmax(ep[2 * slots + 1:2 * slots + 1 + q, 0:CH],
                   ep[2 * slots + 1:2 * slots + 1 + q, 0:CH], dst)
            T.copy(dst, out[1, 0:q, 0:CH])
            # (c) dynamic vmax source offset
            for r in T.serial(2):
                T.vmax(ep[(r + 1) * slots + 1:(r + 1) * slots + 1 + q, 0:CH],
                       ep[(r + 1) * slots + 1:(r + 1) * slots + 1 + q, 0:CH], dst)
                T.copy(dst, out[r + 2, 0:q, 0:CH])

    return main


@_jit
def _t7(dtype):
    slots = 64
    q = 56

    @T.prim_func
    def main(x: T.Tensor((1, CH, W), dtype), out: T.Tensor((2, q, CH), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, q), dtype)
            ep = T.alloc_ub((8 * slots, CH), dtype)
            T.vbrc(T.cast(-T.infinity("float32"), dtype), ep)
            T.copy(x[0, 0:CH, 0:q], src)
            for r in T.serial(2):
                T.transpose(src, ep[r * slots + 1:r * slots + 1 + q, 0:CH],
                            permutation=[1, 0])
            T.copy(ep[1:1 + q, 0:CH], out[0, 0:q, 0:CH])
            T.copy(ep[slots + 1:slots + 1 + q, 0:CH], out[1, 0:q, 0:CH])

    return main


# ---------------------------------------------------------------- T8/T9
# T8: vmax / copy with a DYNAMIC flat row offset (o*OW) on a filled buffer.
# T9: vmax with a 3-D buffer whose leading index is dynamic.
@_jit
def _t8(dtype):
    q = 56
    bh = 4

    @T.prim_func
    def main(x: T.Tensor((1, CH, 4 * q), dtype), out: T.Tensor((4, q, CH), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, 4 * q), dtype)
            at = T.alloc_ub((4 * q, CH), dtype)
            acc = T.alloc_ub((bh * q, CH), dtype)
            T.copy(x[0, 0:CH, 0:4 * q], src)
            T.transpose(src, at, permutation=[1, 0])
            # dynamic flat offsets on BOTH vmax operands (the H-tap pattern)
            for o in T.serial(bh):
                T.vmax(at[o * q:o * q + q, 0:CH], at[o * q:o * q + q, 0:CH],
                       acc[o * q:o * q + q, 0:CH])
            for o in T.serial(bh):
                T.copy(acc[o * q:o * q + q, 0:CH], out[o, 0:q, 0:CH])
            # dynamic index derived from a division (divisibility mapping)
            for r in T.serial(4):
                if r % 2 == 0:
                    o2 = r // 2
                    T.vmax(at[o2 * q:o2 * q + q, 0:CH],
                           at[o2 * q:o2 * q + q, 0:CH], acc[0:q, 0:CH])
                    T.copy(acc[0:q, 0:CH], out[o2, 0:q, 0:CH])

    return main


@_jit
def _t9(dtype):
    q = 56
    bh = 4

    @T.prim_func
    def main(x: T.Tensor((1, CH, 4 * q), dtype), out: T.Tensor((4, q, CH), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((bh, q, CH), dtype)
            src = T.alloc_ub((CH, q), dtype)
            T.copy(x[0, 0:CH, 0:4 * q], src)
            T.transpose(src, a[0, 0:q, 0:CH], permutation=[1, 0])
            for o in T.serial(bh):
                T.vmax(a[o, 0:q, 0:CH], a[o, 0:q, 0:CH], a[o, 0:q, 0:CH])
                T.copy(a[o, 0:q, 0:CH], out[o, 0:q, 0:CH])

    return main


def _data(dtype):
    td = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}[dtype]
    x = torch.arange(1, 1 + CH * W, dtype=torch.float32).reshape(1, CH, W)
    return x.to(td)


def run(test, dtype):
    td = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}[dtype]
    if test == "t1":
        x = _data(dtype)
        out = _t1(dtype)(x.npu()).cpu()
        ok = True
        for i, off in enumerate((0, 1, 2, 4, 8, 16, 32)):
            if off + OW > W:
                continue
            ref = x[0, :, off:off + OW]
            bad = int((out[i] != ref).sum().item())
            print(f"[t1 {dtype}] off={off:3d}: {'PASS' if bad == 0 else f'FAIL {bad}'}")
            ok &= bad == 0
        return ok
    if test == "t2":
        x = torch.randn(1, CH, W, dtype=torch.float32)
        x[0, 0, 3] = float("nan")
        x[0, 1, OW + 3] = float("nan")
        x[0, 2, 0] = float("inf")
        x[0, 2, OW] = float("-inf")
        x = x.to(td)
        out = _t2(dtype)(x.npu()).cpu()
        got0, got1, got2 = out[0], out[1], out[2]
        nan0 = torch.isnan(got0)
        exp0 = torch.isnan(x[0, :, 0:OW]) | torch.isnan(x[0, :, OW:2 * OW])
        nan1 = torch.isnan(got1)
        exp1 = exp0
        ok = True
        for name, g, e in (("nan-op0", nan0, exp0), ("nan-op1", nan1, exp1),
                           ("nan-chain", torch.isnan(got2), exp0)):
            bad = int((g != e).sum().item())
            print(f"[t2 {dtype}] {name}: {'PASS' if bad == 0 else f'FAIL {bad}'} "
                  f"(nan found {int(g.sum())}/{int(e.sum())})")
            ok &= bad == 0
        infv = got0[2, 0].item()
        print(f"[t2 {dtype}] (+inf,-inf)->{infv} "
              f"{'PASS' if infv == float('inf') else 'FAIL'}")
        ok &= infv == float("inf")
        return ok
    if test == "t3":
        x = _data(dtype)
        out = _t3(dtype)(x.npu()).cpu()
        e_ref = x[0, :, 0::2]
        o_ref = x[0, :, 1::2]
        b0 = int((out[0] != e_ref).sum().item())
        b1 = int((out[1] != o_ref).sum().item())
        print(f"[t3 {dtype}] even={b0} odd={b1} "
              f"{'PASS' if b0 == 0 and b1 == 0 else 'FAIL'}")
        return b0 == 0 and b1 == 0
    if test == "t4":
        x = _data(dtype)
        out = _t4(dtype)(x.npu()).cpu()
        e_ref = x[0, :, 0::2].t()  # (Q, CH)
        got = out[0, 1:1 + Q]
        bad = int((got != e_ref).sum().item())
        lead = float(out[0, 0, 0].float().item())
        print(f"[t4 {dtype}] transposed(lead-in) mism={bad} lead={lead} "
              f"{'PASS' if bad == 0 else 'FAIL'}")
        return bad == 0
    if test == "t8":
        x = torch.arange(1, 1 + CH * 4 * Q, dtype=torch.float32).reshape(1, CH, 4 * Q).to(td)
        out = _t8(dtype)(x.npu()).cpu()
        b = [int((out[i] != x[0, :, i * Q:(i + 1) * Q].t()).sum().item())
             for i in range(4)]
        print(f"[t8 {dtype}] dyn-vmax r0={b[0]} r1={b[1]} r2={b[2]} div={b[3]} "
              f"{'PASS' if max(b) == 0 else 'FAIL'}")
        return max(b) == 0
    if test == "t9":
        x = _data(dtype)
        out = _t9(dtype)(x.npu()).cpu()
        b = [int((out[i] != x[0, :, i * Q:(i + 1) * Q].t()).sum().item())
             for i in range(4)]
        print(f"[t9 {dtype}] 3d-dyn-idx {b} "
              f"{'PASS' if max(b) == 0 else 'FAIL'}")
        return max(b) == 0
    if test == "t6":
        x = _data(dtype)
        out = _t6(dtype)(x.npu()).cpu()
        r0 = x[0, :, 0:Q].t()
        r1 = x[0, :, Q:2 * Q].t()
        b = [int((out[i] != r0).sum().item()) for i in (0,)] + \
            [int((out[i] != r1).sum().item()) for i in (1, 2, 3)]
        print(f"[t6 {dtype}] static-dst={b[0]} dyn-dst={b[1]} dyn-dst2={b[2]} "
              f"dyn-vmax={b[3]} {'PASS' if max(b) == 0 else 'FAIL'}")
        return max(b) == 0
    if test == "t7":
        x = _data(dtype)
        out = _t7(dtype)(x.npu()).cpu()
        r0 = x[0, :, 0:Q].t()
        r1 = x[0, :, 0:Q].t()
        b0 = int((out[0] != r0).sum().item())
        b1 = int((out[1] != r1).sum().item())
        print(f"[t7 {dtype}] dyn-r0={b0} dyn-r1={b1} "
              f"{'PASS' if b0 == 0 and b1 == 0 else 'FAIL'}")
        return b0 == 0 and b1 == 0
    if test == "t5":
        x = torch.arange(1, 1 + CH * 55, dtype=torch.float32).reshape(1, CH, 55).to(td)
        out = _t5(dtype)(x.npu()).cpu()
        e_ref = x[0, :, 0::2]
        o_ref = x[0, :, 1::2]
        b0 = int((out[0] != e_ref).sum().item())
        b1 = int((out[1][:, :o_ref.shape[1]] != o_ref).sum().item())
        print(f"[t5 {dtype}] odd-width even={b0} odd={b1} "
              f"{'PASS' if b0 == 0 and b1 == 0 else 'FAIL'}")
        return b0 == 0 and b1 == 0
    raise SystemExit(f"unknown test {test}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"])
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    ok = run(a.test, a.dtype)
    print("RESULT", "PASS" if ok else "FAIL")
