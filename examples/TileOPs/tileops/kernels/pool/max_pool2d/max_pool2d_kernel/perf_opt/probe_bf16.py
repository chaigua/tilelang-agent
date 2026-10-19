"""bf16 vbrc / pad-arming compile probe (codegen bug bisect).

The phase path fails at codegen (VcastCodegen) for bfloat16 only.  This probe
compiles the individual statements to find which one trips
``getBroadcastDim(buffer_shape0, buffer_shape1)``.

  python3 probe_bf16.py
"""
import os
import traceback

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T

DT = "bfloat16"
CH = 64
W = 64


def jit(fn):
    return tilelang.jit(out_idx=[1], target="npuir")(fn)


def _ninf():
    return T.cast(-T.infinity("float32"), DT)


@jit
def _v1():
    """vbrc of a cast scalar into a (33, CH) buffer."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((33, CH), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            et = T.alloc_ub((33, CH), DT)
            T.vbrc(_ninf(), et)
            T.copy(et[0:33, 0:CH], out[0:33, 0:CH])

    return main


@jit
def _v2():
    """vbrc into a (406, CH) buffer."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((406, CH), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            acc = T.alloc_ub((406, CH), DT)
            T.vbrc(_ninf(), acc)
            T.copy(acc, out)

    return main


@jit
def _v3():
    """vbrc into three shapes in one kernel."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((33, CH), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), DT)
            et = T.alloc_ub((33, CH), DT)
            acc = T.alloc_ub((406, CH), DT)
            T.vbrc(_ninf(), src)
            T.vbrc(_ninf(), et)
            T.vbrc(_ninf(), acc)
            T.copy(et, out)

    return main


@jit
def _v4():
    """vbrc + deinterleave + transpose + vmax (the phase statement mix)."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((33, CH), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), DT)
            eo = T.alloc_ub((CH, W // 2), DT)
            oo = T.alloc_ub((CH, W // 2), DT)
            et = T.alloc_ub((33, CH), DT)
            ot = T.alloc_ub((33, CH), DT)
            T.vbrc(_ninf(), src)
            T.vbrc(_ninf(), et)
            T.vbrc(_ninf(), ot)
            T.copy(x[0, 0:CH, 0:W], src)
            T.deinterleave(src, eo, oo, channel_nums=2)
            T.transpose(eo, et[1:1 + W // 2, 0:CH], permutation=[1, 0])
            T.transpose(oo, ot[1:1 + W // 2, 0:CH], permutation=[1, 0])
            T.vmax(et[0:W // 2, 0:CH], ot[0:W // 2, 0:CH], et[0:W // 2, 0:CH])
            T.copy(et, out)

    return main


@jit
def _v5():
    """deinterleave only (bf16)."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((CH, W // 2), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src = T.alloc_ub((CH, W), DT)
            eo = T.alloc_ub((CH, W // 2), DT)
            oo = T.alloc_ub((CH, W // 2), DT)
            T.copy(x[0, 0:CH, 0:W], src)
            T.deinterleave(src, eo, oo, channel_nums=2)
            T.copy(eo, out)

    return main


@jit
def _v6():
    """transpose only (bf16)."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((W // 2, CH), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            eo = T.alloc_ub((CH, W // 2), DT)
            et = T.alloc_ub((33, CH), DT)
            T.copy(x[0, 0:CH, 0:W // 2], eo)
            T.transpose(eo, et[1:1 + W // 2, 0:CH], permutation=[1, 0])
            T.copy(et[1:1 + W // 2, 0:CH], out)

    return main


@jit
def _v7():
    """vmax only (bf16)."""
    @T.prim_func
    def main(x: T.Tensor((1, CH, W), DT), out: T.Tensor((W // 2, CH), DT)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((W // 2, CH), DT)
            T.copy(x[0, 0:CH, 0:W // 2], a)
            T.vmax(a, a, a)
            T.copy(a, out)

    return main


def main():
    for name in ("_v4", "_v5", "_v6", "_v7"):
        try:
            globals()[name]()
            print(f"[{name}] COMPILE OK")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip().splitlines()
            tail = [ln for ln in msg if "Check failed" in ln or "error" in ln.lower()]
            print(f"[{name}] FAIL: {(tail[-1] if tail else msg[-1])[:200]}")


if __name__ == "__main__":
    main()
