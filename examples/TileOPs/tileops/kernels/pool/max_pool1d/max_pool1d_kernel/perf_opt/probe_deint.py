"""probe_deint.py -- minimal repro/feasibility probe for T.deinterleave on npuir.

Answers:
  1. does `T.deinterleave(src, d0, d1, channel_nums=2)` produce the mod-2 phases?
  2. does `channel_nums=3` with three destinations work (needed for stride 3)?
  3. does a *region* destination (`ph[0:ch, 0:T]` of a taller buffer) work?
  4. is the resulting latency acceptable?

Run: python probe_deint.py            # correctness only (cheap)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

CH = 8
L = 12  # multiple of 2 and 3


def build_deint(cn, ntaps, region_dst=False, tall=0):
    Trows = L // cn

    @tilelang.jit(out_idx=[1 + cn], target="npuir")
    def _f(_x):
        outs = [T.Tensor((CH, Trows if not region_dst else tall), "float16")
                for _ in range(cn)]
        # third output is the whole (untouched-free) buffer for validation
        pass
        return None

    return None


def test_case(cn, ntaps):
    Trows = L // cn

    @tilelang.jit(out_idx=list(range(1, 1 + cn)), target="npuir")
    def _f(_tag):
        @T.prim_func
        def main(
            x: T.Tensor((CH, L), "float16"),
            o0: T.Tensor((CH, Trows), "float16"),
            o1: T.Tensor((CH, Trows), "float16"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                src = T.alloc_ub((CH, L), "float16")
                d0 = T.alloc_ub((CH, Trows), "float16")
                d1 = T.alloc_ub((CH, Trows), "float16")
                T.copy(x, src)
                T.deinterleave(src, d0, d1, channel_nums=cn)
                T.copy(d0, o0)
                T.copy(d1, o1)

        return main

    return _f(0)


def test_region_dst(cn=2):
    Trows = L // cn
    tall = Trows + 2

    @tilelang.jit(out_idx=[1, 2, 3], target="npuir")
    def _f(_tag):
        @T.prim_func
        def main(
            x: T.Tensor((CH, L), "float16"),
            o0: T.Tensor((CH, tall), "float16"),
            o1: T.Tensor((CH, tall), "float16"),
            ow: T.Tensor((CH,), "float16"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                src = T.alloc_ub((CH, L), "float16")
                d0 = T.alloc_ub((CH, tall), "float16")
                d1 = T.alloc_ub((CH, tall), "float16")
                seen = T.alloc_ub((CH,), "float16")
                T.vbrc(T.float16(-1000.0), d0)
                T.vbrc(T.float16(-1000.0), d1)
                T.copy(x, src)
                T.deinterleave(src, d0[0:CH, 0:Trows], d1[0:CH, 0:Trows],
                               channel_nums=cn)
                T.copy(d0[0:CH, Trows:Trows + 1], seen[0:CH])
                T.copy(d0, o0)
                T.copy(d1, o1)
                T.copy(seen, ow)

        return main

    return _f(0)


def main():
    x = torch.arange(CH * L, dtype=torch.float32).reshape(CH, L).to(torch.float16).npu()

    print("== case A: channel_nums=2 ==")
    try:
        k = test_case(2, 2)
        a, b = k(x)
        a = a.cpu(); b = b.cpu()
        xc = x.cpu()
        print("  d0 == x[:, 0::2]:", bool(torch.equal(a, xc[:, 0::2])))
        print("  d1 == x[:, 1::2]:", bool(torch.equal(b, xc[:, 1::2])))
    except Exception as exc:  # noqa: BLE001
        print("  FAILED:", repr(exc)[:600])

    print("== case B: channel_nums=3 (3 dsts?) ==")
    try:
        Trows = L // 3

        @tilelang.jit(out_idx=[1, 2, 3], target="npuir")
        def _f3(_tag):
            @T.prim_func
            def main(
                x: T.Tensor((CH, L), "float16"),
                o0: T.Tensor((CH, Trows), "float16"),
                o1: T.Tensor((CH, Trows), "float16"),
                o2: T.Tensor((CH, Trows), "float16"),
            ):
                with T.Kernel(1, is_npu=True) as (cid, _):
                    src = T.alloc_ub((CH, L), "float16")
                    d0 = T.alloc_ub((CH, Trows), "float16")
                    d1 = T.alloc_ub((CH, Trows), "float16")
                    d2 = T.alloc_ub((CH, Trows), "float16")
                    T.copy(x, src)
                    T.deinterleave(src, d0, d1, d2, channel_nums=3)
                    T.copy(d0, o0)
                    T.copy(d1, o1)
                    T.copy(d2, o2)

            return main

        k = _f3(0)
        a, b, c = k(x)
        xc = x.cpu()
        print("  d0 == x[:, 0::3]:", bool(torch.equal(a.cpu(), xc[:, 0::3])))
        print("  d1 == x[:, 1::3]:", bool(torch.equal(b.cpu(), xc[:, 1::3])))
        print("  d2 == x[:, 2::3]:", bool(torch.equal(c.cpu(), xc[:, 2::3])))
    except Exception as exc:  # noqa: BLE001
        print("  FAILED:", repr(exc)[:800])

    print("== case C: region dst into a taller buffer ==")
    try:
        k = test_region_dst(2)
        o0, o1, ow = k(x)
        xc = x.cpu()
        o0 = o0.cpu()
        print("  d0[:,0:T] == x[:,0::2]:", bool(torch.equal(o0[:, :L // 2], xc[:, 0::2])))
        print("  tail row kept at -1000:", bool((o0[:, L // 2:] == -1000.0).all()),
              o0[:, L // 2:].tolist()[:2])
    except Exception as exc:  # noqa: BLE001
        print("  FAILED:", repr(exc)[:800])


if __name__ == "__main__":
    main()
