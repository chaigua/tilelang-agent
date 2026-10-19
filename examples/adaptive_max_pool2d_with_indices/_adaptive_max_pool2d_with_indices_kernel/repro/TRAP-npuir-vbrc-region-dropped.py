# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP repro: a scalar ``T.vbrc`` whose destination is a *buffer region* is
silently dropped on npuir -- the destination keeps its previous contents.

Phenomenon (asserted below, on NPU):
  * ``T.vbrc(T.float32(SEED), acc[ow:ow+1, 0:CH])`` (scalar -> region, inside a
    ``T.serial`` loop with a runtime ``ow``) writes nothing: after a following
    in-place ``T.vmax`` accumulation the seed value is nowhere in ``acc``.
  * the sanctioned equivalent -- a full-buffer ``T.vbrc`` into a ``(1, CH)``
    staging buffer followed by ``T.copy(staging, acc[ow:ow+1, 0:CH])`` --
    lands the seed for every ``ow`` (and for every CH tested: 8 / 32 / 64).
  * a *raw* Python scalar operand fails the frontend rank assert
    (``customize_npuir.py::npuir_brc``: "The input vector and output vector must
    have same rank", raised whenever ``src`` is not a ``tir.PrimExpr``); a typed
    scalar (``T.int32(-1)`` / ``T.float32(3.0)``) is accepted.

Why it matters: ``_adaptive_max_pool2d_with_indices_kernel`` seeds its first-hit
index accumulator per output column with a runtime value
(``ih0 * W_in + iw0(ow)``).  With the region form the seed is silently lost, so
an all-``-inf`` window returns index 0 instead of the first visited element's
flat index -- a silent wrong-answer, not a crash.  The int32 region form is
worse: the dropped write can leave the region uninitialised and the following
read faults the vector core (acl error 507034 / "Vector core execution timed
out"), which is how this trap was first hit.

Run: /workspace/venv/bin/python repro/TRAP-npuir-vbrc-region-dropped.py
Exit code 0 == trap and sanctioned form both confirmed.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

WIN, OUT_W = 2, 2
SEED = 1.0e9  # dominates any window value, so a surviving seed is unmistakable


def build(dtype, seed_form):
    """seed_form in {'region', 'staging'}."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def f(ch):
        @T.prim_func
        def main(
            x: T.Tensor((WIN, OUT_W, ch), dtype),
            out: T.Tensor((OUT_W, ch), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                acc = T.alloc_ub((OUT_W, ch), dtype)
                stg = T.alloc_ub((1, ch), dtype)
                nxt = T.alloc_ub((1, ch), dtype)
                T.vbrc(T.float32(-1.0e30), acc)  # full-buffer scalar fill (works)
                for ow in T.serial(OUT_W):
                    if seed_form == "region":
                        # TRAPPED FORM: scalar -> region with a runtime offset.
                        T.vbrc(T.float32(SEED + ow), acc[ow : ow + 1, 0:ch])
                    else:
                        # SANCTIONED FORM: full buffer + T.copy into the region.
                        T.vbrc(T.float32(SEED + ow), stg)
                        T.copy(stg, acc[ow : ow + 1, 0:ch])
                for kw in T.serial(WIN):
                    for ow in T.serial(OUT_W):
                        T.copy(x[kw, ow : ow + 1, 0:ch], nxt)
                        T.vmax(acc[ow : ow + 1, 0:ch], nxt, acc[ow : ow + 1, 0:ch])
                T.copy(acc, out)

        return main

    return f


def main():
    for ch in (8, 32, 64):
        x = torch.arange(WIN * OUT_W * ch, dtype=torch.float32).reshape(
            WIN, OUT_W, ch
        ) * 2.0
        want = torch.tensor([[SEED], [SEED + 1]], dtype=torch.float32).expand(OUT_W, ch)

        trapped = build("float32", "region")(ch)(x.npu()).cpu()
        sanctioned = build("float32", "staging")(ch)(x.npu()).cpu()

        raised = False
        try:

            @tilelang.jit(out_idx=[1], target="npuir")
            def bad():
                @T.prim_func
                def main(
                    x: T.Tensor((OUT_W, ch), "float32"),
                    out: T.Tensor((OUT_W, ch), "float32"),
                ):
                    with T.Kernel(1, is_npu=True) as (cid, _):
                        a = T.alloc_ub((OUT_W, ch), "float32")
                        T.vbrc(3.0, a)  # raw Python scalar -> frontend rank assert
                        T.copy(a, out)

                return main

            bad()
        except Exception as exc:  # noqa: BLE001
            # The frontend emits the diagnostic on stderr and raises a
            # DiagnosticError; the intercept is what matters here.
            raised = True
            msg = str(exc).splitlines()[0][:90]
        else:
            msg = "no exception"

        print(f"ch={ch}: region-form seed value   : {trapped[0][:3].tolist()}")
        print(f"ch={ch}: staging-form seed value  : {sanctioned[0][:3].tolist()}")
        print(f"ch={ch}: raw-scalar vbrc rejected : {raised} ({msg})")

        assert not torch.equal(trapped, want), (
            f"ch={ch}: TRAP NOT reproduced -- the region-form seed survived"
        )
        assert torch.equal(sanctioned, want), (
            f"ch={ch}: sanctioned staging form lost the seed: {sanctioned[0][:3]}"
        )
        assert raised, f"ch={ch}: raw Python scalar was accepted by T.vbrc"

    print(
        "TRAP confirmed: region-targeted scalar T.vbrc is dropped; "
        "staging buffer + T.copy is the sanctioned form; raw scalars rejected"
    )


if __name__ == "__main__":
    main()
