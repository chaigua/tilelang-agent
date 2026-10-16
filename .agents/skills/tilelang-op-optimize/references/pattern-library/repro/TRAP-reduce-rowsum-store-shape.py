"""TRAP-reduce-rowsum-store-shape  (Home: traps-compiler.md)

Phenomenon (one sentence): a 3-D row-reduction result ``(blk_s, 1, 1)`` stored
straight into an S-contiguous GM segment ``delta[bz, bx, s0:s0+real_s]``
(region ``(1, 1, real_s)``) is rejected by the npuir codegen
(`bufferization.materialize_in_destination` op source/destination shapes are
incompatible); staging the row sums through a 1-D UB buffer
(``T.copy(dlta[0:real_s, 0, 0], mid1[0:real_s])`` then
``T.copy(mid1[0:real_s], delta[bz, bx, s0:s0+real_s])``) compiles and is
numerically correct.

Assertions: (1) mode "bad" -- the direct store form fails to compile and the
error text names ``materialize_in_destination`` + ``shapes are incompatible``;
(2) mode "good" -- the 1-D staging form compiles, runs on the NPU and matches
the torch reference within 1e-4.

Front-end/actionable root cause: the region of ``dlta[0:real_s, 0, 0]`` keeps
rank 3 with the non-unit axis FIRST (``(real_s, 1, 1)``) while
``delta[bz, bx, s0:s0+real_s]`` keeps rank 3 with the non-unit axis LAST
(``(1, 1, real_s)``); the two regions are not shape-compatible and the
bufferization destination materialisation is a literal shape match.  Units
can only be folded from the left, so a source region must also end with the
payload axis to match a contiguous GM segment on the innermost axis.  A
single-axis ``T.transpose(dlta, mid, permutation=[2, 1, 0])`` (``(1, 1, blk_s)``)
is an equally correct alternative, but it introduces a `vtranspose` op.

First verified: Ascend910B3 + CANN 9.0.0 + tilelang 0.1.2+1bc8dbc (dev root),
2026-10-16.
origin_task: multi_head_attention-_flashattn_bwd_preprocess_kernel-20261016
  (probe /tmp/.../probe2.py variants A/D vs B, promoted under ED-C).
Re-verify: run ``python TRAP-reduce-rowsum-store-shape.py`` (needs an NPU) or
  ``python TRAP-reduce-rowsum-store-shape.py --check-only`` (compilation only).
"""

import os
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

B, S, H, D, BLK_S = 1, 64, 2, 64, 64  # S == BLK_S: one block per (b, h)


def _prim_func(use_staging: bool):
    @T.prim_func
    def kern(
        o: T.Tensor((B, S, H, D), "float16"),
        do: T.Tensor((B, S, H, D), "float16"),
        delta: T.Tensor((B, H, S), "float32"),
    ):
        with T.Kernel(B * H, is_npu=True) as (cid, _):
            stage = T.alloc_shared((BLK_S, 1, D), "float16")
            acc32 = T.alloc_shared((BLK_S, 1, D), "float32")
            aux32 = T.alloc_shared((BLK_S, 1, D), "float32")
            dlta = T.alloc_shared((BLK_S, 1, 1), "float32")
            mid1 = T.alloc_shared((BLK_S,), "float32")
            bz = cid // H
            bx = cid % H
            real_s = T.min(BLK_S, S)
            T.copy(o[bz, 0:real_s, bx, 0:D], stage[0:real_s, 0:1, 0:D])
            T.vcast(stage, acc32, round_mode="rint")
            T.copy(do[bz, 0:real_s, bx, 0:D], stage[0:real_s, 0:1, 0:D])
            T.vcast(stage, aux32, round_mode="rint")
            T.vmul(acc32, aux32, acc32)
            T.reduce_sum(acc32, dlta, dim=2)
            if use_staging:
                T.copy(dlta[0:real_s, 0, 0], mid1[0:real_s])
                T.copy(mid1[0:real_s], delta[bz, bx, 0:real_s])
            else:
                T.copy(dlta[0:real_s, 0, 0], delta[bz, bx, 0:real_s])

    return kern


if len(sys.argv) > 1 and sys.argv[1] == "--bad-child":
    tilelang.compile(_prim_func(False), out_idx=[2], target="npuir")
    print("BAD-FORM-COMPILED")
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] != "--check-only":
    raise SystemExit(f"unknown argument {sys.argv[1]!r}")

# ---- assertion 1: the direct (design) store form must be rejected ----------
child = subprocess.run(
    [sys.executable, __file__, "--bad-child"], capture_output=True, text=True
)
blob = child.stdout + child.stderr
assert child.returncode != 0 and "BAD-FORM-COMPILED" not in child.stdout, (
    "the direct (real_s,1,1) -> (1,1,real_s) store unexpectedly compiled"
)
assert "materialize_in_destination" in blob and "shapes are incompatible" in blob, (
    f"unexpected failure text: {blob[-400:]}"
)
print("[repro] assertion 1 PASS: direct store form rejected "
      "(materialize_in_destination shapes incompatible)")

if sys.argv[1:] == ["--check-only"]:
    print("[repro] ALL ASSERTIONS PASS (compilation only)")
    raise SystemExit(0)

# ---- assertion 2: the 1-D staging workaround compiles and is correct -------
torch.manual_seed(0)
o = torch.randn(B, S, H, D, dtype=torch.float32).to(torch.float16)
do = torch.randn(B, S, H, D, dtype=torch.float32).to(torch.float16)
kern = tilelang.compile(_prim_func(True), out_idx=[2], target="npuir")
out = kern(o.npu(), do.npu()).cpu()
ref = (o.float() * do.float()).sum(-1).transpose(1, 2).contiguous()
max_diff = (out - ref).abs().max().item()
assert out.shape == (B, H, S) and max_diff < 1e-4, f"max_diff={max_diff}"
print(f"[repro] assertion 2 PASS: 1-D staging form correct, max_diff={max_diff:.3e}")
print("[repro] ALL ASSERTIONS PASS")
