"""CONST-reduce-ub-price-axis-alignment  (Home: constants.md)

Phenomenon (one sentence): the BishengIR UB requirement of a 3-D flat
row-reduction chain (``vcast`` x2 + in-place ``vmul`` + ``reduce_sum(dim=2)``
+ 1-D store staging) is ~96 B per tile element when the reduction axis length
is NOT a multiple of the vector width, against ~11-12 B/element when it is
(measured Ascend910B3 + CANN 9.0.0 by back-solving the
``ub overflow, requires N bits while M bits available`` message).

Assertions (magnitude + relative relations only; absolute values are printed
for human comparison): (1) aligned D=64 at a large tile overflows at < 20
B/element; (2) unaligned D=63 overflows at > 60 B/element; (3) the
unaligned/aligned unit-price ratio is > 3; (4) the positive control -- the
unaligned shape still compiles (and is numerically correct on the NPU) once
the tile is small enough, i.e. this is a capacity band, not a correctness
defect.

Measurement command family (for the msprof / re-verification steps):
  python CONST-reduce-ub-price-axis-alignment.py            # needs an NPU
  python CONST-reduce-ub-price-axis-alignment.py --check-only
The ``ub overflow`` numbers come from the compiler itself, so no profiler run
is required to re-derive the bands.

First verified: Ascend910B3 + CANN 9.0.0 + tilelang 0.1.2+1bc8dbc (dev root),
2026-10-16.
origin_task: multi_head_attention-_flashattn_bwd_preprocess_kernel-20261016
  (Stage 3, probe3.py sweep over D; the same constant drives the factory's
  alignment-aware element budget).
Re-verify: rerun the command above; the bands are asserted, the exact prices
  are printed.
"""

import os
import re
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


@tilelang.jit(out_idx=[2], target="npuir")
def rowsum(dim, blk_s, seq_len=64):
    @T.prim_func
    def kern(
        o: T.Tensor((1, seq_len, 1, dim), "float16"),
        do: T.Tensor((1, seq_len, 1, dim), "float16"),
        delta: T.Tensor((1, 1, seq_len), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, __):
            stage = T.alloc_shared((blk_s, 1, dim), "float16")
            acc32 = T.alloc_shared((blk_s, 1, dim), "float32")
            aux32 = T.alloc_shared((blk_s, 1, dim), "float32")
            dlta = T.alloc_shared((blk_s, 1, 1), "float32")
            mid1 = T.alloc_shared((blk_s,), "float32")
            real_s = T.min(blk_s, seq_len)
            T.copy(o[0, 0:real_s, 0, 0:dim], stage[0:real_s, 0:1, 0:dim])
            T.vcast(stage, acc32, round_mode="rint")
            T.copy(do[0, 0:real_s, 0, 0:dim], stage[0:real_s, 0:1, 0:dim])
            T.vcast(stage, aux32, round_mode="rint")
            T.vmul(acc32, aux32, acc32)
            T.reduce_sum(acc32, dlta, dim=2)
            T.copy(dlta[0:real_s, 0, 0], mid1[0:real_s])
            T.copy(mid1[0:real_s], delta[0, 0, 0:real_s])

    return kern


def unit_price(dim, blk_s):
    """Return the UB bytes/element BishengIR asks for, or None when it fits."""
    try:
        rowsum(dim, blk_s)
        return None
    except Exception as exc:  # noqa: BLE001
        m = re.search(r"requires (\d+) bits while (\d+) bits available", str(exc))
        assert m, f"unexpected failure for dim={dim} blk_s={blk_s}: {str(exc)[:200]}"
        return int(m.group(1)) / 8.0 / (blk_s * dim)


# ---- 1/2/3: the two capacity bands ---------------------------------------
aligned = unit_price(64, 384)  # E = 24576
unaligned = unit_price(63, 64)  # E = 4032
assert aligned is not None and unaligned is not None, "expected both tiles to overflow"
print(f"[repro] aligned   D=64 blk_s=384: {aligned:.2f} B/elem")
print(f"[repro] unaligned D=63 blk_s= 64: {unaligned:.2f} B/elem")
assert aligned < 20.0, f"aligned unit price {aligned:.2f} >= 20"
assert unaligned > 60.0, f"unaligned unit price {unaligned:.2f} <= 60"
assert unaligned / aligned > 3.0, f"ratio {unaligned / aligned:.2f} <= 3"
print("[repro] assertions 1-3 PASS: unaligned reduction axis costs >3x UB/element")

if sys.argv[1:] == ["--check-only"]:
    print("[repro] ALL ASSERTIONS PASS (compilation only)")
    raise SystemExit(0)

# ---- 4: positive control (small enough tile compiles and is correct) ------
rowsum(64, 64)
torch.manual_seed(0)
o = torch.randn(1, 32, 1, 63, dtype=torch.float32).to(torch.float16)
do = torch.randn(1, 32, 1, 63, dtype=torch.float32).to(torch.float16)
out = rowsum(63, 32, 32)(o.npu(), do.npu()).cpu()
ref = (o.float() * do.float()).sum(-1).transpose(1, 2).contiguous()
max_diff = (out - ref).abs().max().item()
assert out.shape == (1, 1, 32) and max_diff < 1e-4, f"max_diff={max_diff}"
print(f"[repro] assertion 4 PASS: shrunken unaligned tile correct, max_diff={max_diff:.3e}")
print("[repro] ALL ASSERTIONS PASS")
