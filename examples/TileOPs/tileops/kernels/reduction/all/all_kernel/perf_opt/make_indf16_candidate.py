"""Materialise the ``f16 indicator plane`` candidate from the Stage 3 baseline.

Single structural point: the zero-hit indicator plane (and every micro-buffer
derived from it) is stored/reduced in **float16** instead of float32.

  * ``i1 -> f16`` (``T.vcast``, documented ``i1 | f16 | rint``) replaces
    ``i1 -> f32``;
  * ``T.reduce_max`` runs on the f16 plane (documented fp16 support);
  * the complement ``1 - max`` is exact in f16 (the indicator is exactly 0.0 /
    1.0) and the epilogue ``f16 -> i8`` is a documented cast, so the extra
    ``f32 -> f16`` cast of the Stage 3 chain disappears.

Semantics are unchanged bit-for-bit (0/1 is exact in f16, ``reduce_max`` is
exact), which the L0/L1 suites verify with ``torch.equal``.

Usage: python make_indf16_candidate.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "_logical_reduce_kernel.py"))
OUT = os.path.join(HERE, "_logical_reduce_kernel_round3_op6_indf16.py")

REPLACEMENTS = [
    # ---- path R (resident) buffers
    (
        '''                ind = T.alloc_shared((block_m, N), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")''',
        '''                # Stage 4: the zero-hit indicator is exactly 0.0/1.0, so it
                # is materialised and reduced in float16 (half the reduce input
                # bytes, one fewer epilogue cast).
                ind = T.alloc_shared((block_m, N), "float16")
                mx = T.alloc_fragment((block_m, 1), "float16")
                mx_flat = T.alloc_fragment((block_m,), "float16")
                one_v = T.alloc_fragment((block_m,), "float16")
                all_v = T.alloc_fragment((block_m,), "float16")''',
    ),
    # ---- path R epilogue
    (
        '''                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f32
                        T.reduce_max(ind, mx, dim=1, clear=True)  # 1.0 iff any zero
                        T.reshape(mx, mx_flat)  # (bm, 1) -> (bm,) bridge
                        T.vsub(one_v, mx_flat, all_v)  # all = 1 - any_zero
                        T.vcast(all_v, all_f16, round_mode="rint")  # f32 -> f16
                        T.vcast(all_f16, out_ub, round_mode="rint")  # f16 -> i8''',
        '''                        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
                        T.vcast(mask_eq, ind, round_mode="rint")  # i1 -> f16
                        T.reduce_max(ind, mx, dim=1, clear=True)  # 1.0 iff any zero
                        T.reshape(mx, mx_flat)  # (bm, 1) -> (bm,) bridge
                        T.vsub(one_v, mx_flat, all_v)  # all = 1 - any_zero
                        T.vcast(all_v, out_ub, round_mode="rint")  # f16 -> i8''',
    ),
    # ---- path C buffers
    (
        '''                ind = T.alloc_shared((block_m, tile_n), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")''',
        '''                ind = T.alloc_shared((block_m, tile_n), "float16")
                mx = T.alloc_fragment((block_m, 1), "float16")
                acc = T.alloc_fragment((block_m, 1), "float16")
                new_acc = T.alloc_fragment((block_m, 1), "float16")
                acc_flat = T.alloc_fragment((block_m,), "float16")
                one_v = T.alloc_fragment((block_m,), "float16")
                all_v = T.alloc_fragment((block_m,), "float16")''',
    ),
    # ---- path C epilogue
    (
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)  # minuend is always 1
                        T.vcast(all_v, all_f16, round_mode="rint")
                        T.vcast(all_f16, out_ub, round_mode="rint")''',
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)  # minuend is always 1
                        T.vcast(all_v, out_ub, round_mode="rint")''',
    ),
    # ---- path C tail buffers
    (
        '''                ind_tail = T.alloc_shared((block_m, tail), "float32")
                mx = T.alloc_fragment((block_m, 1), "float32")
                mx_tail = T.alloc_fragment((block_m, 1), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                new_acc_t = T.alloc_fragment((block_m, 1), "float32")
                acc_flat = T.alloc_fragment((block_m,), "float32")
                one_v = T.alloc_fragment((block_m,), "float32")
                all_v = T.alloc_fragment((block_m,), "float32")
                all_f16 = T.alloc_fragment((block_m,), "float16")''',
        '''                ind_tail = T.alloc_shared((block_m, tail), "float16")
                mx = T.alloc_fragment((block_m, 1), "float16")
                mx_tail = T.alloc_fragment((block_m, 1), "float16")
                acc = T.alloc_fragment((block_m, 1), "float16")
                new_acc = T.alloc_fragment((block_m, 1), "float16")
                new_acc_t = T.alloc_fragment((block_m, 1), "float16")
                acc_flat = T.alloc_fragment((block_m,), "float16")
                one_v = T.alloc_fragment((block_m,), "float16")
                all_v = T.alloc_fragment((block_m,), "float16")''',
    ),
    # ---- path C tail epilogue
    (
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)
                        T.vcast(all_v, all_f16, round_mode="rint")
                        T.vcast(all_f16, out_ub, round_mode="rint")''',
        '''                        T.reshape(acc, acc_flat)
                        T.vsub(one_v, acc_flat, all_v)
                        T.vcast(all_v, out_ub, round_mode="rint")''',
    ),
]


def main():
    with open(BASE) as handle:
        src = handle.read()
    for old, new in REPLACEMENTS:
        if src.count(old) != 1:
            raise SystemExit(f"replacement not unique ({src.count(old)}): {old[:80]!r}")
        src = src.replace(old, new)
    if "all_f16" in src.split("def _build_resident")[1].split("def _build_tiled")[0]:
        raise SystemExit("resident builder still references all_f16")
    with open(OUT, "w") as handle:
        handle.write(src)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
