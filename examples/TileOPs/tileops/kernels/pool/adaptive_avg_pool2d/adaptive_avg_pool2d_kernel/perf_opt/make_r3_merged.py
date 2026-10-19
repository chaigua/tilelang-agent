#!/usr/bin/env python3
"""Compose the round-3 merged current best from the round-2 best.

Measured winners merged in-kernel at factory level:
  * w1 (C4 global path, fp16): simplified reduce chain -- `reduce_sum(fp16 ->
    fp16)` + one `vcast` for the mean, instead of an fp32 full-plane
    intermediate.  resnet-global 16.61 -> 15.98 us (ab_test interleaved,
    -3.79%, verdict B_better).
  * w2 (general grid path): `_pick_ch_general` v2 -- single-wave occupancy
    first, then least wasted channels.  spp-6x6 16.32 -> 15.46 us (-5.27%,
    verdict B_better); nondiv unchanged (CH=32).
  * w3 (C4 capacity guards): the reduction UB price is charged per *plane
    element* (CONST-reduce-ub-per-element), so a wider CH is only legal for
    small planes.  Replaces the old `global_bytes` heuristic with
    `_pick_ch_global(plane, unit)`; the unit price is dtype-dependent because
    `reduce_sum` has no bf16 accumulation form (bf16 keeps the fp32 plane).

Usage: python3 make_r3_merged.py
"""
import pathlib

SRC = pathlib.Path("_adaptive_avg_pool2d_kernel_round2_best.py")
text = SRC.read_text()

# ---- w2 first: general-path CH heuristic --------------------------------
genh = pathlib.Path("_adaptive_avg_pool2d_kernel_round3_genh.py").read_text()
i = genh.index("def _pick_ch_general(")
j = genh.index("\n\n\n# ------------------------------", i)
new_fn = genh[i:j]
k = text.index("def _pick_ch_general(")
L = text.index("\n\n\n# ------------------------------", k)
text = text[:k] + new_fn + text[L:]

# ---- new constant: dtype-aware reduction UB unit price -------------------
ANCHOR_C = "# fp16/bf16 vector width in elements (alignment of the channel vector axis).\n_VECTOR_WIDTH = 8\n"
NEW_C = ANCHOR_C + '''
# Measured BishengIR reduction UB unit price for this kernel's C4 chain, in
# bytes of UB per *plane element* of the reduction input -- the lowering books
# it independently of the real dtype buffer list
# (constants.md CONST-reduce-ub-per-element).  Two chain forms:
#   * fp16 cheap chain (`reduce_sum(fp16 -> fp16)` + one `vcast`): calibrated
#     on 910B3 / CANN 9.0.0 by compile-time sweep (perf_opt/probe_r3_gf.py):
#     plane=49 fp16 admits CH=64 but not CH=128; plane=81 bf16 rejects CH=64
#     but admits CH=32 with the fp32 chain -> unit prices land in (37.9, 62.7].
#   * bf16 (`reduce_sum` has no bf16 accumulation form -> fp32 full-plane
#     intermediate + `reduce_sum(fp32 -> fp32)`).
_C4_REDUCE_UB_UNIT_F16 = 48
_C4_REDUCE_UB_UNIT_SUBFP32 = 64
'''
assert ANCHOR_C in text
text = text.replace(ANCHOR_C, NEW_C, 1)

# ---- new helper: plane-aware C4 channel block -----------------------------
HELPER_ANCHOR = "\n\n# ---------------------------------------------------------------------------\n# Golden (PyTorch CPU reference implementation)"
HELPER = '''

def _pick_ch_global(c_in: int, plane: int, cap: int, unit_bytes: int) -> int:
    """Channel block of the C4 global path, bounded by the reduction UB price.

    The reduction lowering books UB proportional to the *plane element* count
    of the reduction input, so the usable block is bounded by
    ``unit_bytes * plane * CH <= _UB_CAPACITY_BYTES`` -- a plane-aware cap, not
    a byte cap on the real buffer list.  Returns 0 when even one vector width
    does not fit; the factory then falls back to the general path (always
    correct).
    """
    plane_elems = _UB_CAPACITY_BYTES // unit_bytes
    limit = min(c_in, cap, plane_elems // plane if plane else cap)
    if limit < _VECTOR_WIDTH:
        return 0
    return (limit // _VECTOR_WIDTH) * _VECTOR_WIDTH
'''
assert HELPER_ANCHOR in text
text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)

# ---- factory: dtype-aware unit + plane-aware ch_glb -----------------------
OLD_CH = "    ch_glb = _round_ch(c_in, CH_GLOBAL)\n"
NEW_CH = (
    "    # C4 global: the reduction UB price scales with the plane element\n"
    "    # count and depends on the chain form, so the block cap is plane- and\n"
    "    # dtype-aware (`_pick_ch_global`).\n"
    "    c4_unit = _C4_REDUCE_UB_UNIT_F16 if dtype == \"float16\" else _C4_REDUCE_UB_UNIT_SUBFP32\n"
    "    ch_glb = _pick_ch_global(c_in, plane, CH_GLOBAL, c4_unit)\n"
)
assert OLD_CH in text
text = text.replace(OLD_CH, NEW_CH, 1)

OLD_NBLK = "    nblk_glb = (c_in + ch_glb - 1) // ch_glb\n"
NEW_NBLK = "    nblk_glb = (c_in + ch_glb - 1) // ch_glb if ch_glb else 0\n"
assert OLD_NBLK in text
text = text.replace(OLD_NBLK, NEW_NBLK, 1)

OLD_GUARD = """    # (CH,1,HW) fp16 tile + (CH,1,HW) fp32 accumulator + small fp32/fp16 tails.
    global_bytes = (ch_glb * plane * (elem_bytes + 4) + ch_glb * 6) * _UB_MULTIBUFFER_FACTOR
    global_fits = global_bytes <= _UB_CAPACITY_BYTES
"""
NEW_GUARD = """    # C4 global fit: `ch_glb == 0` means the plane-aware reduction-capacity cap
    # could not fit even one vector width, so the plane is not C4-viable.
    global_fits = ch_glb >= _VECTOR_WIDTH
"""
assert OLD_GUARD in text
text = text.replace(OLD_GUARD, NEW_GUARD, 1)

# ---- w1: rewrite the whole C4 (is_global) block ---------------------------
G0 = "    if is_global:\n"
G1 = "\n\n    # ---------------- general grid ----------------"
a = text.index(G0)
b = text.index(G1, a)

NEW_GLOBAL = '''    if is_global:
        num_kernels, num_local = _grid(n * nblk_glb)

        def _build(block_m=None):
            if block_m is not None and block_m < 1:
                raise ValueError(f"block_m must be positive, got block_m={block_m}")

            @tilelang.jit(out_idx=[1], target="npuir")
            def _func_global(ch_: int):
                @T.prim_func
                def main(
                    # The plane is read in its natural (contiguous) memory order:
                    # one channel block is CH*H*W contiguous elements, so the
                    # spatial H/W axes are merged into a single flat axis.  This
                    # keeps the GM->UB copy a plain contiguous transfer and makes
                    # the reduction axis wide (H*W) instead of narrow (W).
                    x: T.Tensor((n, c_in, 1, plane), dtype),
                    out: T.Tensor((n, c_in, out_h, out_w), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        if dtype == "float16":
                            # Cheap chain (G11/w1): reduce in fp16 and cast once,
                            # dropping the fp32 full-plane intermediate.  Its UB
                            # price (`_C4_REDUCE_UB_UNIT_F16`) admits a wider CH.
                            tile = T.alloc_ub((ch_glb, 1, plane), dtype)
                            s16 = T.alloc_ub((ch_glb, 1, 1), dtype)
                            sred = T.alloc_ub((ch_glb, 1, 1), "float32")
                            gout = T.alloc_ub((ch_glb, 1, 1), dtype)
                            for t in T.serial(num_local):
                                uid = t * num_kernels + cid
                                if uid < n * nblk_glb:
                                    nb = uid // nblk_glb
                                    cb = uid % nblk_glb
                                    real_c = T.min(ch_glb, c_in - cb * ch_glb)
                                    T.copy(
                                        x[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:plane],
                                        tile[0:real_c, 0:1, 0:plane],
                                    )
                                    T.reduce_sum(tile, s16, dim=2)
                                    T.vcast(s16, sred, round_mode="rint")
                                    T.vmul(sred, inv_plane, sred)
                                    T.vcast(sred, gout, round_mode="rint")
                                    T.copy(
                                        gout[0:real_c, 0:1, 0:1],
                                        out[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:1],
                                    )
                        else:
                            # bf16 has no `reduce_sum` accumulation form, so keep
                            # the fp32 full-plane intermediate (round-2 chain)
                            # and the matching price / CH cap.
                            tile = T.alloc_ub((ch_glb, 1, plane), dtype)
                            tile32 = T.alloc_ub((ch_glb, 1, plane), "float32")
                            sred = T.alloc_ub((ch_glb, 1, 1), "float32")
                            gout = T.alloc_ub((ch_glb, 1, 1), dtype)
                            for t in T.serial(num_local):
                                uid = t * num_kernels + cid
                                if uid < n * nblk_glb:
                                    nb = uid // nblk_glb
                                    cb = uid % nblk_glb
                                    real_c = T.min(ch_glb, c_in - cb * ch_glb)
                                    T.copy(
                                        x[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:plane],
                                        tile[0:real_c, 0:1, 0:plane],
                                    )
                                    T.vcast(tile, tile32, round_mode="rint")
                                    T.reduce_sum(tile32, sred, dim=2)
                                    T.vmul(sred, inv_plane, sred)
                                    T.vcast(sred, gout, round_mode="rint")
                                    T.copy(
                                        gout[0:real_c, 0:1, 0:1],
                                        out[nb, cb * ch_glb : cb * ch_glb + real_c, 0:1, 0:1],
                                    )

                return main

            return _func_global(ch_glb)

        return _build'''

text = text[:a] + NEW_GLOBAL + text[b:]

assert "CH_GLOBAL = 32" in text
text = text.replace("CH_GLOBAL = 32", "CH_GLOBAL = 64")
text = text.replace(
    "TUNED.", "TUNED (round3: C4 fp16 cheap reduce chain + plane/dtype-aware CH + single-wave CH heuristic).", 1
)
pathlib.Path("_adaptive_avg_pool2d_kernel_round3_merged.py").write_text(text)
print("wrote _adaptive_avg_pool2d_kernel_round3_merged.py", len(text))
