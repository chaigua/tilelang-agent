#!/usr/bin/env python3
"""Generate the round-3 experiment branch files from the current best.

Branches (each changes exactly one main optimization point):
  r3_f16sum : C4 global path -- simplified reduce chain (reduce f16 -> f16, then
              cast once) to cut the vector UB traffic of the fp32 full-plane
              intermediate; CH_GLOBAL raised to its UB-price cap (64).
  r3_ch48   : C4 global path -- CH_GLOBAL 32 -> 48 (8 rounds instead of 13),
              keeping the current (precision-safe) fp32 chain.
  r3_gen48  : general grid path -- CH_GENERAL 64 -> 48, so spp uses 36/40
              vector cores instead of 24/40 (single wave).
"""
import pathlib
import re

SRC = pathlib.Path("_adaptive_avg_pool2d_kernel.py")
text = SRC.read_text()

# ---------------- r3_ch48 ----------------
ch48 = text.replace("CH_GLOBAL = 32", "CH_GLOBAL = 48")
assert ch48 != text
ch48 = ch48.replace(
    "TUNED.", "TUNED (round3 r3_ch48: C4 CH_GLOBAL 32->48).", 1
)
pathlib.Path("_adaptive_avg_pool2d_kernel_round3_ch48.py").write_text(ch48)

# ---------------- r3_gen48 ----------------
gen48 = text.replace("CH_GENERAL = 64", "CH_GENERAL = 48")
assert gen48 != text
gen48 = gen48.replace(
    "TUNED.", "TUNED (round3 r3_gen48: CH_GENERAL 64->48).", 1
)
pathlib.Path("_adaptive_avg_pool2d_kernel_round3_gen48.py").write_text(gen48)

# ---------------- r3_f16sum ----------------
old_alloc = "                        tile32 = T.alloc_ub((ch_glb, 1, plane), \"float32\")\n"
new_alloc = (
    "                        s16 = T.alloc_ub((ch_glb, 1, 1), dtype)\n"
)
old_chain = (
    "                                T.vcast(tile, tile32, round_mode=\"rint\")\n"
    "                                T.reduce_sum(tile32, sred, dim=2)\n"
)
new_chain = (
    "                                T.reduce_sum(tile, s16, dim=2)\n"
    "                                T.vcast(s16, sred, round_mode=\"rint\")\n"
)
f16sum = text.replace("CH_GLOBAL = 32", "CH_GLOBAL = 64")
assert f16sum != text
assert old_alloc in f16sum, "alloc anchor missing"
assert old_chain in f16sum, "chain anchor missing"
f16sum = f16sum.replace(old_alloc, new_alloc)
f16sum = f16sum.replace(old_chain, new_chain)
f16sum = f16sum.replace(
    "TUNED.", "TUNED (round3 r3_f16sum: C4 f16-direct reduce chain, CH_GLOBAL=64).", 1
)
pathlib.Path("_adaptive_avg_pool2d_kernel_round3_f16sum.py").write_text(f16sum)

for name in ("r3_ch48", "r3_gen48", "r3_f16sum"):
    p = pathlib.Path(f"_adaptive_avg_pool2d_kernel_round3_{name.split('_')[1]}.py")
    print(name, p, p.stat().st_size)
