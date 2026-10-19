#!/usr/bin/env python3
"""Generate `_adaptive_avg_pool2d_kernel_round3_genh.py`: the round-3 candidate
that refines `_pick_ch_general` (general grid path channel-block heuristic).

Everything else is byte-identical to the round-2 current best.
"""
import pathlib

SRC = pathlib.Path("_adaptive_avg_pool2d_kernel.py")
text = SRC.read_text()

START = "def _pick_ch_general("
END = "\n\n\n# ------------------------------"
i = text.index(START)
j = text.index(END, i)

NEW = '''def _pick_ch_general(c_in: int, n: int, out_h: int, vector_cores: int, cap: int) -> int:
    """Effective channel block of the general grid path (G2, refined in round 3).

    The logical-unit count is ``N * out_H * ceil(C/CH)``; it decides how many
    physical vector cores are busy and whether the persistent loop needs more
    than one wave (``num_local > 1``).  A second wave costs more than a wider
    vector saves, so the block is selected **inside the single-wave set**
    (``n * out_h * ceil(C/CH) <= vector_cores``) by:

      1. maximise the number of busy cores (``units``);
      2. among equal occupancy, minimise the last block's wasted channels
         (``ceil(C/CH) * CH - C``);
      3. among those, prefer the wider block.

    Measured with `msprof op` Task Duration on the manifest general-grid shapes
    (round 3, `perf_opt/probe*`, `perf_opt/perf_records.jsonl`):
      * spp-6x6  (C=128, out 6x6, N=2): CH=64 -> 24 busy cores -> 16.13us;
        CH=48 -> 36 busy cores -> **15.37us (-4.7%)**;
        CH=32 -> 48 units -> 2 waves -> 24.32us (round-1 r1_gen32).
      * nondiv   (C=64,  out 7x7, N=2): CH=32 -> 28 units, 0 wasted -> 13.20us;
        CH=48 -> 28 units, 32 wasted channels -> 13.84us (+4.8%).
    Step (2) breaks the tie that would otherwise pick CH=48 for nondiv as well.
    """
    cap = max(_VECTOR_WIDTH, cap)
    limit = min(c_in, cap)
    best_key = None
    best_ch = None
    ch = _VECTOR_WIDTH
    while ch <= limit:
        units = n * out_h * ((c_in + ch - 1) // ch)
        if units <= vector_cores:
            waste = ((c_in + ch - 1) // ch) * ch - c_in
            key = (-units, waste, -ch)
            if best_key is None or key < best_key:
                best_key, best_ch = key, ch
        ch += _VECTOR_WIDTH
    if best_ch is not None:
        return best_ch
    # No single-wave block exists (many logical units): fall back to the widest
    # block within the capacity cap, which keeps the wave count lowest.
    return _round_ch(c_in, cap)'''

genh = text[:i] + NEW + text[j:]
genh = genh.replace(
    "TUNED.", "TUNED (round3 r3_genh: single-wave occupancy CH heuristic).", 1
)
pathlib.Path("_adaptive_avg_pool2d_kernel_round3_genh.py").write_text(genh)
print("wrote", len(genh))
