#!/usr/bin/env python3
"""Offline check of the round-3 `_pick_ch_general` v2 rule (no NPU / no msprof).

v2 chooses the channel block inside the single-wave set (units <= vector_cores)
by: (1) maximise busy cores (units), (2) minimise the last block's wasted
channels, (3) prefer the wider block.
"""
VW = 8
VC = 40
CAP = 64


def round_ch(limit, cap):
    # fill in `cap` at call time
    pass


def _round_ch(limit, cap):
    ch = min(limit, cap)
    return max(VW, ((ch + VW - 1) // VW) * VW)


def v1(c_in, n, out_h, vc, cap):
    base = _round_ch(c_in, cap)
    base_units = n * out_h * ((c_in + base - 1) // base)
    if base_units * 2 > vc:
        return base
    half = max(VW, base // 2)
    half_units = n * out_h * ((c_in + half - 1) // half)
    if half_units <= vc:
        return half
    return base


def v2(c_in, n, out_h, vc, cap):
    cap = max(VW, cap)
    limit = min(c_in, cap)
    best_key = best_ch = None
    ch = VW
    while ch <= limit:
        nblk = (c_in + ch - 1) // ch
        units = n * out_h * nblk
        if units <= vc:
            waste = nblk * ch - c_in
            key = (-units, waste, -ch)
            if best_key is None or key < best_key:
                best_key, best_ch = key, ch
        ch += VW
    return best_ch if best_ch is not None else _round_ch(c_in, cap)


CASES = [
    ("tune spp", 2, 128, 6), ("tune nondiv", 2, 64, 7),
    ("L0-1/smoke", 1, 8, 2), ("L0-2/spp", 2, 128, 6), ("L0-3/resnet", 8, 2048, 1),
    ("L0-4/nondiv", 2, 64, 7), ("L0-5/identity", 1, 8, 9), ("L0-6/half-none", 1, 16, 11),
    ("L0-7/scalar", 1, 16, 10), ("L0-9/cblk-edge", 2, 64, 6), ("L0-10/persist", 2, 2048, 1),
    ("L1/bf16/cblk-tail", 1, 96, 7), ("L1/fp16/nondiv", 1, 128, 7), ("L1/bf16/spp", 2, 128, 6),
    ("L1/fp16/global-2b", 2, 2048, 1), ("L1/bf16/global", 4, 64, 1), ("L1/identity-tail", 1, 96, 1),
    ("L1/expanding", 1, 8, 7), ("L1/h-identity", 1, 16, 11), ("L1/large-n", 16, 32, 3),
    ("L1/1x1", 1, 8, 1), ("L2/5x7x3", 1, 5, 2), ("L2/1x1x1", 1, 1, 1), ("L2/8x32x32", 1, 8, 1),
    ("L2/8x64x64", 1, 8, 1), ("L2/1024x3x3", 3, 1024, 1), ("L2/8x1x37", 1, 8, 1),
    ("L2/2x7x13", 2, 7, 9),
]
print(f"{'case':22s} {'C':>5s} {'N':>3s} {'outH':>4s} | {'v1':>4s} {'u1':>4s} | {'v2':>4s} {'u2':>4s}  changed")
for name, n, c, oh in CASES:
    a = v1(c, n, oh, VC, CAP); b = v2(c, n, oh, VC, CAP)
    u1 = n * oh * ((c + a - 1) // a); u2 = n * oh * ((c + b - 1) // b)
    print(f"{name:22s} {c:5d} {n:3d} {oh:4d} | {a:4d} {u1:4d} | {b:4d} {u2:4d}  {'YES' if a != b else ''}")
