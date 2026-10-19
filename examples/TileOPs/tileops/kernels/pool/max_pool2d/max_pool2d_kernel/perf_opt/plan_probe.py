"""Host-side plan probe (Stage 4 session 7): print the phase-path (bh, ch) that
the auto planner picks for each benchmark case, and which (bh, ch) override
combinations are accepted/rejected by the UB budget guard.

No NPU launch, no profile -- pure host arithmetic on the same planner code.

  python3 plan_probe.py
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import bench  # noqa: E402
import _max_pool2d_kernel as K  # noqa: E402


def show(case_id, grid):
    n, c, h, w, k, s, p, d, ceil_mode, dt = bench.CASES[case_id]
    plan = K._plan(n, c, h, w, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1],
                   ceil_mode, dt, vector_cores=40)
    ph = plan["phase"]
    print(f"== {case_id}: in=({n},{c},{h},{w}) k={k} s={s} p={p} d={d} ceil={ceil_mode}")
    print(f"   auto: path={plan['path']} ch={plan['ch']} num_cblk={plan['num_cblk']} "
          f"num_logical={plan['num_logical']} num_kernels={plan['num_kernels']} "
          f"num_local={plan['num_local']}")
    if ph:
        ub = K._phase_bytes(plan["ch"], ph["rows"], ph["pad_rows"], ph["bh"],
                            plan["out_w"], ph["w_pad"], ph["q_pad"],
                            plan["elem_bytes"], K._phase_acc_bytes(dt),
                            pipe=ph.get("pipe", False))
        print(f"   phase: bh={ph['bh']} ch={plan['ch']} rows={ph['rows']} "
              f"pad_rows={ph['pad_rows']} q_pad={ph['q_pad']} out_w={plan['out_w']} "
              f"pipe={ph.get('pipe', False)} "
              f"UB={ub/1024:.1f}KB x1.7={ub*1.7/1024:.1f}KB budget={K._UB_CAPACITY_BYTES/1024:.0f}KB")
    out_h, out_w = plan["out_h"], plan["out_w"]
    eb = plan["elem_bytes"]
    ab = K._phase_acc_bytes(dt)
    for (bh, ch) in grid:
        q_pad = K._round_up((w + 1) // 2, 8)
        w_pad = 2 * q_pad
        taps = K._phase_taps(k[1], p[1], d[1])
        shift_max = max(b - 1 for _, b in taps)
        tail = max(0, shift_max + out_w - q_pad)
        pad_rows = K._round_up(1 + q_pad + tail, 8) + 8
        rows = (bh - 1) * s[0] + (k[0] - 1) * d[0] + 1
        ub = K._phase_bytes(min(ch, c), rows, pad_rows, bh, out_w, w_pad, q_pad,
                            eb, ab, pipe=True)
        ok_div = (bh <= out_h and out_h % bh == 0)
        ok_ub = ub * K._UB_MULTIBUFFER_FACTOR <= K._UB_CAPACITY_BYTES
        ok_al = (min(ch, c) * eb) % 32 == 0
        print(f"   override bh={bh:2d} ch={ch:3d}: div={ok_div} align={ok_al} "
              f"UB={ub/1024:.1f}KB x1.7={ub*1.7/1024:.1f}KB -> "
              f"{'ACCEPT' if (ok_div and ok_ub and ok_al) else 'REJECT'}")


GRIDS = {
    "resnet-stem-float16": [(b, c) for b in (1, 2, 4, 8) for c in (64, 32)],
    "vgg-block-float16": [(b, 64) for b in (4, 8)],
    "alexnet-ceil-float16": [(3, 64)],
}
if __import__("sys").argv[1:] == ["--auto"]:
    for cid in bench.CASES:
        show(cid, [])
    raise SystemExit(0)
for cid, g in GRIDS.items():
    show(cid, g)
