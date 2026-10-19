"""Session 8 probe: numerics of the L1 cases whose expected *path tag* is stale.

The L1 table in `_max_pool2d_kernel.py` still says `expect_path="general"` for
six shapes that the phase-decimated path (introduced in Stage 4) now handles.
`_run_case` asserts the tag *before* it compares against the golden, so a green
L1 row does not prove those shapes are numerically exact under the phase plan.
This probe re-runs exactly those six cases with `expect_path=None` and prints
max_diff / violations / exactness, so the tag update is backed by data.

  python3 probe_l1_paths.py
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import _max_pool2d_kernel as K  # noqa: E402

CASES = [
    # (n, c, h, w, k, s, p, d, ceil, dtype, label)
    (16, 128, 56, 56, (2, 2), (2, 2), (0, 0), (1, 1), False, "bfloat16",
     "L1/bf16/vgg-block"),
    (32, 64, 55, 55, (3, 3), (2, 2), (0, 0), (1, 1), True, "bfloat16",
     "L1/bf16/alexnet-ceil"),
    (16, 128, 56, 56, (2, 2), (2, 2), (0, 0), (1, 1), False, "float32",
     "L1/fp32/vgg-block"),
    (32, 64, 55, 55, (3, 3), (2, 2), (0, 0), (1, 1), True, "float32",
     "L1/fp32/alexnet-ceil"),
    (2, 16, 12, 14, (2, 2), (2, 2), (0, 0), (1, 1), False, "float16",
     "L1/fp16/disjoint"),
]


def main():
    fails = []
    for (n, c, h, w, k, s, p, d, ceil_mode, dt, label) in CASES:
        try:
            plan = K._plan(n, c, h, w, k[0], k[1], s[0], s[1], p[0], p[1], d[0],
                           d[1], ceil_mode, dt)
            print(f"{label}: path={plan['path']} bh={plan['phase']['bh']} "
                  f"ch={plan['ch']} num_logical={plan['num_logical']} "
                  f"num_local={plan['num_local']}", flush=True)
            K._run_case(n=n, c_in=c, h_in=h, w_in=w, kernel_size=k, stride=s,
                        padding=p, dilation=d, ceil_mode=ceil_mode,
                        dtype_str=dt, tag="PROBE", expect_path=None)
        except AssertionError as exc:
            fails.append((label, exc))
    # nan-window f32 (the dtype that flipped to the phase path)
    for dt in ("float32",):
        label = f"L1/{dt}/nan-window"
        try:
            td = K._TOLERANCE[dt][0]
            x = K._nan_window_x().to(td)
            K._run_case(n=1, c_in=8, h_in=11, w_in=13, kernel_size=(3, 3),
                        stride=(2, 2), padding=(1, 1), dilation=(1, 1),
                        ceil_mode=False, dtype_str=dt, tag="PROBE", x=x,
                        equal_nan=True, expect_path=None)
        except AssertionError as exc:
            fails.append((label, exc))
    print("PROBE FAILURES:", fails, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
