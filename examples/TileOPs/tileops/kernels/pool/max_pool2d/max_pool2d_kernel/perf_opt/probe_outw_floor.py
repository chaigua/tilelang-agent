"""Session 8 probe: which output widths can the phase path handle?

`probe_l1_paths.py` showed the phase path is *not* bit-exact on (2,16,9,11)
(out_w = 6) and (1,8,11,13) (out_w = 7) while it is exact on out_w >= 27.
The suspicion is the per-output-row wide tile: `acc` / `et` / `ot` are
`(out_w, ch)`-shaped and a VEC tile-op row block is 8 rows, so a sub-8-row
output width spills past the buffer end.

This probe runs the same "large magnitude" data through w_in = 11, 13, 15, 17
(out_w = 6, 7, 8, 9 -- everything else fixed) and prints the outcome, so the
applicability guard in `_phase_plan` is chosen from data.

  python3 probe_outw_floor.py
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402

import _max_pool2d_kernel as K  # noqa: E402


def main():
    for w_in, out_w in ((11, 6), (13, 7), (15, 8), (17, 9), (27, 14)):
        x = (torch.randn(2, 16, 9, w_in, dtype=torch.float32) * 1000.0).to(
            torch.float16)
        plan = K._plan(2, 16, 9, w_in, 3, 3, 2, 2, 1, 1, 1, 1, False, "float16")
        ph = plan["phase"] or {}
        try:
            K._run_case(n=2, c_in=16, h_in=9, w_in=w_in, kernel_size=(3, 3),
                        stride=(2, 2), padding=(1, 1), dilation=(1, 1),
                        ceil_mode=False, dtype_str="float16", tag="PROBE", x=x)
            print(f"RESULT w_in={w_in} out_w={out_w} path={plan['path']} "
                  f"bh={ph.get('bh')} -> EXACT", flush=True)
        except AssertionError as exc:
            print(f"RESULT w_in={w_in} out_w={out_w} path={plan['path']} "
                  f"bh={ph.get('bh')} -> MISMATCH "
                  f"{str(exc).splitlines()[-1].strip()}", flush=True)


if __name__ == "__main__":
    main()
