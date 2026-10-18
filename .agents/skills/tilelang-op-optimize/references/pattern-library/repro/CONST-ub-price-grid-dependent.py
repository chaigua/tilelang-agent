"""[repro] CONST-ub-price-grid-dependent -- pattern-library/constants.md

Same buffer structure / same block_m produces DIFFERENT UB byte requirements
depending on the persistent grid size, so "may I raise block_m?" cannot be
answered by linear extrapolation from another shape or another grid.

Measured (Ascend910B4 / CANN 9.0.0 / tilelang 0.1.2+2afbf3b, this task):
  fp16 D=4096, block_m=4, auto multi-buffer ON, 29/32-block grid
      -> 229408 B (14.002 B/elem)  ... does not exist
  fp16 D=4096, block_m=4, auto multi-buffer ON, 40-block grid
      -> 196672 B (12.004 B/elem)  ... also does not exist (64 B over 196608 B)
  fp16 D=4096, block_m=4, auto multi-buffer OFF
      -> COMPILES (<= 12.0 B/elem)  ... the winning configuration
  fp16 D=4096, block_m=3, ON:    COMPILES  (the deployed config before it)
  fp16 D=6272, block_m=3, OFF:   225824 B (12.002 B/elem) ... does not exist
So: the price of a *given* structure is a function of (dtype path, block_m,
grid, buffer count) -- and grid alone moves it by +2.0 B/elem here.

Assertion (constant class): the compile probe must (a) reject block_m=4 with
multi-buffer ON with the compiler's `ub overflow ... requires N bits while M
bits available` message, and (b) accept block_m=4 with multi-buffer OFF.
Absolute byte counts are printed for human comparison only (environment
drift is expected); the pass/fail relation above is the invariant.

Measurement (on demand, NOT run by repro_runner for timing): duration numbers
come from `msprof op --kernel-name=main --launch-count=20 --warm-up=5` around
a bench harness; this file only carries the compile-time (UB accounting) half.

First verified: GroupNormFwdOp Stage 4 2026-10-18 (task
TileOPs-GroupNormFwdOp-20261018T021206Z), Ascend910B4 + CANN 9.0.0.
origin_task (provenance): TileOPs-GroupNormFwdOp-20261018T021206Z
Re-verification log (append-only):
  2026-10-18 PASS (compile probe: ON@bm=4 overflow, OFF@bm=4 compiles)
"""

import os
import re

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T


def build(M, D, block_m, dtype, num_blocks, multi_buffer):
    """GroupNorm-shaped residency: fp16 staging + 2 fp32 planes + 2 stats."""
    num_logical = (M + block_m - 1) // block_m
    num_kernels = min(num_logical, num_blocks)
    num_local_tasks = (num_logical + num_kernels - 1) // num_kernels
    kwargs = {"target": "npuir", "out_idx": [3]}
    if not multi_buffer:
        kwargs["pass_configs"] = {"npuir.enable_auto_multi_buffer": False}
    D_inv = 1.0 / float(D)

    @tilelang.jit(**kwargs)
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor((M, D), dtype),
            w: T.Tensor((D,), dtype),
            b: T.Tensor((D,), dtype),
            y: T.Tensor((M, D), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                stage = T.alloc_shared((block_m, D), dtype)
                x_f32 = T.alloc_shared((block_m, D), "float32")
                sq_f32 = T.alloc_shared((block_m, D), "float32")
                mean = T.alloc_shared((block_m, 1), "float32")
                var = T.alloc_shared((block_m, 1), "float32")
                rstd = T.alloc_shared((block_m, 1), "float32")
                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        off = block_id * block_m
                        real_m = T.min(block_m, M - off)
                        T.copy(x[off : off + real_m, 0:D], stage[0:real_m, 0:D])
                        T.vcast(stage, x_f32, round_mode="rint")
                        T.reduce_sum(x_f32, mean, dim=1)
                        T.vmul(mean, D_inv, mean)
                        T.vsub(x_f32, mean, x_f32)
                        T.vmul(x_f32, x_f32, sq_f32)
                        T.reduce_sum(sq_f32, var, dim=1)
                        T.vmul(var, D_inv, var)
                        T.vadd(var, 1e-5, var)
                        T.vsqrt(var, rstd)
                        T.vdiv(rstd, var, rstd)
                        T.vmul(x_f32, rstd, x_f32)
                        T.vcast(x_f32, stage, round_mode="rint")
                        T.copy(stage[0:real_m, 0:D], y[off : off + real_m, 0:D])

        return main

    return _func


def probe(M, D, block_m, dtype, num_blocks, multi_buffer):
    """Return (compiled, bytes_required_or_None)."""
    try:
        build(M, D, block_m, dtype, num_blocks, multi_buffer)()
        return True, None
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        m = re.search(r"requires\s+(\d+)\s*bits while\s+(\d+)\s*bits available", msg)
        if not m:
            raise
        return False, int(m.group(1)) // 8


def main() -> None:
    M, D, dtype = 256, 4096, "float16"
    for nb in (32, 40):
        ok, need = probe(M, D, 4, dtype, nb, True)
        print(f"mb ON  bm=4 grid={nb}: compiled={ok} bytes={need} "
              f"({(need or 0) / (4 * D):.3f} B/elem)")
        assert not ok, "mb ON + bm=4 must overflow this structure"
        assert need is not None and need > 196608, "overflow must exceed ub_cap"

    ok_on, need_on = probe(M, D, 3, dtype, 32, True)
    assert ok_on, "mb ON + bm=3 is the deployed configuration and must compile"
    print(f"mb ON  bm=3 grid=32: compiled={ok_on}")

    ok_off, need_off = probe(M, D, 4, dtype, 32, False)
    assert ok_off, "mb OFF + bm=4 must compile (this is the round-3 win)"
    print(f"mb OFF bm=4 grid=32: compiled={ok_off} bytes={need_off}")

    ok_off2, need_off2 = probe(M, D, 3, dtype, 32, False)
    print(f"mb OFF bm=3 grid=32: compiled={ok_off2} bytes={need_off2}")
    print("CONST-ub-price-grid-dependent: PASS (overflow iff bm=4 with mb ON)")


if __name__ == "__main__":
    main()
