"""[repro] PL-1.26-rounds-first-then-multibuffer -- pattern-library/elementwise.md

Optimization insight (GroupNorm Stage 4, 2026-10-18): for a persistent
row-reduction kernel the first-order lever is the number of in-core rounds
`R = ceil(ceil(M/bm) / nk)`, not the auto multi-buffer switch. The switch is
merely a *means* to a larger `block_m`: with multi-buffer ON the fp16
residency price is 14.0 B/elem at bm=4 (D=4096) so bm=4 does not exist, while
with it OFF the price is <= 12.0 B/elem and bm=4 compiles. Same `block_m`
comparison says the switch loses 3.4%; the bm-increasing comparison says it
wins 16.1% (interleaved A/B, 3/3 pairs, 9.18 -> 7.70 us). So:
  1) minimise R over (block_m, num_kernels) inside the UB budget,
  2) then pick the multi-buffer mode that makes that block_m legal,
  3) never judge a switching alternative at a fixed block_m.

Mechanism attribution: per-round cost is dominated by the exposed scalar +
vector chain (measured 0.07 us/op issue, ~2.1-2.4 us of scalar per 2-3
rounds); DMA is only 1.6-1.8 us, so removing one round removes ~1.5 us while
the switch alone changes nothing but the residency price
(see CONST-ub-price-grid-dependent).

Assertion (pattern class, delta form): py_compile only -- the effect shows at
full kernel scale. The two skeletons below differ ONLY in the residency /
round-count decision; `rounds()` documents the arithmetic that decides it.
Measured reference: image-g32 fp16, M=256, D=4096, 40 vector cores:
  slow: bm=3, 86 logical tasks, nk=40 -> R=3  (9.18 us)
  fast: bm=4, 64 logical tasks, nk=32 -> R=2  (7.70 us, -16.1%)

First verified: GroupNormFwdOp Stage 4 2026-10-18 (ab_round3_mboff_bm4).
origin_task (provenance): TileOPs-GroupNormFwdOp-20261018T021206Z
Re-verification log (append-only):
  2026-10-18 py_compile PASS (delta skeleton; full-scale effect measured in
  perf_opt/ab_round3_mboff_bm4 and perf_opt/profiles/final_v4)
"""

import math

import tilelang
import tilelang.language as T


def rounds(M, block_m, nk):
    """In-core serial rounds per core: the quantity to minimise first."""
    num_logical = math.ceil(M / block_m)
    num_kernels = min(num_logical, nk)
    return math.ceil(num_logical / num_kernels), num_logical, num_kernels


def balanced_grid(M, block_m, vector_cores):
    """Smallest grid that still achieves the minimal round count."""
    num_logical = math.ceil(M / block_m)
    r = math.ceil(num_logical / min(num_logical, vector_cores))
    return math.ceil(num_logical / r)


def build(M, D, block_m, nk, dtype, multi_buffer):
    """GroupNorm-shaped persistent row reduction (skeleton of the tuned kernel)."""
    num_logical = math.ceil(M / block_m)
    num_kernels = balanced_grid(M, block_m, nk) if num_logical > nk else num_logical
    num_local_tasks = math.ceil(num_logical / num_kernels)
    kwargs = {"target": "npuir", "out_idx": [3]}
    if not multi_buffer:
        kwargs["pass_configs"] = {"npuir.enable_auto_multi_buffer": False}
    d_inv = 1.0 / D

    @tilelang.jit(**kwargs)
    def _f():
        @T.prim_func
        def main(
            x: T.Tensor((M, D), dtype),
            w: T.Tensor((D,), dtype),
            b: T.Tensor((D,), dtype),
            y: T.Tensor((M, D), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                stage = T.alloc_shared((block_m, D), dtype)
                xf = T.alloc_shared((block_m, D), "float32")
                sq = T.alloc_shared((block_m, D), "float32")
                mean = T.alloc_shared((block_m, 1), "float32")
                var = T.alloc_shared((block_m, 1), "float32")
                rstd = T.alloc_shared((block_m, 1), "float32")
                for i in T.serial(num_local_tasks):          # <- rounds
                    blk = i * num_kernels + cid
                    if blk < num_logical:
                        off = blk * block_m
                        rm = T.min(block_m, M - off)
                        T.copy(x[off : off + rm, 0:D], stage[0:rm, 0:D])
                        T.vcast(stage, xf, round_mode="rint")
                        T.reduce_sum(xf, mean, dim=1)
                        T.vmul(mean, d_inv, mean)
                        T.vsub(xf, mean, xf)
                        T.vmul(xf, xf, sq)
                        T.reduce_sum(sq, var, dim=1)
                        T.vmul(var, d_inv, var)
                        T.vadd(var, 1e-5, var)
                        T.vsqrt(var, rstd)
                        T.vdiv(rstd, var, rstd)
                        T.vmul(xf, rstd, xf)
                        T.vcast(xf, stage, round_mode="rint")
                        T.copy(stage[0:rm, 0:D], y[off : off + rm, 0:D])

        return main

    return _f


if __name__ == "__main__":
    M, D = 256, 4096
    slow = rounds(M, 3, 40)      # multi-buffer ON  -> bm=3 legal, R=3
    fast = rounds(M, 4, 40)      # multi-buffer OFF -> bm=4 legal, R=2
    print(f"slow bm=3: rounds/logical/grid = {slow}")
    print(f"fast bm=4: rounds/logical/grid = {fast}")
    assert slow[0] == 3 and fast[0] == 2, "round-count arithmetic"
    assert balanced_grid(M, 4, 40) == 32, "balanced grid for bm=4 is 32 blocks"
    build(M, D, 4, 40, "float16", multi_buffer=False)
    build(M, D, 3, 40, "float16", multi_buffer=True)
    print("PL-1.26-rounds-first-then-multibuffer: delta skeleton builds")
