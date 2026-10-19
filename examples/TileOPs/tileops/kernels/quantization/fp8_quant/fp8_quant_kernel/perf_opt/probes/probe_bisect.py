"""Bisect probe for `_fp8_quant_kernel` scalarization (Stage 4).

Exposes the same factory signature as the baseline so `bench.py --impl
probes/probe_bisect.py` can drive it.  The kernel variant is selected by the
`PROBE_STAGE` environment variable so msprof profiling stays a single harness.

Stages (each strictly adds work on top of the previous):
  s0_dma       : T.copy X -> UB, cross-dtype T.copy UB -> OutU8 (DMA floor)
  s1_cast      : + 2D T.Parallel fp16->fp32 cast loop + 2D u8 store loop
  s2_absreduce : + T.vabs + T.reduce(absmax, dim=1)
  s3_scale     : + amax_c/vdiv scale + vdiv quant + vclamp
  s4_enc       : + software e4m3 encode chain (up to the pre-sign value)
  s5_full      : full baseline-equivalent kernel
"""

import functools
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

FP8_MIN, FP8_MAX = -448.0, 448.0
AMAX_EPS = 1e-4
BLOCK_ELEMS = 2048
_DEFAULT_VECTOR_CORES = 40


def _vector_core_num():
    try:
        from tilelang.utils.npu_utils import NPUUtils

        num = int(NPUUtils.get().get_aivector_core_num())
        return num if num > 0 else _DEFAULT_VECTOR_CORES
    except Exception:
        return _DEFAULT_VECTOR_CORES


@functools.lru_cache(maxsize=64)
def _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype):
    stage = os.environ.get("PROBE_STAGE", "s5_full")
    rows = batch * seq_len_kv * kv_group
    dim = index_dim
    block_m = max(1, min(BLOCK_ELEMS // dim, rows))
    n_tiles = (rows + block_m - 1) // block_m
    num_blocks = min(_vector_core_num(), n_tiles)
    iters = (n_tiles + num_blocks - 1) // num_blocks

    @tilelang.jit(target="npuir")
    def _fwd_func(num_stages=0, block_m_hint=32):
        _ = (num_stages, block_m_hint)

        @T.prim_func
        def _fwd_main(
            X: T.Tensor((rows, dim), in_dtype),
            Scale: T.Tensor((rows,), "float32"),
            OutU8: T.Tensor((rows, dim), "uint8"),
        ):
            with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                x_in = T.alloc_ub((block_m, dim), in_dtype)
                u8 = T.alloc_ub((block_m, dim), "uint8")
                scale1 = T.alloc_ub((block_m,), "float32")
                x32 = T.alloc_ub((block_m, dim), "float32")
                av = T.alloc_ub((block_m, dim), "float32")
                amax = T.alloc_ub((block_m, 1), "float32")
                amax_c = T.alloc_ub((block_m, 1), "float32")
                c448 = T.alloc_ub((block_m, 1), "float32")
                scale2 = T.alloc_ub((block_m, 1), "float32")
                qv = T.alloc_ub((block_m, dim), "float32")
                lv = T.alloc_ub((block_m, dim), "float32")
                t1 = T.alloc_ub((block_m, dim), "float32")
                num = T.alloc_ub((block_m, dim), "float32")
                pwf = T.alloc_ub((block_m, dim), "float32")
                sbv = T.alloc_ub((block_m, dim), "float32")
                ei = T.alloc_ub((block_m, dim), "int32")
                qi = T.alloc_ub((block_m, dim), "int32")
                one_i = T.alloc_ub((block_m, dim), "int32")
                pw = T.alloc_ub((block_m, dim), "int32")
                cond = T.alloc_ub((block_m, dim), "bool")

                for i, j in T.Parallel(block_m, dim):
                    one_i[i, j] = 1
                for i in T.Parallel(block_m):
                    c448[i, 0] = 448.0

                for it in T.serial(iters):
                    tile = it * num_blocks + cid
                    if tile < n_tiles:
                        row0 = T.min(tile * block_m, rows - block_m)
                        T.copy(X[row0, 0], x_in)

                        if stage == "s0_dma":
                            T.copy(x_in, OutU8[row0, 0])
                            for i in T.Parallel(block_m):
                                scale1[i] = 1.0
                            T.copy(scale1, Scale[row0:row0 + block_m])
                        else:
                            for i, j in T.Parallel(block_m, dim):
                                x32[i, j] = T.cast(x_in[i, j], "float32")
                            if stage == "s1_cast":
                                for i, j in T.Parallel(block_m, dim):
                                    u8[i, j] = T.cast(
                                        T.min(T.max(x32[i, j], -128.0), 127.0), "uint8")
                                for i in T.Parallel(block_m):
                                    scale1[i] = 1.0
                                T.copy(u8, OutU8[row0, 0])
                                T.copy(scale1, Scale[row0:row0 + block_m])
                            else:
                                T.vabs(x32, av)
                                T.reduce(av, amax, dims=[1], reduce_mode="max")
                                if stage == "s2_absreduce":
                                    for i, j in T.Parallel(block_m, dim):
                                        u8[i, j] = T.cast(
                                            T.min(T.max(x32[i, j], -128.0), 127.0), "uint8")
                                    for i in T.Parallel(block_m):
                                        scale1[i] = amax[i, 0]
                                    T.copy(u8, OutU8[row0, 0])
                                    T.copy(scale1, Scale[row0:row0 + block_m])
                                else:
                                    for i in T.Parallel(block_m):
                                        amax_c[i, 0] = T.max(amax[i, 0], AMAX_EPS)
                                    T.vdiv(amax_c, c448, scale2)
                                    T.vdiv(x32, scale2, qv)
                                    T.vclamp(qv, qv, FP8_MIN, FP8_MAX)
                                    if stage == "s3_scale":
                                        for i, j in T.Parallel(block_m, dim):
                                            u8[i, j] = T.cast(
                                                T.min(T.max(qv[i, j], -128.0), 127.0),
                                                "uint8")
                                        for i in T.Parallel(block_m):
                                            scale1[i] = scale2[i, 0]
                                        T.copy(u8, OutU8[row0, 0])
                                        T.copy(scale1, Scale[row0:row0 + block_m])
                                    else:
                                        T.vabs(qv, av)
                                        T.vcmp(qv, av, cond, "lt")
                                        T.vcast(cond, sbv, round_mode="rint")
                                        T.vclamp(av, av, 1e-30, FP8_MAX)
                                        T.vlog2(av, lv, t1)
                                        T.vcast(lv, ei, round_mode="floor")
                                        T.vclamp(ei, ei, -6, 8)
                                        T.vcast(ei, t1, round_mode="rint")
                                        for i, j in T.Parallel(block_m, dim):
                                            qi[i, j] = ei[i, j] + 6
                                        T.vshl(one_i, qi, pw)
                                        T.vcast(pw, pwf, round_mode="rint")
                                        for i, j in T.Parallel(block_m, dim):
                                            num[i, j] = av[i, j] * 512.0
                                        T.vdiv(num, pwf, lv)
                                        T.vcast(lv, qi, round_mode="rint")
                                        T.vcast(qi, lv, round_mode="rint")
                                        if stage == "s4_enc":
                                            for i, j in T.Parallel(block_m, dim):
                                                u8[i, j] = T.cast(
                                                    T.min(T.max(lv[i, j], -128.0), 127.0),
                                                    "uint8")
                                            for i in T.Parallel(block_m):
                                                scale1[i] = scale2[i, 0]
                                            T.copy(u8, OutU8[row0, 0])
                                            T.copy(scale1, Scale[row0:row0 + block_m])
                                        else:
                                            for i, j in T.Parallel(block_m, dim):
                                                lv[i, j] = (8.0 * (t1[i, j] + 6.0)
                                                            + lv[i, j] - 128.0 * sbv[i, j])
                                            for i, j in T.Parallel(block_m, dim):
                                                u8[i, j] = T.cast(
                                                    T.min(T.max(lv[i, j], -128.0), 127.0),
                                                    "uint8")
                                            for i in T.Parallel(block_m):
                                                scale1[i] = scale2[i, 0]
                                            T.copy(u8, OutU8[row0, 0])
                                            T.copy(scale1, Scale[row0:row0 + block_m])

        return _fwd_main

    return _fwd_func
