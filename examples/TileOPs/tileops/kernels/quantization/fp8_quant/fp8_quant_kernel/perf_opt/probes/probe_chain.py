"""Cumulative-chain probe for `_fp8_quant_kernel` (Stage 4).

Adds the encode-chain ops in baseline order, one accumulation step at a time,
and stores the latest result (so nothing is DCE'd).  Also honours
``PROBE_BLOCK_M`` to test the UB-capacity hypothesis (block_m drives every
(block_m, dim) fp32 buffer size).

PROBE_STAGE: c0 (base qv) / c1 (+sign) / c2 (+exp/log) / c3 (+vshl) /
             c4 (+vdiv) / c5 (+byte) / full (baseline-equivalent)
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


@functools.lru_cache(maxsize=128)
def _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype):
    variant = os.environ.get("PROBE_STAGE", "full")
    block_elems = int(os.environ.get("PROBE_BLOCK_M", "0")) or BLOCK_ELEMS
    rows = batch * seq_len_kv * kv_group
    dim = index_dim
    block_m = max(1, min(block_elems // dim, rows))
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
                        for i, j in T.Parallel(block_m, dim):
                            x32[i, j] = T.cast(x_in[i, j], "float32")
                        T.vabs(x32, av)
                        T.reduce(av, amax, dims=[1], reduce_mode="max")
                        for i in T.Parallel(block_m):
                            amax_c[i, 0] = T.max(amax[i, 0], AMAX_EPS)
                        T.vdiv(amax_c, c448, scale2)
                        T.vdiv(x32, scale2, qv)
                        T.vclamp(qv, qv, FP8_MIN, FP8_MAX)

                        T.vabs(qv, av)
                        if variant != "c0":
                            T.vcmp(qv, av, cond, "lt")
                            T.vcast(cond, sbv, round_mode="rint")
                        if variant in ("c2", "c3", "c4", "c5", "full"):
                            T.vclamp(av, av, 1e-30, FP8_MAX)
                            T.vlog2(av, lv, t1)
                            T.vcast(lv, ei, round_mode="floor")
                            T.vclamp(ei, ei, -6, 8)
                            T.vcast(ei, t1, round_mode="rint")
                        if variant in ("c3", "c4", "c5", "full"):
                            for i, j in T.Parallel(block_m, dim):
                                qi[i, j] = ei[i, j] + 6
                            T.vshl(one_i, qi, pw)
                            T.vcast(pw, pwf, round_mode="rint")
                        if variant in ("c4", "c5", "full"):
                            for i, j in T.Parallel(block_m, dim):
                                num[i, j] = av[i, j] * 512.0
                            T.vdiv(num, pwf, lv)
                            T.vcast(lv, qi, round_mode="rint")
                            T.vcast(qi, lv, round_mode="rint")
                        if variant in ("c5", "full"):
                            for i, j in T.Parallel(block_m, dim):
                                lv[i, j] = (8.0 * (t1[i, j] + 6.0) + lv[i, j]
                                            - 128.0 * sbv[i, j])

                        # store: pick the freshest f32 buffer for the variant
                        if variant in ("c5", "full"):
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(
                                    T.min(T.max(lv[i, j], -128.0), 127.0), "uint8")
                        elif variant == "c4":
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(
                                    T.min(T.max(lv[i, j], -128.0), 127.0), "uint8")
                        elif variant == "c3":
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(T.min(T.max(pwf[i, j], -128.0), 127.0), "uint8")
                        elif variant == "c2":
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(T.min(T.max(t1[i, j], -128.0), 127.0), "uint8")
                        elif variant == "c1":
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(T.min(T.max(sbv[i, j], -128.0), 127.0), "uint8")
                        else:
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(T.min(T.max(qv[i, j], -128.0), 127.0), "uint8")

                        for i in T.Parallel(block_m):
                            scale1[i] = scale2[i, 0]
                        T.copy(u8, OutU8[row0, 0])
                        T.copy(scale1, Scale[row0:row0 + block_m])

        return _fwd_main

    return _fwd_func
