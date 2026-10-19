"""Floor probes: quantify msprof Task-Duration fixed overhead + DMA span.

PROBE_STAGE:
  empty      : write only the (block_m,) scale vector (no X read, minimal work)
  read       : DMA X -> UB, touch one element, write scale
  readwrite  : DMA X -> UB, cast to u8, DMA UB -> OutU8, write scale
"""

import functools
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

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
    variant = os.environ.get("PROBE_STAGE", "empty")
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

                for it in T.serial(iters):
                    tile = it * num_blocks + cid
                    if tile < n_tiles:
                        row0 = T.min(tile * block_m, rows - block_m)
                        if variant == "empty":
                            for i in T.Parallel(block_m):
                                scale1[i] = 1.0
                        elif variant == "read":
                            T.copy(X[row0, 0], x_in)
                            for i in T.Parallel(block_m):
                                scale1[i] = T.cast(x_in[i, 0], "float32")
                        else:
                            T.copy(X[row0, 0], x_in)
                            for i, j in T.Parallel(block_m, dim):
                                u8[i, j] = T.cast(
                                    T.min(T.max(T.cast(x_in[i, j], "float32"), -128.0), 127.0),
                                    "uint8")
                            T.copy(u8, OutU8[row0, 0])
                            for i in T.Parallel(block_m):
                                scale1[i] = 1.0
                        T.copy(scale1, Scale[row0:row0 + block_m])

        return _fwd_main

    return _fwd_func
