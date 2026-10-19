"""Stage 4 (perf_opt) package for `_max_pool1d_with_indices_kernel`.

The Stage 5 wrapper imports the tuned drop-in kernel from here:

    from .max_pool1d_with_indices_kernel.perf_opt._max_pool1d_with_indices_kernel \
        import _max_pool1d_with_indices_kernel, pool_output_dim

The factory signature and the dual-output contract are identical to the Stage 3
baseline (`main(x) -> (out, indices)`, `out_idx=[1, 2]`), so the import is a pure
drop-in swap.  See `opt_log.md` for the measured deltas and the tuning history.
"""
