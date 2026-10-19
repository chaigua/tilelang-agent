"""Stage 4 tuned kernel package for `_adaptive_max_pool2d_kernel`.

The tuned implementation lives in ``_adaptive_max_pool2d_kernel.py`` and is
imported explicitly by the wrapper's perf_opt branch:

    from .adaptive_max_pool2d_kernel.perf_opt._adaptive_max_pool2d_kernel import (
        _adaptive_max_pool2d_kernel, adaptive_bin, max_adaptive_bin_extent)

Deliberately exports nothing itself (no extra names, no import side effects).
"""
