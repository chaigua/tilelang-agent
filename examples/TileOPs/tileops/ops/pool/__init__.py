"""Pool op family (NPU-adapted).

Ported from the GPU TileOPs ``tileops/ops/pool.py`` module -- only the
operators migrated so far are re-exported here.
"""

from tileops.ops.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d import AdaptiveMaxPool2dFwdOp
from tileops.ops.pool.adaptive_max_pool2d_with_indices import (
    AdaptiveMaxPool2dIndicesFwdOp,
)
from tileops.ops.pool.max_pool1d import MaxPool1dFwdOp
from tileops.ops.pool.max_pool1d_with_indices import MaxPool1dIndicesFwdOp
from tileops.ops.pool.max_pool2d import MaxPool2dFwdOp

__all__ = [
    "AdaptiveAvgPool2dFwdOp",
    "AdaptiveMaxPool2dFwdOp",
    "AdaptiveMaxPool2dIndicesFwdOp",
    "MaxPool1dFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
]
