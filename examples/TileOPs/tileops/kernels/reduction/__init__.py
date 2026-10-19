"""Reduction kernel family (NPU-adapted).

Ported from the GPU TileOPs ``tileops/kernels/reduction/`` package -- only
the operators migrated so far are re-exported here.

Several ops share a GPU kernel class name but live in their own NPU op
packages, so the colliding copies are re-exported under aliases:

- ``ArgmaxFwdOp`` / ``ArgminFwdOp`` share the GPU ``ArgreduceKernel`` class
  name -> the ``argmin`` copy is ``ArgminArgreduceKernel``.
- ``AllFwdOp`` / ``CountNonzeroFwdOp`` share the GPU ``LogicalReduceKernel``
  class name -> the ``all`` copy is ``AllLogicalReduceKernel``.
- ``AnyFwdOp`` also drives the GPU ``LogicalReduceKernel`` class name -> the
  ``any`` copy is ``AnyLogicalReduceKernel``.

Sorted import block (ruff I001); the alias keeps the family-level namespace
collision-free.
"""

from tileops.kernels.reduction.all import LogicalReduceKernel as AllLogicalReduceKernel
from tileops.kernels.reduction.any import LogicalReduceKernel as AnyLogicalReduceKernel
from tileops.kernels.reduction.argmax import ArgreduceKernel
from tileops.kernels.reduction.argmin import ArgreduceKernel as ArgminArgreduceKernel
from tileops.kernels.reduction.count_nonzero import LogicalReduceKernel
from tileops.kernels.reduction.logsumexp import LogSumExpKernel

__all__ = [
    "AllLogicalReduceKernel",
    "AnyLogicalReduceKernel",
    "ArgminArgreduceKernel",
    "ArgreduceKernel",
    "LogSumExpKernel",
    "LogicalReduceKernel",
]
