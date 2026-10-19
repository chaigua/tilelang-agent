"""Workload definitions for the norm op family.

Ported from TileOPs ``workloads/normalization.py`` --
``AdaLayerNormWorkload`` and ``GroupNormWorkload`` (the other norm
workloads are not migrated).
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class AdaLayerNormWorkload(WorkloadBase):
    """Workload definition for AdaLayerNormFwdOp.

    Generates the three same-shape operands (``x`` / ``scale`` / ``shift``)
    via ``randn``.  Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        return x, scale, shift


class GroupNormWorkload(WorkloadBase):
    """Workload definition for GroupNormFwdOp.

    Ported from TileOPs ``workloads/normalization.py:GroupNormWorkload``.
    Input count is preserved: ``(x, weight, bias)`` where ``x`` has shape
    ``(n, c, *spatial)`` and ``weight``/``bias`` are per-channel ``(c,)``
    (the Op layer's affine layout; the kernel's row-broadcast affine is
    derived inside the kernel).  Adaptation point W1: ``device`` is resolved
    from :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, n: int, c: int, spatial: tuple, g: int,
                 dtype: torch.dtype, eps: float = 1e-5):
        self.n = n
        self.c = c
        self.spatial = spatial
        self.g = g
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        shape = (self.n, self.c, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device=device)
        weight = torch.randn(self.c, dtype=self.dtype, device=device)
        bias = torch.randn(self.c, dtype=self.dtype, device=device)
        return x, weight, bias
