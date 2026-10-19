"""Workload definitions for the attention op family (NPU).

Ported from the GPU TileOPs ``workloads/attention/mha.py`` --
``MhaFwdWorkload`` and ``MhaBwdWorkload``.  ``MhaDecodeWorkload`` /
``MhaDecodePagedWorkload`` belong to ops that are not migrated yet.

``MhaBwdWorkload`` calls ``MultiHeadAttentionFwdOp`` to materialize the
forward output ``o`` and needs the per-row square-attention log-sum-exp,
which on the GPU comes from ``workloads/attention/gqa.py``.  The NPU
project has no ``workloads/attention/`` package, so that helper
(``_compute_gqa_square_lse``) is ported verbatim into this module.

The only adaptation (W1) is device placement: ``device='cuda'`` is
resolved through :func:`tileops.device.get_device_backend` instead of the
hard-coded ``"cuda"``.
"""

import math

import torch

from tileops.device import get_device_backend
from tileops.ops.attention.multi_head_attention import MultiHeadAttentionFwdOp
from tileops.workloads.workload_base import WorkloadBase

__all__ = ["MhaBwdWorkload", "MhaFwdWorkload"]


class MhaFwdWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionFwdOp.

    Generates the q/k/v triple in BSHD layout:
    ``(batch, seq_len, heads, dim)`` per tensor.
    """

    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        return q, k, v


def _compute_gqa_square_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
) -> torch.Tensor:
    """Per-row log-sum-exp of the square (S x S) attention scores, base-2.

    Ported verbatim from the GPU ``workloads/attention/gqa.py`` (device-agnostic:
    ``q.device`` / ``masked_fill`` / ``logsumexp``).  The fp32 score path and the
    ``* log2(e)`` conversion match the kernel's e-base online-softmax domain.
    """
    groups = heads // heads_kv
    seq_len = q.shape[1]
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * (dim**-0.5)
    if is_causal:
        pos = torch.arange(seq_len, device=q.device)
        mask = pos[None, :] <= pos[:, None]
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
    return torch.logsumexp(scores, dim=-1) * math.log2(math.e)


class MhaBwdWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionBwdOp.

    Generates the backward input tuple in BSHD layout:
    ``q``, ``k``, ``v`` (all ``requires_grad=True``, for the autograd
    reference), the forward output ``o`` (produced by
    :class:`MultiHeadAttentionFwdOp`), a random ``grad_output``, and the
    base-2 per-row log-sum-exp ``lse`` (shape ``(B, H, S)``, float32).
    """

    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=device,
            requires_grad=True,
        )
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device=device
        )

        fwd_op = MultiHeadAttentionFwdOp(
            self.batch, self.heads, self.seq_len, self.dim, self.is_causal, self.dtype
        )
        with torch.no_grad():
            result = fwd_op(q, k, v)
            o = result[0] if isinstance(result, tuple) else result
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse
