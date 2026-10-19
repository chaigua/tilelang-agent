# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-torch-oracle-maxpool1d-empty-window -- test-design trap (torch oracle).

Entry:      pattern-library/traps-runtime.md (test-oracle hazard, pooling family)
Symptom:    for a *1D windowed* max-pool window with no in-bounds input lane
            (reachable when dilation >= 2 and padding > 0, e.g. L_in=1, k=2,
            s=1, p=1, d=2 -> window lanes {-1, 1}), torch's CPU kernel returns
            an **out-of-range** index while the value is -inf.  A test that
            compares `indices` against `F.max_pool1d(..., return_indices=True)`
            therefore FAILS on a kernel that is correct with respect to the
            source state machine (which initialises `max_idx = 0` and skips
            out-of-bounds lanes -> `(value, index) = (-inf, 0)`).
Assertions: (1) the premise holds -- every window lane is out of range;
            (2) torch returns an out-of-range index (the hazard exists);
            (3) the source-aligned contract value is 0 and differs from torch;
            (4) control: for a non-empty-window config torch IS a valid oracle
                (index in range and the value equals the running maximum).
First verified: tilelang 0.1.2+892e2c18d80c37795727236b5ebbe145626a8313 / dev root
            build ba60e99c16439d6fa6a59acdfca162c17b2bf8b2 + torch 2.9.0+cpu
            + torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3 / 2026-10-14.
origin_task: max_pool1d_with_indices / _max_pool1d_with_indices_kernel Stage 3
            (DESIGN.md section 0.1 "empty-window contract", section 8.3 L0-15;
            verify_equiv.py empty_window_evidence()).
Re-verify:  python repro/TRAP-torch-oracle-maxpool1d-empty-window.py

No NPU and no task module is imported: this is a pure-torch oracle hazard.
"""

import torch
import torch.nn.functional as F


def _pool_output_dim(l_in, k, s, p, ceil_mode, d):
    """PyTorch max_pool1d output extent (source kernels/pool/common.py rule)."""
    effective = d * (k - 1) + 1
    if ceil_mode:
        out = (l_in + 2 * p - effective + s - 1) // s + 1
    else:
        out = (l_in + 2 * p - effective) // s + 1
    if ceil_mode and out > 0 and (out - 1) * s >= l_in + p:
        out -= 1
    return max(out, 0)


def _window_lanes(l_in, ow, k, s, p, d):
    return [ow * s - p + kw * d for kw in range(k)]


def main():
    # ---- empty-window regimes (d >= 2 and p > 0) ----
    empty_cfgs = [("EW-min", 1, 2, 1, 1, 2, False),
                  ("EW-ceil", 1, 2, 3, 1, 3, True)]
    for tag, l_in, k, s, p, d, ceil in empty_cfgs:
        out_l = _pool_output_dim(l_in, k, s, p, ceil, d)
        assert out_l >= 1, f"{tag}: output extent must be positive"
        for ow in range(out_l):
            lanes = _window_lanes(l_in, ow, k, s, p, d)
            assert all(iw < 0 or iw >= l_in for iw in lanes), (
                f"{tag}: window {ow} lanes {lanes} are not all out of [0,{l_in})"
            )
        x = torch.tensor([[5.0], [7.0]])
        val, idx = F.max_pool1d(x, k, s, p, d, ceil, return_indices=True)
        # (1) premise: value is the -inf identity of the max reduction
        assert bool((val == float("-inf")).all()), (
            f"{tag}: empty-window value {val.flatten().tolist()} != -inf"
        )
        # (2) hazard: torch's index is out of the legal [0, L_in) range
        assert int(idx.max()) >= l_in or int(idx.min()) < 0, (
            f"{tag}: torch index {idx.flatten().tolist()} is in range; hazard gone"
        )
        # (3) source-aligned contract (max_idx initialised to 0, lanes skipped)
        contract_idx = torch.zeros_like(idx)
        assert not torch.equal(idx, contract_idx), (
            f"{tag}: torch index unexpectedly equals the contract value"
        )
        print(f"{tag}: lanes out-of-range -> torch idx {idx.flatten().tolist()} "
              f"(OUT OF RANGE) vs contract idx 0; value -inf")

    # ---- control: non-empty windows, torch IS a valid oracle ----
    l_in, k, s, p, d = 6, 3, 1, 1, 1
    x = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]])
    val, idx = F.max_pool1d(x, k, s, p, d, False, return_indices=True)
    assert 0 <= int(idx.min()) and int(idx.max()) < l_in, (
        f"control: torch index {idx.flatten().tolist()} out of range"
    )
    # out_l = (6 + 2*1 - 3)//1 + 1 = 6 windows; the last one {4,5,6} skips iw=6.
    assert val.flatten().tolist() == [3.0, 4.0, 4.0, 5.0, 9.0, 9.0], (
        f"control: unexpected values {val.flatten().tolist()}"
    )
    print(f"control (non-empty windows): torch idx {idx.flatten().tolist()} "
          f"in range -> a valid oracle")


if __name__ == "__main__":
    main()
    print("TRAP-torch-oracle-maxpool1d-empty-window: PASS")
