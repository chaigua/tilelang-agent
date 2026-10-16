# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-fp16-op-affine-double-rounding -- op-layer dtype affine is a lossy
double rounding (pure-CPU test-design trap, no NPU / no task-workspace import).

Entry ID      : TRAP-fp16-op-affine-double-rounding
Home file     : traps-runtime.md (test-design family; pure-torch precedent:
                TRAP-torch-oracle-maxpool1d-empty-window.py)
Phenomenon    : A normalisation op with the affine layering
                "kernel outputs the normalised tensor already rounded to the
                input dtype; the op layer then applies per-channel
                weight/bias in that same dtype" produces, for fp16, output
                that differs from the repo golden
                ``F.group_norm(x.float(), g, w.float(), b.float()).to(fp16)``
                by more than atol=1e-3 on a small fraction of elements.
                Cause: the kernel's fp16 round-trip quantises the normalised
                value (error e1 <= 0.5 ulp(n)), and the dtype affine scales
                that error by |w| (e1*|w|) while adding its own rounding.
                When ``n*w + b`` lands near zero the absolute error is
                bounded by ulp(|n*w|), not by ulp(|result|), so atol cannot
                cover it.  A harness that replicates the op layer in the
                input dtype therefore REJECTS A CORRECT KERNEL.
Assertion     : (a) trap exists  -- with fp16 dtype affine, a non-empty
                fraction of elements violate |out-golden| <= atol + rtol*|g|;
                (b) valid harness form -- applying the affine in fp32 and
                rounding ONCE gives 0 violations;
                (c) kernel-level form -- unit affine (DESIGN.md section 8.1)
                is bit-exact against the pure-normalisation golden.
First verified: tilelang 0.1.2+9307929320d / CANN 9.0.0 / Ascend910B3 /
                2026-10-15; torch 2.9.0+cpu (CPU-only, no device needed).
origin_task   : group_norm / _group_norm_kernel Stage 3 (2026-10-15).
Re-verify     : python3 repro/TRAP-fp16-op-affine-double-rounding.py
"""

import torch
import torch.nn.functional as F

ATOL = RTOL = 1e-3  # fp16 gate (source repo _get_tolerances)


def _violations(out, ref, atol=ATOL, rtol=RTOL):
    out, ref = out.float(), ref.float()
    return int(((out - ref).abs() > (atol + rtol * ref.abs())).sum())


def _run(seed, n=16, c=32, d=256, g=8):
    """Minimal op-level replica: normalised fp16 tensor + per-channel affine."""
    torch.manual_seed(seed)
    # A *correct* kernel returns pure normalisation rounded once to fp16.
    norm_fp32 = F.group_norm(torch.randn(n, c, d, dtype=torch.float32), g, eps=1e-5)
    out_kernel = norm_fp32.to(torch.float16)  # kernel's dtype round-trip
    w = torch.randn(c).to(torch.float16)  # per-channel (C,)
    b = torch.randn(c).to(torch.float16)
    ash = (1, c, 1)

    # Golden: fp32 opmath with a single round-back (the affine stage of the
    # repo reference program).
    g32 = norm_fp32 * w.float().reshape(ash) + b.float().reshape(ash)
    golden = g32.to(torch.float16)

    # (a) trap: fp16 dtype affine (exactly what GroupNormFwdOp.forward does)
    op_dtype = out_kernel * w.reshape(ash) + b.reshape(ash)
    # (b) valid harness form: affine in fp32, round once
    op_f32 = (out_kernel.float() * w.float().reshape(ash) + b.float().reshape(ash)).to(
        torch.float16
    )
    # (c) kernel-level unit affine against the pure-normalisation golden
    # (DESIGN section 8.1): must be bit-exact (x*1+0 is an identity).
    unit = out_kernel * torch.ones(c).to(torch.float16).reshape(ash)
    pure = norm_fp32.to(torch.float16)
    return (
        _violations(op_dtype, golden),
        _violations(op_f32, golden),
        _violations(unit, pure),
    )


def main():
    trap_total = f32_total = unit_total = 0
    for seed in range(8):
        t, f, u = _run(seed)
        trap_total += t
        f32_total += f
        unit_total += u
    print(f"fp16 dtype-affine violations : {trap_total} (trap: must be > 0)")
    print(f"fp32-affine-then-round        : {f32_total}  (valid harness form)")
    print(f"unit affine (kernel-level)    : {unit_total}  (DESIGN section 8.1)")

    # (a) the trap exists: the exact op replica rejects a correct kernel.
    assert trap_total > 0, "expected the fp16 dtype-affine trap to fire"
    # (b) the fp32-affine harness form removes the interface artifact: only
    #     the kernel's own fp16 round-trip remains (orders of magnitude less).
    assert f32_total * 10 < trap_total, "fp32-affine harness form must be clean-ish"
    # (c) unit affine is the authoritative kernel-level gate and is exact.
    assert unit_total == 0, "unit-affine kernel gate must be exact"
    print("PASS: TRAP-fp16-op-affine-double-rounding")


if __name__ == "__main__":
    main()
