# `_argreduce_kernel` (ArgminFwdOp, NPU)

Migrated TileLang kernel for `torch.argmin` (first-occurrence, `int64` output) on
`target="npuir"`, Developer mode. Implements the frozen `DESIGN.md` (v1) in this
directory.

## Files

| File | Role |
|------|------|
| `_argreduce_kernel.py` | Kernel + embedded PyTorch golden + layered L0/L1/L2/Boundary suite + `main` (`--level L0|all`) |
| `DESIGN.md` | Frozen Stage 1 design (algorithm, tiling, budgets, verification plan) |
| `REVIEW.md` | Stage 2 design review (conclusion: pass, 0 blockers) |
| `verify_equiv.py` | Stage 1 machine equivalence proof (E1–E5 `EQUIV_PASS` + controls) |
| `RETROSPECTIVE.md` | Stage 1/2/3 retrospectives (append-only) |
| `repro/` | Promoted probes (self-contained + assertions) |

## Contract

* Factory: `_argreduce_kernel(M, N, op_kind, dtype) -> _func(block_m) -> main(x) -> out`
* `op_kind in {"argmin", "argmax"}`, `dtype in {float16, bfloat16, float32}`
* `x`: `(M, N_padded)`, `N_padded = align_up(N, 256)` — the Op layer
  (`ArgminFwdOp._kernel_handles_padding = False`, `_pad_value() = +inf`) pads
  before calling; the kernel declares `(M, N_padded)` (declared width must equal
  the passed width).
* `out`: `(M,)` `int64`, first-occurrence minimum index.

## Algorithm (DESIGN.md §1.4)

Two paths selected by the N residency threshold (`N_padded * B/elem * bm <= 64 KB`):

* **Resident (path S)** — `reduce_min` row extreme → `vbrc` broadcast →
  `vcmp "eq"` → `vselect(idx_j, BIG)` → `reduce_min` first index, over the whole
  `N_padded` domain (pad columns are `+inf`, index-neutral).
  Explicit vector chain (the 2-D `T.Parallel` + `if_then_else` fused form
  scalarizes at `bm >= 2`).
* **Tiled online (path T)** — for N that does not fit UB (e.g. `(4, 102400)`):
  tile-0 direct-write running state + strict-less `(min, first-idx)` recurrence
  with no-alias `vselect` updates; `bm = 1` forced. The candidate uses the
  `bm == 1` fused form so the frozen `tile_n` budget (fp16 5120 / bf16 4096)
  holds.

`fp16` stays in its native domain; `bf16` is pre-cast to `fp32` (reduce does not
support bf16); `fp32` uses same-dtype copies (`f32 -> f32 vcast` is
non-identity).

## Run

```bash
TILELANG_ASCEND_MODE=Developer python _argreduce_kernel.py --level L0
TILELANG_ASCEND_MODE=Developer python _argreduce_kernel.py --level all
python repro/TRAP-npuir-reshape-dst-buffer-shadow.py
```

## Verified status (Stage 3, attempt 1)

| Level | Result |
|-------|--------|
| L0 (gate, blocking) | 20 cases PASS |
| L1 (blocking) | 11 cases PASS |
| L2 (warn only) | 3 cases PASS + NaN/assertion records |
| Boundary (warn only) | 3 cases PASS |

Comparison is exact (`torch.equal`, `max_diff = 0`). Documented non-equivalence:
a NaN input row yields the sentinel `2**30` (the `+inf` mask never matches),
while `torch.argmin` returns the first NaN index (DESIGN.md §9.2-R1).

Toolchain: tilelang `0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce` +
CANN 9.0.0 + torch_npu 2.9.0.post8 + Ascend910B3 (20 AICore / 40 AIVector / 192 KB UB).
