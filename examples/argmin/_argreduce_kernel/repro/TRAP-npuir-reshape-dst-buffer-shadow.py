# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Probe (promoted): ``T.reshape`` is shadowed by ``npuir_reshape(src, dst)``.

Finding (source-level, verified on this toolchain)
--------------------------------------------------
On the ``npuir`` branch ``tilelang.language`` exports ``reshape`` twice:

  * ``tilelang/language/__init__.py:68``  -> generic ``customize.reshape(src, shape_list)``
  * ``tilelang/language/__init__.py:196`` -> ``npuir_reshape as reshape`` (last wins)

so the *effective* ``T.reshape`` is ``npuir_reshape(src, dst)`` whose SECOND
argument is a **destination buffer** (its shape is the target shape), not a
shape list.  Code (and doc examples) written for the generic form
``T.reshape(A, [N // M, M])`` therefore does not mean what it reads like on
npuir: passing a Python list as ``dst`` fails the ``_to_region`` path.

Correct npuir usage (as in ``_argreduce_kernel.py``):
    ``T.reshape(first, first_flat)``     # (bm, 1) -> (bm,)
    ``T.reshape(idx_src, idx_row)``      # (N,)   -> (1, N)

This probe asserts the shadowing so that a future toolchain that restores the
generic signature (or removes the alias) trips the assertion instead of
silently changing the meaning of existing kernel code.

Run:  python repro/TRAP-npuir-reshape-dst-buffer-shadow.py
"""

import inspect

import tilelang.language as T

# Effective symbol is the npuir destination-buffer form.
assert T.reshape.__name__ == "npuir_reshape", (
    f"T.reshape resolved to {T.reshape.__name__!r}; expected 'npuir_reshape' "
    f"(the npuir alias must shadow the generic reshape)"
)

sig = inspect.signature(T.reshape)
assert list(sig.parameters) == ["src", "dst"], (
    f"T.reshape parameters {list(sig.parameters)} != ['src', 'dst']; the npuir "
    f"form takes a destination BUFFER as the second argument, not a shape list"
)

# The generic (shape-list) form still exists under its own name.
from tilelang.language import customize as _customize  # noqa: E402

assert list(inspect.signature(_customize.reshape).parameters) == ["src", "shape"], (
    "customize.reshape(src, shape) generic form not found; probe assumption stale"
)

print(
    "PASS: T.reshape is npuir_reshape(src, dst) [dst = destination buffer]; "
    "generic customize.reshape(src, shape) is shadowed."
)
