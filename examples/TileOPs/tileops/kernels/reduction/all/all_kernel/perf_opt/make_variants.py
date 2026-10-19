"""Materialise Stage 4 dispatch variants from the tuned template.

Each variant differs from the template in exactly one module-level dispatch
constant (``TUNED_RPRIME`` / ``TUNED_TILE_CAP_MODE``), i.e. one structural
point, and is measured from the same template code otherwise.

For *screening* the variant files relax the Stage 3 design-period dispatch
assertions in ``_run_tensor`` (they assert the pre-tuning path / tile / block_m
values) and instead assert the dispatch **invariants** that must hold for any
variant:

  * the per-block element count never exceeds the measured UB hard cap
    (``_select_config`` already raises otherwise);
  * path ``R``/``R'`` means ``tile_n == N`` and ``num_full == 1``;
  * path ``C`` means ``num_full * tile_n + tail == N`` (coverage, no overlap).

The bit-exact golden comparison (``_exact_compare`` / ``torch.equal``) is left
untouched -- that is the real L0 gate.  The delivered final file restores the
strict assertions with the adopted concrete values.

Usage: python make_variants.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "_logical_reduce_kernel_tuned_tpl.py")

VARIANTS = {
    "_logical_reduce_kernel_round3_op4_rprime.py": {"TUNED_RPRIME": "True", "TUNED_TILE_CAP_MODE": '"ecap"'},
    "_logical_reduce_kernel_round3_op5_tilecap.py": {"TUNED_RPRIME": "False", "TUNED_TILE_CAP_MODE": '"ub"'},
    "_logical_reduce_kernel_round3_op7_rprime_tilecap.py": {"TUNED_RPRIME": "True", "TUNED_TILE_CAP_MODE": '"ub"'},
}

OLD_RUN_TENSOR = '''    M, N = x.shape
    cfg = _select_config(M, N, dtype_str)
    if expect_path is not None and cfg["path"] != expect_path:
        raise AssertionError(f"[{tag}] path={cfg['path']} != expected {expect_path}")
    if expect_block_m is not None and cfg["block_m"] != expect_block_m:
        raise AssertionError(
            f"[{tag}] block_m={cfg['block_m']} != expected {expect_block_m} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )
    if expect_tile_n is not None and cfg["tile_n"] != expect_tile_n:
        raise AssertionError(
            f"[{tag}] tile_n={cfg['tile_n']} != expected {expect_tile_n} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )'''

NEW_RUN_TENSOR = '''    M, N = x.shape
    cfg = _select_config(M, N, dtype_str)
    # Screening variant: the design-period path/tile/block_m expectations are
    # replaced by dispatch invariants (the full bit-exact golden comparison
    # below is unchanged and remains the gate).
    if cfg["path"] == "R":
        assert cfg["tile_n"] == N and cfg["num_full"] == 1 and cfg["tail"] == 0, (
            f"[{tag}] resident path must be a single full-width tile: {cfg}"
        )
    else:
        assert (
            cfg["num_full"] * cfg["tile_n"] + cfg["tail"] == N
        ), f"[{tag}] column tiling must cover N exactly: {cfg}"
    assert cfg["block_m"] * min(cfg["tile_n"], N) <= HARD_CAP_ELEMS[dtype_str], (
        f"[{tag}] element budget violated: {cfg}"
    )'''


TILE_ASSERT_RE = re.compile(
    r'if \(cfg\["num_full"\], cfg\["tail"\]\) != \([0-9]+, [0-9]+\):\n'
    r'\s+raise AssertionError\([^\n]*\)\n'
)


def relax_tile_assertions(src):
    """Screening only: drop the design-period (num_full, tail) assertions."""
    return TILE_ASSERT_RE.sub(
        "# screening variant: design-period tile arithmetic assertion relaxed\n", src
    )


def main():
    with open(TPL) as handle:
        src = handle.read()
    if src.count(OLD_RUN_TENSOR) != 1:
        raise SystemExit("run_tensor block not unique")
    for name, knobs in VARIANTS.items():
        out_src = relax_tile_assertions(src.replace(OLD_RUN_TENSOR, NEW_RUN_TENSOR))
        for key, value in knobs.items():
            pattern = re.compile(rf"^{key} = .*$", re.M)
            if len(pattern.findall(out_src)) != 1:
                raise SystemExit(f"constant {key} not unique")
            out_src = pattern.sub(f"{key} = {value}", out_src)
        with open(os.path.join(HERE, name), "w") as handle:
            handle.write(out_src)
        print("wrote", name, knobs)


if __name__ == "__main__":
    main()
