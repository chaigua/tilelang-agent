"""Generate AnyFwdOp Stage-4 candidate files from the frozen Stage-3 baseline.

Every candidate is a full standalone file (kernel + golden + L0/L1/L2/Boundary
tests + main) derived from ``../_logical_reduce_kernel.py`` by explicit text
transforms, so the harness contract and the precision suite stay byte-identical
apart from the intended change.

Usage:
  python make_candidates.py --spec f16plane --out _logical_reduce_kernel_round1_op1_f16plane.py
  python make_candidates.py --spec f16_tile16k --out _logical_reduce_kernel_round1_op2_f16_tile16k.py
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.abspath(os.path.join(HERE, "..", "_logical_reduce_kernel.py"))

BLOCK_MARKERS = (
    ("resident", "# Path R -- resident, whole row in UB"),
    ("tiled", "# Path C -- column-tiled with no-alias running min accumulator"),
)


def split_blocks(src):
    """Split the source into head / resident / tiled / tiled_tail / tail."""
    i_res = src.index("def _build_resident(")
    i_tiled = src.index("def _build_tiled(")
    i_tail = src.index("def _build_tiled_tail(")
    i_factory = src.index("# Factory (harness contract")
    return {
        "head": src[:i_res],
        "resident": src[i_res:i_tiled],
        "tiled": src[i_tiled:i_tail],
        "tail": src[i_tail:i_factory],
        "rest": src[i_factory:],
    }


def to_f16_plane(block):
    """Switch the indicator / accumulator / epilogue chain of one builder to f16.

    Line-oriented on purpose: the baseline's indentation varies between the
    resident and the column-tiled builders, so substring replacement of whole
    lines would leave stray indentation behind.
    """
    acc_names = ("acc", "new_acc")
    plane_names = ("ind", "mn", "acc_flat", "mn_flat", "one_v", "any_v")
    out_lines = []
    for line in block.splitlines():
        stripped = line.strip()
        # the f32 -> f16 epilogue hop is redundant once the chain is f16
        if stripped.startswith("any_f16 = T.alloc_fragment"):
            continue
        if stripped.startswith("T.vcast(any_v, any_f16"):
            continue
        if "T.vcast(any_f16, out_ub" in line:
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(
                indent + 'T.vcast(any_v, out_ub, round_mode="rint")'
                "  # f16 -> i8 (single hop)"
            )
            continue
        replaced = False
        for name in acc_names + plane_names:
            pat = f"{name} = T.alloc_"
            if pat in line and '"float32"' in line and "zero" not in line:
                dtype_const = "ACC_DTYPE" if name in acc_names else "IND_DTYPE"
                out_lines.append(line.replace('"float32"', dtype_const))
                replaced = True
                break
        if replaced:
            continue
        if "# i1 -> f32" in line:
            out_lines.append(line.replace("# i1 -> f32", "# i1 -> f16"))
            continue
        if "# f32 -> f16" in line and "any_f16" not in line:
            out_lines.append(line.replace("# f32 -> f16", "# f16 carrier"))
            continue
        if "T.fill(one_v, tir.const(1," in line:
            out_lines.append(line.replace('"float32"', "IND_DTYPE"))
            continue
        if "T.fill(acc, tir.const(1," in line:
            out_lines.append(line.replace('"float32"', "ACC_DTYPE"))
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if block.endswith("\n") else "")


def add_constants(head, extra):
    """Insert tuned module-level constants right after the E_CAP block."""
    anchor = 'DEFAULT_ALIGNMENT = 256\n'
    assert anchor in head
    return head.replace(anchor, anchor + extra, 1)


def set_docstring_note(src, note):
    anchor = 'Run: ``python _logical_reduce_kernel.py --level all``\n"""'
    assert anchor in src
    return src.replace(anchor, note + '\n\nRun: ``python _logical_reduce_kernel.py --level all``\n"""', 1)


SPECS = {
    # OP1: indicator / accumulator / epilogue chain in float16 (path R + path C
    # without tail).  The static-width tail block keeps the float32 chain.
    "f16plane": {
        "note": (
            "Stage 4 candidate OP1 (round 1): the zero-hit indicator plane, the\n"
            "running minimum accumulator and the complemented epilogue run in\n"
            "**float16** (0/1 indicator values are exactly representable, so the\n"
            "reduction stays bit-exact); the static-width tail block keeps the\n"
            "float32 chain (narrow-tail float16 mis-reduce, VP-2026-0326)."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: indicator plane + reduction carrier\n'
            'ACC_DTYPE = "float16"  # tuned: running-min accumulator carrier\n'
        ),
        "blocks": ("resident_f16", "tiled_f16", "tail_f32"),
        "select_config": None,
    },
    # OP2: float16 chain + wider column tile (16384 elements) for the
    # column-tiled path.
    "f16_tile16k": {
        "note": (
            "Stage 4 candidate OP2 (round 1): OP1's float16 chain plus a wider\n"
            "column tile (16384 elements, the measured fp16/fp32 hard pass point)\n"
            "for the column-tiled path; 8192 stays for bfloat16."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: indicator plane + reduction carrier\n'
            'ACC_DTYPE = "float16"  # tuned: running-min accumulator carrier\n'
            'TILE_CAP_ELEMS = {"float16": 16384, "float32": 16384, "bfloat16": 8192}\n'
        ),
        "blocks": ("resident_f16", "tiled_f16", "tail_f32"),
        "select_config": "tile_cap",
    },
    # R2-OP1: OP1's float16 chain + software pipelining of the column-tile loop.
    "f16_pipe2": {
        "note": (
            "Stage 4 candidate R2-OP1 (round 2): OP1's float16 chain plus an\n"
            "explicit two-stage ``T.Pipelined`` on the column-tile loop (overlap\n"
            "the GM->UB tile read with the indicator/reduce chain)."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: indicator plane + reduction carrier\n'
            'ACC_DTYPE = "float16"  # tuned: running-min accumulator carrier\n'
            'C_TILE_STAGES = 2  # tuned: T.Pipelined stages on the column-tile loop\n'
        ),
        "blocks": ("resident_f16", "tiled_f16", "tail_f32"),
        "select_config": None,
        "pipe2": True,
    },
    # FINAL: float16 chain on the resident path only (the column-tiled path
    # measured no_gain for every variant and keeps the Stage-3 f32 chain).
    "f16res": {
        "note": (
            "Stage 4 final: the resident path (``N <= E_CAP``) runs the zero-hit\n"
            "indicator plane, the row-minimum and the complemented epilogue in\n"
            "**float16** (0/1 indicator values are exactly representable, so the\n"
            "reduction stays bit-exact and a single f16->i8 hop replaces the\n"
            "f32->f16->i8 epilogue).  The column-tiled path (``N > E_CAP``) keeps\n"
            "the Stage-3 float32 chain: four structural variants (16384-wide tile,\n"
            "deferred element-wise vmin plane, unrolled tile loop, two-stage\n"
            "pipelining of the tile and of the persistent task loop) were each\n"
            "measured against it and all landed inside the +-3% noise band, so the\n"
            "smaller code surface wins.  The static-width tail block keeps the\n"
            "float32 chain as well (narrow-tail float16 mis-reduce, VP-2026-0326)."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: resident-path indicator plane carrier\n'
            'ACC_DTYPE = "float16"  # tuned: resident-path row-minimum carrier\n'
        ),
        "blocks": ("resident_f16",),
        "select_config": None,
    },
    # R3-OP1: static unrolling of the column-tile loop.
    "f16_unroll": {
        "note": (
            "Stage 4 candidate R3-OP1 (round 3): OP1's float16 chain plus a\n"
            "statically unrolled column-tile loop (``T.unroll``) so the scheduler\n"
            "can interleave the tile copies with the indicator/reduce chain."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: indicator plane + reduction carrier\n'
            'ACC_DTYPE = "float16"  # tuned: running-min accumulator carrier\n'
        ),
        "blocks": ("resident_f16", "tiled_f16", "tail_f32"),
        "select_config": None,
        "loop": "unroll",
    },
    # R3-OP2: pipelined outer (persistent task) loop.
    "f16_taskpipe": {
        "note": (
            "Stage 4 candidate R3-OP2 (round 3): OP1's float16 chain plus a\n"
            "two-stage ``T.Pipelined`` on the outer persistent task loop (overlap\n"
            "the next row block's tile reads with the current block's chain)."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: indicator plane + reduction carrier\n'
            'ACC_DTYPE = "float16"  # tuned: running-min accumulator carrier\n'
        ),
        "blocks": ("resident_f16", "tiled_f16", "tail_f32"),
        "select_config": None,
        "loop": "taskpipe",
    },
    # OP3: float16 chain + deferred reduction (element-wise vmin plane).
    "f16_defer": {
        "note": (
            "Stage 4 candidate OP3 (round 1): OP1's float16 chain plus a deferred\n"
            "reduction -- each column tile merges element-wise with ``vmin`` into a\n"
            "resident float16 plane, and a single ``reduce_min`` runs once per row\n"
            "block instead of once per tile."
        ),
        "constants": (
            'IND_DTYPE = "float16"  # tuned: indicator plane + reduction carrier\n'
            'ACC_DTYPE = "float16"  # tuned: running-min accumulator carrier\n'
        ),
        "blocks": ("resident_f16", "tiled_f16", "tail_f32"),
        "select_config": None,
    },
}

SELECT_TILE_CAP = '''        tile_n = _pick_tile_n(N, min(E_CAP, N))
'''
SELECT_TILE_CAP_NEW = '''        tile_n = _pick_tile_n(N, min(TILE_CAP_ELEMS[dtype], N))
'''


def defer_reduce(block):
    """Replace the per-tile (bm,1) reduce+min with an element-wise plane merge."""
    out = block
    out = out.replace(
        '''                mn = T.alloc_fragment((block_m, 1), IND_DTYPE)
                acc = T.alloc_fragment((block_m, 1), ACC_DTYPE)
                new_acc = T.alloc_fragment((block_m, 1), ACC_DTYPE)
                acc_flat = T.alloc_fragment((block_m,), IND_DTYPE)''',
        '''                mn = T.alloc_fragment((block_m, 1), IND_DTYPE)
                acc = T.alloc_shared((block_m, tile_n), ACC_DTYPE)
                acc_new = T.alloc_shared((block_m, tile_n), ACC_DTYPE)
                mn_tile = T.alloc_fragment((block_m, 1), IND_DTYPE)
                acc_flat = T.alloc_fragment((block_m,), IND_DTYPE)''')
    out = out.replace(
        '''                T.fill(acc, tir.const(1, ACC_DTYPE))''',
        '''                T.fill(acc, tir.const(1, ACC_DTYPE))  # min identity''')
    out = out.replace(
        '''                            T.reduce_min(ind, mn, dim=1, clear=True)
                            # No-alias accumulate: the OR merge is a min over
                            # the zero-hit indicator (never vadd / vmax here).
                            T.vmin(acc, mn, new_acc)
                            T.copy(new_acc, acc)''',
        '''                            # Deferred merge: keep a full-width plane and
                            # fold each tile in with an element-wise vmin (the
                            # OR merge is a min over the zero-hit indicator --
                            # never vadd / vmax here); one reduce at the end.
                            T.vmin(acc, ind, acc_new)
                            T.copy(acc_new, acc)''')
    out = out.replace(
        '''                        T.reshape(acc, acc_flat)''',
        '''                        T.reduce_min(acc, mn_tile, dim=1, clear=True)
                        T.reshape(mn_tile, mn_flat_acc)''')
    out = out.replace(
        'T.vsub(one_v, acc_flat, any_v)',
        'T.vsub(one_v, mn_flat_acc, any_v)')
    return out


def build(spec_name, out_path):
    spec = SPECS[spec_name]
    with open(BASELINE) as f:
        src = f.read()
    blocks = split_blocks(src)
    head = add_constants(blocks["head"], spec["constants"])
    head = set_docstring_note(head, spec["note"])

    res = blocks["resident"]
    til = blocks["tiled"]
    tail = blocks["tail"]
    if "resident_f16" in spec["blocks"]:
        res = to_f16_plane(res)
    if "tiled_f16" in spec["blocks"]:
        til = to_f16_plane(til)
        if spec_name == "f16_defer":
            til = defer_reduce(til)
        if spec.get("loop") == "unroll":
            old = "for t in T.serial(num_full):  # static bound"
            assert old in til
            til = til.replace(old, "for t in T.unroll(num_full):  # static bound")
        if spec.get("loop") == "taskpipe":
            old = "for s in T.serial(num_local_tasks):"
            assert old in til
            til = til.replace(old, "for s in T.Pipelined(num_local_tasks, num_stages=2):")
        if spec.get("pipe2"):
            old = "for t in T.serial(num_full):  # static bound"
            assert old in til
            til = til.replace(old, "for t in T.Pipelined(num_full, num_stages=C_TILE_STAGES):  # static bound")
    rest = blocks["rest"]
    if spec["select_config"] == "tile_cap":
        head = head.replace(SELECT_TILE_CAP, SELECT_TILE_CAP_NEW)
        assert SELECT_TILE_CAP_NEW in head

    new_src = head + res + til + tail + rest
    with open(os.path.join(HERE, out_path), "w") as f:
        f.write(new_src)
    print("wrote", out_path, len(new_src), "bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=tuple(SPECS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.spec, args.out)
