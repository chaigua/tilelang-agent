"""Materialise Stage 4 candidate files from the knob sandbox.

Each candidate is a full copy of ``_logical_reduce_kernel_lab.py`` with its own
``KNOBS`` dict, so every branch has a distinct artifact hash (perf_records
traceability) and can run its own L0 regression.  A candidate changes exactly
ONE structural point relative to its parent.

Usage: python make_candidates.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "_logical_reduce_kernel_lab.py")

CANDIDATES = {
    "_logical_reduce_kernel_round1_op1_rprime.py": {
        "E_CAP": 8192, "RPRIME": 1, "TILE_CAP": "ecap", "BM_POLICY": "ladder",
    },
    "_logical_reduce_kernel_round1_op2_tilecap_ub.py": {
        "E_CAP": 8192, "RPRIME": 0, "TILE_CAP": "ub", "BM_POLICY": "ladder",
    },
    "_logical_reduce_kernel_round1_op3_bm_cores.py": {
        "E_CAP": 8192, "RPRIME": 0, "TILE_CAP": "ecap", "BM_POLICY": "cores",
    },
    "_logical_reduce_kernel_round2_op4_rprime_tilecap.py": {
        "E_CAP": 8192, "RPRIME": 1, "TILE_CAP": "ub", "BM_POLICY": "ladder",
    },
    "_logical_reduce_kernel_round2_op5_rprime_bmcores.py": {
        "E_CAP": 8192, "RPRIME": 1, "TILE_CAP": "ecap", "BM_POLICY": "cores",
    },
}


def render(src, knobs):
    block = "KNOBS = {\n" + "".join(
        f'    "{k}": {v!r},\n' for k, v in knobs.items()
    ) + "}"
    return re.sub(r"KNOBS = \{.*?\n\}", block, src, count=1, flags=re.S)


def main():
    with open(LAB) as handle:
        src = handle.read()
    for name, knobs in CANDIDATES.items():
        out = os.path.join(HERE, name)
        with open(out, "w") as handle:
            handle.write(render(src, knobs))
        print("wrote", name, knobs)


if __name__ == "__main__":
    main()
