"""Materialise experiment-branch files from the sandbox (`_logical_reduce_kernel_lab.py`).

Each candidate gets its own file so ``perf_records.jsonl`` can pin a distinct
``artifact_path`` + ``artifact_sha256`` (env-knob variants would share a hash).

Usage: python make_candidates.py            # writes the ROUND1 list below
"""

import hashlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "_logical_reduce_kernel_lab.py")

# candidate file suffix -> KNOBS
ROUND1 = {
    "op1_e8192_rprime0": {"E_CAP": 8192, "UB_ELEMS": 24576, "RPRIME": 0},
    "op2_e8192_rprime1": {"E_CAP": 8192, "UB_ELEMS": 24576, "RPRIME": 1},
    "op3_e16384_rprime0": {"E_CAP": 16384, "UB_ELEMS": 24576, "RPRIME": 0},
}

KNOB_RE = re.compile(r"KNOBS = \{.*?\n\}\n", re.S)


def materialise(suffix, knobs):
    src = open(LAB).read()
    block = "KNOBS = {\n" + "".join(
        f'    "{k}": {v},\n' for k, v in knobs.items()
    ) + "}\n"
    out = KNOB_RE.sub(block, src, count=1)
    assert out != src or block in src, "KNOBS block not replaced"
    path = os.path.join(HERE, f"_logical_reduce_kernel_round1_{suffix}.py")
    open(path, "w").write(out)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    print(f"{os.path.basename(path)}  sha256={digest}  knobs={knobs}")
    return path


if __name__ == "__main__":
    for suffix, knobs in ROUND1.items():
        materialise(suffix, knobs)
