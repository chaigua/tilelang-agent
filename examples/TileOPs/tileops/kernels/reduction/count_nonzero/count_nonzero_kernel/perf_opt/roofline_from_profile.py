"""Extract the GM Read+Write roofline metrics (report calibre) from an msprof dir.

The official TileOPs report's ``Ratio (%)`` and ``Bandwidth (TB/s)`` columns come
from ``visualize_data.bin`` -> roofline entry ``GM/L2: GM Read + Write``
(``benchmark_base._parse_msprof_roofline``).  This helper reproduces that exact
calibre for the Stage 4 profiles so baseline / final / report numbers are
comparable.

CLI: python roofline_from_profile.py <msprof_output_dir> [more dirs...]
"""

import glob
import json
import os
import sys

from tileops.benchmark.msprof_roofline import parse_bin_file


def roofline(output_dir):
    bins = sorted(glob.glob(os.path.join(output_dir, "**", "visualize_data.bin"), recursive=True))
    if not bins:
        return {"error": "no visualize_data.bin"}
    # first captured launch is representative; take the median ratio across bins
    ratios, bws, perfs, comps = [], [], [], []
    for path in bins[:10]:
        data = parse_bin_file(path)
        entries = data.get("roofline_entries", []) if isinstance(data, dict) else []
        gm = [e for e in entries if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", "")]
        if not gm:
            continue
        ratios.append(gm[0]["ratio"] * 100)
        bws.append(gm[0]["bw"])
        perfs.append(gm[0]["performance"])
        comps.append(gm[0]["computility"])
    if not ratios:
        return {"error": "no GM Read + Write roofline entry"}
    ratios.sort()
    return {
        "n_bins": len(ratios),
        "ratio_pct_median": round(ratios[len(ratios) // 2], 3),
        "bw_tb_s": round(bws[0], 4),
        "perf_tops_s": round(perfs[0], 5),
        "computility_tops_s": round(comps[0], 5),
    }


if __name__ == "__main__":
    out = {}
    for d in sys.argv[1:]:
        out[d] = roofline(d)
    print(json.dumps(out, indent=2))
