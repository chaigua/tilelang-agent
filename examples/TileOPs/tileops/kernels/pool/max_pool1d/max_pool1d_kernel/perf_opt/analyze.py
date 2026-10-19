"""analyze.py -- summarise msprof op metrics for a profile run root.

Usage:
  python analyze.py <run_root> [<run_root> ...]

`run_root` = the directory that contains one `OPPROF_*/main/<i>/` launch dir per
launch (e.g. `profiles/baseline/round0/<candidate>/<workload>_<hash>`).

For each run root it prints:
  * median Task Duration(us) over the launches (the only latency metric used),
  * PipeUtilization decomposition (aiv_time / vec / mte2 / mte3 / scalar),
  * Memory traffic (GM<->UB KB), plus the ideal manifest traffic,
  * the **official-report roofline `Ratio(%)` and `Bandwidth(TB/s)`** read from
    `visualize_data.bin` (`parse_bin_file`, i.e. msopprof's own precomputed
    GM/L2 "GM Read + Write" roofline entry -- exactly what
    `tileops.reporting.analyzer` consumes for the report's Ratio column).
"""

import csv
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")))

from tileops.benchmark.msprof_roofline import (  # noqa: E402
    DEFAULT_CONFIG,
    compute_from_csvs,
    parse_bin_file,
)


def launch_dirs(root):
    return sorted(
        os.path.dirname(p)
        for p in glob.glob(os.path.join(root, "**", "OpBasicInfo_*.csv"),
                           recursive=True)
    )


def median_col(dirs_, pattern, columns):
    out = {c: [] for c in columns}
    for d in dirs_:
        for path in glob.glob(os.path.join(d, pattern)):
            with open(path) as f:
                for row in csv.DictReader(f):
                    for c in columns:
                        v = row.get(c)
                        if v not in (None, "", "NA"):
                            try:
                                out[c].append(float(v))
                            except ValueError:
                                pass
    return {c: (round(statistics.median(v), 3) if v else None)
            for c, v in out.items()}


def roofline(bin_path):
    data = parse_bin_file(bin_path)
    gm = [e for e in data["roofline_entries"]
          if e.get("title") == "GM/L2" and "GM Read + Write" in e.get("bw_name", "")]
    if not gm:
        gm = [e for e in data["roofline_entries"] if "GM" in e.get("bw_name", "")]
    if not gm:
        return None
    e = gm[0]
    return {"ratio_pct": round(e["ratio"] * 100, 3), "bw_tb_s": e["bw"],
            "AI": round(e["AI"], 6), "perf_tops": round(e["performance"], 6),
            "computility_tops": round(e["computility"], 6),
            "duration_us": round(data["base_info"]["duration"], 3)}


def official_ratio(launch_dir):
    """Recompute the report's Ratio(%) from the CSVs (Method 2).

    `compute_from_csvs` expects timestamp-free filenames; the msprof dumps carry
    a timestamp suffix, so build a canonical copy farm in a temp dir.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ratio_")
    try:
        for path in glob.glob(os.path.join(launch_dir, "*.csv")):
            canonical = os.path.basename(path).split("_")[0] + ".csv"
            shutil.copyfile(path, os.path.join(tmp, canonical))
        return compute_from_csvs(tmp, DEFAULT_CONFIG)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_warned = [False]


def main():
    for root in sys.argv[1:]:
        print(f"\n### {root}")
        dirs_ = launch_dirs(root)
        if not dirs_:
            print("  no OpBasicInfo.csv found")
            continue
        basic = median_col(dirs_, "OpBasicInfo_*.csv",
                           ["Task Duration(us)", "Block Dim", "Current Freq"])
        pipe = median_col(dirs_, "PipeUtilization_*.csv",
                          ["aiv_time(us)", "aiv_vec_time(us)", "aiv_mte2_time(us)",
                           "aiv_mte3_time(us)", "aiv_scalar_time(us)"])
        mem = median_col(dirs_, "Memory_*.csv",
                         ["GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)"])
        print(f"  Task Duration(us) = {basic['Task Duration(us)']}"
              f"   BlockDim={basic['Block Dim']}   Freq={basic['Current Freq']}"
              f"   launches={len(dirs_)}")
        print(f"  aiv_time={pipe['aiv_time(us)']}  vec={pipe['aiv_vec_time(us)']}"
              f"  mte2={pipe['aiv_mte2_time(us)']}  mte3={pipe['aiv_mte3_time(us)']}"
              f"  scalar={pipe['aiv_scalar_time(us)']}")
        print(f"  GM_to_UB={mem['GM_to_UB_datas(KB)']}KB  UB_to_GM={mem['UB_to_GM_datas(KB)']}KB"
              f"  (per launch; x BlockDim for the kernel total)")
        rl = []
        for d in dirs_:
            try:
                m = official_ratio(d)
            except Exception as exc:  # noqa: BLE001
                if not _warned[0]:
                    print(f"  [csv-ratio] failed on {d}: {exc!r}")
                    _warned[0] = True
                continue
            rl.append(m)
        if rl:
            print(f"  [official report Ratio(%)] median = "
                  f"{round(statistics.median([r['ratio'] * 100 for r in rl]), 3)}"
                  f"   Bandwidth(TB/s) = {rl[0]['bw']}"
                  f"   total_bytes(med) = {round(statistics.median([r['total_bytes'] for r in rl]))}"
                  f"   total_fops(med) = {round(statistics.median([r['total_fops'] for r in rl]))}"
                  f"   AI(med) = {round(statistics.median([r['AI'] for r in rl]), 5)}"
                  f"   computility={rl[0]['computility']}TOps/s")


if __name__ == "__main__":
    main()
