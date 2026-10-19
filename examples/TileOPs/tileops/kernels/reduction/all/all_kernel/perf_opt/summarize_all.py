"""Summarise Task Duration + pipe/memory metrics for every measured profile dir."""
import csv, glob, json, os, statistics, sys

COLS = ["aiv_time(us)", "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
        "aiv_scalar_ratio", "aiv_mte2_active_bw(GB/s)", "GM_to_UB_datas(KB)",
        "UB_to_GM_datas(KB)"]


def agg(d):
    out = {}
    for pat in ("PipeUtilization", "Memory_", "MemoryUB"):
        for f in sorted(glob.glob(os.path.join(d, "**", f"{pat}*.csv"), recursive=True)):
            rows = list(csv.DictReader(open(f)))
            if not rows:
                continue
            for k in rows[0]:
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r[k]))
                    except (TypeError, ValueError):
                        pass
                if vals and len(vals) == len(rows):
                    out[k] = round(statistics.mean(vals), 4)
    info = sorted(glob.glob(os.path.join(d, "**", "OpBasicInfo*.csv"), recursive=True))
    if info:
        rows = list(csv.DictReader(open(info[0])))
        out["task_duration_us"] = round(statistics.mean(float(r["Task Duration(us)"]) for r in rows), 4)
        out["block_dim"] = rows[0]["Block Dim"]
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        a = agg(d)
        keep = {k: a.get(k) for k in ["task_duration_us", "block_dim"] + COLS}
        print(json.dumps({"dir": d, **keep}))
