"""Clean re-check of the device peak bandwidth (session-3, step A).

Runs a small, fixed matrix through `msprof op` serially and prints GB/s per run.
Rationale: the user's target `achieved_bw / 1.75 TB/s >= 70%` is only meaningful
if 1.75 TB/s is reachable on this part; this probe measures the reachable
read / mixed / write ceiling with fully contiguous DMA.

  python3 run_peak2.py            # all cases
  python3 run_peak2.py a1 a4      # subset
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prof  # noqa: E402

LAUNCH = 20
WARM = 2
METRICS = "PipeUtilization,Memory"

# id: (mode, mb, span_elems, stages)
CASES = {
    "a1": ("read1", 256, 61440, 3),   # pure read, 1-elem keep-alive, 120 KB chunks
    "a2": ("read1", 64, 61440, 3),
    "a3": ("read1", 256, 15360, 6),   # 30 KB chunks, deeper pipeline (more outstanding)
    "a4": ("rw", 256, 30720, 1),      # read+write mixed, 60 KB chunks
    "a5": ("rw", 64, 30720, 1),
    "a6": ("write", 256, 61440, 1),   # pure write (confirm the earlier 1082 GB/s)
}


def run(cid):
    mode, mb, span, stages = CASES[cid]
    od = os.path.join("profiles", "peak2", cid)
    os.makedirs(od, exist_ok=True)
    os.chmod(od, 0o700)
    cmd = ["msprof", "op", "--kernel-name=main", f"--output={od}",
           f"--launch-count={LAUNCH}", f"--warm-up={WARM}", "--dump=off",
           f"--aic-metrics={METRICS}", sys.executable, "peak.py",
           "--mode", mode, "--mb", str(mb), "--span", str(span),
           "--stages", str(stages), "--iters", str(WARM + LAUNCH)]
    lg = os.path.join("logs", "peak2")
    os.makedirs(lg, exist_ok=True)
    with open(os.path.join(lg, cid + ".log"), "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        pr = subprocess.run(cmd, capture_output=True, text=True)
        f.write(pr.stdout + pr.stderr)
    if pr.returncode != 0:
        return {"id": cid, "error": f"msprof rc={pr.returncode}"}
    r = prof.parse(od)
    if r["duration_us"] is None:
        return {"id": cid, "error": "no rows"}
    moved = mb * 1024 * 1024                      # one direction, bytes
    read_b = moved if mode in ("read", "read1", "rw") else 0
    write_b = moved if mode in ("write", "rw") else 0
    total = read_b + write_b
    dur = r["duration_us"]
    return {"id": cid, "mode": mode, "mb": mb, "span_bytes": span * 2,
            "stages": stages, "duration_us": dur, "dur_min": r["dur_min"],
            "dur_max": r["dur_max"], "n": r["n"],
            "read_gbs": round(read_b / dur / 1e3, 1) if read_b else None,
            "write_gbs": round(write_b / dur / 1e3, 1) if write_b else None,
            "total_gbs": round(total / dur / 1e3, 1),
            "aiv_time": r["aiv_time"], "mte2": r["mte2"], "mte3": r["mte3"],
            "vec": r["vec"], "scalar": r["scalar"],
            "gm_to_ub_kb": r["gm_to_ub_kb"], "ub_to_gm_kb": r["ub_to_gm_kb"],
            "msprof_raw_path": od,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ids = sys.argv[1:] or list(CASES)
    out = []
    for cid in ids:
        rec = run(cid)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open("peak_recheck.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    ok = [r for r in out if "error" not in r]
    if ok:
        print("\n== summary ==")
        for m in ("read1", "rw", "write"):
            v = [r["total_gbs"] for r in ok if r["mode"] == m]
            if v:
                print(f"{m:6s} total GB/s: median {statistics.median(v):.1f}  all {v}")


if __name__ == "__main__":
    main()
