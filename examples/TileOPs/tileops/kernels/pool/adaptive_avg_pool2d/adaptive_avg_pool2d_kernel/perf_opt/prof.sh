#!/usr/bin/env bash
# prof.sh -- Stage 4 msprof op helper (kernel-only Task Duration + pipe breakdown).
#   ./prof.sh <out_dir> <python-script> [args...]
set -u
OUT="$1"; shift
mkdir -p "$OUT"; rm -rf "$OUT"/OPPROF_*; chmod 700 "$OUT" 2>/dev/null
mkdir -p logs/round3
LOG="logs/round3/$(echo "$OUT" | tr '/' '_').log"
MS="BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio"
{
  echo "# msprof op --kernel-name=main --output=$OUT --launch-count=20 --warm-up=5 --dump=off --aic-metrics=$MS $*"
  msprof op --kernel-name=main --output="$OUT" --launch-count=20 --warm-up=5 \
      --dump=off --aic-metrics="$MS" "$@"
} >"$LOG" 2>&1
python3 - "$OUT" <<'EOF'
import csv, glob, os, statistics, sys
out = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(out, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))):
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("Task Duration(us)"):
                rows.append(float(r["Task Duration(us)"]))
if not rows:
    print("RESULT: NO ROWS (see log)")
    sys.exit(0)
print("RESULT: rows=%d median=%.3f mean=%.3f min=%.3f max=%.3f" % (
    len(rows), statistics.median(rows), statistics.mean(rows), min(rows), max(rows)))

COLS = ["aiv_time(us)", "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)",
        "aiv_scalar_time(us)", "aiv_scalar_wait_time(us)", "aiv_scalar_stall_by_ub_time(us)",
        "aiv_mte2_active_bw(GB/s)", "aiv_mte2_ratio", "aiv_vec_ratio"]
def dump(pattern, cols):
    for p in sorted(glob.glob(os.path.join(out, "OPPROF_*", "*", "*", pattern))):
        with open(p) as f:
            for r in csv.DictReader(f):
                vals = [f"{c.split('(')[0]}={r.get(c,'-')}" for c in cols if c in r]
                if any(v.split('=')[1] not in ('-', 'NA', '0', '0.0', '0.00') for v in vals):
                    print("  " + " ".join(vals))
dump("PipeUtilization_*.csv", COLS)
dump("MemoryUB_*.csv", ["GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)", "GM_to_UB_bw(GB/s)",
                       "UB_to_GM_bw(GB/s)", "GM_to_UB_time(us)", "UB_to_GM_time(us)"])
dump("Memory_*.csv", ["GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)", "GM_to_UB_bw(GB/s)",
                      "UB_to_GM_bw(GB/s)", "GM_to_UB_time(us)", "UB_to_GM_time(us)"])
EOF
