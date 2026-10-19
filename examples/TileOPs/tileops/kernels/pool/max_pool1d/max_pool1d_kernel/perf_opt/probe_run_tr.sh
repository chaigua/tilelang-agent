#!/bin/bash
set -u
source /workspace/env.sh >/dev/null 2>&1
V=${1:?variant}
OUT="profiles/probe/tr_${V}"
rm -rf "$OUT"; mkdir -p "$OUT"; chmod 700 "$OUT"
msprof op --kernel-name=main --output="$OUT" --launch-count=20 --warm-up=5 \
  --dump=off --aic-metrics=BasicInfo,PipeUtilization,Memory \
  /workspace/venv/bin/python3 probe_tr.py --variant "$V" --iters 25 \
  > "logs/probe_tr_${V}.log" 2>&1
echo "--- rc=$? tr/$V ---"
/workspace/venv/bin/python3 analyze.py "$OUT" 2>&1 | grep -E "Task Duration|aiv_time|GM_to_UB"
