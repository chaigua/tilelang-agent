#!/bin/bash
# probe_run.sh -- msprof op a single probe.py variant, print Task Duration + vec/MTE.
# usage: bash probe_run.sh <case> <variant> [profile_tag]
set -u
source /workspace/env.sh >/dev/null 2>&1
CASE=${1:?case}
VARIANT=${2:?variant}
TAG=${3:-$VARIANT}
OUT="profiles/probe/${CASE}_${TAG}"
chmod 700 . 2>/dev/null
rm -rf "$OUT"; mkdir -p "$OUT"; chmod 700 "$OUT"
msprof op --kernel-name=main --output="$OUT" --launch-count=20 --warm-up=5 \
  --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  /workspace/venv/bin/python3 probe.py --case "$CASE" --variant "$VARIANT" --iters 25 \
  > "logs/probe_${CASE}_${TAG}.log" 2>&1
echo "--- rc=$? ${CASE}/${VARIANT} ---"
/workspace/venv/bin/python3 analyze.py "$OUT" 2>&1 | grep -v '^$'
