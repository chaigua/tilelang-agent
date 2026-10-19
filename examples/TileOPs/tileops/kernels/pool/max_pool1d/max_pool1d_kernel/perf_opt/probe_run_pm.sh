#!/bin/bash
set -u
source /workspace/env.sh >/dev/null 2>&1
CASE=${1:?case}; shift
ARGS="$@"
TAG=$(echo "${CASE}_${ARGS}" | tr -d ' -=_' | head -c 40)
OUT="profiles/probe/pm_${TAG}"
rm -rf "$OUT"; mkdir -p "$OUT"; chmod 700 "$OUT"
msprof op --kernel-name=main --output="$OUT" --launch-count=20 --warm-up=5 \
  --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  /workspace/venv/bin/python3 probe_pm.py --case "$CASE" $ARGS --iters 25 \
  > "logs/probe_pm_${TAG}.log" 2>&1
echo "--- rc=$? $CASE $ARGS ---"
/workspace/venv/bin/python3 analyze.py "$OUT" 2>&1 | grep -v '^$'
