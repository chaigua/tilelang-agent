#!/usr/bin/env bash
# Stage 4 measurement driver (kernel-only, msprof op Task Duration).
#
# Every measurement is `msprof op --kernel-name=main --launch-count=20
# --warm-up=5` around `bench.py` / `probe_floor.py`; the median of the 20
# recorded Task Durations is the single latency metric of this stage.
#
# Groups
#   baseline : Stage 3 file, deployed block_m = 1 (the read-only wrapper's
#              structural default)                          -> perf_records
#   final    : perf_opt/GroupNormFwdOp.py, block_m = 0 (tuned default)
#                                                          -> perf_records
#   floor    : empty / dma_rw / chain probes on the deployed grid geometry
#                                                          -> copy_floor_calibration.jsonl only
set -u
cd "$(dirname "$0")"
source /workspace/env.sh >/dev/null 2>&1

PY=/workspace/venv/bin/python3
BASE_IMPL="$PWD/../_group_norm_kernel.py"
OPT_IMPL="$PWD/GroupNormFwdOp.py"

run_prof() {  # run_prof <outdir> <logfile> <python args...>
  local out="$1"; shift
  local log="$1"; shift
  mkdir -p "$out"
  chmod 700 "$out"
  msprof op --kernel-name=main --output="$out" --launch-count=20 --warm-up=5 \
      --dump=off "$PY" "$@" >"$log" 2>&1
  local rc=$?
  echo "[measure] rc=$rc out=$out log=$log"
  return $rc
}

group="${1:-all}"

if [ "$group" = "all" ] || [ "$group" = "baseline" ]; then
  for c in image-g32/fp16 image-g32/bf16 wider-channel-g32/fp16 tail-spatial-g16/fp16; do
    tag="${c/\//_}"
    run_prof "profiles/baseline_v2/$tag" "logs/baseline_v2/$tag.log" \
      bench.py --impl "$BASE_IMPL" --case "$c" --bm 1 --print-bm
  done
fi

if [ "$group" = "all" ] || [ "$group" = "final" ]; then
  for c in image-g32/fp16 image-g32/bf16 wider-channel-g32/fp16 tail-spatial-g16/fp16; do
    tag="${c/\//_}"
    run_prof "profiles/final_v2/$tag" "logs/final_v2/$tag.log" \
      bench.py --impl "$OPT_IMPL" --case "$c" --bm 0 --print-bm
  done
fi

if [ "$group" = "all" ] || [ "$group" = "floor" ]; then
  # empty-kernel (launch/fixed-cost) floor at the grid sizes that occur
  for b in 2 29 32 40; do
    run_prof "profiles/floor_v2/empty_b$b" "logs/floor_v2/empty_b$b.log" \
      probe_floor.py --mode empty --blocks "$b"
  done
  # pure-DMA floor (read+write) and copy-floor (load + 2 v-ops + store) at the
  # deployed grid geometry of each tune workload
  for spec in "image-g32/fp16 29 3" "wider-channel-g32/fp16 32 2" "tail-spatial-g16/fp16 32 2"; do
    set -- $spec
    c="$1"; nb="$2"; bm="$3"
    tag="${c/\//_}"
    run_prof "profiles/floor_v2/dma_rw_$tag" "logs/floor_v2/dma_rw_$tag.log" \
      probe_floor.py --mode dma_rw --case "$c" --bm "$bm" --blocks "$nb"
    run_prof "profiles/floor_v2/chain_$tag" "logs/floor_v2/chain_$tag.log" \
      probe_floor.py --mode chain --case "$c" --bm "$bm" --blocks "$nb"
  done
  # image-g32 was already probed at 40 blocks in round0; redo at the deployed
  # 29-block grid for an apples-to-apples floor
  run_prof "profiles/floor_v2/dma_rw_image-g32_fp16" "logs/floor_v2/dma_rw_image-g32_fp16.log" \
    probe_floor.py --mode dma_rw --case image-g32/fp16 --bm 3 --blocks 29
  run_prof "profiles/floor_v2/chain_image-g32_fp16" "logs/floor_v2/chain_image-g32_fp16.log" \
    probe_floor.py --mode chain --case image-g32/fp16 --bm 3 --blocks 29
fi
echo "[measure] done group=$group"
