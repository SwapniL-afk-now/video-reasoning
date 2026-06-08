#!/bin/bash
set -e
OUT="/workspace/output_4b"

echo "=== Deleting existing output for both videos ==="
rm -rf "$OUT/JTa_Ue2MSwc" "$OUT/KktLi3UifPY" \
        "$OUT/build_JTa_Ue2MSwc.log" "$OUT/build_KktLi3UifPY.log" \
        "$OUT/eval_JTa_Ue2MSwc.log"  "$OUT/eval_KktLi3UifPY.log" \
        "$OUT/results_JTa_Ue2MSwc.jsonl" "$OUT/results_KktLi3UifPY.jsonl" \
        "$OUT/debug_JTa_Ue2MSwc" "$OUT/debug_KktLi3UifPY"
echo "=== Output cleared. Starting timed pipeline ==="

OVERALL_START=$(date +%s)

MODEL="Qwen/Qwen3.5-4B"
GPU_UTIL=0.70

run_video () {
  local VID="$1" VIDEO="$2" CSV="$3"
  local T0=$(date +%s)

  echo ""
  echo "########## BUILD $VID  [$(date '+%H:%M:%S')] ##########"
  python3 /workspace/qvkg/scripts/build_vkg.py \
    --video "$VIDEO" \
    --out "$OUT" \
    --questions-csv "$CSV" \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_UTIL" \
    2>&1 | tee "$OUT/build_${VID}.log"
  local BUILD_END=$(date +%s)
  echo "--- BUILD $VID done in $((BUILD_END - T0))s ---"

  echo ""
  echo "########## EVAL $VID  [$(date '+%H:%M:%S')] ##########"
  python3 /workspace/qvkg/scripts/eval_lvbench.py \
    --csv "$CSV" \
    --vkg-dir "$OUT" \
    --video-dir /workspace \
    --out "$OUT/results_${VID}.jsonl" \
    --two-stage \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --debug-dir "$OUT/debug_${VID}" \
    2>&1 | tee "$OUT/eval_${VID}.log"
  local EVAL_END=$(date +%s)
  echo "--- EVAL $VID done in $((EVAL_END - BUILD_END))s  (video total: $((EVAL_END - T0))s) ---"
}

run_video KktLi3UifPY /workspace/KktLi3UifPY.mp4 /workspace/KktLi3UifPY_dataset.csv
run_video JTa_Ue2MSwc /workspace/JTa_Ue2MSwc.mp4 /workspace/LVBench_full.csv

OVERALL_END=$(date +%s)
echo ""
echo "=== TOTAL WALL TIME: $((OVERALL_END - OVERALL_START))s ==="
echo ""
echo "=== RESULTS ==="
for f in "$OUT"/results_*.jsonl; do
  python3 -c "
import json, sys
rows = [json.loads(l) for l in open('$f') if l.strip()]
c = sum(1 for r in rows if r.get('correct'))
print('$f', c, '/', len(rows), '=', round(100*c/len(rows),1) if rows else 0, '%')
"
done
