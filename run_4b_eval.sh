#!/bin/bash
set -e
MODEL="Qwen/Qwen3.5-4B"
OUT="/workspace/output_4b"        # fresh build root — existing output/ untouched
GPU_UTIL=0.70                     # max KV cache → max concurrent batched seqs
mkdir -p "$OUT"

echo "=== 4B FULL pipeline: build-from-scratch + two-stage eval ==="
echo "Model: $MODEL  out=$OUT  gpu_util=$GPU_UTIL"
echo "vLLM loads lazily (only when scene-extraction / planner first needs it)."

run_video () {
  local VID="$1" VIDEO="$2" CSV="$3"
  echo ""
  echo "########## BUILD $VID ##########"
  python3 /workspace/qvkg/scripts/build_vkg.py \
    --video "$VIDEO" \
    --out "$OUT" \
    --questions-csv "$CSV" \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_UTIL" \
    2>&1 | tee "$OUT/build_${VID}.log"

  echo ""
  echo "########## EVAL $VID ##########"
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
}

run_video KktLi3UifPY /workspace/KktLi3UifPY.mp4 /workspace/KktLi3UifPY_dataset.csv
run_video JTa_Ue2MSwc /workspace/JTa_Ue2MSwc.mp4 /workspace/LVBench_full.csv

echo ""
echo "=== 4B pipeline complete ==="
for f in "$OUT"/results_*.jsonl; do
  python3 -c "import json,sys; rows=[json.loads(l) for l in open('$f') if l.strip()]; c=sum(1 for r in rows if r.get('correct')); print('$f', c, '/', len(rows))"
done
