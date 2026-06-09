#!/bin/bash
set -e

MODEL="${MODEL:-/workspace/models/Qwen3.5-4B}"
OUT="${OUT:-/workspace/vkgs}"
RESULTS_DIR="${RESULTS_DIR:-/workspace}"
GPU_UTIL="${GPU_UTIL:-0.75}"
N_WORKERS="${N_WORKERS:-8}"       # parallel frame-extraction chunks
VIDEO_DIR="${VIDEO_DIR:-/workspace/videos}"

mkdir -p "$OUT"

echo "=== Q-VKG pipeline ==="
echo "Model:      $MODEL"
echo "Output:     $OUT"
echo "Workers:    $N_WORKERS  |  CoarseFPS: 1.0 (fixed)  |  GPU util: $GPU_UTIL"
echo "Whisper:    ON (large-v3)"

run_video () {
  local VID_NAME="$1"   # e.g. KktLi3UifPY
  local CSV="$2"        # path to questions CSV

  echo ""
  echo "########## $VID_NAME ##########"
  python3 /workspace/video-reasoning/qvkg/scripts/run_pipeline.py \
    --csv           "$CSV" \
    --video-dir     "$VIDEO_DIR" \
    --out           "$OUT" \
    --results       "$RESULTS_DIR/results_${VID_NAME}.jsonl" \
    --model         "$MODEL" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --n-chunk-workers "$N_WORKERS" \
    --whisper-model large-v3 \
    --no-cleanup \
    2>&1 | tee "$RESULTS_DIR/run_log_${VID_NAME}.txt"
}

# Add / remove videos here — Whisper and parallel SigLIP apply to all
run_video KktLi3UifPY /workspace/KktLi3UifPY_dataset.csv
run_video JTa_Ue2MSwc /workspace/JTa_Ue2MSwc_dataset.csv

echo ""
echo "=== Pipeline complete ==="
for f in "$RESULTS_DIR"/results_*.jsonl; do
  python3 -c "
import json, os
rows = [json.loads(l) for l in open('$f') if l.strip()]
c = sum(1 for r in rows if r.get('correct'))
print(os.path.basename('$f'), f'{c}/{len(rows)}', f'({100*c/len(rows):.1f}%)' if rows else '')
"
done
