#!/bin/bash
set -e
MODEL="Qwen/Qwen3.5-9B"
LOGDIR="/workspace/output"
mkdir -p "$LOGDIR"

echo "=== Starting 9B evaluation pipeline ==="
echo "Model: $MODEL"
echo ""

# --- KktLi3UifPY ---
echo "[1/4] Building VKG for KktLi3UifPY..."
python3 /workspace/qvkg/scripts/build_vkg.py \
  --video /workspace/KktLi3UifPY.mp4 \
  --out /workspace/output \
  --questions-csv /workspace/KktLi3UifPY_dataset.csv \
  --model "$MODEL" \
  --gpu-memory-utilization 0.60 \
  2>&1 | tee "$LOGDIR/build_KktLi3UifPY_9b.log"
echo "[1/4] KktLi3UifPY VKG done."

echo "[2/4] Evaluating KktLi3UifPY..."
python3 /workspace/qvkg/scripts/eval_lvbench.py \
  --csv /workspace/KktLi3UifPY_dataset.csv \
  --vkg-dir /workspace/output \
  --video-dir /workspace \
  --out "$LOGDIR/results_KktLi3UifPY_9b.jsonl" \
  --two-stage \
  --model "$MODEL" \
  --gpu-memory-utilization 0.60 \
  --debug-dir "$LOGDIR/debug_KktLi3UifPY_9b" \
  2>&1 | tee "$LOGDIR/eval_KktLi3UifPY_9b.log"
echo "[2/4] KktLi3UifPY eval done."

# --- JTa_Ue2MSwc ---
echo "[3/4] Building VKG for JTa_Ue2MSwc..."
python3 /workspace/qvkg/scripts/build_vkg.py \
  --video /workspace/JTa_Ue2MSwc.mp4 \
  --out /workspace/output \
  --questions-csv /workspace/LVBench_full.csv \
  --model "$MODEL" \
  --gpu-memory-utilization 0.60 \
  2>&1 | tee "$LOGDIR/build_JTa_Ue2MSwc_9b.log"
echo "[3/4] JTa_Ue2MSwc VKG done."

echo "[4/4] Evaluating JTa_Ue2MSwc..."
python3 /workspace/qvkg/scripts/eval_lvbench.py \
  --csv /workspace/LVBench_full.csv \
  --vkg-dir /workspace/output \
  --video-dir /workspace \
  --out "$LOGDIR/results_JTa_Ue2MSwc_9b.jsonl" \
  --two-stage \
  --model "$MODEL" \
  --gpu-memory-utilization 0.60 \
  --debug-dir "$LOGDIR/debug_JTa_Ue2MSwc_9b" \
  2>&1 | tee "$LOGDIR/eval_JTa_Ue2MSwc_9b.log"
echo "[4/4] JTa_Ue2MSwc eval done."

echo ""
echo "=== All 9B evaluations complete ==="
