#!/bin/bash
set -e

# Wait for KktLi3UifPY VKG build to complete
echo "[pipeline] Waiting for KktLi3UifPY build..."
until [ -f /workspace/output/KktLi3UifPY/vkg.json ]; do sleep 15; done
echo "[pipeline] KktLi3UifPY build done. Starting eval..."

# Eval KktLi3UifPY
cd /workspace
python3 qvkg/scripts/eval_lvbench.py \
  --csv /workspace/KktLi3UifPY_dataset.csv \
  --vkg-dir /workspace/output \
  --video-dir /workspace \
  --out /workspace/output/results_KktLi3UifPY.jsonl \
  --two-stage \
  --debug-dir /workspace/output/debug_KktLi3UifPY
echo "[pipeline] KktLi3UifPY eval done. Starting JTa_Ue2MSwc build..."

# Build JTa_Ue2MSwc VKG
python3 qvkg/scripts/build_vkg.py \
  --video /workspace/JTa_Ue2MSwc.mp4 \
  --out /workspace/output \
  --questions-csv /workspace/LVBench_full.csv \
  --gpu-memory-utilization 0.65 \
  2>&1 | tee /workspace/output/build_JTa_Ue2MSwc.log
echo "[pipeline] JTa_Ue2MSwc build done. Starting eval..."

# Eval JTa_Ue2MSwc
python3 qvkg/scripts/eval_lvbench.py \
  --csv /workspace/LVBench_full.csv \
  --vkg-dir /workspace/output \
  --video-dir /workspace \
  --out /workspace/output/results_JTa_Ue2MSwc.jsonl \
  --two-stage \
  --debug-dir /workspace/output/debug_JTa_Ue2MSwc
echo "[pipeline] All done."
