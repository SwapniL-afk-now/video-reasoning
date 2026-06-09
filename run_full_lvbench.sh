#!/usr/bin/env bash
# =============================================================================
# run_full_lvbench.sh
#
# Iterates over all 103 LVBench videos one at a time:
#   1. Download video (yt-dlp, 720p)
#   2. Build VKG (build_vkg.py)
#   3. Evaluate questions (eval_lvbench.py)
#   4. Log accuracy to W&B (log_to_wandb.py)
#   5. Delete video + VKG checkpoint to free disk space
#
# Usage:
#   export WANDB_API_KEY=...
#   bash run_full_lvbench.sh 2>&1 | tee run_lvbench.log
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------ Config
VENV_PY="/workspace/video-reasoning/.venv/bin/python"
CSV="/workspace/LVBench_full.csv"
RESULTS="/workspace/results_LVBench.jsonl"
VIDEO_DIR="/workspace/videos"
VKG_DIR="/workspace/output_lvbench"
SCRIPT_DIR="/workspace/video-reasoning/qvkg/scripts"
MODEL="Qwen/Qwen3.5-4B"
GPU_MEM_UTIL=0.65
LOG_FILE="/workspace/run_lvbench.log"

# Source credentials (WANDB_API_KEY, HF_TOKEN)
set -a
source /workspace/.env
set +a

mkdir -p "$VIDEO_DIR" "$VKG_DIR"
touch "$RESULTS"

echo "============================================================"
echo " LVBench Full Evaluation ($(date))"
echo "============================================================"
echo "Model:         $MODEL"
echo "GPU mem util:  $GPU_MEM_UTIL"
echo "Results:       $RESULTS"
echo "Log:           $LOG_FILE"
echo "============================================================"

# --------------------------------------------------- Determine undone videos
echo ""
echo "[INFO] Determining completed/remaining videos..."

UNDONE_VIDEOS=$("$VENV_PY" -c "
import csv, json, os

CSV = '$CSV'
RESULTS = '$RESULTS'

# Load all videos and their question UIDs
with open(CSV) as f:
    reader = csv.DictReader(f)
    vid_uids = {}
    for row in reader:
        v = row['video_path'].replace('.mp4', '')
        vid_uids.setdefault(v, set()).add(row['uid'])

# Load answered UIDs
answered = set()
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    answered.add(json.loads(line)['uid'])
                except Exception:
                    pass

# Find undone videos
undone = []
done = []
for v, uids in sorted(vid_uids.items()):
    if uids.issubset(answered):
        done.append(v)
    else:
        undone.append(v)

print(json.dumps({'done': done, 'undone': undone, 'total': len(vid_uids)}))
")

DONE_COUNT=$(echo "$UNDONE_VIDEOS" | "$VENV_PY" -c "import sys,json; print(len(json.load(sys.stdin)['done']))")
UNDONE_COUNT=$(echo "$UNDONE_VIDEOS" | "$VENV_PY" -c "import sys,json; print(len(json.load(sys.stdin)['undone']))")
TOTAL_COUNT=$(echo "$UNDONE_VIDEOS" | "$VENV_PY" -c "import sys,json; print(json.load(sys.stdin)['total'])")

echo "  Completed: $DONE_COUNT / $TOTAL_COUNT"
echo "  Remaining:  $UNDONE_COUNT / $TOTAL_COUNT"

if [ "$UNDONE_COUNT" -eq 0 ]; then
    echo "[INFO] All videos completed. Nothing to do."
    exit 0
fi

# Build list of undone video IDs as array
mapfile -t VIDEO_QUEUE < <(echo "$UNDONE_VIDEOS" | "$VENV_PY" -c "
import sys, json
data = json.load(sys.stdin)
for v in data['undone']:
    print(v)
")

echo ""
echo "Queue: ${VIDEO_QUEUE[*]}"
echo ""

# --------------------------------------------------- Trap for clean exit
WANDB_INIT_DONE=0
cleanup() {
    echo ""
    echo "[CLEANUP] Finishing wandb run..."
    "$VENV_PY" -c "import wandb; wandb.finish()" 2>/dev/null || true
    echo "[CLEANUP] Done."
}
trap cleanup EXIT SIGINT SIGTERM

# --------------------------------------------------- Main loop
COMPLETED=0
TOTAL=${#VIDEO_QUEUE[@]}

for VID in "${VIDEO_QUEUE[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    VIDEO_FILE="$VIDEO_DIR/$VID.mp4"
    VKG_VID_DIR="$VKG_DIR/$VID"

    echo ""
    echo "============================================================"
    echo "  [$COMPLETED/$TOTAL] Processing: $VID ($(date))"
    echo "============================================================"

    STEP_START=$(date +%s)

    # ------------------------------------------------- Step 1: Download
    echo ""
    echo "  [1/4] Downloading video..."
    if [ -f "$VIDEO_FILE" ]; then
        echo "    Already downloaded, skipping."
    else
        if ! yt-dlp -f "bestvideo[height<=720]+bestaudio" \
            --merge-output-format mp4 \
            -o "$VIDEO_FILE" \
            "https://www.youtube.com/watch?v=$VID" 2>&1; then
            echo "    [SKIP] Download failed for $VID"
            continue
        fi
    fi

    # ------------------------------------------------- Step 2: Build VKG
    echo ""
    echo "  [2/4] Building VKG..."
    if [ -f "$VKG_VID_DIR/vkg.json" ]; then
        echo "    VKG already built, skipping."
    else
        if ! "$VENV_PY" "$SCRIPT_DIR/build_vkg.py" \
            --video "$VIDEO_FILE" \
            --questions-csv "$CSV" \
            --out "$VKG_DIR" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" 2>&1; then
            echo "    [SKIP] VKG build failed for $VID"
            # Cleanup partial VKG
            rm -rf "$VKG_VID_DIR" 2>/dev/null || true
            rm -f "$VIDEO_FILE" 2>/dev/null || true
            continue
        fi
    fi

    # ------------------------------------------------- Step 3: Evaluate
    echo ""
    echo "  [3/4] Evaluating questions..."
    if ! "$VENV_PY" "$SCRIPT_DIR/eval_lvbench.py" \
        --csv "$CSV" \
        --vkg-dir "$VKG_DIR" \
        --video-dir "$VIDEO_DIR" \
        --out "$RESULTS" \
        --model "$MODEL" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" 2>&1; then
        echo "    [WARN] Evaluation encountered errors for $VID"
    fi

    STEP_END=$(date +%s)
    DURATION=$((STEP_END - STEP_START))

    # ------------------------------------------------- Step 4: Log to wandb
    echo ""
    echo "  [4/4] Logging to W&B..."
    "$VENV_PY" "$SCRIPT_DIR/log_to_wandb.py" \
        --results "$RESULTS" \
        --video "$VID" \
        --duration "$DURATION" 2>&1 || echo "    [WARN] wandb logging failed"

    # ------------------------------------------------- Step 5: Cleanup
    echo ""
    echo "  [CLEANUP] Removing video + VKG checkpoints..."
    rm -f "$VIDEO_FILE" 2>/dev/null || true
    rm -rf "$VKG_VID_DIR" 2>/dev/null || true
    echo "    Done."

    # Print progress summary
    echo ""
    "$VENV_PY" -c "
import json, os
results_file = '$RESULTS'
correct = 0
total = 0
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                total += 1
                if r.get('correct'):
                    correct += 1
acc = (correct / total * 100) if total > 0 else 0
print(f'  >>> Cumulative accuracy: {correct}/{total} = {acc:.1f}% <<<')
"

    echo ""
    echo "  Duration: ${DURATION}s ($(printf '%02d:%02d' $((DURATION/60)) $((DURATION%60))))"
    echo "============================================================"
done

# --------------------------------------------------- Final summary
echo ""
echo "============================================================"
echo " ALL DONE ($(date))"
echo "============================================================"
"$VENV_PY" -c "
import json, os
results_file = '$RESULTS'
correct = 0
total = 0
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                total += 1
                if r.get('correct'):
                    correct += 1
acc = (correct / total * 100) if total > 0 else 0
print(f'Final accuracy: {correct}/{total} = {acc:.1f}%')
"
echo ""
echo "Results saved to: $RESULTS"
echo "============================================================"
