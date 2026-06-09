#!/usr/bin/env bash
# run_rolling.sh — Download → build VKG → eval → delete, rolling over all videos in CSV.
# Idempotent: safe to kill and restart at any point.
# Each video is fully deleted after eval to reclaim disk space.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
CSV="${CSV:-/workspace/LVBench_full.csv}"
OUT="${OUT:-/workspace/vkgs}"
VIDEO_DIR="${VIDEO_DIR:-/workspace/videos}"
RESULTS="${RESULTS:-/workspace/results_all.jsonl}"
DONE_FILE="$OUT/done_videos.txt"
MODEL="${MODEL:-/workspace/models/Qwen3.5-4B}"
GPU_UTIL="0.75"
N_WORKERS="8"          # parallel frame-extraction chunks (fixed)
WHISPER_MODEL="large-v3"
LOG="$OUT/run_rolling.log"
BATCH_SIZE="${BATCH_SIZE:-4}"   # videos to download in parallel per batch

# Videos to skip (already being handled separately)
IN_PROGRESS=("KktLi3UifPY" "JTa_Ue2MSwc")

# ─── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

wait_gpu_free() {
    log "  [gpu] Waiting for GPU to free up (>80 GB)..."
    until python3 -c "
import subprocess, sys
r = subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], capture_output=True, text=True)
sys.exit(0 if int(r.stdout.strip()) > 81920 else 1)
" 2>/dev/null; do sleep 5; done
    log "  [gpu] GPU ready."
}

get_all_vids() {
    python3 -c "
import csv
seen, out = [], []
with open('$CSV') as f:
    for r in csv.DictReader(f):
        v = r['video_path'].replace('.mp4','')
        if v not in seen:
            seen.append(v); out.append(v)
for v in out: print(v)
"
}

is_fully_evaluated() {
    local vid="$1"
    python3 -c "
import csv, json, sys
uids = [r['uid'] for r in csv.DictReader(open('$CSV')) if r['video_path'].replace('.mp4','') == '$vid']
if not uids: sys.exit(0)
try:
    answered = {json.loads(l)['uid'] for l in open('$RESULTS') if l.strip()}
except FileNotFoundError:
    answered = set()
sys.exit(0 if all(u in answered for u in uids) else 1)
"
}

free_gb() { df -BG /workspace | awk 'NR==2{gsub("G",""); print $4}'; }
mark_done() { grep -qxF "$1" "$DONE_FILE" 2>/dev/null || echo "$1" >> "$DONE_FILE"; }

print_accuracy() {
    [[ -f "$RESULTS" ]] || return 0
    python3 -c "
import json
rows = [json.loads(l) for l in open('$RESULTS') if l.strip()]
if not rows: print('  No results yet.'); exit()
c = sum(1 for r in rows if r.get('correct'))
print(f'  Running accuracy: {c}/{len(rows)} = {100*c/len(rows):.1f}%')
"
}

download_video() {
    local vid="$1"
    local dest="$VIDEO_DIR/$vid.mp4"
    [[ -f "$dest" ]] && { log "  [download] $vid already present."; return 0; }
    log "  [download] $vid ..."
    yt-dlp \
        -f "bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/bestvideo[vcodec!=av01][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
        --merge-output-format mp4 \
        --no-part \
        --socket-timeout 60 \
        --retries 3 \
        -o "$dest" \
        "https://www.youtube.com/watch?v=$vid" \
        >> "$OUT/download_${vid}.log" 2>&1
}

process_video() {
    local vid="$1"
    local T0; T0=$(date +%s)
    log "━━━ $vid — build + eval  (disk: $(free_gb) GB) ━━━"

    python3 /workspace/video-reasoning/qvkg/scripts/run_pipeline.py \
        --csv            "$CSV" \
        --video-dir      "$VIDEO_DIR" \
        --out            "$OUT" \
        --results        "$RESULTS" \
        --model          "$MODEL" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --n-chunk-workers "$N_WORKERS" \
        --whisper-model  "$WHISPER_MODEL" \
        2>&1 | tee "$OUT/run_${vid}.log" \
    && mark_done "$vid"

    local T1; T1=$(date +%s)
    log "  Wall time: $((T1 - T0))s"
    print_accuracy
}

cleanup_video() {
    local vid="$1"
    log "  [cleanup] $vid — removing video + VKG..."
    rm -f  "$VIDEO_DIR/$vid.mp4"
    rm -rf "$OUT/$vid"
    rm -f  "$OUT/run_${vid}.log" "$OUT/download_${vid}.log"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

mkdir -p "$OUT" "$VIDEO_DIR"
touch "$DONE_FILE"
log "=== run_rolling.sh starting ==="
log "CSV: $CSV  |  workers=$N_WORKERS  |  whisper=$WHISPER_MODEL  |  gpu_util=$GPU_UTIL"
log "Free disk: $(free_gb) GB  |  batch_size=$BATCH_SIZE"

declare -A SKIP
for v in "${IN_PROGRESS[@]}"; do SKIP["$v"]=1; done

mapfile -t ALL_VIDS < <(get_all_vids)
log "Total videos in CSV: ${#ALL_VIDS[@]}"

ALREADY_HAVE=()
NEED_DOWNLOAD=()
for vid in "${ALL_VIDS[@]}"; do
    [[ -n "${SKIP[$vid]+x}" ]] && continue
    is_fully_evaluated "$vid" && { log "  [skip] $vid — already evaluated."; continue; }
    if [[ -f "$VIDEO_DIR/$vid.mp4" ]]; then
        ALREADY_HAVE+=("$vid")
    else
        NEED_DOWNLOAD+=("$vid")
    fi
done

log "Already on disk: ${#ALREADY_HAVE[@]}  |  Need download: ${#NEED_DOWNLOAD[@]}"

# ─── Phase 1: Videos already on disk ──────────────────────────────────────────
for vid in "${ALREADY_HAVE[@]}"; do
    wait_gpu_free
    process_video "$vid"
    cleanup_video "$vid"
done

# ─── Phase 2: Batch download → process → delete ───────────────────────────────
N="${#NEED_DOWNLOAD[@]}"
i=0

while (( i < N )); do
    # Cap batch by available disk (~3 GB per video)
    disk_now=$(free_gb)
    max_by_disk=$(( disk_now / 3 ))
    (( max_by_disk < 1 )) && max_by_disk=1
    effective=$(( BATCH_SIZE < max_by_disk ? BATCH_SIZE : max_by_disk ))

    batch=()
    for (( j=i; j<N && j<i+effective; j++ )); do
        batch+=("${NEED_DOWNLOAD[$j]}")
    done
    i=$(( i + ${#batch[@]} ))

    log "─── Batch: downloading ${#batch[@]} videos in parallel ───"
    for vid in "${batch[@]}"; do download_video "$vid" & done
    wait
    log "  Downloads complete."

    # Process + delete each video sequentially (single GPU)
    for vid in "${batch[@]}"; do
        if [[ ! -f "$VIDEO_DIR/$vid.mp4" ]]; then
            log "  [ERROR] $vid.mp4 missing after download — skipping."
            continue
        fi
        wait_gpu_free
        process_video "$vid"
        cleanup_video "$vid"
    done
done

# ─── Summary ──────────────────────────────────────────────────────────────────
log "=== run_rolling.sh complete ==="
log "Free disk: $(free_gb) GB"
print_accuracy
