#!/usr/bin/env bash
# run_rolling.sh — Batch download → build VKG → eval → delete pipeline
# Downloads BATCH_SIZE videos in parallel, processes them one by one on GPU,
# deletes everything after eval (keeps only results_all.jsonl).
# Idempotent: safe to kill and restart at any point.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
CSV="/workspace/LVBench_full.csv"
OUT="/workspace/output_4b"
VIDEOS="/workspace/videos"
RESULTS="$OUT/results_all.jsonl"
DONE_FILE="$OUT/done_videos.txt"
MODEL="Qwen/Qwen3.5-4B"
GPU_UTIL=0.70
LOG="$OUT/run_rolling.log"
BATCH_SIZE=8   # parallel downloads per batch

# Videos currently being processed by run_timed.sh — never touch these
IN_PROGRESS=("KktLi3UifPY" "JTa_Ue2MSwc")

# ─── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

get_all_vids() {
    python3 -c "
import csv
seen = []
with open('$CSV') as f:
    for r in csv.DictReader(f):
        vid = r['video_path'].replace('.mp4','')
        if vid not in seen:
            seen.append(vid)
for v in seen:
    print(v)
"
}

is_fully_evaluated() {
    local vid="$1"
    python3 -c "
import csv, json, sys
uids = [r['uid'] for r in csv.DictReader(open('$CSV')) if r['video_path'].replace('.mp4','') == '$vid']
if not uids:
    sys.exit(0)
try:
    answered = {json.loads(l)['uid'] for l in open('$RESULTS') if l.strip()}
except FileNotFoundError:
    answered = set()
sys.exit(0 if all(u in answered for u in uids) else 1)
"
}

is_vkg_built() { [[ -f "$OUT/$1/vkg.json" ]]; }

download_video() {
    local vid="$1"
    local dest="$VIDEOS/$vid.mp4"
    if [[ -f "$dest" ]]; then
        log "  [download] $vid already present, skipping."
        return 0
    fi
    log "  [download] Starting $vid ..."
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

build_siglip_phase() {
    local vid="$1"
    # Skip if both step1 and step2 already checkpointed
    if [[ -f "$OUT/$vid/.ckpt_step1_sampled" && -f "$OUT/$vid/.ckpt_step2_transcribed" ]]; then
        log "  [siglip] $vid already checkpointed, skipping."
        return 0
    fi
    log "  [siglip] Running SigLIP+Whisper phase for $vid ..."
    mkdir -p "$OUT/$vid"
    python3 /workspace/qvkg/scripts/build_vkg.py \
        --video "$VIDEOS/$vid.mp4" \
        --out "$OUT" \
        --questions-csv "$CSV" \
        --model "$MODEL" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --phase siglip \
        >> "$OUT/build_${vid}.log" 2>&1
}

build_llm_phase() {
    local vid="$1"
    if is_vkg_built "$vid"; then
        log "  [llm] VKG already present for $vid, skipping."
        return 0
    fi
    log "  [llm] Running LLM phase for $vid ..."
    python3 /workspace/qvkg/scripts/build_vkg.py \
        --video "$VIDEOS/$vid.mp4" \
        --out "$OUT" \
        --questions-csv "$CSV" \
        --model "$MODEL" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --phase llm \
        2>&1 | tee -a "$OUT/build_${vid}.log"
}

eval_video() {
    local vid="$1"
    log "  [eval] Evaluating $vid ..."
    python3 /workspace/qvkg/scripts/eval_lvbench.py \
        --csv "$CSV" \
        --vkg-dir "$OUT" \
        --video-dir "$VIDEOS" \
        --out "$RESULTS" \
        --two-stage \
        --model "$MODEL" \
        --gpu-memory-utilization "$GPU_UTIL" \
        2>&1 | tee "$OUT/eval_${vid}.log"
}

cleanup_video() {
    local vid="$1"
    log "  [cleanup] Removing all artifacts for $vid ..."
    rm -f  "$VIDEOS/$vid.mp4"
    rm -rf "$OUT/$vid"
    rm -f  "$OUT/build_${vid}.log" "$OUT/eval_${vid}.log" "$OUT/download_${vid}.log"
    rm -rf "$OUT/debug_${vid}"
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

process_video() {
    local vid="$1"
    local T0
    T0=$(date +%s)
    log "━━━ Processing: $vid ━━━  (free: $(free_gb) GB)"

    local BUILD_OK=true
    if ! build_llm_phase "$vid"; then
        log "  [ERROR] LLM phase failed for $vid."
        BUILD_OK=false
    fi

    if $BUILD_OK; then
        if ! eval_video "$vid"; then
            log "  [ERROR] eval failed for $vid (results may be partial)."
        fi
    fi

    cleanup_video "$vid"
    $BUILD_OK && mark_done "$vid"

    local T1
    T1=$(date +%s)
    log "  Wall time for $vid: $((T1 - T0))s"
    print_accuracy
}

# ─── Main ─────────────────────────────────────────────────────────────────────

mkdir -p "$OUT" "$VIDEOS"
touch "$DONE_FILE"
log "=== run_rolling.sh starting ==="
log "Free disk: $(free_gb) GB  |  batch_size=$BATCH_SIZE"

declare -A SKIP
for v in "${IN_PROGRESS[@]}"; do SKIP["$v"]=1; done

mapfile -t ALL_VIDS < <(get_all_vids)
log "Total videos in CSV: ${#ALL_VIDS[@]}"

ALREADY_HAVE=()
NEED_DOWNLOAD=()
for vid in "${ALL_VIDS[@]}"; do
    [[ -n "${SKIP[$vid]+x}" ]] && { log "  [skip] $vid — currently in progress."; continue; }
    if is_fully_evaluated "$vid"; then
        log "  [skip] $vid — already fully evaluated."
        continue
    fi
    if [[ -f "$VIDEOS/$vid.mp4" ]]; then
        ALREADY_HAVE+=("$vid")
    else
        NEED_DOWNLOAD+=("$vid")
    fi
done

log "Already downloaded (process first): ${#ALREADY_HAVE[@]}"
log "Need download: ${#NEED_DOWNLOAD[@]}"

# ─── Phase 1: Already-downloaded videos ───────────────────────────────────────
# Run SigLIP phase for all in parallel, then LLM+eval sequentially.
if [[ ${#ALREADY_HAVE[@]} -gt 0 ]]; then
    log "--- SigLIP phase for ${#ALREADY_HAVE[@]} already-downloaded videos (parallel) ---"
    for vid in "${ALREADY_HAVE[@]}"; do
        build_siglip_phase "$vid" &
    done
    wait
    log "--- SigLIP phase complete ---"
fi

for vid in "${ALREADY_HAVE[@]}"; do
    process_video "$vid"
done

# ─── Phase 2: Batch download → process → delete ───────────────────────────────
N="${#NEED_DOWNLOAD[@]}"
i=0

while (( i < N )); do
    # Disk guard: cap effective batch size by available space (~2 GB per video)
    disk_now=$(free_gb)
    max_by_disk=$(( disk_now / 3 ))
    (( max_by_disk < 1 )) && max_by_disk=1
    effective=$(( BATCH_SIZE < max_by_disk ? BATCH_SIZE : max_by_disk ))

    # Slice next batch
    batch=()
    for (( j=i; j<N && j<i+effective; j++ )); do
        batch+=("${NEED_DOWNLOAD[$j]}")
    done
    i=$(( i + ${#batch[@]} ))

    log "─── Batch $((i / effective)) / $(( (N + effective - 1) / effective )) — downloading ${#batch[@]} videos in parallel ───"

    # Download all in parallel
    for vid in "${batch[@]}"; do
        download_video "$vid" &
    done
    wait
    log "  Batch downloads complete."

    # SigLIP phase for all downloaded videos in parallel
    log "  Running SigLIP phase in parallel for downloaded videos..."
    for vid in "${batch[@]}"; do
        [[ -f "$VIDEOS/$vid.mp4" ]] && build_siglip_phase "$vid" &
    done
    wait
    log "  SigLIP phase complete."

    # LLM phase + eval + cleanup — sequential (single GPU)
    for vid in "${batch[@]}"; do
        if [[ ! -f "$VIDEOS/$vid.mp4" ]]; then
            log "  [ERROR] $vid.mp4 missing after download. Skipping."
            continue
        fi
        process_video "$vid"
    done
done

# ─── Final summary ────────────────────────────────────────────────────────────
log "=== run_rolling.sh complete ==="
log "Free disk: $(free_gb) GB"
print_accuracy
