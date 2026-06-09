#!/usr/bin/env python3
"""Single-GPU end-to-end pipeline: build VKG → eval → cleanup, one video at a time.

Usage:
    python scripts/run_pipeline.py \\
        --csv /workspace/LVBench_full.csv \\
        --video-dir /workspace/videos \\
        --out /workspace/output \\
        --results results.jsonl

    # With chunked parallel frame extraction (4 CPU threads per video):
    python scripts/run_pipeline.py \\
        --csv /workspace/LVBench_full.csv \\
        --video-dir /workspace/videos \\
        --out /workspace/output \\
        --results results.jsonl \\
        --n-chunk-workers 4

The script processes each video sequentially:
  1. Build VKG (with n_chunk_workers parallel frame extraction chunks)
  2. Run LVBench evaluation for that video
  3. Delete the video file and VKG directory to free disk space
  4. Move to next video

Resume-safe: already-answered question UIDs are skipped; videos whose VKG
already exists skip the build step.
"""

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import types
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# CSV helpers (shared with eval_lvbench.py)
# ---------------------------------------------------------------------------

def load_questions(csv_path: str) -> Dict[str, List[dict]]:
    by_video: Dict[str, List[dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_video[row["video_path"]].append(row)
    return dict(by_video)


def load_answered_uids(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    answered = set()
    with open(out_path) as f:
        for line in f:
            try:
                answered.add(json.loads(line)["uid"])
            except Exception:
                pass
    return answered


def parse_qt(raw: str) -> List[str]:
    raw = raw.strip().strip("[]").replace("'", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def load_question_time_refs(csv_path: str, video_filename: str):
    from qvkg.query.intent import parse_time_reference
    time_refs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("video_path", "") == video_filename:
                tr = row.get("time_reference", "").strip()
                if tr:
                    parsed = parse_time_reference(tr)
                    if parsed:
                        time_refs.append(parsed)
    return time_refs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_vkg(video_path: str, output_dir: str, args, siglip, llm, whisper_model) -> bool:
    """Build VKG for one video. Returns True on success."""
    from qvkg.builder import VKGBuilder

    video_filename = os.path.basename(video_path)
    question_time_refs = load_question_time_refs(args.csv, video_filename)
    if question_time_refs:
        print(f"  Question-aware pre-sampling: {len(question_time_refs)} windows")

    config = {
        "frame_budget":          args.budget,
        "semantic_threshold":    0.78,
        "semantic_k_neighbors":  10,
        "causal_min_confidence": 0.6,
        "hard_boundary_thresh":  args.hard_boundary,
        "soft_boundary_thresh":  args.soft_boundary,
        "question_time_refs":    question_time_refs,
        "n_chunk_workers":       args.n_chunk_workers,
        "coarse_fps":            args.coarse_fps,
    }

    builder = VKGBuilder(llm, whisper_model, siglip, config)
    try:
        graph = builder.build(video_path, output_dir, phase="all")
        if graph is not None:
            print(f"  VKG built: {len(graph.nodes)} nodes, "
                  f"{sum(len(v) for v in graph.edges.values())} edges")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [ERR] VKG build failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_video(
    video_filename: str,
    questions: List[dict],
    vkg_dir: str,
    video_dir: str,
    answered_uids: set,
    out_f,
    all_results: List[dict],
    args,
    llm,
    siglip,
) -> None:
    from qvkg.faiss_index import load_faiss_index
    from qvkg.frame_store import FrameStore
    from qvkg.schema import VKGraph
    from qvkg.query.qa import batch_answer_questions
    from qvkg.query.two_stage import batch_two_stage_answer_questions
    from qvkg.query.agent import agent_answer_question
    from qvkg.query.react_agent import react_answer_question

    video_id   = os.path.splitext(video_filename)[0]
    vkg_path   = os.path.join(vkg_dir, video_id, "vkg.json")
    index_path = os.path.join(vkg_dir, video_id, "vkg.index")
    video_path = os.path.join(video_dir, video_filename)

    if not os.path.exists(vkg_path):
        print(f"  [SKIP eval] No VKG at {vkg_path}")
        return

    pending = [q for q in questions if q["uid"] not in answered_uids]
    if not pending:
        print(f"  All questions already answered, skipping eval")
        return

    print(f"  Evaluating {len(pending)} questions...")

    graph       = VKGraph.load(vkg_path)
    faiss_index = load_faiss_index(index_path)
    frame_store = FrameStore(os.path.join(vkg_dir, video_id), mode="r")
    video_type  = questions[0].get("type", "")
    vp = video_path if os.path.exists(video_path) else None

    if args.react:
        results = []
        for q in pending:
            try:
                result = react_answer_question(
                    question=q["question"],
                    graph=graph,
                    faiss_index=faiss_index,
                    llm=llm,
                    siglip_encoder=siglip,
                    video_path=vp,
                    question_type=parse_qt(q.get("question_type", "")),
                    time_reference=q.get("time_reference", "").strip() or None,
                    mcq=True,
                    max_turns=args.max_turns,
                )
            except Exception as e:
                print(f"    [ERR] uid={q['uid']}: {e}")
                result = None
            results.append(result)

    elif args.two_stage:
        batch_input = [
            {
                "question":      q["question"],
                "question_type": parse_qt(q.get("question_type", "")),
                "time_reference": q.get("time_reference", "").strip() or None,
                "uid":           q["uid"],
            }
            for q in pending
        ]
        try:
            results = batch_two_stage_answer_questions(
                questions=batch_input,
                graph=graph,
                faiss_index=faiss_index,
                llm=llm,
                siglip_encoder=siglip,
                video_path=vp,
                mcq=True,
            )
        except Exception as e:
            print(f"  [ERR] two-stage batch failed: {e}")
            results = [None] * len(pending)

    elif args.agent:
        results = []
        for q in pending:
            try:
                result = agent_answer_question(
                    question=q["question"],
                    graph=graph,
                    faiss_index=faiss_index,
                    llm=llm,
                    siglip_encoder=siglip,
                    video_path=vp,
                    question_type=parse_qt(q.get("question_type", "")),
                    time_reference=q.get("time_reference", "").strip() or None,
                    mcq=True,
                    max_turns=args.max_turns,
                )
            except Exception as e:
                print(f"    [ERR] uid={q['uid']}: {e}")
                result = None
            results.append(result)

    else:
        batch_input = [
            {
                "question":      q["question"],
                "question_type": parse_qt(q.get("question_type", "")),
                "time_reference": q.get("time_reference", "").strip() or None,
            }
            for q in pending
        ]
        try:
            results = batch_answer_questions(
                questions=batch_input,
                graph=graph,
                faiss_index=faiss_index,
                frame_store=frame_store,
                llm=llm,
                siglip_encoder=siglip,
                video_path=vp,
                video_type=video_type,
                mcq=True,
            )
        except Exception as e:
            print(f"  [ERR] batch failed: {e}")
            results = [None] * len(pending)

    for q, result in zip(pending, results):
        gt_answer = q["answer"].strip().upper()
        qt        = parse_qt(q.get("question_type", ""))
        if result is None:
            pred, correct = "ERROR", False
        else:
            pred    = result.answer.strip().upper()
            correct = pred == gt_answer

        record = {
            "uid":            q["uid"],
            "video":          video_filename,
            "video_type":     video_type,
            "question_types": qt,
            "time_reference": q.get("time_reference", "").strip() or None,
            "predicted":      pred,
            "ground_truth":   gt_answer,
            "correct":        correct,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        all_results.append(record)
        answered_uids.add(q["uid"])

        status = "✓" if correct else "✗"
        print(f"    {status} uid={q['uid']} pred={pred} gt={gt_answer} [{', '.join(qt)}]")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup(video_path: str, vkg_dir: str, keep_results: bool = True) -> None:
    """Delete video file and VKG directory to free disk space."""
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"  Deleted video: {video_path}")

    if os.path.exists(vkg_dir):
        if keep_results:
            # Keep only the results JSONL inside the VKG dir (if any), delete everything else
            for entry in os.listdir(vkg_dir):
                entry_path = os.path.join(vkg_dir, entry)
                if not entry.endswith(".jsonl"):
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
            # If dir is now empty, remove it too
            if not os.listdir(vkg_dir):
                os.rmdir(vkg_dir)
        else:
            shutil.rmtree(vkg_dir)
        print(f"  Cleaned up VKG dir: {vkg_dir}")


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------

def print_accuracy_report(results: List[dict]) -> None:
    if not results:
        return
    from collections import defaultdict
    correct_total = sum(1 for r in results if r["correct"])
    print(f"\n{'='*60}")
    print(f"Overall: {correct_total}/{len(results)} = {correct_total/len(results)*100:.1f}%")

    by_qt: Dict = defaultdict(lambda: [0, 0])
    for r in results:
        for qt in r.get("question_types", []):
            by_qt[qt][1] += 1
            if r["correct"]:
                by_qt[qt][0] += 1
    print("\nPer question type:")
    for qt, (c, t) in sorted(by_qt.items(), key=lambda x: -x[1][1]):
        print(f"  {qt:<35} {c:>3}/{t:<3} = {c/t*100:5.1f}%")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-GPU: build VKG → eval → cleanup, one video at a time"
    )
    parser.add_argument("--csv",        required=True, help="LVBench_full.csv path")
    parser.add_argument("--video-dir",  required=True, help="Directory with .mp4 files")
    parser.add_argument("--out",        required=True, help="Output directory for VKGs")
    parser.add_argument("--results",    required=True, help="Output JSONL path for eval results")

    # Model args
    parser.add_argument("--model",     default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tp",        type=int,   default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len",  type=int, default=65536)
    parser.add_argument("--min-pixels",     type=int, default=200704)
    parser.add_argument("--max-pixels",     type=int, default=1003520)
    parser.add_argument("--whisper-model",  default="large-v3")
    parser.add_argument("--no-whisper",     action="store_true",
                        help="Disable Whisper transcription (default: Whisper ON)")

    # Sampler args
    parser.add_argument("--budget",          type=int,   default=500)
    parser.add_argument("--hard-boundary",   type=float, default=0.75)
    parser.add_argument("--soft-boundary",   type=float, default=0.5)
    parser.add_argument("--n-chunk-workers", type=int,   default=8,
                        help="Parallel video chunks for frame extraction (default 8)")
    parser.add_argument("--coarse-fps",      type=float, default=1.0,
                        help="FPS for coarse frame extraction (default 1.0)")

    # QA mode
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--agent",     action="store_true")
    parser.add_argument("--react",     action="store_true")
    parser.add_argument("--max-turns", type=int, default=6)

    # Pipeline control
    parser.add_argument("--no-cleanup",   action="store_true",
                        help="Keep video and VKG files after eval (default: delete them)")
    parser.add_argument("--skip-build",   action="store_true",
                        help="Skip VKG build (only eval + cleanup)")
    parser.add_argument("--skip-eval",    action="store_true",
                        help="Skip eval (only build + cleanup)")
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="Process at most N videos (for debugging)")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Load models once — kept alive for the full run
    print("Loading SigLIP encoder...")
    from qvkg.vllm_client import build_llm, build_siglip_encoder
    siglip = build_siglip_encoder()

    print(f"Preparing {args.model} via vLLM...")
    llm = build_llm(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        lazy=True,
    )

    # Whisper runs in an isolated worker process (spawned by builder._transcribe_background).
    # The main process only holds the model-size string — no GPU memory consumed here.
    whisper_model = None
    if not args.no_whisper and not args.skip_build:
        try:
            import faster_whisper  # noqa: F401 — just verify it's installed
            whisper_model = types.SimpleNamespace(
                _model_size=args.whisper_model
            )
            print(f"Whisper {args.whisper_model} — will run in worker process")
        except ImportError:
            print("  faster-whisper not installed — skipping audio")

    questions_by_video = load_questions(args.csv)
    answered_uids      = load_answered_uids(args.results)
    all_results: List[dict] = []

    videos = list(questions_by_video.keys())
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    out_f = open(args.results, "a")
    try:
        for video_idx, video_filename in enumerate(videos):
            video_id   = os.path.splitext(video_filename)[0]
            video_path = os.path.join(args.video_dir, video_filename)
            vkg_dir    = os.path.join(args.out, video_id)
            questions  = questions_by_video[video_filename]

            print(f"\n[{video_idx+1}/{len(videos)}] {video_filename}")

            # --- Build ---
            vkg_exists = os.path.exists(os.path.join(vkg_dir, "vkg.json"))
            if not args.skip_build and not vkg_exists:
                if not os.path.exists(video_path):
                    print(f"  [SKIP] Video not found: {video_path}")
                    continue
                print(f"  Building VKG (n_chunk_workers={args.n_chunk_workers})...")
                os.makedirs(vkg_dir, exist_ok=True)
                build_ok = build_vkg(video_path, vkg_dir, args, siglip, llm, whisper_model)
                if not build_ok:
                    print(f"  [SKIP eval] Build failed for {video_filename}")
                    continue
            elif vkg_exists:
                print(f"  VKG already built, skipping build step")

            # --- Eval ---
            if not args.skip_eval:
                eval_video(
                    video_filename=video_filename,
                    questions=questions,
                    vkg_dir=args.out,
                    video_dir=args.video_dir,
                    answered_uids=answered_uids,
                    out_f=out_f,
                    all_results=all_results,
                    args=args,
                    llm=llm,
                    siglip=siglip,
                )

            # --- Cleanup ---
            if not args.no_cleanup:
                print(f"  Cleaning up...")
                cleanup(video_path, vkg_dir, keep_results=False)

    finally:
        out_f.close()

    if all_results:
        print_accuracy_report(all_results)
    else:
        print("No new questions evaluated.")


if __name__ == "__main__":
    main()
