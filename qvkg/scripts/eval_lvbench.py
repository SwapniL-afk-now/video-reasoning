#!/usr/bin/env python3
"""LVBench evaluation runner.

Usage:
    python scripts/eval_lvbench.py \\
        --csv /workspace/LVBench_full.csv \\
        --vkg-dir ./output \\
        --video-dir /workspace \\
        --out results.jsonl

Loads each video's VKG once, runs all questions for that video, writes results
incrementally (resume-safe). Prints per-category accuracy at the end.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_questions(csv_path: str) -> Dict[str, List[dict]]:
    """Returns dict: video_path → list of question dicts."""
    by_video = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_video[row["video_path"]].append(row)
    return dict(by_video)


def load_answered_uids(out_path: str) -> set:
    """Load already-answered UIDs for resume support."""
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
    """Parse question_type field like "['event understanding', 'reasoning']"."""
    raw = raw.strip().strip("[]").replace("'", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def print_accuracy_report(results: List[dict]) -> None:
    overall_correct = sum(1 for r in results if r["correct"])
    print(f"\n{'='*60}")
    print(f"Overall accuracy: {overall_correct}/{len(results)} "
          f"= {overall_correct/len(results)*100:.1f}%")

    # Per question type
    by_qt = defaultdict(lambda: [0, 0])
    for r in results:
        for qt in r.get("question_types", []):
            by_qt[qt][1] += 1
            if r["correct"]:
                by_qt[qt][0] += 1

    print("\nPer question type:")
    for qt, (correct, total) in sorted(by_qt.items(), key=lambda x: -x[1][1]):
        print(f"  {qt:<35} {correct:>3}/{total:<3} = {correct/total*100:5.1f}%")

    # Per video type
    by_vt = defaultdict(lambda: [0, 0])
    for r in results:
        vt = r.get("video_type", "unknown")
        by_vt[vt][1] += 1
        if r["correct"]:
            by_vt[vt][0] += 1

    print("\nPer video type:")
    for vt, (correct, total) in sorted(by_vt.items(), key=lambda x: -x[1][1]):
        print(f"  {vt:<20} {correct:>3}/{total:<3} = {correct/total*100:5.1f}%")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q-VKG on LVBench")
    parser.add_argument("--csv",       required=True, help="LVBench_full.csv path")
    parser.add_argument("--vkg-dir",   required=True, help="Directory with per-video VKG subdirs")
    parser.add_argument("--video-dir", required=True, help="Directory containing .mp4 files")
    parser.add_argument("--out",       required=True, help="Output JSONL path")
    parser.add_argument("--model",     default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tp",        type=int, default=1)
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max questions to evaluate (for debugging)")
    parser.add_argument("--max-model-len", type=int, default=65536,
                        help="Maximum model context length (default 65536)")
    parser.add_argument("--min-pixels",    type=int, default=200704,
                        help="Minimum pixels per image (default 256*28*28)")
    parser.add_argument("--max-pixels",    type=int, default=1003520,
                        help="Maximum pixels per image (default 1280*28*28)")
    args = parser.parse_args()

    # Load models once
    print("Loading SigLIP encoder...")
    from qvkg.vllm_client import build_llm, build_siglip_encoder
    siglip = build_siglip_encoder()

    print(f"Loading {args.model} via vLLM...")
    llm = build_llm(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    questions_by_video = load_questions(args.csv)
    answered_uids      = load_answered_uids(args.out)
    all_results: List[dict] = []

    from qvkg.faiss_index import load_faiss_index
    from qvkg.frame_store import FrameStore
    from qvkg.query.qa import answer_question
    from qvkg.schema import VKGraph

    total_done = 0
    out_f = open(args.out, "a")

    try:
        for video_filename, questions in questions_by_video.items():
            video_id   = os.path.splitext(video_filename)[0]
            vkg_path   = os.path.join(args.vkg_dir, video_id, "vkg.json")
            index_path = os.path.join(args.vkg_dir, video_id, "vkg.index")
            video_path = os.path.join(args.video_dir, video_filename)

            if not os.path.exists(vkg_path):
                print(f"  [SKIP] No VKG for {video_filename}")
                continue

            print(f"\nVideo: {video_filename} ({len(questions)} questions)")

            # Load VKG once per video
            graph       = VKGraph.load(vkg_path)
            faiss_index = load_faiss_index(index_path)
            frame_store = FrameStore(os.path.join(args.vkg_dir, video_id), mode="r")
            video_type  = questions[0].get("type", "")

            if faiss_index is None:
                print(f"  [WARN] No FAISS index for {video_filename} — FAISS path disabled")

            for q in questions:
                uid = q["uid"]
                if uid in answered_uids:
                    continue
                if args.limit and total_done >= args.limit:
                    break

                question_text = q["question"]
                gt_answer     = q["answer"].strip().upper()
                time_ref      = q.get("time_reference", "").strip() or None
                qt            = parse_qt(q.get("question_type", ""))

                try:
                    result = answer_question(
                        question       = question_text,
                        graph          = graph,
                        faiss_index    = faiss_index,
                        frame_store    = frame_store,
                        llm            = llm,
                        siglip_encoder = siglip,
                        question_type  = qt,
                        time_reference = time_ref,
                        video_path     = video_path if os.path.exists(video_path) else None,
                        video_type     = video_type,
                        mcq            = True,
                    )
                    pred = result.answer.strip().upper()
                    correct = pred == gt_answer
                except Exception as e:
                    pred    = "ERROR"
                    correct = False
                    print(f"    [ERR] uid={uid}: {e}")

                record = {
                    "uid":            uid,
                    "video":          video_filename,
                    "video_type":     video_type,
                    "question_types": qt,
                    "time_reference": time_ref,
                    "predicted":      pred,
                    "ground_truth":   gt_answer,
                    "correct":        correct,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_results.append(record)
                answered_uids.add(uid)
                total_done += 1

                status = "✓" if correct else "✗"
                print(f"  {status} uid={uid} pred={pred} gt={gt_answer} "
                      f"[{', '.join(qt)}] t={time_ref}")

            if args.limit and total_done >= args.limit:
                break

    finally:
        out_f.close()

    if all_results:
        print_accuracy_report(all_results)
    else:
        print("No new questions evaluated.")


if __name__ == "__main__":
    main()
