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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65,
                        help="vLLM GPU memory utilization (lower = more room for SigLIP)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max questions to evaluate (for debugging)")
    parser.add_argument("--max-model-len", type=int, default=65536,
                        help="Maximum model context length (default 65536)")
    parser.add_argument("--min-pixels",    type=int, default=200704,
                        help="Minimum pixels per image (default 256*28*28)")
    parser.add_argument("--max-pixels",    type=int, default=1003520,
                        help="Maximum pixels per image (default 1280*28*28)")
    parser.add_argument("--agent",         action="store_true", default=False,
                        help="Use agentic graph search (VLM drives retrieval) instead of one-shot")
    parser.add_argument("--two-stage",     action="store_true", default=False,
                        help="Use two-stage QA: planner call → context assembly → answerer call")
    parser.add_argument("--react",         action="store_true", default=False,
                        help="Use ReAct agent (Thought/Action/Observation loop, plain-text actions)")
    parser.add_argument("--walker",        action="store_true", default=False,
                        help="Use inference-time agentic graph traversal (warrant-gated walker)")
    parser.add_argument("--max-hops",      type=int, default=4,
                        help="Max traversal hops per question in walker mode (default 4)")
    parser.add_argument("--theta-cov",     type=float, default=1.0,
                        help="Coverage threshold for the walker stop rule (default 1.0)")
    parser.add_argument("--max-turns",     type=int, default=6,
                        help="Max tool-call turns per question in agent/react mode (default 6)")
    parser.add_argument("--debug-dir",     type=str, default=None,
                        help="Directory to write per-question debug logs and frame images")
    args = parser.parse_args()

    # Load models once
    print("Loading SigLIP encoder...")
    from qvkg.vllm_client import build_llm, build_siglip_encoder
    siglip = build_siglip_encoder()

    print(f"Preparing {args.model} (vLLM engine loads lazily on first use)...")
    llm = build_llm(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        lazy=True,
    )

    questions_by_video = load_questions(args.csv)
    answered_uids      = load_answered_uids(args.out)
    all_results: List[dict] = []

    from qvkg.faiss_index import load_faiss_index
    from qvkg.frame_store import FrameStore
    from qvkg.query.qa import batch_answer_questions
    from qvkg.query.agent import agent_answer_question
    from qvkg.query.two_stage import two_stage_answer_question, batch_two_stage_answer_questions
    from qvkg.query.react_agent import react_answer_question
    from qvkg.query.walker import batch_walk_answer_questions
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

            # Filter to unanswered questions
            pending = []
            for q in questions:
                if q["uid"] in answered_uids:
                    continue
                if args.limit and total_done + len(pending) >= args.limit:
                    break
                pending.append(q)

            if not pending:
                print(f"\nVideo: {video_filename} — all questions already answered, skipping")
                continue

            print(f"\nVideo: {video_filename} ({len(pending)} pending / {len(questions)} total)")

            graph       = VKGraph.load(vkg_path)
            faiss_index = load_faiss_index(index_path)
            frame_store = FrameStore(os.path.join(args.vkg_dir, video_id), mode="r")
            video_type  = questions[0].get("type", "")

            if faiss_index is None:
                print(f"  [WARN] No FAISS index for {video_filename}")

            vp = video_path if os.path.exists(video_path) else None

            if args.walker:
                # Inference-time agentic graph traversal, batched by hop.
                print(f"  Walker mode: {len(pending)} questions "
                      f"(max_hops={args.max_hops}, theta_cov={args.theta_cov})...")
                batch_input = [
                    {
                        "question":       q["question"],
                        "question_type":  parse_qt(q.get("question_type", "")),
                        "time_reference": q.get("time_reference", "").strip() or None,
                        "uid":            q["uid"],
                    }
                    for q in pending
                ]
                try:
                    results = batch_walk_answer_questions(
                        questions   = batch_input,
                        graph       = graph,
                        faiss_index = faiss_index,
                        llm         = llm,
                        siglip      = siglip,
                        video_path  = vp,
                        frame_store = frame_store,
                        mcq         = True,
                        debug_dir   = args.debug_dir,
                        max_hops    = args.max_hops,
                        theta_cov   = args.theta_cov,
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [ERR] walker batch failed: {e}")
                    results = [None] * len(pending)
            elif args.react:
                # ReAct: Thought/Action/Observation loop with plain-text actions (serial)
                print(f"  ReAct mode: {len(pending)} questions (max {args.max_turns} turns each)...")
                results = []
                for q in pending:
                    qt       = parse_qt(q.get("question_type", ""))
                    time_ref = q.get("time_reference", "").strip() or None
                    try:
                        result = react_answer_question(
                            question       = q["question"],
                            graph          = graph,
                            faiss_index    = faiss_index,
                            llm            = llm,
                            siglip_encoder = siglip,
                            video_path     = vp,
                            question_type  = qt,
                            time_reference = time_ref,
                            mcq            = True,
                            max_turns      = args.max_turns,
                        )
                    except Exception as e:
                        print(f"    [ERR] uid={q['uid']}: {e}")
                        result = None
                    results.append(result)
            elif args.two_stage:
                # Two-stage batched: 1 planner call + 1 answerer call for all questions
                print(f"  Two-stage (batch) mode: {len(pending)} questions...")
                batch_input = [
                    {
                        "question":       q["question"],
                        "question_type":  parse_qt(q.get("question_type", "")),
                        "time_reference": q.get("time_reference", "").strip() or None,
                        "uid":            q["uid"],
                    }
                    for q in pending
                ]
                try:
                    results = batch_two_stage_answer_questions(
                        questions      = batch_input,
                        graph          = graph,
                        faiss_index    = faiss_index,
                        llm            = llm,
                        siglip_encoder = siglip,
                        video_path     = vp,
                        mcq            = True,
                        debug_dir      = args.debug_dir,
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [ERR] batch failed: {e}")
                    results = [None] * len(pending)
            elif args.agent:
                # Agentic mode: VLM drives graph traversal per question (serial)
                print(f"  Agent mode: {len(pending)} questions (max {args.max_turns} turns each)...")
                results = []
                for q in pending:
                    qt       = parse_qt(q.get("question_type", ""))
                    time_ref = q.get("time_reference", "").strip() or None
                    try:
                        result = agent_answer_question(
                            question       = q["question"],
                            graph          = graph,
                            faiss_index    = faiss_index,
                            llm            = llm,
                            siglip_encoder = siglip,
                            video_path     = vp,
                            question_type  = qt,
                            time_reference = time_ref,
                            mcq            = True,
                            max_turns      = args.max_turns,
                        )
                    except Exception as e:
                        print(f"    [ERR] uid={q['uid']}: {e}")
                        result = None
                    results.append(result)
            else:
                # One-shot batch mode: all questions in a single vLLM call
                batch_input = [
                    {
                        "question":       q["question"],
                        "question_type":  parse_qt(q.get("question_type", "")),
                        "time_reference": q.get("time_reference", "").strip() or None,
                    }
                    for q in pending
                ]
                print(f"  Sending {len(pending)} questions to vLLM in one batch...")
                try:
                    results = batch_answer_questions(
                        questions      = batch_input,
                        graph          = graph,
                        faiss_index    = faiss_index,
                        frame_store    = frame_store,
                        llm            = llm,
                        siglip_encoder = siglip,
                        video_path     = vp,
                        video_type     = video_type,
                        mcq            = True,
                    )
                except Exception as e:
                    print(f"  [ERR] batch failed: {e}")
                    results = [None] * len(pending)

            for q, result in zip(pending, results):
                uid       = q["uid"]
                gt_answer = q["answer"].strip().upper()
                qt        = parse_qt(q.get("question_type", ""))
                time_ref  = q.get("time_reference", "").strip() or None

                if result is None:
                    pred, correct = "ERROR", False
                else:
                    pred    = result.answer.strip().upper()
                    correct = pred == gt_answer

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

                status = "✓" if correct else "✗"
                print(f"  {status} uid={uid} pred={pred} gt={gt_answer} "
                      f"[{', '.join(qt)}] t={time_ref}")

            total_done += len(pending)
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
