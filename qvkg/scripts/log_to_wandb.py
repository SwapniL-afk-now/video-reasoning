#!/usr/bin/env python3
"""Log LVBench evaluation results to W&B.

Usage:
    ./log_to_wandb.py --results results.jsonl --video JTa_Ue2MSwc --duration 123.4
    ./log_to_wandb.py --results results.jsonl                         # cumulative only
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import wandb

WANDB_RUN_ID_FILE = "/workspace/.wandb_run_id"


def _load_run_id():
    if os.path.exists(WANDB_RUN_ID_FILE):
        with open(WANDB_RUN_ID_FILE) as f:
            return f.read().strip()
    return None


def _save_run_id(rid: str):
    os.makedirs(os.path.dirname(WANDB_RUN_ID_FILE), exist_ok=True)
    with open(WANDB_RUN_ID_FILE, "w") as f:
        f.write(rid)


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def log_metrics(results: list[dict], video_id: str | None, duration_sec: float | None):
    rid = _load_run_id()
    if rid is None:
        run = wandb.init(project="LVBench-eval", name="full-run")
        _save_run_id(run.id)
    else:
        run = wandb.init(project="LVBench-eval", id=rid, resume="must")

    if video_id is not None:
        video_results = [r for r in results if r.get("video", "").replace(".mp4", "") == video_id]
        if video_results:
            correct = sum(1 for r in video_results if r["correct"])
            total = len(video_results)
            acc = correct / total if total > 0 else 0
            log = {
                f"per_video/{video_id}/correct": correct,
                f"per_video/{video_id}/total": total,
                f"per_video/{video_id}/accuracy": acc,
            }
            if duration_sec is not None:
                log[f"per_video/{video_id}/duration_sec"] = duration_sec
            try:
                import torch
                log[f"per_video/{video_id}/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                pass
            wandb.log(log)

    overall_correct = sum(1 for r in results if r["correct"])
    overall_total = len(results)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0

    by_qt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        for qt in r.get("question_types", []):
            by_qt[qt][1] += 1
            if r["correct"]:
                by_qt[qt][0] += 1

    log = {
        "overall/correct": overall_correct,
        "overall/total": overall_total,
        "overall/accuracy": overall_acc,
    }
    wandb.log(log)

    for qt, (corr, tot) in sorted(by_qt.items()):
        if tot > 0:
            wandb.log({
                f"per_type/{qt}/correct": corr,
                f"per_type/{qt}/total": tot,
                f"per_type/{qt}/accuracy": corr / tot,
            })

    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Log LVBench results to W&B")
    parser.add_argument("--results", required=True, help="Path to results JSONL")
    parser.add_argument("--video", default=None, help="Video ID for per-video metrics")
    parser.add_argument("--duration", type=float, default=None, help="Wall-clock seconds for this video")
    args = parser.parse_args()

    results = load_results(args.results)
    log_metrics(results, args.video, args.duration)


if __name__ == "__main__":
    main()
