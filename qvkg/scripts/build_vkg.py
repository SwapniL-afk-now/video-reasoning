#!/usr/bin/env python3
"""Offline VKG construction CLI.

Usage:
    python scripts/build_vkg.py --video foo.mp4 --out ./output
    python scripts/build_vkg.py --video foo.mp4 --out ./output \\
        --questions-csv LVBench_full.csv   # enables question-aware dense sampling
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_question_time_refs(csv_path: str, video_filename: str):
    """Parse time_references for a specific video from LVBench CSV."""
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


def main():
    parser = argparse.ArgumentParser(description="Build Video Knowledge Graph")
    parser.add_argument("--video",          required=True, help="Path to input video")
    parser.add_argument("--out",            required=True, help="Output directory")
    parser.add_argument("--budget",         type=int, default=500, help="Keyframe budget")
    parser.add_argument("--model",          default="Qwen/Qwen3.5-4B")
    parser.add_argument("--whisper-model",  default="large-v3-turbo",
                        help="faster-whisper model size (default large-v3-turbo, ~8× faster)")
    parser.add_argument("--whisper-compute-type", default="int8_float16",
                        help="faster-whisper compute_type (default int8_float16)")
    parser.add_argument("--no-whisper",     action="store_true")
    parser.add_argument("--coarse-fps",     type=float, default=1.0,
                        help="FPS for coarse frame extraction (default 1.0)")
    parser.add_argument("--coarse-frame-cap", type=int, default=0,
                        help="Cap total coarse frames (0 = uncapped); tapers fps on long videos")
    parser.add_argument("--flow-max-dim",   type=int, default=256,
                        help="Downscale long-side px before optical flow (default 256)")
    parser.add_argument("--no-optical-flow", action="store_true",
                        help="Drop the motion term and renormalize keyframe-score weights")
    parser.add_argument("--tp",             type=int, default=1,
                        help="Tensor parallel size (GPUs)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65,
                        help="vLLM GPU memory utilization (lower = more room for SigLIP)")
    parser.add_argument("--questions-csv",  default=None,
                        help="LVBench CSV — enables question-aware dense pre-sampling")
    parser.add_argument("--video-type",     default=None,
                        help="Video type hint (sport/live/cartoon/…) for motion ranking")
    parser.add_argument("--hard-boundary",  type=float, default=0.75,
                        help="Hard scene boundary threshold (histogram diff)")
    parser.add_argument("--soft-boundary",  type=float, default=0.5,
                        help="Soft scene boundary threshold (embedding distance)")
    parser.add_argument("--max-model-len",  type=int, default=65536,
                        help="Maximum model context length (default 65536)")
    parser.add_argument("--min-pixels",     type=int, default=200704,
                        help="Minimum pixels per image for VLM processor (default 256*28*28)")
    parser.add_argument("--max-pixels",     type=int, default=1003520,
                        help="Maximum pixels per image for VLM processor (default 1280*28*28)")
    parser.add_argument("--subtitles",      default=None,
                        help="Path to .srt/.vtt subtitle track (auto-discovered next "
                             "to the video if omitted) — authoritative text for "
                             "text-referred questions")
    parser.add_argument("--phase",          choices=["all", "siglip", "llm"], default="all",
                        help="all: full pipeline (default); "
                             "siglip: Steps 1+2+2b only (parallel-safe, no vLLM); "
                             "llm: Steps 0+3-10 only (requires siglip phase done first)")
    args = parser.parse_args()

    video_filename = os.path.basename(args.video)
    video_id       = os.path.splitext(video_filename)[0]
    output_dir     = os.path.join(args.out, video_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Building VKG for: {args.video}")
    print(f"Output dir:       {output_dir}")

    from qvkg.vllm_client import build_llm, build_siglip_encoder

    print("Loading SigLIP encoder...")
    siglip = build_siglip_encoder()

    llm = None
    if args.phase in ("all", "llm"):
        print("Preparing Qwen VLM (vLLM engine loads lazily on first use)...")
        llm = build_llm(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            lazy=True,
        )

    whisper_model = None
    if args.phase in ("all", "siglip") and not args.no_whisper:
        # The builder transcribes in an isolated worker process keyed by model
        # size + compute type — pass those via a lightweight namespace rather
        # than eagerly loading the model here (which the worker would re-load).
        import types
        whisper_model = types.SimpleNamespace(
            _model_size=args.whisper_model,
            _compute_type=args.whisper_compute_type,
        )
        print(f"Whisper {args.whisper_model} ({args.whisper_compute_type}) "
              f"— will run in worker process")

    # Question-aware pre-sampling
    question_time_refs = []
    if args.questions_csv:
        question_time_refs = load_question_time_refs(args.questions_csv, video_filename)
        print(f"  Loaded {len(question_time_refs)} question time-references for dense pre-sampling")

    config = {
        "frame_budget":            args.budget,
        "semantic_threshold":      0.78,
        "semantic_k_neighbors":    10,
        "causal_min_confidence":   0.6,
        "hard_boundary_thresh":    args.hard_boundary,
        "soft_boundary_thresh":    args.soft_boundary,
        "question_time_refs":      question_time_refs,
        "video_type":              args.video_type,
        "subtitle_path":           args.subtitles,
        "coarse_fps":              args.coarse_fps,
        "coarse_frame_cap":        args.coarse_frame_cap,
        "flow_max_dim":            args.flow_max_dim,
        "use_optical_flow":        not args.no_optical_flow,
        "whisper_compute_type":    args.whisper_compute_type,
    }

    from qvkg.builder import VKGBuilder
    builder = VKGBuilder(llm, whisper_model, siglip, config)
    graph   = builder.build(args.video, output_dir, phase=args.phase)

    if graph is not None:
        print(f"\nDone. VKG saved to {output_dir}/vkg.json")
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Edges: {sum(len(v) for v in graph.edges.values())}")
    else:
        # Explicitly release GPU memory before process exits so vLLM can claim it.
        import gc
        import torch
        del builder, siglip, whisper_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\nSigLIP phase done. GPU memory released. Checkpoints saved to {output_dir}/")


if __name__ == "__main__":
    main()
