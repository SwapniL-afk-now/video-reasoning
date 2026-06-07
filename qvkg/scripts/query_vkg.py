#!/usr/bin/env python3
"""Online QA CLI — query a pre-built VKG.

Usage:
    python scripts/query_vkg.py --vkg ./output/myvideo --question "Why did X happen?"
    python scripts/query_vkg.py --vkg ./output/myvideo --interactive
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Query a Video Knowledge Graph")
    parser.add_argument("--vkg",         required=True, help="VKG output directory")
    parser.add_argument("--question",    help="Single question to answer")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--model",       default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tp",          type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=65536,
                        help="Maximum model context length (default 65536)")
    parser.add_argument("--min-pixels",  type=int, default=200704,
                        help="Minimum pixels per image (default 256*28*28)")
    parser.add_argument("--max-pixels",  type=int, default=1003520,
                        help="Maximum pixels per image (default 1280*28*28)")
    args = parser.parse_args()

    vkg_path   = os.path.join(args.vkg, "vkg.json")
    index_path = os.path.join(args.vkg, "vkg.index")

    if not os.path.exists(vkg_path):
        print(f"ERROR: VKG not found at {vkg_path}")
        sys.exit(1)

    print("Loading VKG...")
    from qvkg.schema import VKGraph
    graph = VKGraph.load(vkg_path)
    print(f"  {len(graph.nodes)} nodes, {sum(len(v) for v in graph.edges.values())} edges")

    print("Loading FAISS index...")
    from qvkg.faiss_index import load_faiss_index
    faiss_index = load_faiss_index(index_path)
    if faiss_index is None:
        print(f"ERROR: FAISS index not found at {index_path}")
        sys.exit(1)

    from qvkg.frame_store import FrameStore
    frame_store = FrameStore(args.vkg, mode="r")

    print("Loading SigLIP encoder...")
    from qvkg.vllm_client import build_siglip_encoder
    siglip = build_siglip_encoder()

    print("Loading Qwen VLM...")
    from qvkg.vllm_client import build_llm
    llm = build_llm(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    from qvkg.query.qa import answer_question

    def ask(question: str):
        result = answer_question(
            question, graph, faiss_index, frame_store, llm, siglip
        )
        print(f"\nIntents:   {result.intents}")
        print(f"Subgraph:  {result.subgraph_size} nodes")
        print(f"Keyframes: {result.keyframes_used}")
        print(f"Evidence:  {result.evidence_nodes}")
        print(f"\nAnswer:\n{result.answer}\n")

    if args.interactive:
        print("\nQ-VKG Interactive Mode (Ctrl+C to exit)\n")
        while True:
            try:
                q = input("Question: ").strip()
                if q:
                    ask(q)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.question:
        ask(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
