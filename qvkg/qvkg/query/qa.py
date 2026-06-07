from __future__ import annotations

"""Single-call online QA with time-reference-aware retrieval and MCQ output."""

from typing import List, Optional, Tuple

import faiss
import numpy as np

from ..faiss_index import load_faiss_index
from ..frame_store import FrameStore
from ..schema import AnswerResult, VKGraph
from ..vllm_client import (
    MCQ_REASONING_SAMPLING,
    MCQ_SYSTEM_PROMPT,
    QA_SAMPLING,
    extract_mcq_answer,
)
from .activator import SubgraphActivator
from .frame_extractor import extract_frames_for_window, frames_to_b64_urls
from .intent import classify_intent, parse_time_reference
from .serializer import ContextSerializer

# Sport and live content benefits from motion-ranked frame selection
MOTION_RANK_TYPES = {"sport", "live"}


def answer_question(
    question: str,
    graph: VKGraph,
    faiss_index: faiss.Index,
    frame_store: FrameStore,
    llm,
    siglip_encoder,
    question_type: Optional[List[str]] = None,
    time_reference: Optional[str] = None,
    video_path: Optional[str] = None,
    video_type: Optional[str] = None,
    gap_threshold_sec: float = 8.0,
    max_keyframes: int = 8,
    mcq: bool = True,
) -> AnswerResult:
    """Answer a question using the VKG with optional time-reference-aware retrieval.

    If time_reference is given:
      - Use temporal range query (O(log N)) to activate relevant VKG nodes
      - Detect gaps (temporal_precision > gap_threshold) → extract frames on-demand
    Else:
      - FAISS semantic search path (unchanged original behaviour)

    mcq=True: uses MCQ_REASONING_SAMPLING (free-form reasoning output,
      then extracts answer letter via regex). mcq=False: uses QA_SAMPLING
      (free-form, for open-ended questions).
    """
    intents = classify_intent(question)
    activator = SubgraphActivator(graph, faiss_index, siglip_encoder)
    motion_rank = video_type in MOTION_RANK_TYPES

    # ------------------------------------------------------------------
    # Subgraph activation
    # ------------------------------------------------------------------
    t_start: Optional[float] = None
    t_end:   Optional[float] = None
    min_precision = 0.0

    if time_reference:
        parsed = parse_time_reference(time_reference)
        if parsed:
            t_start, t_end = parsed
            subgraph, min_precision = activator.activate_by_time_reference(
                t_start, t_end, intents
            )
        else:
            subgraph = activator.activate(question, intents)
    else:
        subgraph = activator.activate(question, intents)

    # ------------------------------------------------------------------
    # Frame retrieval
    # ------------------------------------------------------------------
    b64_urls: List[str] = []
    keyframe_timestamps: List[float] = []

    if t_start is not None and video_path and min_precision > gap_threshold_sec:
        # Gap detected — extract frames on-demand from the raw video
        live_frames = extract_frames_for_window(
            video_path, t_start, t_end,
            max_frames=max_keyframes,
            motion_rank=motion_rank,
        )
        b64_urls = frames_to_b64_urls(live_frames)
        keyframe_timestamps = [f.timestamp for f in live_frames]
    else:
        # Use pre-sampled keyframes from HDF5 (question-seeded frames prioritised)
        visual_nodes = subgraph.get_visual_nodes()
        # Prioritise question-seeded nodes (id starts with 'qs_')
        visual_nodes.sort(
            key=lambda n: (0 if (n.keyframe_id or "").startswith("qs_") else 1,
                           -n.confidence)
        )
        for n in visual_nodes[:max_keyframes]:
            if n.keyframe_id:
                fi = frame_store.load(n.keyframe_id)
                if fi:
                    try:
                        url = frame_store.get_b64_url(n.keyframe_id)
                        b64_urls.append(url)
                        keyframe_timestamps.append(fi.timestamp)
                    except Exception:
                        pass

        # Fallback: if still no frames and we have a time ref + video, extract
        if not b64_urls and t_start is not None and video_path:
            live_frames = extract_frames_for_window(
                video_path, t_start, t_end,
                max_frames=max_keyframes,
                motion_rank=motion_rank,
            )
            b64_urls = frames_to_b64_urls(live_frames)
            keyframe_timestamps = [f.timestamp for f in live_frames]

    # ------------------------------------------------------------------
    # Context serialization
    # ------------------------------------------------------------------
    context_text = ContextSerializer().serialize(
        subgraph, question, intents,
        question_types=question_type,
        t_start=t_start,
        t_end=t_end,
    )

    # ------------------------------------------------------------------
    # Build multimodal prompt
    # ------------------------------------------------------------------
    content = []
    for url, ts in zip(b64_urls, keyframe_timestamps):
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"[t={ts:.0f}s]"})
    content.append({"type": "text", "text": context_text})

    sampling = MCQ_REASONING_SAMPLING if mcq else QA_SAMPLING
    system_prompt = MCQ_SYSTEM_PROMPT if mcq else (
        "You are a video reasoning assistant. "
        "Study the provided frames and knowledge context carefully. "
        "First, think step-by-step about the evidence from the frames and "
        "the structured knowledge. Then answer the question clearly. "
        "Cite specific timestamps as evidence where relevant."
    )

    # ------------------------------------------------------------------
    # Single vLLM call
    # ------------------------------------------------------------------
    output = llm.chat(
        messages=[[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]],
        sampling_params=sampling,
    )[0]

    raw_answer = output.outputs[0].text.strip()

    # For MCQ: extract answer letter from free-form reasoning output
    final_answer = extract_mcq_answer(raw_answer) if mcq else raw_answer

    return AnswerResult(
        answer=final_answer,
        intents=intents,
        subgraph_size=len(subgraph.nodes),
        keyframes_used=keyframe_timestamps,
        evidence_nodes=[n.label for n in subgraph.get_sorted_events()[:5]],
    )
