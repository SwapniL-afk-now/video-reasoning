from __future__ import annotations

"""QA engine: single-question and batched multi-question inference via vLLM."""

from typing import List, Optional

import faiss
import numpy as np
from collections import Counter

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

MOTION_RANK_TYPES = {"sport", "live"}

_QA_SYSTEM = (
    "You are a video reasoning assistant. "
    "Study the provided frames and knowledge context carefully. "
    "Think step-by-step about the evidence, then answer clearly. "
    "Cite specific timestamps where relevant."
)


def _build_prompt(
    question: str,
    graph: VKGraph,
    faiss_index: faiss.Index,
    frame_store: FrameStore,
    siglip_encoder,
    question_type: Optional[List[str]] = None,
    time_reference: Optional[str] = None,
    video_path: Optional[str] = None,
    video_type: Optional[str] = None,
    gap_threshold_sec: float = 8.0,
    max_keyframes: int = 8,
    mcq: bool = True,
) -> tuple:
    """Build multimodal prompt for one question.

    Returns (messages_list, sampling_params, intents, keyframe_timestamps).
    No LLM call is made — caller batches these and fires one llm.chat().
    """
    intents = classify_intent(
        question,
        siglip_encoder=siglip_encoder,
        question_types=question_type,
    )
    activator = SubgraphActivator(graph, faiss_index, siglip_encoder)
    motion_rank = video_type in MOTION_RANK_TYPES

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

    # Frame retrieval
    b64_urls: List[str] = []
    keyframe_timestamps: List[float] = []

    # Frame selection strategy:
    #   ≤90s  → force_live: re-extract exact frames from the raw video so
    #            point-in-time and short-span questions always see the right moment.
    #   >90s  → use VKG stored keyframes, but sampled in temporal buckets so the
    #            model gets a uniform "film strip" across the full span rather than
    #            8 confidence-ranked frames that may all cluster at one moment.
    window_sec = (t_end - t_start) if (t_start is not None and t_end is not None) else None
    force_live = (window_sec is not None and window_sec <= 90.0 and video_path)

    if t_start is not None and video_path and (min_precision > gap_threshold_sec or force_live):
        live_frames = extract_frames_for_window(
            video_path, t_start, t_end,
            max_frames=max_keyframes,
            motion_rank=motion_rank,
        )
        b64_urls = frames_to_b64_urls(live_frames)
        keyframe_timestamps = [f.timestamp for f in live_frames]
    else:
        visual_nodes = subgraph.get_visual_nodes()

        if window_sec and window_sec > 90.0 and t_start is not None and len(visual_nodes) > max_keyframes:
            # Wide span: divide the query window into max_keyframes evenly-spaced
            # buckets and pick the highest-confidence frame from each bucket.
            # This ensures the model sees the full temporal span, not just a
            # confidence-clustered subset of it.
            bucket_w = window_sec / max_keyframes
            selected: List = []
            for i in range(max_keyframes):
                lo = t_start + i * bucket_w
                hi = t_start + (i + 1) * bucket_w
                bucket = [n for n in visual_nodes if lo <= n.t_start < hi and n.keyframe_id]
                if bucket:
                    selected.append(max(bucket, key=lambda n: n.confidence))
            # Fill any empty buckets with qs_live nodes (pre-sampled question frames)
            if len(selected) < max_keyframes:
                qs = [n for n in visual_nodes
                      if (n.keyframe_id or "").startswith("qs_") and n not in selected]
                selected.extend(qs[:max_keyframes - len(selected)])
            visual_nodes = selected
        else:
            visual_nodes.sort(
                key=lambda n: (0 if (n.keyframe_id or "").startswith("qs_") else 1,
                               -n.confidence)
            )
            visual_nodes = visual_nodes[:max_keyframes]

        for n in visual_nodes:
            if n.keyframe_id:
                fi = frame_store.load(n.keyframe_id)
                if fi:
                    try:
                        url = frame_store.get_b64_url(n.keyframe_id)
                        b64_urls.append(url)
                        keyframe_timestamps.append(fi.timestamp)
                    except Exception:
                        pass

        if not b64_urls and t_start is not None and video_path:
            live_frames = extract_frames_for_window(
                video_path, t_start, t_end,
                max_frames=max_keyframes,
                motion_rank=motion_rank,
            )
            b64_urls = frames_to_b64_urls(live_frames)
            keyframe_timestamps = [f.timestamp for f in live_frames]

    context_text = ContextSerializer().serialize(
        subgraph, question, intents,
        question_types=question_type,
        t_start=t_start,
        t_end=t_end,
        visual_provided=bool(b64_urls),
    )

    content = []
    for url, ts in zip(b64_urls, keyframe_timestamps):
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"[t={ts:.0f}s]"})
    content.append({"type": "text", "text": context_text})

    sampling = MCQ_REASONING_SAMPLING if mcq else QA_SAMPLING
    system_prompt = MCQ_SYSTEM_PROMPT if mcq else _QA_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content},
    ]

    meta = {
        "intents":    intents,
        "kf_ts":      keyframe_timestamps,
        "subgraph_sz": len(subgraph.nodes),
        "evidence":   [n.label for n in subgraph.get_sorted_events()[:5]],
        "mcq":        mcq,
    }
    return messages, sampling, meta


def batch_answer_questions(
    questions: List[dict],
    graph: VKGraph,
    faiss_index: faiss.Index,
    frame_store: FrameStore,
    llm,
    siglip_encoder,
    video_path: Optional[str] = None,
    video_type: Optional[str] = None,
    gap_threshold_sec: float = 8.0,
    max_keyframes: int = 8,
    mcq: bool = True,
) -> List[AnswerResult]:
    """Build prompts for all questions, fire ONE batched llm.chat() call.

    Each element of `questions` is a dict with keys:
      question, question_type (list), time_reference (str|None)
    """
    all_messages = []
    all_meta = []

    for q in questions:
        msgs, sampling, meta = _build_prompt(
            question=q["question"],
            graph=graph,
            faiss_index=faiss_index,
            frame_store=frame_store,
            siglip_encoder=siglip_encoder,
            question_type=q.get("question_type"),
            time_reference=q.get("time_reference"),
            video_path=video_path,
            video_type=video_type,
            gap_threshold_sec=gap_threshold_sec,
            max_keyframes=max_keyframes,
            mcq=mcq,
        )
        all_messages.append(msgs)
        all_meta.append(meta)

    # Single batched GPU call — vLLM parallelises across all prompts
    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling,          # same params for all
        chat_template_kwargs={"enable_thinking": mcq},
        use_tqdm=True,
    )

    results = []
    for meta, out in zip(all_meta, outputs):
        raw = out.outputs[0].text.strip()
        answer = extract_mcq_answer(raw) if meta["mcq"] else raw
        results.append(AnswerResult(
            answer=answer,
            intents=meta["intents"],
            subgraph_size=meta["subgraph_sz"],
            keyframes_used=meta["kf_ts"],
            evidence_nodes=meta["evidence"],
        ))
    return results


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
    """Single-question wrapper around batch_answer_questions."""
    results = batch_answer_questions(
        questions=[{
            "question":      question,
            "question_type": question_type,
            "time_reference": time_reference,
        }],
        graph=graph,
        faiss_index=faiss_index,
        frame_store=frame_store,
        llm=llm,
        siglip_encoder=siglip_encoder,
        video_path=video_path,
        video_type=video_type,
        gap_threshold_sec=gap_threshold_sec,
        max_keyframes=max_keyframes,
        mcq=mcq,
    )
    return results[0]


def answer_question_majority(
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
    n_votes: int = 3,
    **kwargs,
) -> AnswerResult:
    """Run answer_question n_votes times, return majority answer."""
    votes = []
    last = None
    for _ in range(n_votes):
        last = answer_question(
            question=question,
            graph=graph,
            faiss_index=faiss_index,
            frame_store=frame_store,
            llm=llm,
            siglip_encoder=siglip_encoder,
            question_type=question_type,
            time_reference=time_reference,
            video_path=video_path,
            video_type=video_type,
            **kwargs,
        )
        votes.append(last.answer)
    majority = Counter(votes).most_common(1)[0][0]
    return AnswerResult(
        answer=majority,
        intents=last.intents,
        subgraph_size=last.subgraph_size,
        keyframes_used=last.keyframes_used,
        evidence_nodes=last.evidence_nodes,
    )
