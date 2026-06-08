from __future__ import annotations

"""Two-stage QA: LLM planner → deterministic context assembly → LLM answerer.

Stage 1 (Planner): text-only, fast. Given episode summaries + question, outputs
a JSON retrieval plan: which time windows to inspect, what to search for, whether
frames are needed.

Stage 2 (Answerer): code executes the plan — searches graph, extracts frames,
serializes context — then one VLM call answers the question.

No iterative tool calls, no model self-direction at inference time.
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..schema import AnswerResult, VKGraph
from ..vllm_client import (
    MCQ_REASONING_SAMPLING,
    PLANNER_SAMPLING,
    PLANNER_SYSTEM_PROMPT,
    QA_SAMPLING,
    extract_mcq_answer,
)
from .agent import GraphToolExecutor          # reuse search/scene-detail/list helpers
from .frame_extractor import extract_frames_for_window, frames_to_b64_urls
from .intent import parse_time_reference

# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def _save_debug_record(
    debug_dir: str,
    record: Dict[str, Any],
    frame_b64_urls: List[str],
    uid: str,
) -> None:
    """Save one question's full context to disk for analysis."""
    os.makedirs(debug_dir, exist_ok=True)
    frame_dir = os.path.join(debug_dir, "frames", uid)
    os.makedirs(frame_dir, exist_ok=True)

    frame_paths = []
    for i, url in enumerate(frame_b64_urls):
        # url is "data:image/jpeg;base64,<data>"
        try:
            header, b64data = url.split(",", 1)
            ext = "jpg" if "jpeg" in header else "png"
            img_path = os.path.join(frame_dir, f"frame_{i:02d}.{ext}")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64data))
            frame_paths.append(img_path)
        except Exception:
            frame_paths.append(f"<decode error for frame {i}>")

    record["frame_paths"] = frame_paths
    record["n_frames"] = len(frame_b64_urls)

    jsonl_path = os.path.join(debug_dir, "debug_two_stage.jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


_ANSWERER_SYSTEM = (
    "You are a video QA assistant. "
    "Study the provided knowledge graph context and video frames carefully, "
    "then answer the question. "
    "Treat any time window in the question as approximate: the event being asked "
    "about may begin a few seconds before, or complete a few seconds after, the "
    "stated range — weigh frames just outside the window rather than dismissing them. "
    "Take explicit narration and on-screen text literally: if the audio says an action "
    "is performed (e.g. 'boiled', 'fried') prefer that over your own visual guess about "
    "an adjacent step. "
    "For multiple-choice questions, output exactly one letter: A, B, C, or D."
)

MAX_NODES = 40       # max nodes to include in answerer context
MAX_FRAMES = 8       # default max frames (non-visual questions)
VLM_IMAGE_LIMIT = 10 # hard vLLM limit per prompt


# ---------------------------------------------------------------------------
# Stage 1 — Planner
# ---------------------------------------------------------------------------

def run_planner(
    question: str,
    executor: GraphToolExecutor,
    question_type: Optional[List[str]] = None,
    time_reference: Optional[str] = None,
    llm=None,
) -> dict:
    """Run the planner LLM call. Returns the JSON plan dict."""
    episodes_text = executor._list_episodes()

    qt_str = ", ".join(question_type) if question_type else ""
    header_parts = []
    if qt_str:
        header_parts.append(f"[Question type: {qt_str}]")
    if time_reference:
        parsed_tr = parse_time_reference(time_reference)
        if parsed_tr:
            t0_s, t1_s = parsed_tr
            header_parts.append(
                f"[Time reference: {time_reference} (= {int(t0_s)}s–{int(t1_s)}s from video start)]"
            )
        else:
            header_parts.append(f"[Time reference: {time_reference}]")

    user_text = (
        f"{episodes_text}\n\n"
        + ("\n".join(header_parts) + "\n\n" if header_parts else "")
        + f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    output = llm.chat(
        messages=[messages],
        sampling_params=PLANNER_SAMPLING,
        use_tqdm=False,
    )[0]

    raw = output.outputs[0].text.strip()
    try:
        plan = json.loads(raw)
    except Exception:
        # Fallback: safe empty plan
        plan = {"windows": [], "search_queries": [question[:60]], "needs_frames": False}

    return plan


# ---------------------------------------------------------------------------
# Stage 2 — Context assembly + Answerer
# ---------------------------------------------------------------------------

def _assemble_context(
    plan: dict,
    executor: GraphToolExecutor,
    video_path: Optional[str],
    time_reference: Optional[str],
    question: str,
    question_type: Optional[List[str]],
) -> tuple:
    """Returns (content_list, node_count, frame_timestamps)."""

    # --- Adaptive frame budget based on question type ---
    # Visual/entity/event/temporal questions need denser frame sampling so that
    # a specific person's appearance/outfit or the order of events across a window
    # is actually captured (not just a clustered burst at one timestamp).
    _dense_frame_types = {
        "entity recognition", "key information retrieval",
        "event understanding", "temporal grounding",
    }
    qtypes_lower = {qt.strip().lower() for qt in (question_type or [])}
    if qtypes_lower & _dense_frame_types:
        frame_budget = 14   # denser for visual/OCR/event-ordering questions
    else:
        frame_budget = MAX_FRAMES

    # --- Graph nodes ---
    seen_ids = set()
    all_nodes = []

    SKIP_TYPES = {"EpisodeNode", "VideoNode"}

    def _add_nodes_in_range(t_start: float, t_end: float, max_nodes: int = 30) -> None:
        """Fetch graph nodes that overlap [t_start, t_end] and append to all_nodes.
        Uses span-overlap filter so ongoing speech/action nodes that START before
        the window but extend into it are included (e.g. a sentence starting 2s
        before the window and ending inside it)."""
        buf = 5.0
        lo, hi = t_start - buf, t_end + buf
        window_nodes = [
            n for n in executor.graph.nodes.values()
            if (n.t_end >= lo
                and n.t_start <= hi
                and n.node_type not in SKIP_TYPES
                and n.id not in seen_ids)
        ]
        window_nodes.sort(key=lambda n: n.t_start)
        for n in window_nodes[:max_nodes]:
            seen_ids.add(n.id)
            all_nodes.append(n)

    # Fix 2: time_reference → guaranteed graph node context (prepended for priority).
    # For short non-point windows, extend the trailing end by 60s to capture events that
    # START within the window but complete just after (e.g. fish head → placed in dumpling).
    tr_nodes: List = []
    _tr_t0: Optional[float] = None
    _tr_t1: Optional[float] = None
    if time_reference:
        parsed_tr = parse_time_reference(time_reference)
        if parsed_tr:
            t0_tr, t1_tr = parsed_tr
            span = abs(t1_tr - t0_tr)
            if span < 3600.0:  # skip whole-video references
                # Point references (span≈0) mark a specific moment — don't extend.
                # Longer windows (≥120s) already cover enough — don't over-extend.
                tail_ext = 60.0 if 5.0 < span < 120.0 else 0.0
                _tr_t0, _tr_t1 = min(t0_tr, t1_tr), max(t0_tr, t1_tr) + tail_ext
                _old_seen = set(seen_ids)
                _add_nodes_in_range(_tr_t0, _tr_t1)
                tr_nodes = [n for n in all_nodes if n.id not in _old_seen]

    import re as _re_mcq
    _mcq_opts = _re_mcq.findall(r'\([A-D]\)\s*(.+)', question)

    for q in plan.get("search_queries", []):
        text = executor._search_graph(q)
        for line in text.splitlines():
            if "] id=" in line:
                nid = line.split("id=")[1].split()[0]
                node = executor.graph.nodes.get(nid)
                if node and nid not in seen_ids:
                    seen_ids.add(nid)
                    all_nodes.append(node)

    # Fix 7: MCQ option searches — run AFTER plan queries so plan results get
    # insertion-order priority for 0-score cap slots. Ensures all candidate answers
    # have graph coverage (catches rare late-video entities like "Deer meat").
    for opt_text in _mcq_opts[:4]:
        opt_text = opt_text.strip()
        if len(opt_text) > 3:
            text = executor._search_graph(opt_text)
            for line in text.splitlines():
                if "] id=" in line:
                    nid = line.split("id=")[1].split()[0]
                    node = executor.graph.nodes.get(nid)
                    if node and nid not in seen_ids:
                        seen_ids.add(nid)
                        all_nodes.append(node)

    # Fix: Entity-introduction expansion — for each MCQ option, find if it matches
    # a CharacterNode or entity in the graph, then include that entity's earliest
    # appearance (introduction point) plus surrounding context. This ensures that
    # when a question references an entity (e.g. "Sierra's roommate"), the entity's
    # name and role are available even if stated before the time-reference window.
    for opt_text in _mcq_opts[:4]:
        opt_text = opt_text.strip()
        if len(opt_text) < 3:
            continue
        opt_lower = opt_text.lower()
        # Check if option matches any CharacterNode label or entity_id
        for char_node in executor.graph.get_nodes_by_type("CharacterNode"):
            char_label = (char_node.label or "").lower()
            char_desc = (char_node.canonical_description or "").lower()
            char_id = (char_node.entity_id or "").lower()
            if (opt_lower in char_label or opt_lower in char_desc
                    or opt_lower in char_id or char_label in opt_lower):
                # Found a matching entity — get its earliest appearance
                appearances = executor.graph.entity_idx.get(char_node.entity_id, [])
                earliest_node = None
                for aid in appearances:
                    anode = executor.graph.nodes.get(aid)
                    if anode and (earliest_node is None or anode.t_start < earliest_node.t_start):
                        earliest_node = anode
                if earliest_node and earliest_node.id not in seen_ids:
                    _add_nodes_in_range(
                        earliest_node.t_start - 10,
                        earliest_node.t_start + 30,
                        max_nodes=10,
                    )
                # Also include the CharacterNode itself
                if char_node.id not in seen_ids:
                    seen_ids.add(char_node.id)
                    all_nodes.append(char_node)
                break

    for w in plan.get("windows", []):
        _add_nodes_in_range(float(w["t_start"]), float(w["t_end"]))

    # Fix 3: Temporal cluster expansion — find clusters among search results and
    # expand each cluster to capture adjacent speech/action context.
    search_nodes_so_far = [n for n in all_nodes if n not in tr_nodes]
    if search_nodes_so_far:
        sorted_seeds = sorted(search_nodes_so_far[:20], key=lambda n: n.t_start)
        clusters: list = []
        cur: list = []
        for node in sorted_seeds:
            if not cur or node.t_start - cur[-1].t_end < 120:
                cur.append(node)
            else:
                clusters.append(cur)
                cur = [node]
        if cur:
            clusters.append(cur)
        for cluster in clusters[:3]:
            c_start = cluster[0].t_start
            c_end = cluster[-1].t_end
            _add_nodes_in_range(c_start - 30, c_end + 30)

    # Fix 6: Keyword re-ranking — boost nodes whose text overlaps with question keywords
    # before capping. Uses a deduped keyword set so repeated MCQ option words (e.g.
    # "meat" appearing in all options) don't inflate scores for early-video nodes.
    import re as _re
    _stop = {"what", "when", "where", "which", "this", "that", "with", "from", "have",
             "does", "into", "about", "type", "last", "first", "video"}
    _kws = list(dict.fromkeys(  # dedup, preserve order
        w for w in _re.findall(r'\b\w{4,}\b', question.lower()) if w not in _stop
    ))
    if _kws:
        def _kw_score(node) -> float:
            text = (node.label + " " + (getattr(node, "description", "") or "")).lower()
            matched = sum(1 for kw in _kws if kw in text)
            return matched / len(_kws)
        tr_set_ids = {n.id for n in tr_nodes}
        non_tr_raw = [n for n in all_nodes if n.id not in tr_set_ids]
        # Stable-sort by keyword score descending (preserves FAISS order for ties)
        non_tr_ranked = sorted(non_tr_raw, key=_kw_score, reverse=True)
    else:
        tr_set_ids = {n.id for n in tr_nodes}
        non_tr_ranked = [n for n in all_nodes if n.id not in tr_set_ids]

    # When a specific time window was retrieved (tr_nodes), it already contains the
    # primary evidence. Cap non_tr aggressively to reduce cross-video noise from
    # broad keyword matches (e.g. 'fish' matching 50+ nodes in a food documentary).
    if tr_nodes:
        non_tr_cap = min(len(non_tr_ranked), MAX_NODES - len(tr_nodes), 12)
    else:
        non_tr_cap = MAX_NODES
    combined = tr_nodes + non_tr_ranked[:non_tr_cap]

    # Fix 1: Cap BEFORE timestamp sort to preserve relevance order.
    combined = combined[:MAX_NODES]

    # Save search-rank-ordered snapshot for frame auto-extraction (before display sort)
    ranked_search_nodes = combined[len(tr_nodes):]

    all_nodes = combined
    all_nodes.sort(key=lambda n: n.t_start)

    # --- Frames ---
    all_frames = []

    # Always extract from the explicitly provided time_reference (given metadata).
    # Use the same extended window used for tr_nodes so frames cover event completions.
    if time_reference and video_path:
        parsed = parse_time_reference(time_reference)
        if parsed:
            t0, t1 = parsed
            frame_t0 = _tr_t0 if _tr_t0 is not None else min(t0, t1)
            frame_t1 = _tr_t1 if _tr_t1 is not None else max(t0, t1)
            # Scale frame count with window span so a multi-minute window
            # (e.g. "last to appear during the interview") gets evenly-spaced
            # coverage instead of a fixed sparse 6. Reserve most of the budget
            # for the explicitly-referenced window; extract_frames_for_window
            # already spaces them uniformly across [frame_t0, frame_t1].
            tr_span = max(0.0, frame_t1 - frame_t0)
            tr_max = min(frame_budget, max(6, int(tr_span / 12)))
            frames = extract_frames_for_window(video_path, frame_t0, frame_t1, max_frames=tr_max)
            all_frames.extend(frames)

    # Auto-extract from top visual search result nodes.
    # The planner can't know WHERE to extract before search runs; we do it here.
    # De-duplicate by 60-second buckets so we don't pull many frames from the same scene.
    _VISUAL_TYPES = {"ClipNode", "ObjectNode", "StateChangeNode", "SceneNode"}
    if video_path:
        seen_buckets: set = set()
        for node in ranked_search_nodes[:20]:
            if len(all_frames) >= frame_budget:
                break
            if node.node_type not in _VISUAL_TYPES:
                continue
            mid = (node.t_start + node.t_end) / 2
            bucket = int(mid // 60)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            t0 = max(0.0, mid - 5.0)
            t1 = mid + 5.0
            frames = extract_frames_for_window(video_path, t0, t1, max_frames=2)
            all_frames.extend(frames)

    # Also extract from planner-identified windows if frames needed
    if plan.get("needs_frames") and video_path:
        for w in plan.get("windows", []):
            if len(all_frames) >= frame_budget:
                break
            frames = extract_frames_for_window(
                video_path, float(w["t_start"]), float(w["t_end"]),
                max_frames=4,
            )
            all_frames.extend(frames)

    # Cap total frames at budget (may exceed VLM_IMAGE_LIMIT — batching handles it)
    all_frames = all_frames[:frame_budget]
    b64_urls = frames_to_b64_urls(all_frames)
    frame_ts = [f.timestamp for f in all_frames]

    # --- Build context text (shared across all frame batches) ---
    episodes_text = executor._list_episodes()
    qt_str = ", ".join(question_type) if question_type else ""

    node_lines = []
    for n in all_nodes:
        meta_bits = []
        if n.node_type == "SpeechNode":
            for e in executor.graph.get_edges(n.id):
                if e.relation_type == "SPOKEN_BY":
                    ch = executor.graph.nodes.get(e.target_id)
                    if ch:
                        meta_bits.append(f"speaker={ch.label}")
        elif n.node_type == "StateChangeNode" and (n.prev_state or n.next_state):
            meta_bits.append(f"{n.prev_state}→{n.next_state}")
        elif n.node_type == "OCRNode":
            sem = n.metadata.get("semantic_type", "")
            if sem:
                meta_bits.append(f"type={sem}")
        meta_str = f"  ({', '.join(meta_bits)})" if meta_bits else ""
        node_lines.append(
            f"  [{n.t_start:.0f}s–{n.t_end:.0f}s] [{n.node_type}] {n.label}{meta_str}"
        )

    context_parts = [
        f"## Video Structure\n{episodes_text}",
        f"## Retrieved Knowledge\n" + ("\n".join(node_lines) if node_lines else "(none)"),
    ]
    if time_reference:
        context_parts.append(f"## Focus\nTime reference: {time_reference}")
    if qt_str:
        context_parts.append(f"Question type: {qt_str}")
    context_parts.append(f"\nQuestion: {question}")

    context_text = "\n\n".join(context_parts)

    return context_text, len(all_nodes), frame_ts, b64_urls


# ---------------------------------------------------------------------------
# Frame batching — overlapping windows with context propagation
# ---------------------------------------------------------------------------

def _make_frame_batches(
    b64_urls: List[str],
    frame_ts: List[float],
    context_text: str,
    batch_size: int = VLM_IMAGE_LIMIT,
    overlap: int = 2,
) -> List[List[dict]]:
    """Split frames into overlapping batches, each carrying the shared context.

    Each batch is a content list: [image, timestamp, image, timestamp, ..., context_text].
    Overlap ensures temporal continuity — the last `overlap` frames of batch N
    are the first `overlap` frames of batch N+1.
    """
    if len(b64_urls) <= batch_size:
        # Single batch — no splitting needed
        content = []
        for url, ts in zip(b64_urls, frame_ts):
            content.append({"type": "image_url", "image_url": {"url": url}})
            content.append({"type": "text", "text": f"[t={ts:.1f}s]"})
        content.append({"type": "text", "text": context_text})
        return [content]

    batches = []
    step = batch_size - overlap
    for start in range(0, len(b64_urls), step):
        end = min(start + batch_size, len(b64_urls))
        batch_urls = b64_urls[start:end]
        batch_ts = frame_ts[start:end]

        content = []
        for url, ts in zip(batch_urls, batch_ts):
            content.append({"type": "image_url", "image_url": {"url": url}})
            content.append({"type": "text", "text": f"[t={ts:.1f}s]"})
        content.append({"type": "text", "text": context_text})
        batches.append(content)

        if end >= len(b64_urls):
            break
    return batches


def _merge_batch_answers(answers: List[str]) -> str:
    """Merge per-batch MCQ answers via majority vote."""
    from collections import Counter
    if not answers:
        return "ERROR"
    # Filter out ERROR/empty
    valid = [a for a in answers if a and a != "ERROR"]
    if not valid:
        return "ERROR"
    return Counter(valid).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def two_stage_answer_question(
    question: str,
    graph: VKGraph,
    faiss_index: Optional[faiss.Index],
    llm,
    siglip_encoder,
    video_path: Optional[str] = None,
    question_type: Optional[List[str]] = None,
    time_reference: Optional[str] = None,
    mcq: bool = True,
    debug_dir: Optional[str] = None,
    uid: str = "unknown",
) -> AnswerResult:
    """Two-stage answer: planner call → context assembly → batched answerer call.

    When > VLM_IMAGE_LIMIT frames are retrieved, they are split into overlapping
    batches. Each batch carries the full graph context + the previous batch's
    answer as continuity signal. Final answer via majority vote.
    """

    executor = GraphToolExecutor(graph, faiss_index, siglip_encoder, video_path)

    # Stage 1: planner
    plan = run_planner(
        question=question,
        executor=executor,
        question_type=question_type,
        time_reference=time_reference,
        llm=llm,
    )

    # Stage 2: assemble context (returns context_text + all frames uncapped)
    context_text, node_count, frame_ts, b64_urls = _assemble_context(
        plan=plan,
        executor=executor,
        video_path=video_path,
        time_reference=time_reference,
        question=question,
        question_type=question_type,
    )

    # Stage 3: frame batching with context propagation
    batches = _make_frame_batches(b64_urls, frame_ts, context_text)
    sampling = MCQ_REASONING_SAMPLING if mcq else QA_SAMPLING
    batch_answers: List[str] = []
    batch_raws: List[str] = []

    prev_answer_hint = ""
    for batch_idx, batch_content in enumerate(batches):
        # Prepend previous batch's answer as continuity signal
        if prev_answer_hint:
            # Insert before the context text block
            hint_text = (
                f"## Previous batch analysis\n"
                f"{prev_answer_hint}\n\n"
                f"Consider the above when answering. "
                f"If the evidence in the current frames conflicts, prefer the current frames."
            )
            # Find the context text entry and prepend hint
            for i, item in enumerate(batch_content):
                if item.get("type") == "text" and "Question:" in item.get("text", ""):
                    batch_content[i] = {"type": "text", "text": hint_text + "\n\n" + item["text"]}
                    break

        messages = [
            {"role": "system", "content": _ANSWERER_SYSTEM},
            {"role": "user",   "content": batch_content},
        ]

        output = llm.chat(
            messages=[messages],
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )[0]

        raw = output.outputs[0].text.strip()
        answer = extract_mcq_answer(raw) if mcq else raw
        batch_answers.append(answer)
        batch_raws.append(raw)
        prev_answer_hint = raw

    # Merge batch answers
    final_answer = _merge_batch_answers(batch_answers)

    # Debug logging
    if debug_dir:
        _save_debug_record(
            debug_dir=debug_dir,
            uid=uid,
            frame_b64_urls=b64_urls,
            record={
                "uid": uid,
                "question": question,
                "question_type": question_type,
                "time_reference": time_reference,
                "plan": plan,
                "n_nodes": node_count,
                "frame_timestamps": frame_ts,
                "context_text": context_text,
                "answerer_system": _ANSWERER_SYSTEM,
                "answerer_raw_output": "\n---BATCH---\n".join(batch_raws),
                "predicted_answer": final_answer,
                "batch_answers": batch_answers,
                "n_batches": len(batches),
            },
        )

    return AnswerResult(
        answer=final_answer,
        intents=[plan.get("reasoning", "")],
        subgraph_size=node_count,
        keyframes_used=frame_ts,
        evidence_nodes=[],
    )


# ---------------------------------------------------------------------------
# Batched public API — one planner call + one answerer call for all questions
# ---------------------------------------------------------------------------

def batch_two_stage_answer_questions(
    questions: List[dict],
    graph: VKGraph,
    faiss_index,
    llm,
    siglip_encoder,
    video_path: Optional[str] = None,
    mcq: bool = True,
    debug_dir: Optional[str] = None,
) -> List[AnswerResult]:
    """Batch two-stage: 1 planner LLM call + frame-batched answerer calls.

    For each question:
      - Frames are split into overlapping batches of VLM_IMAGE_LIMIT
      - Each batch carries the full graph context + previous batch's answer
      - Final answer via majority vote across batches
    """
    executor = GraphToolExecutor(graph, faiss_index, siglip_encoder, video_path)
    episodes_text = executor._list_episodes()

    # --- Stage 1: batch planner (all questions in one llm.chat call) ---
    planner_msgs = []
    for q in questions:
        qt_str = ", ".join(q.get("question_type") or [])
        header_parts = []
        if qt_str:
            header_parts.append(f"[Question type: {qt_str}]")
        if q.get("time_reference"):
            tr_p = parse_time_reference(q["time_reference"])
            if tr_p:
                header_parts.append(
                    f"[Time reference: {q['time_reference']} (= {int(tr_p[0])}s–{int(tr_p[1])}s from video start)]"
                )
            else:
                header_parts.append(f"[Time reference: {q['time_reference']}]")
        user_text = (
            episodes_text + "\n\n"
            + ("\n".join(header_parts) + "\n\n" if header_parts else "")
            + f"Question: {q['question']}"
        )
        planner_msgs.append([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ])

    planner_outputs = llm.chat(
        messages=planner_msgs,
        sampling_params=PLANNER_SAMPLING,
        use_tqdm=False,
    )

    plans = []
    for idx, o in enumerate(planner_outputs):
        try:
            plans.append(json.loads(o.outputs[0].text.strip()))
        except Exception:
            plans.append({
                "windows": [],
                "search_queries": [questions[idx]["question"][:60]],
                "needs_frames": False,
                "reasoning": "",
            })

    # --- Stage 2: context assembly + frame batching (CPU/IO) ---
    all_context_texts: List[str] = []
    all_node_counts: List[int] = []
    all_frame_ts: List = []
    all_b64_urls: List = []
    all_batches: List[List[List[dict]]] = []  # per-question list of batches

    for q, plan in zip(questions, plans):
        context_text, node_count, frame_ts, b64_urls = _assemble_context(
            plan=plan,
            executor=executor,
            video_path=video_path,
            time_reference=q.get("time_reference"),
            question=q["question"],
            question_type=q.get("question_type"),
        )
        batches = _make_frame_batches(b64_urls, frame_ts, context_text)
        all_context_texts.append(context_text)
        all_node_counts.append(node_count)
        all_frame_ts.append(frame_ts)
        all_b64_urls.append(b64_urls)
        all_batches.append(batches)

    # --- Stage 3: batched answerer (all prompts in one llm.chat call per pass) ---
    sampling = MCQ_REASONING_SAMPLING if mcq else QA_SAMPLING
    results: List[AnswerResult] = []

    # Organize prompts by batch index: batch_groups[bi] = [(qi, messages), ...]
    max_batches = max(len(batches) for batches in all_batches) if all_batches else 0
    batch_groups: List[List[tuple]] = [[] for _ in range(max_batches)]

    for i, batches in enumerate(all_batches):
        for bi, batch_content in enumerate(batches):
            messages = [
                {"role": "system", "content": _ANSWERER_SYSTEM},
                {"role": "user",   "content": batch_content},
            ]
            batch_groups[bi].append((i, messages))

    # Process each batch group in one llm.chat() call
    question_answers: Dict[int, List[tuple]] = {}  # qi -> [(batch_idx, answer, raw)]

    for bi, group in enumerate(batch_groups):
        if not group:
            continue

        # For batch_idx > 0, prepend previous batch's answer as hint
        if bi > 0:
            for pi, qi in enumerate(group):
                qi_val = group[pi][0]
                prev_raw = None
                for prev_bi, _, raw in question_answers.get(qi_val, []):
                    if prev_bi == bi - 1:
                        prev_raw = raw
                        break
                if prev_raw:
                    hint_text = (
                        f"## Previous batch analysis\n"
                        f"{prev_raw}\n\n"
                        f"Consider the above when answering. "
                        f"If the evidence in the current frames conflicts, prefer the current frames."
                    )
                    content = group[pi][1][1]["content"]  # user content
                    for ci, item in enumerate(content):
                        if item.get("type") == "text" and "Question:" in item.get("text", ""):
                            content[ci] = {"type": "text", "text": hint_text + "\n\n" + item["text"]}
                            break

        prompts = [msgs for _, msgs in group]
        qi_indices = [qi for qi, _ in group]

        outputs = llm.chat(
            messages=prompts,
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )

        for qi, output in zip(qi_indices, outputs):
            raw = output.outputs[0].text.strip()
            answer = extract_mcq_answer(raw) if mcq else raw
            if qi not in question_answers:
                question_answers[qi] = []
            question_answers[qi].append((bi, answer, raw))

    # Merge per-question and build results
    for i, (q, plan, batches) in enumerate(zip(questions, plans, all_batches)):
        qa_list = sorted(question_answers.get(i, []), key=lambda x: x[0])
        batch_answers = [ans for _, ans, _ in qa_list]
        batch_raws = [raw for _, _, raw in qa_list]
        final_answer = _merge_batch_answers(batch_answers)

        if debug_dir:
            _save_debug_record(
                debug_dir=debug_dir,
                uid=q.get("uid", str(i)),
                frame_b64_urls=all_b64_urls[i],
                record={
                    "uid":             q.get("uid", str(i)),
                    "question":        q["question"],
                    "question_type":   q.get("question_type"),
                    "time_reference":  q.get("time_reference"),
                    "plan":            plan,
                    "n_nodes":         all_node_counts[i],
                    "frame_timestamps": all_frame_ts[i],
                    "context_text":    all_context_texts[i],
                    "answerer_system": _ANSWERER_SYSTEM,
                    "answerer_raw_output": "\n---BATCH---\n".join(batch_raws),
                    "predicted_answer": final_answer,
                    "batch_answers":   batch_answers,
                    "n_batches":       len(batches),
                },
            )

        results.append(AnswerResult(
            answer=final_answer,
            intents=[plan.get("reasoning", "")],
            subgraph_size=all_node_counts[i],
            keyframes_used=all_frame_ts[i],
            evidence_nodes=[],
        ))

    return results
