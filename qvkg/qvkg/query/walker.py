from __future__ import annotations

"""Inference-time agentic graph traversal — the batched-by-hop walker.

A training-free reasoning loop: the controller LLM picks one typed action over a
visible sub-graph, a deterministic executor applies it, and a coverage–elasticity
warrant decides when to stop. The loop is iterative *per question* but
synchronous *across questions*: each hop fires at most three batched ``llm.chat``
calls (1 controller + 2 answerer read-outs) for the whole question set.

See AGENTIC_TRAVERSAL.md for the design rationale.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from ..schema import AnswerResult, VKGEdge, VKGNode, VKGraph
from ..vllm_client import (
    BUILD_CAUSAL_SAMPLING,
    BUILD_CAUSAL_SYSTEM,
    OFFLINE_SAMPLING,
    WALKER_ANSWER_SAMPLING,
    WALKER_ANSWER_SYSTEM,
    WALKER_CONTROLLER_SYSTEM,
    WALKER_PROBE_SAMPLING,
    build_scene_system_prompt,
    extract_mcq_answer,
    walker_controller_sampling,
)
from . import graph_ops
from . import warrant as warrant_mod
from .actions import (
    MAX_IMAGE_FRAMES_PER_PROMPT,
    ACTION_SCHEMA,
    WalkState,
    execute_action,
)
from .activator import SubgraphActivator
from .frame_extractor import extract_frames_for_window, frames_to_b64_urls
from .intent import classify_intent, parse_time_reference
from .serializer import ContextSerializer
from .verifier import verify

MOTION_RANK_TYPES = {"sport", "live"}

_SERIALIZER = ContextSerializer()
_CONTROLLER_SAMPLING = walker_controller_sampling(ACTION_SCHEMA)


# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

_OPTION_RE = re.compile(r"\(([A-D])\)\s*([^()]*?)(?=\s*\([A-D]\)|$)", re.DOTALL)


def parse_options(question: str) -> Dict[str, str]:
    """Extract ``{"A": text, ...}`` from an MCQ question string."""
    opts: Dict[str, str] = {}
    for m in _OPTION_RE.finditer(question):
        opts[m.group(1)] = " ".join(m.group(2).split())
    return opts


# ---------------------------------------------------------------------------
# Seeding S_0 (deterministic — no LLM call)
# ---------------------------------------------------------------------------

def _resolve_entities(graph: VKGraph, texts: List[str]) -> Set[str]:
    """Fuzzy-match stem/option strings to CharacterNode appearances."""
    seeds: Set[str] = set()
    chars = graph.get_nodes_by_type("CharacterNode")
    for raw in texts:
        low = raw.lower()
        toks = {w for w in re.findall(r"\b\w{4,}\b", low)}
        if not toks:
            continue
        for ch in chars:
            label = (ch.label or "").lower()
            desc = (ch.canonical_description or "").lower()
            cid = (ch.entity_id or "").lower()
            desc_toks = {w for w in re.findall(r"\b\w{4,}\b", desc)}
            if (label and (label in low or low in label)) or \
               (cid and cid in low) or \
               (desc_toks and (toks & desc_toks)):
                if ch.entity_id:
                    seeds |= set(graph.entity_idx.get(ch.entity_id, []))
                seeds.add(ch.id)
    return seeds


def seed_state(
    q: dict,
    graph: VKGraph,
    faiss_index,
    siglip_encoder,
    frame_store,
    video_path: Optional[str],
    mcq: bool,
    seed_faiss_k: int = 12,
) -> WalkState:
    """Build S_0 using the same SubgraphActivator pipeline as one-shot.

    Ports the full power of ``qa._build_prompt`` — intent-driven FAISS seeding,
    typed BFS expansion (temporal spine, causal chains, entity threads, etc.),
    temporal bucketing for wide windows, and confidence-ranked frame selection.
    The result is an S_0 as rich as what one-shot sees in a single pass.
    """
    question = q["question"]
    options = parse_options(question) if mcq else {}
    qtype = q.get("question_type") or []
    tr = parse_time_reference(q["time_reference"]) if q.get("time_reference") else None

    intents = classify_intent(
        question, siglip_encoder=siglip_encoder, question_types=qtype)
    activator = SubgraphActivator(graph, faiss_index, siglip_encoder, max_nodes=80)

    if tr is not None:
        t0, t1 = tr
        # Temporal-range activation — does not require FAISS.
        subgraph, _min_prec = activator.activate_by_time_reference(t0, t1, intents)
    elif faiss_index is not None:
        subgraph = activator.activate(question, intents)
    else:
        # No time reference and no FAISS index — activate() would dereference a
        # None index. Seed from episodes / first nodes instead.
        eps = graph.get_episodes()
        seed_ids = {e.id for e in eps[:6]} or \
                   {n.id for n in list(graph.nodes.values())[:20]}
        subgraph = graph_ops.build_subgraph(graph, seed_ids)

    node_ids = set(subgraph.nodes.keys())
    # Fallback: if still empty, grab first nodes.
    if not node_ids:
        eps = graph.get_episodes()
        node_ids = {e.id for e in eps[:6]} or \
                   {n.id for n in list(graph.nodes.values())[:10]}

    state = WalkState(
        qid=q.get("uid", question[:24]),
        question=question,
        options=options,
        qtype=qtype,
        time_reference=tr,
        node_ids=set(node_ids),
        frontier=set(node_ids),
    )

    # ---- Frame selection (ported from qa._build_prompt) ----
    gap_threshold_sec = 8.0
    max_keyframes = 8
    motion_rank = any(vt in question.lower() for vt in ("sport", "live"))
    t_start, t_end = tr if tr else (None, None)

    window_sec = (t_end - t_start) if (t_start is not None and t_end is not None) else None
    force_live = (window_sec is not None and window_sec <= 90.0 and video_path)

    if t_start is not None and video_path and (window_sec is None or window_sec <= gap_threshold_sec or force_live):
        live_frames = extract_frames_for_window(
            video_path, t_start, t_end,
            max_frames=max_keyframes, motion_rank=motion_rank)
        urls_ts = list(zip(frames_to_b64_urls(live_frames),
                           [f.timestamp for f in live_frames]))
        state.add_frames(urls_ts)
    else:
        visual_nodes = subgraph.get_visual_nodes()

        if window_sec and window_sec > 90.0 and t_start is not None and len(visual_nodes) > max_keyframes:
            bucket_w = window_sec / max_keyframes
            selected = []
            for i in range(max_keyframes):
                lo = t_start + i * bucket_w
                hi = t_start + (i + 1) * bucket_w
                bucket = [n for n in visual_nodes if lo <= n.t_start < hi and n.keyframe_id]
                if bucket:
                    selected.append(max(bucket, key=lambda n: n.confidence))
            if len(selected) < max_keyframes:
                qs = [n for n in visual_nodes
                      if (n.keyframe_id or "").startswith("qs_") and n not in selected]
                selected.extend(qs[:max_keyframes - len(selected)])
            visual_nodes = selected[:max_keyframes]
        else:
            visual_nodes.sort(key=lambda n: -n.confidence)
            visual_nodes = visual_nodes[:max_keyframes]

        urls_ts = []
        for n in visual_nodes:
            if n.keyframe_id and frame_store is not None:
                try:
                    url = frame_store.get_b64_url(n.keyframe_id)
                    urls_ts.append((url, n.t_start))
                except Exception:
                    pass
        state.add_frames(urls_ts)

        if not state.frames and t_start is not None and video_path:
            live_frames = extract_frames_for_window(
                video_path, t_start, t_end,
                max_frames=max_keyframes, motion_rank=motion_rank)
            urls_ts = list(zip(frames_to_b64_urls(live_frames),
                               [f.timestamp for f in live_frames]))
            state.add_frames(urls_ts)

    return state


# ---------------------------------------------------------------------------
# Frame batching helpers
# ---------------------------------------------------------------------------

def frame_to_text_summary(graph: VKGraph, timestamp: float,
                          window_sec: float = 3.0) -> str:
    """Build a text description of VKG content near a frame's timestamp.

    Used by :meth:`WalkState.archive_old_frames` to convert image frames
    that exceed the per-prompt image limit into persistent text summaries.
    """
    nodes = graph.get_nodes_in_window(timestamp - window_sec,
                                      timestamp + window_sec)
    texts: List[str] = []
    for n in nodes:
        if n.label and len(n.label) > 3:
            texts.append(n.label)
        if n.node_type == "OCRNode" and getattr(n, 'text', None):
            texts.append(f"OCR: {n.text[:120]}")
        if n.node_type == "ASRNode" and getattr(n, 'text', None):
            texts.append(f"ASR: {n.text[:120]}")
        if n.canonical_description:
            texts.append(n.canonical_description[:200])
    unique = list(dict.fromkeys(texts))
    return " | ".join(unique)[:600] if unique else "(no caption)"


# ---------------------------------------------------------------------------
# BUILD action — online KG construction
# ---------------------------------------------------------------------------

def _build_anchor(state: WalkState, graph: VKGraph, faiss_index,
                  siglip_encoder) -> float:
    """Find the best timestamp anchor for BUILD.

    Priority: FAISS question search → CSV time_reference → frontier midpoint
    → all-node midpoint → 0.
    """
    if faiss_index is not None:
        from . import graph_ops
        hits = graph_ops.faiss_search(graph, faiss_index, siglip_encoder,
                                       state.question, k=3)
        if hits:
            ts = [graph.nodes[n].t_start for n in hits
                  if n in graph.nodes and graph.nodes[n].t_start is not None]
            if ts:
                return sum(ts) / len(ts)
    if state.time_reference:
        return (state.time_reference[0] + state.time_reference[1]) / 2
    frontier_ts = [graph.nodes[n].t_start for n in state.frontier
                   if n in graph.nodes and graph.nodes[n].t_start is not None]
    if frontier_ts:
        return (min(frontier_ts) + max(frontier_ts)) / 2
    all_ts = [graph.nodes[n].t_start for n in state.node_ids
              if n in graph.nodes and graph.nodes[n].t_start is not None]
    return sum(all_ts) / len(all_ts) if all_ts else 0.0


def _build_params(gap_slot: str, multiplier: int) -> tuple:
    """Return (radius_sec, n_frames) for a given gap slot and multiplier."""
    params = {
        "causal_edge":         (30.0, 8),
        "entity_or_frame":     (10.0, 12),
        "temporal_occurrences": (60.0, 6),
        "mcq_option_coverage": (15.0, 8),
        "episode_summary":     (90.0, 6),
    }
    radius, n_frames = params.get(gap_slot, (30.0, 8))
    return radius * multiplier, min(n_frames * multiplier, 32)


def _first_unfilled_gap_slot(state: WalkState) -> str:
    if state.warrant and state.warrant.gap_slots:
        return state.warrant.gap_slots[0]
    return "entity_or_frame"


_EVENT_TYPES_FOR_CAUSAL = {
    "ActionNode", "InteractionNode", "StateChangeNode",
    "SpeechNode", "SceneNode", "AudioEventNode",
}


def _build_causal_edges(state: WalkState, graph: VKGraph,
                        llm, t0: float, t1: float) -> None:
    """Infer missing causal edges among events in [t0, t1] via the LLM."""
    nodes = [n for n in graph.get_nodes_in_window(t0, t1, buffer_sec=5.0)
             if n.node_type in _EVENT_TYPES_FOR_CAUSAL]
    nodes.sort(key=lambda n: n.t_start)
    if len(nodes) < 2:
        return

    timeline = "\n".join(
        f"  [{n.t_start:.0f}s] [{n.node_type}] "
        f"{n.label or '(unnamed)'}"
        f"{' — ' + n.canonical_description[:100] if n.canonical_description else ''}"
        for n in nodes[:40]
    )

    user_msg = (
        f"Question: {state.question}\n\n"
        f"Video events at [{t0:.0f}s–{t1:.0f}s]:\n{timeline}\n\n"
        "Identify causal cause→effect relationships between these events "
        "that are relevant to answering the question above. "
        "Use the exact event descriptions as cause/effect values."
    )

    out = llm.chat(
        messages=[[
            {"role": "system", "content": BUILD_CAUSAL_SYSTEM},
            {"role": "user", "content": user_msg},
        ]],
        sampling_params=BUILD_CAUSAL_SAMPLING,
        use_tqdm=False,
    )

    import json
    try:
        text = out[0].outputs[0].text
        links = json.loads(text)
    except Exception:
        return

    if not isinstance(links, list):
        return

    added = set()
    touched: Set[str] = set()
    for link in links:
        if link.get("confidence", 0) < 0.6:
            continue
        cause_desc = (link.get("cause") or "").lower()
        effect_desc = (link.get("effect") or "").lower()
        cause_node = _match_event_description(nodes, cause_desc)
        effect_node = _match_event_description(nodes, effect_desc)
        if cause_node and effect_node and cause_node.id != effect_node.id:
            key = (cause_node.id, effect_node.id, link.get("relation_type", "CAUSES"))
            if key not in added:
                graph.add_edge(VKGEdge(
                    source_id=cause_node.id,
                    target_id=effect_node.id,
                    relation_type=link.get("relation_type", "CAUSES"),
                    weight=link.get("confidence", 1.0),
                    confidence=link.get("confidence", 1.0),
                    metadata={
                        "reasoning": link.get("reasoning", ""),
                        "source": "build",
                        "build_hop": state.hop,
                    },
                ))
                added.add(key)
                touched.add(cause_node.id)
                touched.add(effect_node.id)

    # The new edge is only visible to the induced subgraph (warrant, answerer,
    # serializer) if BOTH endpoints are in the working set. Absorb them so the
    # causal_edge slot can actually be filled and the elasticity probe can test
    # whether they matter.
    if touched:
        state.absorb(touched)


def _match_event_description(nodes: list, desc: str) -> Optional[VKGNode]:
    """Fuzzy-match a description string to a node label/description."""
    if not desc:
        return None
    desc_lower = desc.lower()
    for n in nodes:
        label = (n.label or "").lower().strip()
        canon = (n.canonical_description or "").lower().strip()
        # Guard against empty strings: `"" in desc_lower` is always True and
        # would match the first node for every description, collapsing cause
        # and effect onto the same node so no edge is ever built.
        if label and (desc_lower in label or label in desc_lower):
            return n
        if canon and (desc_lower in canon or canon in desc_lower):
            return n
    # Word-level fallback
    desc_words = set(w for w in re.findall(r"\b\w{4,}\b", desc_lower))
    if not desc_words:
        return None
    best, best_score = None, 0
    for n in nodes:
        label = (n.label or "").lower()
        canon = (n.canonical_description or "").lower()
        text = label + " " + canon
        words = set(w for w in re.findall(r"\b\w{4,}\b", text))
        overlap = len(desc_words & words)
        if overlap > best_score:
            best_score = overlap
            best = n
    return best if best_score >= 2 else None


def _perceive_window(state: WalkState, graph: VKGraph, video_path: str,
                     llm, t0: float, t1: float, n_frames: int,
                     video_type: str = "") -> int:
    """Interactively perceive a raw-video window into NEW graph nodes.

    Extracts frames from the raw video, runs the scene-extraction VLM on them,
    and mints OCRNode / CharacterNode / ActionNode / StateChangeNode nodes —
    then absorbs them so the warrant slot (entity_or_frame / mcq) can actually be
    filled. This is the step that turns "look at the video" into queryable,
    storable knowledge rather than transient pixels. Returns #nodes minted.
    """
    if not video_path:
        return 0
    frames = extract_frames_for_window(video_path, t0, t1,
                                       max_frames=min(n_frames, 10),
                                       motion_rank=True)
    if not frames:
        return 0
    urls = frames_to_b64_urls(frames)
    state.add_frames(list(zip(urls, [f.timestamp for f in frames])))

    content: List[dict] = []
    for url, fr in zip(urls, frames):
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"[t={fr.timestamp:.1f}s]"})
    content.append({"type": "text",
                    "text": "Extract structured information from these frames as JSON."})

    try:
        out = llm.chat(
            messages=[[
                {"role": "system", "content": build_scene_system_prompt(video_type)},
                {"role": "user", "content": content},
            ]],
            sampling_params=OFFLINE_SAMPLING,
            use_tqdm=False,
        )
        data = json.loads(out[0].outputs[0].text)
    except Exception:
        return 0

    mid = (t0 + t1) / 2.0
    minted: Set[str] = set()
    idx = len(graph.nodes)

    def _mint(kind: str, label: str, ts: float, **extra) -> None:
        nonlocal idx
        label = (label or "").strip()
        if not label:
            return
        nid = f"build_h{state.hop}_{kind}_{idx}"
        idx += 1
        graph.add_node(VKGNode(
            id=nid, node_type=kind, label=label[:200], level=0,
            t_start=ts, t_end=ts,
            metadata={"source": "build_perceive", "build_hop": state.hop, **extra},
        ))
        minted.add(nid)

    for o in data.get("ocr_text", []) or []:
        _mint("OCRNode", o.get("text", ""), mid)
    for c in data.get("characters", []) or []:
        _mint("CharacterNode", c.get("description", ""), mid,
              emotion=c.get("emotion", ""))
    for a in data.get("actions", []) or []:
        ts = float(a.get("approx_timestamp", mid) or mid)
        _mint("ActionNode", a.get("description", ""), ts,
              actor=a.get("actor", ""))
    for sc in data.get("state_changes", []) or []:
        _mint("StateChangeNode",
              f"{sc.get('entity','')}: {sc.get('from_state','')}→{sc.get('to_state','')}",
              float(sc.get("approx_timestamp", mid) or mid))

    if minted:
        state.absorb(minted)
    return len(minted)


def _execute_build(state: WalkState, graph: VKGraph, faiss_index,
                   siglip_encoder, video_path: str, llm, frame_store,
                   video_type: str = "") -> None:
    """Execute BUILD action — online KG construction for missing evidence.

    Two modes:
    - **Frame extraction** only if anchor timestamp not already covered by
      existing frames (avoids adding noisy duplicates).
    - **Causal edge inference** always runs for causal/temporal gap slots.

    No subprocesses or multiprocessing workers are spawned.
    """
    # BUILD bypasses execute_action, which is what normally resets these — do it
    # here so the elasticity ablation only removes nodes BUILD actually added.
    state.last_ring = set()
    state.stuck = False
    anchor = _build_anchor(state, graph, faiss_index, siglip_encoder)
    gap_slot = _first_unfilled_gap_slot(state)
    radius, n_frames = _build_params(gap_slot, state.build_radius_multiplier)

    t0 = max(0, anchor - radius)
    t1 = anchor + radius

    if gap_slot in ("entity_or_frame", "mcq_option_coverage"):
        # Visual/identity gap → interactively perceive the window into NEW
        # OCR/Character/Action nodes (also adds the frames for the answerer).
        _perceive_window(state, graph, video_path, llm, t0, t1, n_frames,
                         video_type=video_type)
    else:
        # Causal/temporal/episode gap → pull frames (for the answerer) and infer
        # causal edges among the existing events in the window.
        anchor_bucket = int(anchor // 10)
        existing_buckets = {int(ts // 10) for _, ts in state.frames}
        if anchor_bucket not in existing_buckets:
            frames = extract_frames_for_window(
                video_path, t0, t1, max_frames=n_frames, motion_rank=True)
            state.add_frames(list(zip(frames_to_b64_urls(frames),
                                      [f.timestamp for f in frames])))
        _build_causal_edges(state, graph, llm, t0, t1)

    # Bump expansion for next BUILD call
    state.build_radius_multiplier *= 2
    state.build_density *= 2


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


# Hard cap on serialized-context characters per prompt. The model max is 65,536
# tokens including image tokens (~10k for 8 frames); 140k chars ≈ ~35k text
# tokens leaves comfortable headroom. Without this cap, large subgraphs
# serialized past the model limit and every question in the batch ERRORed.
MAX_CTX_CHARS = 140_000


def _cap_ctx(ctx: str) -> str:
    if len(ctx) <= MAX_CTX_CHARS:
        return ctx
    return ctx[:MAX_CTX_CHARS] + "\n...[context truncated]"


def _subgraph_of(graph: VKGraph, node_ids: Set[str]):

    return graph_ops.build_subgraph(graph, node_ids)


def _controller_messages(state: WalkState, graph: VKGraph,
                         max_image_frames: int = MAX_IMAGE_FRAMES_PER_PROMPT) -> list:
    """Multimodal controller prompt — frames + serialized subgraph + gaps.

    Unlike the answerer, the controller's job is *navigation*: it sees the
    same frame archive and image batch so it can visually verify whether a
    ZOOM target is promising, or whether an EXPAND actually reached a scene
    relevant to the question — eliminating blind text-only turns.
    """
    sub = _subgraph_of(graph, state.node_ids)
    ctx = _SERIALIZER.serialize(
        sub, state.question, intents=[], question_types=state.qtype,
        t_start=state.time_reference[0] if state.time_reference else None,
        t_end=state.time_reference[1] if state.time_reference else None,
        visual_provided=bool(state.frames),
        include_edges=True,
    )

    gap_text = "(none — every required slot is filled)"
    coverage = 0.0
    if state.warrant:
        coverage = state.warrant.coverage
        if state.warrant.gaps:
            gap_text = "\n".join(f"  - {g}" for g in state.warrant.gaps)

    frontier_preview = ", ".join(sorted(state.frontier)[:12]) or "(empty)"
    ans = state.current_answer or "(none yet)"
    stuck = ""
    if state.stuck:
        stuck = "\n[STUCK] Last EXPAND added no new nodes. Try RECALL(query) instead — it searches the entire graph."

    content: List[dict] = []

    # 1) Frame archive — text summaries of previously analyzed frame batches.
    if state.frame_archive:
        archive_block = "\n".join(
            f"- {e}" for e in state.frame_archive[-30:])
        content.append({"type": "text", "text":
            f"## Previously analyzed frames (text summaries)\n{archive_block}\n"})

    # 2) Current image batch — at most max_image_frames.
    batch = state.frames
    if len(batch) > max_image_frames:
        batch = batch[-max_image_frames:]
    for url, ts in batch:
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"[t={ts:.0f}s]"})

    # 3) Navigation context (same as before, as text).
    content.append({"type": "text", "text":
        f"{_cap_ctx(ctx)}\n\n"
        f"## Frontier node ids\n{frontier_preview}\n\n"
        f"## Missing evidence (gaps)\n{gap_text}\n\n"
        f"## Coverage so far\n{coverage:.2f}\n\n"
        f"## Your last tentative answer\n{ans}\n\n"
        f"{stuck}\n"
        f"Choose ONE action (JSON)."})

    return [
        {"role": "system", "content": WALKER_CONTROLLER_SYSTEM},
        {"role": "user", "content": content},
    ]


def _answerer_messages(state: WalkState, graph: VKGraph, node_ids: Set[str],
                       max_image_frames: int = MAX_IMAGE_FRAMES_PER_PROMPT) -> list:
    """Multimodal answerer prompt — respects Qwen's 10-image-per-prompt limit.

    Frames beyond ``max_image_frames`` are converted to text summaries (from
    VKG metadata) and fed as context alongside the current image batch. This
    lets the model process arbitrarily many frames across hops without
    exceeding the architecture's image budget.
    """
    sub = _subgraph_of(graph, node_ids)
    ctx = _SERIALIZER.serialize(
        sub, state.question, intents=[], question_types=state.qtype,
        t_start=state.time_reference[0] if state.time_reference else None,
        t_end=state.time_reference[1] if state.time_reference else None,
        visual_provided=bool(state.frames),
        include_edges=True,
    )
    content: List[dict] = []

    # 1) Frame archive — text summaries of previously analyzed frame batches.
    if state.frame_archive:
        archive_block = "\n".join(
            f"- {e}" for e in state.frame_archive[-30:])
        content.append({"type": "text", "text":
            f"## Previously analyzed frames (text summaries)\n{archive_block}\n"})

    # 2) Current image batch — at most max_image_frames.
    batch = state.frames
    if len(batch) > max_image_frames:
        batch = batch[-max_image_frames:]
    for url, ts in batch:
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"[t={ts:.0f}s]"})

    # 3) Serialized sub-graph context.
    content.append({"type": "text", "text": _cap_ctx(ctx)})
    return [
        {"role": "system", "content": WALKER_ANSWER_SYSTEM},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Forced backtrack (rejected ANSWER / unsatisfied STOP_REQUEST)
# ---------------------------------------------------------------------------

def _forced_action(state: WalkState, graph: VKGraph) -> Dict[str, Any]:
    """Pick a deterministic action targeting the first unfilled slot."""
    w = state.warrant
    slots = list(w.gap_slots) if w else []
    if "mcq_option_coverage" in slots and w and w.missing_options:
        return {"action": "DISCRIMINATE", "option": w.missing_options[0]}
    if "causal_edge" in slots:
        return {"action": "BUILD"}
    if "temporal_occurrences" in slots:
        return {"action": "EXPAND", "relation": "ENTITY"}
    if "episode_summary" in slots:
        return {"action": "EXPAND", "relation": "CONTAINS"}
    if "entity_or_frame" in slots:
        return {"action": "BUILD"}
    return {"action": "RECALL", "query": state.question}


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> Dict[str, Any]:
    """Parse the controller's constrained-JSON action; tolerate stray prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find the first JSON object.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return {"action": "RECALL", "query": ""}


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def _log_hop(debug_dir: Optional[str], rec: dict) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "debug_walker.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API — batched-by-hop walk
# ---------------------------------------------------------------------------

def batch_walk_answer_questions(
    questions: List[dict],
    graph: VKGraph,
    faiss_index,
    llm,
    siglip,
    video_path: Optional[str] = None,
    frame_store=None,
    mcq: bool = True,
    debug_dir: Optional[str] = None,
    max_hops: int = 4,
    theta_cov: float = 1.0,
    k: int = 5,
) -> List[AnswerResult]:
    """Answer all questions via the warrant-gated traversal, batched by hop.

    Per hop, at most three batched ``llm.chat`` calls are issued across all live
    states: one controller pass and two answerer read-outs (full + ablated).
    """
    # Seed all states deterministically (no LLM).
    states: List[WalkState] = [
        seed_state(q, graph, faiss_index, siglip, frame_store, video_path, mcq)
        for q in questions
    ]

    # Initial warrant (coverage only; elasticity unknown → treated as 1).
    for st in states:
        sub = _subgraph_of(graph, st.node_ids)
        st.warrant = warrant_mod.compute_warrant(
            sub, st.qtype, st.question, st.options, bool(st.frames),
            answer_full=None, answer_ablated=None, mcq=mcq, theta_cov=theta_cov,
            discriminated=st.discriminated,
        )

    def live() -> List[WalkState]:
        return [s for s in states if not s.done]

    for hop in range(max_hops):
        live_states = live()
        if not live_states:
            break

        # ---- 1) Controller batch: one action per live state. ----
        ctrl_msgs = [_controller_messages(s, graph) for s in live_states]
        ctrl_out = llm.chat(
            messages=ctrl_msgs,
            sampling_params=_CONTROLLER_SAMPLING,
            use_tqdm=False,
        )
        actions = [_parse_action(o.outputs[0].text) for o in ctrl_out]

        # ---- 2) Resolve each action (verify ANSWER / STOP, else execute). ----
        for st, action in zip(live_states, actions):
            st.hop = hop
            st.last_action = action
            name = (action.get("action") or "").upper()

            if name == "BUILD":
                st.history.append({"hop": hop, "action": "BUILD"})
                _execute_build(st, graph, faiss_index, siglip,
                               video_path, llm, frame_store)
                continue
            elif name == "ANSWER":
                letter = (action.get("letter") or "").upper()
                cited = action.get("cited_node_ids") or []
                sub = _subgraph_of(graph, st.node_ids)
                ok, why = verify(letter, cited, sub, st.qtype, st.question,
                                 has_frames=bool(st.frames))
                st.history.append({"hop": hop, "action": "ANSWER", "letter": letter,
                                   "cited": cited, "verify_ok": ok, "verify_why": why})
                if ok and st.warrant and st.warrant.satisfied:
                    st.final_answer = letter
                    st.current_answer = letter
                    st.cited_node_ids = cited
                    st.done = True
                    _log_hop(debug_dir, _hop_record(st, action, ok, why))
                    continue
                # Rejected / not yet warranted → forced backtrack toward a gap.
                action = _forced_action(st, graph)
                st.forced_action = action
                st.last_action = {"forced_from": "ANSWER", **action}
            elif name == "STOP_REQUEST":
                st.history.append({"hop": hop, "action": "STOP_REQUEST"})
                if st.warrant and st.warrant.satisfied:
                    st.final_answer = st.current_answer
                    st.done = True
                    _log_hop(debug_dir, _hop_record(st, action, True, "stop satisfied"))
                    continue
                action = _forced_action(st, graph)
                st.forced_action = action
                st.last_action = {"forced_from": "STOP_REQUEST", **action}
            else:
                st.history.append({"hop": hop, "action": name,
                                   "args": {k2: v for k2, v in action.items()
                                            if k2 != "action"}})

            execute_action(action, st, graph, faiss_index, siglip,
                           frame_store, video_path, k=k)

        # ---- 3) Answerer read-outs (full + ablated), batched. ----
        # Both are GREEDY (WALKER_PROBE_SAMPLING) so the elasticity comparison is
        # a true deterministic finite difference, not two noisy temp>0 draws. The
        # high-temperature thinking answer is generated once, at emission (below).
        pending = [s for s in live_states if not s.done]
        if not pending:
            continue

        full_msgs = [_answerer_messages(s, graph, s.node_ids) for s in pending]
        full_out = llm.chat(
            messages=full_msgs,
            sampling_params=WALKER_PROBE_SAMPLING,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )
        for s, o in zip(pending, full_out):
            s.prev_answer = s.current_answer
            ans = extract_mcq_answer(o.outputs[0].text) if mcq \
                else o.outputs[0].text.strip()
            # Empty = truncated/parse failure: keep the previous answer rather
            # than letting "" flow into the warrant as a real prediction.
            s.current_answer = ans or s.current_answer

        # Ablated probe — only for states whose last action added a ring.
        # States whose action added NO ring get no free pass: their elasticity
        # falls back to answer stability across hops (prev vs current), which is
        # None→elastic at hop 0. Defaulting to current_answer here made
        # elasticity trivially 0 and retired states at hop 0 untested.
        ablate = [s for s in pending if s.last_ring]
        ablated_answers: Dict[str, Optional[str]] = {s.qid: s.prev_answer for s in pending}
        if ablate:
            ab_msgs = [_answerer_messages(s, graph, s.node_ids - s.last_ring)
                       for s in ablate]
            ab_out = llm.chat(
                messages=ab_msgs,
                sampling_params=WALKER_PROBE_SAMPLING,
                chat_template_kwargs={"enable_thinking": mcq},
                use_tqdm=False,
            )
            for s, o in zip(ablate, ab_out):
                ab = extract_mcq_answer(o.outputs[0].text) if mcq \
                    else o.outputs[0].text.strip()
                ablated_answers[s.qid] = ab or None

        # ---- 3b) Archive old frames, store answerer analysis as text. ----
        # Frames beyond the per-prompt image limit are converted to VKG-text
        # summaries so no visual evidence is lost. The answerer's own output
        # is stored alongside as a persistent "analysis" that survives frame
        # eviction.
        for s in pending:
            if s.current_answer:
                s.frame_archive.append(
                    f"[hop {s.hop}] Answerer analysis: {s.current_answer}"
                )
            s.archive_old_frames(graph)

        # ---- 4) Recompute warrant; retire satisfied states. ----
        for s in pending:
            sub = _subgraph_of(graph, s.node_ids)
            s.warrant = warrant_mod.compute_warrant(
                sub, s.qtype, s.question, s.options, bool(s.frames),
                answer_full=s.current_answer,
                answer_ablated=ablated_answers.get(s.qid),
                mcq=mcq, theta_cov=theta_cov,
                discriminated=s.discriminated,
            )
            _log_hop(debug_dir, _hop_record(s, s.last_action, None, None))
            if s.warrant.satisfied:
                s.final_answer = s.current_answer
                # Capture citations from a verified ANSWER for this letter, if any.
                cur = (s.current_answer or "").upper()
                for h in reversed(s.history):
                    if h.get("action") == "ANSWER" and h.get("verify_ok") \
                       and (h.get("letter") or "").upper() == cur:
                        s.cited_node_ids = h.get("cited", [])
                        break
                s.done = True

    # ---- Final emission pass: full-thinking-budget greedy answer. ----
    # Only for states the walk did NOT retire (hop cap reached without a
    # satisfied warrant). States that converged — warrant-satisfied or
    # citation-verified — keep the answer the walk certified; re-answering them
    # here was overwriting converged-correct letters.
    final_states = [s for s in states if not s.final_answer]
    if final_states:
        fin_msgs = [_answerer_messages(s, graph, s.node_ids) for s in final_states]
        fin_out = llm.chat(
            messages=fin_msgs,
            sampling_params=WALKER_ANSWER_SAMPLING,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )
        for s, o in zip(final_states, fin_out):
            ans = extract_mcq_answer(o.outputs[0].text) if mcq \
                else o.outputs[0].text.strip()
            if ans:
                s.final_answer = ans

    # Emit: verified/satisfied answer, else current read-out (lowest-risk).
    results: List[AnswerResult] = []
    for s in states:
        answer = s.final_answer or s.current_answer or "ERROR"
        results.append(AnswerResult(
            answer=answer,
            intents=[h.get("action", "") for h in s.history],
            subgraph_size=len(s.node_ids),
            keyframes_used=[ts for _, ts in s.frames],
            evidence_nodes=s.cited_node_ids,
        ))
    return results


def _hop_record(state: WalkState, action, verify_ok, verify_why) -> dict:
    w = state.warrant
    return {
        "qid": state.qid,
        "hop": state.hop,
        "action": action,
        "frontier_size": len(state.frontier),
        "subgraph_size": len(state.node_ids),
        "n_frames": len(state.frames),
        "n_frame_archive": len(state.frame_archive),
        "coverage": w.coverage if w else None,
        "gaps": w.gaps if w else None,
        "elasticity": w.elasticity if w else None,
        "current_answer": state.current_answer,
        "citations": state.cited_node_ids,
        "verify_ok": verify_ok,
        "verify_why": verify_why,
        "satisfied": w.satisfied if w else None,
    }
