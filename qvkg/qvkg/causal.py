from __future__ import annotations

"""Per-episode causal chain inference via visually-grounded vLLM batch."""

import json
from typing import List

from .frame_store import FrameStore
from .schema import Episode, VKGEdge, VKGraph
from .vllm_client import CAUSAL_SAMPLING, CAUSAL_SYSTEM_PROMPT


def build_causal_request(
    episode: Episode,
    graph: VKGraph,
    frame_store: FrameStore,
) -> dict:
    events = graph.get_events_in_episode(episode)
    # Cap events to avoid exceeding the model's context window on long episodes.
    MAX_EVENTS = 300
    if len(events) > MAX_EVENTS:
        events = events[:MAX_EVENTS]
    event_timeline = "\n".join(
        f"  [{e.t_start:.1f}s] [{e.node_type}] {e.label}"
        for e in events
    )

    keyframes = episode.get_representative_frames(max_frames=8)

    content = []
    for frame in keyframes:
        if frame.keyframe_id if hasattr(frame, "keyframe_id") else frame.id:
            fid = getattr(frame, "keyframe_id", frame.id)
            try:
                b64_url = frame_store.get_b64_url(fid)
                content.append({"type": "image_url",
                                 "image_url": {"url": b64_url}})
            except (KeyError, FileNotFoundError):
                pass
        content.append({"type": "text",
                         "text": f"[t={frame.timestamp:.1f}s]"})

    content.append({"type": "text", "text": (
        f"\nEpisode: \"{episode.label}\" "
        f"({episode.t_start:.0f}s - {episode.t_end:.0f}s)\n"
        f"Narrative role: {episode.narrative_role}\n\n"
        f"Event timeline:\n{event_timeline}\n\n"
        f"Identify causal relationships as JSON."
    )})

    return {
        "episode_id": episode.id,
        "messages": [
            {"role": "system", "content": CAUSAL_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    }


def parse_causal_edges(
    json_text: str,
    graph: VKGraph,
    episode: Episode,
    min_confidence: float = 0.6,
) -> List[VKGEdge]:
    try:
        causal_data = json.loads(json_text)
    except json.JSONDecodeError:
        return []

    events = graph.get_events_in_episode(episode)
    edges = []

    for link in causal_data:
        conf = float(link.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        cause_node = graph.find_event_by_description(link.get("cause", ""), events)
        effect_node = graph.find_event_by_description(link.get("effect", ""), events)

        if cause_node and effect_node and cause_node.id != effect_node.id:
            edges.append(VKGEdge(
                source_id=cause_node.id,
                target_id=effect_node.id,
                relation_type=link.get("relation", "CAUSES"),
                weight=conf,
                confidence=conf,
                metadata={"reasoning": link.get("reasoning", ""),
                          "source": "qwen_causal"},
            ))

    return edges


def infer_episode_causality(
    episode: Episode,
    graph: VKGraph,
    frame_store: FrameStore,
    llm,
) -> List[VKGEdge]:
    req = build_causal_request(episode, graph, frame_store)
    outputs = llm.chat(
        messages=[req["messages"]],
        sampling_params=CAUSAL_SAMPLING,
    )
    raw = outputs[0].outputs[0].text
    return parse_causal_edges(raw, graph, episode)
