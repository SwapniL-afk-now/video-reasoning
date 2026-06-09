from __future__ import annotations

"""ContextSerializer: converts SubGraph → structured NL for the VLM."""

from typing import List, Optional, Set

from ..schema import SubGraph

# Node types that are relevant per question type — keeps prompts lean
RELEVANT_NODE_TYPES: dict = {
    "key information retrieval": {"OCRNode", "SpeechNode", "ClipNode", "CharacterNode"},
    "entity recognition":        {"CharacterNode", "ObjectNode", "ClipNode", "SceneNode"},
    "event understanding":       {"CharacterNode", "ActionNode", "InteractionNode",
                                   "StateChangeNode", "ClipNode", "SceneNode"},
    "reasoning":                 {"CharacterNode", "ActionNode", "CauseNode", "GoalNode",
                                   "SpeechNode", "StateChangeNode", "EpisodeNode"},
    "temporal grounding":        {"CharacterNode", "ClipNode", "SceneNode",
                                   "ActionNode", "StateChangeNode"},
    "summarization":             {"CharacterNode", "EpisodeNode", "SceneNode",
                                   "ActionNode", "SpeechNode"},
    "emotional reasoning":       {"CharacterNode", "SceneNode", "SpeechNode"},
    "intentional reasoning":     {"CharacterNode", "ActionNode", "SpeechNode",
                                   "EpisodeNode"},
    "prospective reasoning":     {"SceneNode", "EpisodeNode", "ActionNode"},
    "counterfactual reasoning":  {"CharacterNode", "ActionNode", "StateChangeNode",
                                   "SceneNode", "EpisodeNode"},
    "spatial understanding":     {"SceneNode", "ObjectNode", "CharacterNode",
                                   "LocationNode"},
}

# Always include these regardless of question type
ALWAYS_INCLUDE: Set[str] = {"SceneNode", "EpisodeNode"}


def _relevant_types(question_types: List[str]) -> Optional[Set[str]]:
    """Return the union of relevant node types for the given question types.
    Returns None if we can't narrow down (all types allowed)."""
    if not question_types:
        return None
    result: Set[str] = set(ALWAYS_INCLUDE)
    for qt in question_types:
        qt_clean = qt.strip().strip("'[]")
        if qt_clean in RELEVANT_NODE_TYPES:
            result |= RELEVANT_NODE_TYPES[qt_clean]
    return result if len(result) > len(ALWAYS_INCLUDE) else None


class ContextSerializer:

    def serialize(
        self,
        subgraph: SubGraph,
        question: str,
        intents: List[str],
        question_types: Optional[List[str]] = None,
        t_start: Optional[float] = None,
        t_end:   Optional[float] = None,
        visual_provided: bool = False,
    ) -> str:
        """Serialize a subgraph to a VLM-readable context string.

        visual_provided=True signals that the caller is also sending video frames
        alongside this text.  In that mode we skip ActionNode and ObjectNode
        descriptions — the model can read those directly from the images, and
        including noisy per-window descriptions only adds misleading signal.
        """
        sections: List[str] = []

        # Window header when time reference is present
        if t_start is not None and t_end is not None:
            sections.append(
                f"## Focus Window: {t_start:.0f}s – {t_end:.0f}s\n"
                f"(The answer is located within this time window)"
            )

        # Filter nodes by question type relevance
        allowed_types = _relevant_types(question_types or [])

        def include(node_type: str) -> bool:
            return allowed_types is None or node_type in allowed_types

        # When visual frames are provided, skip per-window visual descriptors.
        # ActionNodes and ObjectNodes are noisy VLM extractions over 10-frame
        # windows; when the model can see the actual frames it should use those
        # directly. OCR, speech, and characters carry non-visual signal that
        # frames alone can't provide.
        VISUAL_ONLY_TYPES = {"ActionNode", "ObjectNode"}

        def include_node(node_type: str) -> bool:
            if visual_provided and node_type in VISUAL_ONLY_TYPES:
                return False
            return include(node_type)

        # Filter subgraph nodes for timeline
        timeline = [
            n for n in subgraph.get_sorted_events()
            if include_node(n.node_type)
        ]

        # Window filter: prefer nodes inside the focus window
        if t_start is not None and timeline:
            lo, hi = t_start - 30, t_end + 30
            in_window = [n for n in timeline if lo <= n.t_start <= hi]
            if in_window:
                timeline = in_window

        if timeline:
            # Wide span (>60s): group non-SceneNode content under SceneNode
            # time headers so the model can track what happened when.
            # Narrow span: flat list (fast to read, precise time already shown).
            span = (t_end - t_start) if (t_start is not None and t_end is not None) else 0
            scene_nodes_in_tl = [n for n in timeline if n.node_type == "SceneNode"]
            use_grouped = span > 60 and len(scene_nodes_in_tl) >= 2

            if use_grouped:
                sections.append("## Timeline (grouped by segment)")
                # Build scene boundary list sorted by time
                scene_boundaries = sorted(scene_nodes_in_tl, key=lambda n: n.t_start)
                # Assign each non-SceneNode to the closest scene segment
                non_scene = [n for n in timeline if n.node_type != "SceneNode"]
                for sc in scene_boundaries:
                    children = [
                        n for n in non_scene
                        if sc.t_start <= n.t_start <= sc.t_end
                    ]
                    header = f"  --- [{sc.t_start:.0f}s–{sc.t_end:.0f}s] {sc.label} ---"
                    sections.append(header)
                    for n in children:
                        conf_str = (f" (conf:{n.confidence:.2f})"
                                    if n.confidence < 0.8 else "")
                        sections.append(
                            f"    [{n.node_type}] {n.label}{conf_str}"
                        )
            else:
                sections.append("## Timeline")
                for n in timeline:
                    conf_str = (f" (confidence: {n.confidence:.2f})"
                                if n.confidence < 0.8 else "")
                    prec_str = (f" [gap:{n.temporal_precision:.0f}s]"
                                if n.temporal_precision > 8 else "")
                    func = n.metadata.get("narrative_function", "")
                    func_str = f" [{func}]" if func else ""
                    sections.append(
                        f"  [{n.t_start:.0f}s–{n.t_end:.0f}s] "
                        f"[{n.node_type}]{func_str} {n.label}{conf_str}{prec_str}"
                    )

        # Causal chains (reasoning / causal intent)
        if "CAUSAL" in intents or (question_types and
                                    any("reasoning" in qt for qt in question_types)):
            chains = subgraph.get_causal_chains()
            if chains:
                sections.append("\n## Causal Relationships")
                for c in chains:
                    sections.append(
                        f"  [{c.source.t_start:.0f}s] {c.source.label}\n"
                        f"    ──[{c.relation}, conf={c.confidence:.2f}]──▶\n"
                        f"  [{c.target.t_start:.0f}s] {c.target.label}\n"
                        f"    Reason: {c.metadata.get('reasoning', 'not specified')}"
                    )

        # Characters (always included — critical for all question types)
        chars = subgraph.get_characters()
        if chars:
            sections.append("\n## Characters")
            for ch in chars:
                appearances = ch.metadata.get("appearances", [])
                times = [f"{a['timestamp']:.0f}s" for a in appearances[:8]]
                emotions = [a["emotion"] for a in appearances if a.get("emotion")]
                emotion_str = f" [{emotions[-1]}]" if emotions else ""
                sections.append(
                    f"  {ch.label}{emotion_str}: {ch.canonical_description or ch.label}\n"
                    f"    Appears at: {', '.join(times) or 'unknown'}"
                )

        # Emotional arc (EMOTIONAL intent or when EMOTION_SHIFT edges are present)
        if "EMOTIONAL" in intents or any(
            e.relation_type == "EMOTION_SHIFT" for e in subgraph.edges
        ):
            emotion_arcs = []
            for edge in sorted(subgraph.edges, key=lambda e: e.metadata.get("t", 0)):
                if edge.relation_type != "EMOTION_SHIFT":
                    continue
                src = subgraph.nodes.get(edge.source_id)
                tgt = subgraph.nodes.get(edge.target_id)
                if not src or not tgt:
                    continue
                from_e = edge.metadata.get("from", "?")
                to_e   = edge.metadata.get("to", "?")
                char_label = src.label
                emotion_arcs.append(
                    f"  {char_label}: {from_e} → {to_e} "
                    f"(at {src.t_start:.0f}s → {tgt.t_start:.0f}s)"
                )
            if emotion_arcs:
                sections.append("\n## Emotional Arc")
                sections.extend(emotion_arcs)

        # State changes (always included)
        states = subgraph.get_state_changes()
        if states:
            sections.append("\n## State Changes")
            for s in states:
                sections.append(
                    f"  [{s.t_start:.0f}s] {s.metadata.get('entity', s.label)}: "
                    f"{s.prev_state} → {s.next_state}"
                )

        # Spatial layout
        if "SPATIAL" in intents:
            spatial = subgraph.get_spatial_relations()
            if spatial:
                sections.append("\n## Spatial Layout")
                for r in spatial[:15]:
                    src = subgraph.nodes.get(r.source_id)
                    tgt = subgraph.nodes.get(r.target_id)
                    if src and tgt:
                        rel_str = r.relation_type.lower().replace("_", " ")
                        sections.append(
                            f"  [{src.t_start:.0f}s] {src.label} {rel_str} {tgt.label}"
                        )

        # Episode summaries for summarization questions
        if question_types and any("summarization" in qt for qt in question_types):
            eps = sorted(
                [n for n in subgraph.nodes.values() if n.node_type == "EpisodeNode"],
                key=lambda n: n.t_start,
            )
            if eps:
                sections.append("\n## Episode Summaries")
                for ep in eps:
                    summary = ep.metadata.get("summary", "")
                    sections.append(
                        f"  [{ep.t_start:.0f}s–{ep.t_end:.0f}s] {ep.label}"
                        + (f" — {summary[:100]}" if summary else "")
                    )

        # Dialogue — narrator lines first (authoritative), then character speech
        speeches = subgraph.get_speech_nodes()
        if speeches:
            sections.append("\n## Dialogue")
            spoken_by: dict = {}
            for edge in subgraph.edges:
                if edge.relation_type == "SPOKEN_BY":
                    char = subgraph.nodes.get(edge.target_id)
                    if char:
                        spoken_by[edge.source_id] = char.label[:30]

            # Sort: narrator lines first, then by timestamp
            def _speech_sort_key(s):
                is_narrator = s.metadata.get("source", "") == "narrator"
                return (0 if is_narrator else 1, s.t_start)

            for s in sorted(speeches, key=_speech_sort_key)[:20]:
                is_narrator = s.metadata.get("source", "") == "narrator"
                speaker = spoken_by.get(s.id, "")
                if is_narrator:
                    speaker_str = "[NARRATOR]: "
                elif speaker:
                    speaker_str = f"{speaker}: "
                else:
                    speaker_str = ""
                sections.append(f'  [{s.t_start:.0f}s] {speaker_str}"{s.label}"')

        # What happens next (PROSPECTIVE intent)
        if "PROSPECTIVE" in intents:
            future_events = sorted(
                [
                    n for n in subgraph.nodes.values()
                    if n.node_type in ("SceneNode", "EpisodeNode", "ActionNode")
                    and (not timeline or n.t_start > max(e.t_end for e in timeline))
                ],
                key=lambda n: n.t_start,
            )
            if future_events:
                sections.append("\n## What Happens Next")
                for n in future_events[:6]:
                    sections.append(
                        f"  [{n.t_start:.0f}s–{n.t_end:.0f}s] [{n.node_type}] {n.label}"
                    )

        context = "\n".join(sections)
        return (
            "You are answering a question about a video. "
            "The following knowledge was extracted from the video:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the above knowledge and the provided video frames. "
            "Cite specific timestamps as evidence where relevant."
        )
