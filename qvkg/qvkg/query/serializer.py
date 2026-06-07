from __future__ import annotations

"""ContextSerializer: converts SubGraph → structured NL for the VLM."""

from typing import List, Optional, Set

from ..schema import SubGraph

# Node types that are relevant per question type — keeps prompts lean
RELEVANT_NODE_TYPES: dict = {
    "key information retrieval": {"OCRNode", "SpeechNode", "ClipNode"},
    "entity recognition":        {"CharacterNode", "ObjectNode", "ClipNode", "SceneNode"},
    "event understanding":       {"ActionNode", "InteractionNode", "StateChangeNode",
                                   "ClipNode", "SceneNode"},
    "reasoning":                 {"ActionNode", "CauseNode", "GoalNode", "SpeechNode",
                                   "StateChangeNode", "EpisodeNode"},
    "temporal grounding":        {"ClipNode", "SceneNode", "ActionNode", "StateChangeNode"},
    "summarization":             {"EpisodeNode", "SceneNode", "ActionNode", "SpeechNode"},
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
    ) -> str:
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

        # Filter subgraph nodes for timeline
        timeline = [
            n for n in subgraph.get_sorted_events()
            if include(n.node_type)
        ]

        # Window filter: prefer nodes inside the focus window
        if t_start is not None and timeline:
            lo, hi = t_start - 30, t_end + 30
            in_window = [n for n in timeline if lo <= n.t_start <= hi]
            if in_window:
                timeline = in_window

        if timeline:
            sections.append("## Timeline")
            for n in timeline:
                conf_str = (f" (confidence: {n.confidence:.2f})"
                            if n.confidence < 0.8 else "")
                prec_str = (f" [gap:{n.temporal_precision:.0f}s]"
                            if n.temporal_precision > 8 else "")
                sections.append(
                    f"  [{n.t_start:.0f}s–{n.t_end:.0f}s] "
                    f"[{n.node_type}] {n.label}{conf_str}{prec_str}"
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

        # Characters (identity / entity recognition)
        if "IDENTITY" in intents or (question_types and
                                      any("entity" in qt for qt in question_types)):
            chars = subgraph.get_characters()
            if chars:
                sections.append("\n## Characters")
                for ch in chars:
                    appearances = ch.metadata.get("appearances", [])
                    times = [f"{a['timestamp']:.0f}s" for a in appearances[:8]]
                    sections.append(
                        f"  {ch.label}: {ch.canonical_description or ch.label}\n"
                        f"    Appears at: {', '.join(times) or 'unknown'}"
                    )

        # State changes
        if "STATE" in intents or (question_types and
                                   any("event" in qt for qt in question_types)):
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

        # Dialogue
        speeches = subgraph.get_speech_nodes()
        if speeches:
            sections.append("\n## Dialogue")
            for s in speeches[:12]:
                sections.append(f'  [{s.t_start:.0f}s] "{s.label}"')

        context = "\n".join(sections)
        return (
            "You are answering a question about a video. "
            "The following knowledge was extracted from the video:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the above knowledge and the provided video frames. "
            "Cite specific timestamps as evidence where relevant."
        )
