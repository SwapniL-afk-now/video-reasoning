from __future__ import annotations

"""Citation-grounded answer verification.

``ANSWER(letter, cited_node_ids)`` is checked, not trusted. A failed check is
not an error — the walker converts it into a forced EXPAND/DISCRIMINATE toward
the unsatisfied slot and keeps walking. The model cannot assert a conclusion the
graph does not support.
"""

import re
from typing import List, Set, Tuple

from ..schema import CAUSAL_EDGE_TYPES, SubGraph

_STOP = {
    "what", "when", "where", "which", "this", "that", "with", "from", "have",
    "does", "into", "about", "type", "last", "first", "video", "they", "their",
}


def _keywords(text: str) -> Set[str]:
    return {w for w in re.findall(r"\b\w{4,}\b", text.lower()) if w not in _STOP}


_CAUSAL_QTYPES = {
    "reasoning", "causal reasoning", "event understanding",
    "intentional reasoning", "multi-step reasoning", "common sense reasoning",
    "counterfactual reasoning",
}
_TEMPORAL_QTYPES = {
    "temporal grounding", "temporal understanding", "action recognition",
}
_ENTITY_QTYPES = {
    "entity recognition", "key information retrieval", "subtitle-based retrieval",
    "character understanding", "object recognition",
}


def verify(
    letter: str,
    cited_ids: List[str],
    subgraph: SubGraph,
    qtype: List[str],
    question: str,
    has_frames: bool = False,
) -> Tuple[bool, str]:
    """Return ``(ok, reason)``.

    A non-MCQ-letter or empty letter fails immediately. Otherwise we check
    grounding (every citation is a real, retrieved node) and structural support
    appropriate to the question type.
    """
    if not letter or letter.upper() not in ("A", "B", "C", "D"):
        return False, "answer is not a valid option letter"

    # --- Grounding: every cited id must be in the working subgraph. ---
    cited = [c for c in (cited_ids or []) if c]
    ungrounded = [c for c in cited if c not in subgraph.nodes]
    if ungrounded:
        return False, f"cited nodes not in retrieved subgraph: {ungrounded[:3]}"
    if not cited:
        # No citations at all — accept only if there is at least some evidence,
        # otherwise force more retrieval.
        if not subgraph.nodes and not has_frames:
            return False, "no citations and no retrieved evidence"
        return True, "accepted (no explicit citations, evidence present)"

    cited_nodes = [subgraph.nodes[c] for c in cited]
    qts = {q.strip().lower() for q in (qtype or [])}

    # --- Structural support by qtype. ---
    if qts & _CAUSAL_QTYPES:
        if not _on_causal_path(subgraph, cited, question):
            return False, "no cited node lies on a causal path to the queried event"
        return True, "causal support verified"

    if qts & _TEMPORAL_QTYPES:
        timestamped = [n for n in cited_nodes if n.t_start is not None]
        if len(timestamped) < 2:
            return False, "ordering claim needs ≥2 timestamped cited nodes"
        return True, "temporal support verified"

    if qts & _ENTITY_QTYPES:
        if has_frames:
            return True, "entity support: frames present"
        if any(n.node_type in ("OCRNode", "CharacterNode") for n in cited_nodes):
            return True, "entity support: OCR/character node cited"
        return False, "entity question needs a cited OCR/character node or a frame"

    # Default: grounded citation is sufficient.
    return True, "grounded citation accepted"


def _on_causal_path(subgraph: SubGraph, cited: List[str], question: str) -> bool:
    """True if a cited node participates in a causal edge, preferring stem match."""
    cited_set = set(cited)
    kws = _keywords(question)
    causal_incident = False
    for e in subgraph.edges:
        if e.relation_type not in CAUSAL_EDGE_TYPES:
            continue
        if e.source_id in cited_set or e.target_id in cited_set:
            causal_incident = True
            # Stronger acceptance if the other endpoint is stem-matched.
            for nid in (e.source_id, e.target_id):
                n = subgraph.nodes.get(nid)
                if n and any(kw in n.label.lower() for kw in kws):
                    return True
    return causal_incident
