from __future__ import annotations

"""Coverage–elasticity warrant: the deterministic stop controller.

The model never decides when it is done. Termination is computed externally over
the subgraph the model built:

* **coverage** — is the structurally-required evidence (per qtype) present?
* **elasticity** — has the predicted answer stopped moving as evidence arrives?

``satisfied = (coverage >= theta_cov) AND (elasticity == 0)``.

The unmet slots are surfaced as ``gaps`` and fed back into the controller prompt
so self-refinement is *directed* (fill the missing slot) rather than random.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..schema import SubGraph

_STOP = {
    "what", "when", "where", "which", "this", "that", "with", "from", "have",
    "does", "into", "about", "type", "last", "first", "video", "they", "their",
    "would", "could", "there", "after", "before", "during", "while",
}


def _keywords(text: str) -> Set[str]:
    return {w for w in re.findall(r"\b\w{4,}\b", text.lower()) if w not in _STOP}


# ---------------------------------------------------------------------------
# Warrant dataclass
# ---------------------------------------------------------------------------

@dataclass
class Warrant:
    coverage: float = 0.0
    elasticity: int = 1
    gaps: List[str] = field(default_factory=list)          # human-readable, for the prompt
    gap_slots: List[str] = field(default_factory=list)     # slot names, for forced backtrack
    missing_options: List[str] = field(default_factory=list)  # uncovered MCQ letters
    satisfied: bool = False


# ---------------------------------------------------------------------------
# Required slots per qtype
# ---------------------------------------------------------------------------

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
_SUMMARY_QTYPES = {"summarization", "video understanding"}


def required_slots(qtype: List[str], question: str, mcq: bool) -> List[str]:
    """Return the list of slot names required for this question.

    Slot names are human-readable so they double as the controller's gap list.
    """
    qts = {q.strip().lower() for q in (qtype or [])}
    slots: List[str] = []

    if qts & _CAUSAL_QTYPES:
        slots.append("causal_edge")
    if qts & _TEMPORAL_QTYPES:
        slots.append("temporal_occurrences")
    if qts & _ENTITY_QTYPES:
        slots.append("entity_or_frame")
    if qts & _SUMMARY_QTYPES:
        slots.append("episode_summary")

    if mcq:
        slots.append("mcq_option_coverage")

    # Always require at least one piece of grounded evidence.
    if not slots:
        slots.append("any_evidence")
    return slots


# ---------------------------------------------------------------------------
# Slot satisfaction checks
# ---------------------------------------------------------------------------

def _causal_edge_present(subgraph: SubGraph, question: str) -> bool:
    chains = subgraph.get_causal_chains()
    if not chains:
        return False
    kws = _keywords(question)
    if not kws:
        return True
    # Prefer a causal chain incident to a stem-matched node.
    for c in chains:
        text = (c.source.label + " " + c.target.label).lower()
        if any(kw in text for kw in kws):
            return True
    # A causal edge exists but isn't stem-matched — still partial evidence.
    return True


def _temporal_occurrences_present(subgraph: SubGraph) -> bool:
    # ≥2 distinct timestamped appearances sharing an entity_id.
    by_entity: Dict[str, Set[float]] = {}
    for n in subgraph.nodes.values():
        if n.entity_id:
            by_entity.setdefault(n.entity_id, set()).add(round(n.t_start, 1))
    return any(len(ts) >= 2 for ts in by_entity.values())


def _entity_or_frame_present(subgraph: SubGraph, has_frames: bool) -> bool:
    if has_frames:
        return True
    for n in subgraph.nodes.values():
        if n.node_type in ("OCRNode", "CharacterNode"):
            return True
    return False


def _episode_summary_present(subgraph: SubGraph) -> bool:
    for n in subgraph.nodes.values():
        if n.node_type == "EpisodeNode":
            return True
    return False


def _mcq_option_coverage(
    subgraph: SubGraph,
    options: Dict[str, str],
    discriminated: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """Each option needs ≥1 supporting-or-refuting node, OR to have been probed.

    An option counts as *addressed* if a node's text overlaps it (supporting/
    refuting evidence present) or the controller has already issued a
    DISCRIMINATE for it — a found-nothing probe still settles the option.
    Returns ``(all_covered, missing_letters)``.
    """
    if not options:
        return True, []
    discriminated = discriminated or set()
    node_texts = [
        (n.label + " " + (n.canonical_description or "")).lower()
        for n in subgraph.nodes.values()
    ]
    missing: List[str] = []
    for letter, text in options.items():
        if letter in discriminated:
            continue
        kws = _keywords(text)
        if not kws:
            continue
        covered = any(any(kw in nt for kw in kws) for nt in node_texts)
        if not covered:
            missing.append(letter)
    return (len(missing) == 0), missing


def _any_evidence(subgraph: SubGraph, has_frames: bool) -> bool:
    return bool(subgraph.nodes) or has_frames


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_coverage(
    subgraph: SubGraph,
    slots: List[str],
    question: str,
    options: Dict[str, str],
    has_frames: bool,
    discriminated: Optional[Set[str]] = None,
) -> Tuple[float, List[str], List[str], List[str]]:
    """Return ``(coverage in [0,1], gap_descriptions, gap_slots, missing_options)``."""
    if not slots:
        return 1.0, [], [], []

    filled = 0
    gaps: List[str] = []
    gap_slots: List[str] = []
    missing_options: List[str] = []

    for slot in slots:
        ok = True
        gap_msg = ""
        if slot == "causal_edge":
            ok = _causal_edge_present(subgraph, question)
            gap_msg = "missing: a causal edge linking the queried event to a cause/effect"
        elif slot == "temporal_occurrences":
            ok = _temporal_occurrences_present(subgraph)
            gap_msg = "missing: ≥2 timestamped occurrences of the queried entity to order them"
        elif slot == "entity_or_frame":
            ok = _entity_or_frame_present(subgraph, has_frames)
            gap_msg = "missing: an OCR/character node or a frame at the referenced moment"
        elif slot == "episode_summary":
            ok = _episode_summary_present(subgraph)
            gap_msg = "missing: an episode summary spanning the queried range"
        elif slot == "mcq_option_coverage":
            ok, missing = _mcq_option_coverage(subgraph, options, discriminated)
            missing_options = missing
            gap_msg = f"missing: supporting/refuting evidence for option(s) {', '.join(missing)}"
        elif slot == "any_evidence":
            ok = _any_evidence(subgraph, has_frames)
            gap_msg = "missing: any grounded evidence for the question"

        if ok:
            filled += 1
        else:
            gaps.append(gap_msg)
            gap_slots.append(slot)

    return filled / len(slots), gaps, gap_slots, missing_options


# ---------------------------------------------------------------------------
# Elasticity probe (no LLM call — consumes pre-computed answerer read-outs)
# ---------------------------------------------------------------------------

def elasticity_probe(answer_full: Optional[str], answer_ablated: Optional[str]) -> int:
    """1 if the answer changes when the last evidence ring is removed, else 0.

    Both arguments are letters already produced by the batched answerer calls in
    the walker — this function makes no LLM call itself.
    """
    if answer_full is None or answer_ablated is None:
        return 1
    return 0 if answer_full.strip().upper() == answer_ablated.strip().upper() else 1


# ---------------------------------------------------------------------------
# Top-level warrant computation
# ---------------------------------------------------------------------------

def compute_warrant(
    subgraph: SubGraph,
    qtype: List[str],
    question: str,
    options: Dict[str, str],
    has_frames: bool,
    answer_full: Optional[str],
    answer_ablated: Optional[str],
    mcq: bool = True,
    theta_cov: float = 1.0,
    discriminated: Optional[Set[str]] = None,
) -> Warrant:
    slots = required_slots(qtype, question, mcq)
    coverage, gaps, gap_slots, missing_options = compute_coverage(
        subgraph, slots, question, options, has_frames, discriminated)
    elasticity = elasticity_probe(answer_full, answer_ablated)
    satisfied = (coverage >= theta_cov) and (elasticity == 0)
    return Warrant(
        coverage=coverage,
        elasticity=elasticity,
        gaps=gaps,
        gap_slots=gap_slots,
        missing_options=missing_options,
        satisfied=satisfied,
    )
