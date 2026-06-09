from __future__ import annotations

"""Intent classification for Q-VKG query pipeline.

Three-tier classifier (in priority order):
  1. Benchmark question_types metadata → deterministic mapping (zero cost)
  2. SigLIP zero-shot embedding similarity vs intent prototype sentences
  3. Minimal unambiguous keyword fallback (closed-set tokens only)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tier 1: benchmark question_type → intent mapping
# ---------------------------------------------------------------------------

QUESTION_TYPE_TO_INTENT: Dict[str, List[str]] = {
    # LVBench categories
    "key information retrieval":  ["SEMANTIC", "TEMPORAL"],
    "entity recognition":         ["IDENTITY", "SEMANTIC"],
    "event understanding":        ["CAUSAL", "TEMPORAL", "STATE"],
    "reasoning":                  ["CAUSAL", "INTENTIONAL"],
    "temporal grounding":         ["TEMPORAL"],
    "summarization":              ["SUMMARY"],
    # LongVideoBench categories
    "subtitle-based retrieval":   ["SEMANTIC", "TEMPORAL"],
    "visual-related reasoning":   ["CAUSAL", "SPATIAL", "SEMANTIC"],
    "temporal understanding":     ["TEMPORAL", "STATE"],
    "action recognition":         ["SEMANTIC", "STATE"],
    "character understanding":    ["IDENTITY", "EMOTIONAL", "INTENTIONAL"],
    "causal reasoning":           ["CAUSAL"],
    "counterfactual reasoning":   ["COUNTERFACT", "CAUSAL"],
    "emotional reasoning":        ["EMOTIONAL"],
    "intentional reasoning":      ["INTENTIONAL"],
    "prospective reasoning":      ["PROSPECTIVE"],
    "spatial understanding":      ["SPATIAL"],
    "multi-step reasoning":       ["CAUSAL", "TEMPORAL", "INTENTIONAL"],
    "common sense reasoning":     ["CAUSAL", "SEMANTIC"],
    # Generic catch-alls
    "video understanding":        ["SUMMARY", "SEMANTIC"],
    "scene understanding":        ["SEMANTIC", "SPATIAL"],
    "object recognition":         ["SEMANTIC", "IDENTITY"],
}


def _classify_by_question_types(question_types: List[str]) -> List[str]:
    seen: set = set()
    result: List[str] = []
    for qt in question_types:
        qt_norm = qt.strip().strip("'[]").lower()
        for intent in QUESTION_TYPE_TO_INTENT.get(qt_norm, []):
            if intent not in seen:
                seen.add(intent)
                result.append(intent)
    return result


# ---------------------------------------------------------------------------
# Tier 2: SigLIP zero-shot embedding similarity
# ---------------------------------------------------------------------------

INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "TEMPORAL": [
        "In what order did these events occur?",
        "Which happened first, A or B?",
        "How much time elapsed between the two scenes?",
        "At what timestamp does this event take place?",
        "What is the sequence of actions shown?",
    ],
    "CAUSAL": [
        "What caused this event to happen?",
        "What led to this outcome?",
        "Explain what triggered the change.",
        "What was the consequence of this action?",
        "Due to what circumstances did this occur?",
    ],
    "SPATIAL": [
        "Where is this object located in the scene?",
        "What is to the left of the person?",
        "Describe the position of the items on the table.",
        "Is the building near or far from the camera?",
        "Which room or setting does this take place in?",
    ],
    "IDENTITY": [
        "Who is the person wearing the red jacket?",
        "Which character appears in both scenes?",
        "Identify the individual shown here.",
        "Whose face appears at this moment?",
        "Name the person speaking in this clip.",
    ],
    "STATE": [
        "What was the state of the object before and after?",
        "When did the light switch from off to on?",
        "How did the situation change over time?",
        "Describe the transition that occurred.",
        "At what point did the machine turn off?",
    ],
    "SEMANTIC": [
        "What type of object is shown?",
        "What category does this scene belong to?",
        "Classify what is happening here.",
        "What kind of activity is being performed?",
        "What is the nature of the item on screen?",
    ],
    "SUMMARY": [
        "Summarize the main events of this video.",
        "Describe overall what happens in the clip.",
        "What is this video primarily about?",
        "Give an overview of the story shown.",
        "What are the key things that occurred throughout?",
    ],
    "COUNTERFACT": [
        "What would have happened if the door had not opened?",
        "Suppose the character had made a different choice — what then?",
        "Had this not occurred, what might the outcome be?",
        "If the weather had been different, how would the scene change?",
        "What would be the result if this step were skipped?",
    ],
    "EMOTIONAL": [
        "How does the character feel in this scene?",
        "What emotion is the person expressing?",
        "Describe the mood of the individual shown.",
        "Is the character happy, sad, or anxious here?",
        "What emotional reaction does the event trigger?",
    ],
    "INTENTIONAL": [
        "Why did the character decide to leave?",
        "What was the person's goal in doing this?",
        "What motivated this action?",
        "What did the character intend to achieve?",
        "What was the purpose behind this behavior?",
    ],
    "PROSPECTIVE": [
        "What will most likely happen next?",
        "Based on this scene, predict the following events.",
        "What comes after this moment in the video?",
        "What is about to occur given the current situation?",
        "Following this event, what do you expect to see?",
    ],
}

_EMBEDDING_THRESHOLD: float = 0.25
_PROTO_CACHE: Dict[int, Dict[str, np.ndarray]] = {}  # keyed by id(siglip_encoder)


def _build_proto_cache(siglip_encoder) -> Dict[str, np.ndarray]:
    """Encode all prototype sentences in one batched GPU call and cache mean embeddings."""
    all_texts: List[str] = []
    intent_slices: Dict[str, Tuple[int, int]] = {}
    for intent, sents in INTENT_PROTOTYPES.items():
        s = len(all_texts)
        all_texts.extend(sents)
        intent_slices[intent] = (s, s + len(sents))

    all_embs = siglip_encoder.encode_text(all_texts).astype(np.float32)  # (55, D)

    cache: Dict[str, np.ndarray] = {}
    for intent, (s, e) in intent_slices.items():
        mean_emb = all_embs[s:e].mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        cache[intent] = mean_emb / norm if norm > 0 else mean_emb
    return cache


def _classify_by_embedding(
    question: str,
    siglip_encoder,
    threshold: float = _EMBEDDING_THRESHOLD,
) -> List[str]:
    key = id(siglip_encoder)
    if key not in _PROTO_CACHE:
        _PROTO_CACHE[key] = _build_proto_cache(siglip_encoder)
    protos = _PROTO_CACHE[key]

    q_emb = siglip_encoder.encode_text([question]).astype(np.float32)[0]
    norm = np.linalg.norm(q_emb)
    if norm > 0:
        q_emb = q_emb / norm

    scores = {intent: float(np.dot(q_emb, proto)) for intent, proto in protos.items()}
    matched = sorted(
        [(intent, score) for intent, score in scores.items() if score >= threshold],
        key=lambda x: -x[1],
    )
    return [intent for intent, _ in matched]


# ---------------------------------------------------------------------------
# Tier 3: minimal unambiguous keyword fallback
# ---------------------------------------------------------------------------

# Only tokens/phrases that cannot appear in another intent's correct questions
_UNAMBIGUOUS_KEYWORDS: List[Tuple[str, str]] = [
    ("who ",      "IDENTITY"),
    ("whose ",    "IDENTITY"),
    ("where ",    "SPATIAL"),
    ("what if ",  "COUNTERFACT"),
    ("had not ",  "COUNTERFACT"),
    ("summarize", "SUMMARY"),
    ("overview",  "SUMMARY"),
]


def _classify_by_keywords(question: str) -> List[str]:
    q = question.lower() + " "  # trailing space prevents prefix false matches
    seen: set = set()
    result: List[str] = []
    for kw, intent in _UNAMBIGUOUS_KEYWORDS:
        if kw in q and intent not in seen:
            seen.add(intent)
            result.append(intent)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(
    question: str,
    siglip_encoder=None,
    question_types: Optional[List[str]] = None,
) -> List[str]:
    """Classify the retrieval intent(s) of a question.

    Returns a list of intent strings ordered by confidence. Falls back to
    ["SEMANTIC"] if nothing fires.
    """
    # Tier 1: benchmark metadata (deterministic, zero inference cost)
    if question_types:
        intents = _classify_by_question_types(question_types)
        if intents:
            return intents

    # Tier 2: SigLIP zero-shot embedding similarity
    if siglip_encoder is not None:
        intents = _classify_by_embedding(question, siglip_encoder)
        if intents:
            return intents

    # Tier 3: unambiguous keyword fallback
    intents = _classify_by_keywords(question)
    return intents or ["SEMANTIC"]


# ---------------------------------------------------------------------------
# Time-reference helpers (unchanged)
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> Optional[float]:
    """Parse 'MM:SS', 'HH:MM:SS', or plain seconds string → float seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        return None
    return None


def parse_time_reference(time_ref: str) -> Optional[Tuple[float, float]]:
    """Parse 'MM:SS-MM:SS' or 'HH:MM:SS-HH:MM:SS' → (t_start_secs, t_end_secs).

    Handles pinpoint (both sides equal) and reversed ranges.
    Returns None if unparseable.
    """
    if not time_ref or not time_ref.strip():
        return None
    raw = time_ref.strip()
    parts = raw.split("-")
    if len(parts) == 2:
        t0 = _parse_ts(parts[0])
        t1 = _parse_ts(parts[1])
    elif len(parts) == 4:
        t0 = _parse_ts(":".join(parts[:2]))
        t1 = _parse_ts(":".join(parts[2:]))
    elif len(parts) == 3:
        t0 = _parse_ts(":".join(parts[:2]))
        t1 = _parse_ts(parts[2])
        if t0 is None or t1 is None:
            t0 = _parse_ts(parts[0])
            t1 = _parse_ts(":".join(parts[1:]))
    else:
        return None

    if t0 is None or t1 is None:
        return None
    return (min(t0, t1), max(t0, t1))
