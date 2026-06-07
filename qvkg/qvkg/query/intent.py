from __future__ import annotations

from typing import List, Optional, Tuple

INTENT_PATTERNS = {
    "TEMPORAL":    ["when", "before", "after", "first", "then", "sequence",
                    "order", "timeline", "next", "previous", "how long"],
    "CAUSAL":      ["why", "cause", "reason", "because", "lead to", "result",
                    "how did", "what caused", "explain why"],
    "SPATIAL":     ["where", "location", "position", "left", "right", "near",
                    "behind", "in front", "distance", "close to"],
    "IDENTITY":    ["who", "which person", "same person", "character", "identify",
                    "name", "describe the person"],
    "STATE":       ["when did", "change", "turn on", "turn off", "become", "state",
                    "before it was", "after it became"],
    "SEMANTIC":    ["what kind", "type", "category", "classify", "what is"],
    "SUMMARY":     ["summarize", "describe", "happened", "overall", "story",
                    "what is this video", "tell me about"],
    "COUNTERFACT": ["if", "would", "could", "what if", "hypothetically",
                    "had not", "instead"],
}


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
    # Split on '-' but be careful: timestamps like '01:30-02:00'
    # We split on the last '-' that follows a ':' group
    # Strategy: split on '-', then pair up parts
    raw = time_ref.strip()
    # Find the separator: '-' between two timestamp parts
    # Timestamps contain ':', so the separator '-' won't be inside a part
    # Split on '-' and re-join parts by twos
    parts = raw.split("-")
    if len(parts) == 2:
        t0 = _parse_ts(parts[0])
        t1 = _parse_ts(parts[1])
    elif len(parts) == 4:
        # HH:MM:SS-HH:MM:SS splits into 4 on '-'
        t0 = _parse_ts(":".join(parts[:2]))
        t1 = _parse_ts(":".join(parts[2:]))
    elif len(parts) == 3:
        # Could be HH:MM:SS-MM:SS or MM:SS-HH:MM:SS
        # Try both splits
        t0 = _parse_ts(":".join(parts[:2]))
        t1 = _parse_ts(parts[2])
        if t0 is None or t1 is None:
            t0 = _parse_ts(parts[0])
            t1 = _parse_ts(":".join(parts[1:]))
    else:
        return None

    if t0 is None or t1 is None:
        return None
    # Swap if reversed (some LVBench entries have end < start)
    return (min(t0, t1), max(t0, t1))


def classify_intent(question: str) -> List[str]:
    q = question.lower()
    matched = [intent for intent, kws in INTENT_PATTERNS.items()
               if any(kw in q for kw in kws)]
    return matched or ["SEMANTIC"]
