from __future__ import annotations

"""Subtitle-track ingestion: parse .srt / .vtt files into timestamped segments.

The VKG normally reconstructs dialogue from Whisper ASR. When an authoritative
subtitle track is available it is preferable for *text-referred* questions
(exact wording, on-screen-only captions, precise timing). This module parses
SRT and WebVTT into a uniform list of segments that the builder turns into
SpeechNodes tagged ``source="subtitle"``.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional

# 00:01:02,500  (SRT) or 00:01:02.500 (VTT) → seconds
_TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")
# A cue's time line: "<start> --> <end>" (VTT may append positioning settings)
_ARROW_RE = re.compile(r"(.+?)\s*-->\s*([0-9:.,]+)")
# Inline tags to strip: <i>, <b>, <c.colorE5E5E5>, {\an8}, etc.
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]*\}")


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


def _parse_ts(ts: str) -> Optional[float]:
    m = _TS_RE.search(ts)
    if not m:
        return None
    h, mm, ss, frac = m.groups()
    frac = frac.ljust(3, "0")[:3]  # normalise to milliseconds
    return int(h) * 3600 + int(mm) * 60 + int(ss) + int(frac) / 1000.0


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ")
    # Drop speaker-label prefixes like "JERRY:" that aren't part of the caption
    return re.sub(r"\s+", " ", text).strip()


def parse_subtitle_file(path: str) -> List[SubtitleSegment]:
    """Parse an .srt or .vtt file into a list of SubtitleSegment (seconds).

    Format is auto-detected by content (both are cue blocks separated by blank
    lines with a ``start --> end`` line). Returns [] on any failure rather than
    raising, so a missing/corrupt subtitle never breaks a build.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            raw = f.read()
    except OSError:
        return []

    # Normalise line endings and split into cue blocks on blank lines.
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n"))
    segments: List[SubtitleSegment] = []

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        # Locate the timing line (skip optional numeric index / "WEBVTT" header).
        arrow_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if arrow_idx is None:
            continue
        m = _ARROW_RE.search(lines[arrow_idx])
        if not m:
            continue
        start = _parse_ts(m.group(1))
        end = _parse_ts(m.group(2))
        if start is None or end is None or end < start:
            continue
        text = _clean(" ".join(lines[arrow_idx + 1:]))
        if text:
            segments.append(SubtitleSegment(start=start, end=end, text=text))

    return segments


def discover_subtitle_path(video_path: str) -> Optional[str]:
    """Look for a sibling subtitle file next to the video (foo.mp4 → foo.srt/.vtt)."""
    base, _ = os.path.splitext(video_path)
    for ext in (".srt", ".vtt", ".en.srt", ".en.vtt"):
        cand = base + ext
        if os.path.isfile(cand):
            return cand
    return None
