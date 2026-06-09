from __future__ import annotations

"""Scene extraction: multi-frame per-scene batch vLLM calls.

For scenes with many keyframes, frames are split into rolling windows of
WINDOW_SIZE (default 10) with WINDOW_STRIDE overlap. Each window is a
separate vLLM request; results are merged back into one SceneData per scene.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .frame_store import FrameStore
from .schema import Scene
from .vllm_client import OFFLINE_SAMPLING, build_scene_system_prompt

WINDOW_SIZE   = 10  # frames per VLM call
WINDOW_STRIDE = 8   # step between windows (2-frame overlap for context)


@dataclass
class SceneData:
    scene_id:           str
    time_range:         tuple
    scene_label:        str = ""
    location:           str = "unknown"
    objects:            List[dict] = field(default_factory=list)
    actions:            List[dict] = field(default_factory=list)
    spatial_relations:  List[dict] = field(default_factory=list)
    characters:         List[dict] = field(default_factory=list)
    ocr_text:           List[dict] = field(default_factory=list)
    ocr_semantics:      List[dict] = field(default_factory=list)
    state_changes:      List[dict] = field(default_factory=list)
    mentioned_entities: List[dict] = field(default_factory=list)
    scene_mood:         str = "neutral"
    current_speaker:    str = ""
    speaker_on_screen:  bool = False
    narrative_function: str = "action"
    # which original scene boundary this window belongs to (for episode linkage)
    parent_scene_id:    str = ""


def _build_window_content(frames, frame_store) -> List[dict]:
    """Build the image+text content list for one window of frames."""
    content = []
    for frame in frames:
        try:
            b64_url = frame_store.get_b64_url(frame.id)
            content.append({"type": "image_url",
                             "image_url": {"url": b64_url}})
        except (KeyError, FileNotFoundError):
            pass
        content.append({"type": "text",
                         "text": f"[t={frame.timestamp:.1f}s]"})
    content.append({"type": "text",
                     "text": "Extract structured information from these scene frames as JSON."})
    return content


def build_scene_extraction_requests(
    scenes: List[Scene],
    frame_store: FrameStore,
    system_prompt: str = "",
) -> List[dict]:
    """Build one request per rolling window per scene.

    Each request carries scene_id and window_idx so results can be merged.
    """
    if not system_prompt:
        system_prompt = build_scene_system_prompt()

    requests = []
    for scene in scenes:
        frames = scene.keyframes
        if not frames:
            continue

        # Produce rolling windows; single window if ≤ WINDOW_SIZE frames
        starts = list(range(0, max(1, len(frames) - WINDOW_SIZE + 1), WINDOW_STRIDE))
        if not starts:
            starts = [0]

        for win_idx, start in enumerate(starts):
            window_frames = frames[start:start + WINDOW_SIZE]
            content = _build_window_content(window_frames, frame_store)
            # Compute precise time range for this window from its actual frames
            win_t_start = window_frames[0].timestamp  if window_frames else scene.t_start
            win_t_end   = window_frames[-1].timestamp if window_frames else scene.t_end
            requests.append({
                "scene_id":         scene.id,
                "scene_time_range": (scene.t_start, scene.t_end),
                "window_idx":       win_idx,
                "window_start":     start,
                "window_t_start":   win_t_start,
                "window_t_end":     win_t_end,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": content},
                ],
            })

    return requests


def _merge_scene_results(window_results: List[dict], scene_id: str,
                          time_range: Tuple) -> SceneData:
    """Merge VLM outputs from multiple rolling windows into one SceneData.

    List fields (objects, actions, etc.) are concatenated and de-duplicated
    by label. Scalar fields (scene_label, mood, etc.) come from window 0.
    """
    if not window_results:
        return SceneData(scene_id=scene_id, time_range=time_range)

    # Scalar fields from first window (highest quality — most context frames)
    first = window_results[0]
    sd = SceneData(
        scene_id=scene_id,
        time_range=time_range,
        scene_label=first.get("scene_label", ""),
        location=first.get("location", "unknown") or "unknown",
        scene_mood=first.get("scene_mood", "neutral"),
        current_speaker=first.get("current_speaker", "unknown"),
        speaker_on_screen=bool(first.get("speaker_on_screen", False)),
        narrative_function=first.get("narrative_function", "action"),
    )

    # List fields: merge across all windows, de-dup by label
    list_fields = [
        "objects", "actions", "spatial_relations", "characters",
        "ocr_text", "ocr_semantics", "state_changes", "mentioned_entities",
    ]
    for field_name in list_fields:
        seen_labels: set = set()
        merged: List[dict] = []
        for win in window_results:
            for item in win.get(field_name, []):
                label = str(item.get("label", item.get("text", item.get("description", ""))))
                key = label.lower().strip()
                if key and key not in seen_labels:
                    seen_labels.add(key)
                    merged.append(item)
                elif not key:
                    merged.append(item)
        setattr(sd, field_name, merged)

    return sd


def run_scene_extraction(
    scenes: List[Scene],
    frame_store: FrameStore,
    llm,
    video_type: str = "",
    has_narrator: bool = False,
) -> Dict[str, SceneData]:
    if not scenes:
        return {}

    system_prompt = build_scene_system_prompt(video_type, has_narrator)
    requests = build_scene_extraction_requests(scenes, frame_store, system_prompt)

    if not requests:
        return {}

    outputs = llm.chat(
        messages=[r["messages"] for r in requests],
        sampling_params=OFFLINE_SAMPLING,
        use_tqdm=True,
    )

    # Parse each window into its own SceneData with a precise time range.
    # Keys are window-specific IDs (e.g., "scene_0007_w03") so each window
    # becomes a separate SceneNode in the graph — preserving temporal attribution.
    results: Dict[str, SceneData] = {}

    for req, out in zip(requests, outputs):
        parent_sid = req["scene_id"]
        win_idx    = req["window_idx"]
        win_id     = f"{parent_sid}_w{win_idx:02d}"

        # Compute precise time range from the actual frames in this window
        scene_t_start, scene_t_end = req["scene_time_range"]
        win_t_start = req.get("window_t_start", scene_t_start)
        win_t_end   = req.get("window_t_end",   scene_t_end)

        try:
            raw  = out.outputs[0].text
            data = json.loads(raw)
        except (json.JSONDecodeError, IndexError, AttributeError):
            data = {}

        sd = SceneData(
            scene_id=win_id,
            time_range=(win_t_start, win_t_end),
            parent_scene_id=parent_sid,
            scene_label=data.get("scene_label", ""),
            location=data.get("location", "unknown") or "unknown",
            scene_mood=data.get("scene_mood", "neutral"),
            current_speaker=data.get("current_speaker", ""),
            speaker_on_screen=bool(data.get("speaker_on_screen", False)),
            narrative_function=data.get("narrative_function", "action"),
            objects=data.get("objects", []),
            actions=data.get("actions", []),
            spatial_relations=data.get("spatial_relations", []),
            characters=data.get("characters", []),
            ocr_text=data.get("ocr_text", []),
            ocr_semantics=data.get("ocr_semantics", []),
            state_changes=data.get("state_changes", []),
            mentioned_entities=data.get("mentioned_entities", []),
        )
        results[win_id] = sd

    return results
