from __future__ import annotations

"""Scene extraction: multi-frame per-scene batch vLLM calls."""

import json
from dataclasses import dataclass, field
from typing import Dict, List

from .frame_store import FrameStore
from .schema import Scene
from .vllm_client import OFFLINE_SAMPLING, build_scene_system_prompt


@dataclass
class SceneData:
    scene_id:           str
    time_range:         tuple
    scene_label:        str = ""
    objects:            List[dict] = field(default_factory=list)
    actions:            List[dict] = field(default_factory=list)
    spatial_relations:  List[dict] = field(default_factory=list)
    characters:         List[dict] = field(default_factory=list)
    ocr_text:           List[dict] = field(default_factory=list)
    ocr_semantics:      List[dict] = field(default_factory=list)
    state_changes:      List[dict] = field(default_factory=list)
    scene_mood:         str = "neutral"
    current_speaker:    str = ""
    speaker_on_screen:  bool = False
    narrative_function: str = "action"


def build_scene_extraction_requests(
    scenes: List[Scene],
    frame_store: FrameStore,
    system_prompt: str = "",
) -> List[dict]:
    if not system_prompt:
        system_prompt = build_scene_system_prompt()
    requests = []
    for scene in scenes:
        content = []
        for frame in scene.keyframes:
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

        requests.append({
            "scene_id":         scene.id,
            "scene_time_range": (scene.t_start, scene.t_end),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": content},
            ],
        })
    return requests


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
    outputs = llm.chat(
        messages=[r["messages"] for r in requests],
        sampling_params=OFFLINE_SAMPLING,
        use_tqdm=True,
    )

    results: Dict[str, SceneData] = {}
    for req, out in zip(requests, outputs):
        try:
            raw = out.outputs[0].text
            data = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            data = {}

        results[req["scene_id"]] = SceneData(
            scene_id=req["scene_id"],
            time_range=req["scene_time_range"],
            scene_label=data.get("scene_label", ""),
            objects=data.get("objects", []),
            actions=data.get("actions", []),
            spatial_relations=data.get("spatial_relations", []),
            characters=data.get("characters", []),
            ocr_text=data.get("ocr_text", []),
            ocr_semantics=data.get("ocr_semantics", []),
            state_changes=data.get("state_changes", []),
            scene_mood=data.get("scene_mood", "neutral"),
            current_speaker=data.get("current_speaker", "unknown"),
            speaker_on_screen=bool(data.get("speaker_on_screen", False)),
            narrative_function=data.get("narrative_function", "action"),
        )

    return results
