from __future__ import annotations

"""Scene extraction: multi-frame per-scene batch vLLM calls."""

import json
from dataclasses import dataclass, field
from typing import Dict, List

from .frame_store import FrameStore
from .schema import Scene
from .vllm_client import OFFLINE_SAMPLING, SCENE_SYSTEM_PROMPT


@dataclass
class SceneData:
    scene_id:          str
    time_range:        tuple
    scene_label:       str = ""
    objects:           List[dict] = field(default_factory=list)
    actions:           List[dict] = field(default_factory=list)
    spatial_relations: List[dict] = field(default_factory=list)
    characters:        List[dict] = field(default_factory=list)
    ocr_text:          List[dict] = field(default_factory=list)
    state_changes:     List[dict] = field(default_factory=list)
    scene_mood:        str = "neutral"


def build_scene_extraction_requests(
    scenes: List[Scene],
    frame_store: FrameStore,
) -> List[dict]:
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
            "scene_id":        scene.id,
            "scene_time_range": (scene.t_start, scene.t_end),
            "messages": [
                {"role": "system", "content": SCENE_SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
        })
    return requests


def run_scene_extraction(
    scenes: List[Scene],
    frame_store: FrameStore,
    llm,
) -> Dict[str, SceneData]:
    if not scenes:
        return {}

    requests = build_scene_extraction_requests(scenes, frame_store)
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
            state_changes=data.get("state_changes", []),
            scene_mood=data.get("scene_mood", "neutral"),
        )

    return results
