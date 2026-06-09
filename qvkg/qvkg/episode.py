from __future__ import annotations

"""Episode segmentation: single LLM call on scene labels."""

import json
from typing import Dict, List

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from .extraction import SceneData
from .schema import Episode, Scene
from .vllm_client import EPISODE_SCHEMA, EPISODE_SYSTEM_PROMPT


def segment_episodes(
    scenes: List[Scene],
    scene_data: Dict[str, SceneData],
    llm,
) -> List[Episode]:
    if not scenes:
        return []

    # scene_data keys are window IDs (e.g., "scene_0007_w00"); find the first
    # window whose parent_scene_id matches each scene to get its label.
    def _scene_label(s) -> str:
        for sd in scene_data.values():
            if sd.parent_scene_id == s.id:
                return sd.scene_label or "unknown"
        return "unknown"

    scene_descriptions = [
        f"Scene {i} [{s.t_start:.0f}s-{s.t_end:.0f}s]: {_scene_label(s)}"
        for i, s in enumerate(scenes)
    ]

    sampling = SamplingParams(
        temperature=0,
        max_tokens=2048,
        structured_outputs=StructuredOutputsParams(json=EPISODE_SCHEMA),
    )

    outputs = llm.chat(
        messages=[[
            {"role": "system", "content": EPISODE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Video has {len(scenes)} scenes. Group into narrative episodes:\n\n"
                + "\n".join(scene_descriptions)
            )},
        ]],
        sampling_params=sampling,
    )

    try:
        episode_data = json.loads(outputs[0].outputs[0].text)
    except (json.JSONDecodeError, IndexError):
        # Fallback: one episode covering everything
        episode_data = [{
            "label": "Full Video",
            "start_scene_idx": 0,
            "end_scene_idx": len(scenes) - 1,
            "narrative_role": "other",
            "summary": "Full video content",
        }]

    episodes = []
    for ep in episode_data:
        s_idx = max(0, ep.get("start_scene_idx", 0))
        e_idx = min(len(scenes) - 1, ep.get("end_scene_idx", len(scenes) - 1))
        ep_scenes = scenes[s_idx:e_idx + 1]
        if not ep_scenes:
            continue
        episodes.append(Episode(
            id=f"ep_{len(episodes):03d}",
            label=ep.get("label", f"Episode {len(episodes)}"),
            t_start=ep_scenes[0].t_start,
            t_end=ep_scenes[-1].t_end,
            narrative_role=ep.get("narrative_role", "other"),
            summary=ep.get("summary", ""),
            scenes=ep_scenes,
        ))

    return episodes
