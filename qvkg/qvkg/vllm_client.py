from __future__ import annotations

"""vLLM client factory, sampling params, JSON schemas, and system prompts."""

import re
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ---------------------------------------------------------------------------
# JSON schemas for guided decoding
# ---------------------------------------------------------------------------

SCENE_EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["scene_label", "objects", "actions", "spatial_relations",
                 "characters", "ocr_text", "state_changes", "scene_mood"],
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step analysis of what is happening in this scene before providing the structured output"
        },
        "scene_label": {
            "type": "string",
            "description": "One concise label, e.g. 'kitchen cooking', 'street argument'"
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "bbox_norm", "confidence"],
                "properties": {
                    "label":      {"type": "string"},
                    "bbox_norm":  {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] normalized 0-1"
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "state":      {"type": "string",
                                   "description": "current state, e.g. 'on', 'open', 'broken'"}
                }
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "actor"],
                "properties": {
                    "description": {"type": "string"},
                    "actor":       {"type": "string",
                                    "description": "character description or 'unknown'"},
                    "object":      {"type": "string",
                                    "description": "object being acted upon, if any"},
                    "confidence":  {"type": "number"}
                }
            }
        },
        "spatial_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subject", "relation", "object"],
                "properties": {
                    "subject":  {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["left_of", "right_of", "above", "below",
                                 "in_front_of", "behind", "near",
                                 "contains", "overlaps"]
                    },
                    "object":   {"type": "string"}
                }
            }
        },
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "bbox_norm"],
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Distinctive appearance: gender, age, clothing, hair"
                    },
                    "bbox_norm":   {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4
                    },
                    "emotion": {"type": "string"},
                    "action":  {"type": "string"}
                }
            }
        },
        "ocr_text": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":       {"type": "string"},
                    "bbox_norm":  {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4
                    },
                    "confidence": {"type": "number"}
                }
            }
        },
        "state_changes": {
            "type": "array",
            "description": "Transitions visible across the provided frames",
            "items": {
                "type": "object",
                "required": ["entity", "from_state", "to_state", "approx_timestamp"],
                "properties": {
                    "entity":           {"type": "string"},
                    "from_state":       {"type": "string"},
                    "to_state":         {"type": "string"},
                    "approx_timestamp": {
                        "type": "number",
                        "description": "seconds into video"
                    }
                }
            }
        },
        "scene_mood": {
            "type": "string",
            "enum": ["tense", "calm", "joyful", "sad", "urgent",
                     "neutral", "comedic", "dramatic"]
        }
    }
}

CAUSAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["cause", "effect", "relation", "confidence", "reasoning"],
        "properties": {
            "cause":      {"type": "string",
                           "description": "Exact event description from the timeline"},
            "effect":     {"type": "string",
                           "description": "Exact event description from the timeline"},
            "relation":   {"type": "string",
                           "enum": ["CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"]},
            "confidence": {"type": "number", "minimum": 0.6, "maximum": 1.0},
            "reasoning":  {"type": "string",
                           "description": "Step-by-step reasoning explaining the causal link grounded in visual evidence"}
        }
    }
}

EPISODE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["label", "start_scene_idx", "end_scene_idx", "narrative_role"],
        "properties": {
            "label":           {"type": "string"},
            "start_scene_idx": {"type": "integer"},
            "end_scene_idx":   {"type": "integer"},
            "narrative_role": {
                "type": "string",
                "enum": ["introduction", "rising_action", "climax",
                         "falling_action", "resolution", "subplot",
                         "transition", "flashback", "other"]
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence episode summary"
            }
        }
    }
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SCENE_SYSTEM_PROMPT = (
    "You are a video analysis system. You receive multiple frames from a single "
    "video scene in chronological order. Each frame is labeled with its timestamp. "
    "Extract structured information about the scene as JSON.\n\n"
    "First, think step-by-step about what is happening in this scene — identify the "
    "setting, the people present, their actions, object positions, and any visible "
    "changes. Write your reasoning in the 'reasoning' field of the output.\n\n"
    "Be precise about spatial positions (use normalized coordinates 0-1). "
    "For characters, provide distinctive appearance descriptions to enable "
    "cross-scene re-identification (e.g., 'young woman, black jacket, short brown "
    "hair, glasses'). For state changes, note the approximate timestamp when the "
    "transition occurs. Only report observations you can directly verify in the frames."
)

CAUSAL_SYSTEM_PROMPT = (
    "You are analyzing a video episode to identify causal relationships between "
    "events. You have access to:\n"
    "1. Keyframe images from this episode (in chronological order)\n"
    "2. A timeline of extracted events with timestamps\n\n"
    "First, think step-by-step about the narrative flow: what events precede others, "
    "what visual or contextual evidence suggests a causal link. Then identify causal "
    "links: what caused what, what enabled what, what motivated whom. "
    "Only report high-confidence links (>0.6). Ground each link in specific "
    "observations from the frames or timeline. Return JSON."
)

EPISODE_SYSTEM_PROMPT = (
    "You are a narrative analyst. You receive an ordered list of scene descriptions "
    "from a video. Group them into 10-30 narrative episodes — coherent story segments "
    "with a clear beginning and end.\n\n"
    "Each episode should represent a distinct narrative beat: an introduction, a "
    "conflict, a resolution, a transition, etc. Return a JSON array of episodes."
)

# ---------------------------------------------------------------------------
# Sampling params
# ---------------------------------------------------------------------------

OFFLINE_SAMPLING = SamplingParams(
    temperature=0,
    max_tokens=2048,
    structured_outputs=StructuredOutputsParams(json=SCENE_EXTRACTION_SCHEMA),
)

CAUSAL_SAMPLING = SamplingParams(
    temperature=0.1,
    max_tokens=1024,
    structured_outputs=StructuredOutputsParams(json=CAUSAL_SCHEMA),
)

QA_SAMPLING = SamplingParams(
    temperature=0,
    max_tokens=2048,
)

# Multiple-choice: free-form reasoning + answer letter extraction.
# The model thinks step-by-step then outputs a single answer letter.
MCQ_REASONING_SAMPLING = SamplingParams(
    temperature=0,
    max_tokens=2048,
    # No structured_outputs — let the model reason freely
)

MCQ_SYSTEM_PROMPT = (
    "You are answering a multiple choice question about a video. "
    "Study the provided frames and knowledge context carefully. "
    "First, think step-by-step about the evidence from the frames and knowledge. "
    "Consider each choice in turn. Then output ONLY the letter of the correct "
    "answer: A, B, C, or D. The answer letter must be on its own line at the end."
)


def extract_mcq_answer(text: str) -> str:
    """Extract answer letter (A/B/C/D) from free-form model output.

    Strips thinking tags (<think>...</think>) if present, then searches for
    a standalone answer letter on its own line or at end of text.
    Falls back to scanning the last few characters.
    """
    # Strip Qwen3 thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Look for a line that is just a single letter A-D
    for line in text.split("\n"):
        line = line.strip()
        if re.fullmatch(r"[A-D]", line):
            return line

    # Fallback: look for last standalone letter in the text
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]

    # Last resort: return the raw text trimmed
    return text.strip()[:20]

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def build_llm(
    model: str = _DEFAULT_MODEL,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 65536,
    max_images_per_prompt: int = 10,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> LLM:
    return LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_num_seqs=64,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": max_images_per_prompt},
        mm_processor_kwargs={
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    )


def build_siglip_encoder(model_name: str = "google/siglip-so400m-patch14-384"):
    """Load SigLIP encoder for embedding nodes and queries."""
    from transformers import AutoProcessor, SiglipModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return SigLIPEncoder(model, processor, device)


class SigLIPEncoder:
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    @property
    def embedding_dim(self) -> int:
        return self.model.config.vision_config.hidden_size

    def encode_text(self, texts: list) -> "np.ndarray":
        import numpy as np
        import torch

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_image(self, image) -> "np.ndarray":
        import numpy as np
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()[0]

    def encode_images_batch(self, images: list) -> "np.ndarray":
        import numpy as np
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()
