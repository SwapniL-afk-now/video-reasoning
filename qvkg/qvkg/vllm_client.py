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
    "required": ["scene_label", "location", "objects", "actions", "spatial_relations",
                 "characters", "ocr_text", "ocr_semantics", "state_changes",
                 "scene_mood", "current_speaker", "speaker_on_screen",
                 "narrative_function", "mentioned_entities"],
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step analysis of what is happening in this scene before providing the structured output"
        },
        "scene_label": {
            "type": "string",
            "description": "Descriptive 2-6 word label capturing the key activity, e.g. 'hosts eating ramen outside 711', 'chef grilling wagyu beef', 'aerial view of coastal town'"
        },
        "location": {
            "type": "string",
            "description": "The specific place or setting where this scene occurs, e.g. 'kitchen', 'Times Square', 'soccer stadium', 'living room', 'hospital corridor'. Use 'unknown' if the setting is unclear."
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "bbox_norm", "confidence", "attributes", "state"],
                "properties": {
                    "label":      {"type": "string",
                                   "description": "object name e.g. 'red sports car', 'wooden table', 'glass bottle'"},
                    "bbox_norm":  {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] normalized 0-1"
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "attributes": {"type": "array", "items": {"type": "string"},
                                   "description": "visual attributes e.g. red, wooden, large, shiny"},
                    "state":      {"type": "string",
                                   "description": "current state, e.g. 'on', 'open', 'broken', 'full', 'empty'"}
                }
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "actor"],
                "properties": {
                    "description":      {"type": "string"},
                    "actor":            {"type": "string",
                                         "description": "character description or 'unknown'"},
                    "object":           {"type": "string",
                                         "description": "object being acted upon, if any"},
                    "confidence":       {"type": "number"},
                    "approx_timestamp": {
                        "type": "number",
                        "description": "estimated seconds from video start when this action occurs, within the scene's time range"
                    }
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
            "description": "CRITICAL: Every visible person in this scene MUST be listed here",
            "items": {
                "type": "object",
                "required": ["description", "bbox_norm", "emotion"],
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Distinctive appearance: gender, age, clothing, hair, accessories"
                    },
                    "bbox_norm":   {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] normalized 0-1"
                    },
                    "emotion": {"type": "string",
                                "description": "emotional state of this person e.g. happy, sad, neutral, angry, surprised"},
                    "action":  {"type": "string",
                                "description": "what this person is doing e.g. walking, talking, eating, cooking"}
                }
            }
        },
        "ocr_text": {
            "type": "array",
            "description": "CRITICAL: Every piece of visible text in the scene",
            "items": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text":       {"type": "string",
                                   "description": "Exact text content, read carefully"},
                    "bbox_norm":  {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] normalized 0-1"
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
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
        },
        "current_speaker": {
            "type": "string",
            "description": (
                "Who is speaking during this scene. If a visible character is speaking, "
                "use their description matching the characters array (e.g., 'woman in red jacket'). "
                "If speech is heard but no speaker is visible (voice-over, off-screen commentary), "
                "use 'narrator'. If no speech or unclear, use 'unknown'."
            )
        },
        "speaker_on_screen": {
            "type": "boolean",
            "description": "Is the speaking person visible in any frame of this scene?"
        },
        "narrative_function": {
            "type": "string",
            "enum": ["action", "dialogue", "exposition", "transition",
                     "reaction", "commentary", "highlight", "interview"],
            "description": "The narrative role this scene plays in the overall video structure"
        },
        "ocr_semantics": {
            "type": "array",
            "description": "Semantic interpretation of each OCR text item — parallel to ocr_text",
            "items": {
                "type": "object",
                "required": ["text", "semantic_type", "refers_to"],
                "properties": {
                    "text": {"type": "string", "description": "Exact text matching an entry in ocr_text"},
                    "semantic_type": {
                        "type": "string",
                        "enum": ["price", "name", "score", "time", "title",
                                 "caption", "subtitle", "label", "brand",
                                 "location", "stat", "other"]
                    },
                    "refers_to": {
                        "type": "string",
                        "description": (
                            "What entity or concept does this text describe? "
                            "e.g. 'salmon sashimi dish on left plate', 'game clock', "
                            "'chapter heading', 'player wearing jersey #23'"
                        )
                    }
                }
            }
        },
        "mentioned_entities": {
            "type": "array",
            "description": "Entities explicitly named or referred to in dialogue/speech during this scene",
            "items": {
                "type": "object",
                "required": ["text", "refers_to_label"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact wording used in speech, e.g. 'the red car', 'John', 'that knife'"
                    },
                    "refers_to_label": {
                        "type": "string",
                        "description": "Label of the visible entity this refers to, matching an entry in objects or characters"
                    }
                }
            }
        }
    }
}

VIDEO_TYPE_SCHEMA = {
    "type": "object",
    "required": ["video_type", "has_narrator_voiceover", "has_multiple_speakers",
                 "dominant_language"],
    "properties": {
        "video_type": {
            "type": "string",
            "enum": ["movie", "sports", "documentary", "vlog", "tutorial",
                     "news", "educational", "entertainment", "other"]
        },
        "has_narrator_voiceover": {"type": "boolean"},
        "has_multiple_speakers":  {"type": "boolean"},
        "dominant_language":      {"type": "string"},
        "key_characteristics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "e.g. 'score overlays', 'cooking demonstrations', 'interview segments'"
        }
    }
}

ENTITY_RESOLUTION_SCHEMA = {
    "type": "array",
    "description": "One entry per unique real-world person identified across all scenes",
    "items": {
        "type": "object",
        "required": ["canonical_id", "canonical_name", "canonical_description",
                     "description_variants", "entity_type"],
        "properties": {
            "canonical_id": {
                "type": "string",
                "description": "Short stable identifier, e.g. 'host', 'player_23', 'detective_brown'"
            },
            "canonical_name": {
                "type": "string",
                "description": "Best known name or role, e.g. 'David', 'Goalkeeper #1', 'Narrator'"
            },
            "canonical_description": {
                "type": "string",
                "description": "Most complete single appearance description"
            },
            "description_variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All raw description strings from scenes that belong to this entity"
            },
            "entity_type": {
                "type": "string",
                "enum": ["main_character", "supporting_character", "background",
                         "narrator", "interviewer", "interviewee",
                         "athlete", "anchor", "host", "other"]
            }
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

_SCENE_BASE_PROMPT = (
    "You are a video analysis system. You receive multiple frames from a single "
    "video scene in chronological order. Each frame is labeled with its timestamp. "
    "Extract structured information about the scene as JSON.\n\n"
    "First, think step-by-step about what is happening in this scene — identify the "
    "setting, the people present, their actions, object positions, and any visible "
    "changes. Write your reasoning in the 'reasoning' field of the output.\n\n"
    "Be precise about spatial positions (use normalized coordinates 0-1). "
    "For characters, provide distinctive appearance descriptions to enable "
    "cross-scene re-identification (e.g., 'young woman, black jacket, short brown "
    "hair, glasses').\n\n"
    "CRITICAL — EMOTION ANALYSIS: For EVERY character you detect, you MUST analyze "
    "and report their emotional state based on facial expression, body language, "
    "and context. Possible emotions include: happy, sad, angry, surprised, neutral, "
    "excited, disappointed, nervous, relaxed, confused, hopeful, or describe other "
    "emotions you observe. Never leave emotion empty.\n\n"
    "CRITICAL — CHARACTER DETECTION: You MUST detect EVERY person visible in ANY frame "
    "of this scene, including people who are partially visible, in the background, "
    "or shown from behind. For EVERY person you see, you MUST:\n"
    "  1. Add them to the 'characters' array with full description (gender, age, "
    "clothing, hair, distinctive features), bbox, emotion, and action\n"
    "  2. ALSO add them to the 'objects' array with label like 'man', 'woman', 'chef'\n"
    "If NO people are visible in ANY frame, the 'characters' array should be empty. "
    "But if even one person appears, they MUST be included.\n\n"
    "CRITICAL — OCR TEXT EXTRACTION: You MUST read ALL visible text from EVERY frame. "
    "Do not skip any frame. For each frame, check for:\n"
    "  - Signs, posters, banners, billboards\n"
    "  - Product labels, prices, menus\n"
    "  - On-screen titles, subtitles, captions\n"
    "  - Social media comments, usernames, timestamps\n"
    "  - Phone numbers, URLs, brand names\n"
    "  - Any other text of any size or font\n\n"
    "Transcribe the EXACT text content — every character matters, including digits, "
    "symbols, and punctuation. If there are multiple text elements, list EACH ONE "
    "separately in the 'ocr_text' array with its approximate location. "
    "DO NOT SKIP any text. If text resolution is low, do your best to read it "
    "character by character.\n\n"
    "For state changes, note the approximate timestamp when the transition occurs. "
    "Only report observations you can directly verify in the frames.\n\n"
    "SPEAKER ATTRIBUTION: Set 'current_speaker' to the description of the visible "
    "person who is speaking (matching the characters array), 'narrator' if speech "
    "is heard but no speaker is visible on screen, or 'unknown' if silent/unclear. "
    "Set 'speaker_on_screen' accordingly. "
    "Set 'narrative_function' to the role this scene plays: action, dialogue, "
    "exposition, transition, reaction, commentary, highlight, or interview.\n\n"
    "OCR SEMANTICS: For each item in ocr_text, add a parallel entry in ocr_semantics "
    "describing what the text refers to (e.g., '¥980' refers to 'salmon dish price', "
    "'23:47' refers to 'game clock', 'CHAPTER 3' refers to 'chapter heading').\n\n"
    "LOCATION: Set 'location' to the specific setting name (e.g. 'kitchen', "
    "'hospital corridor', 'Times Square', 'soccer pitch'). Use 'unknown' only if truly "
    "indeterminate. This helps link scenes that occur in the same place.\n\n"
    "ACTION TIMESTAMPS: For each action, set 'approx_timestamp' to the estimated "
    "video time in seconds when it occurs. Use the frame timestamps shown to estimate.\n\n"
    "MENTIONED ENTITIES: In 'mentioned_entities', list every object or character that "
    "is explicitly named or referred to in speech/dialogue during this scene, with the "
    "matching label from the objects or characters array."
)

_TYPE_ADDENDA = {
    "sports": (
        "\nSPORTS VIDEO: Jersey numbers and team colors are primary player identity cues — "
        "always include them in character descriptions. OCR overlays contain scores, "
        "game clocks, and player name chyrons — extract them exactly with semantic_type "
        "'score', 'time', or 'name'. Note game events (goal, foul, tackle, serve) as actions. "
        "The commentator/announcer is typically off-screen; set current_speaker='narrator' "
        "when their voice is heard with no visible speaker."
    ),
    "movie": (
        "\nMOVIE/DRAMA: Focus on character emotional performance and precise dialogue "
        "attribution. Match current_speaker to the character description in the characters "
        "array — the speaking character is usually the one with the most expressive face or "
        "open mouth. Note scene blocking (who is in foreground vs. background). "
        "Identify characters by consistent appearance markers (costume, hairstyle)."
    ),
    "documentary": (
        "\nDOCUMENTARY: This video likely has narrator voice-over. When narration is heard "
        "with no visible speaker, set current_speaker='narrator' and speaker_on_screen=false. "
        "For interview segments (person talking directly to camera or interviewer), set "
        "narrative_function='interview'. Identify experts by on-screen text labels (chyrons) "
        "when present — extract them as ocr_text with semantic_type='name'."
    ),
    "vlog": (
        "\nVLOG/TUTORIAL: The main host is typically the primary on-screen speaker throughout. "
        "Extract food, product, and location names precisely in object labels. "
        "Price tags and menus are critical OCR targets — use semantic_type='price' for costs "
        "and refers_to the specific item. Capture the host's emotional reactions closely."
    ),
    "news": (
        "\nNEWS BROADCAST: Chyrons (lower-third graphics showing names/titles) are critical "
        "named entity information — extract them verbatim with semantic_type='name'. "
        "Identify on-screen anchors, field reporters, and interview subjects. "
        "Breaking news tickers should be extracted as ocr_text with semantic_type='caption'."
    ),
    "tutorial": (
        "\nTUTORIAL/EDUCATIONAL: Focus on what is being demonstrated. Extract all on-screen "
        "text (step numbers, labels, captions) with their semantic meaning. The instructor "
        "is typically the current_speaker when visible on screen."
    ),
}


def build_scene_system_prompt(
    video_type: str = "",
    has_narrator: bool = False,
) -> str:
    """Return the scene extraction system prompt conditioned on video type."""
    prompt = _SCENE_BASE_PROMPT
    addendum = _TYPE_ADDENDA.get(video_type, "")
    if addendum:
        prompt += addendum
    if has_narrator and video_type not in ("documentary", "sports"):
        prompt += (
            "\nIMPORTANT: This video contains narrator voice-over. When speech is heard "
            "but no speaker face is visible, always set current_speaker='narrator' and "
            "speaker_on_screen=false."
        )
    return prompt


# Keep the constant for backward compatibility — resolves to the generic prompt
SCENE_SYSTEM_PROMPT = build_scene_system_prompt()

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

VIDEO_TYPE_SYSTEM_PROMPT = (
    "You are a video content classifier. You receive a sample of frames from a video. "
    "Determine the video type, whether it has narrator voice-over, whether multiple "
    "speakers are present, the dominant spoken language, and key visual characteristics "
    "that should guide content extraction. Return JSON."
)

ENTITY_RESOLUTION_SYSTEM_PROMPT = (
    "You are resolving character identity across a long video. "
    "You will receive a list of character descriptions extracted from different scenes "
    "at different timestamps. Your task: identify which descriptions refer to the same "
    "real-world person and group them into canonical entities.\n\n"
    "Use ALL available cues:\n"
    "  - Physical features: face, hair, build, age\n"
    "  - Clothing and accessories (consistent across scenes)\n"
    "  - Jersey numbers and team colors (sports)\n"
    "  - On-screen name labels or chyrons\n"
    "  - Narrative role (host, interviewer, athlete, etc.)\n"
    "  - Temporal context (same person likely has consistent role)\n\n"
    "For sports: jersey numbers and team colors are the strongest identity cues. "
    "For interviews: distinguish interviewer (behind camera) from interviewee (facing camera). "
    "For documentaries: 'narrator' is a special entity for off-screen voice-over.\n\n"
    "Assign each entity a short canonical_id like 'host', 'player_23_red', "
    "'detective_brown', 'narrator', 'vendor_1'. "
    "Return a JSON array where each entry is one unique real-world person."
)

EPISODE_SYSTEM_PROMPT = (
    "You are a narrative analyst. You receive an ordered list of scene descriptions "
    "from a video. Group them into 10-30 narrative episodes — coherent story segments "
    "with a clear beginning and end.\n\n"
    "Each episode should represent a distinct narrative beat: an introduction, a "
    "conflict, a resolution, a transition, etc. Return a JSON array of episodes.\n\n"
    "CRITICAL: Each episode label MUST be a short descriptive phrase (4-8 words) that "
    "captures the specific topic or activity — for example 'Raw chicken tasting at "
    "Tokyo market' or 'Exploring Tsukiji seafood stalls'. Do NOT use generic labels "
    "like 'Episode 1' or 'Introduction'."
)

# ---------------------------------------------------------------------------
# Planner schema + prompt (two-stage QA)
# ---------------------------------------------------------------------------

PLANNER_SCHEMA = {
    "type": "object",
    "required": ["windows", "search_queries", "needs_frames"],
    "properties": {
        "windows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["t_start", "t_end"],
                "properties": {
                    "t_start": {"type": "number"},
                    "t_end":   {"type": "number"},
                },
            },
        },
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "needs_frames":   {"type": "boolean"},
        "reasoning":      {"type": "string"},
    },
}

PLANNER_SYSTEM_PROMPT = (
    "You are a retrieval planner for a video QA system. "
    "You will receive a question, its type, a time reference, and the video's episode structure. "
    "Output a JSON retrieval plan:\n\n"
    "- windows: time windows (seconds) to retrieve detailed nodes from. "
    "For pinpoint timestamp questions, include that timestamp window. "
    "For broad narrative questions, leave empty and rely on search_queries instead.\n"
    "- search_queries: 1-4 text queries to search the knowledge graph. "
    "Use specific terms from the question. For ordering questions (last/first), "
    "include the entity type so all occurrences can be found and compared.\n"
    "- needs_frames: true if the question requires visual identification at a specific moment "
    "(on-screen text, object appearance, action at a timestamp). "
    "False for narrative, ordering, causal, or summary questions.\n"
    "- reasoning: one sentence explaining your plan."
)

# ---------------------------------------------------------------------------
# Sampling params
# ---------------------------------------------------------------------------

PLANNER_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=512,
    structured_outputs=StructuredOutputsParams(json=PLANNER_SCHEMA),
)

VIDEO_TYPE_SAMPLING = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=512,
    structured_outputs=StructuredOutputsParams(json=VIDEO_TYPE_SCHEMA),
)

ENTITY_RESOLUTION_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=32768,
    structured_outputs=StructuredOutputsParams(json=ENTITY_RESOLUTION_SCHEMA),
)

OFFLINE_SAMPLING = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=8192,
    structured_outputs=StructuredOutputsParams(json=SCENE_EXTRACTION_SCHEMA),
)

CAUSAL_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=1024,
    structured_outputs=StructuredOutputsParams(json=CAUSAL_SCHEMA),
)

QA_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=32768,
)

# Multiple-choice: model reasons freely inside <think>…</think> then answers.
# max_tokens covers both the thinking trace and the final answer letter.
MCQ_REASONING_SAMPLING = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    max_tokens=8192,
)

# ---------------------------------------------------------------------------
# Walker (inference-time agentic graph traversal) sampling + prompts
# ---------------------------------------------------------------------------

# Answerer / final emission read-out. GREEDY: the walk's convergence signal
# (elasticity) is computed over greedy probes, so the emitted answer must come
# from the same deterministic distribution — a temp-1.0 final pass can flip a
# converged correct answer on sampling noise. Thinking budget kept at 8192.
WALKER_ANSWER_SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=8192,
)


# Internal convergence probe: GREEDY so the elasticity finite difference
# (answer(S_full) vs answer(S∖last_ring)) is deterministic — comparing two
# temperature>0 draws would flip on sampling noise, not on evidence. Thinking is
# kept (capped) so the probe still reasons; only the randomness is removed.
WALKER_PROBE_SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
)


def walker_controller_sampling(action_schema: dict) -> SamplingParams:
    """Greedy, schema-constrained single-action decode (≤256 tokens)."""
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(json=action_schema),
    )


WALKER_CONTROLLER_SYSTEM = (
    "You are navigating a video knowledge graph to answer a question. "
    "You see frames from the video, the current sub-graph, and missing evidence "
    "gaps. Be concise — pick ONE action that fills the largest gap fastest.\n\n"
    "Two search modes:\n"
    "- EXPAND(relation): follow graph edges from current frontier. "
    "Use when you know the evidence is nearby.\n"
    "- RECALL(query): semantic search over the entire graph (bypasses topology). "
    "Use when EXPAND finds nothing new or evidence may be anywhere.\n\n"
    "Actions:\n"
    "- EXPAND(relation): follow an edge family. "
    "relation ∈ {CAUSAL, ENTITY, SPEAKER, TEMPORAL, EMOTION, SIMILAR, CONTAINS}.\n"
    "- ZOOM(node_id): materialise frames at a node.\n"
    "- DISCRIMINATE(option): retrieve evidence for an MCQ option.\n"
    "- RECALL(query): semantic search (topology-free — reaches any node).\n"
    "- BUILD: extract fresh frames + infer missing edges at the question's "
    "location. Use when:\n"
    "  * Gaps persist despite EXPAND/RECALL (stuck)\n"
    "  * Causal edges are missing — BUILD infers them from existing scene labels\n"
    "  * Entity evidence is missing — BUILD extracts denser frames at the anchor\n"
    "- ANSWER(letter, cited_node_ids): answer with supporting node ids.\n"
    "- STOP_REQUEST: you believe you are done.\n\n"
    "Do NOT over-explore. Once you have enough evidence to pick an answer, "
    "ANSWER immediately. Output one JSON action."
)

WALKER_ANSWER_SYSTEM = (
    "You are a video analyst answering a multiple choice question. "
    "You are given frames and a knowledge sub-graph with explicit edges "
    "(causal chains, same-entity threads, speaker links).\n\n"
    "Think carefully and thoroughly inside your thinking block: examine each "
    "frame, review the timeline and characters, then evaluate each option "
    "(A, B, C, D) against the evidence — timestamps, visual details, "
    "dialogue, and OCR text.\n\n"
    "After your thinking, output ONE line containing ONLY the answer letter: "
    "A, B, C, or D."
)

# ---------------------------------------------------------------------------
# BUILD action — online KG construction prompts
# ---------------------------------------------------------------------------

BUILD_CAUSAL_SYSTEM = (
    "You are a causal reasoning engine for video event analysis. "
    "Given a timeline of timestamped events from a video, identify causal "
    "cause→effect relationships.\n\n"
    "Rules:\n"
    "- Only link events that have a clear causal relationship.\n"
    "- A cause must precede its effect in time.\n"
    "- Use CAUSES for direct cause-effect, ENABLES for precondition, "
    "PREVENTS for blocking, MOTIVATES for psychological drive.\n"
    "- Assign confidence (0.0–1.0) based on how certain the link is.\n"
    "- Provide brief reasoning for each link.\n\n"
    "Output a JSON array of objects with fields:\n"
    "cause (string), effect (string), relation_type (CAUSES|ENABLES|PREVENTS|MOTIVATES), "
    "confidence (float), reasoning (string)."
)

BUILD_CAUSAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["cause", "effect", "relation_type", "confidence", "reasoning"],
        "properties": {
            "cause": {"type": "string", "description": "Description of the cause event"},
            "effect": {"type": "string", "description": "Description of the effect event"},
            "relation_type": {
                "type": "string",
                "enum": ["CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"}
        }
    }
}

BUILD_CAUSAL_SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    structured_outputs=StructuredOutputsParams(json=BUILD_CAUSAL_SCHEMA),
)

MCQ_SYSTEM_PROMPT = (
    "You are a video analyst answering a multiple choice question. "
    "You are given frames from the video and structured knowledge extracted from it.\n\n"
    "Think carefully and thoroughly inside your thinking block: examine each frame, "
    "review the timeline and characters, then evaluate each option (A, B, C, D) "
    "against the evidence — timestamps, visual details, dialogue, and OCR text.\n\n"
    "After your thinking, output ONE line containing ONLY the answer letter: A, B, C, or D."
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


class LazyLLM:
    """Defers vLLM engine construction until the first real use.

    The engine reserves its full GPU budget the moment it is created. By
    deferring construction to the first attribute access (e.g. ``.chat``),
    SigLIP encoding, Whisper transcription and frame sampling all run with the
    GPU otherwise free — the engine spins up exactly when first needed.
    Transparent: any attribute/method access proxies to the real ``LLM``.
    """

    def __init__(self, **kwargs):
        # Store via __dict__ to avoid triggering __getattr__ during init.
        self.__dict__["_kwargs"] = kwargs
        self.__dict__["_llm"] = None

    def _ensure(self) -> LLM:
        if self.__dict__["_llm"] is None:
            print("  [vLLM] Initializing engine on first use...")
            self.__dict__["_llm"] = _construct_llm(**self.__dict__["_kwargs"])
        return self.__dict__["_llm"]

    @property
    def is_loaded(self) -> bool:
        return self.__dict__["_llm"] is not None

    def __getattr__(self, name):
        # Only called for attributes not found normally → proxy to real engine.
        return getattr(self._ensure(), name)


def _construct_llm(
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_images_per_prompt: int,
    min_pixels: int,
    max_pixels: int,
) -> LLM:
    return LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_num_seqs=256,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": max_images_per_prompt},
        mm_processor_kwargs={
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    )


def build_llm(
    model: str = _DEFAULT_MODEL,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 65536,
    max_images_per_prompt: int = 10,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
    lazy: bool = False,
):
    kwargs = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_images_per_prompt=max_images_per_prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if lazy:
        return LazyLLM(**kwargs)
    return _construct_llm(**kwargs)


def build_siglip_encoder(model_name: str = "google/siglip-so400m-patch14-384"):
    """Load SigLIP encoder for embedding nodes and queries."""
    from transformers import AutoProcessor, SiglipModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return SigLIPEncoder(model, processor, device, model_name=model_name)


class SigLIPEncoder:
    def __init__(self, model, processor, device: str, model_name: str = ""):
        self.model = model
        self.processor = processor
        self.device = device
        self.model_name = model_name

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
            output = self.model.get_text_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output[1]
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_image(self, image) -> "np.ndarray":
        import numpy as np
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output[1]
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()[0]

    def encode_images_batch(self, images: list) -> "np.ndarray":
        import numpy as np
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output[1]
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()
