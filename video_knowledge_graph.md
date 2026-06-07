# Q-VKG: Query-Conditioned Video Knowledge Graph
## Powered by vLLM + Qwen3-VL for Efficient Long-Video Reasoning

---

## 1. Problem Statement & Core Insight

### 1.1 The Problem

Long video reasoning is unsolved. A 100-minute video has 180,000 frames at 30fps.
No VLM can process all of them. Existing approaches fail in one of three ways:

| Approach | Method | Failure Mode |
|----------|--------|-------------|
| Uniform sampling | Pick every N-th frame | Misses rare critical events |
| Dense captioning | VLM every frame | O(N) API calls, costs $100s per video |
| Multi-agent memory | Agents summarize segments | Hallucination accumulates; no causal structure |
| Static scene graphs | YOLO + rule-based edges | No causality, no identity, no reasoning interface |

### 1.2 Our Core Insight

**Build the graph once. Query it forever. Ground answers in both structure and pixels.**

Three ideas combined:

1. **Unified VLM backbone**: Replace 8 specialized models (YOLO, SigLIP zero-shot,
   EasyOCR, BLIP-2, separate LLM, etc.) with a single Qwen3-VL model served via
   vLLM. One model, one API, one deployment.

2. **Graph as persistent index**: Build the Video Knowledge Graph (VKG) offline once
   per video. At query time, retrieve only the relevant subgraph — O(1) VLM calls
   per question regardless of video length.

3. **Visual + structural grounding**: At query time, the VLM receives both the
   structured graph context (timeline, causal chains, character identities) AND the
   actual keyframe images. It reasons over symbols and pixels simultaneously.

### 1.3 System Overview

Two phases: offline VKG construction (once per video) and online QA (per question).

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — OFFLINE VKG CONSTRUCTION  (once per video, ~12 min/A100) │
│                                                                        │
│  Video ──→ Hierarchical Frame Sampler ──→ ~500 keyframes             │
│                    │                                                   │
│                    ├──→ vLLM [Qwen3-VL 7B]                           │
│                    │         └─→ batched scene extraction JSON:       │
│                    │             objects, labels, OCR, actions,       │
│                    │             spatial relations, state changes      │
│                    │                                                   │
│                    └──→ Whisper large-v3 ──→ timestamped transcript   │
│                                                                        │
│  VKG Nodes ──→ SigLIP encoder ──→ FAISS index                        │
│  Episode events ──→ vLLM [Qwen3-VL] ──→ causal edges (LLM-inferred) │
│                                                                        │
│  Output: VKG.json + FAISS index + frame store (HDF5)                 │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  VKGs persisted to disk
┌────────────────────────────▼─────────────────────────────────────────┐
│  PHASE 2 — ONLINE QA  (per question, ~2-5 seconds)                   │
│                                                                        │
│  Question ──→ Intent Classifier                                       │
│           ──→ Subgraph Activator (FAISS + typed graph traversal)     │
│           ──→ Context Serializer (graph → structured NL)             │
│           ──→ Qwen3-VL (context + keyframe images, single call)      │
│           ──→ Answer + evidence chain                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.4 VLM Call Count vs. Alternatives

| System | Offline VLM Calls | Online VLM Calls | Models in Memory |
|--------|------------------|-----------------|-----------------|
| Dense captioning | O(N) ≈ 6,000 | 1 | 1 VLM |
| HAVEN / DVD | O(S) ≈ 500 | 1 | 1 VLM + tools |
| Multi-model pipeline | 0 (specialized) | 1 | 8+ models |
| **Q-VKG (ours)** | **~115 batched** | **1** | **Qwen3-VL + Whisper + SigLIP** |

---

## 2. Related Works & Collision Analysis

This section maps the full prior-art landscape and explicitly identifies where Q-VKG
overlaps with existing work and where it is genuinely differentiated. All claims in
Section 13 (Novel Contributions) are grounded in this analysis.

### 2.1 Long Video Understanding: Hierarchical & Agentic Methods

These papers tackle the same core problem (long video QA) but via hierarchical
databases or agentic tool-use — not knowledge graphs.

| Paper | Venue | Method Summary | Graph? | Causal? | Identity? | Open-src? |
|-------|-------|---------------|--------|---------|-----------|-----------|
| **HAVEN** (Yin et al., arXiv 2601.13719) | arXiv Jan 2026 | 4-level hierarchy + audiovisual entity cohesion + agentic search. LVBench 84.1 | ✗ DB | ✗ | ✓ audio | ✗ GPT-4.1 |
| **DVD / Deep Video Discovery** (Zhang et al., arXiv 2505.18079) | arXiv May 2025 | 3-tool agentic loop (Browse/Clip/Frame) over multi-grained DB. LVBench 74.2 | ✗ DB | ✗ | ✗ | ✗ GPT-4.1+o3 |
| **Symphony** (arXiv 2603.17307) | arXiv Mar 2026 | 5-agent system (planner/reflector/grounder/subtitle/visual). LVBench +5% | ✗ | ✗ | ✗ | ✗ |
| **MR.Video** (NeurIPS 2025) | NeurIPS 2025 | MapReduce: per-segment Map summaries → global Reduce reasoning | ✗ | ✗ | ✗ | ✗ |
| **VCA** (Yang et al., arXiv 2412.10471) | ICCV 2025 | Curiosity-driven tree-search frame exploration | ✗ | ✗ | ✗ | ✗ |
| **VideoTree** (arXiv 2405.19209) | 2024 | Adaptive coarse-to-fine tree for LLM reasoning. EgoSchema 66.2% | ✗ | ✗ | ✗ | Partial |
| **LLoVi** (CVPR 2024) | CVPR 2024 | Summarize clips with VLM, reason over summaries. EgoSchema 57.6% | ✗ | ✗ | ✗ | Partial |
| **MovieChat** (CVPR 2024) | CVPR 2024 | Short-term dense + long-term compressed token memory | ✗ | ✗ | ✗ | Partial |
| **VideoAgent** (Wu et al., 2024) | 2024 | CLIP-similarity iterative frame retrieval agent. EgoSchema 54.1% | ✗ | ✗ | ✗ | Partial |
| **AVA** (Yan et al., arXiv 2505.00254) | NDSI 2026 | Event Knowledge Graph (EKG) for 10h+ streaming video; agentic retrieval. LVBench 62.3% | Partial (EKG) | ✗ | ✗ | Partial |

**Key gap across all these**: None builds a typed knowledge graph with traversable
edges. None has explicit causal edges. HAVEN has identity tracking but only via
audio. All use multiple LLM calls per question at inference time.

---

### 2.2 Video Knowledge Graphs & Graph-Based Video Reasoning

This cluster is the most critical for novelty assessment.

#### Graph-to-Frame RAG (Yang et al., CVPR 2026, arXiv 2604.04372) — HIGHEST COLLISION RISK

**What it does:** Builds an offline video KG with two views: (1) **event-causal
view** — participants, actions, intent, preconditions, postconditions, causal links;
(2) scene-affordance view — objects, functional areas, world knowledge. Retrieves
a minimal relevant subgraph at query time via `argmax R(q,S) − λ·C(S)`. Renders
the subgraph as a Graphviz visual frame appended to the video for joint LMM reasoning.
GPT-4o for construction, GPT-4o-mini for query mapping. Training-free. CVPR 2026.

This paper shares the core architectural insight: **offline KG + query-time subgraph
retrieval**. It must be cited prominently and differentiated explicitly.

| Dimension | Graph-to-Frame RAG | Q-VKG |
|-----------|-------------------|-------|
| Construction model | GPT-4o (expensive, API-only) | Qwen3-VL via vLLM (open-source) |
| Context delivery | Subgraph rendered as **visual frame** | Structured NL text + actual keyframe images |
| Causal inference | Implicit in graph structure (no explicit LLM reasoning step) | Explicit LLM batch inference per episode with confidence scores |
| Audio/speech modality | ✗ Not present | ✓ Whisper + cross-modal edges |
| Character identity | ✗ Not mentioned | ✓ Description-based cross-scene resolver |
| State transitions | ✗ Not present | ✓ StateChangeNode (entity: prev → next state) |
| Edge taxonomy | Dual-view, unspecified types | 6 categories, 20+ explicitly named types |
| Temporal structure | ✗ Not present | ✓ 4-level backbone (clip→scene→episode→video) |
| Retrieval mechanism | GPT-4o-mini natural language query mapping | FAISS ANN + intent classifier + typed BFS |
| Reproducibility | Requires GPT-4o API | Fully open weights, runs on 2× A100 |

#### Vgent (Shen et al., NeurIPS 2025 Spotlight, arXiv 2510.14032)

Builds structured graphs with semantic relationships across video clips for
retrieval-augmented long video QA. Adds a structured verification step before
generation. 3.0–5.4% improvement on MLVU. No causal edges, no character tracking,
no audio, no explicit edge taxonomy. NeurIPS 2025 Spotlight.

**Collision level: MODERATE.** Graph-based RAG for long video — same family, but
no causal reasoning, no multimodal integration, no character tracking.

#### MAGIC-Video / Bridging Modalities (Li et al., arXiv 2605.08271, May 2026)

Multimodal memory graph with interleaved narrative chain. **Six typed edges**
unifying episodic, semantic, and visual content. Tracks **entity biographies**
across extended timeframes (days/weeks). Training-free. Focused on ultra-long
egocentric video (EgoLifeQA, Ego-R1: +10.1 pts over prior best).

**Collision level: MODERATE.** Has typed edges (6 types) and entity tracking. Key
differences: egocentric/lifelog domain only, no causal edges, no audio integration,
no intent-driven activation, no offline-offline-online cost separation.

#### Mind Palace / VideoMindPalace (Huang et al., arXiv 2501.04336, Jan 2025)

Environment-grounded semantic graphs for long video: hand-object interactions,
activity zones, environment layout. Strong spatial focus. No causal edges, no
character tracking, no audio. Egocentric-oriented (Ego4D, EgoSchema).

**Collision level: LOW.** Different graph focus (spatial environment vs. narrative
KG). No causal reasoning. Narrower modality coverage.

#### GraphVideoAgent (Chu et al., arXiv 2501.15953, Jan 2025)

Entity-relation graphs used only for frame selection guidance; not for direct
reasoning. 2.2-pt improvement on EgoSchema (marginal). No causal edges.

**Collision level: LOW.** Graph is a selection heuristic, not a reasoning substrate.

#### EgoGraph (Sun et al., arXiv 2602.23709, Feb 2026)

Temporal KG for egocentric video: people, objects, locations, event nodes with
temporal relational modeling. Accumulates stable long-term memory over multiple
days. Egocentric only (EgoLifeQA, EgoR1-bench).

**Collision level: LOW-MODERATE.** Egocentric only. No causal edges. No audio.
Separate domain and target video type.

#### AVA Event Knowledge Graph (Yan et al., arXiv 2505.00254, NDSI 2026)

Indexes long video streams (10h+) via Event Knowledge Graphs (EKG) for near-
real-time analytics. Supports complex queries via agentic retrieval-generation.
LVBench 62.3%, VideoMME-Long 64.1%. No explicit causal edges. Streaming-first
design (not batch offline). Agentic (multiple LLM calls per query).

**Collision level: MODERATE.** Closest to a production video KG system. Key
differences: no causal edges, streaming vs. offline-first, agentic vs. single-call
retrieval, different graph schema focus (events/entities vs. full multimodal narrative).

#### Short-video Scene Graph Papers (Pre-LLM Era)

Action Genome (CVPR 2020), VGT (ECCV 2022), HQGA (AAAI 2022), STTran (ICCV 2021),
DualVGR, PGAT — per-frame scene graphs on short videos (<3 min) using GNNs. None
handles long video, audio, causal reasoning, or LLM-based extraction.
**No meaningful collision at our target scale.**

---

### 2.3 Causal Reasoning in Video

#### MECD+ (Chen et al., IEEE TPAMI 2025, arXiv 2501.07227) — COLLISION ON CAUSAL EDGES

**What it does:** Builds event-level causal graphs from video using an Event Granger
Test: masks premise events, measures whether their absence reduces prediction
accuracy of the result event. Produces a binary causal relation DAG. Uses a
purpose-built transformer (NOT LLM-based). Operates on short videos with 4–11
discrete events (ActivityNet, NExT-Video, EgoSchema). Outperforms GPT-4o by 5.77%.
Published in IEEE TPAMI.

**Collision level: MODERATE on the concept of building causal edges from video.**

| Dimension | MECD+ | Q-VKG |
|-----------|-------|-------|
| Causal inference mechanism | Granger-inspired mask-based specialized model (trained) | LLM batch visual-grounded inference (training-free) |
| Video length | Short clips, 4–11 events, <2 min | Long video, 100+ min, 10–30 episodes |
| Task scope | Causal graph only | Causal is one of 6 edge categories in a full pipeline |
| Integration | Standalone benchmark | Integrated into graph retrieval and QA answering |
| Generalization | Trained on specific causal datasets | Zero-shot via LLM reasoning over any domain |

Our differentiation: LLM-based causal inference is **training-free and domain-general**,
operating on 100-minute videos rather than isolated short clips. MECD+ is a specialist
model for a specialized task; ours is a component in a general-purpose pipeline.

#### NExT-QA, Causal-VidQA, CLEVRER

Benchmarks for evaluating causal reasoning — they do not build causal graphs as
inference-time pipeline components. We evaluate on NExT-QA causal split to measure
our causal edge contribution.

---

### 2.4 Character & Entity Identity in Video

| Paper | Method | Works without audio? | General domain? |
|-------|--------|---------------------|-----------------|
| HAVEN | Speaker diarization → entity cohesion | ✗ Audio required | ✓ |
| MAGIC-Video | Entity biographies (temporal tracking) | ✓ | ✗ Egocentric only |
| EgoGraph | People nodes, temporal relations | ✓ | ✗ Egocentric only |
| AVA | Entity nodes in EKG | ✓ | ✓ |
| **Q-VKG** | **Description-based clustering (visual only)** | **✓** | **✓** |

**Key gap**: No paper does audio-free character identity resolution in
**general-domain** long video QA. HAVEN requires diarizable audio. MAGIC-Video
and EgoGraph are egocentric-only. Our visual description clustering works for
silent films, dubbed content, and any language.

---

### 2.5 Summary: The Exact White Space Q-VKG Occupies

No existing paper simultaneously achieves all five of the following:

1. **Typed multimodal KG** with traversable cross-modal edges (visual + audio + text)
   for general-domain long video
2. **LLM-inferred causal edges** on long video episodes (MECD+ does this only for
   short clips with a specialized trained model)
3. **Audio-free character identity resolution** (HAVEN does it only with audio)
4. **Single-call online QA** via deterministic graph activation (all competitors
   use multiple agentic LLM calls)
5. **Fully open-source reproducible pipeline** (HAVEN, DVD, Graph-to-Frame RAG all
   depend on GPT-4.1 or GPT-4o)

Graph-to-Frame RAG (CVPR 2026) is the closest prior work. It must be cited as the
primary baseline and differentiated along the dimensions in §2.2.

---

## 3. Model Stack

### 3.1 Architecture: Three Models, Two Phases

Three specialised models cover the full pipeline. Qwen3-VL is served via vLLM for both
offline construction and online QA. Whisper and SigLIP run as standalone inference
services outside the main serving loop.

```
┌──────────────────────────────────────────────────────────────────────┐
│  MODEL 1: Qwen3-VL-7B  ← served via vLLM                            │
│                                                                        │
│  Phase 1 (offline graph build):                                       │
│    ├─ Scene extraction JSON (objects, OCR, actions, spatial, states) │
│    ├─ Episode causal chain inference                                  │
│    └─ Episode segmentation (text-only, no images)                    │
│                                                                        │
│  Phase 2 (online QA):                                                 │
│    └─ Single-call QA with structured context + keyframes             │
│                                                                        │
├──────────────────────────────────────────────────────────────────────┤
│  MODEL 2: Whisper large-v3  ← standalone (audio only)               │
│    ├─ Full-audio transcription with word timestamps                  │
│    └─ Runs once offline; output feeds SpeechNodes                   │
│                                                                        │
├──────────────────────────────────────────────────────────────────────┤
│  MODEL 3: SigLIP-SO400M encoder  ← standalone (encode only)         │
│    ├─ Encodes all VKG nodes → 1152-d embeddings                     │
│    └─ FAISS HNSW index for query-time subgraph seed retrieval       │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Keep SigLIP Alongside Qwen3-VL

Qwen3-VL could technically produce embeddings (via its vision encoder), but SigLIP
is kept for one specific task: **fast cosine similarity search at scale**.

| Property | SigLIP encoder | Qwen3-VL vision encoder |
|----------|---------------|------------------------|
| Embedding dim | 1152-d | 3584-d (7B) |
| Inference time per frame | ~5ms | ~80ms |
| Memory footprint | 400MB | 15GB (7B) |
| FAISS indexing | Trivial | Possible but wasteful |

For 500 nodes queried at inference time, SigLIP + FAISS takes ~50ms.
Using Qwen's encoder would take ~800ms and require keeping the model hot during
graph traversal. SigLIP pays for itself.

### 3.3 Why Keep Whisper

Qwen3-VL and all current VLMs process images and text, not raw audio waveforms.
Whisper is unavoidable for speech transcription. It runs once per video on the full
audio track and produces timestamped transcripts that become `SpeechNode`s.

---

## 4. vLLM Setup & Configuration

### 4.1 Serving Configuration

```python
# server.py — start once, reuse for all offline and online requests
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",   # swap for Qwen3-VL when available
    tensor_parallel_size=2,                  # 2× A100 80GB
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,   # caches system prompt KV across all requests
    max_num_seqs=64,              # 64 concurrent sequences in flight
    max_model_len=32768,          # Qwen2.5-VL supports 32K context
    limit_mm_per_prompt={
        "image": 10               # up to 10 frames per single prompt
    },
    mm_processor_kwargs={
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,  # dynamic resolution
    }
)
```

**Why these settings matter:**

- `enable_prefix_caching=True`: Every offline extraction request shares the same
  system prompt (~200 tokens). vLLM caches its KV representation once and reuses
  it across all 100+ scene calls. Saves ~15% of compute on the offline pass.

- `max_num_seqs=64`: Continuous batching — vLLM fills GPU with up to 64 concurrent
  decode streams. All scene extraction requests are queued simultaneously and
  processed at peak utilization.

- `limit_mm_per_prompt={"image": 10}`: Allows sending an entire scene (5-10 frames)
  as one multi-image prompt. One call extracts everything about a scene instead of
  one call per frame.

- Dynamic resolution (`mm_processor_kwargs`): Qwen2.5-VL natively supports
  variable-resolution inputs. Frames are tiled into visual tokens at their natural
  resolution up to the max, preserving detail in text-heavy or object-dense scenes.

### 3.2 Structured Output with Guided Decoding

All offline extraction uses JSON schema-guided decoding. This guarantees parseable
output with zero prompt engineering for output format:

```python
from vllm.sampling_params import GuidedDecodingParams

OFFLINE_SAMPLING = SamplingParams(
    temperature=0,        # deterministic
    max_tokens=2048,
    guided_decoding=GuidedDecodingParams(json=SCENE_EXTRACTION_SCHEMA)
)

CAUSAL_SAMPLING = SamplingParams(
    temperature=0.1,      # slight randomness for causal creativity
    max_tokens=1024,
    guided_decoding=GuidedDecodingParams(json=CAUSAL_SCHEMA)
)

QA_SAMPLING = SamplingParams(
    temperature=0,
    max_tokens=512
    # no guided decoding — free-form answer
)
```

---

## 5. Hierarchical Frame Sampling

### 4.1 Four-Level Temporal Hierarchy

The sampling strategy determines how many VLM calls are needed offline. The goal:
capture all semantically important moments within a frame budget of ~500 keyframes
for a 100-minute video.

```
Level 0: Raw video  ─── 30 fps ─── 180,000 frames (never processed)
    │
    ▼
Level 1: CLIP nodes ─── 1-3 fps importance sampling ─── 3,000-6,000 clips
    │    (individual action moments, object appearances)
    │
    ▼
Level 2: SCENE nodes ── visual discontinuity detection ── 50-150 scenes
    │    (coherent visual context: one location/lighting/cast)
    │
    ▼
Level 3: EPISODE nodes ─ LLM semantic segmentation ─── 10-30 episodes
    │    (narrative segments: "introduction", "conflict", "resolution")
    │
    ▼
Level 4: VIDEO node ──── single root ──── 1 per video
         (global context, genre, setting, main characters)
```

Each level is a node type in the VKG. The hierarchy is the graph backbone — every
other node (object, event, speech) attaches to a clip or scene node.

### 4.2 Importance-Scored Sampling

```python
class HierarchicalSampler:

    def sample(self, video_path: str, budget: int = 500) -> SampleResult:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Step 1: Extract at 1 fps (coarse grid)
        coarse = self._extract_uniform(cap, target_fps=1.0)  # ~6,000 frames

        # Step 2: Score every coarse frame
        scores = self._score_all(coarse)                      # vectorized

        # Step 3: Detect scene boundaries
        boundaries = self._detect_boundaries(coarse, scores)  # ~100 scenes

        # Step 4: Per-scene keyframe selection
        keyframes = []
        for scene in boundaries:
            kf = self._select_keyframes(
                frames=coarse[scene.start_idx:scene.end_idx],
                scores=scores[scene.start_idx:scene.end_idx],
                k=max(2, int(scene.duration_sec / 15)),  # 1 frame per 15s
                max_gap_sec=30.0,                         # coverage constraint
                diversity_weight=0.4                       # spread in embedding space
            )
            keyframes.extend(kf)

        # Step 5: Budget rebalancing across scenes
        keyframes = self._rebalance(keyframes, scores, budget)

        # Step 6: Episode segmentation (LLM call on scene captions)
        scene_captions = self._fast_caption_scenes(boundaries)
        episodes = self._segment_episodes(scene_captions)    # 1 LLM call

        return SampleResult(keyframes, boundaries, episodes)

    def _score_frame(self, frame, prev, next, audio_rms) -> float:
        visual_delta   = self._histogram_diff(frame, prev)         # fast
        semantic_shift = self._siglip_cosine_dist(frame, prev)     # batched
        motion         = self._optical_flow_mag(prev, next)        # TV-L1
        obj_delta      = self._yolo_lite_count_delta(frame, prev)  # YOLOv8n
        audio_energy   = audio_rms                                 # pre-computed

        return (0.25 * visual_delta   +
                0.25 * semantic_shift +
                0.20 * motion         +
                0.15 * obj_delta      +
                0.15 * audio_energy)

    def _select_keyframes(self, frames, scores, k, max_gap_sec, diversity_weight):
        # Greedy: pick highest score, then penalize nearby frames (NMS-like)
        selected = []
        last_selected_t = -999

        sorted_idx = np.argsort(scores)[::-1]
        for idx in sorted_idx:
            if len(selected) >= k:
                break
            t = frames[idx].timestamp
            # Enforce minimum gap (no two keyframes within 5s)
            if t - last_selected_t < 5.0:
                continue
            selected.append(frames[idx])
            last_selected_t = t

        # Coverage pass: fill any gaps > max_gap_sec
        selected = self._fill_coverage_gaps(frames, selected, max_gap_sec)
        return selected
```

### 4.3 Scene Boundary Detection

Uses a combination of histogram distance and SigLIP embedding shift to catch both
hard cuts (scene change) and soft transitions (gradual scene drift):

```python
def _detect_boundaries(self, frames, scores, hard_thresh=0.6, soft_thresh=0.35):
    boundaries = []
    embeddings = self._siglip_batch(frames)  # batched, fast

    scene_start = 0
    for i in range(1, len(frames)):
        hist_dist = self._histogram_diff(frames[i], frames[i-1])
        emb_dist  = 1 - cosine_similarity(embeddings[i], embeddings[i-1])

        is_hard_cut = hist_dist > hard_thresh
        is_soft_cut = emb_dist > soft_thresh and hist_dist > 0.3

        if is_hard_cut or is_soft_cut:
            boundaries.append(SceneBoundary(
                start_idx=scene_start,
                end_idx=i - 1,
                start_time=frames[scene_start].timestamp,
                end_time=frames[i-1].timestamp,
            ))
            scene_start = i

    # Final scene
    boundaries.append(SceneBoundary(scene_start, len(frames)-1, ...))
    return boundaries
```

---

## 6. Offline VLM Extraction Pipeline

### 5.1 JSON Schemas for Guided Decoding

These schemas define exactly what Qwen3-VL must output for each call.
Guided decoding guarantees conformance — no regex parsing, no failure modes.

```python
SCENE_EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["scene_label", "objects", "actions", "spatial_relations",
                 "characters", "ocr_text", "state_changes", "scene_mood"],
    "properties": {

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
                        "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] normalized 0-1"
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "state":      {"type": "string", "description": "current state, e.g. 'on', 'open', 'broken'"}
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
                    "actor":       {"type": "string", "description": "character description or 'unknown'"},
                    "object":      {"type": "string", "description": "object being acted upon, if any"},
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
                                 "in_front_of", "behind", "near", "contains", "overlaps"]
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
                        "description": "Distinctive appearance description: gender, age, clothing, hair"
                    },
                    "bbox_norm":   {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "emotion":     {"type": "string"},
                    "action":      {"type": "string"}
                }
            }
        },

        "ocr_text": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":       {"type": "string"},
                    "bbox_norm":  {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
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
                    "approx_timestamp": {"type": "number", "description": "seconds into video"}
                }
            }
        },

        "scene_mood": {
            "type": "string",
            "enum": ["tense", "calm", "joyful", "sad", "urgent", "neutral", "comedic", "dramatic"]
        }
    }
}

CAUSAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["cause", "effect", "relation", "confidence", "reasoning"],
        "properties": {
            "cause":      {"type": "string", "description": "Exact event description from the timeline"},
            "effect":     {"type": "string", "description": "Exact event description from the timeline"},
            "relation":   {"type": "string", "enum": ["CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"]},
            "confidence": {"type": "number", "minimum": 0.6, "maximum": 1.0},
            "reasoning":  {"type": "string", "description": "One sentence explaining the causal link"}
        }
    }
}
```

### 5.2 Scene Extraction: Multi-Frame Per Call

The key throughput optimization: send all keyframes for a scene in one prompt.
Qwen3-VL sees temporal context across frames and can detect state changes,
character continuity, and actions that span multiple frames — things a per-frame
call cannot see.

```python
SCENE_SYSTEM_PROMPT = """You are a video analysis system. You receive multiple frames
from a single video scene in chronological order. Each frame is labeled with its
timestamp. Extract structured information about the scene as JSON.

Be precise about spatial positions (use normalized coordinates 0-1).
For characters, provide distinctive appearance descriptions to enable cross-scene
re-identification (e.g., "young woman, black jacket, short brown hair, glasses").
For state changes, note the approximate timestamp when the transition occurs.
Only report observations you can directly verify in the frames."""

def build_scene_extraction_requests(
    scenes: List[Scene],
    frame_store: FrameStore
) -> List[ChatCompletionRequest]:

    requests = []
    for scene in scenes:
        # Build interleaved image+text content
        content = []
        for frame in scene.keyframes:
            content.append({
                "type": "image_url",
                "image_url": {"url": frame_store.get_b64_url(frame.id)}
            })
            content.append({
                "type": "text",
                "text": f"[t={frame.timestamp:.1f}s]"
            })

        content.append({
            "type": "text",
            "text": "Extract structured information from these scene frames as JSON."
        })

        requests.append({
            "scene_id": scene.id,
            "scene_time_range": (scene.t_start, scene.t_end),
            "messages": [
                {"role": "system", "content": SCENE_SYSTEM_PROMPT},
                {"role": "user",   "content": content}
            ]
        })

    return requests

def run_scene_extraction(
    scenes: List[Scene],
    frame_store: FrameStore,
    llm: LLM
) -> Dict[str, SceneData]:

    requests = build_scene_extraction_requests(scenes, frame_store)

    # Submit ALL scenes as one batch — vLLM continuous batching handles the queue
    outputs = llm.chat(
        messages=[r["messages"] for r in requests],
        sampling_params=OFFLINE_SAMPLING,
        use_tqdm=True
    )

    results = {}
    for req, out in zip(requests, outputs):
        scene_data = json.loads(out.outputs[0].text)
        results[req["scene_id"]] = SceneData(
            scene_id=req["scene_id"],
            time_range=req["scene_time_range"],
            **scene_data
        )

    return results
```

### 5.3 Episode Segmentation (1 LLM Call Per Video)

After scene extraction, group scenes into narrative episodes using a single LLM
call on the scene labels. This is cheap (text-only, no images):

```python
EPISODE_SYSTEM_PROMPT = """You are a narrative analyst. You receive an ordered list
of scene descriptions from a video. Group them into 10-30 narrative episodes —
coherent story segments with a clear beginning and end.

Each episode should represent a distinct narrative beat: an introduction, a conflict,
a resolution, a transition, etc. Return a JSON array of episodes."""

EPISODE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["label", "start_scene_idx", "end_scene_idx", "narrative_role"],
        "properties": {
            "label":          {"type": "string"},
            "start_scene_idx":{"type": "integer"},
            "end_scene_idx":  {"type": "integer"},
            "narrative_role": {
                "type": "string",
                "enum": ["introduction", "rising_action", "climax", "falling_action",
                         "resolution", "subplot", "transition", "flashback", "other"]
            },
            "summary": {"type": "string", "description": "2-3 sentence episode summary"}
        }
    }
}

def segment_episodes(scenes: List[Scene], scene_data: Dict, llm: LLM) -> List[Episode]:
    scene_descriptions = [
        f"Scene {i} [{s.t_start:.0f}s-{s.t_end:.0f}s]: {scene_data[s.id].scene_label}"
        for i, s in enumerate(scenes)
    ]

    output = llm.chat(
        messages=[[
            {"role": "system", "content": EPISODE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Video has {len(scenes)} scenes. Group into narrative episodes:\n\n"
                + "\n".join(scene_descriptions)
            )}
        ]],
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=2048,
            guided_decoding=GuidedDecodingParams(json=EPISODE_SCHEMA)
        )
    )[0]

    episode_data = json.loads(output.outputs[0].text)
    return [Episode(scenes=scenes[e["start_scene_idx"]:e["end_scene_idx"]+1], **e)
            for e in episode_data]
```

### 5.4 Character Resolution (Description Clustering)

Instead of ArcFace metric embeddings, Qwen3-VL provides rich textual descriptions
of each character appearance. We cluster these descriptions using text embeddings
to resolve cross-scene identity:

```python
class DescriptionBasedCharacterResolver:
    """
    Resolve character identity across scenes using appearance descriptions.

    Limitation vs. ArcFace: weaker for visually similar people (twins, uniforms).
    Advantage: no additional model, works for occluded or distant people.
    """

    def resolve(
        self,
        all_character_mentions: List[CharacterMention],  # from scene extraction
        siglip_encoder,                                   # reuse for text encoding
        similarity_threshold: float = 0.80
    ) -> List[CharacterNode]:

        # Encode all descriptions as text embeddings
        descriptions = [m.description for m in all_character_mentions]
        embeddings = siglip_encoder.encode_text(descriptions)
        faiss.normalize_L2(embeddings)

        # Cluster by description similarity
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(
            eps=1 - similarity_threshold,
            min_samples=1,
            metric='cosine'
        ).fit_predict(embeddings)

        # Build CharacterNode per cluster
        characters = {}
        for mention, label in zip(all_character_mentions, labels):
            if label not in characters:
                characters[label] = CharacterNode(
                    id=f"char_{label}",
                    label=f"Person_{label}",
                    canonical_description=mention.description,
                    appearances=[]
                )
            characters[label].appearances.append(
                CharacterAppearance(
                    scene_id=mention.scene_id,
                    timestamp=mention.timestamp,
                    bbox=mention.bbox,
                    action=mention.action,
                    emotion=mention.emotion
                )
            )

        # Refine canonical description: pick most detailed across appearances
        for char in characters.values():
            char.canonical_description = max(
                [a.description for a in char.appearances],
                key=len
            )

        return list(characters.values())
```

**Note on ArcFace fallback:** For videos where character accuracy is critical
(e.g., MovieChat benchmark), optionally add ArcFace/OSNet on detected person crops.
This adds one more model but dramatically improves ReID for visually similar people.
The description-based resolver is the default; ArcFace is an optional upgrade.

### 5.5 Causal Chain Inference (Multi-Frame LLM Batch)

For each episode, Qwen3-VL receives the keyframe images AND the extracted event
timeline. Visual grounding improves causal judgment compared to text-only LLM:

```python
CAUSAL_SYSTEM_PROMPT = """You are analyzing a video episode to identify causal
relationships between events. You have access to:
1. Keyframe images from this episode (in chronological order)
2. A timeline of extracted events with timestamps

Identify causal links: what caused what, what enabled what, what motivated whom.
Only report high-confidence links (>0.6). Ground each link in specific observations
from the frames or timeline. Return JSON."""

def infer_episode_causality(
    episode: Episode,
    graph: VKGraph,
    frame_store: FrameStore,
    llm: LLM
) -> List[CausalEdge]:

    # Get all events in this episode
    events = graph.get_events_in_episode(episode)
    event_timeline = "\n".join(
        f"  [{e.t_start:.1f}s] [{e.node_type}] {e.label}"
        for e in events
    )

    # Pick up to 8 representative keyframes for the episode
    keyframes = episode.get_representative_frames(max_frames=8)

    content = []
    for frame in keyframes:
        content.append({"type": "image_url",
                        "image_url": {"url": frame_store.get_b64_url(frame.id)}})
        content.append({"type": "text", "text": f"[t={frame.timestamp:.1f}s]"})

    content.append({"type": "text", "text": (
        f"\nEpisode: \"{episode.label}\" ({episode.t_start:.0f}s - {episode.t_end:.0f}s)\n"
        f"Narrative role: {episode.narrative_role}\n\n"
        f"Event timeline:\n{event_timeline}\n\n"
        f"Identify causal relationships as JSON."
    )})

    output = llm.chat(
        messages=[[
            {"role": "system", "content": CAUSAL_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]],
        sampling_params=CAUSAL_SAMPLING
    )[0]

    causal_data = json.loads(output.outputs[0].text)

    edges = []
    for link in causal_data:
        cause_node = graph.find_event_by_description(link["cause"], events)
        effect_node = graph.find_event_by_description(link["effect"], events)
        if cause_node and effect_node and link["confidence"] >= 0.6:
            edges.append(CausalEdge(
                source_id=cause_node.id,
                target_id=effect_node.id,
                relation=link["relation"],
                confidence=link["confidence"],
                reasoning=link["reasoning"]
            ))

    return edges
```

### 5.6 FAISS Index Construction

After all nodes are created from scene extraction, build the similarity index:

```python
def build_faiss_index(graph: VKGraph, siglip_encoder, index_path: str):
    nodes = list(graph.nodes.values())

    # Encode all node labels + context as text for cross-modal search
    texts = [
        f"{n.node_type}: {n.label} at {n.t_start:.0f}s"
        for n in nodes
    ]
    embeddings = siglip_encoder.encode_text(texts)  # batched

    # Also encode visual nodes with their image
    for i, node in enumerate(nodes):
        if node.keyframe_id:
            img_emb = siglip_encoder.encode_image(
                node.get_image(graph.frame_store)
            )
            # Fuse text and image embeddings (average)
            embeddings[i] = (embeddings[i] + img_emb) / 2.0

    faiss.normalize_L2(embeddings)

    # HNSW: better recall than IVF at this scale, no training needed
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)

    # Save
    faiss.write_index(index, index_path)
    graph.node_id_list = [n.id for n in nodes]  # maps FAISS index → node ID

    return index
```

---

## 7. Knowledge Graph Schema

### 6.1 Node Types

```
VKG NODE HIERARCHY
│
├─ TEMPORAL BACKBONE (always built, hierarchical)
│   ├─ VideoNode      ── root node, global video context
│   ├─ EpisodeNode    ── narrative segment (LLM-segmented, 10-30 per video)
│   ├─ SceneNode      ── visual context window (50-150 per video)
│   └─ ClipNode       ── individual keyframe moment (300-500 per video)
│
├─ ENTITY NODES (persistent across time)
│   ├─ CharacterNode  ── resolved person identity (cross-scene)
│   └─ ObjectNode     ── tracked physical object (cross-frame)
│
├─ EVENT NODES (instantiated — occur at a specific time)
│   ├─ ActionNode        ── character/object does something
│   ├─ InteractionNode   ── between two entities
│   └─ StateChangeNode   ── entity changes state (stove: off → on)
│
├─ PERCEPTION NODES (raw multimodal observations)
│   ├─ SpeechNode     ── Whisper transcript segment
│   ├─ OCRNode        ── on-screen text (from Qwen3-VL)
│   └─ AudioEventNode ── non-speech audio event
│
└─ ABSTRACT NODES (LLM-inferred)
    ├─ CauseNode      ── inferred cause of an event
    └─ GoalNode       ── inferred character goal
```

### 6.2 Node Dataclass

```python
@dataclass
class VKGNode:
    id:         str
    node_type:  str          # from taxonomy above
    label:      str          # human-readable, e.g. "person cooking at stove"
    level:      int          # 0=clip, 1=scene, 2=episode, 3=video

    # Temporal grounding
    t_start:    float        # seconds
    t_end:      float        # seconds

    # Visual grounding
    keyframe_id: Optional[str]          # reference to frame in HDF5 store
    bbox:        Optional[List[float]]  # [x1,y1,x2,y2] normalized
    confidence:  float = 1.0

    # Embeddings
    siglip_embedding: Optional[np.ndarray] = None  # for FAISS search
    faiss_idx:        Optional[int] = None          # position in FAISS index

    # Entity-specific
    entity_id:           Optional[str] = None  # for character/object continuity
    canonical_description: Optional[str] = None  # for character ReID

    # State-specific
    prev_state: Optional[str] = None
    next_state: Optional[str] = None

    # Parent in hierarchy
    parent_id: Optional[str] = None   # scene → episode, clip → scene

    metadata: Dict = field(default_factory=dict)
```

### 6.3 Edge Types

```python
EDGE_TYPES = {

    # Temporal backbone
    "PRECEDES":     "a ends before b starts",
    "OVERLAPS":     "a and b overlap in time",
    "DURING":       "a is contained within b's time span",

    # Hierarchical backbone
    "CONTAINS":     "parent contains child (episode→scene, scene→clip)",
    "INSTANCE_OF":  "event is an instance within a scene",

    # Entity continuity
    "SAME_ENTITY":     "same person/object at different times",
    "PERFORMS":        "character performs action",
    "INTERACTS_WITH":  "two entities interact",
    "LOCATED_IN":      "entity appears in scene",

    # Spatial (from Qwen3-VL extraction)
    "LEFT_OF":      "spatial: left",
    "RIGHT_OF":     "spatial: right",
    "ABOVE":        "spatial: above",
    "BELOW":        "spatial: below",
    "IN_FRONT_OF":  "spatial: closer to camera",
    "BEHIND":       "spatial: further from camera",
    "NEAR":         "spatial: in close proximity",
    "CONTAINS_SPATIAL": "bounding box containment",

    # Causal (from Qwen3-VL causal inference)
    "CAUSES":       "a directly causes b",
    "ENABLES":      "a creates condition for b",
    "PREVENTS":     "a prevents b from occurring",
    "MOTIVATES":    "a is character's motivation for b",

    # Semantic
    "SIMILAR_TO":   "high cosine similarity (FAISS)",
    "CONTRADICTS":  "conflicting observations",

    # Cross-modal
    "DESCRIBES":    "speech describes concurrent visual",
    "MENTIONS":     "speech mentions a visible entity",
    "LABELS":       "OCR text labels a visible object",
    "ACCOMPANIES":  "audio event tied to visual action",
}

@dataclass
class VKGEdge:
    source_id:     str
    target_id:     str
    relation_type: str   # from EDGE_TYPES
    weight:        float  # 0-1 edge strength
    confidence:    float  # 0-1 extraction confidence
    metadata:      Dict = field(default_factory=dict)
    # metadata may include: {"reasoning": "...", "source": "qwen3vl|faiss|whisper"}
```

---

## 8. Graph Construction Orchestrator

### 7.1 Full Offline Pipeline

```python
class VKGBuilder:

    def __init__(self, llm: LLM, whisper_model, siglip_encoder, config: Config):
        self.llm = llm
        self.whisper = whisper_model
        self.siglip = siglip_encoder
        self.config = config

    def build(self, video_path: str, output_dir: str) -> VKGraph:
        graph = VKGraph()
        frame_store = FrameStore(output_dir)

        print("Step 1: Hierarchical frame sampling...")
        sampler = HierarchicalSampler(self.siglip)
        sample = sampler.sample(video_path, budget=self.config.frame_budget)
        frame_store.save_keyframes(sample.keyframes)

        print("Step 2: Audio transcription (Whisper)...")
        transcript = self.whisper.transcribe(video_path, word_timestamps=True)
        speech_nodes = self._build_speech_nodes(transcript)
        graph.add_nodes(speech_nodes)

        print("Step 3: Scene extraction (Qwen3-VL batch)...")
        scene_requests = build_scene_extraction_requests(sample.scenes, frame_store)
        scene_outputs = self.llm.chat(
            messages=[r["messages"] for r in scene_requests],
            sampling_params=OFFLINE_SAMPLING,
            use_tqdm=True
        )
        scene_data = {
            r["scene_id"]: json.loads(o.outputs[0].text)
            for r, o in zip(scene_requests, scene_outputs)
        }

        print("Step 4: Node creation...")
        self._create_temporal_backbone(graph, sample, scene_data)
        self._create_entity_nodes(graph, scene_data, frame_store)
        self._create_event_nodes(graph, scene_data)
        self._create_perception_nodes(graph, scene_data, speech_nodes)

        print("Step 5: Temporal + hierarchical edges...")
        self._build_temporal_edges(graph)
        self._build_hierarchical_edges(graph, sample)

        print("Step 6: Spatial edges (from Qwen3-VL output)...")
        self._build_spatial_edges_from_extraction(graph, scene_data)

        print("Step 7: Cross-modal edges...")
        self._build_crossmodal_edges(graph)

        print("Step 8: Character resolution...")
        resolver = DescriptionBasedCharacterResolver()
        characters = resolver.resolve(
            graph.get_all_character_mentions(),
            self.siglip
        )
        self._link_characters_to_events(graph, characters)

        print("Step 9: FAISS index...")
        faiss_index = build_faiss_index(graph, self.siglip,
                                         f"{output_dir}/vkg.index")

        print("Step 10: Causal chain inference (Qwen3-VL batch)...")
        episode_requests = [
            build_causal_request(ep, graph, frame_store)
            for ep in graph.get_episodes()
        ]
        causal_outputs = self.llm.chat(
            messages=[r["messages"] for r in episode_requests],
            sampling_params=CAUSAL_SAMPLING,
            use_tqdm=True
        )
        for req, out in zip(episode_requests, causal_outputs):
            causal_edges = parse_causal_edges(out.outputs[0].text, graph)
            graph.add_edges(causal_edges)

        print("Step 11: Semantic edges (FAISS ANN)...")
        build_semantic_edges_faiss(graph, faiss_index,
                                    threshold=0.78, k_neighbors=10)

        graph.save(f"{output_dir}/vkg.json")
        print(f"VKG built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
```

### 7.2 Edge Construction: Spatial (From Qwen Output, Not Heuristics)

Unlike the original plan's IoU-based spatial edges, spatial relations now come
directly from Qwen3-VL's visual understanding — no bounding box math required:

```python
def _build_spatial_edges_from_extraction(
    self,
    graph: VKGraph,
    scene_data: Dict
) -> None:
    for scene_id, data in scene_data.items():
        scene_node = graph.get_node(scene_id)

        for rel in data.get("spatial_relations", []):
            subject_node = graph.find_entity_in_scene(rel["subject"], scene_id)
            object_node  = graph.find_entity_in_scene(rel["object"], scene_id)

            if subject_node and object_node:
                graph.add_edge(VKGEdge(
                    source_id=subject_node.id,
                    target_id=object_node.id,
                    relation_type=rel["relation"].upper(),
                    weight=0.85,
                    confidence=0.85,
                    metadata={"source": "qwen3vl", "scene": scene_id}
                ))
```

### 7.3 Efficient Semantic Edge Construction (FAISS)

```python
def build_semantic_edges_faiss(
    graph: VKGraph,
    faiss_index: faiss.Index,
    threshold: float = 0.78,
    k_neighbors: int = 10
) -> None:
    n = len(graph.node_id_list)
    embeddings = faiss_index.reconstruct_n(0, n).astype(np.float32)

    # Query all nodes for their k nearest neighbors in one pass
    similarities, indices = faiss_index.search(embeddings, k_neighbors + 1)

    added = 0
    for i, (sims, nbrs) in enumerate(zip(similarities, indices)):
        node_a = graph.nodes[graph.node_id_list[i]]
        for sim, j in zip(sims[1:], nbrs[1:]):
            if sim < threshold:
                break
            node_b = graph.nodes[graph.node_id_list[j]]
            # Don't create SIMILAR_TO between parent/child pairs
            if node_b.id == node_a.parent_id or node_a.id == node_b.parent_id:
                continue
            graph.add_edge(VKGEdge(
                source_id=node_a.id,
                target_id=node_b.id,
                relation_type="SIMILAR_TO",
                weight=float(sim),
                confidence=float(sim),
                metadata={"source": "faiss_hnsw"}
            ))
            added += 1

    print(f"Added {added} semantic edges (FAISS HNSW, threshold={threshold})")
```

---

## 9. Online Phase: Query-Conditioned Reasoning

### 8.1 Query Intent Classification

Fast keyword-based classification, no LLM call needed:

```python
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
                    "had not", "instead"]
}

def classify_intent(question: str) -> List[str]:
    q = question.lower()
    return [intent for intent, kws in INTENT_PATTERNS.items()
            if any(kw in q for kw in kws)] or ["SEMANTIC"]
```

### 8.2 Subgraph Activator

```python
class SubgraphActivator:

    def __init__(self, graph: VKGraph, faiss_index, siglip_encoder, config):
        self.graph   = graph
        self.index   = faiss_index
        self.siglip  = siglip_encoder
        self.config  = config

    def activate(self, question: str, intents: List[str]) -> SubGraph:
        # Step 1: Embed question, find semantically relevant seed nodes
        q_emb = self.siglip.encode_text([question])
        faiss.normalize_L2(q_emb)
        sims, idx = self.index.search(q_emb, k=20)
        seeds = [self.graph.nodes[self.graph.node_id_list[i]]
                 for i in idx[0] if sims[0][list(idx[0]).index(i)] > 0.5]

        activated = {n.id for n in seeds}

        # Step 2: Intent-driven graph expansion
        for intent in intents:

            if intent == "TEMPORAL":
                for node in seeds:
                    activated |= self._walk_temporal_spine(node, hops=4)

            elif intent == "CAUSAL":
                for node in seeds:
                    activated |= self._follow_causal_edges(node, depth=3)

            elif intent == "IDENTITY":
                for node in seeds:
                    if node.entity_id:
                        # Get ALL appearances of this character
                        activated |= set(
                            self.graph.entity_index.get(node.entity_id, [])
                        )

            elif intent == "SPATIAL":
                for node in seeds:
                    activated |= self._expand_spatial(node, scene_only=True)

            elif intent == "STATE":
                state_nodes = self.graph.get_nodes_by_type("StateChangeNode")
                activated |= {n.id for n in state_nodes
                              if self._matches_query(n, question)}

            elif intent == "SUMMARY":
                # Activate all episode nodes + their direct children
                for ep in self.graph.get_episodes():
                    activated.add(ep.id)
                    activated |= {n.id for n in
                                  self.graph.get_children(ep, depth=1)}

        # Step 3: Include parent context for all activated nodes
        activated |= {self.graph.nodes[nid].parent_id
                      for nid in activated
                      if self.graph.nodes[nid].parent_id}

        # Step 4: Prune to context budget (~60 nodes)
        subgraph = self.graph.induced_subgraph(activated)
        return self._prune_to_budget(subgraph, max_nodes=60)

    def _walk_temporal_spine(self, node: VKGNode, hops: int) -> Set[str]:
        activated = set()
        curr = node
        for direction in ["PRECEDES", "PRECEDES_REV"]:
            c = curr
            for _ in range(hops):
                neighbor = self.graph.get_neighbor(c, direction)
                if not neighbor:
                    break
                activated.add(neighbor.id)
                c = neighbor
        return activated

    def _follow_causal_edges(self, node: VKGNode, depth: int) -> Set[str]:
        activated = set()
        queue = [(node, 0)]
        causal_types = {"CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"}
        while queue:
            curr, d = queue.pop(0)
            if d >= depth:
                continue
            for edge in self.graph.get_edges(curr.id):
                if edge.relation_type in causal_types:
                    neighbor = self.graph.nodes[edge.target_id]
                    activated.add(neighbor.id)
                    queue.append((neighbor, d+1))
            for edge in self.graph.get_incoming_edges(curr.id):
                if edge.relation_type in causal_types:
                    neighbor = self.graph.nodes[edge.source_id]
                    activated.add(neighbor.id)
                    queue.append((neighbor, d+1))
        return activated
```

### 8.3 Context Serializer

Converts the activated subgraph into structured natural language for the VLM.
Format adapts based on detected query intent:

```python
class ContextSerializer:

    def serialize(self, subgraph: SubGraph, question: str, intents: List[str]) -> str:
        sections = []

        # Always include: temporal spine
        timeline = subgraph.get_sorted_events()
        if timeline:
            sections.append("## Timeline")
            sections.extend(
                f"  [{n.t_start:.0f}s–{n.t_end:.0f}s] "
                f"[{n.node_type}] {n.label}"
                + (f" (confidence: {n.confidence:.2f})" if n.confidence < 0.8 else "")
                for n in timeline
            )

        # Causal chains (if causal intent)
        if "CAUSAL" in intents:
            chains = subgraph.get_causal_chains()
            if chains:
                sections.append("\n## Causal Relationships")
                for c in chains:
                    sections.append(
                        f"  [{c.source.t_start:.0f}s] {c.source.label}\n"
                        f"    ──[{c.relation}, conf={c.confidence:.2f}]──▶\n"
                        f"  [{c.target.t_start:.0f}s] {c.target.label}\n"
                        f"    Reason: {c.metadata.get('reasoning', 'not specified')}"
                    )

        # Character appearances (if identity intent)
        if "IDENTITY" in intents:
            chars = subgraph.get_characters()
            if chars:
                sections.append("\n## Characters")
                for ch in chars:
                    times = [f"{a.timestamp:.0f}s" for a in ch.appearances[:8]]
                    sections.append(
                        f"  {ch.label}: {ch.canonical_description}\n"
                        f"    Appears at: {', '.join(times)}"
                    )

        # State changes (if state intent)
        if "STATE" in intents:
            states = subgraph.get_state_changes()
            if states:
                sections.append("\n## State Changes")
                for s in states:
                    sections.append(
                        f"  [{s.t_start:.0f}s] {s.label}: "
                        f"{s.prev_state} → {s.next_state}"
                    )

        # Spatial layout (if spatial intent)
        if "SPATIAL" in intents:
            spatial = subgraph.get_spatial_relations()
            if spatial:
                sections.append("\n## Spatial Layout")
                for r in spatial[:15]:
                    sections.append(
                        f"  [{r.scene_time:.0f}s] {r.source.label} "
                        f"{r.relation_type.lower().replace('_', ' ')} "
                        f"{r.target.label}"
                    )

        # Dialogue
        speeches = subgraph.get_speech_nodes()
        if speeches:
            sections.append("\n## Dialogue")
            for s in speeches[:12]:
                sections.append(f"  [{s.t_start:.0f}s] \"{s.label}\"")

        context = "\n".join(sections)

        return (
            "You are answering a question about a video. "
            "The following knowledge was extracted from the video:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the above knowledge. "
            "Cite specific timestamps as evidence. "
            "If information is insufficient, say so explicitly."
        )
```

### 8.4 Final VLM Answer Call

The critical moment: VLM receives structured text context + actual keyframe images.
Reasoning over both symbols and pixels simultaneously:

```python
def answer_question(
    question: str,
    graph: VKGraph,
    faiss_index,
    frame_store: FrameStore,
    llm: LLM
) -> AnswerResult:

    # Step 1: Classify intent
    intents = classify_intent(question)

    # Step 2: Activate relevant subgraph
    activator = SubgraphActivator(graph, faiss_index, siglip_encoder, config)
    subgraph = activator.activate(question, intents)

    # Step 3: Serialize subgraph to structured text
    context_text = ContextSerializer().serialize(subgraph, question, intents)

    # Step 4: Retrieve keyframe images for visual grounding
    # (cap at 6 frames to stay within context budget)
    visual_nodes = subgraph.get_visual_nodes()
    visual_nodes.sort(key=lambda n: n.confidence, reverse=True)
    keyframes = [frame_store.load(n.keyframe_id)
                 for n in visual_nodes[:6]
                 if n.keyframe_id]

    # Step 5: Build multi-modal prompt
    content = []
    for kf in keyframes:
        content.append({"type": "image_url",
                        "image_url": {"url": kf.to_b64_url()}})
        content.append({"type": "text", "text": f"[t={kf.timestamp:.0f}s]"})
    content.append({"type": "text", "text": context_text})

    # Step 6: Single vLLM call
    output = llm.chat(
        messages=[[
            {"role": "system", "content": "You are a video reasoning assistant."},
            {"role": "user",   "content": content}
        ]],
        sampling_params=QA_SAMPLING
    )[0]

    return AnswerResult(
        answer=output.outputs[0].text,
        intents=intents,
        subgraph_size=len(subgraph.nodes),
        keyframes_used=[kf.timestamp for kf in keyframes],
        evidence_nodes=[n.label for n in subgraph.get_sorted_events()[:5]]
    )
```

---

## 10. Data Storage Layout

```
output/<video_id>/
├── vkg.json           # full graph (nodes + edges), ~5-15MB per 100-min video
├── vkg.index          # FAISS HNSW index, ~10MB for 500 nodes at 1152-d
├── frames.h5          # HDF5 frame store, ~500MB (500 frames at 720p JPEG)
└── meta.json          # video metadata, config, build stats

# In-memory layout during query time:
VKGraph:
  nodes:         Dict[str, VKGNode]         # O(1) lookup by ID
  edges:         Dict[str, List[VKGEdge]]   # adjacency list, outgoing
  edges_in:      Dict[str, List[VKGEdge]]   # adjacency list, incoming
  temporal_idx:  SortedList[VKGNode]        # sorted by t_start
  entity_idx:    Dict[str, List[str]]       # entity_id → [node_ids]
  type_idx:      Dict[str, List[str]]       # node_type → [node_ids]
  node_id_list:  List[str]                  # FAISS index position → node_id
```

---

## 11. Complexity Analysis

### 10.1 Offline Build Time (100-minute video)

| Step | Operation | Complexity | Est. Time (A100) |
|------|-----------|-----------|-----------------|
| Frame sampling | SigLIP on 6K coarse frames | O(N_coarse) | 30s |
| Scene extraction | ~100 vLLM calls, 5 frames each | O(S) | 4 min |
| Whisper transcription | Single pass on full audio | O(T) | 3 min |
| Node creation | Graph construction from JSON | O(N_nodes) | 10s |
| Temporal edges | Sequential scan | O(N_nodes) | 5s |
| Spatial edges | From Qwen output (no computation) | O(N_relations) | 5s |
| Character resolution | DBSCAN on N_chars embeddings | O(N_chars²) | 10s |
| FAISS index build | HNSW construction | O(N log N) | 20s |
| Causal inference | ~15 vLLM calls (one/episode) | O(E) | 1.5 min |
| Semantic edges | FAISS ANN search | O(N log N) | 10s |
| **Total** | | | **~10-12 min** |

### 10.2 Online Query Time

| Step | Operation | Est. Time |
|------|-----------|-----------|
| Intent classification | Keyword match | <1ms |
| FAISS seed search | k=20 ANN query | 5ms |
| Graph traversal | BFS/DFS on subgraph | 20ms |
| Context serialization | String construction | 5ms |
| Frame loading (6 frames) | HDF5 read + encode | 200ms |
| VLM answer generation | Single vLLM call | 1-3s |
| **Total** | | **~1.5-4 seconds** |

### 10.3 Model VLM Call Comparison

| System | Offline Calls | Online Calls | Total for 100 QA |
|--------|--------------|-------------|-----------------|
| Dense VLM captioning | ~6,000 | 1 per QA | ~6,100 |
| HAVEN / DVD | ~500 | 1 per QA | ~600 |
| Multi-model (YOLO+etc.) | 0 VLM | 1 per QA | 100 |
| **Q-VKG** | **~115** | **1 per QA** | **~215** |

Q-VKG's offline investment pays back at >3 questions per video compared to
dense captioning. For benchmarks with many questions per video (Video-MME has
30+ per video), the savings are 30-100×.

---

## 12. Evaluation Plan

### 11.1 Benchmarks

| Benchmark | Task | # Questions | Video Length | Primary Metric |
|-----------|------|-------------|-------------|---------------|
| **Video-MME** | Multi-task VQA | 30/video | 30s–60min | Accuracy |
| **MLVU** | Long-form understanding | 10/video | 3–120min | Accuracy |
| **LongVideoBench** | Long-form QA | 20/video | 1–60min | Accuracy |
| **NExT-QA** | Causal + temporal QA | 5/video | 1–2min | WUPS |
| **EgoSchema** | Egocentric QA | 1/video | 3min | Accuracy |
| **MovieChat-1K** | Movie dialogue QA | 10/video | 100+min | GPT-4 score |

### 11.2 Ablation Studies

| Ablation | What Is Removed | Expected Impact |
|----------|----------------|----------------|
| No causal edges | Remove CAUSES/ENABLES edges | −8–12% NExT-QA causal split |
| No character resolution | Treat each mention as new entity | −10–15% identity questions |
| Uniform sampling vs. HTS | Replace adaptive with uniform 1fps | −8–12% long-video benchmarks |
| No subgraph activation | Pass full graph context | ±accuracy, −50% token efficiency |
| No keyframes in QA call | Text context only, no images | −5–10% visual detail questions |
| 7B vs. 72B Qwen | Model capacity | +5–15% across all tasks |
| Description ReID vs. ArcFace | Character resolver quality | −5% identity on MovieChat |

### 11.3 Graph Quality Evaluation

Evaluate graph construction quality independently from downstream QA:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Entity recall | % of human-annotated entities in VKG | >80% |
| Causal precision | Human eval on random 50 causal edges | >75% correct |
| Timeline ordering | % of event pairs correctly ordered | >90% |
| Character ReID accuracy | % of cross-scene links correct | >70% |
| OCR accuracy | Character error rate vs. ground truth | <10% CER |

### 11.4 Efficiency Benchmarks

| Metric | Target |
|--------|--------|
| Offline build time (100-min video, 1× A100) | <15 min |
| Online query latency (7B, A100) | <4 seconds |
| VRAM during build | <20GB (7B + SigLIP + Whisper) |
| VRAM during query | <16GB (7B + FAISS) |
| VKG file size (100-min video) | <20MB (graph + index, excl. frames) |

---

## 13. Comparison with SOTA

| System | Venue | Typed KG? | Causal edges? | Audio-free identity? | Open-source? | Online VLM calls |
|--------|-------|-----------|--------------|---------------------|-------------|-----------------|
| HAVEN (Yin et al.) | arXiv Jan 2026 | ✗ DB | ✗ | ✗ audio-only | ✗ GPT-4.1 | O(S) agentic |
| DVD / Deep Video Disc. | arXiv May 2025 | ✗ DB | ✗ | ✗ | ✗ GPT-4.1+o3 | O(S) tool loop |
| Symphony | arXiv Mar 2026 | ✗ | ✗ | ✗ | ✗ | O(S) multi-agent |
| MR.Video | NeurIPS 2025 | ✗ | ✗ | ✗ | ✗ | O(S) |
| VCA | ICCV 2025 | ✗ | ✗ | ✗ | ✗ | O(S) tree search |
| AVA (EKG) | NDSI 2026 | Partial EKG | ✗ | ✗ | Partial | O(S) agentic |
| **Graph-to-Frame RAG** | **CVPR 2026** | **✓ dual-view** | **Implicit** | **✗** | **✗ GPT-4o** | **O(1)** |
| Vgent | NeurIPS 2025 | Partial | ✗ | ✗ | Partial | O(1) RAG |
| MAGIC-Video | arXiv 2026 | ✓ 6-type | ✗ | ✓ (ego only) | ✓ | O(1) agentic |
| MECD+ | IEEE TPAMI 2025 | ✓ causal DAG | ✓ (short clips) | ✗ | ✓ | N/A (offline task) |
| LLoVi / VideoTree | CVPR / 2024 | ✗ | ✗ | ✗ | Partial | O(1) |
| **Q-VKG (ours)** | — | **✓ 20+ typed** | **✓ long video** | **✓ general domain** | **✓ Qwen3-VL** | **O(1)** |

**Reading the table**: Graph-to-Frame RAG (CVPR 2026) is the primary baseline to
beat — it has typed KG + O(1) online calls, but lacks audio, character tracking,
causal inference rigor, and open-source reproducibility. MECD+ does causal edges
but only for short clips with a specialized trained model. No paper does all five
columns simultaneously.

---

## 14. Novel Contributions

> **Framing note for paper writing**: Graph-to-Frame RAG (CVPR 2026) shares our
> offline-KG + online-retrieval architecture. It must be cited as the primary
> baseline. Our novelty is a combination of five properties no single prior work
> has: open-source pipeline, LLM-grounded causal edges on long video, audio-free
> identity, state-change tracking, and deterministic intent-driven activation.
> Claiming any one of these alone as "first" is incorrect; claiming the combination
> is defensible.

### Contribution 1 — Visually-Grounded LLM Causal Inference on Long Video *(primary)*

**Claim**: First pipeline to infer and store explicit causal edges on general-domain
long video (100+ min) using LLM visual grounding, with no task-specific training.

**Prior work gap**:
- MECD+ builds causal edges but uses a trained Granger-causality model on short
  clips (4–11 events, <2 min). Not applicable to 100-min open-domain video.
- Graph-to-Frame RAG has an "event-causal view" but provides no explicit LLM
  inference step — causal structure is implicit in the graph, not a dedicated
  component with confidence scores.
- All other long-video methods (HAVEN, DVD, Symphony, MR.Video) have no causal
  edges at all.

**Our method**: Episode-level batch vLLM calls where Qwen3-VL sees both keyframe
images and the extracted event timeline, producing confidence-scored causal edges
(CAUSES / ENABLES / PREVENTS / MOTIVATES). Training-free. Zero-shot on any domain.

---

### Contribution 2 — Fully Open-Source Reproducible Pipeline *(practical)*

**Claim**: First competitive long-video KG system that runs entirely on open-weight
models (Qwen3-VL + Whisper + SigLIP) with no proprietary API dependency.

**Prior work gap**: HAVEN uses GPT-4.1. DVD uses GPT-4.1 + o3. Graph-to-Frame RAG
uses GPT-4o + GPT-4o-mini. AVA uses unspecified VLMs. No competitive pipeline is
fully reproducible on accessible hardware.

**Impact**: Enables academic reproducibility, ablation studies, and deployment
without per-query API costs. The vLLM batched inference reduces offline build cost
to ~10-12 min / 100-min video on 2× A100.

---

### Contribution 3 — Audio-Free Visual Character Identity Resolution *(fills HAVEN gap)*

**Claim**: Cross-scene character identity resolution that works for silent video,
dubbed content, and any language — without speaker diarization.

**Prior work gap**:
- HAVEN's entity cohesion requires diarizable audio. Silent films, music videos,
  dubbed foreign-language content all fail.
- MAGIC-Video tracks entities via temporal biographies but only for egocentric
  first-person video.
- No paper resolves character identity in general-domain long video using visual
  descriptions alone.

**Our method**: Qwen3-VL generates rich appearance descriptions per character
appearance; SigLIP text embeddings + DBSCAN cluster them into persistent
CharacterNodes. Evaluated via cross-scene identity accuracy on MovieChat-1K.

---

### Contribution 4 — StateChangeNode: Direct Entity State Tracking *(novel node type)*

**Claim**: Explicit modeling of entity state transitions (entity: prev\_state →
next\_state) as queryable graph nodes, enabling O(1) lookup for "when did X change?"
questions.

**Prior work gap**: No video KG paper — including Graph-to-Frame RAG, MAGIC-Video,
EgoGraph, or AVA — has an explicit state-change node type. State queries in prior
work require repeated frame inspection (DVD Frame Inspect trap) or agentic search.

**Our method**: Qwen3-VL's guided JSON output includes `state_changes` as a
first-class field. Each change becomes a `StateChangeNode` with `prev_state`,
`next_state`, and grounded timestamp. Enables direct index lookup.

---

### Contribution 5 — Deterministic Intent-Driven Subgraph Activation *(engineering)*

**Claim**: A keyword-based intent classifier routes queries to typed graph traversal
strategies (temporal BFS, causal chain following, entity expansion, spatial
neighborhood), avoiding the "trap" failure modes documented in DVD.

**Prior work gap**:
- DVD documents "Frame Inspect Trap" and "Clip Search Trap" — the agentic loop
  degrades when the reasoning LLM makes poor tool choices.
- HAVEN's agentic navigation makes multiple GPT-4.1 calls per query.
- Graph-to-Frame RAG uses GPT-4o-mini to map queries to subgraph queries
  (LLM-dependent, failure-prone on edge cases).

**Our method**: Deterministic keyword intent classification + FAISS seed search +
typed BFS/DFS. No LLM call for retrieval. No failure modes from LLM planning.
Latency: <30ms for subgraph activation vs. 2-5s for LLM-based routing.

---

### Contribution 6 — Comprehensive Typed Multimodal Edge Taxonomy *(architectural)*

**Claim**: A 6-category, 20+ type edge taxonomy (temporal, hierarchical, entity,
spatial, causal, cross-modal) that enables multi-hop queries spanning modalities —
e.g., "What speech was recorded while the object mentioned in the sign was on screen?"

**Prior work gap**:
- Graph-to-Frame RAG has dual-view nodes but no explicit edge taxonomy.
- MAGIC-Video has 6 typed edges but no cross-modal (audio/OCR) edges and no causal.
- Action Genome has spatial edges but only for short clips, no audio/causal.
- No single prior work covers temporal + spatial + causal + cross-modal + entity
  + hierarchical edges in one graph.

---

### What Is NOT a Novel Contribution (honest accounting)

| Claim | Why it's not novel | What to say instead |
|-------|-------------------|---------------------|
| "Offline KG + online subgraph retrieval" | Graph-to-Frame RAG (CVPR 2026) does this | "We extend the offline-KG paradigm of [G2F-RAG] with..." |
| "Hierarchical temporal structure" | HAVEN, DVD, MR.Video, VideoTree all use hierarchies | "Our 4-level temporal backbone serves as the graph spine, unlike flat hierarchies in prior work" |
| "Multi-frame per scene VLM extraction" | HAVEN uses 20 frames/segment | Frame this as part of the vLLM efficiency contribution |
| "Character identity tracking" | HAVEN (audio), MAGIC-Video (ego), EgoGraph (ego) all do this | "First audio-free, general-domain character resolver" |
| "Graph-based video RAG" | Vgent (NeurIPS 2025), Graph-to-Frame RAG | "We add causal, cross-modal, and state-change dimensions absent from prior graph RAG" |

---

## 15. Implementation Timeline

| Phase | Duration | Deliverables | Key Risk & Mitigation |
|-------|----------|--------------|----------------------|
| **Phase 1: Core Pipeline** | 2 weeks | VKGraph + FrameStore + SubGraph data structures, HTS sampler, Whisper integration | Write unit tests for VKGraph before building on top |
| **Phase 2: VKG Construction** | 3 weeks | vLLM scene extraction, full node/edge pipeline, FAISS index, character resolver, causal inference | Guided decoding failures → fallback regex + retry |
| **Phase 3: Query Interface** | 2 weeks | Intent classifier, subgraph activator, serializer, single-call QA | Context budget overflow → tune max_nodes per intent type |
| **Phase 4: Evaluation** | 3 weeks | Video-MME / MLVU / NExT-QA / MovieChat results, ablation study (no causal edges, no audio, no state nodes) | SOTA gap → ablation isolates weak component |
| **Phase 5: Paper** | 4 weeks | CVPR / NeurIPS / ICLR submission | Reviewer asks "vs. Graph-to-Frame RAG?" → §2.2 diff table; "vs. MECD+?" → §2.3 diff table |

**Total: 14 weeks (~3.5 months)**

---

## 16. Open Questions & Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| 7B vs. 72B Qwen3-VL | 7B for experiments, 72B for final eval | Speed vs. quality tradeoff; 72B adds ~8× build time |
| Description ReID vs. ArcFace | Description-based default, ArcFace optional | Fewer models; add ArcFace only if MovieChat results are weak |
| Whisper tiny vs. large-v3 | large-v3 | Runs once per video; quality matters for cross-modal edges |
| vLLM version | Pin to tested version (0.8.x) | Multi-image API is evolving; avoid breaking changes |
| Graph storage: NetworkX vs. Neo4j | NetworkX + JSON | Simpler deployment; Neo4j only if video library grows to 100+ |
| FAISS IVF vs. HNSW | HNSW | No training step needed; better recall at this scale (500-5K nodes) |
| Causal hallucination | Confidence threshold 0.6 + human eval on 10% sample | Filters noise; measures precision separately from QA accuracy |
