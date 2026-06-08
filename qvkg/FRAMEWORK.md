# Q-VKG: Query-Conditioned Video Knowledge Graphs for Long-Video Reasoning

*A framework combining offline multimodal graph construction with planner-guided,
agentic graph traversal for question answering over hour-long video.*

---

## Abstract

Long-video question answering stresses two failure points of end-to-end
video–language models: (i) the quadratic cost of attending to thousands of
frames, and (ii) the loss of *long-range, cross-modal, and causal* structure
when a video is flattened into a token stream. We present **Q-VKG**, a framework
that decouples *perception* from *reasoning*. Offline, a single multimodal pass
distills a video into a typed, timestamped **Video Knowledge Graph (VKG)** whose
nodes are scenes, events, objects, state-changes, characters, speech, subtitles,
and on-screen text, and whose edges encode temporal order, containment, spatial
relations, cross-modal co-occurrence, semantic similarity, and causality. Online,
a question is answered by a **two-stage agentic traversal**: a text-only *planner*
emits a structured retrieval plan over the graph, a *deterministic executor*
assembles a compact, relevance-ranked subgraph plus a small set of on-demand
frames, and a vision–language *answerer* produces the final response. The design
yields three properties reviewers can check directly: the expensive perception
runs **once per video** and is amortized across all questions; inference is
**deterministic and reproducible** (no open-ended tool-calling loop at test time);
and every answer is **grounded** in explicit, inspectable graph evidence with
timestamps. We instantiate Q-VKG with Qwen-VL served by vLLM and evaluate on
LVBench.

---

## 1. Introduction

A 90-minute video at 25 fps is ~135k frames. Two dominant paradigms struggle
here. **Uniform-sampling VLMs** subsample to a fixed budget (e.g. 32–256 frames)
and read them jointly; they discard most of the video and have no explicit memory
of *when* or *in what order* things happened. **Caption-then-retrieve** pipelines
summarize clips into text and run RAG; they recover scalability but lose visual
detail, identity tracking across scenes, and causal/temporal relations that the
question may hinge on.

We argue the right intermediate representation is an explicit, **typed multimodal
graph** built once and queried many times. The graph makes long-range structure
*first-class*: temporal ordering is an edge, an object's reappearance 40 minutes
later is an entity link, "X happened because of Y" is a causal edge. A question
then becomes a *traversal* of this structure rather than a re-reading of the video.

**Contributions.**

1. **A heterogeneous, timestamped VKG schema** (10 node types, 6 relation
   families) spanning vision, audio (ASR), subtitles, and on-screen text, with a
   hierarchical Video→Episode→Scene→Clip backbone and explicit causal edges
   (§3, §4).
2. **A query-conditioned two-stage traversal** that separates a cheap text-only
   *planner* from a *deterministic* context-assembly executor and a single
   *answerer* call — avoiding the latency, nondeterminism, and cost of iterative
   tool-calling agents while retaining their selectivity (§5).
3. **Text-aware perception for "text-referred" questions**: caption-aware frame
   sampling plus first-class subtitle-track ingestion and OCR fusion, targeting a
   question class that flattened VLMs systematically miss (§4.1, §4.3).
4. **Context-assembly principles** that materially affect accuracy on long video:
   *relevance-then-chronology* ("cap-before-sort"), temporal cluster expansion,
   and VLM frame-batching with context propagation (§5.2–§5.3).
5. An **amortized, reproducible** systems instantiation (vLLM continuous batching,
   prefix-cached shared prompts, schema-guided JSON decoding, lazy engine load),
   with ablations isolating each design choice (§6).

---

## 2. Related Work (condensed)

- **Long-video VLMs / token reduction** (memory tokens, token merging, streaming
  attention): scale the *reader*, not the *representation*; no explicit temporal
  or causal index.
- **Video RAG / caption graphs**: scalable but text-lossy; identity and causality
  are not modeled as structure.
- **Scene graphs / spatio-temporal graphs**: typically per-clip and spatial; we
  add cross-scene entity resolution, episode hierarchy, ASR/subtitle/OCR fusion,
  and causal edges, and we *use* the graph for QA rather than only for frame
  selection.
- **LLM agents over tools**: powerful but expensive and nondeterministic at
  inference; we keep the *planning* benefit while making *execution* deterministic.

---

## 3. The Video Knowledge Graph

A VKG is a directed, attributed graph $G=(V,E)$.

### 3.1 Nodes

Each node $v\in V$ carries a type, a natural-language `label`, a temporal extent
$[t_\text{start}, t_\text{end}]$, a hierarchy `level`, optional modality-specific
attributes, and a `confidence`. Node types:

| Level | Node type | Content |
|-------|-----------|---------|
| 3 | `VideoNode` | root |
| 2 | `EpisodeNode` | coherent multi-scene segment (narrative unit) |
| 1 | `SceneNode` | shot/scene bounded by visual change |
| 0 | `ClipNode` | fine visual unit within a scene |
| 0 | `ActionNode` | an action/event with participants |
| 0 | `ObjectNode` | a detected object (+ bbox, attributes) |
| 0 | `StateChangeNode` | `prev_state → next_state` of an entity/attribute |
| 0 | `CharacterNode` | a resolved person identity (video-wide) |
| 0 | `SpeechNode` | a timestamped utterance (ASR **or** subtitle, tagged by `source`) |
| 0 | `OCRNode` | on-screen text, with `semantic_type` (name / caption / price / …) |

### 3.2 Edges

Edges $e=(u,v,r)$ use relation family $r$:

| Relation | Meaning | Built from |
|----------|---------|-----------|
| `CONTAINS` | hierarchical / temporal containment | backbone (§4.2) |
| temporal `BEFORE/AFTER` | chronological order | timestamps (§4.2) |
| spatial | `subject–relation–object` in a frame | VLM extraction (§4.3) |
| `SPOKEN_BY` | utterance → speaker | speaker attribution (§4.5) |
| `LABELS` | OCR text → object/scene it annotates | OCR fusion (§4.5) |
| `SAME_ENTITY` | cross-scene identity | entity resolution (§4.5) |
| `SIMILAR_TO` / `DESCRIBES` | semantic neighbors | FAISS kNN (§4.5) |
| causal | cause → effect | causal inference (§4.6) |

An `entity_idx` maps each resolved `entity_id` to the list of node
appearances, giving O(1) cross-scene object/character tracking — the substrate
for "track X across scenes" and "before/after" relational queries.

---

## 4. Offline VKG Construction

Construction is an 11-step pipeline (orchestrated in `builder.py`), checkpointed
per step for resumability. The expensive perception (the VLM) is invoked in a
**small number of large batched calls**, not per-scene loops.

```
Algorithm 1: BuildVKG(video)
  0. video-type detection (VLM, 1 call)
  1. hierarchical frame sampling                → keyframes, scene boundaries
  2. ASR transcription (faster-whisper)         → SpeechNode(source=asr)
  2b. subtitle-track ingestion (.srt/.vtt)      → SpeechNode(source=subtitle)
  3. scene extraction (VLM, ONE batched call)   → objects/actions/OCR/chars/relations
  ·  episode segmentation (VLM)
  4. node creation (backbone, entities, events, perception)
  5. temporal + hierarchical edges
  6. spatial edges
  7. cross-modal edges + speaker attribution + OCR-semantic edges
  8. character resolution (LLM-native, DBSCAN fallback)
  9. semantic edges (FAISS kNN)
 10. causal inference (VLM, ONE batched call)   → cause→effect DAG
  ·  build FAISS index; serialize G
```

### 4.1 Hierarchical, caption-aware frame sampling (`sampler.py`)

Frames are extracted at ~1 fps as a coarse pool. Each frame is scored by a
multi-signal importance function

$$
s(f_i)=0.25\,d_\text{hist}+0.25\,d_\text{sem}+0.15\,m_\text{flow}+0.15\,a_\text{rms}+0.20\,\tau(f_i),
$$

combining color-histogram change, SigLIP embedding distance, optical-flow
magnitude, audio RMS energy, and a **text-region score** $\tau$. $\tau$ detects
on-screen text via gradient → Otsu → horizontal morphological closing, counting
wide/short line-shaped blobs and up-weighting the lower third (where lower-thirds
and name captions live). Scene boundaries are detected by a hard
histogram threshold and a soft embedding threshold; per scene, keyframes are
selected greedily by score with min/max temporal-gap constraints and a
coverage-gap fill pass, then rebalanced to a global budget. A **question-aware**
pass additionally densifies sampling at benchmark-provided time references.

*Why $\tau$ matters:* names of speakers in talk-shows, scores in sports, and
prices in vlogs appear as transient captions that uniform sampling drops; $\tau$
biases those frames into the keyframe set so they reach OCR.

### 4.2 Temporal & hierarchical backbone

A `VideoNode → EpisodeNode → SceneNode → ClipNode` containment tree gives the
multi-resolution index used by the planner. Adjacent same-level nodes are linked
with temporal `BEFORE/AFTER` edges; because every node carries
$[t_\text{start},t_\text{end}]$, *ordering queries reduce to timestamp
comparisons*.

### 4.3 Scene extraction (the perception core)

For each scene, its keyframes + a shared system prompt are sent to the VLM, which
returns **schema-guided JSON**: scene label, objects (with bboxes/attributes),
actions, `spatial_relations` (subject–relation–object), characters, and crucially
`ocr_text` + `ocr_semantics` (each on-screen string tagged name/caption/price/…).
All scenes are submitted in **one** `llm.chat` call; vLLM continuous-batches them
and reuses the shared prompt's KV via prefix caching.

### 4.4 Audio, subtitle, and OCR text fusion

Three text channels are unified as `SpeechNode`/`OCRNode` with a `source` tag:
**ASR** (faster-whisper), **subtitle track** (.srt/.vtt parsed to timestamped
cues — authoritative wording/timing for *text-referred* questions when present),
and **OCR** (burned-in captions). This is the key to the "Text-referred" question
family (event/object/attribute *while a subtitle is shown*), which pure-visual
pipelines cannot ground.

### 4.5 Cross-modal structure & entity resolution

Speaker attribution links utterances to characters (`SPOKEN_BY`); OCR strings are
attached to the object/scene they annotate by bbox-IoU and LLM semantics
(`LABELS`); FAISS kNN over node embeddings adds `SIMILAR_TO`/`DESCRIBES`. **Character
resolution** clusters per-scene person mentions into video-wide identities via an
LLM-native resolver (all description chunks submitted in a single batched call),
with a description-similarity DBSCAN fallback, populating `entity_idx`.

### 4.6 Causal inference

Per episode, the VLM is asked to infer a binary cause→effect DAG over the
episode's events (one batched call across episodes), with confidence thresholding
and cycle/length guards. These edges support "why" questions without test-time
re-watching.

---

## 5. Online Reasoning: Query-Conditioned Agentic Traversal

We answer a question $q$ (optionally with a type and a time reference) by a
**two-stage** procedure that keeps the *selectivity* of an agent while making
execution deterministic.

```
Algorithm 2: Answer(q, G, video)
  Stage 1 — Planner (text-only LLM):
      input  : episode summaries + q (+ type, time-ref)
      output : plan = { windows[], search_queries[], needs_frames, reasoning }
  Stage 2 — Deterministic executor (no LLM):
      assemble subgraph S ⊆ G  and  frame set F      (Algorithm 3)
  Stage 3 — Answerer (VLM):
      serialize(S) + F  →  answer
```

### 5.1 Planner

A text-only LLM sees the compact episode index and the question and emits a JSON
plan: which **time windows** to inspect, which **semantic search queries** to
issue, and whether **frames** are needed. The planner *plans*; it never sees
pixels and never executes retrieval — making Stage 1 cheap and cacheable.

### 5.2 Deterministic context assembly (`_assemble_context`)

```
Algorithm 3: AssembleContext(q, plan, G, video)
  1. time-reference window  → node-range lookup (priority block t_nodes)
  2. for each search query  → FAISS semantic retrieval
  3. MCQ option searches + entity-introduction expansion   (coverage)
  4. temporal cluster expansion (±30s around dense clusters)
  5. keyword re-rank (deduped) ; CAP to MAX_NODES *before* chronological sort
  6. chronological sort for display ; serialize node lines
  7. frames: span-proportional, de-clustered extraction at the time-ref window
            + top visual search nodes (60s-bucketed)
```

Two principles are load-bearing:

- **Relevance-then-chronology ("cap-before-sort").** Selecting by relevance, then
  *capping*, then sorting by time — never sort-then-cap — prevents dense early
  content from evicting the actually-relevant late-video nodes. Time windows are
  filtered by `t_start` (not span) to avoid pulling in video-wide nodes.
- **Coverage for distractor-rich MCQs.** Searching each answer option (and each
  named entity's introduction) guarantees rare, late-appearing candidates are
  represented in the subgraph.

Frames are extracted **on demand** from the raw video at assembly time:
de-clustered and **span-proportional** so a multi-minute window yields evenly
spaced coverage (needed for "last to appear / order" and outfit/identity
questions) rather than a redundant burst at one timestamp.

### 5.3 Answerer with frame-batching + context propagation

When the retrieved frame set exceeds the VLM's per-prompt image limit, frames are
split into **overlapping batches**, each carrying the full graph context and a
*continuity hint* (the previous batch's analysis); per-batch answers are merged by
majority vote. This lets the answerer cover more of a long window than a single
prompt allows while preserving temporal continuity across batches.

### 5.4 Fully-agentic variants (ablation arms)

For comparison, the same graph supports two open-ended agents over a shared tool
set — `list_episodes`, `search_graph`, `get_scene_detail(t0,t1)`,
`extract_frames(t0,t1)`: a **function-calling agent** and a **ReAct**
(Thought/Action/Observation) loop. These let us measure what the deterministic
two-stage executor gives up (if anything) relative to iterative traversal, at
known cost in latency and reproducibility.

---

## 6. Experiments (protocol)

**Benchmarks.** LVBench (hour-scale); the framework targets the L1-Perception and
L2-Relation question taxonomy (scene/object/event/text-referred, single- and
multi-moment). Subtitle ingestion is exercised on subtitle-bearing corpora
(e.g. LongVideoBench); LVBench is evaluated without subtitles.

**Models / serving.** Qwen-VL (4B/9B) served by vLLM (continuous batching,
`max_num_seqs=256`, prefix caching, schema-guided JSON, paged-attention KV at
0.90 GPU utilization), faster-whisper for ASR, SigLIP for frame embeddings; a
single Blackwell-class GPU. The engine is **lazy-loaded** so sampling/ASR run with
the GPU free.

**Metric.** MCQ accuracy, reported per question category to expose where structure
helps (relation/causal) vs. where perception caps accuracy (object-existence).

**Ablations (each isolates one claim).**
1. two-stage vs. function-calling agent vs. ReAct (accuracy / latency / determinism);
2. cap-before-sort vs. sort-then-cap;
3. with/without temporal cluster expansion;
4. with/without caption-aware sampling $\tau$ and subtitle/OCR fusion (text-referred slice);
5. span-proportional de-clustered frames vs. fixed sparse sampling;
6. frame-batching + context propagation vs. truncation to the image limit.

**Reproducibility.** Greedy/low-temperature decoding, checkpointed deterministic
build, and per-question debug records (plan, retrieved nodes, frame timestamps,
serialized context, model rationale) are emitted for every answer, enabling exact
error attribution (model vs. retrieval vs. frame-selection).

---

## 7. Limitations & Honest Failure Analysis

- **Perception ceiling.** Any object/attribute/tracking answer is upper-bounded by
  what the build captured; a frame never sampled is unrecoverable at query time.
  Caption-aware sampling raises, but does not remove, this ceiling.
- **Chunked entity resolution** can split an identity that appears in different
  description chunks.
- **Boundary effects.** Strict time-window adherence can miss an event that
  completes just outside the referenced range; we mitigate with window-tail
  extension and an answerer instruction to treat windows as approximate.
- **Text channels.** Without a subtitle track, text-referred questions fall back to
  ASR+OCR, which diverge from exact subtitle wording and miss fast captions.
- **Build cost.** Perception is paid once but is non-trivial; it amortizes only
  when a video is queried multiple times.

---

## 8. Why this should interest the community

Q-VKG is a concrete argument that **long-video understanding is a representation
problem, not only a context-length problem**: precomputing an explicit, typed,
causal, multimodal graph turns hour-scale QA into bounded graph traversal with
inspectable, timestamped evidence — cheaper, more reproducible, and more
debuggable than both flattened VLMs and free-form tool-using agents, while
remaining model-agnostic.

---

### Appendix A — Component map (for reproduction)

| Paper section | Module |
|---------------|--------|
| §3 schema | `qvkg/schema.py` |
| §4 pipeline | `qvkg/builder.py` |
| §4.1 sampling, $\tau$ | `qvkg/sampler.py` |
| §4.3 extraction, schema-guided JSON | `qvkg/extraction.py`, `qvkg/vllm_client.py` |
| §4.4 subtitles | `qvkg/subtitle.py` |
| §4.5 entities | `qvkg/character.py`, `qvkg/faiss_index.py` |
| §4.6 causal | `qvkg/causal.py` |
| §5 two-stage | `qvkg/query/two_stage.py` |
| §5.4 agents | `qvkg/query/agent.py`, `qvkg/query/react_agent.py` |
| §6 serving | `qvkg/vllm_client.py` (`LazyLLM`, batched `llm.chat`) |
