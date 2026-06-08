# Q-VKG Architecture

## System Overview

```mermaid
flowchart TB
    subgraph OFFLINE["Offline Phase — VKG Construction (once per video)"]
        direction TB
        VIDEO[(Video.mp4)] --> SAMPLER[Hierarchical Frame Sampler]
        VIDEO --> WHISPER[Whisper large-v3<br/>Audio Transcription]
        
        SAMPLER --> KEYFRAMES[Keyframes + Scenes + Episodes]
        KEYFRAMES --> SCENE_EXTRACT[Qwen3-VL via vLLM<br/>Scene Extraction]
        
        WHISPER --> SPEECH[SpeechNodes]
        
        SCENE_EXTRACT --> NODE_BUILD[Node Creation]
        SPEECH --> NODE_BUILD
        
        NODE_BUILD --> |TemporalBackbone<br/>EntityNodes<br/>EventNodes<br/>PerceptionNodes| GRAPH[(VKGraph<br/>nodes + edges)]
        
        GRAPH --> EDGE_BUILD[Edge Construction]
        EDGE_BUILD --> |Temporal<br/>Hierarchical<br/>Spatial<br/>Cross-modal| GRAPH
        
        GRAPH --> CHAR_RESOLVE[DescriptionBasedCharacterResolver<br/>DBSCAN clustering]
        CHAR_RESOLVE --> |Resolved CharacterNodes| GRAPH
        
        GRAPH --> FAISS_BUILD[FAISS HNSW Index<br/>SigLIP text + image embeddings]
        FAISS_BUILD --> FAISS[(vkg.index)]
        
        GRAPH --> CAUSAL[Qwen3-VL<br/>Causal Chain Inference]
        CAUSAL --> |CAUSES/ENABLES/PREVENTS/MOTIVATES| GRAPH
        
        GRAPH --> SEMANTIC[FAISS ANN<br/>Semantic Edges]
        SEMANTIC --> |SIMILAR_TO| GRAPH
        
        GRAPH --> VKG_JSON[(vkg.json)]
        KEYFRAMES --> HDF5[(frames.h5)]
    end

    subgraph ONLINE["Online Phase — QA (per question, ~2-5s)"]
        direction TB
        Q[Question] --> INTENT[Intent Classifier]
        Q --> ACTIVATOR[Subgraph Activator<br/>FAISS + Typed BFS]
        
        INTENT --> ACTIVATOR
        
        ACTIVATOR --> SUBGRAPH[Activated Subgraph<br/>≤60 nodes]
        
        SUBGRAPH --> CONTEXT[Context Serializer<br/>Graph → Structured NL]
        SUBGRAPH --> FRAMES[Frame Selector<br/>Prioritise question-seeded keyframes]
        
        CONTEXT --> CTXT[Structured Context Text]
        FRAMES --> KF[Keyframe Images]
        
        Q --> PROMPT_BUILD[Build Multimodal Prompt]
        CTXT --> PROMPT_BUILD
        KF --> PROMPT_BUILD
        
        PROMPT_BUILD --> VLM[Qwen3-VL via vLLM<br/>Single call]
        VLM --> ANSWER[Answer + Evidence]
    end

    subgraph AGENTIC["QA Variants"]
        direction TB
        TWO_STAGE[Two-Stage QA<br/>Stage 1: LLM Planner → Plan<br/>Stage 2: Execute plan + VLM Answer]
        AGENT[Agentic QA<br/>VLM drives iterative<br/>graph search with tools]
    end

    OFFLINE --> |saved to disk| ONLINE
    ONLINE --> TWO_STAGE
    ONLINE --> AGENT
```

## Model Stack

```mermaid
flowchart LR
    subgraph MODELS["Three Models"]
        QWEN[Qwen3-VL 4B/7B<br/>via vLLM<br/>Scene extraction<br/>Causal inference<br/>Episode segmentation<br/>QA answering]
        WHISPER[Whisper large-v3<br/>GPU<br/>Audio transcription<br/>Word timestamps]
        SIGLIP[SigLIP-SO400M<br/>GPU<br/>Text + image embeddings<br/>FAISS indexing]
    end
    
    subgraph STORAGE["Storage"]
        HDF5[frames.h5<br/>JPEG-compressed keyframes]
        JSON[vkg.json<br/>Graph state]
        FAISS_FILE[vkg.index<br/>HNSW index]
    end
    
    QWEN --> STORAGE
    SIGLIP --> FAISS_FILE
```

## Data Flow — Step by Step

| Step | Component | Input | Output | Time (100 min video) |
|------|-----------|-------|--------|---------------------|
| 1 | HierarchicalSampler | Video | ~500 keyframes, ~206 scenes | ~8-15 min |
| 2 | Whisper | Video | SpeechNodes (timestamped transcript) | ~2-5 min (overlaps with Step 1) |
| 3 | Scene Extraction (Qwen) | Keyframes + scenes | SceneData (labels, objects, actions, OCR, spatial) | ~2-3 min |
| 3b | Episode Segmentation (Qwen) | Scene labels | EpisodeNodes (10-30 episodes) | ~10 sec |
| 4-8 | Node/Edge builder | SceneData | Fully connected VKGraph | ~30 sec |
| 9 | FAISS builder | Graph nodes | vkg.index (HNSW) | ~10 sec |
| 10 | Causal inference (Qwen) | Episodes + keyframes | CausalEdges | ~1-2 min |
| 11 | Semantic edges (FAISS) | FAISS index | SIMILAR_TO edges | ~5 sec |

## Edge Taxonomy

```
Temporal:    PRECEDES, OVERLAPS, DURING
Hierarchy:   CONTAINS, INSTANCE_OF
Entity:      SAME_ENTITY, PERFORMS, INTERACTS_WITH, LOCATED_IN
Spatial:     LEFT_OF, RIGHT_OF, ABOVE, BELOW, IN_FRONT_OF, BEHIND, NEAR
Causal:      CAUSES, ENABLES, PREVENTS, MOTIVATES
Semantic:    SIMILAR_TO, CONTRADICTS
Cross-modal: DESCRIBES, MENTIONS, LABELS, ACCOMPANIES
```

## Query Paths

```mermaid
flowchart LR
    subgraph PATHS["Three QA Approaches"]
        BASIC[Basic QA<br/>Intent → FAISS/BFS<br/>→ Subgraph → VLM]
        TWO_STAGE[Two-Stage<br/>Planner LLM → Plan<br/>→ Execute → VLM]
        AGENT[Agentic QA<br/>VLM with tools<br/>→ Iterative search]
    end
    
    BASIC --> |~2-5s, deterministic| ANS1[Answer]
    TWO_STAGE --> |~3-8s, structured| ANS2[Answer]
    AGENT --> |~5-15s, flexible| ANS3[Answer]
```

## Project Layout

```
qvkg/
├── qvkg/                     ← library code
│   ├── builder.py            ← 11-step VKG construction orchestrator
│   ├── sampler.py            ← hierarchical frame sampling (CPU + SigLIP)
│   ├── extraction.py         ← VLM scene extraction (Qwen via vLLM)
│   ├── causal.py             ← LLM causal chain inference
│   ├── character.py          ← DBSCAN description-based character resolution
│   ├── episode.py            ← LLM episode segmentation
│   ├── schema.py             ← VKGNode, VKGEdge, VKGraph, SubGraph
│   ├── vllm_client.py        ← LLM/SigLIP factory, sampling params, JSON schemas
│   ├── faiss_index.py        ← FAISS HNSW index + semantic edges
│   ├── frame_store.py        ← HDF5 keyframe storage (JPEG-compressed)
│   └── query/                ← online QA pipeline
│       ├── qa.py             ← single-call answer_question()
│       ├── intent.py         ← question intent classification
│       ├── activator.py      ← subgraph activation (FAISS + typed BFS)
│       ├── serializer.py     ← graph → structured NL context
│       ├── frame_extractor.py← on-demand frame extraction from raw video
│       ├── two_stage.py      ← planner → execute → answer
│       ├── agent.py          ← VLM-driven iterative graph search
│       └── react_agent.py    ← ReAct-style agent for complex queries
├── scripts/
│   ├── build_vkg.py          ← offline VKG construction CLI
│   ├── query_vkg.py          ← online QA CLI
│   ├── eval_lvbench.py       ← LVBench evaluation runner (resume-safe)
│   └── verify_flash_attn.py  ← flash-attn smoke test
├── configs/default.yaml      ← default configuration
├── setup.py                  ← Python package install
├── requirements.txt          ← pip dependencies
├── SETUP.md                  ← setup guide
├── ARCHITECTURE.md           ← this file
├── perf_optimizations.md     ← performance plan
└── tests/                    ← pytest unit tests
    ├── test_sampler.py
    ├── test_schema.py
    ├── test_intent.py
    └── test_frame_extractor.py
```
