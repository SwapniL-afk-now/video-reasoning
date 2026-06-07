# Q-VKG Setup Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 24 GB | 80+ GB (A100/H100) |
| CUDA | 12.1 | 12.8+ |
| Python | 3.10 | 3.12 |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB | 200 GB (model weights + videos) |

## 1. Prerequisites

```bash
# Verify GPU and CUDA
nvidia-smi
nvcc --version
python --version   # >= 3.10
```

## 2. Clone & Install

```bash
# Clone repository
git clone <repo-url> && cd video-reasoning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Q-VKG package
pip install -e qvkg/

# Install extra dependencies
pip install faster-whisper faiss-gpu-cu12
```

## 3. Environment Variables

Models are downloaded from Hugging Face Hub. Set your token:

```bash
# Required for gated models (Qwen)
export HF_TOKEN="hf_..."
```

Create `.env` in the project root for persistent storage (already in `.gitignore`):

```
HF_TOKEN=hf_...
```

## 4. GPU-Specific: Flash Attention

### Blackwell (RTX PRO 6000, B200, etc.)
Flash Attention must be built from source for sm_120:

```bash
cd /tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && git checkout v2.8.3
FLASH_ATTN_CUDA_ARCHS=120 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  MAX_JOBS=4 \
  pip install --no-build-isolation .
```

### Other GPUs (A100, H100, etc.)
Pre-built wheels work:

```bash
pip install flash-attn==2.8.3
```

### Verify

```bash
python qvkg/scripts/verify_flash_attn.py
```

## 5. Build VKG (Video Knowledge Graph)

Process a video offline (one-time per video):

```bash
python qvkg/scripts/build_vkg.py \
  --video /path/to/video.mp4 \
  --out ./output \
  --model Qwen/Qwen3.5-4B \
  --tp 1
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--budget` | 500 | Keyframe budget |
| `--model` | Qwen/Qwen3.5-4B | VLM model |
| `--tp` | 1 | Tensor parallel GPUs |
| `--no-whisper` | false | Skip audio transcription |
| `--hard-boundary` | 0.75 | Scene cut threshold |
| `--soft-boundary` | 0.5 | Scene transition threshold |
| `--questions-csv` | null | Enable question-aware dense sampling |
| `--video-type` | null | Hint: `sport`, `live`, `cartoon` |

Output written to `./output/<video_id>/`:
- `vkg.json` — graph (nodes + edges)
- `vkg.index` — FAISS HNSW index
- `frames.h5` — HDF5 keyframe store
- `meta.json` — build metadata

## 6. Run Inference (QA)

Query a pre-built VKG:

```bash
# Single question
python qvkg/scripts/query_vkg.py \
  --vkg ./output/<video_id> \
  --question "Why did the character open the door?"

# Interactive mode
python qvkg/scripts/query_vkg.py \
  --vkg ./output/<video_id> \
  --interactive
```

## 7. Evaluate on LVBench

```bash
python qvkg/scripts/eval_lvbench.py \
  --csv /path/to/LVBench_full.csv \
  --vkg-dir ./output \
  --video-dir /path/to/videos \
  --out results.jsonl \
  --model Qwen/Qwen3.5-4B
```

Results stream to `results.jsonl` incrementally (resume-safe). Per-category accuracy printed at the end.

## 8. Project Layout

```
qvkg/
├── SETUP.md                  ← this file
├── setup.py                  ← package install
├── requirements.txt          ← pip dependencies
├── configs/default.yaml      ← default configuration
├── qvkg/                     ← library code
│   ├── builder.py            ← 11-step VKG construction orchestrator
│   ├── sampler.py            ← hierarchical frame sampling
│   ├── extraction.py         ← VLM scene extraction
│   ├── causal.py             ← LLM causal chain inference
│   ├── character.py          ← description-based character resolution
│   ├── episode.py            ← LLM episode segmentation
│   ├── schema.py             ← VKGNode, VKGEdge, VKGraph dataclasses
│   ├── vllm_client.py        ← LLM factory, sampling params, JSON schemas
│   ├── faiss_index.py        ← FAISS HNSW index + semantic edges
│   ├── frame_store.py        ← HDF5 keyframe storage
│   └── query/                ← online QA pipeline
│       ├── qa.py             ← single-call answer_question()
│       ├── intent.py         ← question intent classification
│       ├── activator.py      ← subgraph activation (FAISS + typed BFS)
│       ├── serializer.py     ← graph → structured NL context
│       └── frame_extractor.py← on-demand frame extraction
├── scripts/
│   ├── build_vkg.py          ← offline VKG construction CLI
│   ├── query_vkg.py          ← online QA CLI
│   ├── eval_lvbench.py       ← LVBench evaluation runner
│   └── verify_flash_attn.py  ← flash-attn smoke test
└── tests/
    ├── test_sampler.py
    ├── test_schema.py
    ├── test_intent.py
    └── test_frame_extractor.py
```
