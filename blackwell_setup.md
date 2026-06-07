# Blackwell RTX PRO 6000 Setup Guide for verl / TAFR-GRPO

This document describes how to set up the verl training framework with Flash Attention on NVIDIA Blackwell (sm_120) GPUs, specifically the RTX PRO 6000 Blackwell Workstation Edition.

## Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (98 GB VRAM) |
| Compute Capability | 12.0 (sm_120) |
| Driver | 595.71.05 |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12.13 |
| flash-attn | 2.8.3 (built from source) |
| vLLM | 0.21.0 |
| verl | 0.8.0.dev |
| NCCL | 2.28.9+cuda13.0 |

## 1. Flash Attention 2 (sm_120)

Pre-built `flash-attn` wheels on PyPI do **not** include sm_120 kernels. You must build from source.

### Why FA2, not FA4?

- `flash-attn-4` (4.0.0b16) uses TCGEN05/TMEM instructions that are absent on sm_120.
- The `feat/sm120-support` branch from `blake-snc` fork works for FA4 CuTe DSL but is beta.
- FA2 2.8.3 compiles cleanly for sm_120 from source and is production-stable.

### Build from source

```bash
# Clone FA2
cd /tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3

# Build for sm_120
FLASH_ATTN_CUDA_ARCHS=120 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=4 \
pip install --no-build-isolation .
```

- `FLASH_ATTN_CUDA_ARCHS=120` -- targets only sm_120 (avoids compiling for all architectures).
- `MAX_JOBS=4` -- limits parallel compilation to avoid OOM during build.

### Verification

```python
import flash_attn
print(flash_attn.__version__)  # 2.8.3

# Smoke tests (all should pass)
from flash_attn import flash_attn_func
import torch
q = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device='cuda')
out = flash_attn_func(q, k, v, causal=True)
print(f"FA2 output shape: {out.shape}")
```

