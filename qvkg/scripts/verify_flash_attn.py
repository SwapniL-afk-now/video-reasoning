#!/usr/bin/env python3
"""Smoke test for flash-attn 2.x on Blackwell sm_120."""
import sys


def check_flash_attn():
    try:
        import flash_attn
        print(f"flash_attn version: {flash_attn.__version__}")
    except ImportError:
        print("ERROR: flash_attn not installed. Build from source:")
        print("  cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git")
        print("  cd flash-attention && git checkout v2.8.3")
        print("  FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \\")
        print("  pip install --no-build-isolation .")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    from flash_attn import flash_attn_func

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    q = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_func(q, k, v, causal=True)
    print(f"FA2 output shape: {out.shape}  ✓")
    print("Flash Attention smoke test PASSED")


if __name__ == "__main__":
    check_flash_attn()
