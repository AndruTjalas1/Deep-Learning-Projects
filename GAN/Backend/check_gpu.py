#!/usr/bin/env python3
"""Quick script to check GPU/CUDA availability."""

import torch

print("=" * 50)
print("PyTorch Configuration Check")
print("=" * 50)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPU: Not available (using CPU)")

# Check for Apple MPS
if hasattr(torch.backends, 'mps'):
    print(f"Apple MPS Available: {torch.backends.mps.is_available()}")

print("\n" + "=" * 50)
