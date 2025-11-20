# GPU Setup Guide

Complete instructions for enabling GPU acceleration on your system.

## Current Status

✓ PyTorch installed: 2.1.1 (CPU version)
✗ CUDA not available
⚠️ GPU not being utilized

## Quick GPU Setup (Windows with NVIDIA GPU)

### Step 1: Verify NVIDIA GPU

```bash
# Check if you have NVIDIA GPU
nvidia-smi
```

You should see output with GPU model and CUDA Compute Capability.

### Step 2: Install CUDA-Enabled PyTorch

**Option A: Using pip (Recommended)**

```bash
# Remove CPU version
pip uninstall torch torchvision torchaudio -y

# Install CUDA 11.8 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Option B: Using Conda (Alternative)**

```bash
# Deactivate current venv
deactivate

# Create new conda environment with CUDA support
conda create -n dcgan-gpu python=3.10 pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Activate
conda activate dcgan-gpu

# Install remaining requirements
pip install -r requirements.txt
```

### Step 3: Verify GPU Setup

```bash
python device.py
```

You should see output like:
```
Using NVIDIA GPU: NVIDIA GeForce RTX 4090
```

### Step 4: Update config.yaml

Ensure GPU usage is enabled:

```yaml
device:
  use_gpu: true
```

## CUDA Version Compatibility

| PyTorch Version | CUDA Versions | Command |
|-----------------|---------------|---------|
| 2.1.1 | 11.8, 12.1 | `--index-url https://download.pytorch.org/whl/cu118` |
| 2.0.1 | 11.7, 11.8 | `--index-url https://download.pytorch.org/whl/cu117` |
| Latest | See PyTorch site | Check pytorch.org |

## Troubleshooting GPU Setup

### Issue: "CUDA is not available"

**Check 1: NVIDIA Drivers**
```bash
nvidia-smi
```
If this fails, install drivers from: https://www.nvidia.com/Download/driverDetails.aspx

**Check 2: Verify PyTorch Installation**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Check 3: Check CUDA Toolkit**
```bash
nvcc --version
```

### Issue: "Out of GPU Memory"

Reduce batch size or resolution in `config.yaml`:

```yaml
training:
  batch_size: 32  # Reduce from 64
  
image:
  resolution: 64  # Reduce from 128
```

### Issue: PyTorch still using CPU after installation

```bash
# Uninstall all versions
pip uninstall torch torchvision torchaudio -y

# Clear cache
pip cache purge

# Reinstall CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Mac (Apple Silicon) Setup

If you're working on Mac with Apple Silicon:

```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio -y

# Install Metal Performance Shaders version
pip install torch torchvision torchaudio

# Verify
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Performance Comparison

| Device | 64×64, Batch 64 | Speed | Memory |
|--------|-----------------|-------|--------|
| CPU | ~2.5s/batch | 1x | 2-3 GB |
| NVIDIA GPU | ~0.3s/batch | 8x faster | 4-6 GB |
| Mac M1/M2 | ~1.2s/batch | 2x faster | 2-3 GB |

## VRAM Requirements by Configuration

| Resolution | Batch Size | VRAM Needed |
|------------|-----------|-------------|
| 64×64 | 64 | 4 GB |
| 64×64 | 128 | 8 GB |
| 128×128 | 32 | 6 GB |
| 128×128 | 64 | 12 GB |
| 256×256 | 16 | 8 GB |
| 256×256 | 32 | 14 GB |

## Batch Size Recommendations by GPU

| GPU | Max Batch | Recommended |
|-----|-----------|-------------|
| RTX 3060 (12GB) | 64 @ 64×64 | 32-48 |
| RTX 4080 (16GB) | 128 @ 128×128 | 64-96 |
| RTX 4090 (24GB) | 256 @ 256×256 | 128-192 |
| A100 (40GB) | 512 @ 256×256 | 256+ |

## Monitoring GPU Usage

### Real-time GPU Monitor

```bash
# Watch GPU usage while training
nvidia-smi -l 1  # Updates every 1 second
```

### In Python

```python
import torch

# Check GPU usage
gpu_used = torch.cuda.memory_allocated() / 1024**3
gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"GPU Memory: {gpu_used:.1f}GB / {gpu_total:.1f}GB")
```

## Next Steps

1. ✅ Install CUDA-enabled PyTorch
2. ✅ Verify with `nvidia-smi` and `torch.cuda.is_available()`
3. ✅ Test with small training run (5 epochs at 64×64)
4. ✅ Monitor GPU usage with `nvidia-smi -l 1`
5. ✅ Increase batch size/resolution as needed

Your system should now be **8x faster** with GPU! 🚀
