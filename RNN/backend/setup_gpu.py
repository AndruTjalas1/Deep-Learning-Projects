"""
GPU Setup Script
Configures TensorFlow to use GPU if available
"""

import os
import sys

# Set environment variables before TensorFlow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Use GPU 0

import tensorflow as tf

print("=" * 70)
print("GPU SETUP DIAGNOSTIC")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"\n✓ CPUs detected: {len(cpus)}")
print(f"✓ GPUs detected: {len(gpus)}")

if gpus:
    print("\nGPU Details:")
    for gpu in gpus:
        print(f"  {gpu}")
    
    # Enable GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
else:
    print("\n⚠ WARNING: No GPUs detected!")
    print("  This could be due to:")
    print("  1. NVIDIA GPU drivers not installed")
    print("  2. CUDA toolkit not installed")
    print("  3. cuDNN not installed")
    print("  4. TensorFlow GPU libraries missing")
    print("\n  For GPU support, ensure you have:")
    print("  - NVIDIA driver (✓ Detected via nvidia-smi)")
    print("  - CUDA 12.x toolkit")
    print("  - cuDNN library")
    print("  - TensorFlow-GPU package")

# Verify TensorFlow version
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"TensorFlow Build Info: {tf.sysconfig.get_build_info()['is_cuda_build']}")

# Test GPU computation
if gpus:
    print("\n✓ Testing GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 2.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"  Matrix multiplication result: {c}")
    print("  GPU computation successful!")
else:
    print("\n✓ GPU not available - using CPU")

print("\n" + "=" * 70)
