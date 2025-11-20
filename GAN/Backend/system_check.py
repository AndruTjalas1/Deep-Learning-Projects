#!/usr/bin/env python3
"""Comprehensive system readiness check for DCGAN training."""

import os
import sys
from pathlib import Path

print("\n" + "=" * 60)
print("DCGAN SYSTEM READINESS CHECK")
print("=" * 60 + "\n")

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check(condition, text):
    """Print check result."""
    if condition:
        print(f"{GREEN}✓{RESET} {text}")
        return True
    else:
        print(f"{RED}✗{RESET} {text}")
        return False

all_passed = True

# 1. Python & PyTorch
print("📦 DEPENDENCIES")
print("-" * 60)

try:
    import torch
    check(True, f"PyTorch: {torch.__version__}")
    check(torch.cuda.is_available(), "CUDA Available")
    if torch.cuda.is_available():
        check(True, f"  GPU: {torch.cuda.get_device_name(0)}")
        check(True, f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except:
    all_passed = False
    check(False, "PyTorch not found")

try:
    import fastapi
    check(True, "FastAPI installed")
except:
    all_passed = False
    check(False, "FastAPI not installed")

try:
    import pydantic
    check(True, "Pydantic installed")
except:
    all_passed = False
    check(False, "Pydantic not installed")

# 2. Directory Structure
print("\n📁 DIRECTORY STRUCTURE")
print("-" * 60)

base_dir = Path(__file__).parent
dirs_to_check = {
    'data/cats': 'Dataset: Cats',
    'data/dogs': 'Dataset: Dogs',
    'samples': 'Output: Samples',
    'saved_models': 'Output: Models',
    'logs': 'Output: Logs',
}

for dir_path, label in dirs_to_check.items():
    full_path = base_dir / dir_path
    exists = full_path.exists()
    check(exists, f"{label}: {dir_path}")
    if not exists:
        all_passed = False

# Check for images
cats_dir = base_dir / 'data' / 'cats'
dogs_dir = base_dir / 'data' / 'dogs'

if cats_dir.exists():
    cat_images = list(cats_dir.glob('*.[jJpP][nNgG]*')) + list(cats_dir.glob('*.[pP][nN][gG]')) + list(cats_dir.glob('*.[bB][mM][pP]'))
    check(len(cat_images) > 0, f"Cat images found: {len(cat_images)}")
    if len(cat_images) == 0:
        all_passed = False

if dogs_dir.exists():
    dog_images = list(dogs_dir.glob('*.[jJpP][nNgG]*')) + list(dogs_dir.glob('*.[pP][nN][gG]')) + list(dogs_dir.glob('*.[bB][mM][pP]'))
    check(len(dog_images) > 0, f"Dog images found: {len(dog_images)}")
    if len(dog_images) == 0:
        print(f"{YELLOW}⚠{RESET}  Need at least 50+ images per animal type for good results")

# 3. Configuration
print("\n⚙️  CONFIGURATION")
print("-" * 60)

config_path = base_dir / 'config.yaml'
check(config_path.exists(), "config.yaml exists")

if config_path.exists():
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        check(True, f"Configuration valid")
        check(True, f"  Epochs: {config['training']['epochs']}")
        check(True, f"  Batch size: {config['training']['batch_size']}")
        check(True, f"  Resolution: {config['image']['resolution']}x{config['image']['resolution']}")
    except Exception as e:
        check(False, f"Configuration error: {e}")

# 4. Python Modules
print("\n🔧 PYTHON MODULES")
print("-" * 60)

modules_to_check = {
    'config': 'config.py',
    'device': 'device.py',
    'models': 'models.py',
    'data_loader': 'data_loader.py',
    'trainer': 'trainer.py',
}

for module_name, file_name in modules_to_check.items():
    file_path = base_dir / file_name
    check(file_path.exists(), f"{module_name}: {file_name}")
    if not file_path.exists():
        all_passed = False

# 5. API Server
print("\n🌐 API SERVER")
print("-" * 60)

main_py = base_dir / 'main.py'
check(main_py.exists(), "main.py exists")

# 6. Recommendations
print("\n💡 RECOMMENDATIONS")
print("-" * 60)

vram_needed = 4  # GB
if config and config['image']['resolution'] == 128:
    vram_needed = 8
elif config and config['image']['resolution'] == 256:
    vram_needed = 12

gpu_vram = 8
if gpu_vram >= vram_needed:
    check(True, f"GPU memory sufficient for {config['image']['resolution']}x{config['image']['resolution']} training")
else:
    print(f"{YELLOW}⚠{RESET}  GPU memory may be tight. Consider reducing batch_size or resolution")

batch_size = config['training']['batch_size'] if config else 64
if batch_size <= 64:
    check(True, f"Batch size reasonable: {batch_size}")
else:
    print(f"{YELLOW}⚠{RESET}  Large batch size ({batch_size}) may cause GPU OOM. Try 32-64.")

# 7. Summary
print("\n" + "=" * 60)
if all_passed:
    print(f"{GREEN}✓ SYSTEM READY FOR TRAINING{RESET}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add images to Backend/data/cats/ and Backend/data/dogs/")
    print("2. Run: python main.py")
    print("3. In another terminal: cd ../Frontend && npm run dev")
    print("4. Open: http://localhost:3000")
    print("\nHappy training! 🚀\n")
else:
    print(f"{RED}✗ SYSTEM NOT FULLY READY{RESET}")
    print("=" * 60)
    print("\nFix the issues above before training.\n")
    sys.exit(1)

print("=" * 60 + "\n")
