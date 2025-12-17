"""
Helper script to inspect all epoch models and copy the best one.
"""
from pathlib import Path
import torch
from models import CharacterCNN
from config import MODELS_DIR, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*70}")
print("EPOCH MODEL INSPECTOR")
print(f"{'='*70}\n")

# Find all epoch models
epoch_models = sorted(MODELS_DIR.glob("epoch_*.pt"))

if not epoch_models:
    print("No epoch models found!")
    exit(1)

print(f"Found {len(epoch_models)} epoch models:\n")
print(f"{'Epoch':<10} {'File Size (MB)':<20} {'Weight Stats':<30}")
print("-" * 60)

for model_path in epoch_models:
    epoch_num = int(model_path.stem.split("_")[1])
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    
    # Load and check weights
    model = CharacterCNN(num_classes=NUM_CLASSES, dropout_rate=0.25)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Check first layer weights
    first_param = list(model.parameters())[0]
    mean_val = first_param.mean().item()
    std_val = first_param.std().item()
    
    print(f"Epoch {epoch_num:<3} {file_size_mb:<19.2f} Mean: {mean_val:.6f}, Std: {std_val:.6f}")

print(f"\n{'='*70}")
print("\nTo use a specific epoch model:")
print("1. Check the sample visualizations in logs/ folder")
print("2. Identify the epoch with best visual results")
print("3. Copy it: cp saved_models/epoch_XYZ.pt saved_models/character_cnn.pt")
print("4. Restart the server to load it")
print(f"\n{'='*70}\n")
