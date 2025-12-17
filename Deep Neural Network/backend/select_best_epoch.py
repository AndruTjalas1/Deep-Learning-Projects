"""
Quick script to select and copy the best epoch model.
Usage: python select_best_epoch.py <epoch_number>
Example: python select_best_epoch.py 35
"""
import sys
import shutil
from pathlib import Path
from config import MODELS_DIR

if len(sys.argv) < 2:
    print("Usage: python select_best_epoch.py <epoch_number>")
    print("Example: python select_best_epoch.py 35")
    sys.exit(1)

epoch_num = int(sys.argv[1])
epoch_model = MODELS_DIR / f"epoch_{epoch_num:03d}.pt"
target_model = MODELS_DIR / "character_cnn.pt"

if not epoch_model.exists():
    print(f"Error: {epoch_model} does not exist!")
    sys.exit(1)

print(f"\nCopying {epoch_model.name} to {target_model.name}...")
shutil.copy(epoch_model, target_model)
print(f"✓ Done! Restart the server to use epoch {epoch_num} model.")
print(f"  Command: python main.py")
print()
