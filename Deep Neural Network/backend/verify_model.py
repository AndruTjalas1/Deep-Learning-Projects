"""
Quick verification script to check if the model is properly trained.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from models import CharacterCNN
from config import NUM_CLASSES, MODELS_DIR, IDX_TO_CHAR, IMG_SIZE
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*70}")
print("MODEL VERIFICATION")
print(f"{'='*70}\n")

# Load the model
model = CharacterCNN(num_classes=NUM_CLASSES, dropout_rate=0.25)

best_model_path = MODELS_DIR / "character_cnn_best.pt"

print(f"Loading model from: {best_model_path}")
print(f"Model exists: {best_model_path.exists()}")

if best_model_path.exists():
    print(f"File size: {best_model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Model loaded successfully")
else:
    print("✗ Model file not found!")
    exit(1)

model.to(device)
model.eval()

# Check weights
print("\nModel Weight Statistics:")
for name, param in model.named_parameters():
    if 'weight' in name and 'conv' in name:
        print(f"  {name}:")
        print(f"    Mean: {param.mean():.6f}, Std: {param.std():.6f}")
        print(f"    Min: {param.min():.6f}, Max: {param.max():.6f}")
        break

# Test with random input
print("\nTest Inference with Random Input:")
random_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

with torch.no_grad():
    logits = model(random_input)
    probs = F.softmax(logits, dim=1)

top_probs, top_indices = torch.topk(probs[0], 10)

print("Top 10 predictions on random input:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    char = IDX_TO_CHAR[idx.item()]
    print(f"  {i+1}. {char:3s}: {prob.item():.4f}")

# Count predictions
predictions = []
for _ in range(100):
    random_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        logits = model(random_input)
        probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs[0]).item()
    predictions.append(pred_idx)

print(f"\nPrediction distribution on 100 random inputs:")
unique, counts = np.unique(predictions, return_counts=True)
for idx, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]:
    char = IDX_TO_CHAR[idx]
    print(f"  {char}: {count} times ({100*count/100:.1f}%)")

print("\n" + "="*70)
