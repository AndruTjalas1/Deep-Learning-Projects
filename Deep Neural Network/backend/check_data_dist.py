"""
Check training data distribution to identify class imbalance.
"""
from pathlib import Path
from config import DATA_DIR, CHAR_TO_IDX, IDX_TO_CHAR
import numpy as np

print(f"\n{'='*70}")
print("DATA DISTRIBUTION ANALYSIS")
print(f"{'='*70}\n")

data_path = Path(DATA_DIR)

# Find the actual data location (might be nested)
if not list(data_path.glob('*/')) or not any(p.is_dir() for p in data_path.glob('*/')):
    print(f"No class directories found in {data_path}")
    exit(1)

class_counts = {}

# Check nested structure
search_path = data_path
for subdir in data_path.iterdir():
    if subdir.is_dir() and subdir.name in ['train', 'test', 'combined_folder']:
        if subdir.name == 'combined_folder':
            for nested in subdir.iterdir():
                if nested.is_dir() and nested.name == 'train':
                    search_path = nested
                    break
        else:
            search_path = subdir
        break

print(f"Searching in: {search_path}\n")

# Count images per class
class_dirs = sorted([d for d in search_path.iterdir() if d.is_dir()])

for class_dir in class_dirs:
    class_name = class_dir.name
    image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))
    class_counts[class_name] = len(image_files)

if not class_counts:
    print("No images found!")
    exit(1)

# Sort by count
sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

print("Image count per class:")
print(f"{'Class':<10} {'Count':<10} {'%':<10}")
print("-" * 30)

total = sum(c for _, c in sorted_counts)
for class_name, count in sorted_counts:
    pct = 100 * count / total
    print(f"{class_name:<10} {count:<10} {pct:>6.1f}%")

print(f"\nTotal images: {total}")
print(f"Total classes: {len(class_counts)}")
print(f"Average per class: {total / len(class_counts):.0f}")
print(f"\nMax imbalance: {max(class_counts.values()) / min(class_counts.values()):.1f}x")

# Find the problematic classes
top_5 = sorted_counts[:5]
bottom_5 = sorted_counts[-5:]

print(f"\nTop 5 most frequent:")
for class_name, count in top_5:
    print(f"  {class_name}: {count}")

print(f"\nTop 5 least frequent:")
for class_name, count in bottom_5:
    print(f"  {class_name}: {count}")

print("\n" + "="*70)
