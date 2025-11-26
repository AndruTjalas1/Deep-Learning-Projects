"""
Training script for Handwriting Recognition CNN Model

This script trains the CharacterCNN model on the Kaggle handwritten characters dataset.
The data should be in the 'data/' directory with structure:
    data/
        ├── 0/
        ├── 1/
        ├── ...
        ├── a/
        └── Z/

Usage:
    python train.py
"""

# CRITICAL: Set CUDA environment BEFORE importing torch
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch

# CRITICAL: Import torch FIRST, set device early
print(f"\n{'='*70}")
print(f"PyTorch Initialization:")
print(f"  torch.__version__ = {torch.__version__}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  torch.version.cuda = {torch.version.cuda}")
print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from models import CharacterCNN
from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, MODEL_CONFIG, 
    NUM_CLASSES, IMG_SIZE, CHAR_TO_IDX, IDX_TO_CHAR
)
from hyperparameters import TRAINING, DATA_PREPROCESSING, LOGGING

# Device will be set in main()
device = None


class HandwritingDataset(Dataset):
    """
    Custom Dataset for loading handwritten character images.
    
    Directory structure:
        data/
            ├── 0/
            ├── 1/
            ├── ...
            ├── A/
            ├── ...
            ├── Z/
    """
    
    def __init__(self, data_dir, transform=None, split='train', split_ratio=0.8):
        """
        Args:
            data_dir: Path to data directory
            transform: Image transformations
            split: 'train', 'val', or 'test'
            split_ratio: Ratio for train/val split
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        
        self.images = []
        self.labels = []
        
        # Load images from directories
        self._load_data()
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        if split == 'train':
            print(f"  Train samples: {len(self.images)}")
        elif split == 'val':
            print(f"  Val samples: {len(self.images)}")
    
    def _load_data(self):
        """Load images and labels from directory structure.
        
        Handles both flat structure (data/0/, data/1/, etc.)
        and nested structure (data/train/0/, data/test/0/, etc.)
        """
        
        # Check if we have nested train/test folders
        nested_path = None
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and subdir.name in ['train', 'test', 'combined_folder']:
                nested_path = subdir
                break
        
        if nested_path and nested_path.name == 'combined_folder':
            # Go one level deeper
            for subdir in nested_path.iterdir():
                if subdir.is_dir() and subdir.name == 'train':
                    nested_path = subdir
                    break
        
        search_dir = nested_path if nested_path else self.data_dir
        
        # Get all character directories
        class_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir()])
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {search_dir}")
        
        # Load images for each class
        for class_idx, class_dir in enumerate(class_dirs):
            char_name = class_dir.name
            
            # Map directory names to character indices
            # Handle both single char (0, 1, ..., a, b, ..., A_caps, etc.) and variants
            if char_name in CHAR_TO_IDX:
                label = CHAR_TO_IDX[char_name]
            elif char_name.endswith('_caps'):
                # Remove _caps and look up
                base_char = char_name.replace('_caps', '').upper()
                if base_char in CHAR_TO_IDX:
                    label = CHAR_TO_IDX[base_char]
                else:
                    try:
                        label = int(char_name)
                    except:
                        continue
            else:
                # Try numeric or uppercase version
                try:
                    label = int(char_name)
                except:
                    upper_char = char_name.upper()
                    if upper_char in CHAR_TO_IDX:
                        label = CHAR_TO_IDX[upper_char]
                    else:
                        continue
            
            # Get all image files
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))
            
            if not image_files:
                continue
            
            # Split into train/val/test
            np.random.seed(42)
            indices = np.arange(len(image_files))
            np.random.shuffle(indices)
            
            train_split = int(len(image_files) * self.split_ratio)
            val_split = train_split + int(len(image_files) * (1 - self.split_ratio) / 2)
            
            if self.split == 'train':
                selected_indices = indices[:train_split]
            elif self.split == 'val':
                selected_indices = indices[train_split:val_split]
            else:  # test
                selected_indices = indices[val_split:]
            
            for idx in selected_indices:
                self.images.append(image_files[idx])
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get single image and label."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            blank_img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, label


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        # Non-blocking GPU transfer for faster pipeline
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            # Non-blocking GPU transfer for faster pipeline
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def visualize_predictions(model, val_loader, device, epoch, num_samples=36):
    """Visualize one sample from each character class with predictions as image grid."""
    model.eval()
    
    images_dict = {}  # {label: (image, confidence, prediction)}
    
    # Collect one sample from each class
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            for i in range(len(images)):
                label = labels[i].item()
                
                # Skip if we already have this class
                if label in images_dict:
                    continue
                
                images_dict[label] = (
                    images[i].cpu(),
                    confidences[i].item() * 100,
                    predicted[i].item()
                )
                
                if len(images_dict) >= NUM_CLASSES:
                    break
            
            if len(images_dict) >= NUM_CLASSES:
                break
    
    # Sort by label to get consistent ordering (0-9, a-z, A-Z)
    sorted_labels = sorted(images_dict.keys())
    
    # Create grid (6x6 for 36 classes)
    grid_size = int(np.ceil(np.sqrt(NUM_CLASSES)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle(f'Epoch {epoch} - All Classes (One Sample Each)', fontsize=18, fontweight='bold')
    
    correct = 0
    for idx, label in enumerate(sorted_labels):
        img, conf, pred_label = images_dict[label]
        true_char = IDX_TO_CHAR[label]
        pred_char = IDX_TO_CHAR[pred_label]
        match = label == pred_label
        
        if match:
            correct += 1
            color = 'green'
            edge_color = 'green'
        else:
            color = 'red'
            edge_color = 'red'
        
        # Get subplot
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]
        
        # Display image
        img_display = img.squeeze().numpy()
        ax.imshow(img_display, cmap='gray')
        ax.axis('off')
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        
        # Title with true label, predicted label, and confidence
        title = f'{true_char}\n→ {pred_char} ({conf:.0f}%)'
        ax.set_title(title, color=color, fontweight='bold', fontsize=11)
    
    # Hide empty subplots
    for idx in range(len(sorted_labels), grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        axes[row, col].axis('off')
    
    # Save image
    viz_path = LOGS_DIR / f"epoch_{epoch}_all_classes.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    accuracy = (correct / len(sorted_labels)) * 100
    print(f"\n✓ Saved visualization to {viz_path}")
    print(f"  Accuracy (all classes): {accuracy:.1f}% ({correct}/{len(sorted_labels)})")
    print(f"  Classes shown: {', '.join([IDX_TO_CHAR[l] for l in sorted_labels])}\n")


def main():
    """Main training function."""
    global device
    
    print("=" * 60)
    print("HANDWRITING RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    
    # Device detection (must happen inside main for proper CUDA initialization)
    print(f"\n{'='*60}")
    print(f"CUDA DETECTION INFO:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ USING GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print(f"⚠ USING CPU (GPU not available)")

    print(f"Device object: {device}\n")
    
    # Hyperparameters from hyperparameters.py
    batch_size = TRAINING["batch_size"]
    epochs = TRAINING["num_epochs"]
    learning_rate = TRAINING["learning_rate"]
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Input Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Number of Classes: {NUM_CLASSES}")
    print(f"  Device: {device}")
    print(f"  Data Directory: {DATA_DIR}")
    
    # Data transforms with augmentation from hyperparameters
    aug_config = TRAINING["augmentation"]
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(aug_config["rotation"]),
        transforms.RandomAffine(
            degrees=0, 
            translate=(aug_config["translation"], aug_config["translation"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TRAINING["normalize_mean"], 
            std=TRAINING["normalize_std"]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TRAINING["normalize_mean"], 
            std=TRAINING["normalize_std"]
        )
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_dataset = HandwritingDataset(
            DATA_DIR, 
            transform=train_transform, 
            split='train',
            split_ratio=TRAINING["train_split"]
        )
        val_dataset = HandwritingDataset(
            DATA_DIR, 
            transform=val_transform, 
            split='val',
            split_ratio=TRAINING["train_split"]
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure data is in {DATA_DIR} with structure:")
        print("  data/")
        print("      ├── 0/")
        print("      ├── 1/")
        print("      ├── ...etc")
        return
    
    # Data loaders with worker optimization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Parallel CPU threads for data loading
        pin_memory=True,
        prefetch_factor=2,  # Prefetch batches for smoother GPU pipeline
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Parallel CPU threads for data loading
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Model
    print("\nInitializing model...")
    model = CharacterCNN(num_classes=NUM_CLASSES, dropout_rate=0.25)
    model = model.to(device)
    
    # GPU optimizations for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN for faster convolutions
        torch.backends.cudnn.deterministic = False  # Faster but less reproducible
    
    # Calculate class weights to handle imbalance
    print("\nCalculating class weights for imbalanced data...")
    class_counts = {}
    for labels in [train_dataset.labels]:
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # Compute weights inversely proportional to class frequency
    total_samples = len(train_dataset)
    class_weights = torch.ones(NUM_CLASSES)
    for class_id in range(NUM_CLASSES):
        if class_id in class_counts:
            class_weights[class_id] = total_samples / (NUM_CLASSES * class_counts[class_id])
        else:
            class_weights[class_id] = 1.0  # Default weight for unseen classes
    
    class_weights = class_weights.to(device)
    
    print(f"  Sample class weights:")
    for i in range(min(10, NUM_CLASSES)):
        print(f"    {IDX_TO_CHAR[i]}: {class_weights[i]:.4f}")
    print(f"  ...")
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # SAVE MODEL FOR EVERY EPOCH (guaranteed no loss)
        epoch_model_path = MODELS_DIR / f"epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"  ✓ Model saved: {epoch_model_path}")
        
        # Clear GPU cache to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # GENERATE SAMPLES EVERY 2 EPOCHS (all 36 characters)
        if epoch % 2 == 0:
            print(f"  Generating sample predictions for all characters...")
            visualize_predictions(model, val_loader, device, epoch, num_samples=16)
            print(f"  ✓ Samples saved for epoch {epoch}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")
        
        # Learning rate schedule
        scheduler.step(val_loss)
    
    print(f"\n✓ Training complete! All epoch models saved to {MODELS_DIR}/")
    print(f"✓ Best validation accuracy achieved: {best_val_acc:.2f}%")
    
    # Save history
    if LOGGING["save_history"]:
        history_path = LOGS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved to {history_path}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Plot results
    if LOGGING["save_plots"]:
        print("\nGenerating training plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train', marker='o')
        axes[0].plot(history['val_loss'], label='Val', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train', marker='o')
        axes[1].plot(history['val_acc'], label='Val', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = LOGS_DIR / "training_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {plot_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"\nEpoch models saved to: {MODELS_DIR}/epoch_*.pt")
    print(f"Start the server with: python main.py")


if __name__ == "__main__":
    main()
