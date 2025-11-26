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
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import struct
import numpy as np

from models import CharacterCNN
from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, MODEL_CONFIG, 
    NUM_CLASSES, IMG_SIZE, CHAR_TO_IDX, IDX_TO_CHAR, EMNIST_CLASSES
)
from hyperparameters import TRAINING, DATA_PREPROCESSING, LOGGING

device = None
EMNIST_DATA_DIR = Path(__file__).parent / "emnist_data"


def read_idx_file(filename):
    """Read IDX format file (MNIST/EMNIST standard format)."""
    with open(filename, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        if magic == 2049:  # Labels
            return np.frombuffer(f.read(n), dtype=np.uint8)
        elif magic == 2051:  # Images
            rows, cols = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, rows, cols)
    return None


class HandwritingDataset(Dataset):
    """Load EMNIST balanced dataset from extracted binary files."""
    
    def __init__(self, transform=None, split='train'):
        self.transform = transform
        
        if split == 'train':
            img_file = EMNIST_DATA_DIR / "emnist-balanced-train-images-idx3-ubyte"
            lbl_file = EMNIST_DATA_DIR / "emnist-balanced-train-labels-idx1-ubyte"
        else:
            img_file = EMNIST_DATA_DIR / "emnist-balanced-test-images-idx3-ubyte"
            lbl_file = EMNIST_DATA_DIR / "emnist-balanced-test-labels-idx1-ubyte"
        
        if not img_file.exists() or not lbl_file.exists():
            raise FileNotFoundError(f"EMNIST files not found in {EMNIST_DATA_DIR}.")
        
        self.images = read_idx_file(str(img_file))
        self.labels = read_idx_file(str(lbl_file))
        
        if self.images is None or self.labels is None:
            raise RuntimeError("Failed to read EMNIST files")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Read image and convert to PIL
        img_array = self.images[idx]
        # EMNIST images: transpose only for correct orientation
        img_array_transposed = np.transpose(img_array)
        img = Image.fromarray(img_array_transposed.astype(np.uint8))
        label = int(self.labels[idx])
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, label


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
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


def visualize_predictions(model, val_loader, device, epoch):
    """Visualize predictions for one sample from each class."""
    model.eval()
    images_dict = {}
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            for i in range(len(images)):
                label = labels[i].item()
                if label not in images_dict:
                    images_dict[label] = {
                        'image': images[i].cpu(),
                        'confidence': confidences[i].item() * 100,
                        'predicted': predicted[i].item()
                    }
                
                if len(images_dict) >= NUM_CLASSES:
                    break
            
            if len(images_dict) >= NUM_CLASSES:
                break
    
    # Display in order: 0, 1, 2, ..., 46 (following EMNIST_CLASSES order)
    grid_size = int(np.ceil(np.sqrt(NUM_CLASSES)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle(f'Epoch {epoch} - Model Predictions (Organized by Index)', fontsize=16, fontweight='bold')
    
    correct = 0
    displayed = 0
    
    for idx in range(NUM_CLASSES):
        if idx not in images_dict:
            continue
        
        data = images_dict[idx]
        img = data['image']
        conf = data['confidence']
        pred_label = data['predicted']
        
        is_correct = (idx == pred_label)
        if is_correct:
            correct += 1
        
        row = displayed // grid_size
        col = displayed % grid_size
        ax = axes[row, col]
        
        img_display = img.squeeze().numpy()
        ax.imshow(img_display, cmap='gray')
        ax.axis('off')
        
        true_char = IDX_TO_CHAR.get(idx, '?')
        pred_char = IDX_TO_CHAR.get(pred_label, '?')
        color = 'green' if is_correct else 'red'
        
        title = f'{idx}: {true_char} -> {pred_char}\n{conf:.0f}%'
        ax.set_title(title, color=color, fontweight='bold', fontsize=9)
        
        displayed += 1
    
    for idx in range(displayed, grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        axes[row, col].axis('off')
    
    viz_path = LOGS_DIR / f"epoch_{epoch:03d}_predictions.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Saved visualization: {viz_path}")
    print(f"  Accuracy: {correct}/{displayed} ({100*correct/displayed:.1f}%)")


def main():
    global device
    
    print("=" * 60)
    print("HANDWRITING RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    
    print(f"\n{'='*60}")
    print(f"CUDA DETECTION INFO:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[OK] USING GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print(f"[WARNING] USING CPU (GPU not available)")

    print(f"Device object: {device}\n")
    
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
    
    print("\nLoading datasets...")
    try:
        train_dataset = HandwritingDataset(
            transform=train_transform, 
            split='train'
        )
        val_dataset = HandwritingDataset(
            transform=val_transform, 
            split='test'
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    print("\nInitializing model...")
    model = CharacterCNN(num_classes=NUM_CLASSES, dropout_rate=0.25)
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print("\nUsing uniform class weights (EMNIST is already balanced)...")
    class_weights = torch.ones(NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    print("\nStarting training...\n")
    
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
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        epoch_model_path = MODELS_DIR / f"epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"  [OK] Model saved: {epoch_model_path}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if epoch % 2 == 0:
            print(f"  Generating sample predictions for all characters...")
            visualize_predictions(model, val_loader, device, epoch)
            print(f"  [OK] Samples saved for epoch {epoch}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  [OK] New best validation accuracy: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
    
    print(f"\n[OK] Training complete! All epoch models saved to {MODELS_DIR}/")
    print(f"[OK] Best validation accuracy achieved: {best_val_acc:.2f}%")
    
    if LOGGING["save_history"]:
        history_path = LOGS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[OK] Training history saved to {history_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if LOGGING["save_plots"]:
        print("\nGenerating training plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history['train_loss'], label='Train', marker='o')
        axes[0].plot(history['val_loss'], label='Val', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
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
        print(f"[OK] Plot saved to {plot_path}")
    
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
