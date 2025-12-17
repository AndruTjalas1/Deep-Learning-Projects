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
import struct

from models import CharacterCNN
from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, NUM_CLASSES, IMG_SIZE,
    EMNIST_CLASSES, MODEL_TYPE_FOR_CLASS,
    DIGIT_CLASSES, UPPERCASE_CLASSES, LOWERCASE_CLASSES,
    DIGIT_CHAR_TO_IDX, UPPERCASE_CHAR_TO_IDX, LOWERCASE_CHAR_TO_IDX,
    DIGIT_IDX_TO_CHAR, UPPERCASE_IDX_TO_CHAR, LOWERCASE_IDX_TO_CHAR
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


class SpecialistDataset(Dataset):
    """Load EMNIST balanced dataset filtered for a specific character type."""
    
    def __init__(self, transform=None, split='train', model_type='digit'):
        """
        Args:
            model_type: 'digit', 'uppercase', or 'lowercase'
            split: 'train' or 'val'
        """
        self.transform = transform
        self.model_type = model_type
        
        # Create mapping from EMNIST label to specialist model label
        if model_type == 'digit':
            self.target_emnist_indices = list(range(10))  # EMNIST indices 0-9 are digits
            self.emnist_to_specialist = {i: i for i in range(10)}  # 0->0, 1->1, ..., 9->9
        elif model_type == 'uppercase':
            self.target_emnist_indices = list(range(10, 36))  # EMNIST indices 10-35 are A-Z
            self.emnist_to_specialist = {i: i-10 for i in range(10, 36)}  # 10->0, 11->1, ..., 35->25
        elif model_type == 'lowercase':
            self.target_emnist_indices = list(range(36, 47))  # EMNIST indices 36-46 are lowercase
            self.emnist_to_specialist = {i: i-36 for i in range(36, 47)}  # 36->0, 37->1, ..., 46->10
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        if split == 'train':
            img_file = EMNIST_DATA_DIR / "emnist-balanced-train-images-idx3-ubyte"
            lbl_file = EMNIST_DATA_DIR / "emnist-balanced-train-labels-idx1-ubyte"
        else:
            img_file = EMNIST_DATA_DIR / "emnist-balanced-test-images-idx3-ubyte"
            lbl_file = EMNIST_DATA_DIR / "emnist-balanced-test-labels-idx1-ubyte"
        
        if not img_file.exists() or not lbl_file.exists():
            raise FileNotFoundError(f"EMNIST files not found in {EMNIST_DATA_DIR}.")
        
        # Load all images and labels
        all_images = read_idx_file(str(img_file))
        all_labels = read_idx_file(str(lbl_file))
        
        if all_images is None or all_labels is None:
            raise RuntimeError("Failed to read EMNIST files")
        
        # Filter to only include samples of the target type
        mask = np.array([label in self.target_emnist_indices for label in all_labels])
        self.images = all_images[mask]
        raw_labels = all_labels[mask]
        
        # Remap EMNIST labels to specialist model labels (0 to N-1)
        self.labels = np.array([self.emnist_to_specialist[label] for label in raw_labels])
        
        print(f"  {model_type.upper()} Dataset ({split}): {len(self.images)} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Read image and convert to PIL (grayscale)
        img_array = self.images[idx]
        img_array_transposed = np.transpose(img_array)
        img = Image.fromarray(img_array_transposed.astype(np.uint8), mode='L')
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
    
    with tqdm(train_loader, desc="Training") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
    
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'val_loss': total_loss / (pbar.n + 1)})
    
    return total_loss / len(val_loader), 100 * correct / total


def train_specialist_model(model_type='digit'):
    """Train a specialist model for a specific character type."""
    global device
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}\n")
    
    print(f"\n{'='*70}")
    print(f"TRAINING SPECIALIST MODEL: {model_type.upper()}")
    print(f"{'='*70}\n")
    
    # Determine number of classes
    if model_type == 'digit':
        num_classes = len(DIGIT_CLASSES)
    elif model_type == 'uppercase':
        num_classes = len(UPPERCASE_CLASSES)
    elif model_type == 'lowercase':
        num_classes = len(LOWERCASE_CLASSES)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"Model Type: {model_type}")
    print(f"Number of classes: {num_classes}")
    
    # Create datasets
    print(f"\nLoading {model_type} training dataset...")
    train_dataset = SpecialistDataset(
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]),
        split='train',
        model_type=model_type
    )
    
    print(f"Loading {model_type} validation dataset...")
    val_dataset = SpecialistDataset(
        transform=transforms.ToTensor(),
        split='val',
        model_type=model_type
    )
    
    # Create dataloaders
    batch_size = TRAINING["batch_size"]
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
    
    # Initialize model
    print(f"\nInitializing Enhanced CNN model for {model_type}...")
    model = CharacterCNN(num_classes=num_classes, dropout_rate=0.25)
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=TRAINING["scheduler_patience"],
        min_lr=1e-6,
        verbose=True
    )
    
    # Training loop
    num_epochs = TRAINING["num_epochs"]
    best_val_loss = float('inf')
    best_model_path = None
    patience_counter = 0
    early_stopping_patience = TRAINING["early_stopping_patience"]
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = MODELS_DIR / f"best_{model_type}_model.pt"
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"✓ Best model saved: {best_model_path}")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save epoch checkpoint
        epoch_path = MODELS_DIR / f"epoch_{model_type}_{epoch+1:03d}.pt"
        torch.save(model.state_dict(), epoch_path)
    
    # Save training history
    history_path = LOGS_DIR / f"history_{model_type}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training Complete for {model_type.upper()}")
    print(f"Best Model: {best_model_path}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {history['val_acc'][history['val_loss'].index(best_val_loss)]:.2f}%")
    print(f"{'='*70}\n")
    
    return best_model_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPECIALIST MODEL TRAINING PIPELINE")
    print("Training 3 separate models for digits, uppercase, and lowercase")
    print("="*70)
    
    # Train all three models
    model_types = ['digit', 'uppercase', 'lowercase']
    best_models = {}
    
    for model_type in model_types:
        try:
            best_model_path = train_specialist_model(model_type)
            best_models[model_type] = str(best_model_path)
        except Exception as e:
            print(f"ERROR training {model_type} model: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_type, path in best_models.items():
        print(f"✓ {model_type.upper():12} - {path}")
    print("="*70 + "\n")
