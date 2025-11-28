# CRITICAL: Set CUDA environment BEFORE importing torch
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import struct

from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, NUM_CLASSES,
    EMNIST_CLASSES, MODEL_TYPE_FOR_CLASS,
    DIGIT_CLASSES, UPPERCASE_CLASSES, LOWERCASE_CLASSES
)

device = None
EMNIST_DATA_DIR = Path(__file__).parent / "emnist_data"

# Character type mapping
CHAR_TYPE_CLASSES = ['digit', 'uppercase', 'lowercase']
CHAR_TYPE_TO_IDX = {char_type: idx for idx, char_type in enumerate(CHAR_TYPE_CLASSES)}
CHAR_TYPE_IDX_TO_NAME = {idx: char_type for char_type, idx in CHAR_TYPE_TO_IDX.items()}

print(f"\n{'='*70}")
print(f"Character Type Classifier - Training Configuration")
print(f"Classes: {CHAR_TYPE_CLASSES}")
print(f"Mapping: {CHAR_TYPE_TO_IDX}")
print(f"{'='*70}\n")


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


class CharacterTypeDataset(Dataset):
    """Dataset for character type classification (digit/uppercase/lowercase)."""
    
    def __init__(self, transform=None, split='train'):
        """
        Args:
            split: 'train' or 'val'
        """
        self.transform = transform
        
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
        
        # Map EMNIST labels (0-46) to character types (0-2)
        type_labels = []
        for emnist_label in all_labels:
            if emnist_label in MODEL_TYPE_FOR_CLASS:
                model_type, _ = MODEL_TYPE_FOR_CLASS[emnist_label]
                type_idx = CHAR_TYPE_TO_IDX[model_type]
                type_labels.append(type_idx)
            else:
                # Fallback (shouldn't happen)
                type_labels.append(0)
        
        self.images = all_images
        self.labels = np.array(type_labels, dtype=np.int64)
        
        print(f"  Character Type Dataset ({split}): {len(self.images)} samples")
        # Print distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for idx, count in zip(unique, counts):
            print(f"    {CHAR_TYPE_IDX_TO_NAME[idx]}: {count} samples")
    
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


class LightCharacterTypeClassifier(nn.Module):
    """
    Lightweight classifier for character type (digit/uppercase/lowercase).
    Uses a simple CNN to quickly classify character types.
    """
    
    def __init__(self, num_classes=3):
        super(LightCharacterTypeClassifier, self).__init__()
        
        # Lightweight architecture - 2 conv blocks only
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x


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


def train_character_type_classifier():
    """Train the light character type classifier."""
    global device
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}\n")
    
    print(f"\n{'='*70}")
    print(f"TRAINING CHARACTER TYPE CLASSIFIER")
    print(f"Purpose: Automatically detect digit/uppercase/lowercase")
    print(f"Architecture: Lightweight 2-layer CNN")
    print(f"{'='*70}\n")
    
    # Create datasets
    print(f"Loading character type training dataset...")
    train_dataset = CharacterTypeDataset(
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ]),
        split='train'
    )
    
    print(f"Loading character type validation dataset...")
    val_dataset = CharacterTypeDataset(
        transform=transforms.ToTensor(),
        split='val'
    )
    
    # Create dataloaders
    batch_size = 256  # Larger batch for faster training
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
    print(f"\nInitializing Light Character Type Classifier...")
    model = LightCharacterTypeClassifier(num_classes=3)
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Training setup with class weight balancing to reduce digit bias
    unique_labels, counts = np.unique(train_dataset.labels, return_counts=True)
    total_samples = len(train_dataset)
    class_weights = torch.tensor([total_samples / (len(counts) * count) for count in counts], dtype=torch.float32)
    class_weights = class_weights.to(device)
    print(f"  Class weights (digit/uppercase/lowercase): {[round(w.item(), 3) for w in class_weights]}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=True
    )
    
    # Training loop
    num_epochs = 20  # Fewer epochs for light model
    best_val_loss = float('inf')
    best_model_path = None
    patience_counter = 0
    early_stopping_patience = 5
    
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
            best_model_path = MODELS_DIR / "character_type_classifier.pt"
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
        epoch_path = MODELS_DIR / f"epoch_character_type_{epoch+1:03d}.pt"
        torch.save(model.state_dict(), epoch_path)
    
    # Save training history
    history_path = LOGS_DIR / "history_character_type_classifier.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training Complete for Character Type Classifier")
    print(f"Best Model: {best_model_path}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {history['val_acc'][history['val_loss'].index(best_val_loss)]:.2f}%")
    print(f"{'='*70}\n")
    
    return best_model_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHARACTER TYPE CLASSIFIER TRAINING")
    print("Lightweight model to classify digit/uppercase/lowercase")
    print("="*70)
    
    try:
        best_model_path = train_character_type_classifier()
        print(f"\n✓ SUCCESS: Character type classifier trained!")
        print(f"  Model saved to: {best_model_path}")
        print(f"  This classifier will be used in the OCR pipeline to route")
        print(f"  characters to the appropriate specialist model.")
    except Exception as e:
        print(f"ERROR training character type classifier: {e}")
        import traceback
        traceback.print_exc()
