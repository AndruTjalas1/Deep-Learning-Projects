"""Data loading utilities for DCGAN training."""

from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class AnimalDataset(Dataset):
    """Dataset for loading animal images (cats, dogs, etc.)."""
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 64,
        transform: Optional[transforms.Compose] = None,
        animal_types: Optional[list] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing animal subdirectories
            resolution: Image resolution to resize to
            transform: Optional torchvision transforms
            animal_types: List of animal types to include (e.g., ['cats', 'dogs'])
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.animal_types = animal_types or ['cats', 'dogs']
        self.image_paths = []
        self.labels = []
        
        # Default transforms if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.CenterCrop((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])
        
        self.transform = transform
        self._load_images()
    
    def _load_images(self) -> None:
        """Load image paths from directory structure."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for animal_idx, animal_type in enumerate(self.animal_types):
            animal_dir = self.data_dir / animal_type
            
            if not animal_dir.exists():
                print(f"Warning: Directory not found: {animal_dir}")
                continue
            
            # Find all images in the directory
            for image_file in animal_dir.iterdir():
                if image_file.suffix.lower() in valid_extensions:
                    self.image_paths.append(str(image_file))
                    self.labels.append(animal_idx)
        
        if not self.image_paths:
            raise ValueError(
                f"No images found in {self.data_dir}. "
                f"Expected subdirectories: {self.animal_types}"
            )
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.animal_types)} animal types")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label by index."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a random image on error
            return self[np.random.randint(0, len(self))]


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    resolution: int = 64,
    train_split: float = 0.8,
    animal_types: Optional[list] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_dir: Root directory containing animal subdirectories
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        resolution: Image resolution
        train_split: Fraction of data for training
        animal_types: List of animal types to include
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = AnimalDataset(
        data_dir=data_dir,
        resolution=resolution,
        animal_types=animal_types,
    )
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader


def create_train_loader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    resolution: int = 64,
    animal_types: Optional[list] = None,
) -> DataLoader:
    """
    Create only a training data loader (no validation split).
    
    Args:
        data_dir: Root directory containing animal subdirectories
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        resolution: Image resolution
        animal_types: List of animal types to include
        
    Returns:
        DataLoader for training
    """
    dataset = AnimalDataset(
        data_dir=data_dir,
        resolution=resolution,
        animal_types=animal_types,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return loader
