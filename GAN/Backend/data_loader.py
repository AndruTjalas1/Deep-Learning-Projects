"""Data loading utilities for DCGAN training - uses Microsoft Cats and Dogs dataset."""

from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np


class MicrosoftCatsDogsDataset(Dataset):
    """Custom dataset for Microsoft Cats and Dogs."""
    
    def __init__(self, root_dir: str, animal_filter: Optional[List[str]] = None, 
                 resolution: int = 64, transform=None, max_images: int = None):
        """Initialize dataset.
        
        Args:
            root_dir: Root directory containing animal folders
            animal_filter: List of animal types to include
            resolution: Image resolution
            transform: Image transforms
            max_images: Maximum number of images to load (useful for faster training during testing)
        """
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.image_paths = []
        self.labels = []
        
        # Default transforms
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.CenterCrop((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.transform = transform
        
        # Normalize animal filter
        if animal_filter is None:
            animal_filter = ['cat', 'dog']
        animal_filter = [a.lower().rstrip('s') for a in animal_filter]
        
        # Load images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for animal_idx, animal in enumerate(animal_filter):
            animal_dir = self.root_dir / animal
            if animal_dir.exists():
                for img_file in animal_dir.glob('**/*'):
                    if img_file.suffix.lower() in valid_extensions:
                        self.image_paths.append(str(img_file))
                        self.labels.append(animal_idx)
                        
                        # Stop if we've reached max_images
                        if max_images and len(self.image_paths) >= max_images:
                            break
            if max_images and len(self.image_paths) >= max_images:
                break
        
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root_dir}")
        
        print(f"Loaded {len(self.image_paths)} images" + (f" (limited to {max_images})" if max_images else ""))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self[np.random.randint(0, len(self))]


def _load_from_directory(
    batch_size: int = 64,
    resolution: int = 64,
    animal_filter: Optional[List[str]] = None,
    num_workers: int = 0,
    max_images: int = None,
) -> DataLoader:
    """Load from custom directory structure."""
    import sys
    if sys.platform == "win32":
        num_workers = 0
    
    if animal_filter is None:
        animal_filter = ['cat', 'dog']
    
    dataset = MicrosoftCatsDogsDataset(
        root_dir="./data",
        animal_filter=animal_filter,
        resolution=resolution,
        max_images=max_images,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return loader


def load_dataset(
    batch_size: int = 64,
    resolution: int = 64,
    animal_filter: Optional[List[str]] = None,
    num_workers: int = 0,
    max_images: int = None,
) -> DataLoader:
    """
    Main entry point for loading dataset.
    Loads from ./data/cat and ./data/dog directories.
    
    Args:
        batch_size: Batch size for data loader
        resolution: Image resolution to resize to
        animal_filter: List of animal types (['cat'], ['dog'], or ['cat', 'dog'])
        num_workers: Number of workers (default 0 for Windows)
        max_images: Max images to load (useful for faster testing, e.g., 2000)
        
    Returns:
        DataLoader for training
    """
    import sys
    if sys.platform == "win32":
        num_workers = 0
    
    if animal_filter is None:
        animal_filter = ['cat', 'dog']
    
    # Normalize filter
    animal_filter = [a.lower().rstrip('s') for a in animal_filter]
    
    print(f"\n{'='*60}")
    print(f"Loading dataset for: {animal_filter}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Batch size: {batch_size}")
    if max_images:
        print(f"Max images: {max_images}")
    print(f"{'='*60}\n")
    
    # Load from directory
    data_dir = Path("./data")
    
    # Check if we have local data
    has_local_data = False
    for animal in animal_filter:
        animal_dir = data_dir / animal
        if animal_dir.exists():
            images = list(animal_dir.glob('**/*.jpg')) + list(animal_dir.glob('**/*.jpeg')) + list(animal_dir.glob('**/*.png'))
            if images:
                has_local_data = True
                print(f"Found {len(images)} images in {animal_dir}")
    
    if not has_local_data:
        print("\n" + "!"*60)
        print("ERROR: Dataset not found!")
        print("!"*60)
        print("\nTo set up the Microsoft Cats and Dogs dataset:")
        print("\n1. Download from:")
        print("   https://www.microsoft.com/en-us/download/details.aspx?id=54765")
        print("\n2. Extract the dataset so the directory structure looks like:")
        print("   ./data/")
        print("   ├── cat/")
        print("   │   ├── cat.0.jpg")
        print("   │   ├── cat.1.jpg")
        print("   │   └── ...")
        print("   └── dog/")
        print("       ├── dog.0.jpg")
        print("       ├── dog.1.jpg")
        print("       └── ...")
        print("\n3. Then run training:")
        print(f"   python train.py --animals {','.join(animal_filter)}")
        print("!"*60 + "\n")
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}. "
            "Please follow the setup instructions above."
        )
    
    return _load_from_directory(
        batch_size=batch_size,
        resolution=resolution,
        animal_filter=animal_filter,
        num_workers=num_workers,
        max_images=max_images,
    )
