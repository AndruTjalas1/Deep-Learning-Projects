# Code Examples & Advanced Usage

This guide shows how to use the DCGAN system programmatically and perform advanced tasks.

## Table of Contents

1. [Python API Examples](#python-api-examples)
2. [JavaScript/Frontend Examples](#javascriptfrontend-examples)
3. [Advanced Training](#advanced-training)
4. [Custom Models](#custom-models)
5. [Data Processing](#data-processing)

## Python API Examples

### Example 1: Basic Training Script

```python
"""
Example: Training DCGAN from Python script
"""

from pathlib import Path
from config import get_config
from device import get_device
from models import Generator, Discriminator, weights_init
from data_loader import create_train_loader
from trainer import DCGANTrainer

# Load configuration
config = get_config("config.yaml")

# Get device
device = get_device(config.device.use_gpu)

# Create models
generator = Generator(
    latent_dim=config.generator.latent_dim,
    feature_maps=config.generator.feature_maps,
    image_channels=config.image.channels,
    image_resolution=config.image.resolution,
)

discriminator = Discriminator(
    feature_maps=config.discriminator.feature_maps,
    image_channels=config.image.channels,
    image_resolution=config.image.resolution,
)

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Create trainer
trainer = DCGANTrainer(
    generator=generator,
    discriminator=discriminator,
    config=config.model_dump(),
    device=device,
)

# Load data
train_loader = create_train_loader(
    data_dir=config.data.dataset_path,
    batch_size=config.training.batch_size,
    num_workers=config.data.num_workers,
    resolution=config.image.resolution,
    animal_types=['cats', 'dogs'],
)

# Train
results = trainer.train(
    train_loader=train_loader,
    num_epochs=config.training.epochs,
)

print(f"Training completed in {results['total_time']:.2f} seconds")
print(f"Final G Loss: {results['final_g_loss']:.4f}")
print(f"Final D Loss: {results['final_d_loss']:.4f}")
```

### Example 2: Generate Images

```python
"""
Example: Generate images with trained model
"""

import torch
from pathlib import Path
from models import Generator
from torchvision.utils import save_image

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("Backend/saved_models/checkpoint_epoch_050_final.pt", map_location=device)

generator = Generator(
    latent_dim=100,
    feature_maps=64,
    image_channels=3,
    image_resolution=64,
)
generator.load_state_dict(checkpoint['generator_state'])
generator.to(device)
generator.eval()

# Generate random images
with torch.no_grad():
    z = torch.randn(16, 100, device=device)
    fake_images = generator(z)

# Save
save_image(
    fake_images.cpu(),
    "generated_samples.png",
    nrow=4,
    normalize=True,
    range=(-1, 1),
)
```

### Example 3: Load and Resume Training

```python
"""
Example: Resume training from checkpoint
"""

from trainer import DCGANTrainer

# Create trainer (same as before)
trainer = DCGANTrainer(...)

# Load checkpoint
trainer.load_checkpoint("Backend/saved_models/checkpoint_epoch_050.pt")

# Resume training
results = trainer.train(
    train_loader=train_loader,
    num_epochs=100,  # Continue to epoch 100
)
```

### Example 4: Custom Configuration

```python
"""
Example: Create and use custom configuration
"""

from config import Config, TrainingConfig, ImageConfig

# Create custom config
custom_config = Config(
    training=TrainingConfig(
        epochs=200,
        batch_size=32,
        learning_rate=0.0001,
    ),
    image=ImageConfig(
        resolution=256,
        channels=3,
    ),
)

# Save for later use
custom_config.save_to_yaml("my_config.yaml")

# Load later
loaded_config = Config.load_from_yaml("my_config.yaml")
```

### Example 5: Evaluate Training Metrics

```python
"""
Example: Analyze training metrics
"""

import torch
import matplotlib.pyplot as plt

# Load checkpoint
checkpoint = torch.load("Backend/saved_models/checkpoint_epoch_100_final.pt")

g_losses = checkpoint['g_losses']
d_losses = checkpoint['d_losses']

# Plot losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator Loss Over Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(d_losses, label='Discriminator Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Loss Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
```

## JavaScript/Frontend Examples

### Example 1: API Integration

```javascript
// Custom fetch function with error handling
import trainingApi from './api';

async function startAndMonitorTraining() {
  try {
    // Start training
    const result = await trainingApi.startTraining(['cats', 'dogs'], 50);
    console.log('Training started:', result);

    // Poll status every 5 seconds
    const statusInterval = setInterval(async () => {
      const status = await trainingApi.getTrainingStatus();
      console.log(`Epoch: ${status.current_epoch}/${status.total_epochs}`);
      
      if (status.training_completed) {
        clearInterval(statusInterval);
        console.log('Training completed!');
      }
    }, 5000);

  } catch (error) {
    console.error('Training failed:', error);
  }
}
```

### Example 2: Custom React Component

```jsx
// Component with auto-updating metrics
import React, { useState, useEffect } from 'react';
import trainingApi from '../api';

export function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await trainingApi.getTrainingMetrics();
        setMetrics(data);
      } catch (err) {
        console.error('Failed to fetch metrics:', err);
      }
    }, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, []);

  if (!metrics) return <div>Loading...</div>;

  return (
    <div>
      <h3>Epoch: {metrics.current_epoch + 1} / {metrics.total_epochs}</h3>
      <p>G Loss: {metrics.latest_g_loss?.toFixed(4)}</p>
      <p>D Loss: {metrics.latest_d_loss?.toFixed(4)}</p>
      <progress
        value={metrics.current_epoch}
        max={metrics.total_epochs}
      />
    </div>
  );
}
```

### Example 3: Download Generated Samples

```javascript
async function downloadSamples() {
  try {
    const data = await trainingApi.listSamples();
    
    // Create zip of all samples
    for (const sample of data.samples) {
      const url = trainingApi.getSampleUrl(sample);
      const link = document.createElement('a');
      link.href = url;
      link.download = sample;
      link.click();
    }
  } catch (error) {
    console.error('Download failed:', error);
  }
}
```

## Advanced Training

### Example 1: Progressive Training (Increase Resolution)

```python
"""
Example: Progressive DCGAN training
Start at 64x64, then 128x128, then 256x256
"""

from config import Config

resolutions = [64, 128, 256]

for resolution in resolutions:
    print(f"\n{'='*50}")
    print(f"Training at {resolution}x{resolution}")
    print(f"{'='*50}\n")
    
    # Update config
    config = get_config()
    config.image.resolution = resolution
    config.training.epochs = 50
    
    # Create models and train
    generator = Generator(
        latent_dim=100,
        feature_maps=64,
        image_channels=3,
        image_resolution=resolution,
    )
    # ... rest of training code
```

### Example 2: Multi-Animal Training with Switches

```python
"""
Example: Train different animal types separately
"""

animals = ['cats', 'dogs', 'birds', 'bears']

for animal in animals:
    print(f"\nTraining {animal}...")
    
    # Create animal-specific loader
    loader = create_train_loader(
        data_dir=f"data/{animal}",
        batch_size=64,
        resolution=128,
        animal_types=[animal],
    )
    
    # Train model
    trainer = DCGANTrainer(generator, discriminator, config, device)
    trainer.train(loader, num_epochs=100)
    
    # Save model specific to animal
    trainer.save_checkpoint(
        epoch=100,
        suffix=f"_final_{animal}",
    )
```

### Example 3: Hyperparameter Sweep

```python
"""
Example: Test different hyperparameters
"""

from itertools import product

learning_rates = [0.0001, 0.0002, 0.0005]
batch_sizes = [32, 64, 128]

results = []

for lr, bs in product(learning_rates, batch_sizes):
    print(f"\nTesting LR={lr}, BS={bs}")
    
    config = get_config()
    config.training.learning_rate = lr
    config.training.batch_size = bs
    
    trainer = DCGANTrainer(generator, discriminator, config, device)
    result = trainer.train(train_loader, num_epochs=50)
    
    results.append({
        'lr': lr,
        'batch_size': bs,
        'final_g_loss': result['final_g_loss'],
        'final_d_loss': result['final_d_loss'],
    })

# Find best configuration
best = min(results, key=lambda x: x['final_g_loss'])
print(f"\nBest config: LR={best['lr']}, BS={best['batch_size']}")
```

## Custom Models

### Example 1: Extend Generator

```python
"""
Example: Custom generator with different architecture
"""

import torch.nn as nn
from models import Generator

class CustomGenerator(Generator):
    """Generator with extra layers for better stability"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add spectral normalization or other improvements
        # ... custom implementation
        pass
```

### Example 2: Conditional Generator (cGAN)

```python
"""
Example: Conditional GAN that can generate specific animals
"""

import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    """Generator that takes animal class as input"""
    
    def __init__(self, latent_dim, num_classes, feature_maps, image_resolution):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding layer for class information
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Rest of architecture similar to regular Generator
        # but with class information concatenated to latent vector
```

## Data Processing

### Example 1: Data Augmentation

```python
"""
Example: Custom data augmentation pipeline
"""

from torchvision import transforms
import albumentations as A

# Using albumentations for advanced augmentation
transform = A.Compose([
    A.Resize(64, 64),
    A.CenterCrop(64, 64),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

### Example 2: Dataset Inspection

```python
"""
Example: Visualize dataset
"""

import matplotlib.pyplot as plt
from data_loader import AnimalDataset

dataset = AnimalDataset("./data", resolution=64)

# Show random samples
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    image, label = dataset[i]
    # Denormalize
    image = (image + 1) / 2
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(['Cats', 'Dogs'][label])
    ax.axis('off')

plt.tight_layout()
plt.savefig('dataset_samples.png')
```

### Example 3: Calculate Dataset Statistics

```python
"""
Example: Get dataset statistics
"""

from data_loader import AnimalDataset
import torch

dataset = AnimalDataset("./data", resolution=64)

# Calculate mean and std
mean = torch.zeros(3)
std = torch.zeros(3)
total = 0

for image, _ in dataset:
    mean += image.mean(dim=(1, 2))
    std += image.std(dim=(1, 2))
    total += 1

mean /= total
std /= total

print(f"Dataset mean: {mean}")
print(f"Dataset std: {std}")

# Use these in your config
```

## Performance Profiling

### Example: Profile Training Speed

```python
"""
Example: Measure training performance
"""

import time
import torch

def profile_training(trainer, train_loader, num_batches=100):
    """Profile training performance"""
    
    trainer.generator.train()
    trainer.discriminator.train()
    
    times = {'data': [], 'forward': [], 'backward': []}
    
    start = time.time()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # Data loading time
        data_start = time.time()
        images = images.to(trainer.device)
        times['data'].append(time.time() - data_start)
        
        # Forward pass
        forward_start = time.time()
        # ... forward pass code
        times['forward'].append(time.time() - forward_start)
        
        # Backward pass
        backward_start = time.time()
        # ... backward pass code
        times['backward'].append(time.time() - backward_start)
    
    # Print results
    print(f"Average data loading: {sum(times['data'])/len(times['data']):.4f}s")
    print(f"Average forward pass: {sum(times['forward'])/len(times['forward']):.4f}s")
    print(f"Average backward pass: {sum(times['backward'])/len(times['backward']):.4f}s")
```

---

**Use these examples to customize and extend the DCGAN system for your needs!** 🚀
