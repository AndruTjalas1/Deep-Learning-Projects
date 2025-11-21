"""Simple DCGAN training script for cats and dogs."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
from pathlib import Path

from config import get_config
from device import get_device


# =========================================================
# SPECTRAL NORMALIZATION
# =========================================================
class SpectralNorm(nn.Module):
    """Spectral normalization wrapper for weight matrices."""
    def __init__(self, module, n_power_iterations=1):
        super().__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        self.register_buffer('u', torch.randn(1, module.weight.shape[0]))
        
    def _normalize_weight(self):
        w = self.module.weight.data
        w_shape = w.shape
        w_mat = w.reshape(w_shape[0], -1)
        
        u = self.u
        for _ in range(self.n_power_iterations):
            v = torch.nn.functional.normalize(u @ w_mat, p=2, dim=1)
            u = torch.nn.functional.normalize(v @ w_mat.t(), p=2, dim=1)
        
        self.u = u
        sigma = (u @ w_mat @ v.t()).squeeze()
        return w / (sigma + 1e-12)
    
    def forward(self, *args, **kwargs):
        self.module.weight.data = self._normalize_weight()
        return self.module(*args, **kwargs)

# =========================================================
# SIMPLE MODELS (Working version)
# =========================================================

class SimpleGenerator(nn.Module):
    """Simple generator that works reliably - supports 64x64, 128x128, and 256x256."""
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 4x4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8x8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16x16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32x32

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64x64

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # 128x128

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 256x256
        )

    def forward(self, z):
        # Reshape latent vector from [batch, nz] to [batch, nz, 1, 1]
        x = z.view(z.size(0), self.nz, 1, 1)
        return self.main(x)


class SimpleDiscriminator(nn.Module):
    """Simple discriminator that works reliably - supports 64x64, 128x128, and 256x256."""
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 256x256 -> 128x128

            SpectralNorm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x128 -> 64x64

            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 -> 32x32

            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16

            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8

            nn.AdaptiveAvgPool2d(1),
            SpectralNorm(nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1)


def weights_init(m):
    """Initialize weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# =========================================================
# SETUP
# =========================================================
config = get_config()
device = get_device(use_gpu=config.device.use_gpu)

print(f"Using device: {device}")
print(f"Image resolution: {config.image.resolution}x{config.image.resolution}")


# =========================================================
# DATASET LOADER
# =========================================================
def load_dataset(animal_filter=None, batch_size=64, max_images=None, use_cropped=True):
    """Load dataset from ./data/cat and ./data/dog directories (or cropped versions if available)."""
    from data_loader import load_dataset as load_data
    
    if animal_filter is None:
        animal_filter = ["cat", "dog"]
    
    # Normalize to singular form
    animal_filter = [a.lower().rstrip('s') for a in animal_filter]
    
    return load_data(
        batch_size=batch_size,
        resolution=config.image.resolution,
        animal_filter=animal_filter,
        num_workers=0,
        max_images=max_images,
        use_cropped=use_cropped,
    )


# =========================================================
# TRAINING LOOP
# =========================================================
def train(animal_types=None, epochs=None, batch_size=None, max_images=None, use_cropped=True):
    """Train DCGAN model.
    
    Args:
        animal_types: List of animal types to train on
        epochs: Number of epochs
        batch_size: Batch size
        max_images: Maximum images to load (for faster testing)
        use_cropped: Use cropped images if available (default: True)
    """
    
    if animal_types is None:
        animal_types = ["cat", "dog"]
    if epochs is None:
        epochs = config.training.epochs
    if batch_size is None:
        batch_size = config.training.batch_size

    print(f"\n{'='*60}")
    print(f"Training DCGAN for: {animal_types}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    if max_images:
        print(f"Max images: {max_images}")
    print(f"Using cropped images: {use_cropped}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    dataloader = load_dataset(animal_filter=animal_types, batch_size=batch_size, max_images=max_images, use_cropped=use_cropped)
    print(f"Dataset loaded: {len(dataloader)} batches")

    # Initialize models
    print("Initializing models...")
    generator = SimpleGenerator(
        nz=config.generator.latent_dim,
        ngf=config.generator.feature_maps,
        nc=config.image.channels,
    ).to(device)
    
    discriminator = SimpleDiscriminator(
        nc=config.image.channels,
        ndf=config.discriminator.feature_maps,
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)
    print("Models initialized")

    # Setup optimizers and loss
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config.training.generator_learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config.training.discriminator_learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )
    
    # Label smoothing values
    real_label_value = 1.0 - config.training.label_smoothing  # 0.9
    fake_label_value = 0.0

    # Create sample directory
    sample_dir = Path("./samples")
    sample_dir.mkdir(exist_ok=True)

    # Training loop
    print("\nStarting training...\n")
    fixed_noise = torch.randn(
        config.sampling.num_samples,
        config.generator.latent_dim,
        device=device
    )

    for epoch in range(1, epochs + 1):
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch"
        )

        g_losses = []
        d_losses = []

        for real_images, _ in progress:
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)

            # Labels with smoothing (real=0.9 instead of 1.0 to prevent overconfidence)
            real_labels = torch.full((batch_size_actual,), real_label_value, device=device)
            fake_labels = torch.full((batch_size_actual,), fake_label_value, device=device)

            # ============
            # Train Discriminator
            # ============
            d_optimizer.zero_grad()

            # Real images
            d_real_output = discriminator(real_images)
            d_real_loss = criterion(d_real_output, real_labels)

            # Fake images
            z = torch.randn(batch_size_actual, config.generator.latent_dim, device=device)
            fake_images = generator(z)
            d_fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            d_optimizer.step()

            # ============
            # Train Generator
            # ============
            g_optimizer.zero_grad()

            z = torch.randn(batch_size_actual, config.generator.latent_dim, device=device)
            fake_images = generator(z)
            d_fake_output = discriminator(fake_images)
            g_loss = criterion(d_fake_output, real_labels)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            progress.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
            })

        # Save samples
        if epoch % config.sampling.sample_interval == 0:
            with torch.no_grad():
                samples = generator(fixed_noise)
                samples = samples * 0.5 + 0.5  # Denormalize
                save_image(
                    samples,
                    sample_dir / f"epoch_{epoch:03d}.png",
                    nrow=8
                )
                print(f"Saved samples at epoch {epoch}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_dir = Path("./saved_models")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(
                generator.state_dict(),
                checkpoint_dir / f"generator_epoch_{epoch}.pt"
            )
            torch.save(
                discriminator.state_dict(),
                checkpoint_dir / f"discriminator_epoch_{epoch}.pt"
            )
            print(f"Saved checkpoint at epoch {epoch}")

    # Save final models
    checkpoint_dir = Path("./saved_models")
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), checkpoint_dir / "generator_final.pt")
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_final.pt")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Samples saved to: {sample_dir}")
    print(f"{'='*60}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DCGAN on cats and dogs")
    parser.add_argument(
        "--animals",
        type=str,
        default="cat, dog",
        help="Comma-separated animal types to train on (default: cat,dog)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to load for faster testing (e.g., 2000)"
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Use original (non-cropped) images instead of cropped versions"
    )
    
    args = parser.parse_args()
    
    # Parse animal types
    animal_types = [a.strip() for a in args.animals.split(",")]
    use_cropped = not args.no_crop
    
    train(
        animal_types=animal_types,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_images=args.max_images,
        use_cropped=use_cropped,
    )
