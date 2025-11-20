"""Training loop and utilities for DCGAN."""

import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import time
from datetime import datetime

from models import Generator, Discriminator, weights_init
from device import get_device


class DCGANTrainer:
    """Trainer class for DCGAN models."""
    
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        config: dict,
        device: torch.device,
    ):
        """
        Initialize trainer.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            config: Configuration dictionary
            device: Device to train on (cuda, mps, cpu)
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.config = config
        
        # Setup optimizers
        self.lr = config['training']['learning_rate']
        self.beta1 = config['training']['beta1']
        self.beta2 = config['training']['beta2']
        
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.g_losses = []
        self.d_losses = []
        self.training_started = False
        self.training_completed = False
        
        # Create output directories
        self._create_output_dirs()
    
    def _create_output_dirs(self) -> None:
        """Create output directories for samples and models."""
        Path(self.config['output']['samples_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output']['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['output']['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (g_loss, d_loss) for the epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batch_count = 0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)
            
            # ============
            # Train Discriminator
            # ============
            self.d_optimizer.zero_grad()
            
            # Real images
            d_real_output = self.discriminator(real_images)
            d_real_loss = self.criterion(d_real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, self.config['generator']['latent_dim'], device=self.device)
            fake_images = self.generator(z)
            d_fake_output = self.discriminator(fake_images.detach())
            d_fake_loss = self.criterion(d_fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # ============
            # Train Generator
            # ============
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config['generator']['latent_dim'], device=self.device)
            fake_images = self.generator(z)
            d_fake_output = self.discriminator(fake_images)
            
            # Generator wants to fool discriminator
            g_loss = self.criterion(d_fake_output, real_labels)
            g_loss.backward()
            self.g_optimizer.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            batch_count += 1
            self.current_batch += 1
            
            # Generate and save samples
            if self.current_batch % self.config['sampling']['sample_interval'] == 0:
                self._save_sample(epoch, batch_idx)
                print(
                    f"Epoch [{epoch}/{self.config['training']['epochs']}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"D Loss: {d_loss.item():.4f} "
                    f"G Loss: {g_loss.item():.4f}"
                )
        
        # Average losses
        avg_g_loss = epoch_g_loss / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        
        return avg_g_loss, avg_d_loss
    
    def _save_sample(self, epoch: int, batch: int) -> None:
        """Save generated sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(
                self.config['sampling']['num_samples'],
                self.config['generator']['latent_dim'],
                device=self.device,
            )
            fake_images = self.generator(z)
        
        # Create filename
        sample_path = Path(self.config['output']['samples_dir']) / f"epoch_{epoch:03d}_batch_{batch:05d}.png"
        
        # Save grid of images
        save_image(
            fake_images.cpu(),
            str(sample_path),
            nrow=4,
            normalize=True,
            range=(-1, 1),
        )
        
        self.generator.train()
    
    def save_checkpoint(self, epoch: int, suffix: str = "") -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'config': self.config,
        }
        
        models_dir = Path(self.config['output']['models_dir'])
        checkpoint_path = models_dir / f"checkpoint_epoch_{epoch:03d}{suffix}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.current_epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: Optional[int] = None,
        checkpoint_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the DCGAN model.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train (uses config if None)
            checkpoint_interval: Save checkpoint every N epochs
            
        Returns:
            Dictionary with training results
        """
        num_epochs = num_epochs or self.config['training']['epochs']
        self.training_started = True
        
        start_time = time.time()
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {num_epochs}, Batch size: {self.config['training']['batch_size']}")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                
                epoch_start = time.time()
                g_loss, d_loss = self.train_epoch(train_loader, epoch)
                epoch_time = time.time() - epoch_start
                
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] - "
                    f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Save checkpoint
                if (epoch + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(epoch + 1)
            
            # Save final checkpoint
            self.save_checkpoint(num_epochs, suffix="_final")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint(self.current_epoch, suffix="_interrupted")
        
        total_time = time.time() - start_time
        self.training_completed = True
        
        return {
            'epochs': num_epochs,
            'total_time': total_time,
            'avg_time_per_epoch': total_time / num_epochs,
            'final_g_loss': self.g_losses[-1] if self.g_losses else None,
            'final_d_loss': self.d_losses[-1] if self.d_losses else None,
        }
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return {
            'training_started': self.training_started,
            'training_completed': self.training_completed,
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'total_epochs': self.config['training']['epochs'],
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'latest_g_loss': self.g_losses[-1] if self.g_losses else None,
            'latest_d_loss': self.d_losses[-1] if self.d_losses else None,
        }
    
    def generate_images(self, num_images: int = 16) -> torch.Tensor:
        """Generate random images."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(
                num_images,
                self.config['generator']['latent_dim'],
                device=self.device,
            )
            images = self.generator(z)
        
        self.generator.train()
        return images.cpu()
