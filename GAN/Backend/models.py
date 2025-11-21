"""DCGAN model architecture."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN Generator network."""
    
    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        image_channels: int = 3,
        image_resolution: int = 64,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize Generator.
        
        Args:
            latent_dim: Dimension of latent vector
            feature_maps: Base number of feature maps
            image_channels: Number of output image channels (3 for RGB)
            image_resolution: Target image resolution (64, 128, 256, etc.)
            dropout_rate: Dropout probability (default: 0.2)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.image_channels = image_channels
        self.image_resolution = image_resolution
        self.dropout_rate = dropout_rate
        
        # Calculate number of layers based on resolution
        # 64: 4 layers, 128: 5 layers, 256: 6 layers
        self.num_layers = int(torch.log2(torch.tensor(image_resolution))) - 2
        
        # Initial linear layer to generate feature maps
        self.fc = nn.Linear(
            latent_dim,
            feature_maps * 8 * 4 * 4
        )
        
        # Build transpose convolution layers
        layers = []
        in_channels = feature_maps * 8
        
        for i in range(self.num_layers):
            out_channels = in_channels // 2
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout_rate))
            in_channels = out_channels
        
        # Final layer to get RGB image
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated image of shape (batch_size, channels, height, width)
        """
        x = self.fc(z)
        x = x.view(-1, self.feature_maps * 8, 4, 4)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    """DCGAN Discriminator network."""
    
    def __init__(
        self,
        feature_maps: int = 64,
        image_channels: int = 3,
        image_resolution: int = 64,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize Discriminator.
        
        Args:
            feature_maps: Base number of feature maps
            image_channels: Number of input image channels (3 for RGB)
            image_resolution: Input image resolution (64, 128, 256, etc.)
            dropout_rate: Dropout probability (default: 0.2)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Simple discriminator architecture that works for 64x64 images
        # Based on proven DCGAN architecture from the literature
        self.main = nn.Sequential(
            # Input: (batch, 3, 64, 64)
            nn.Conv2d(image_channels, feature_maps, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            # (batch, 64, 32, 32)
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            # (batch, 128, 16, 16)
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            # (batch, 256, 8, 8)
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout_rate),
            # (batch, 512, 4, 4)
            
            nn.Conv2d(feature_maps * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            # (batch, 1, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image of shape (batch_size, channels, height, width)
            
        Returns:
            Binary classification (real/fake) of shape (batch_size,)
        """
        output = self.main(x)
        return output.view(-1)


def weights_init(m: nn.Module) -> None:
    """Initialize network weights."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
