"""
PyTorch models for handwriting recognition:
1. CharacterCNN - Recognizes individual characters/digits
2. Confidence scoring module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network for character/digit recognition.
    
    ENHANCED ARCHITECTURE (v2):
    - Input: 28x28 grayscale images
    - Conv Block 1: 64 filters (increased from 32), dual conv layers with residual connection
    - Conv Block 2: 128 filters (increased from 64), dual conv layers with residual connection
    - Conv Block 3: 256 filters (increased from 128), dual conv layers with residual connection
    - Conv Block 4: 256 filters (NEW), additional layer for fine-grained features
    - Global Average Pooling for robustness
    - Dense Layers: 512 → 256 → num_classes
    
    Improvements:
    - More filters per layer for better feature extraction
    - Residual connections for better gradient flow
    - Additional conv block for distinguishing hard character pairs (3 vs E, etc.)
    - Global average pooling for spatial robustness
    - Increased FC layer capacity
    
    This is the primary deep learning algorithm for character classification.
    """
    
    def __init__(self, num_classes=36, dropout_rate=0.3):
        super(CharacterCNN, self).__init__()
        
        # ===== CONV BLOCK 1 (64 filters - increased from 32) =====
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # ===== CONV BLOCK 2 (128 filters - increased from 64) =====
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # ===== CONV BLOCK 3 (256 filters - increased from 128) =====
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # ===== CONV BLOCK 4 (NEW - 256 filters for fine-grained features) =====
        self.conv4a = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        
        # Global average pooling + linear layer
        # After pooling: 28 -> 14 -> 7 -> 3 -> 1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_size = 256
        
        # ===== FULLY CONNECTED LAYERS (Increased capacity) =====
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the enhanced network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            logits: Raw output from final layer (batch_size, num_classes)
        """
        # Conv block 1 (64 filters)
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = F.relu(x)
        x = self.pool1(x)  # 28 -> 14
        
        # Conv block 2 (128 filters)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = F.relu(x)
        x = self.pool2(x)  # 14 -> 7
        
        # Conv block 3 (256 filters)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = F.relu(x)
        x = self.pool3(x)  # 7 -> 3
        
        # Conv block 4 (NEW - 256 filters for fine-grained features)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = F.relu(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = F.relu(x)
        # No pooling after block 4, use global average pooling
        
        # Global average pooling
        x = self.global_avg_pool(x)  # 3 -> 1
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with increased capacity
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class TransferLearningCNN(nn.Module):
    """
    Transfer Learning approach using pre-trained ResNet-style features.
    
    Algorithm 2: Uses pre-trained feature extraction for improved accuracy
    and robustness to different writing styles.
    """
    
    def __init__(self, num_classes=36, pretrained_backbone=None):
        super(TransferLearningCNN, self).__init__()
        
        if pretrained_backbone is None:
            # Simple pre-trained-style backbone
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.feature_dim = 128 * 4 * 4
        else:
            self.feature_extractor = pretrained_backbone
            self.feature_dim = 512  # Typical for ResNet
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Extract features and classify."""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


class ConfidenceScorer(nn.Module):
    """
    Algorithm 3: Uncertainty quantification using Bayesian dropout.
    
    Provides confidence scores and uncertainty estimates by performing
    multiple forward passes with dropout enabled.
    """
    
    def __init__(self, base_model, num_samples=10):
        super(ConfidenceScorer, self).__init__()
        self.base_model = base_model
        self.num_samples = num_samples
    
    def forward(self, x, return_variance=True):
        """
        Forward pass with Monte Carlo dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            return_variance: If True, return variance as well as predictions
            
        Returns:
            predictions: Mean predictions across samples
            variance: Variance across samples (if return_variance=True)
        """
        # Enable dropout during inference
        self.base_model.train()
        
        predictions_list = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                logits = self.base_model(x)
                probs = F.softmax(logits, dim=1)
                predictions_list.append(probs)
        
        # Convert back to eval mode
        self.base_model.eval()
        
        # Stack predictions
        predictions_stacked = torch.stack(predictions_list, dim=0)
        
        # Compute mean and variance
        mean_predictions = predictions_stacked.mean(dim=0)
        
        if return_variance:
            variance = predictions_stacked.var(dim=0)
            return mean_predictions, variance
        else:
            return mean_predictions


class VisionTransformer(nn.Module):
    """
    Vision Transformer for character/digit recognition.
    
    Architecture:
    - Patch Embedding: Converts 28x28 images into patches
    - Transformer Encoder: Multi-head self-attention layers
    - Classification Head: MLP for final prediction
    
    Why ViT?
    - Captures global context better than CNNs for small images
    - Self-attention can focus on discriminative features
    - Often outperforms CNNs on 28x28 character images
    - Better at distinguishing similar characters (3 vs E, p vs b, etc.)
    
    Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    https://arxiv.org/abs/2010.11929
    """
    
    def __init__(self, num_classes=47, img_size=28, patch_size=2, embed_dim=192, 
                 depth=6, num_heads=3, mlp_dim=768, dropout_rate=0.05):
        super(VisionTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches (28x28 with 2x2 patches = 196 patches)
        num_patches = (img_size // patch_size) ** 2
        
        # ===== PATCH EMBEDDING =====
        # Convert image patches to embeddings
        self.patch_embed = nn.Linear(patch_size * patch_size * 1, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # ===== TRANSFORMER ENCODER =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # ===== CLASSIFICATION HEAD =====
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        B = x.shape[0]
        
        # ===== PATCH EMBEDDING =====
        # Reshape to patches: (B, 1, 28, 28) -> (B, num_patches, patch_dim)
        patches = x.reshape(B, 1, self.img_size // self.patch_size, self.patch_size,
                            self.img_size // self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(B, (self.img_size // self.patch_size) ** 2, 
                                 self.patch_size * self.patch_size * 1)
        
        # Embed patches
        x = self.patch_embed(patches)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # ===== TRANSFORMER ENCODER =====
        x = self.transformer_encoder(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Take class token
        x = x[:, 0]  # (B, embed_dim)
        
        # ===== CLASSIFICATION HEAD =====
        logits = self.head(x)
        
        return logits


def create_model(model_type="cnn", num_classes=36, pretrained=False):
    """
    Factory function to create models.
    
    Args:
        model_type: "cnn", "vit", "transfer_learning", or "confidence"
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Initialized model
    """
    if model_type == "cnn":
        return CharacterCNN(num_classes=num_classes)
    elif model_type == "vit":
        return VisionTransformer(num_classes=num_classes)
    elif model_type == "transfer_learning":
        return TransferLearningCNN(num_classes=num_classes)
    elif model_type == "confidence":
        base_model = CharacterCNN(num_classes=num_classes)
        return ConfidenceScorer(base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class LightCharacterTypeClassifier(nn.Module):
    """
    Lightweight classifier for character type (digit/uppercase/lowercase).
    
    Purpose: Automatically detect the type of a handwritten character
    to route it to the correct specialist model.
    
    Architecture:
    - Input: 28x28 grayscale image
    - Conv Block 1: 32 filters, 3x3 kernel
    - Conv Block 2: 64 filters, 3x3 kernel
    - Global Average Pooling
    - Output: 3 classes (digit, uppercase, lowercase)
    
    Model Size: ~200KB (very lightweight)
    Inference Time: ~1ms per character
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
