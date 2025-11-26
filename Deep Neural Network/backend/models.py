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
    Convolutional Neural Network for character/digit recognition.
    
    Architecture:
    - Input: 28x28 grayscale images
    - Conv1: 32 filters, 3x3 kernel
    - Conv2: 64 filters, 3x3 kernel
    - FC1: 128 neurons
    - FC2: 36 neurons (output)
    
    This is the primary deep learning algorithm for character classification.
    """
    
    def __init__(self, num_classes=36, dropout_rate=0.25):
        super(CharacterCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.flatten_size = 128 * 3 * 3  # 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            logits: Raw output from final layer (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
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


def create_model(model_type="cnn", num_classes=36, pretrained=False):
    """
    Factory function to create models.
    
    Args:
        model_type: "cnn", "transfer_learning", or "confidence"
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Initialized model
    """
    if model_type == "cnn":
        return CharacterCNN(num_classes=num_classes)
    elif model_type == "transfer_learning":
        return TransferLearningCNN(num_classes=num_classes)
    elif model_type == "confidence":
        base_model = CharacterCNN(num_classes=num_classes)
        return ConfidenceScorer(base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
