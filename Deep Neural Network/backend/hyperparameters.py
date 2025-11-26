"""
Hyperparameters Configuration File

All training, model, and inference hyperparameters in one place.
Edit this file to adjust settings without modifying code.
"""

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

MODEL_ARCHITECTURE = {
    "name": "CharacterCNN",
    "input_channels": 1,  # Grayscale
    "input_size": 28,  # 28x28 images
    "num_classes": 36,  # 0-9 (10) + A-Z (26)
    
    # Conv layers: (out_channels, kernel_size)
    "conv_layers": [
        {"out_channels": 32, "kernel_size": 3, "padding": 1},
        {"out_channels": 64, "kernel_size": 3, "padding": 1},
        {"out_channels": 128, "kernel_size": 3, "padding": 1},
    ],
    
    # Fully connected layers
    "fc_layers": [256, 128],  # Hidden layers before output
    
    # Dropout rate
    "dropout_rate": 0.5,  # Increased for better regularization across writing styles
    
    # Batch normalization
    "use_batch_norm": True,
    
    # Activation function
    "activation": "relu",  # relu, elu, gelu
    
    # Pooling
    "pool_type": "maxpool",  # maxpool, avgpool
    "pool_size": 2,
}


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAINING = {
    # Epochs
    "num_epochs": 100,  # More epochs for better convergence and style generalization
    
    # Batch sizes (optimized for GPU with parallel data loading)
    "batch_size": 128,  # Reduced for more stable gradient updates across styles
    "val_batch_size": 256,
    "test_batch_size": 256,
    
    # Learning rate
    "learning_rate": 0.0005,  # Reduced for more stable training across styles
    
    # Optimizer
    "optimizer": "adam",  # adam, sgd, rmsprop
    "momentum": 0.9,  # For SGD
    "weight_decay": 1e-5,  # L2 regularization
    
    # Learning rate scheduler
    "scheduler": "reduce_on_plateau",  # reduce_on_plateau, step, exponential
    "scheduler_factor": 0.5,  # Multiply LR by this when triggered
    "scheduler_patience": 3,  # Epochs to wait before reducing LR
    
    # Loss function
    "loss_function": "cross_entropy",  # cross_entropy, focal_loss
    
    # Data augmentation - INCREASED for better style generalization
    "augmentation": {
        "rotation": 25,  # Increased from 10 - handle rotated writing
        "translation": 0.2,  # Increased from 0.1 - handle different positions
        "shear": 15,  # Increased from 5 - handle slanted writing
        "scale": 0.25,  # Increased from 0.1 - handle different sizes
        "brightness": 0.4,  # Increased from 0.2 - handle ink variations
        "contrast": 0.4,  # Increased from 0.2 - handle thickness variations
    },
    
    # Data normalization
    "normalize_mean": 0.5,
    "normalize_std": 0.5,
    
    # Data split ratios
    "train_split": 0.8,  # 80% for training
    "val_split": 0.1,  # 10% for validation (of remaining)
    "test_split": 0.1,  # 10% for testing (of remaining)
    
    # Early stopping (optional)
    "early_stopping": False,
    "early_stopping_patience": 5,  # Stop if no improvement for N epochs
}


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

DATA_PREPROCESSING = {
    # Image size
    "image_size": 28,
    
    # Threshold for binary conversion
    "binary_threshold": 127,
    
    # Character segmentation
    "min_component_size": 30,  # Minimum pixels for valid character
    "max_components": 50,  # Max characters per sentence
    "segmentation_padding": 5,  # Padding around character
    
    # Morphological operations
    "morph_kernel_size": 3,
    "morph_iterations_close": 2,
    "morph_iterations_open": 1,
}


# ============================================================================
# MODEL EVALUATION & INFERENCE
# ============================================================================

EVALUATION = {
    # Confidence threshold for grading
    "grade_thresholds": {
        "A": 0.90,  # >= 90% confidence
        "B": 0.80,  # >= 80% confidence
        "C": 0.70,  # >= 70% confidence
        "D": 0.60,  # >= 60% confidence
        "F": 0.00,  # < 60% confidence
    },
    
    # Monte Carlo dropout samples for uncertainty
    "mc_dropout_samples": 10,
    
    # Metrics to compute
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    ],
}


# ============================================================================
# API & DEPLOYMENT
# ============================================================================

DEPLOYMENT = {
    # Flask server
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,  # Set to True for development
    
    # CORS
    "cors_enabled": True,
    "cors_origins": [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    
    # API timeouts
    "request_timeout": 30,  # Seconds
    
    # Device
    "device": "auto",  # "auto", "cuda", "cpu"
    
    # Model checkpoint path
    "model_checkpoint": "saved_models/character_cnn.pt",
}


# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOGGING = {
    # Log level
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Save training history
    "save_history": True,
    "history_file": "logs/training_history.json",
    
    # Save plots
    "save_plots": True,
    "plot_dir": "logs/",
    
    # Checkpoint frequency
    "save_checkpoint_every_n_epochs": 5,
    "save_best_model": True,
    "best_model_path": "saved_models/character_cnn_best.pt",
}


# ============================================================================
# SUMMARY
# ============================================================================
"""
Quick Reference:

KEY PARAMETERS TO ADJUST:

1. For faster training:
   - Reduce num_epochs (currently 20)
   - Increase batch_size (currently 64)
   - Use "adam" optimizer (already set)

2. For better accuracy:
   - Increase num_epochs (try 50-100)
   - Reduce learning_rate (try 0.0005)
   - Increase dropout_rate (try 0.5)
   - Add more augmentation

3. For deployment on CPU:
   - Reduce model_architecture.conv_layers channels
   - Reduce batch_size
   - Reduce num_epochs for testing

4. For mobile/edge deployment:
   - Use quantization (post-training)
   - Reduce model size
   - Use smaller input_size (20x20 instead of 28x28)

TYPICAL TRAINING TIME:
- GPU (RTX 3060): 30-60 minutes for 20 epochs
- CPU: 2-4 hours for 20 epochs

EXPECTED ACCURACY:
- With good data: 90-95%
- With limited data: 70-85%
- With untrained model: ~3% (random)
"""


def get_config():
    """Get complete configuration dictionary."""
    return {
        "model": MODEL_ARCHITECTURE,
        "training": TRAINING,
        "preprocessing": DATA_PREPROCESSING,
        "evaluation": EVALUATION,
        "deployment": DEPLOYMENT,
        "logging": LOGGING,
    }


if __name__ == "__main__":
    # Print all hyperparameters
    import json
    config = get_config()
    print(json.dumps(config, indent=2))
