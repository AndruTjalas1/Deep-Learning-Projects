"""
Configuration file for the Handwriting Recognition System
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "char_classifier": {
        "architecture": "CNN",
        "input_size": 28,
        "num_classes": 36,  # 26 letters + 10 digits
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 0.001,
    },
    "segmentation": {
        "method": "connected_components",  # or "cnn_segmentation"
        "min_component_size": 10,
        "dilation_kernel": 3,
    },
    "confidence": {
        "method": "softmax",  # or "monte_carlo_dropout"
        "dropout_samples": 10,
    },
}

# Flask Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development")
DEBUG = FLASK_ENV == "development"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,https://yourvercel.app").split(",")

# Data Configuration
NUM_CLASSES = 36  # 26 letters (A-Z) + 10 digits (0-9)
IMG_SIZE = 28  # Standard MNIST-like size

# Character mapping
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS[:NUM_CLASSES])}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

print(f"Configuration loaded. Base directory: {BASE_DIR}")
