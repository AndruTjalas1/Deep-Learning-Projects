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
NUM_CLASSES = 47  # EMNIST balanced split
IMG_SIZE = 28  # Standard MNIST-like size

# EMNIST balanced split - OFFICIAL mapping from emnist-balanced-mapping.txt
# Order: 0-9 (digits), A-Z (all uppercase), then lowercase non-merged (a,b,d,e,f,g,h,n,q,r,t)
EMNIST_CLASSES = [
    # 0-9: digits
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # 10-35: ALL uppercase A-Z
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    # 36-46: lowercase non-merged (skipping c, i, j, k, l, m, o, p, s, u, v, w, x, y, z which are merged with uppercase)
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

CHAR_TO_IDX = {char: idx for idx, char in enumerate(EMNIST_CLASSES)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# ============================================================================
# 3-MODEL SPECIALIST SYSTEM
# ============================================================================
# Character type classes for specialist models
DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
UPPERCASE_CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
LOWERCASE_CLASSES = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# Mappings for each specialist model
DIGIT_CHAR_TO_IDX = {char: idx for idx, char in enumerate(DIGIT_CLASSES)}
DIGIT_IDX_TO_CHAR = {idx: char for char, idx in DIGIT_CHAR_TO_IDX.items()}

UPPERCASE_CHAR_TO_IDX = {char: idx for idx, char in enumerate(UPPERCASE_CLASSES)}
UPPERCASE_IDX_TO_CHAR = {idx: char for char, idx in UPPERCASE_CHAR_TO_IDX.items()}

LOWERCASE_CHAR_TO_IDX = {char: idx for idx, char in enumerate(LOWERCASE_CLASSES)}
LOWERCASE_IDX_TO_CHAR = {idx: char for char, idx in LOWERCASE_CHAR_TO_IDX.items()}

# Combined mapping from EMNIST index to specialist model info
MODEL_TYPE_FOR_CLASS = {}
for idx, char in enumerate(EMNIST_CLASSES):
    if char in DIGIT_CLASSES:
        MODEL_TYPE_FOR_CLASS[idx] = ('digit', DIGIT_CHAR_TO_IDX[char])
    elif char in UPPERCASE_CLASSES:
        MODEL_TYPE_FOR_CLASS[idx] = ('uppercase', UPPERCASE_CHAR_TO_IDX[char])
    elif char in LOWERCASE_CLASSES:
        MODEL_TYPE_FOR_CLASS[idx] = ('lowercase', LOWERCASE_CHAR_TO_IDX[char])

print(f"Configuration loaded. Base directory: {BASE_DIR}")
