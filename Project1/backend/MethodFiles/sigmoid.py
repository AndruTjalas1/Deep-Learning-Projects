# MethodFiles/sigmoid.py
import math
def sigmoid(z):
    """Sigmoid activation: maps real number to (0,1)."""
    return 1.0 / (1.0 + math.exp(-z))
