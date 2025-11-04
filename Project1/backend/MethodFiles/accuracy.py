# MethodFiles/accuracy.py
from .dot import dot
from .step_fn import step_fn

def accuracy(w, b, data):
    """Compute perceptron accuracy over data."""
    return sum(1 for r in data if step_fn(dot(w, r["x"]) + b) == r["y"]) / len(data)
