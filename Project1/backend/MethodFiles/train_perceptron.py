# MethodFiles/train_perceptron.py
import random
from .dot import dot
from .step_fn import step_fn

def train_perceptron(data, lr=0.15, max_epochs=1000, seed=7, shuffle=True):
    """Classic perceptron training with your update rule. Returns (weights, bias, history)."""
    n = len(data[0]["x"])
    random.seed(seed)
    w = [random.uniform(-0.5, 0.5) for _ in range(n)]
    b = random.uniform(-0.5, 0.5)
    hist = []
    for ep in range(1, max_epochs+1):
        errs = 0
        if shuffle:
            random.shuffle(data)
        for row in data:
            x, yt = row["x"], row["y"]
            z = dot(w, x) + b
            yhat = step_fn(z)
            if yhat != yt:
                errs += 1
                if yt == +1:
                    for i in range(n): w[i] += lr * x[i]
                    b += lr
                else:
                    for i in range(n): w[i] -= lr * x[i]
                    b -= lr
        hist.append((ep, errs))
        if errs == 0:
            break
    return w, b, hist
