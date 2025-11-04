# MethodFiles/step_fn.py
def step_fn(z):
    """Binary step activation used for perceptron updates."""
    return 1 if z >= 0 else -1
