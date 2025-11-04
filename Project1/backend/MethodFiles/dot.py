# MethodFiles/dot.py
def dot(a, b):
    """Dot product of two equal-length lists/tuples."""
    return sum(i * j for i, j in zip(a, b))
