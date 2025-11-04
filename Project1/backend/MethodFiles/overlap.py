# MethodFiles/overlap.py
def overlap(a0, a1, b0, b1):
    """Return True if [a0,a1] overlaps [b0,b1]."""
    return (a0 <= b1) and (b0 <= a1)
