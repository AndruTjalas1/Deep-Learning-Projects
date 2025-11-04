# MethodFiles/apply_scaling.py
def apply_scaling(vec, mins, maxs):
    """Apply min-max scaling with previously computed mins & maxs."""
    return [(vec[i]-mins[i])/(maxs[i]-mins[i]) if maxs[i]!=mins[i] else 0.0 for i in range(len(vec))]
