# MethodFiles/scale_features.py
def scale_features(dataset):
    """Min-max scale dataset features to [0,1]. Returns (scaled, mins, maxs)."""
    n = len(dataset[0]["x"])
    mins = [min(row["x"][i] for row in dataset) for i in range(n)]
    maxs = [max(row["x"][i] for row in dataset) for i in range(n)]
    def s(v):
        return [(v[i]-mins[i])/(maxs[i]-mins[i]) if maxs[i]!=mins[i] else 0.0 for i in range(n)]
    scaled = [{"x":s(r["x"]), "y":r["y"]} for r in dataset]
    return scaled, mins, maxs
