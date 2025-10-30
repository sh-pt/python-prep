import numpy as np, pandas as pd
from src import ewm_mean

try:
    profile
except NameError:
    def profile(func):
        return func

def make_data(n=1_000_000, n_groups=500, nan_ratio=0.05, seed=0, weighted=True):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=n), name="val")
    g = pd.Series(rng.integers(0, n_groups, size=n), name="grp")
    if nan_ratio:
        mask = rng.random(n) < nan_ratio
        x[mask] = np.nan
    w = pd.Series(rng.random(size=n)) if weighted else None

    perm = rng.permutation(n)
    x, g = x.iloc[perm], g.iloc[perm]
    if w is not None:
        w = w.iloc[perm]

    s = pd.DataFrame({"x": x, "grp": g}).set_index("grp", append=True)["x"]
    return s, w

@profile
def run_once():
    s, w = make_data()
    ewm_mean(s, halflife=120.0, group="grp", weight=w)


if __name__ == "__main__":
    run_once()
