import numpy as np, pandas as pd
from ewmcore import ewm_mean

rng = np.random.default_rng(0)
n = 20000
x = pd.Series(rng.normal(size=n))
g = pd.Series(rng.integers(0, 50, size=n), name="grp")

s = pd.DataFrame({"x": x, "grp": g}).set_index("grp", append=True)["x"]
y = ewm_mean(s, halflife=120, group="grp", weight=None)

p_base = (
    s.groupby(level="grp")
     .apply(lambda ser: ser.ewm(halflife=120, adjust=True, ignore_na=False).mean())
     .droplevel(0)
     .reindex(s.index)
)

diff = np.abs(p_base.values - y.values)
print("max abs diff:", np.nanmax(diff))
