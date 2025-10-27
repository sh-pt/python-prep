import pandas as pd
import numpy as np
from ._ewm_fast import ewm_kernel

def ewm_mean(input, halflife: float, group=None, weight=None):
    if isinstance(input, pd.Series):
        return ewm_mean_series(input, halflife, group, weight)
    elif isinstance(input, pd.DataFrame):
        return ewm_mean_df(input, halflife, group, weight)
    else:
        raise ValueError('Input must be either a Pandas Series or a Pandas DataFrame')

def ewm_mean_df(input: pd.DataFrame, halflife: float, group=None, weight=None) -> pd.DataFrame:
    out_cols = {}
    for col in input.columns:
        s = input[col].astype(np.float64, copy=False)
        out_cols[col] = ewm_mean_series(s, halflife, group, weight)
    return pd.DataFrame(out_cols, index=input.index, columns=input.columns)

def ewm_mean_series(input: pd.Series, halflife: float, group=None, weight=None) -> pd.Series:
    x = input.to_numpy(dtype=np.float64, copy=False)

    if not np.isscalar(halflife) or halflife <= 0:
        raise ValueError('"halflife" must be a positive float')

    if group is None:
        g_codes = np.zeros(len(x), dtype=np.int64)
        n_groups = 1
    elif np.isscalar(group):
        if group not in input.index.names:
            raise ValueError('If "group" is a scalar it has to be a valid index name')
        g_vals = input.index.get_level_values(group)
        codes, uniques = pd.factorize(g_vals, sort=False)
        if (codes == -1).any():
            codes = codes.copy()
            codes[codes == -1] = len(uniques)
            n_groups = len(uniques) + 1
        else:
            n_groups = len(uniques)
        g_codes = codes.astype(np.int64, copy=False)
    elif isinstance(group, pd.Series):
        if len(group) != len(x):
            raise ValueError('If "group" is a Series, it needs to be the same length as input')
        codes, uniques = pd.factorize(group, sort=False)
        if (codes == -1).any():
            codes = codes.copy()
            codes[codes == -1] = len(uniques)
            n_groups = len(uniques) + 1
        else:
            n_groups = len(uniques)
        g_codes = codes.astype(np.int64, copy=False)
    else:
        raise TypeError('"group" must be a Series or scalar')

    if weight is None:
        w_arr = np.ones(len(input), dtype=np.float64)
    else:
        w_arr = weight.to_numpy(dtype=np.float64, copy=False)
        if len(w_arr) != len(input):
            raise ValueError('"weight" must have the same length as "input"')
        if (w_arr < 0).any():
            raise ValueError('"weight" must have non-negative values')
        w_arr = np.where(np.isnan(w_arr), 0.0, w_arr)

    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

    x_c = np.ascontiguousarray(x, dtype=np.float64)
    w_c = np.ascontiguousarray(w_arr, dtype=np.float64)
    g_c = np.ascontiguousarray(g_codes, dtype=np.int64)
    out = ewm_kernel(x_c, w_c, g_c, float(alpha), int(n_groups))

    return pd.Series(out, index=input.index, name=f'{input.name}_ewm_mean')
