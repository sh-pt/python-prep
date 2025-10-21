# 0.1 first try
# Simplest version, no Cython, no Group, only take in data, halflife, weight, and return ewm

import pandas as pd
import numpy as np
from collections import defaultdict


def ewm_mean(input: pd.Series, halflife: float, group=None, weight=None) -> pd.Series:
    if not isinstance(input, pd.Series):
        raise TypeError('"Input" must be a pandas Series')

    x = input.to_numpy(dtype=np.float64, copy=False)

    if not np.isscalar(halflife) or halflife <= 0:
        raise ValueError('"halflife" must be a positive float')

    # if group is a scalar, it should match one of the level of index name
    # if group is a Series, it should match input length and result will group on that
    if group is None:
        g_arr = np.ones(len(input.index))
    elif np.isscalar(group):
        if group not in input.index.names:
            raise ValueError('If group is a scalar it has to be one of the index name')
        else:
            g_arr = input.index.get_level_values(group).to_numpy(copy=False)
    elif isinstance(group, pd.Series):
        if len(group) != len(input.index):
            raise ValueError('If "Group" is a Series, it needs to be the same length as input')
        else:
            g_arr = group.to_numpy(copy=False)
    else:
        raise TypeError('"group" must be a pandas Series or a scalar')

    # if there's no weight, we set it as 1, 1, 1, ...
    if weight is None:
        w_arr = np.ones(len(input), dtype=np.float64)
    else:
        w_arr = weight.to_numpy(dtype=np.float64, copy=False)
        if len(w_arr) != len(input):
            raise ValueError('"weight" must have the same length as "input"')
        if (w_arr < 0).any():
            raise ValueError('"weight" must have non-negative values')
        w_arr = np.where(np.isnan(w_arr), 0.0, w_arr) # skip the na weights

    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

    s_dict = defaultdict(float)
    w_dict = defaultdict(float)
    out = np.empty(len(x), dtype=np.float64)

    for i in range(len(x)):
        xi = x[i]
        gi = g_arr[i]
        wi = w_arr[i]

        s_prev = s_dict[gi]
        w_prev = w_dict[gi]

        if np.isnan(xi) or wi == 0.0:  # if weight is 0, it is the same as input as N/A, skip that row
            s_now = s_prev * (1.0 - alpha)
            w_now = w_prev * (1.0 - alpha)
        else:
            s_now = alpha * wi * xi + (1 - alpha) * s_prev
            w_now = alpha * wi + (1 - alpha) * w_prev

        s_dict[gi] = s_now
        w_dict[gi] = w_now

        out[i] = (s_now / w_now) if w_now != 0.0 else np.nan

    return pd.Series(out, index=input.index, name=f'{input.name}_ewm_mean')


if __name__ == '__main__':
    # normal
    ret = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15], name='return', dtype='float64')
    out1 = ewm_mean(ret, halflife=2, group=None, weight=None)
    print('out1')
    print(out1)

    # with na
    ret_nan = pd.Series([-0.05, 0.00, 0.05, np.nan, 0.15], name='return', dtype='float64')
    out2 = ewm_mean(ret_nan, halflife=2, group=None, weight=None)
    print('out2')
    print(out2)

    # with weight
    ret_w = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15], name='return', dtype='float64')
    w = pd.Series([1.0, 2.0, 10.0, 1.0, 1.0], index=ret_w.index, dtype='float64')
    out3 = ewm_mean(ret_w, halflife=2, group=None, weight=w)
    print('out3')
    print(out3)

    # If group input is a series, group based on that
    stock_id = pd.Series(['A', 'A', 'A', 'B', 'A'])
    ret_g = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15], dtype='float64')
    w_g = pd.Series([1.0, 0.5, 2.0, 0.0, 1.0], dtype='float64')
    out4 = ewm_mean(ret_g, halflife=2, group=stock_id, weight=w_g)
    print('out4')
    print(out4)

    # If group input is a string, group based on that index level
    idx = pd.MultiIndex.from_product([["A", "B"], pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])],
                                     names=["stock", "ts"])
    ret_mi = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15, 0.10], index=idx, dtype="float64")
    w_mi = pd.Series([1.0, 0.5, 2.0, 0.0, 1.0, 2.0], index=idx, dtype="float64")
    out5 = ewm_mean(ret_mi, halflife=2, group='stock', weight=w_mi)
    print('out5')
    print(out5)
