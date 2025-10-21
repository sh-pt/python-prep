# 0.1 first try
# Simplest version, no Cython, no Group, only take in data, halflife, weight, and return ewm

import pandas as pd
import numpy as np


def ewm_mean(input: pd.Series, halflife: float, group=None, weight=None) -> pd.Series:

    if not isinstance(input, pd.Series):
        raise TypeError('"Input" must be a pandas Series')

    x = input.to_numpy(dtype=np.float64, copy=False)

    if not np.isscalar(halflife) or halflife <= 0:
        raise ValueError('"halflife" must be a positive float')

    if weight is None:
        w_arr = np.ones(len(input), dtype=np.float64)
    else:
        w_arr = np.asarray(weight, dtype=np.float64)
        if len(w_arr) != len(input):
            raise ValueError('"weight" must have the same length as "input"')
        if (w_arr < 0).any():
            raise ValueError('"weight" must have non-negative values')

    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

    s_prev = 0.0
    w_prev = 0.0
    out = np.empty_like(x, dtype=np.float64)

    for i in range(len(x)):
        xi = x[i]
        wi = w_arr[i]

        if np.isnan(xi):
            s_prev = s_prev * (1.0 - alpha)
            w_prev = w_prev * (1.0 - alpha)
        else:
            s_prev = alpha * wi * xi + (1 - alpha) * s_prev
            w_prev = alpha * wi + (1 - alpha) * w_prev

        out[i] = s_prev / w_prev if w_prev != 0 else np.nan

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
    out3 = ewm_mean(ret_w, halflife=2, group=None, weight=None)
    print('out3')
    print(out3)

    # If group input is a series, group based on that
    stock_id = pd.Series(['A','A','A','B','A'])
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

