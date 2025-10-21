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
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.05], name='return')
    result_no_weight = ewm_mean(returns, halflife=2)
    print('result without weight')
    print(result_no_weight)

    returns_na = pd.Series([0.01, 0.02, -0.01, 0.03, np.nan, np.nan, np.nan, 0.05], name='return')
    result_na = ewm_mean(returns_na, halflife=2)
    print('result with na')
    print(result_na)

    weights = [1,2,1,1,3]
    result_with_weight = ewm_mean(returns, halflife=2, weight=weights)
    print('result with weight')
    print(result_with_weight)



