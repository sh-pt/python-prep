import pandas as pd
import numpy as np
import ewm_fast
import concurrent.futures


# aggregated func, calls "ewm_mean_df" or "ewm_mean_series" based on the input
def ewm_mean(input, halflife: float, group=None, weight=None):

    if isinstance(input, pd.Series):
        return ewm_mean_series(input, halflife, group, weight)
    elif isinstance(input, pd.DataFrame):
        return ewm_mean_df(input, halflife, group, weight)
    else:
        raise ValueError('Input must be either a Pandas Series or a Pandas DataFrame')


def ewm_mean_df(input: pd.DataFrame, halflife: float, group=None, weight=None) -> pd.DataFrame:

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for col in input.columns:
            s = input[col].astype(np.float64, copy=False)
            futures[col] = executor.submit(ewm_mean_series, s, halflife, group, weight)

        out_cols = {}
        for col in input.columns:
            out_cols[col] = futures[col].result()

    return pd.DataFrame(out_cols, index=input.index, columns=input.columns)


def ewm_mean_series(input: pd.Series, halflife: float, group=None, weight=None) -> pd.Series:
    x = input.to_numpy(dtype=np.float64, copy=False)

    if not np.isscalar(halflife) or halflife <= 0:
        raise ValueError('"halflife" must be a positive float')

    # if group is a scalar, it should match one of the level of index name
    # if group is a Series, it should match input length and result will group on that
    if group is None:
        g_codes = np.zeros(len(x), dtype=np.int64)
        n_groups = 1

    # clean up the group input, get "group_data"
    else:
        if np.isscalar(group):
            if group not in input.index.names:
                raise ValueError('If "group" is a scalar it has to be a valid index name')
            group_data = input.index.get_level_values(group)
        elif isinstance(group, pd.Series):
            if len(group) != len(x):
                raise ValueError('If "group" is a Series, it needs to be the same length as input')
            group_data = group
        else:
            raise TypeError('"group" must be a Series or scalar')

        # factorize "group_data"
        codes, uniques = pd.factorize(group_data)

        # Handle NaNs, factorize will return -1
        if (codes == -1).any():
            codes = codes.copy()
            codes[codes == -1] = len(uniques)
            n_groups = len(uniques) + 1
        else:
            n_groups = len(uniques)

        g_codes = codes.astype(np.int64, copy=False)

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

    s_acc = np.zeros(n_groups, dtype=np.float64)
    w_acc = np.zeros(n_groups, dtype=np.float64)
    out = np.empty(len(x), dtype=np.float64)

    x_c = np.ascontiguousarray(x, dtype=np.float64)
    w_c = np.ascontiguousarray(w_arr, dtype=np.float64)
    g_c = np.ascontiguousarray(g_codes, dtype=np.int64)

    ewm_fast.ewm_kernel(x_c, w_c, g_c, float(alpha), int(n_groups), s_acc, w_acc, out)

    return pd.Series(out, index=input.index, name=f'{input.name}_ewm_mean')