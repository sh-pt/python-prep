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

    # code below is too slow, should use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for col in input.columns:
            s = input[col].astype(np.float64, copy=False)
            futures[col] = executor.submit(ewm_mean_series, s, halflife, group, weight)

        # build new out_cols to collect results, .result() will wait till future complete
        out_cols = {}
        for col in input.columns:
            out_cols[col] = futures[col].result()

    return pd.DataFrame(out_cols, index=input.index, columns=input.columns)

    # out_cols = {}
    # for col in input.columns:
    #     s = input[col].astype(np.float64, copy=False)
    #     out_cols[col] = ewm_mean_series(s, halflife, group, weight)
    # return pd.DataFrame(out_cols, index=input.index, columns=input.columns)


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

        g_codes = codes.astype(np.float64, copy=False)

    # elif np.isscalar(group):
    #     if group not in input.index.names:
    #         raise ValueError('If "group" is a scalar it has to be a valid index name')
    #     g_vals = input.index.get_level_values(group)
    #     codes, uniques = pd.factorize(g_vals, sort=False)
    #     if (codes == -1).any():
    #         codes = codes.copy()
    #         codes[codes == -1] = len(uniques)
    #         n_groups = len(uniques) + 1
    #     else:
    #         n_groups = len(uniques)
    #     g_codes = codes.astype(np.int64, copy=False)
    # elif isinstance(group, pd.Series):
    #     if len(group) != len(x):
    #         raise ValueError('If "group" is a Series, it needs to be the same length as input')
    #     codes, uniques = pd.factorize(group, sort=False)
    #     if (codes == -1).any():
    #         codes = codes.copy()
    #         codes[codes == -1] = len(uniques)
    #         n_groups = len(uniques) + 1
    #     else:
    #         n_groups = len(uniques)
    #     g_codes = codes.astype(np.int64, copy=False)
    # else:
    #     raise TypeError('"group" must be a Series or scalar')

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

    # new part, for Cython
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    w_c = np.ascontiguousarray(w_arr, dtype=np.float64)
    g_c = np.ascontiguousarray(g_codes, dtype=np.int64)
    out = ewm_fast.ewm_kernel(x_c, w_c, g_c, float(alpha), int(n_groups))

    # following was the old calculation been replaced by Cython
    '''
    s_acc = np.zeros(n_groups, dtype=np.float64)
    w_acc = np.zeros(n_groups, dtype=np.float64)

    out = np.empty(len(x), dtype=np.float64)

    for i in range(len(x)):
        xi = x[i]
        gi = g_codes[i]
        wi = w_arr[i]

        s_prev = s_acc[gi]
        w_prev = w_acc[gi]

        if np.isnan(xi) or wi == 0.0:  # if weight is 0, it is the same as input as N/A, skip that row
            s_now = s_prev * (1.0 - alpha)
            w_now = w_prev * (1.0 - alpha)
        else:
            s_now = alpha * wi * xi + (1 - alpha) * s_prev
            w_now = alpha * wi + (1 - alpha) * w_prev

        s_acc[gi] = s_now
        w_acc[gi] = w_now

        out[i] = (s_now / w_now) if w_now != 0.0 else np.nan
    '''

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

    # If input is a dataframe
    idx = pd.MultiIndex.from_product([["A", "B"], pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])],
                                     names=["stock", "ts"])
    input_df = pd.DataFrame(np.random.randn(6,2), index=idx, columns=['ret', 'chg'], dtype='float64')
    w_df = pd.Series([1.0, 0.5, 2.0, 0.0, 1.0, 2.0], index=idx, dtype="float64")
    out6 = ewm_mean(input_df, halflife=2, group='stock', weight=w_df)
    print('out6')
    print(out6)
