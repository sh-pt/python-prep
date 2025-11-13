import pandas as pd
import numpy as np
import ewm_fast
import concurrent.futures


def _get_args(input_len, halflife, group_data, weight):

    if not np.isscalar(halflife) or halflife <= 0:
        raise ValueError('"halflife" must be a positive float')

    # 1. Process Group
    if group_data is None:
        g_codes = np.zeros(input_len, dtype=np.int64)
        n_groups = 1
    else:
        if not isinstance(group_data, (pd.Series, pd.Index)):
            raise TypeError(f"Internal error: _get_args expected a Series")

        if len(group_data) != input_len:
            raise ValueError('If "group" is a Series, it needs to be the same length as input')

        codes, uniques = pd.factorize(group_data)
        if (codes == -1).any():
            codes = codes.copy()
            codes[codes == -1] = len(uniques)
            n_groups = len(uniques) + 1
        else:
            n_groups = len(uniques)

        g_codes = codes.astype(np.int64, copy=False)

    # 2. Process Weight
    if weight is None:
        w_arr = np.ones(input_len, dtype=np.float64)
    else:
        w_arr = weight.to_numpy(dtype=np.float64, copy=False)
        if len(w_arr) != input_len:
            raise ValueError('"weight" must have the same length as "input"')
        if (w_arr < 0).any():
            raise ValueError('"weight" must have non-negative values')
        w_arr = np.where(np.isnan(w_arr), 0.0, w_arr)  # skip the na weights

    # 3. Process Alpha
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

    # 4. Ensure C-Contiguous
    w_c = np.ascontiguousarray(w_arr, dtype=np.float64)
    g_c = np.ascontiguousarray(g_codes, dtype=np.int64)

    return w_c, g_c, float(alpha), int(n_groups)


# aggregated func, calls "ewm_mean_df" or "ewm_mean_series" based on the input
def ewm_mean(input, halflife: float, group=None, weight=None):

    if isinstance(input, pd.Series):
        return ewm_mean_series(input, halflife, group, weight)
    elif isinstance(input, pd.DataFrame):
        return ewm_mean_df(input, halflife, group, weight)
    else:
        raise ValueError('Input must be either a Pandas Series or a Pandas DataFrame')


def ewm_mean_df(input: pd.DataFrame, halflife: float, group=None, weight=None) -> pd.DataFrame:

    if np.isscalar(group) and group in input.index.names:
        group_data = input.index.get_level_values(group)
    else:
        group_data = group

    w_c, g_c, alpha, n_groups = _get_args(len(input), halflife, group_data, weight)

    out_cols_data = {col: np.empty(len(input), dtype=np.float64) for col in input.columns}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for col in input.columns:
            s = input[col].to_numpy(dtype=np.float64, copy=False)
            x_c = np.ascontiguousarray(s, dtype=np.float64)

            out = out_cols_data[col]

            futures.append(
                executor.submit(
                    ewm_fast.ewm_kernel, x_c, w_c, g_c, alpha, n_groups, out
                )
            )

        for f in futures:
            f.result()

    return pd.DataFrame(out_cols_data, index=input.index, columns=input.columns)


def ewm_mean_series(input: pd.Series, halflife: float, group=None, weight=None) -> pd.Series:

    if np.isscalar(group) and group in input.index.names:
        group_data = input.index.get_level_values(group)
    else:
        group_data = group

    w_c, g_c, alpha, n_groups = _get_args(len(input), halflife, group_data, weight)

    x = input.to_numpy(dtype=np.float64, copy=False)
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty(len(x), dtype=np.float64)

    ewm_fast.ewm_kernel(x_c, w_c, g_c, alpha, n_groups, out)

    return pd.Series(out, index=input.index, name=f'{input.name}_ewm_mean')