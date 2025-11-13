import time
import pandas as pd
import numpy as np
import src


def make_data_df(n_rows, n_cols, n_groups):
    data = {f'col_{i}': np.random.normal(size=n_rows) for i in range(n_cols)}
    df = pd.DataFrame(data)
    g = pd.Series(np.random.randint(0, n_groups, size=n_rows))
    w = pd.Series(np.random.random(size=n_rows))

    return df, g, w


def ewm_mean_df_benchmark(input: pd.DataFrame, halflife: float, group=None, weight=None) -> pd.DataFrame:

    out_cols = {}
    for col in input.columns:
        s = input[col].astype(np.float64, copy=False)
        out_cols[col] = src.ewm_mean_series(s, halflife, group, weight)
    return pd.DataFrame(out_cols, index=input.index, columns=input.columns)


if __name__ == "__main__":

    df, g, w = make_data_df(n_rows=10_000_000, n_cols=100, n_groups=1000)
    halflife = 120.0

    print("Running benchmark...")
    start_benchmark = time.perf_counter()
    res_benchmark = ewm_mean_df_benchmark(df, halflife, group=g, weight=w)
    end_benchmark = time.perf_counter()
    benchmark_time = end_benchmark - start_benchmark
    print(f"  Benchmark time: {benchmark_time:.4f} s")

    print("Running parallel...")
    start_parallel = time.perf_counter()
    res_parallel = src.ewm_mean(df, halflife, group=g, weight=w)
    end_parallel = time.perf_counter()
    parallel_time = end_parallel - start_parallel
    print(f"  Parallel time: {parallel_time:.4f} s")