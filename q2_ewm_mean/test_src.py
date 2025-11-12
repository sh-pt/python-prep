import test_src
import pandas as pd
import numpy as np
import pandas.testing as tm
from src import ewm_mean


def test_out1_normal():
    """Test a normal series with no groups or weights."""
    ret = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15], name='return', dtype='float64')

    result = ewm_mean(ret, halflife=2, group=None, weight=None)

    expected_vals = [-0.050000, -0.020711, 0.011327, 0.045956, 0.082974]
    expected = pd.Series(expected_vals, name='return_ewm_mean', dtype='float64')

    tm.assert_series_equal(result, expected, check_index_type=False, atol=1e-5)


def test_out2_with_na():
    """Test a series with an internal NaN value."""
    ret_nan = pd.Series([-0.05, 0.00, 0.05, np.nan, 0.15], name='return', dtype='float64')

    result = ewm_mean(ret_nan, halflife=2, group=None, weight=None)

    expected_vals = [-0.050000, -0.020711, 0.011327, 0.011327, 0.077250]
    expected = pd.Series(expected_vals, name='return_ewm_mean', dtype='float64')

    tm.assert_series_equal(result, expected, check_index_type=False, atol=1e-5)


def test_out3_with_weight():
    """Test a series with a corresponding weight series."""
    ret_w = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15], name='return', dtype='float64')
    w = pd.Series([1.0, 2.0, 10.0, 1.0, 1.0], index=ret_w.index, dtype='float64')

    result = ewm_mean(ret_w, halflife=2, group=None, weight=w)

    expected_vals = [-0.050000, -0.013060, 0.039868, 0.046249, 0.059786]
    expected = pd.Series(expected_vals, name='return_ewm_mean', dtype='float64')

    tm.assert_series_equal(result, expected, check_index_type=False, atol=1e-5)


def test_out4_group_series():
    """Test grouping by an external Series."""
    stock_id = pd.Series(['A', 'A', 'A', 'B', 'A'])
    ret_g = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15], dtype='float64')
    w_g = pd.Series([1.0, 0.5, 2.0, 0.0, 1.0], dtype='float64')

    result = ewm_mean(ret_g, halflife=2, group=stock_id, weight=w_g)

    expected_vals = [-0.050000, -0.029289, 0.026283, np.nan, 0.067279]
    expected = pd.Series(expected_vals, name='None_ewm_mean', dtype='float64')

    tm.assert_series_equal(result, expected, check_index_type=False, atol=1e-5)


def test_out5_group_index_level():
    """Test grouping by a MultiIndex level name."""
    idx = pd.MultiIndex.from_product([["A", "B"], pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])],
                                     names=["stock", "ts"])
    ret_mi = pd.Series([-0.05, 0.00, 0.05, 0.10, 0.15, 0.10], index=idx, dtype="float64")
    w_mi = pd.Series([1.0, 0.5, 2.0, 0.0, 1.0, 2.0], index=idx, dtype="float64")

    result = ewm_mean(ret_mi, halflife=2, group='stock', weight=w_mi)

    expected_vals = [-0.050000, -0.029289, 0.026283, np.nan, 0.150000, 0.113060]
    expected = pd.Series(expected_vals, index=idx, name='None_ewm_mean', dtype='float64')

    tm.assert_series_equal(result, expected, check_index_type=False, atol=1e-5)