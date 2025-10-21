import pytest
import numpy as np
import pandas as pd
from src import ewm_mean

# ---- reference using default packages --------

def _alpha(halflife):
    if halflife <= 0:
        raise ValueError('halflife must be positive')
    return 1.0 - np.exp(-np.log(2.0) / float(halflife))


def _cleanup_weights(weight, index):
    if weight is None:
        return pd.Series(1.0, index=index, dtype=np.float64)
    if len(weight) != len(index):
        raise ValueError('index must have same length as weight')
    w = pd.Series(weight, index=index, dtype=np.float64)
    w.fillna(0, inplace=True) # N/A in weights considered as skip
    if (w<0).any():
        raise ValueError('weights must be non-negative')
    return w

def ref_ewm_no_weights(s, halflife, groups=None, weights=None):
    ...



# ------- test cases ---------
