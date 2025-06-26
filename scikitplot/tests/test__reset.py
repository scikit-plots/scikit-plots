import numpy as np
import pytest
import sklearn
import matplotlib.pyplot as plt

from .._reset import (
    reset,
    reset_numpy,
    reset_sklearn,
    reset_matplotlib,
)

def test_numpy_reset():
    np.set_printoptions(precision=2)
    np.seterr(all='ignore')
    reset_numpy()
    assert np.get_printoptions()["precision"] != 2
    assert np.geterr() == {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}

def test_sklearn_reset():
    sklearn.set_config(assume_finite=True)
    reset_sklearn()
    assert sklearn.get_config()["assume_finite"] is False

def test_matplotlib_reset():
    plt.figure()
    assert plt.get_fignums()  # should be at least one open
    reset_matplotlib()
    assert not plt.get_fignums()  # all figures closed

@pytest.mark.parametrize("reset_fn", [reset])
def test_reset_all_executes(reset_fn):
    # Change global states
    np.seterr(all="ignore")
    sklearn.set_config(assume_finite=True)
    plt.figure()

    reset_fn()

    # Check they are restored
    assert not plt.get_fignums()
    assert sklearn.get_config()["assume_finite"] is False
    assert np.geterr()["divide"] == "warn"
