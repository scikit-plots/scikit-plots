# scikitplot/utils/tests/test__dataframe.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# This module was copied from the scikit-learn project.
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_dataframe.py
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dataframe detection functions."""

import numpy as np
import pytest

from sklearn._min_dependencies import dependent_packages
from .._dataframe import is_df_or_series, is_pandas_df, is_polars_df
from .._testing import _convert_container


@pytest.mark.parametrize("constructor_name", ["pyarrow", "dataframe", "polars"])
def test_is_df_or_series(constructor_name):
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)

    assert is_df_or_series(df)
    assert not is_df_or_series(np.asarray([1, 2, 3]))


@pytest.mark.parametrize("constructor_name", ["pyarrow", "dataframe", "polars"])
def test_is_pandas_df_other_libraries(constructor_name):
    df = _convert_container([[1, 4, 2], [3, 3, 6]], constructor_name)
    if constructor_name in ("pyarrow", "polars"):
        assert not is_pandas_df(df)
    else:
        assert is_pandas_df(df)


def test_is_pandas_df():
    """Check behavior of is_pandas_df when pandas is installed."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3]])
    assert is_pandas_df(df)
    assert not is_pandas_df(np.asarray([1, 2, 3]))
    assert not is_pandas_df(1)


def test_is_pandas_df_pandas_not_installed(hide_available_pandas):
    """Check is_pandas_df when pandas is not installed."""

    assert not is_pandas_df(np.asarray([1, 2, 3]))
    assert not is_pandas_df(1)


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        ("pyarrow", dependent_packages["pyarrow"][0]),
        ("dataframe", dependent_packages["pandas"][0]),
        ("polars", dependent_packages["polars"][0]),
    ],
)
def test_is_polars_df_other_libraries(constructor_name, minversion):
    df = _convert_container(
        [[1, 4, 2], [3, 3, 6]],
        constructor_name,
        minversion=minversion,
    )
    if constructor_name in ("pyarrow", "dataframe"):
        assert not is_polars_df(df)
    else:
        assert is_polars_df(df)


def test_is_polars_df_for_duck_typed_polars_dataframe():
    """Check is_polars_df for object that looks like a polars dataframe"""

    class NotAPolarsDataFrame:
        def __init__(self):
            self.columns = [1, 2, 3]
            self.schema = "my_schema"

    not_a_polars_df = NotAPolarsDataFrame()
    assert not is_polars_df(not_a_polars_df)


def test_is_polars_df():
    """Check that is_polars_df return False for non-dataframe objects."""

    class LooksLikePolars:
        def __init__(self):
            self.columns = ["a", "b"]
            self.schema = ["a", "b"]

    assert not is_polars_df(LooksLikePolars())
