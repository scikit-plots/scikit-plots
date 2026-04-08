# scikitplot/utils/_dataframe.py
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

"""
Functions to determine if an object is a dataframe or series.

.. seealso::
  * https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_dataframe.py
"""

import sys


def is_df_or_series(X):
    """Return True if the X is a dataframe or series.

    Parameters
    ----------
    X : {array-like, dataframe}
        The array-like or dataframe object to check.

    Returns
    -------
    bool
        True if the X is a dataframe or series, False otherwise.
    """
    return is_pandas_df_or_series(X) or is_polars_df_or_series(X) or is_pyarrow_data(X)


def is_pandas_df_or_series(X):
    """Return True if the X is a pandas dataframe or series.

    Parameters
    ----------
    X : {array-like, dataframe}
        The array-like or dataframe object to check.

    Returns
    -------
    bool
        True if the X is a pandas dataframe or series, False otherwise.
    """
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(X, (pd.DataFrame, pd.Series))


def is_pandas_df(X):
    """Return True if the X is a pandas dataframe.

    Parameters
    ----------
    X : {array-like, dataframe}
        The array-like or dataframe object to check.

    Returns
    -------
    bool
        True if the X is a pandas dataframe, False otherwise.
    """
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(X, pd.DataFrame)


def is_pyarrow_data(X):
    """Return True if the X is a pyarrow Table, RecordBatch, Array or ChunkedArray.

    Parameters
    ----------
    X : {array-like, dataframe}
        The array-like or dataframe object to check.

    Returns
    -------
    bool
        True if the X is a pyarrow Table, RecordBatch, Array or ChunkedArray,
        False otherwise.
    """
    try:
        pa = sys.modules["pyarrow"]
    except KeyError:
        return False
    return isinstance(X, (pa.Table, pa.RecordBatch, pa.Array, pa.ChunkedArray))


def is_polars_df_or_series(X):
    """Return True if the X is a polars dataframe or series.

    Parameters
    ----------
    X : {array-like, dataframe}
        The array-like or dataframe object to check.

    Returns
    -------
    bool
        True if the X is a polars dataframe or series, False otherwise.
    """
    try:
        pl = sys.modules["polars"]
    except KeyError:
        return False
    return isinstance(X, (pl.DataFrame, pl.Series))


def is_polars_df(X):
    """Return True if the X is a polars dataframe.

    Parameters
    ----------
    X : {array-like, dataframe}
        The array-like or dataframe object to check.

    Returns
    -------
    bool
        True if the X is a polarsdataframe, False otherwise.
    """
    try:
        pl = sys.modules["polars"]
    except KeyError:
        return False
    return isinstance(X, pl.DataFrame)
