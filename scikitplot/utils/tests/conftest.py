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
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/conftest.py
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import platform
import sys
from contextlib import suppress
from functools import wraps
from os import environ
from unittest import SkipTest

import joblib
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from scipy.datasets import face
from threadpoolctl import threadpool_limits


@pytest.fixture
def hide_available_pandas(monkeypatch):
    """Pretend pandas was not installed."""
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
