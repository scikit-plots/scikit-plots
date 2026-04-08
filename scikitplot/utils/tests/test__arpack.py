# scikitplot/utils/tests/test__arpack.py
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
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_arpack.py
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from numpy.testing import assert_allclose

from ..validation import check_random_state
from .._arpack import _init_arpack_v0


@pytest.mark.parametrize("seed", range(100))
def test_init_arpack_v0(seed):
    # check that the initialization a sampling from a uniform distribution
    # where we can fix the random state
    size = 1000
    v0 = _init_arpack_v0(size, seed)

    rng = check_random_state(seed)
    assert_allclose(v0, rng.uniform(-1, 1, size=size))
