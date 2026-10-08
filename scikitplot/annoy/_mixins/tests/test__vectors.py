# scikitplot/annoy/_mixins/tests/test__vectors.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of ``scikitplot/annoy/_mixins/_vectors.py``: scikit-learn compatibility.

Query behaviour of :class:`~scikitplot.annoy._mixins._vectors.VectorOpsMixin`
is covered in ``test_mixins.py``. These tests cover the compatibility block at
the top of the module: the project supports scikit-learn from 1.3 on, and two
names the module uses (``validate_data`` and the ``ensure_all_finite``
parameter of ``check_array``) exist from 1.6 on only.

The module must behave the same whichever of its three import branches was
taken. The branch is decided by what is installed, so each test states the
contract for every branch and forces the one that the installed scikit-learn
does not take.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.utils.validation import check_array

from .. import _vectors


class _Estimator:
    """An object scikit-learn knows nothing about, like a bare index."""

    f = 3


QUERY = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


class TestFiniteKeyword:
    """The finiteness policy is passed under the name the installed release takes."""

    def test_it_is_a_parameter_of_the_installed_check_array(self):
        assert _vectors._FINITE_KEYWORD in inspect.signature(check_array).parameters

    def test_it_is_one_of_the_two_spellings(self):
        assert _vectors._FINITE_KEYWORD in ("ensure_all_finite", "force_all_finite")

    def test_the_new_spelling_wins_where_both_exist(self):
        # scikit-learn 1.6 and 1.7 accept both and deprecate the old one.
        if "ensure_all_finite" in inspect.signature(check_array).parameters:
            assert _vectors._FINITE_KEYWORD == "ensure_all_finite"


# scikit-learn 1.6 and 1.7 warn, and 1.8 raises, when ``validate_data`` is given
# an object without ``__sklearn_tags__``. ``_Estimator`` is such an object on
# purpose: the module must fall back to ``check_array`` for it either way.
@pytest.mark.filterwarnings(
    "ignore:The following error was raised:DeprecationWarning"
)
class TestValidateDataSource:
    """The branch in use is recorded, so it can be logged and asked for."""

    def test_names_the_branch_that_was_taken(self):
        if _vectors.validate_data is None:
            expected = {"sklearn.utils.validation.check_array"}
        else:
            expected = {
                "sklearn.utils.validation.validate_data",
                "scikitplot.utils.validation.validate_data",
            }
        assert _vectors._VALIDATE_DATA_SOURCE in expected

    def test_the_public_function_is_preferred_when_scikit_learn_has_it(self):
        import sklearn.utils.validation as validation

        if hasattr(validation, "validate_data"):
            assert _vectors.validate_data is validation.validate_data
            assert _vectors._VALIDATE_DATA_SOURCE == (
                "sklearn.utils.validation.validate_data"
            )


class TestValidateQueryMatrix:
    """Validation gives the same result with and without ``validate_data``."""

    @pytest.fixture(params=["as imported", "without validate_data"])
    def module(self, request, monkeypatch):
        if request.param == "without validate_data":
            # Branch 3: scikit-learn < 1.6 and no ``scikitplot.utils``.
            monkeypatch.setattr(_vectors, "validate_data", None)
        return _vectors

    def test_a_valid_matrix_becomes_contiguous_float32(self, module):
        out = module._validate_query_matrix(
            _Estimator(), QUERY, ensure_all_finite=True, copy=False
        )
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, np.asarray(QUERY, dtype=np.float32))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_are_rejected(self, module, bad):
        with pytest.raises(ValueError, match="NaN|infinity|inf"):
            module._validate_query_matrix(
                _Estimator(), [[1.0, bad, 3.0]], ensure_all_finite=True, copy=False
            )

    def test_allow_nan_accepts_nan_and_still_rejects_infinity(self, module):
        out = module._validate_query_matrix(
            _Estimator(), [[1.0, np.nan, 3.0]], ensure_all_finite="allow-nan", copy=False
        )
        assert np.isnan(out[0, 1])
        with pytest.raises(ValueError, match="infinity|inf"):
            module._validate_query_matrix(
                _Estimator(),
                [[1.0, np.inf, 3.0]],
                ensure_all_finite="allow-nan",
                copy=False,
            )

    def test_the_policy_can_be_switched_off(self, module):
        out = module._validate_query_matrix(
            _Estimator(), [[1.0, np.inf, 3.0]], ensure_all_finite=False, copy=False
        )
        assert np.isinf(out[0, 1])

    def test_a_sparse_matrix_is_rejected(self, module):
        sparse = pytest.importorskip("scipy.sparse")
        with pytest.raises(TypeError):
            module._validate_query_matrix(
                _Estimator(),
                sparse.csr_matrix(np.asarray(QUERY)),
                ensure_all_finite=True,
                copy=False,
            )

    def test_copy_returns_an_array_that_does_not_share_memory(self, module):
        source = np.asarray(QUERY, dtype=np.float32)
        out = module._validate_query_matrix(
            _Estimator(), source, ensure_all_finite=True, copy=True
        )
        assert not np.shares_memory(out, source)


class TestFallbackWhenValidateDataRaises:
    """``check_array`` decides; a ``validate_data`` that raises does not."""

    def test_check_array_is_used(self, monkeypatch):
        calls = []

        def raising(*args, **kwargs):
            calls.append(kwargs)
            raise TypeError("not an estimator scikit-learn recognises")

        monkeypatch.setattr(_vectors, "validate_data", raising)
        out = _vectors._validate_query_matrix(
            _Estimator(), QUERY, ensure_all_finite=True, copy=False
        )
        assert out.shape == (2, 3)
        # ``validate_data`` is always given the 1.6 spelling of the keyword.
        assert calls == [
            {
                "reset": False,
                "ensure_all_finite": True,
                "accept_sparse": False,
                "dtype": _vectors.FLOAT_DTYPES,
                "copy": False,
            }
        ]
