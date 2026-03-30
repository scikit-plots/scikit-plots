# scikitplot/tests/test_exceptions.py
#
# flake8: noqa: D213
# pylint: disable=import-outside-toplevel
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/_globals.py

"""
Comprehensive test suite for ``scikitplot.exceptions``.

Structure
---------
Each public class and module-level behaviour has its own ``Test*`` class.
Tests are ordered to mirror the source file layout:

  TestModuleIntegrity
  TestTooHardError
  TestComplexWarning
  TestRankWarning
  TestDTypePromotionError
  TestAxisError
  TestModuleDeprecationWarning
  TestVisibleDeprecationWarning
  TestScikitplotWarning
  TestScikitplotUserWarning
  TestScikitplotDeprecationWarning
  TestScikitplotPendingDeprecationWarning
  TestScikitplotBackwardsIncompatibleChangeWarning
  TestDuplicateRepresentationWarning
  TestScikitplotException
  TestScikitplotExceptionSubclasses
  TestScikitplotTracingException
  TestScikitplotTraceDataException
  TestScikitplotTraceDataNotFound
  TestScikitplotTraceDataCorrupted
  TestPrivateUnsupportedMultipartUploadException
  TestModuleGetattr

Notes
-----
**Developer notes**

* The module import uses ``importlib`` to handle both installed and
  editable/source-tree layouts without hardcoding ``sys.path``.
* Tests that exercise ``ErfaError`` / ``ErfaWarning`` mock the ``erfa``
  package to avoid an optional dependency.
* All branch-coverage targets in :class:`ScikitplotTraceDataException`
  (9 parametrize paths) are explicit.
* ``pytest.warns`` is used for warning assertions; ``pytest.raises`` is
  used for exception assertions.  No bare ``try/except`` in tests.

Running
-------
From the project root::

    pytest tests/test_exceptions.py -v --tb=short

For coverage::

    pytest tests/test_exceptions.py --cov=scikitplot.exceptions --cov-report=term-missing
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------
# Support both installed package and editable/source-tree layout.  If neither
# works, skip the entire file with a helpful message rather than letting every
# test fail with an ImportError.
try:
    from .. import exceptions as exc
except ImportError:
    pytest.skip(
        "scikitplot.exceptions is not importable. "
        "Install the package or add the project root to PYTHONPATH.",
        allow_module_level=True,
    )


# ===========================================================================
# TestModuleIntegrity
# ===========================================================================

class TestModuleIntegrity:
    """Verify the module-level contracts: __all__, reload guard."""

    def test_all_exports_exist_as_module_attributes(self):
        """Every name declared in __all__ must be reachable as a module attribute."""
        for name in exc.__all__:
            assert hasattr(exc, name), (
                f"__all__ declares {name!r} but the module has no such attribute."
            )

    def test_all_exports_are_classes_or_functions(self):
        """Every name in __all__ must resolve to a type or callable."""
        for name in exc.__all__:
            obj = getattr(exc, name)
            assert isinstance(obj, type), (
                f"{name!r} in __all__ is {type(obj)!r}, expected a class (type)."
            )

    def test_all_contains_expected_numpy_names(self):
        expected = {
            "TooHardError",
            "ComplexWarning",
            "RankWarning",
            "VisibleDeprecationWarning",
            "ModuleDeprecationWarning",
            "DTypePromotionError",
            "AxisError",
        }
        assert expected.issubset(set(exc.__all__))

    def test_all_contains_expected_astropy_names(self):
        expected = {
            "ScikitplotWarning",
            "ScikitplotUserWarning",
            "ScikitplotDeprecationWarning",
            "ScikitplotPendingDeprecationWarning",
            "ScikitplotBackwardsIncompatibleChangeWarning",
            "DuplicateRepresentationWarning",
        }
        assert expected.issubset(set(exc.__all__))

    def test_all_contains_expected_mlflow_names(self):
        expected = {
            "ScikitplotException",
            "ExecutionException",
            "MissingConfigException",
            "InvalidUrlException",
            "CommandError",
            "ScikitplotTracingException",
            "ScikitplotTraceDataException",
            "ScikitplotTraceDataNotFound",
            "ScikitplotTraceDataCorrupted",
        }
        assert expected.issubset(set(exc.__all__))

    def test_private_unsupported_multipart_not_in_all(self):
        """_UnsupportedMultipartUploadException is private and must not appear in __all__."""
        assert "_UnsupportedMultipartUploadException" not in exc.__all__

    def test_is_loaded_guard_present(self):
        """``_is_loaded`` must be set to prevent accidental reloads."""
        assert hasattr(exc, "_is_loaded"), (
            "_is_loaded not found; the reload guard is broken."
        )
        assert exc._is_loaded is True

    def test_reload_raises_runtime_error(self):
        """Reloading the module must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Reloading scikitplot.exceptions"):
            importlib.reload(exc)

    def test_no_duplicate_names_in_all(self):
        """__all__ must not contain duplicate entries."""
        assert len(exc.__all__) == len(set(exc.__all__))


# ===========================================================================
# TestTooHardError
# ===========================================================================

class TestTooHardError:

    def test_is_runtime_error(self):
        assert issubclass(exc.TooHardError, RuntimeError)

    def test_is_exception(self):
        assert issubclass(exc.TooHardError, Exception)

    def test_raise_and_catch_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise exc.TooHardError("max_work exceeded")

    def test_raise_and_catch_as_too_hard_error(self):
        with pytest.raises(exc.TooHardError):
            raise exc.TooHardError("max_work exceeded")

    def test_message_preserved(self):
        msg = "computation too hard"
        err = exc.TooHardError(msg)
        assert str(err) == msg

    def test_no_args(self):
        """TooHardError() with no message must not raise at construction."""
        err = exc.TooHardError()
        assert isinstance(err, RuntimeError)


# ===========================================================================
# TestComplexWarning
# ===========================================================================

class TestComplexWarning:

    def test_is_runtime_warning(self):
        assert issubclass(exc.ComplexWarning, RuntimeWarning)

    def test_is_warning(self):
        assert issubclass(exc.ComplexWarning, Warning)

    def test_can_be_issued(self):
        with pytest.warns(exc.ComplexWarning, match="casting"):
            warnings.warn("casting complex to real", exc.ComplexWarning, stacklevel=1)

    def test_caught_by_runtime_warning(self):
        with pytest.warns(RuntimeWarning):
            warnings.warn("test", exc.ComplexWarning, stacklevel=1)


# ===========================================================================
# TestRankWarning
# ===========================================================================

class TestRankWarning:

    def test_is_runtime_warning(self):
        assert issubclass(exc.RankWarning, RuntimeWarning)

    def test_is_in_all(self):
        assert "RankWarning" in exc.__all__

    def test_can_be_issued(self):
        with pytest.warns(exc.RankWarning):
            warnings.warn("rank deficient", exc.RankWarning, stacklevel=1)


# ===========================================================================
# TestDTypePromotionError
# ===========================================================================

class TestDTypePromotionError:

    def test_is_type_error(self):
        assert issubclass(exc.DTypePromotionError, TypeError)

    def test_raise_and_catch_as_type_error(self):
        with pytest.raises(TypeError):
            raise exc.DTypePromotionError("cannot promote")

    def test_raise_and_catch_as_dtype_promotion_error(self):
        with pytest.raises(exc.DTypePromotionError):
            raise exc.DTypePromotionError("cannot promote")

    def test_message_preserved(self):
        msg = "incompatible dtypes"
        err = exc.DTypePromotionError(msg)
        assert str(err) == msg


# ===========================================================================
# TestAxisError
# ===========================================================================

class TestAxisError:

    # --- inheritance ---

    def test_is_value_error(self):
        assert issubclass(exc.AxisError, ValueError)

    def test_is_index_error(self):
        assert issubclass(exc.AxisError, IndexError)

    def test_both_bases_catchable(self):
        err = exc.AxisError(1, 1)
        assert isinstance(err, ValueError)
        assert isinstance(err, IndexError)

    # --- single-argument (string message) form ---

    def test_single_arg_str_message(self):
        msg = "Custom error message"
        err = exc.AxisError(msg)
        assert str(err) == msg

    def test_single_arg_axis_is_none(self):
        err = exc.AxisError("Custom")
        assert err.axis is None

    def test_single_arg_ndim_is_none(self):
        err = exc.AxisError("Custom")
        assert err.ndim is None

    # --- structured (axis + ndim) form ---

    def test_structured_basic_message(self):
        err = exc.AxisError(1, 2)
        assert str(err) == "axis 1 is out of bounds for array of dimension 2"

    def test_structured_axis_attribute(self):
        err = exc.AxisError(3, 5)
        assert err.axis == 3

    def test_structured_ndim_attribute(self):
        err = exc.AxisError(3, 5)
        assert err.ndim == 5

    # --- structured form with msg_prefix ---

    def test_structured_with_msg_prefix(self):
        err = exc.AxisError(2, 1, msg_prefix="error")
        assert str(err) == "error: axis 2 is out of bounds for array of dimension 1"

    def test_structured_msg_prefix_none_no_colon(self):
        err = exc.AxisError(0, 3)
        result = str(err)
        assert ":" not in result

    # --- negative axis ---

    def test_negative_axis_preserved_in_message(self):
        err = exc.AxisError(-2, 1)
        assert "axis -2" in str(err)

    # --- raise and catch ---

    def test_raise_and_catch_as_value_error(self):
        with pytest.raises(ValueError):
            raise exc.AxisError(99, 2)

    def test_raise_and_catch_as_index_error(self):
        with pytest.raises(IndexError):
            raise exc.AxisError(99, 2)

    def test_raise_and_catch_as_axis_error(self):
        with pytest.raises(exc.AxisError):
            raise exc.AxisError(99, 2)

    # --- slots ---

    def test_slots_defined(self):
        """AxisError.__slots__ must be declared on the class itself."""
        assert hasattr(exc.AxisError, "__slots__")

    def test_slots_contains_expected_names(self):
        """__slots__ must declare exactly the three expected attribute names."""
        assert set(exc.AxisError.__slots__) == {"_msg", "axis", "ndim"}

    def test_slots_attributes_are_accessible(self):
        """Slot attributes must be readable and writable on instances.

        Notes
        -----
        ``AxisError`` inherits from ``ValueError`` and ``IndexError``, both of
        which are C-level built-ins that expose ``__dict__``.  Because at least
        one base class in the MRO does not define ``__slots__``, Python adds a
        ``__dict__`` to every ``AxisError`` instance even though ``AxisError``
        itself declares ``__slots__``.  This is correct Python semantics — the
        ``__slots__`` declaration still protects the *named* slots from
        accidental renaming or deletion, it just cannot suppress ``__dict__``
        when an ancestor already provides one.  The test below verifies the
        slot contract that *can* be enforced.
        """
        err = exc.AxisError(2, 5, msg_prefix="pfx")
        # Each slot attribute holds the value set by __init__.
        assert err.axis == 2
        assert err.ndim == 5
        assert err._msg == "pfx"


# ===========================================================================
# TestModuleDeprecationWarning
# ===========================================================================

class TestModuleDeprecationWarning:

    def test_is_deprecation_warning(self):
        assert issubclass(exc.ModuleDeprecationWarning, DeprecationWarning)

    def test_is_warning(self):
        assert issubclass(exc.ModuleDeprecationWarning, Warning)

    def test_can_be_issued(self):
        with pytest.warns(exc.ModuleDeprecationWarning):
            warnings.warn("module deprecated", exc.ModuleDeprecationWarning, stacklevel=1)


# ===========================================================================
# TestVisibleDeprecationWarning
# ===========================================================================

class TestVisibleDeprecationWarning:

    def test_is_user_warning(self):
        assert issubclass(exc.VisibleDeprecationWarning, UserWarning)

    def test_is_warning(self):
        assert issubclass(exc.VisibleDeprecationWarning, Warning)

    def test_can_be_issued(self):
        with pytest.warns(exc.VisibleDeprecationWarning):
            warnings.warn("visible deprecation", exc.VisibleDeprecationWarning, stacklevel=1)


# ===========================================================================
# TestScikitplotWarning
# ===========================================================================

class TestScikitplotWarning:

    def test_is_warning(self):
        assert issubclass(exc.ScikitplotWarning, Warning)

    def test_not_user_warning(self):
        """ScikitplotWarning inherits Warning directly, not UserWarning."""
        assert not issubclass(exc.ScikitplotWarning, UserWarning)

    def test_can_be_issued(self):
        with pytest.warns(exc.ScikitplotWarning):
            warnings.warn("skplot warning", exc.ScikitplotWarning, stacklevel=1)


# ===========================================================================
# TestScikitplotUserWarning
# ===========================================================================

class TestScikitplotUserWarning:

    def test_is_user_warning(self):
        assert issubclass(exc.ScikitplotUserWarning, UserWarning)

    def test_not_scikitplot_warning(self):
        """ScikitplotUserWarning inherits UserWarning, not ScikitplotWarning."""
        assert not issubclass(exc.ScikitplotUserWarning, exc.ScikitplotWarning)

    def test_can_be_issued(self):
        with pytest.warns(exc.ScikitplotUserWarning):
            warnings.warn("user warning", exc.ScikitplotUserWarning, stacklevel=1)


# ===========================================================================
# TestScikitplotDeprecationWarning
# ===========================================================================

class TestScikitplotDeprecationWarning:

    def test_is_deprecation_warning(self):
        assert issubclass(exc.ScikitplotDeprecationWarning, DeprecationWarning)

    def test_is_warning(self):
        assert issubclass(exc.ScikitplotDeprecationWarning, Warning)

    def test_can_be_issued(self):
        with pytest.warns(exc.ScikitplotDeprecationWarning):
            warnings.warn(
                "deprecated feature",
                exc.ScikitplotDeprecationWarning,
                stacklevel=1,
            )


# ===========================================================================
# TestScikitplotPendingDeprecationWarning
# ===========================================================================

class TestScikitplotPendingDeprecationWarning:

    def test_is_pending_deprecation_warning(self):
        assert issubclass(
            exc.ScikitplotPendingDeprecationWarning, PendingDeprecationWarning
        )

    def test_is_warning(self):
        assert issubclass(exc.ScikitplotPendingDeprecationWarning, Warning)


# ===========================================================================
# TestScikitplotBackwardsIncompatibleChangeWarning
# ===========================================================================

class TestScikitplotBackwardsIncompatibleChangeWarning:

    def test_is_scikitplot_warning(self):
        assert issubclass(
            exc.ScikitplotBackwardsIncompatibleChangeWarning, exc.ScikitplotWarning
        )

    def test_is_warning(self):
        assert issubclass(
            exc.ScikitplotBackwardsIncompatibleChangeWarning, Warning
        )

    def test_can_be_issued(self):
        with pytest.warns(exc.ScikitplotBackwardsIncompatibleChangeWarning):
            warnings.warn(
                "backwards incompatible change",
                exc.ScikitplotBackwardsIncompatibleChangeWarning,
                stacklevel=1,
            )


# ===========================================================================
# TestDuplicateRepresentationWarning
# ===========================================================================

class TestDuplicateRepresentationWarning:

    def test_is_scikitplot_warning(self):
        assert issubclass(
            exc.DuplicateRepresentationWarning, exc.ScikitplotWarning
        )

    def test_can_be_issued(self):
        with pytest.warns(exc.DuplicateRepresentationWarning):
            warnings.warn(
                "duplicate representation",
                exc.DuplicateRepresentationWarning,
                stacklevel=1,
            )


# ===========================================================================
# TestScikitplotException
# ===========================================================================

class TestScikitplotException:

    # --- construction ---

    def test_is_exception(self):
        assert issubclass(exc.ScikitplotException, Exception)

    def test_default_error_code_is_zero(self):
        err = exc.ScikitplotException("oops")
        assert err.error_code == 0

    def test_custom_error_code_int(self):
        err = exc.ScikitplotException("oops", error_code=42)
        assert err.error_code == 42

    def test_custom_error_code_string(self):
        """error_code may be a string sentinel (used by trace-data subclasses)."""
        err = exc.ScikitplotException("oops", error_code="NOT_FOUND")
        assert err.error_code == "NOT_FOUND"

    def test_message_converted_to_str(self):
        """message is always str even when an Exception is passed."""
        inner = ValueError("inner error")
        err = exc.ScikitplotException(inner)
        assert err.message == str(inner)
        assert isinstance(err.message, str)

    def test_message_plain_string(self):
        err = exc.ScikitplotException("plain message")
        assert err.message == "plain message"

    def test_str_representation(self):
        err = exc.ScikitplotException("boom")
        assert str(err) == "boom"

    def test_json_kwargs_stored(self):
        err = exc.ScikitplotException("err", key1="val1", key2=99)
        assert err.json_kwargs == {"key1": "val1", "key2": 99}

    def test_json_kwargs_empty_by_default(self):
        err = exc.ScikitplotException("err")
        assert err.json_kwargs == {}

    # --- raise and catch ---

    def test_raise_and_catch_as_exception(self):
        with pytest.raises(Exception):
            raise exc.ScikitplotException("test")

    def test_raise_and_catch_as_scikitplot_exception(self):
        with pytest.raises(exc.ScikitplotException):
            raise exc.ScikitplotException("test")

    # --- serialize_as_json ---

    def test_serialize_as_json_returns_string(self):
        err = exc.ScikitplotException("msg")
        result = err.serialize_as_json()
        assert isinstance(result, str)

    def test_serialize_as_json_is_valid_json(self):
        err = exc.ScikitplotException("msg", error_code=1, extra="data")
        result = err.serialize_as_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_serialize_as_json_contains_error_code(self):
        err = exc.ScikitplotException("msg", error_code=7)
        parsed = json.loads(err.serialize_as_json())
        assert parsed["error_code"] == 7

    def test_serialize_as_json_contains_message(self):
        err = exc.ScikitplotException("hello world")
        parsed = json.loads(err.serialize_as_json())
        assert parsed["message"] == "hello world"

    def test_serialize_as_json_includes_extra_kwargs(self):
        err = exc.ScikitplotException("msg", alpha="A", beta=2)
        parsed = json.loads(err.serialize_as_json())
        assert parsed["alpha"] == "A"
        assert parsed["beta"] == 2

    def test_serialize_as_json_non_serializable_kwargs_do_not_raise(self):
        """Non-JSON-serialisable kwargs must be coerced to str, not raise."""
        class Unserializable:
            def __repr__(self):
                return "<Unserializable>"
        err = exc.ScikitplotException("msg", bad=Unserializable())
        result = err.serialize_as_json()
        parsed = json.loads(result)
        assert "bad" in parsed
        assert isinstance(parsed["bad"], str)

    def test_serialize_as_json_string_error_code_round_trips(self):
        err = exc.ScikitplotException("msg", error_code="NOT_FOUND")
        parsed = json.loads(err.serialize_as_json())
        assert parsed["error_code"] == "NOT_FOUND"

    # --- get_http_status_code ---

    def test_get_http_status_code_is_500(self):
        err = exc.ScikitplotException("err")
        assert err.get_http_status_code() == 500

    # --- invalid_parameter_value classmethod ---

    def test_invalid_parameter_value_returns_instance(self):
        result = exc.ScikitplotException.invalid_parameter_value("bad value")
        assert isinstance(result, exc.ScikitplotException)

    def test_invalid_parameter_value_error_code_is_zero(self):
        result = exc.ScikitplotException.invalid_parameter_value("bad value")
        assert result.error_code == 0

    def test_invalid_parameter_value_message_preserved(self):
        result = exc.ScikitplotException.invalid_parameter_value("bad value")
        assert result.message == "bad value"

    def test_invalid_parameter_value_extra_kwargs(self):
        result = exc.ScikitplotException.invalid_parameter_value("bad", key="v")
        assert result.json_kwargs == {"key": "v"}

    def test_invalid_parameter_value_subclass_returns_subclass(self):
        """When called on a subclass, the classmethod must return that subclass."""
        result = exc.ExecutionException.invalid_parameter_value("fail")
        assert isinstance(result, exc.ExecutionException)


# ===========================================================================
# TestScikitplotExceptionSubclasses  (CommandError, Execution, Missing, Invalid)
# ===========================================================================

class TestScikitplotExceptionSubclasses:

    @pytest.mark.parametrize("cls_name", [
        "CommandError",
        "ExecutionException",
        "MissingConfigException",
        "InvalidUrlException",
    ])
    def test_is_scikitplot_exception(self, cls_name):
        cls = getattr(exc, cls_name)
        assert issubclass(cls, exc.ScikitplotException)

    @pytest.mark.parametrize("cls_name", [
        "CommandError",
        "ExecutionException",
        "MissingConfigException",
        "InvalidUrlException",
    ])
    def test_is_exception(self, cls_name):
        cls = getattr(exc, cls_name)
        assert issubclass(cls, Exception)

    @pytest.mark.parametrize("cls_name", [
        "CommandError",
        "ExecutionException",
        "MissingConfigException",
        "InvalidUrlException",
    ])
    def test_message_forwarded(self, cls_name):
        cls = getattr(exc, cls_name)
        err = cls("specific error message")
        assert err.message == "specific error message"
        assert str(err) == "specific error message"

    @pytest.mark.parametrize("cls_name", [
        "CommandError",
        "ExecutionException",
        "MissingConfigException",
        "InvalidUrlException",
    ])
    def test_raise_and_catch_as_scikitplot_exception(self, cls_name):
        cls = getattr(exc, cls_name)
        with pytest.raises(exc.ScikitplotException):
            raise cls("err")

    @pytest.mark.parametrize("cls_name", [
        "CommandError",
        "ExecutionException",
        "MissingConfigException",
        "InvalidUrlException",
    ])
    def test_serialize_as_json_valid(self, cls_name):
        cls = getattr(exc, cls_name)
        err = cls("msg")
        parsed = json.loads(err.serialize_as_json())
        assert parsed["message"] == "msg"


# ===========================================================================
# TestScikitplotTracingException
# ===========================================================================

class TestScikitplotTracingException:

    def test_is_scikitplot_exception(self):
        assert issubclass(exc.ScikitplotTracingException, exc.ScikitplotException)

    def test_is_exception(self):
        assert issubclass(exc.ScikitplotTracingException, Exception)

    def test_default_error_code_is_zero(self):
        err = exc.ScikitplotTracingException("tracing failed")
        assert err.error_code == 0

    def test_custom_error_code(self):
        err = exc.ScikitplotTracingException("tracing failed", error_code=99)
        assert err.error_code == 99

    def test_message_forwarded(self):
        err = exc.ScikitplotTracingException("trace msg")
        assert err.message == "trace msg"

    def test_raise_and_catch_as_scikitplot_exception(self):
        with pytest.raises(exc.ScikitplotException):
            raise exc.ScikitplotTracingException("err")

    def test_raise_and_catch_as_tracing_exception(self):
        with pytest.raises(exc.ScikitplotTracingException):
            raise exc.ScikitplotTracingException("err")


# ===========================================================================
# TestScikitplotTraceDataException
# ===========================================================================

class TestScikitplotTraceDataException:
    """
    Covers all branches of :class:`ScikitplotTraceDataException.__init__`.

    Branch matrix (3 error_code × 3 context combos = 9 paths):

    error_code       | request_id | artifact_path | ctx           | message prefix
    -----------------+------------+---------------+---------------+------------------
    NOT_FOUND        | set        | -             | request_id=X  | not found for
    NOT_FOUND        | None       | set           | path=Y        | not found for
    NOT_FOUND        | None       | None          | unknown       | not found for
    INVALID_STATE    | set        | -             | request_id=X  | corrupted for
    INVALID_STATE    | None       | set           | path=Y        | corrupted for
    INVALID_STATE    | None       | None          | unknown       | corrupted for
    <other>          | set        | -             | request_id=X  | error (...) for
    <other>          | None       | set           | path=Y        | error (...) for
    <other>          | None       | None          | unknown       | error (...) for
    """

    def test_is_tracing_exception(self):
        assert issubclass(exc.ScikitplotTraceDataException, exc.ScikitplotTracingException)

    def test_is_scikitplot_exception(self):
        assert issubclass(exc.ScikitplotTraceDataException, exc.ScikitplotException)

    # --- NOT_FOUND branch ---

    def test_not_found_with_request_id_message(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND", request_id="req-123")
        assert "not found" in err.message.lower()
        assert "req-123" in err.message

    def test_not_found_with_request_id_ctx(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND", request_id="req-123")
        assert err.ctx == "request_id=req-123"

    def test_not_found_with_request_id_error_code(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND", request_id="req-123")
        assert err.error_code == "NOT_FOUND"

    def test_not_found_with_artifact_path_message(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND", artifact_path="/data/trace")
        assert "not found" in err.message.lower()
        assert "/data/trace" in err.message

    def test_not_found_with_artifact_path_ctx(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND", artifact_path="/data/trace")
        assert err.ctx == "path=/data/trace"

    def test_not_found_no_context_ctx_is_unknown(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND")
        assert err.ctx == "unknown"

    def test_not_found_no_context_message_contains_unknown(self):
        err = exc.ScikitplotTraceDataException("NOT_FOUND")
        assert "unknown" in err.message

    # --- INVALID_STATE branch ---

    def test_invalid_state_with_request_id_message(self):
        err = exc.ScikitplotTraceDataException("INVALID_STATE", request_id="req-456")
        assert "corrupted" in err.message.lower()
        assert "req-456" in err.message

    def test_invalid_state_with_request_id_ctx(self):
        err = exc.ScikitplotTraceDataException("INVALID_STATE", request_id="req-456")
        assert err.ctx == "request_id=req-456"

    def test_invalid_state_with_artifact_path_message(self):
        err = exc.ScikitplotTraceDataException(
            "INVALID_STATE", artifact_path="/corrupt/file"
        )
        assert "corrupted" in err.message.lower()
        assert "/corrupt/file" in err.message

    def test_invalid_state_with_artifact_path_ctx(self):
        err = exc.ScikitplotTraceDataException(
            "INVALID_STATE", artifact_path="/corrupt/file"
        )
        assert err.ctx == "path=/corrupt/file"

    def test_invalid_state_no_context_ctx_is_unknown(self):
        err = exc.ScikitplotTraceDataException("INVALID_STATE")
        assert err.ctx == "unknown"

    def test_invalid_state_no_context_message_contains_unknown(self):
        err = exc.ScikitplotTraceDataException("INVALID_STATE")
        assert "unknown" in err.message

    # --- generic error_code branch ---

    def test_generic_error_code_message_contains_code(self):
        err = exc.ScikitplotTraceDataException("CUSTOM_CODE", request_id="r1")
        assert "CUSTOM_CODE" in err.message

    def test_generic_error_code_message_contains_ctx(self):
        err = exc.ScikitplotTraceDataException("CUSTOM_CODE", request_id="r1")
        assert "r1" in err.message

    def test_generic_error_code_error_code_stored(self):
        err = exc.ScikitplotTraceDataException("CUSTOM_CODE")
        assert err.error_code == "CUSTOM_CODE"

    def test_generic_no_context_ctx_is_unknown(self):
        err = exc.ScikitplotTraceDataException("OTHER")
        assert err.ctx == "unknown"

    # --- request_id priority over artifact_path ---

    def test_request_id_takes_priority_over_artifact_path(self):
        """When both are supplied, request_id must win."""
        err = exc.ScikitplotTraceDataException(
            "NOT_FOUND",
            request_id="req-789",
            artifact_path="/some/path",
        )
        assert err.ctx == "request_id=req-789"
        assert "req-789" in err.message
        # artifact_path must NOT appear in the message
        assert "/some/path" not in err.message

    # --- ctx attribute is always initialised (Fix 1) ---

    def test_ctx_attribute_always_set(self):
        """ctx must never raise AttributeError regardless of arguments."""
        for args, kwargs in [
            (("NOT_FOUND",), {}),
            (("INVALID_STATE",), {"request_id": "r"}),
            (("INVALID_STATE",), {"artifact_path": "/p"}),
            (("INVALID_STATE",), {}),
        ]:
            err = exc.ScikitplotTraceDataException(*args, **kwargs)
            assert hasattr(err, "ctx"), "ctx attribute missing"
            assert isinstance(err.ctx, str)

    # --- raise and catch ---

    def test_raise_and_catch_as_scikitplot_exception(self):
        with pytest.raises(exc.ScikitplotException):
            raise exc.ScikitplotTraceDataException("NOT_FOUND")

    def test_raise_and_catch_as_tracing_exception(self):
        with pytest.raises(exc.ScikitplotTracingException):
            raise exc.ScikitplotTraceDataException("INVALID_STATE")

    def test_raise_and_catch_as_trace_data_exception(self):
        with pytest.raises(exc.ScikitplotTraceDataException):
            raise exc.ScikitplotTraceDataException("NOT_FOUND")


# ===========================================================================
# TestScikitplotTraceDataNotFound
# ===========================================================================

class TestScikitplotTraceDataNotFound:

    def test_is_trace_data_exception(self):
        assert issubclass(exc.ScikitplotTraceDataNotFound, exc.ScikitplotTraceDataException)

    def test_is_tracing_exception(self):
        assert issubclass(exc.ScikitplotTraceDataNotFound, exc.ScikitplotTracingException)

    def test_error_code_is_not_found(self):
        err = exc.ScikitplotTraceDataNotFound()
        assert err.error_code == "NOT_FOUND"

    def test_no_args_ctx_is_unknown(self):
        err = exc.ScikitplotTraceDataNotFound()
        assert err.ctx == "unknown"

    def test_no_args_message_contains_unknown(self):
        err = exc.ScikitplotTraceDataNotFound()
        assert "unknown" in err.message

    def test_with_request_id(self):
        err = exc.ScikitplotTraceDataNotFound(request_id="req-abc")
        assert err.ctx == "request_id=req-abc"
        assert "req-abc" in err.message

    def test_with_artifact_path(self):
        err = exc.ScikitplotTraceDataNotFound(artifact_path="/tmp/trace")
        assert err.ctx == "path=/tmp/trace"
        assert "/tmp/trace" in err.message

    def test_request_id_over_artifact_path(self):
        err = exc.ScikitplotTraceDataNotFound(
            request_id="req-abc", artifact_path="/tmp/trace"
        )
        assert err.ctx == "request_id=req-abc"

    def test_message_indicates_not_found(self):
        err = exc.ScikitplotTraceDataNotFound(request_id="x")
        assert "not found" in err.message.lower()

    def test_raise_and_catch_as_trace_data_exception(self):
        with pytest.raises(exc.ScikitplotTraceDataException):
            raise exc.ScikitplotTraceDataNotFound(request_id="req")

    def test_raise_and_catch_as_scikitplot_exception(self):
        with pytest.raises(exc.ScikitplotException):
            raise exc.ScikitplotTraceDataNotFound()


# ===========================================================================
# TestScikitplotTraceDataCorrupted
# ===========================================================================

class TestScikitplotTraceDataCorrupted:

    def test_is_trace_data_exception(self):
        assert issubclass(exc.ScikitplotTraceDataCorrupted, exc.ScikitplotTraceDataException)

    def test_is_tracing_exception(self):
        assert issubclass(exc.ScikitplotTraceDataCorrupted, exc.ScikitplotTracingException)

    def test_error_code_is_invalid_state(self):
        err = exc.ScikitplotTraceDataCorrupted()
        assert err.error_code == "INVALID_STATE"

    def test_no_args_ctx_is_unknown(self):
        err = exc.ScikitplotTraceDataCorrupted()
        assert err.ctx == "unknown"

    def test_no_args_message_contains_unknown(self):
        err = exc.ScikitplotTraceDataCorrupted()
        assert "unknown" in err.message

    def test_with_request_id(self):
        err = exc.ScikitplotTraceDataCorrupted(request_id="req-xyz")
        assert err.ctx == "request_id=req-xyz"
        assert "req-xyz" in err.message

    def test_with_artifact_path(self):
        err = exc.ScikitplotTraceDataCorrupted(artifact_path="/data/corrupted")
        assert err.ctx == "path=/data/corrupted"
        assert "/data/corrupted" in err.message

    def test_request_id_over_artifact_path(self):
        err = exc.ScikitplotTraceDataCorrupted(
            request_id="req-xyz", artifact_path="/data/corrupted"
        )
        assert err.ctx == "request_id=req-xyz"

    def test_message_indicates_corrupted(self):
        err = exc.ScikitplotTraceDataCorrupted(request_id="x")
        assert "corrupted" in err.message.lower()

    def test_raise_and_catch_as_trace_data_exception(self):
        with pytest.raises(exc.ScikitplotTraceDataException):
            raise exc.ScikitplotTraceDataCorrupted(request_id="req")

    def test_raise_and_catch_as_scikitplot_exception(self):
        with pytest.raises(exc.ScikitplotException):
            raise exc.ScikitplotTraceDataCorrupted()


# ===========================================================================
# TestPrivateUnsupportedMultipartUploadException
# ===========================================================================

class TestPrivateUnsupportedMultipartUploadException:
    """The private class is still tested; it just isn't public API."""

    def test_exists_as_module_attribute(self):
        assert hasattr(exc, "_UnsupportedMultipartUploadException")

    def test_is_scikitplot_exception(self):
        cls = exc._UnsupportedMultipartUploadException
        assert issubclass(cls, exc.ScikitplotException)

    def test_message_is_fixed(self):
        err = exc._UnsupportedMultipartUploadException()
        assert "Multipart upload is not supported" in err.message

    def test_error_code_is_zero(self):
        err = exc._UnsupportedMultipartUploadException()
        assert err.error_code == 0

    def test_takes_no_arguments(self):
        """Must be constructable with zero arguments."""
        err = exc._UnsupportedMultipartUploadException()
        assert isinstance(err, exc.ScikitplotException)

    def test_class_attribute_message(self):
        cls = exc._UnsupportedMultipartUploadException
        assert isinstance(cls.MESSAGE, str)
        assert len(cls.MESSAGE) > 0


# ===========================================================================
# TestModuleGetattr
# ===========================================================================

class TestModuleGetattr:
    """Module-level __getattr__ handles ErfaError/ErfaWarning and rejects unknowns."""

    def _make_erfa_mock(self):
        """Return a minimal mock of the ``erfa`` module."""
        erfa_mock = types.ModuleType("erfa")
        erfa_mock.ErfaError = type("ErfaError", (Exception,), {})
        erfa_mock.ErfaWarning = type("ErfaWarning", (Warning,), {})
        return erfa_mock

    def test_erfa_error_issues_deprecation_warning(self):
        erfa_mock = self._make_erfa_mock()
        with patch.dict(sys.modules, {"erfa": erfa_mock}):
            with pytest.warns(exc.ScikitplotDeprecationWarning):
                _ = exc.__getattr__("ErfaError")

    def test_erfa_warning_issues_deprecation_warning(self):
        erfa_mock = self._make_erfa_mock()
        with patch.dict(sys.modules, {"erfa": erfa_mock}):
            with pytest.warns(exc.ScikitplotDeprecationWarning):
                _ = exc.__getattr__("ErfaWarning")

    def test_erfa_error_warning_message_mentions_version(self):
        erfa_mock = self._make_erfa_mock()
        with patch.dict(sys.modules, {"erfa": erfa_mock}):
            with pytest.warns(exc.ScikitplotDeprecationWarning, match="0.4"):
                _ = exc.__getattr__("ErfaError")

    def test_erfa_error_warning_message_mentions_name(self):
        erfa_mock = self._make_erfa_mock()
        with patch.dict(sys.modules, {"erfa": erfa_mock}):
            with pytest.warns(exc.ScikitplotDeprecationWarning, match="ErfaError"):
                _ = exc.__getattr__("ErfaError")

    def test_erfa_error_returns_erfa_class(self):
        erfa_mock = self._make_erfa_mock()
        with patch.dict(sys.modules, {"erfa": erfa_mock}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = exc.__getattr__("ErfaError")
        assert result is erfa_mock.ErfaError

    def test_erfa_warning_returns_erfa_class(self):
        erfa_mock = self._make_erfa_mock()
        with patch.dict(sys.modules, {"erfa": erfa_mock}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = exc.__getattr__("ErfaWarning")
        assert result is erfa_mock.ErfaWarning

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            exc.__getattr__("NonExistentName")

    def test_unknown_name_error_mentions_module(self):
        with pytest.raises(AttributeError, match=r"scikitplot\.exceptions"):
            exc.__getattr__("SomethingMissing")

    def test_unknown_name_error_mentions_attribute_name(self):
        with pytest.raises(AttributeError, match="SomethingMissing"):
            exc.__getattr__("SomethingMissing")
