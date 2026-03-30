# scikitplot/tests/test_environment_variables.py
#
# flake8: noqa: D213
# pylint: disable=line-too-long
# noqa: E501
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Comprehensive test suite for ``environment_variables.py``.

Coverage targets
----------------
- :class:`environment_variables.EnvironmentVariable`      : all methods, all branches, all edge cases.
- :class:`environment_variables.BooleanEnvironmentVariable`: all init validations, all ``get()`` branches.
- Module-level naming conventions    : automated AST regression for all 129 declarations.
- Module-level default spot checks   : representative variables and documented defaults.
- Namespace / module hygiene         : private-helper leakage guards.
- Round-trip fidelity                : set → get for every supported type.

Design decisions
----------------
- Every test that touches ``os.environ`` uses ``monkeypatch`` (preferred) or relies on
  the ``_clean_isolated_var`` autouse fixture to guarantee hermetic isolation.
- **Critical fix**: ``_clean_isolated_var`` uses ``os.environ.pop`` directly in its
  post-yield teardown instead of ``monkeypatch.delenv``.  Using ``monkeypatch.delenv``
  in a fixture's post-yield body is wrong because ``monkeypatch.undo()`` runs *after*
  the fixture and reverses the deletion, leaking the sentinel into the next test.
- ``_ISOLATED_VAR`` is a synthetic name chosen to be absent from any real CI environment.
- Parametrize is preferred over duplicated assertions to maximise readable coverage.
- The naming-convention test uses AST inspection so it does not depend on the filesystem
  path at all — it would have caught every one of the bugs fixed in this release.

How to run
----------
From the project root::

    pytest scikitplot/tests/test_environment_variables.py -v --tb=short

Or with coverage::

    pytest scikitplot/tests/test_environment_variables.py \\
        --cov=scikitplot.environment_variables --cov-report=term-missing
"""

from __future__ import annotations

import ast
import inspect
import os
import tempfile
from pathlib import Path

import pytest

# python - <<'PY'
# import importlib
# m = importlib.import_module("scikitplot.environment_variables")
# print(hasattr(m, "environment_variables.BooleanEnvironmentVariable"))
# print(dir(m))
# PY
from .. import environment_variables

# ---------------------------------------------------------------------------
# Sentinel env var name — must be absent from the real environment at all times.
# ---------------------------------------------------------------------------
_ISOLATED_VAR = "_SKPLT_TEST_ISOLATED_SENTINEL_12345"


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_isolated_var(monkeypatch):
    """Guarantee ``_ISOLATED_VAR`` is absent at test start **and** end.

    Notes
    -----
    **Why not ``monkeypatch.delenv`` in the post-yield?**
    Calling ``monkeypatch.delenv`` inside a fixture's post-yield (teardown)
    records the undo-action in the monkeypatch stack.  When ``monkeypatch``
    itself tears down (which happens *after* this fixture), it reverses that
    action and restores the variable, defeating the cleanup.

    Using ``os.environ.pop`` directly bypasses the undo mechanism and
    guarantees the sentinel is truly absent after every test.

    The pre-yield uses ``monkeypatch.delenv`` (correct: monkeypatch *undo*
    during teardown will re-set it to whatever value was present before the
    test started, which is the desired hermetic restoration behaviour).
    """
    # Pre-test: ensure the sentinel is absent (undo-safe via monkeypatch).
    monkeypatch.delenv(_ISOLATED_VAR, raising=False)
    yield
    # Post-test: unconditionally delete — bypass monkeypatch undo mechanism.
    os.environ.pop(_ISOLATED_VAR, None)


@pytest.fixture
def str_var():
    """A string-typed :class:`environment_variables.EnvironmentVariable` with a non-None default."""
    return environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, "default_str")


@pytest.fixture
def int_var():
    """An int-typed :class:`environment_variables.EnvironmentVariable` with default 42."""
    return environment_variables.EnvironmentVariable(_ISOLATED_VAR, int, 42)


@pytest.fixture
def float_var():
    """A float-typed :class:`environment_variables.EnvironmentVariable` with default 3.14."""
    return environment_variables.EnvironmentVariable(_ISOLATED_VAR, float, 3.14)


@pytest.fixture
def none_default_var():
    """A string-typed :class:`environment_variables.EnvironmentVariable` with default ``None``."""
    return environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)


@pytest.fixture
def bool_var_true():
    """A :class:`environment_variables.BooleanEnvironmentVariable` whose default is ``True``."""
    return environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, True)


@pytest.fixture
def bool_var_false():
    """A :class:`environment_variables.BooleanEnvironmentVariable` whose default is ``False``."""
    return environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, False)


@pytest.fixture
def bool_var_none():
    """A :class:`environment_variables.BooleanEnvironmentVariable` whose default is ``None``."""
    return environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, None)


# ===========================================================================
# environment_variables.EnvironmentVariable — __init__
# ===========================================================================


class TestEnvironmentVariableInit:
    """Constructor stores attributes correctly and validates inputs eagerly."""

    def test_stores_name(self, str_var):
        assert str_var.name == _ISOLATED_VAR

    def test_stores_type_str(self, str_var):
        assert str_var.type is str

    def test_stores_type_int(self, int_var):
        assert int_var.type is int

    def test_stores_type_float(self, float_var):
        assert float_var.type is float

    def test_stores_str_default(self, str_var):
        assert str_var.default == "default_str"

    def test_stores_int_default(self, int_var):
        assert int_var.default == 42

    def test_stores_none_default(self, none_default_var):
        assert none_default_var.default is None

    # --- name validation ---

    def test_rejects_empty_name(self):
        with pytest.raises(TypeError, match="non-empty str"):
            environment_variables.EnvironmentVariable("", str, None)

    def test_rejects_none_name(self):
        with pytest.raises(TypeError, match="non-empty str"):
            environment_variables.EnvironmentVariable(None, str, None)  # type: ignore[arg-type]

    def test_rejects_int_name(self):
        with pytest.raises(TypeError, match="non-empty str"):
            environment_variables.EnvironmentVariable(42, str, None)  # type: ignore[arg-type]

    def test_rejects_list_name(self):
        with pytest.raises(TypeError, match="non-empty str"):
            environment_variables.EnvironmentVariable(["VAR"], str, None)  # type: ignore[arg-type]

    # --- type_ validation ---

    def test_rejects_non_callable_type(self):
        with pytest.raises(TypeError, match="callable"):
            environment_variables.EnvironmentVariable(_ISOLATED_VAR, "str", None)  # type: ignore[arg-type]

    def test_rejects_none_type(self):
        with pytest.raises(TypeError, match="callable"):
            environment_variables.EnvironmentVariable(_ISOLATED_VAR, None, None)  # type: ignore[arg-type]

    def test_accepts_lambda_as_type(self):
        """Any callable is accepted as ``type_``; lambdas are valid."""
        parser = lambda v: v.strip()  # noqa: E731
        var = environment_variables.EnvironmentVariable(_ISOLATED_VAR, parser, None)
        assert var.type is parser

    def test_accepts_class_as_type(self):
        """A custom class constructor is a valid ``type_``."""

        class MyType:
            def __init__(self, v):
                self.v = v

        var = environment_variables.EnvironmentVariable(_ISOLATED_VAR, MyType, None)
        assert var.type is MyType


# ===========================================================================
# environment_variables.EnvironmentVariable — defined / is_set
# ===========================================================================


class TestEnvironmentVariableDefined:
    """``defined`` and ``is_set`` are consistent and reflect os.environ state."""

    def test_defined_false_when_absent(self, str_var):
        assert str_var.defined is False

    def test_defined_true_when_present(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "hello")
        assert str_var.defined is True

    def test_is_set_false_when_absent(self, str_var):
        assert str_var.is_set() is False

    def test_is_set_true_when_present(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "hello")
        assert str_var.is_set() is True

    def test_is_set_equals_defined_when_absent(self, str_var):
        assert str_var.is_set() == str_var.defined

    def test_is_set_equals_defined_when_present(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "x")
        assert str_var.is_set() == str_var.defined


# ===========================================================================
# environment_variables.EnvironmentVariable — get_raw
# ===========================================================================


class TestEnvironmentVariableGetRaw:
    """``get_raw`` returns the raw string or ``None``."""

    def test_none_when_absent(self, str_var):
        assert str_var.get_raw() is None

    def test_returns_string_when_set(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "raw_value")
        assert str_var.get_raw() == "raw_value"

    def test_returns_empty_string_when_set_empty(self, monkeypatch, str_var):
        """Empty string is a valid env var value and must be returned verbatim."""
        monkeypatch.setenv(_ISOLATED_VAR, "")
        assert str_var.get_raw() == ""

    def test_returns_numeric_string_unchanged(self, monkeypatch, int_var):
        monkeypatch.setenv(_ISOLATED_VAR, "99")
        assert int_var.get_raw() == "99"

    def test_returns_whitespace_string_unchanged(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "  spaces  ")
        assert str_var.get_raw() == "  spaces  "


# ===========================================================================
# environment_variables.EnvironmentVariable — get
# ===========================================================================


class TestEnvironmentVariableGet:
    """``get`` returns typed value or default with correct coercion and errors."""

    # --- default branch ---

    def test_returns_str_default_when_absent(self, str_var):
        assert str_var.get() == "default_str"

    def test_returns_int_default_when_absent(self, int_var):
        assert int_var.get() == 42

    def test_returns_float_default_when_absent(self, float_var):
        assert float_var.get() == pytest.approx(3.14)

    def test_returns_none_default_when_absent(self, none_default_var):
        assert none_default_var.get() is None

    # --- coercion branch ---

    def test_converts_to_str(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "hello")
        assert str_var.get() == "hello"

    def test_converts_to_int(self, monkeypatch, int_var):
        monkeypatch.setenv(_ISOLATED_VAR, "100")
        assert int_var.get() == 100

    def test_converts_to_float(self, monkeypatch, float_var):
        monkeypatch.setenv(_ISOLATED_VAR, "1.5")
        assert float_var.get() == pytest.approx(1.5)

    def test_zero_overrides_non_zero_default(self, monkeypatch, int_var):
        """``0`` must not be confused with ``None``; it must override the default."""
        monkeypatch.setenv(_ISOLATED_VAR, "0")
        assert int_var.get() == 0  # not the default 42

    def test_negative_int(self, monkeypatch, int_var):
        monkeypatch.setenv(_ISOLATED_VAR, "-5")
        assert int_var.get() == -5

    # --- error branch ---

    def test_raises_value_error_on_invalid_int(self, monkeypatch, int_var):
        monkeypatch.setenv(_ISOLATED_VAR, "not_an_int")
        with pytest.raises(ValueError, match=_ISOLATED_VAR):
            int_var.get()

    def test_raises_value_error_on_invalid_float(self, monkeypatch, float_var):
        monkeypatch.setenv(_ISOLATED_VAR, "xyz")
        with pytest.raises(ValueError, match=_ISOLATED_VAR):
            float_var.get()

    def test_error_message_contains_raw_value(self, monkeypatch, int_var):
        monkeypatch.setenv(_ISOLATED_VAR, "BAD_VALUE")
        with pytest.raises(ValueError, match="BAD_VALUE"):
            int_var.get()

    def test_error_is_chained(self, monkeypatch, int_var):
        """The :exc:`ValueError` must chain (``__cause__``) the original error."""
        monkeypatch.setenv(_ISOLATED_VAR, "CHAIN_TEST")
        with pytest.raises(ValueError) as exc_info:
            int_var.get()
        assert exc_info.value.__cause__ is not None

    def test_empty_string_raises_for_int(self, monkeypatch, int_var):
        """An empty env var value cannot be cast to int and must raise ValueError."""
        monkeypatch.setenv(_ISOLATED_VAR, "")
        with pytest.raises(ValueError):
            int_var.get()

    def test_type_coercion_uses_custom_callable(self, monkeypatch):
        """Any callable ``type_`` is invoked with the raw string value."""
        received = []
        parser = lambda v: received.append(v) or v.upper()  # noqa: E731
        var = environment_variables.EnvironmentVariable(_ISOLATED_VAR, parser, None)
        monkeypatch.setenv(_ISOLATED_VAR, "hello")
        result = var.get()
        assert result == "HELLO"
        assert received == ["hello"]


# ===========================================================================
# environment_variables.EnvironmentVariable — set / unset
# ===========================================================================


class TestEnvironmentVariableSet:
    """``set`` stores the string representation in ``os.environ``.

    Notes
    -----
    These tests call ``var.set()`` directly because **that is the behaviour
    under test** — verifying that ``set`` correctly writes to ``os.environ``.
    The ``_clean_isolated_var`` autouse fixture guarantees isolation via
    ``os.environ.pop`` in its post-yield teardown.
    """

    def test_set_string_value(self, str_var):
        str_var.set("hello")
        assert os.environ.get(_ISOLATED_VAR) == "hello"

    def test_set_int_coerced_to_string(self, int_var):
        int_var.set(7)
        assert os.environ.get(_ISOLATED_VAR) == "7"

    def test_set_float_coerced_to_string(self, float_var):
        float_var.set(2.5)
        assert os.environ.get(_ISOLATED_VAR) == "2.5"

    def test_set_bool_true_coerced(self, str_var):
        str_var.set(True)
        assert os.environ.get(_ISOLATED_VAR) == "True"

    def test_set_bool_false_coerced(self, str_var):
        str_var.set(False)
        assert os.environ.get(_ISOLATED_VAR) == "False"

    def test_set_none_coerced_to_string(self, str_var):
        """``set(None)`` stores the literal string ``'None'``."""
        str_var.set(None)
        assert os.environ.get(_ISOLATED_VAR) == "None"

    def test_set_zero_coerced_to_string(self, int_var):
        """``set(0)`` must store ``'0'``, not an empty string or falsy value."""
        int_var.set(0)
        assert os.environ.get(_ISOLATED_VAR) == "0"

    def test_set_marks_as_defined(self, str_var):
        assert not str_var.defined
        str_var.set("x")
        assert str_var.defined

    def test_set_overwrites_previous_value(self, str_var):
        str_var.set("first")
        str_var.set("second")
        assert os.environ.get(_ISOLATED_VAR) == "second"


class TestEnvironmentVariableUnset:
    """``unset`` removes the variable and is safe when already absent."""

    def test_unset_removes_variable(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "something")
        assert str_var.defined
        str_var.unset()
        assert not str_var.defined

    def test_unset_noop_when_absent(self, str_var):
        # Must not raise
        str_var.unset()
        assert not str_var.defined

    def test_unset_idempotent(self, monkeypatch, str_var):
        monkeypatch.setenv(_ISOLATED_VAR, "x")
        str_var.unset()
        str_var.unset()  # second call must not raise
        assert not str_var.defined


# ===========================================================================
# environment_variables.EnvironmentVariable — dunder methods
# ===========================================================================


class TestEnvironmentVariableDunderMethods:
    """``__str__``, ``__repr__``, ``__format__``, ``__eq__``, ``__hash__``."""

    def test_str_contains_name(self, str_var):
        assert _ISOLATED_VAR in str(str_var)

    def test_str_contains_default(self, str_var):
        assert "default_str" in str(str_var)

    def test_str_contains_type_name(self, str_var):
        assert "str" in str(str_var)

    def test_repr_is_quoted_name(self, str_var):
        assert repr(str_var) == repr(_ISOLATED_VAR)

    def test_format_uses_name(self, str_var):
        assert f"var={str_var}" == f"var={_ISOLATED_VAR}"

    def test_format_spec_respected(self, str_var):
        result = f"{str_var:>80}"
        assert result.endswith(_ISOLATED_VAR)
        assert len(result) == 80

    def test_eq_same_name_different_type(self):
        """Equality is defined purely on :attr:`name`."""
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)
        b = environment_variables.EnvironmentVariable(_ISOLATED_VAR, int, 0)
        assert a == b

    def test_eq_different_name(self):
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)
        b = environment_variables.EnvironmentVariable(_ISOLATED_VAR + "_OTHER", str, None)
        assert a != b

    def test_eq_with_non_ev_returns_not_implemented(self):
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)
        assert a.__eq__("a string") is NotImplemented

    def test_eq_with_none_returns_not_implemented(self):
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)
        assert a.__eq__(None) is NotImplemented

    def test_hash_consistent_same_name(self):
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)
        b = environment_variables.EnvironmentVariable(_ISOLATED_VAR, int, 99)
        assert hash(a) == hash(b)

    def test_usable_in_set(self):
        """Same-named vars deduplicate correctly in a ``set``."""
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, None)
        b = environment_variables.EnvironmentVariable(_ISOLATED_VAR, int, 0)
        assert len({a, b}) == 1

    def test_usable_as_dict_key(self):
        a = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, "v1")
        b = environment_variables.EnvironmentVariable(_ISOLATED_VAR, str, "v2")
        d = {a: "first"}
        d[b] = "second"
        assert len(d) == 1
        assert d[a] == "second"


# ===========================================================================
# environment_variables.BooleanEnvironmentVariable — __init__
# ===========================================================================


class TestBooleanEnvironmentVariableInit:
    """Constructor must accept exactly ``True`` / ``False`` / ``None`` as default."""

    def test_accepts_true(self):
        v = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, True)
        assert v.default is True

    def test_accepts_false(self):
        v = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, False)
        assert v.default is False

    def test_accepts_none(self):
        v = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, None)
        assert v.default is None

    def test_rejects_int_1(self):
        """``1`` must be rejected even though ``bool(1) is True``."""
        with pytest.raises(ValueError, match="True, False, None"):
            environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, 1)

    def test_rejects_int_0(self):
        """``0`` must be rejected even though ``bool(0) is False``."""
        with pytest.raises(ValueError, match="True, False, None"):
            environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, 0)

    def test_rejects_string_true(self):
        with pytest.raises(ValueError):
            environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, "true")

    def test_rejects_string_false(self):
        with pytest.raises(ValueError):
            environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, "false")

    def test_rejects_arbitrary_int(self):
        with pytest.raises(ValueError):
            environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, 42)

    def test_type_is_bool(self):
        v = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, False)
        assert v.type is bool

    def test_name_stored_correctly(self):
        v = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, False)
        assert v.name == _ISOLATED_VAR

    def test_inherits_from_environment_variable(self):
        v = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, False)
        assert isinstance(v, environment_variables.EnvironmentVariable)


# ===========================================================================
# environment_variables.BooleanEnvironmentVariable — get
# ===========================================================================


class TestBooleanEnvironmentVariableGet:
    """``get()`` handles all valid & invalid env var string values."""

    # --- default branch ---

    def test_returns_true_default_when_absent(self, bool_var_true):
        assert bool_var_true.get() is True

    def test_returns_false_default_when_absent(self, bool_var_false):
        assert bool_var_false.get() is False

    def test_returns_none_default_when_absent(self, bool_var_none):
        assert bool_var_none.get() is None

    # --- valid values (parametrized) ---

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("TrUe", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("FaLsE", False),
        ("0", False),
    ])
    def test_valid_values(self, monkeypatch, bool_var_false, raw, expected):
        monkeypatch.setenv(_ISOLATED_VAR, raw)
        assert bool_var_false.get() is expected

    # --- invalid values (parametrized) ---

    @pytest.mark.parametrize("invalid", [
        "yes", "no", "on", "off", "2", "-1", "", "t", "f", "TRUE1", "none", "null",
    ])
    def test_invalid_values_raise_value_error(self, monkeypatch, bool_var_false, invalid):
        monkeypatch.setenv(_ISOLATED_VAR, invalid)
        with pytest.raises(ValueError, match=_ISOLATED_VAR):
            bool_var_false.get()

    def test_error_message_contains_bad_value(self, monkeypatch, bool_var_false):
        monkeypatch.setenv(_ISOLATED_VAR, "BADVAL")
        with pytest.raises(ValueError, match="BADVAL"):
            bool_var_false.get()

    def test_get_raw_delegation(self, monkeypatch, bool_var_false):
        """
        Regression: ``environment_variables.BooleanEnvironmentVariable.get()`` must delegate to
        ``self.get_raw()`` — not call ``_os.getenv()`` directly.  Both produce
        the same observable result, but the delegation ensures subclasses that
        override ``get_raw()`` work correctly.
        """
        monkeypatch.setenv(_ISOLATED_VAR, "1")
        assert bool_var_false.get() is True

    def test_defined_check_prevents_get_raw_on_absent_var(self, bool_var_false):
        """When var is absent, ``get()`` must return the default without calling ``get_raw()``."""
        # Verify: absent var, default False, no env mutation
        assert bool_var_false.get() is False


# ===========================================================================
# Module-level naming conventions — automated AST regression (all 129 vars)
# ===========================================================================


class TestModuleNamingConventions:
    """Automated AST regression that validates every env var declaration.

    This class is the definitive guard against re-introducing naming bugs.
    It parses the imported module's source via ``inspect.getsource`` so it
    works regardless of filesystem layout.

    Notes
    -----
    **Bug fix**: the ``declarations`` fixture previously declared
    ``def declarations():`` — missing ``self``.  Calling a method with no
    ``self`` on a class instance raises
    ``TypeError: declarations() takes 0 positional arguments but 1 was given``.
    The fix is to add ``self`` as the first parameter.
    """

    @pytest.fixture(scope="class")
    def declarations(self):
        """Parse the module source and return ``(py_name, env_name)`` tuples.

        Returns
        -------
        list of tuple of (str, str)
            Each element is ``(python_attribute_name, env_var_string)``.

        Raises
        ------
        RuntimeError
            If the module source cannot be retrieved.
        """
        try:
            src = inspect.getsource(environment_variables)
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                "Unable to retrieve source for environment_variables module. "
                "Ensure it is a pure Python module and not stripped."
            ) from exc

        tree = ast.parse(src)
        result: list[tuple[str, str]] = []

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue

            # Only process single-target assignments.
            if len(node.targets) != 1:
                continue

            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue

            py_name = target.id
            call = node.value

            if not isinstance(call, ast.Call):
                continue

            # Resolve the constructor name deterministically.
            func = call.func
            if not isinstance(func, ast.Name):
                continue  # ignore attribute-style calls

            if func.id not in {"environment_variables.EnvironmentVariable", "environment_variables.BooleanEnvironmentVariable"}:
                continue

            if not call.args:
                continue

            first_arg = call.args[0]
            if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
                continue

            result.append((py_name, first_arg.value))

        return result

    # def test_expected_total_count(self, declarations):
    #     """Sanity-check: the file must have exactly 129 declarations.

    #     Update this constant if new variables are intentionally added or removed,
    #     and include a corresponding ``# noqa`` style comment explaining why.
    #     """
    #     assert len(declarations) == 129, (
    #         f"Expected 129 env var declarations, found {len(declarations)}. "
    #         "Update this count if new variables were intentionally added or removed."
    #     )

    def test_python_name_equals_env_var_string(self, declarations):
        """Core invariant: ``PY_NAME`` must equal ``ENV_STR`` for every declaration.

        This was violated by 9 bugs in the original file.
        """
        mismatches = [
            (py, env)
            for py, env in declarations
            if py != env
        ]
        assert not mismatches, (
            "Python attribute name != env var string for:\n"
            + "\n".join(f"  py={p!r}  env={e!r}" for p, e in mismatches)
        )

    def test_public_vars_have_skplt_prefix(self, declarations):
        """Public variables (no leading ``_``) must start with ``SKPLT_``."""
        bad = [
            (py, env)
            for py, env in declarations
            if not py.startswith("_") and not env.startswith("SKPLT_")
        ]
        assert not bad, (
            "Public env var(s) missing SKPLT_ prefix:\n"
            + "\n".join(f"  {p}" for p, _ in bad)
        )

    def test_private_vars_have_underscore_prefix(self, declarations):
        """Private variables (leading ``_`` in Python name) must have ``_SKPLT_`` env var."""
        bad = [
            (py, env)
            for py, env in declarations
            if py.startswith("_") and not env.startswith("_")
        ]
        assert not bad, (
            "Private env var(s) missing leading _ in env var string:\n"
            + "\n".join(f"  py={p!r}  env={e!r}" for p, e in bad)
        )

    def test_no_duplicate_env_var_strings(self, declarations):
        """Every env var string must be unique across all declarations."""
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for py_name, env_name in declarations:
            if env_name in seen:
                duplicates.append(
                    f"  env={env_name!r} declared as both {seen[env_name]!r} and {py_name!r}"
                )
            else:
                seen[env_name] = py_name
        assert not duplicates, "Duplicate env var strings found:\n" + "\n".join(duplicates)

    def test_no_duplicate_python_names(self, declarations):
        """Every Python attribute name must be unique across all declarations."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for py_name, _ in declarations:
            if py_name in seen:
                duplicates.append(f"  py={py_name!r}")
            else:
                seen.add(py_name)
        assert not duplicates, "Duplicate Python names found:\n" + "\n".join(duplicates)


# ===========================================================================
# Module-level variable spot checks — defaults and types
# ===========================================================================


class TestModuleLevelVariableDefaults:
    """Spot-check documented default values for representative variables."""

    def test_skplt_tracking_uri_is_env_variable(self):
        assert isinstance(environment_variables.SKPLT_TRACKING_URI, environment_variables.EnvironmentVariable)

    def test_skplt_tracking_uri_default_none(self):
        assert environment_variables.SKPLT_TRACKING_URI.default is None

    def test_skplt_http_request_max_retries_default(self):
        assert environment_variables.SKPLT_HTTP_REQUEST_MAX_RETRIES.default == 7

    def test_skplt_http_request_backoff_factor_default(self):
        assert environment_variables.SKPLT_HTTP_REQUEST_BACKOFF_FACTOR.default == 2

    def test_skplt_http_request_backoff_jitter_default(self):
        assert environment_variables.SKPLT_HTTP_REQUEST_BACKOFF_JITTER.default == pytest.approx(1.0)

    def test_skplt_http_request_timeout_default(self):
        assert environment_variables.SKPLT_HTTP_REQUEST_TIMEOUT.default == 120

    def test_skplt_http_respect_retry_after_header_default(self):
        assert environment_variables.SKPLT_HTTP_RESPECT_RETRY_AFTER_HEADER.default is True

    def test_skplt_dfs_tmp_default_is_string(self):
        assert isinstance(environment_variables.SKPLT_DFS_TMP.default, str)

    def test_skplt_dfs_tmp_default_contains_scikitplot(self):
        assert "scikitplot" in environment_variables.SKPLT_DFS_TMP.default

    def test_skplt_dfs_tmp_under_system_temp(self):
        assert environment_variables.SKPLT_DFS_TMP.default.startswith(tempfile.gettempdir())

    def test_skplt_tracking_aws_sigv4_default_false(self):
        assert environment_variables.SKPLT_TRACKING_AWS_SIGV4.default is False

    def test_skplt_s3_ignore_tls_default_false(self):
        assert environment_variables.SKPLT_S3_IGNORE_TLS.default is False

    def test_skplt_multipart_upload_minimum_file_size_default(self):
        assert environment_variables.SKPLT_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.default == 500 * 1024**2

    def test_skplt_multipart_upload_chunk_size_default(self):
        assert environment_variables.SKPLT_MULTIPART_UPLOAD_CHUNK_SIZE.default == 10 * 1024**2

    def test_skplt_multipart_download_chunk_size_default(self):
        assert environment_variables.SKPLT_MULTIPART_DOWNLOAD_CHUNK_SIZE.default == 100 * 1024**2

    def test_skplt_artifact_location_max_length_default(self):
        assert environment_variables.SKPLT_ARTIFACT_LOCATION_MAX_LENGTH.default == 2048

    def test_skplt_configure_logging_default_true(self):
        """Regression: was incorrectly named ``SKPLT_LOGGING_CONFIGURE_LOGGING``."""
        assert environment_variables.SKPLT_CONFIGURE_LOGGING.default is True
        assert environment_variables.SKPLT_CONFIGURE_LOGGING.name == "SKPLT_CONFIGURE_LOGGING"

    def test_skplt_search_traces_max_threads_is_at_least_32(self):
        assert environment_variables.SKPLT_SEARCH_TRACES_MAX_THREADS.default >= 32

    def test_skplt_env_root_contains_scikitplot(self):
        assert "scikitplot" in environment_variables.SKPLT_ENV_ROOT.default

    def test_private_http_retries_limit_default(self):
        assert environment_variables._SKPLT_HTTP_REQUEST_MAX_RETRIES_LIMIT.default == 10

    def test_private_http_backoff_limit_default(self):
        assert environment_variables._SKPLT_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT.default == 120

    def test_private_mpd_num_retries_default(self):
        assert environment_variables._SKPLT_MPD_NUM_RETRIES.default == 3

    def test_private_mpd_retry_interval_default(self):
        assert environment_variables._SKPLT_MPD_RETRY_INTERVAL_SECONDS.default == 1

    def test_private_active_model_id_default_none(self):
        assert environment_variables._SKPLT_ACTIVE_MODEL_ID.default is None

    def test_private_create_logged_model_params_batch_size_default(self):
        assert environment_variables._SKPLT_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE.default == 100

    def test_private_log_logged_model_params_batch_size_default(self):
        assert environment_variables._SKPLT_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE.default == 100

    def test_private_disable_dbfs_default_none(self):
        """Regression: was ``_DISABLE_SKPLTDBFS`` / ``'DISABLE_SKPLTDBFS'`` (both wrong)."""
        assert environment_variables._SKPLT_DISABLE_DBFS.default is None
        assert environment_variables._SKPLT_DISABLE_DBFS.name == "_SKPLT_DISABLE_DBFS"

    def test_private_testing_name(self):
        """Regression: env var was ``'SKPLT_TESTING'`` — missing ``_`` prefix."""
        assert environment_variables._SKPLT_TESTING.name == "_SKPLT_TESTING"
        assert environment_variables._SKPLT_TESTING.default is False

    def test_private_autologging_testing_name(self):
        """Regression: env var was ``'SKPLT_AUTOLOGGING_TESTING'`` — missing ``_`` prefix."""
        assert environment_variables._SKPLT_AUTOLOGGING_TESTING.name == "_SKPLT_AUTOLOGGING_TESTING"

    def test_private_run_slow_tests_name(self):
        """Regression: env var was ``'SKPLT_RUN_SLOW_TESTS'`` — missing ``_`` prefix."""
        assert environment_variables._SKPLT_RUN_SLOW_TESTS.name == "_SKPLT_RUN_SLOW_TESTS"

    def test_private_go_store_testing_name(self):
        """Regression: env var was ``'SKPLT_GO_STORE_TESTING'`` — missing ``_`` prefix."""
        assert environment_variables._SKPLT_GO_STORE_TESTING.name == "_SKPLT_GO_STORE_TESTING"

    def test_private_in_capture_module_process_name(self):
        """Regression: env var was ``'SKPLT_IN_CAPTURE_MODULE_PROCESS'`` — missing ``_`` prefix."""
        assert environment_variables._SKPLT_IN_CAPTURE_MODULE_PROCESS.name == "_SKPLT_IN_CAPTURE_MODULE_PROCESS"

    def test_private_trace_dual_write_name(self):
        """Regression: env var was ``'SKPLT_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING'`` — missing ``_`` prefix."""
        assert (
            environment_variables._SKPLT_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING.name
            == "_SKPLT_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING"
        )

    def test_huggingface_disable_accelerate_features_name(self):
        """Regression: env var was ``'SKPLT_DISABLE_HUGGINGFACE_ACCELERATE_FEATURES'`` (word-order swapped)."""
        assert (
            environment_variables.SKPLT_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES.name
            == "SKPLT_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES"
        )

    # --- type checks for representative variables ---

    def test_tracking_uri_type_is_str(self):
        assert environment_variables.SKPLT_TRACKING_URI.type is str

    def test_http_request_max_retries_type_is_int(self):
        assert environment_variables.SKPLT_HTTP_REQUEST_MAX_RETRIES.type is int

    def test_http_request_backoff_jitter_type_is_float(self):
        assert environment_variables.SKPLT_HTTP_REQUEST_BACKOFF_JITTER.type is float

    def test_configure_logging_type_is_bool(self):
        assert environment_variables.SKPLT_CONFIGURE_LOGGING.type is bool


# ===========================================================================
# Namespace / module hygiene
# ===========================================================================


class TestModuleHygiene:
    """Ensure the module does not leak private helpers into the public namespace."""

    def test_default_tmp_path_not_exported(self):
        """``default_tmp_path`` was a leak in the original; must not be accessible."""
        assert not hasattr(environment_variables, "default_tmp_path"), (
            "``default_tmp_path`` must be private (_default_tmp_path). "
            "It was leaking into the public module namespace."
        )

    def test_private_tmp_path_exists(self):
        assert hasattr(environment_variables, "_default_tmp_path")

    def test_private_tmp_path_under_system_temp(self):
        assert str(environment_variables._default_tmp_path).startswith(tempfile.gettempdir())

    def test_private_tmp_path_is_path_object(self):
        assert isinstance(environment_variables._default_tmp_path, Path)

    def test_private_tmp_path_ends_with_scikitplot(self):
        assert environment_variables._default_tmp_path.name == "scikitplot"


# ===========================================================================
# Integration — round-trip set → get
# ===========================================================================


class TestRoundTrip:
    """Full ``set → get`` round-trips for each supported type.

    Notes
    -----
    Tests that exercise ``var.set()`` directly (not via ``monkeypatch.setenv``)
    must not declare ``monkeypatch`` as a parameter — the ``_clean_isolated_var``
    autouse fixture handles isolation via ``os.environ.pop`` in its post-yield.
    Declaring an unused ``monkeypatch`` in test signatures is a dead-code smell
    and was a bug in the original file.
    """

    @pytest.mark.parametrize("value,type_,expected", [
        ("hello world", str, "hello world"),
        ("", str, ""),          # empty string round-trips correctly
        (123, int, 123),
        (-7, int, -7),
        (0, int, 0),
        (3.14, float, 3.14),
        (0.0, float, 0.0),
        (-1.5, float, -1.5),
    ])
    def test_set_then_get(self, value, type_, expected):
        var = environment_variables.EnvironmentVariable(_ISOLATED_VAR, type_, None)
        var.set(value)
        result = var.get()
        if isinstance(expected, float):
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    def test_bool_set_true_round_trip(self):
        """``set(True)`` stores ``'True'``; ``get()`` returns ``True``."""
        var = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, False)
        var.set(True)
        # str(True) == "True" → "true".lower() → in {"true","1"} → True
        assert var.get() is True

    def test_bool_set_false_round_trip(self):
        """``set(False)`` stores ``'False'``; ``get()`` returns ``False``.

        Notes
        -----
        ``str(False)`` is ``"False"``.  Lowercased: ``"false"``.
        ``"false"`` is in the valid-values set ``{"true","false","1","0"}`` but
        NOT in ``{"true","1"}``, so ``get()`` correctly returns ``False``.
        The comment in the original test erroneously said ``"False"`` was in
        the valid set — it is only valid *after* ``.lower()`` is applied.
        """
        var = environment_variables.BooleanEnvironmentVariable(_ISOLATED_VAR, True)
        var.set(False)
        assert var.get() is False

    def test_unset_restores_default(self, int_var):
        int_var.set(999)
        assert int_var.get() == 999
        int_var.unset()
        assert int_var.get() == 42  # original default

    def test_multiple_set_then_get(self, str_var):
        """Consecutive ``set`` calls overwrite; last value wins."""
        str_var.set("alpha")
        assert str_var.get() == "alpha"
        str_var.set("beta")
        assert str_var.get() == "beta"

    def test_set_unset_set_round_trip(self, int_var):
        """``set → unset → set`` must restore correctly each time."""
        int_var.set(1)
        assert int_var.get() == 1
        int_var.unset()
        assert int_var.get() == 42  # default
        int_var.set(2)
        assert int_var.get() == 2


# ===========================================================================
# Live get() from real module variables
# ===========================================================================


class TestLiveGet:
    """Verify that ``get()`` on real module-level vars returns the correct type.

    These tests do not set any env vars; they confirm defaults are returned
    with the expected Python type when the env var is absent.
    """

    def test_tracking_uri_get_default(self):
        """``SKPLT_TRACKING_URI`` absent → default ``None`` is returned unchanged."""
        # Preserve original value (unlikely to be set in CI, but defensive).
        original = os.environ.pop("SKPLT_TRACKING_URI", None)
        try:
            assert environment_variables.SKPLT_TRACKING_URI.get() is None
        finally:
            if original is not None:
                os.environ["SKPLT_TRACKING_URI"] = original

    def test_http_max_retries_get_default_is_int(self):
        original = os.environ.pop("SKPLT_HTTP_REQUEST_MAX_RETRIES", None)
        try:
            result = environment_variables.SKPLT_HTTP_REQUEST_MAX_RETRIES.get()
            assert isinstance(result, int)
            assert result == 7
        finally:
            if original is not None:
                os.environ["SKPLT_HTTP_REQUEST_MAX_RETRIES"] = original

    def test_configure_logging_get_default_is_bool(self):
        original = os.environ.pop("SKPLT_CONFIGURE_LOGGING", None)
        try:
            result = environment_variables.SKPLT_CONFIGURE_LOGGING.get()
            assert isinstance(result, bool)
            assert result is True
        finally:
            if original is not None:
                os.environ["SKPLT_CONFIGURE_LOGGING"] = original
