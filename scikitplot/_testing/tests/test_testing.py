# scikitplot/_testing/tests/test_testing.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive tests for :mod:`scikitplot._testing._testing`.

Coverage targets
----------------
* :class:`scikitplot._testing._testing._IgnoreWarnings`
    - __init__ validation (valid, invalid category types)
    - context-manager protocol (suppression, category filtering, filter
      restoration, exception propagation, enter returns self)
    - decorator protocol (wrapping, metadata preservation, non-callable error)
    - sequential reuse of the same instance
    - re-entrant guard (RuntimeError)
    - __exit__ without __enter__ guard (RuntimeError)
    - __repr__

* :class:`scikitplot._testing._testing._AssertNoWarningsContext`
    - __init__ validation
    - context-manager protocol (no-warn passes, matching warn raises,
      non-matching warn passes, subclass warns trigger, multiple warns,
      exception propagation, filter restoration)
    - sequential reuse
    - re-entrant guard
    - __exit__ without __enter__ guard
    - __repr__

* :func:`_testing._testing.ignore_warnings` (public factory)
    - returns _IgnoreWarnings when obj is None
    - immediate callable wrap
    - decorator-factory mode
    - Warning subclass as obj raises ValueError
    - non-callable non-None obj raises TypeError
    - category keyword propagated

* :func:`scikitplot._testing._testing.assert_no_warnings` (public factory)
    - returns _AssertNoWarningsContext
    - default and custom warning_class
    - functional integration (no-warn, warn)

* :data:`scikitplot._testing.SkipTest` re-export alias
"""

from __future__ import annotations

import functools
import unittest
import warnings

import pytest

# ---------------------------------------------------------------------------
# Imports under test — relative so this file works as part of the scikitplot
# package tree AND as a standalone package when _testing is the root.
# ---------------------------------------------------------------------------
from .._testing import (  # noqa: TID252
    _AssertNoWarningsContext,
    _IgnoreWarnings,
    assert_no_warnings,
    ignore_warnings,
)
from .. import SkipTest  # noqa: TID252


# ===========================================================================
# Helpers
# ===========================================================================


def _emit(category: type = UserWarning, msg: str = "test warning") -> None:
    """Emit a single warning of *category* with *msg*."""
    warnings.warn(msg, category, stacklevel=1)


# ===========================================================================
# SkipTest re-export
# ===========================================================================


class TestSkipTestAlias:
    """SkipTest must be the exact same object as unittest.case.SkipTest."""

    def test_is_unittest_skip_test(self) -> None:
        assert SkipTest is unittest.case.SkipTest

    def test_raises_as_skip(self) -> None:
        with pytest.raises(SkipTest):
            raise SkipTest("intentional skip")


# ===========================================================================
# _IgnoreWarnings — __init__ validation
# ===========================================================================


class TestIgnoreWarningsInit:
    """Constructor must accept Warning subclasses and reject everything else."""

    def test_default_category_is_warning(self) -> None:
        iw = _IgnoreWarnings()
        assert iw.category is Warning

    def test_specific_subclass_accepted(self) -> None:
        iw = _IgnoreWarnings(category=DeprecationWarning)
        assert iw.category is DeprecationWarning

    def test_warning_itself_accepted(self) -> None:
        iw = _IgnoreWarnings(category=Warning)
        assert iw.category is Warning

    def test_non_warning_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Warning subclass"):
            _IgnoreWarnings(category=Exception)  # type: ignore[arg-type]

    def test_non_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Warning subclass"):
            _IgnoreWarnings(category="UserWarning")  # type: ignore[arg-type]

    def test_int_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _IgnoreWarnings(category=42)  # type: ignore[arg-type]


# ===========================================================================
# _IgnoreWarnings — context-manager protocol
# ===========================================================================


class TestIgnoreWarningsContextManager:
    """Context-manager behaviour: suppression, scope, restoration, exceptions."""

    def test_suppresses_default_category(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _IgnoreWarnings():
                warnings.warn("gone", UserWarning)
        assert not any(issubclass(w.category, UserWarning) for w in captured)

    def test_suppresses_specific_category(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _IgnoreWarnings(category=DeprecationWarning):
                warnings.warn("gone", DeprecationWarning)
        assert not any(
            issubclass(w.category, DeprecationWarning) for w in captured
        )

    def test_does_not_suppress_other_categories(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _IgnoreWarnings(category=DeprecationWarning):
                warnings.warn("kept", UserWarning)
        assert any(issubclass(w.category, UserWarning) for w in captured)

    def test_restores_filters_after_normal_exit(self) -> None:
        original = warnings.filters[:]
        with _IgnoreWarnings():
            pass
        assert warnings.filters == original

    def test_restores_filters_after_exception(self) -> None:
        original = warnings.filters[:]
        with pytest.raises(RuntimeError):
            with _IgnoreWarnings():
                raise RuntimeError("boom")
        assert warnings.filters == original

    def test_exception_inside_block_propagates(self) -> None:
        with pytest.raises(ValueError, match="oops"):
            with _IgnoreWarnings():
                raise ValueError("oops")

    def test_enter_returns_self(self) -> None:
        iw = _IgnoreWarnings()
        with iw as ctx:
            assert ctx is iw

    def test_suppresses_warning_subclass(self) -> None:
        """A subclass of the watched category should also be suppressed."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with _IgnoreWarnings(category=Warning):
                warnings.warn("gone", DeprecationWarning)
        assert not any(issubclass(w.category, DeprecationWarning) for w in captured)


# ===========================================================================
# _IgnoreWarnings — sequential reuse and re-entry guards
# ===========================================================================


class TestIgnoreWarningsReuseAndGuards:
    """Re-entrant use must raise; sequential reuse must succeed."""

    def test_sequential_reuse_works(self) -> None:
        iw = _IgnoreWarnings()
        with iw:
            pass
        # Re-entering the same instance after exit must succeed.
        with iw:
            pass

    def test_reentrant_raises_runtime_error(self) -> None:
        iw = _IgnoreWarnings()
        with iw:
            with pytest.raises(RuntimeError, match="twice"):
                iw.__enter__()

    def test_exit_without_enter_raises_runtime_error(self) -> None:
        iw = _IgnoreWarnings()
        with pytest.raises(RuntimeError, match="without entering"):
            iw.__exit__(None, None, None)

    def test_entered_flag_reset_after_exit(self) -> None:
        iw = _IgnoreWarnings()
        with iw:
            assert iw._entered is True
        assert iw._entered is False

    def test_catch_warnings_cleared_after_exit(self) -> None:
        iw = _IgnoreWarnings()
        with iw:
            pass
        assert iw._catch_warnings is None


# ===========================================================================
# _IgnoreWarnings — decorator protocol
# ===========================================================================


class TestIgnoreWarningsDecorator:
    """Decorator use: wrapping behaviour, metadata, type guard."""

    def test_decorated_function_suppresses_warnings(self) -> None:
        iw = _IgnoreWarnings(category=UserWarning)

        @iw
        def noisy():
            warnings.warn("gone", UserWarning)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            noisy()
        assert not any(issubclass(w.category, UserWarning) for w in captured)

    def test_decorated_function_preserves_name(self) -> None:
        iw = _IgnoreWarnings()

        @iw
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"

    def test_decorated_function_preserves_docstring(self) -> None:
        iw = _IgnoreWarnings()

        @iw
        def documented():
            """Important docs."""

        assert documented.__doc__ == "Important docs."

    def test_decorated_function_passes_args_and_kwargs(self) -> None:
        iw = _IgnoreWarnings()

        @iw
        def add(a, b, *, c=0):
            return a + b + c

        assert add(1, 2, c=3) == 6

    def test_non_callable_raises_type_error(self) -> None:
        iw = _IgnoreWarnings()
        with pytest.raises(TypeError, match="callable"):
            iw(42)  # type: ignore[arg-type]

    def test_decorated_function_can_be_called_multiple_times(self) -> None:
        iw = _IgnoreWarnings(category=DeprecationWarning)
        call_count = 0

        @iw
        def fn():
            nonlocal call_count
            call_count += 1
            warnings.warn("dep", DeprecationWarning)

        fn()
        fn()
        assert call_count == 2


# ===========================================================================
# _IgnoreWarnings — __repr__
# ===========================================================================


class TestIgnoreWarningsRepr:
    def test_repr_default(self) -> None:
        iw = _IgnoreWarnings()
        assert repr(iw) == "_IgnoreWarnings(category=Warning)"

    def test_repr_custom_category(self) -> None:
        iw = _IgnoreWarnings(category=DeprecationWarning)
        assert repr(iw) == "_IgnoreWarnings(category=DeprecationWarning)"


# ===========================================================================
# _AssertNoWarningsContext — __init__ validation
# ===========================================================================


class TestAssertNoWarningsContextInit:
    def test_default_warning_class_is_warning(self) -> None:
        ctx = _AssertNoWarningsContext()
        assert ctx.warning_class is Warning

    def test_specific_subclass_accepted(self) -> None:
        ctx = _AssertNoWarningsContext(warning_class=DeprecationWarning)
        assert ctx.warning_class is DeprecationWarning

    def test_non_warning_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Warning subclass"):
            _AssertNoWarningsContext(warning_class=Exception)  # type: ignore[arg-type]

    def test_non_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Warning subclass"):
            _AssertNoWarningsContext(warning_class=3.14)  # type: ignore[arg-type]


# ===========================================================================
# _AssertNoWarningsContext — context-manager behaviour
# ===========================================================================


class TestAssertNoWarningsContextBehavior:
    """Functional: no-warn passes; matching warn raises; non-matching passes."""

    def test_no_warnings_passes_silently(self) -> None:
        with _AssertNoWarningsContext():
            pass  # no warning — should not raise

    def test_matching_warning_raises_assertion_error(self) -> None:
        with pytest.raises(AssertionError, match="Expected no.*Warning"):
            with _AssertNoWarningsContext():
                warnings.warn("oops", UserWarning)

    def test_non_matching_category_passes(self) -> None:
        with _AssertNoWarningsContext(warning_class=DeprecationWarning):
            warnings.warn("unrelated", UserWarning)
        # No AssertionError raised — non-matching category is invisible.

    def test_subclass_of_watched_triggers_assertion(self) -> None:
        """DeprecationWarning is a subclass of Warning; it should be caught."""
        with pytest.raises(AssertionError):
            with _AssertNoWarningsContext(warning_class=Warning):
                warnings.warn("dep", DeprecationWarning)

    def test_multiple_warnings_all_appear_in_message(self) -> None:
        with pytest.raises(AssertionError, match="got 2"):
            with _AssertNoWarningsContext():
                warnings.warn("first", UserWarning)
                warnings.warn("second", UserWarning)

    def test_exception_inside_block_propagates_without_assertion(self) -> None:
        """A real exception in the block must not be swallowed or shadowed."""
        with pytest.raises(RuntimeError, match="real error"):
            with _AssertNoWarningsContext():
                raise RuntimeError("real error")

    def test_exception_with_warning_propagates_exception_not_assertion(self) -> None:
        """When both an exception and a warning occur, the exception wins."""
        with pytest.raises(ValueError, match="exception wins"):
            with _AssertNoWarningsContext():
                warnings.warn("ignored in presence of exception", UserWarning)
                raise ValueError("exception wins")

    def test_restores_warning_filters_after_exit(self) -> None:
        original = warnings.filters[:]
        with _AssertNoWarningsContext():
            pass
        assert warnings.filters == original

    def test_assertion_error_names_the_category(self) -> None:
        with pytest.raises(AssertionError, match="DeprecationWarning"):
            with _AssertNoWarningsContext(warning_class=DeprecationWarning):
                warnings.warn("dep", DeprecationWarning)

    def test_enter_returns_self(self) -> None:
        ctx = _AssertNoWarningsContext()
        with ctx as bound:
            assert bound is ctx


# ===========================================================================
# _AssertNoWarningsContext — sequential reuse and guards
# ===========================================================================


class TestAssertNoWarningsContextEdgeCases:
    def test_sequential_reuse_works(self) -> None:
        ctx = _AssertNoWarningsContext()
        with ctx:
            pass
        with ctx:
            pass

    def test_reentrant_raises_runtime_error(self) -> None:
        ctx = _AssertNoWarningsContext()
        with ctx:
            with pytest.raises(RuntimeError, match="twice"):
                ctx.__enter__()

    def test_exit_without_enter_raises_runtime_error(self) -> None:
        ctx = _AssertNoWarningsContext()
        with pytest.raises(RuntimeError, match="without entering"):
            ctx.__exit__(None, None, None)

    def test_entered_flag_reset_after_exit(self) -> None:
        ctx = _AssertNoWarningsContext()
        with ctx:
            assert ctx._entered is True
        assert ctx._entered is False

    def test_recorded_none_before_enter(self) -> None:
        ctx = _AssertNoWarningsContext()
        assert ctx._recorded is None


# ===========================================================================
# _AssertNoWarningsContext — __repr__
# ===========================================================================


class TestAssertNoWarningsContextRepr:
    def test_repr_default(self) -> None:
        ctx = _AssertNoWarningsContext()
        assert repr(ctx) == "_AssertNoWarningsContext(warning_class=Warning)"

    def test_repr_custom_class(self) -> None:
        ctx = _AssertNoWarningsContext(warning_class=FutureWarning)
        assert repr(ctx) == "_AssertNoWarningsContext(warning_class=FutureWarning)"


# ===========================================================================
# ignore_warnings — public factory
# ===========================================================================


class TestIgnoreWarningsFactory:
    """Public factory :func:`ignore_warnings` — all three usage modes."""

    # --- returns _IgnoreWarnings ---

    def test_no_args_returns_ignore_warnings_instance(self) -> None:
        result = ignore_warnings()
        assert isinstance(result, _IgnoreWarnings)

    def test_default_category_is_warning(self) -> None:
        result = ignore_warnings()
        assert result.category is Warning

    def test_category_kwarg_propagated(self) -> None:
        result = ignore_warnings(category=DeprecationWarning)
        assert result.category is DeprecationWarning

    # --- context-manager mode ---

    def test_as_context_manager_suppresses_warnings(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with ignore_warnings():
                warnings.warn("gone", UserWarning)
        assert not any(issubclass(w.category, UserWarning) for w in captured)

    # --- decorator-factory mode ---

    def test_decorator_factory_suppresses_on_call(self) -> None:
        @ignore_warnings(category=UserWarning)
        def noisy():
            warnings.warn("gone", UserWarning)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            noisy()
        assert not any(issubclass(w.category, UserWarning) for w in captured)

    # --- immediate wrap mode ---

    def test_immediate_wrap_suppresses_warnings(self) -> None:
        def noisy():
            warnings.warn("gone", UserWarning)

        wrapped = ignore_warnings(noisy)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            wrapped()
        assert not any(issubclass(w.category, UserWarning) for w in captured)

    def test_immediate_wrap_returns_callable(self) -> None:
        wrapped = ignore_warnings(lambda: 42)
        assert callable(wrapped)
        assert wrapped() == 42

    # --- error guards ---

    def test_warning_subclass_as_obj_raises_value_error(self) -> None:
        """Common mistake: passing the warning class instead of category=."""
        with pytest.raises(ValueError, match="not a warning class"):
            ignore_warnings(DeprecationWarning)  # type: ignore[arg-type]

    def test_warning_itself_as_obj_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ignore_warnings(Warning)  # type: ignore[arg-type]

    def test_non_callable_non_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="callable or None"):
            ignore_warnings(42)  # type: ignore[arg-type]

    def test_none_is_valid_and_returns_instance(self) -> None:
        result = ignore_warnings(None)
        assert isinstance(result, _IgnoreWarnings)

    def test_functools_wraps_preserves_metadata(self) -> None:
        """Immediate wrap must carry __name__ and __doc__ from the original."""

        def annotated():
            """Docstring."""

        wrapped = ignore_warnings(annotated)
        assert wrapped.__name__ == "annotated"
        assert wrapped.__doc__ == "Docstring."


# ===========================================================================
# assert_no_warnings — public factory
# ===========================================================================


class TestAssertNoWarningsFactory:
    """Public factory :func:`assert_no_warnings` — basic coverage."""

    def test_returns_assert_no_warnings_context(self) -> None:
        result = assert_no_warnings()
        assert isinstance(result, _AssertNoWarningsContext)

    def test_default_warning_class_is_warning(self) -> None:
        result = assert_no_warnings()
        assert result.warning_class is Warning

    def test_custom_class_propagated(self) -> None:
        result = assert_no_warnings(DeprecationWarning)
        assert result.warning_class is DeprecationWarning

    def test_no_warnings_passes(self) -> None:
        with assert_no_warnings():
            x = 1 + 1  # no warning
        assert x == 2

    def test_matching_warning_raises_assertion(self) -> None:
        with pytest.raises(AssertionError):
            with assert_no_warnings():
                warnings.warn("oops", UserWarning)

    def test_non_matching_warning_passes(self) -> None:
        with assert_no_warnings(DeprecationWarning):
            warnings.warn("unrelated", UserWarning)

    def test_invalid_class_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            assert_no_warnings(int)  # type: ignore[arg-type]

    def test_context_manager_enter_returns_self(self) -> None:
        ctx = assert_no_warnings()
        with ctx as bound:
            assert bound is ctx


# ===========================================================================
# Integration: ignore_warnings + assert_no_warnings interaction
# ===========================================================================


class TestIntegration:
    """Nested usage and interaction between the two utilities."""

    def test_inner_catch_warnings_record_overrides_outer_ignore(self) -> None:
        """Inner catch_warnings(record=True) resets filters to 'always', so
        a warning IS captured even when an outer context has 'ignore'.

        This is the documented behavior of warnings.catch_warnings nesting:
        each context manager saves and restores filter state independently,
        and the inner simplefilter('always') replaces whatever outer filter
        was in effect.  The outer 'ignore' has no effect inside the inner
        record context.
        """
        with ignore_warnings(category=UserWarning):
            with pytest.raises(AssertionError):
                with assert_no_warnings(UserWarning):
                    warnings.warn("visible inside inner record ctx", UserWarning)

    def test_ignore_warnings_inside_assert_no_warnings_suppresses(self) -> None:
        """Inner suppression prevents outer assertion checker from seeing it."""
        with assert_no_warnings(UserWarning):
            with ignore_warnings(category=UserWarning):
                warnings.warn("suppressed before assert_no_warnings sees it", UserWarning)

    def test_ignore_warnings_decorator_stacks_with_functools_wraps(self) -> None:
        """Stacking two ignore_warnings decorators must preserve metadata."""

        @ignore_warnings(category=DeprecationWarning)
        @ignore_warnings(category=UserWarning)
        def doubly_wrapped():
            """Two-layer suppression."""
            warnings.warn("dep", DeprecationWarning)
            warnings.warn("user", UserWarning)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            doubly_wrapped()
        assert len(captured) == 0
        assert doubly_wrapped.__doc__ == "Two-layer suppression."
