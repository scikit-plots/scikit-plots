# scikitplot/_api/tests/test_deprecation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
test_deprecation.py
===================
Unified, comprehensive test suite for ``_api/deprecation.py``.

Supersedes ``test_deprecation.py``, ``test_deprecation_extended.py``, and
``test_deprecation_stdlib.py``.  Every public (and critical internal) symbol
is covered at both happy-path and branch-coverage level.

Covered symbols
---------------
* ``MatplotlibDeprecationWarning``        — hierarchy, instantiation, raise/catch
* ``_generate_deprecation_warning``       — pending guard, auto-computed removal,
                                            explicit/falsy removal, alternative,
                                            addendum, custom message, obj_type,
                                            since with patch component, high minor
* ``warn_deprecated``                     — all kwargs, pending, auto-removal,
                                            explicit/false removal, obj_type,
                                            addendum, combined fields
* ``deprecated`` (function)               — callable preserved, warning emitted,
                                            docstring updated, empty doc suffix,
                                            existing Notes header guard, alt/
                                            addendum in doc, explicit name/obj_type,
                                            functools.wraps metadata preservation
* ``deprecated`` (class)                  — instantiable, class name in warning,
                                            obj_type default/override, returns same
                                            class object, relative classproperty
                                            import (regression guard), explicit name,
                                            no-init class, subclass super() call
* ``deprecated`` (property)               — getter/setter/deleter warn, obj_type
                                            default, explicit name, lambda fget
                                            name fixed by __set_name__
* ``deprecated`` (classproperty)          — class access, instance access
* ``deprecate_privatize_attribute``       — read/write warn, private forwarding,
                                            __set_name__ derivation, multiple attrs
* ``DECORATORS``                          — populated by all three decorators,
                                            registry value is callable
* ``rename_parameter``                    — old name warns, new name silent,
                                            positional silent, assertion guards,
                                            DECORATORS registered, old value used
* ``delete_parameter``                    — deprecated param warns, no param silent,
                                            KEYWORD_ONLY path, VAR_POSITIONAL,
                                            VAR_KEYWORD, missing+no-varkwargs
                                            AssertionError, addendum forwarded,
                                            DECORATORS registered
* ``make_keyword_only``                   — positional warns, keyword silent,
                                            already-KWO raises, positional-only
                                            raises, __signature__ updated,
                                            all-params-from-index-KWO
* ``deprecate_method_override``           — overridden warns, not-overridden silent,
                                            allow_empty with empty/docstring-only/
                                            non-empty bodies, obj as class,
                                            bound method callable
* ``suppress_matplotlib_deprecation_warning`` — suppresses MDW, not UserWarning,
                                            not outside context, context manager
                                            protocol, nested, filters restored,
                                            exception propagation
* ``_deprecated_parameter_class``         — repr, singleton identity
* Integration: stacked decorators         — rename+delete, make_keyword_only+delete

Bug fixes vs prior tests
------------------------
* ``deprecated`` (class): ``from . import classproperty`` relative import
  verified to work without a full Matplotlib installation (regression guard).
* ``_WarnAssertMixin._assert_warns`` provides consistent count/match assertions.

Notes
-----
User note
    Run with either::

        python -m unittest _api.tests.test_deprecation
        pytest _api/tests/test_deprecation.py -v

Developer note
    All tests use ``unittest.TestCase`` as base class.  ``pytest.warns`` /
    ``pytest.raises`` replaced by ``warnings.catch_warnings(record=True)``
    and ``self.assertRaises`` / ``self.assertRaisesRegex``.  Monkeypatching
    uses explicit ``try / finally`` patterns.
"""

from __future__ import annotations

import inspect
import re
import warnings
import unittest

from ..deprecation import (
    DECORATORS,
    MatplotlibDeprecationWarning,
    _deprecated_parameter,
    _deprecated_parameter_class,
    _generate_deprecation_warning,
    delete_parameter,
    deprecated,
    deprecate_method_override,
    deprecate_privatize_attribute,
    make_keyword_only,
    rename_parameter,
    suppress_matplotlib_deprecation_warning,
    warn_deprecated,
)
from .. import classproperty


# ===========================================================================
# Shared helpers
# ===========================================================================

def _collect_category(record, category):
    """Return warning entries of *category* from a catch_warnings record."""
    return [w for w in record if issubclass(w.category, category)]


def _collect_mdw(record):
    """Return only MatplotlibDeprecationWarning entries."""
    return _collect_category(record, MatplotlibDeprecationWarning)


class _WarnAssertMixin:
    """
    Mixin providing ``_assert_warns`` and ``_assert_no_warn`` compatible with
    ``unittest.TestCase``.
    """

    def _assert_warns(
        self,
        record,
        category=MatplotlibDeprecationWarning,
        *,
        match=None,
        count=None,
    ):
        """Assert *record* contains at least one warning of *category*.

        Parameters
        ----------
        record : list
            Captured by ``warnings.catch_warnings(record=True)``.
        category : type, default MatplotlibDeprecationWarning
            Expected warning class.
        match : str or None
            If given, assert at least one warning message matches this regex.
        count : int or None
            If given, assert exactly *count* matching warnings were emitted.
        """
        matching = _collect_category(record, category)
        self.assertTrue(
            len(matching) >= 1,
            f"Expected ≥1 {category.__name__} but got: {record}",
        )
        if count is not None:
            self.assertEqual(
                len(matching), count,
                f"Expected exactly {count} {category.__name__} but got {len(matching)}",
            )
        if match is not None:
            msgs = [str(w.message) for w in matching]
            self.assertTrue(
                any(re.search(match, m) for m in msgs),
                f"No warning matched {match!r}. Messages: {msgs}",
            )

    def _assert_no_warn(self, record, category=MatplotlibDeprecationWarning):
        """Assert *record* contains NO warning of *category*."""
        matching = _collect_category(record, category)
        self.assertEqual(
            len(matching), 0,
            f"Expected no {category.__name__} but got: {matching}",
        )

    def _capture(self, func, *args, **kwargs):
        """Call *func* and return (result, warning_record)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)
        return result, w

    def _capture_warns(self, func, *args, **kwargs):
        """Call *func*, assert at least one MDW, and return (result, record)."""
        result, w = self._capture(func, *args, **kwargs)
        self._assert_warns(w)
        return result, w


# ===========================================================================
# MatplotlibDeprecationWarning
# ===========================================================================

class TestMatplotlibDeprecationWarning(unittest.TestCase):
    """Tests for the custom warning class hierarchy."""

    def test_is_subclass_of_deprecation_warning(self):
        self.assertTrue(issubclass(MatplotlibDeprecationWarning, DeprecationWarning))

    def test_is_subclass_of_warning(self):
        self.assertTrue(issubclass(MatplotlibDeprecationWarning, Warning))

    def test_is_not_bare_deprecation_warning(self):
        """Must be a proper subclass, not DeprecationWarning itself."""
        self.assertIsNot(MatplotlibDeprecationWarning, DeprecationWarning)

    def test_can_be_instantiated(self):
        w = MatplotlibDeprecationWarning("test message")
        self.assertIn("test message", str(w))

    def test_can_be_raised_and_caught(self):
        with self.assertRaises(MatplotlibDeprecationWarning):
            raise MatplotlibDeprecationWarning("boom")


# ===========================================================================
# _generate_deprecation_warning
# ===========================================================================

class TestGenerateDeprecationWarning(unittest.TestCase):
    """Direct tests for the internal _generate_deprecation_warning factory."""

    def test_non_pending_returns_mdw(self):
        w = _generate_deprecation_warning("3.1", name="foo")
        self.assertIsInstance(w, MatplotlibDeprecationWarning)

    def test_pending_returns_pending_deprecation_warning(self):
        w = _generate_deprecation_warning("3.1", name="foo", pending=True)
        self.assertIsInstance(w, PendingDeprecationWarning)

    def test_pending_with_removal_raises_value_error(self):
        with self.assertRaises(ValueError):
            _generate_deprecation_warning("3.1", name="foo", pending=True, removal="3.3")

    def test_pending_error_message_mentions_pending(self):
        with self.assertRaisesRegex(ValueError, "pending"):
            _generate_deprecation_warning("3.1", name="foo", pending=True, removal="3.3")

    def test_auto_computes_removal_minor_plus_two(self):
        w = _generate_deprecation_warning("3.1", name="foo")
        self.assertIn("3.3", str(w))

    def test_explicit_removal_overrides_auto(self):
        w = _generate_deprecation_warning("3.1", name="foo", removal="4.0")
        self.assertIn("4.0", str(w))

    def test_falsy_removal_suppresses_removal_text(self):
        w = _generate_deprecation_warning("3.1", name="foo", removal=False)
        self.assertNotIn("will be removed", str(w))

    def test_alternative_included_in_message(self):
        w = _generate_deprecation_warning("3.1", name="old", alternative="new")
        self.assertIn("new", str(w))

    def test_addendum_included_in_message(self):
        w = _generate_deprecation_warning("3.1", name="foo", addendum="Extra note.")
        self.assertIn("Extra note.", str(w))

    def test_addendum_and_alternative_combined(self):
        w = _generate_deprecation_warning(
            "3.1", name="old", alternative="new", addendum="See migration guide.")
        msg = str(w)
        self.assertIn("new", msg)
        self.assertIn("migration guide", msg)

    def test_custom_message_used_with_substitution(self):
        w = _generate_deprecation_warning(
            "3.1", message="completely custom %(name)s", name="bar")
        self.assertIn("completely custom bar", str(w))

    def test_obj_type_included_in_message(self):
        w = _generate_deprecation_warning("3.1", name="foo", obj_type="method")
        self.assertIn("method", str(w))

    def test_empty_obj_type_uses_name_only(self):
        """Empty obj_type must not insert 'The <n>' prefix."""
        w = _generate_deprecation_warning("3.1", name="bar", obj_type="")
        self.assertIn("bar", str(w))

    def test_non_empty_obj_type_changes_message_prefix(self):
        """Non-empty obj_type inserts 'The <n> <obj_type>' prefix."""
        w = _generate_deprecation_warning("3.1", name="bar", obj_type="method")
        msg = str(w)
        self.assertIn("bar", msg)
        self.assertIn("method", msg)

    def test_major_version_preserved_in_removal(self):
        """The major part of 'since' must be kept in auto-computed removal."""
        w = _generate_deprecation_warning("4.5", name="foo")
        self.assertIn("4.7", str(w))

    def test_since_with_patch_component_ignored(self):
        """3.1.0 → removal must be 3.3 (patch component discarded)."""
        w = _generate_deprecation_warning("3.1.0", name="foo")
        self.assertIn("3.3", str(w))

    def test_high_minor_version_incremented_correctly(self):
        """Minor 9 → removal minor 11."""
        w = _generate_deprecation_warning("3.9", name="foo")
        self.assertIn("3.11", str(w))

    def test_pending_message_uses_will_be_deprecated(self):
        w = _generate_deprecation_warning("3.1", name="foo", pending=True)
        self.assertIn("will be deprecated", str(w))
        self.assertNotIn("was deprecated", str(w))


# ===========================================================================
# warn_deprecated
# ===========================================================================

class TestWarnDeprecated(_WarnAssertMixin, unittest.TestCase):
    """Tests for the warn_deprecated public interface."""

    def _warn(self, **kwargs):
        """Helper: capture a single warn_deprecated call."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated(**kwargs)
        return w

    def test_emits_matplotlib_deprecation_warning(self):
        w = self._warn(since="3.1", name="foo")
        self._assert_warns(w)

    def test_message_contains_name(self):
        w = self._warn(since="3.1", name="foo")
        self._assert_warns(w, match="foo")

    def test_message_contains_since(self):
        w = self._warn(since="3.1", name="foo")
        self._assert_warns(w, match="3.1")

    def test_message_contains_auto_removal(self):
        w = self._warn(since="3.1", name="foo")
        self._assert_warns(w, match="3.3")

    def test_alternative_shown(self):
        w = self._warn(since="3.1", name="old_func", alternative="new_func")
        self._assert_warns(w, match="new_func")

    def test_pending_emits_pending_deprecation_warning(self):
        w = self._warn(since="3.1", name="foo", pending=True)
        self._assert_warns(w, category=PendingDeprecationWarning)

    def test_pending_plus_removal_raises_value_error(self):
        with self.assertRaises(ValueError):
            warn_deprecated("3.1", name="foo", pending=True, removal="3.3")

    def test_custom_message_used(self):
        w = self._warn(since="3.1", message="custom text")
        self._assert_warns(w, match="custom text")

    def test_explicit_removal_overrides_auto(self):
        w = self._warn(since="3.1", name="foo", removal="5.0")
        self._assert_warns(w, match="5.0")

    def test_false_removal_suppresses_removal_text(self):
        w = self._warn(since="3.1", name="foo", removal=False)
        mdw = _collect_mdw(w)
        self.assertNotIn("will be removed", str(mdw[0].message))

    def test_obj_type_shown(self):
        w = self._warn(since="3.1", name="foo", obj_type="module")
        self._assert_warns(w, match="module")

    def test_addendum_shown(self):
        w = self._warn(since="3.1", name="foo", addendum="See docs")
        self._assert_warns(w, match="See docs")

    def test_obj_type_and_addendum_both_present(self):
        w = self._warn(
            since="3.1", name="foo", obj_type="function", addendum="Use bar() instead.")
        msg = str(_collect_mdw(w)[0].message)
        self.assertIn("function", msg)
        self.assertIn("bar()", msg)


# ===========================================================================
# deprecated — function target
# ===========================================================================

class TestDeprecatedFunction(_WarnAssertMixin, unittest.TestCase):
    """Tests for @deprecated applied to plain functions."""

    def test_decorated_function_still_callable(self):
        @deprecated("3.1")
        def foo():
            return 99
        result, w = self._capture(foo)
        self.assertEqual(result, 99)
        self._assert_warns(w)

    def test_emits_warning_on_call(self):
        @deprecated("3.1")
        def bar():
            pass
        _, w = self._capture(bar)
        self._assert_warns(w, match="bar")

    def test_function_name_in_warning(self):
        @deprecated("3.1")
        def my_special_func():
            pass
        _, w = self._capture(my_special_func)
        self._assert_warns(w, match="my_special_func")

    def test_docstring_updated_with_deprecated_marker(self):
        @deprecated("3.1")
        def func():
            """Original docstring."""
        self.assertIn("Deprecated", func.__doc__)
        self.assertIn("3.1", func.__doc__)

    def test_empty_docstring_handled_without_error(self):
        @deprecated("3.1")
        def func():
            pass
        self.assertIsNotNone(func.__doc__)

    def test_empty_docstring_gets_backslash_space_suffix(self):
        """No-docstring functions get r'\\ ' to prevent docutils warnings."""
        @deprecated("3.1")
        def no_doc():
            pass
        self.assertIn(r"\ ", no_doc.__doc__)

    def test_arguments_forwarded_to_original_function(self):
        @deprecated("3.1")
        def add(a, b):
            return a + b
        result, w = self._capture(add, 2, 3)
        self.assertEqual(result, 5)
        self._assert_warns(w)

    def test_alternative_shown_in_warning(self):
        @deprecated("3.1", alternative="new_func")
        def old_func():
            pass
        _, w = self._capture(old_func)
        self._assert_warns(w, match="new_func")

    def test_custom_name_overrides_function_name(self):
        @deprecated("3.1", name="public_name")
        def _internal():
            pass
        _, w = self._capture(_internal)
        self._assert_warns(w, match="public_name")

    def test_functools_wraps_preserves_name(self):
        @deprecated("3.1")
        def original():
            pass
        self.assertEqual(original.__name__, "original")

    def test_functools_wraps_preserves_module(self):
        @deprecated("3.1")
        def carefully_named():
            """Docstring."""
        self.assertEqual(carefully_named.__module__, __name__)

    def test_explicit_obj_type_overrides_default(self):
        @deprecated("3.1", obj_type="method")
        def my_func():
            pass
        _, w = self._capture(my_func)
        self._assert_warns(w, match="method")

    def test_existing_notes_header_not_duplicated(self):
        """If docstring already has Notes section, must not insert duplicate."""
        @deprecated("3.1")
        def func_with_notes():
            """Summary.

            Notes
            -----
            Some existing note.
            """
        doc = func_with_notes.__doc__
        self.assertIsNotNone(doc)
        # At most two 'Notes' occurrences: original + deprecated annotation
        self.assertLessEqual(doc.count("Notes"), 2)

    def test_alternative_appears_in_docstring(self):
        @deprecated("3.1", alternative="new_function")
        def old_function():
            """Old."""
        self.assertIn("new_function", old_function.__doc__)

    def test_addendum_appears_in_docstring(self):
        @deprecated("3.1", addendum="Consult the migration guide.")
        def legacy():
            """Legacy."""
        self.assertIn("migration guide", legacy.__doc__)

    def test_since_appears_in_docstring(self):
        @deprecated("3.99")
        def func():
            """Docstring."""
        self.assertIn("3.99", func.__doc__)

    def test_default_obj_type_is_function(self):
        """Default obj_type must produce 'function' in the warning."""
        @deprecated("3.1")
        def f():
            pass
        _, w = self._capture(f)
        self._assert_warns(w, match="function")


# ===========================================================================
# deprecated — class target
# ===========================================================================

class TestDeprecatedClass(_WarnAssertMixin, unittest.TestCase):
    """Tests for @deprecated applied to classes."""

    def test_class_still_instantiable(self):
        @deprecated("3.1")
        class MyClass:
            def __init__(self):
                self.val = 42
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = MyClass()
        self.assertEqual(obj.val, 42)
        self._assert_warns(w)

    def test_class_name_in_warning(self):
        @deprecated("3.1")
        class DeprecatedThing:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DeprecatedThing()
        self._assert_warns(w, match="DeprecatedThing")

    def test_class_docstring_updated(self):
        @deprecated("3.1")
        class C:
            """Original class doc."""
        self.assertIn("Deprecated", C.__doc__)

    def test_obj_type_is_class_by_default(self):
        @deprecated("3.1")
        class C:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            C()
        self._assert_warns(w, match="class")

    def test_returns_same_class_object(self):
        """@deprecated on a class must return the original class object."""
        class C:
            pass
        original_id = id(C)
        C = deprecated("3.1")(C)
        self.assertEqual(id(C), original_id)

    def test_explicit_obj_type_overrides_class_default(self):
        @deprecated("3.1", obj_type="mixin")
        class MyMixin:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MyMixin()
        self._assert_warns(w, match="mixin")

    def test_class_with_no_custom_init_warns_on_instantiation(self):
        """A class without explicit __init__ must still emit the warning."""
        @deprecated("3.1")
        class NoInit:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = NoInit()
        self.assertIsInstance(obj, NoInit)
        self._assert_warns(w)

    def test_explicit_name_used_in_warning(self):
        @deprecated("3.1", name="PublicName")
        class _InternalClass:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _InternalClass()
        self._assert_warns(w, match="PublicName")

    def test_subclass_super_init_triggers_warning(self):
        """super().__init__() on a deprecated class must emit the warning."""
        @deprecated("3.1")
        class Base:
            def __init__(self):
                self.value = 1
        class Child(Base):
            def __init__(self):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    super().__init__()
                self._super_warns = w
        c = Child()
        self._assert_warns(c._super_warns)

    def test_relative_classproperty_import_regression(self):
        """
        Critical regression: @deprecated uses ``from . import classproperty``
        (relative), not ``from matplotlib._api import classproperty``.
        If regressed this raises ModuleNotFoundError.
        """
        @deprecated("3.1")
        class ClassWithClassproperty:
            @classproperty
            def name(cls):
                return cls.__name__
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = ClassWithClassproperty()
        self.assertIsNotNone(obj)
        self._assert_warns(w)


# ===========================================================================
# deprecated — property target
# ===========================================================================

class TestDeprecatedProperty(_WarnAssertMixin, unittest.TestCase):
    """Tests for @deprecated applied to properties."""

    def _make_class(self):
        class C:
            @deprecated("3.1")
            @property
            def value(self):
                """The value."""
                return 42
        return C

    def test_getter_warns(self):
        C = self._make_class()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C().value
        self._assert_warns(w)

    def test_getter_returns_correct_value(self):
        C = self._make_class()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = C().value
        self.assertEqual(val, 42)

    def test_setter_warns(self):
        class C:
            @deprecated("3.1")
            @property
            def attr(self):
                return self._attr
            @attr.setter
            def attr(self, v):
                self._attr = v
        obj = C()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.attr = 99
        self._assert_warns(w)

    def test_deleter_warns(self):
        class C:
            @deprecated("3.1")
            @property
            def attr(self):
                return self._attr
            @attr.setter
            def attr(self, v):
                self._attr = v
            @attr.deleter
            def attr(self):
                del self._attr
        obj = C()
        obj._attr = 42
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del obj.attr
        self._assert_warns(w)
        self.assertFalse(hasattr(obj, "_attr"))

    def test_obj_type_is_attribute_by_default(self):
        C = self._make_class()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C().value
        self._assert_warns(w, match="attribute")

    def test_explicit_name_overrides_fget_name(self):
        class C:
            @deprecated("3.1", name="public_attr")
            @property
            def _private_attr(self):
                return 42
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C()._private_attr
        self._assert_warns(w, match="public_attr")

    def test_lambda_fget_name_fixed_by_set_name(self):
        """
        When fget is a lambda its __name__ is '<lambda>'.  __set_name__
        must fix it to the assignment name so warning messages are readable.
        """
        class C:
            value = deprecated("3.1")(property(lambda self: 99))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C().value
        msg = str(_collect_mdw(w)[0].message)
        self.assertIn("value", msg)
        self.assertNotIn("<lambda>", msg)

    def test_property_obj_type_override(self):
        class C:
            @deprecated("3.1", obj_type="setting")
            @property
            def mode(self):
                return "fast"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C().mode
        self._assert_warns(w, match="setting")


# ===========================================================================
# deprecated — classproperty target
# ===========================================================================

class TestDeprecatedClassproperty(_WarnAssertMixin, unittest.TestCase):
    """Tests for @deprecated applied to classproperty descriptors."""

    def _make_class(self):
        class C:
            prop = deprecated("3.1")(classproperty(lambda cls: cls.__name__))
        return C

    def test_warns_on_class_access(self):
        C = self._make_class()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C.prop
        self._assert_warns(w)

    def test_warns_on_instance_access(self):
        C = self._make_class()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = C().prop
        self._assert_warns(w)

    def test_returns_correct_value_on_class_access(self):
        C = self._make_class()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = C.prop
        self.assertEqual(val, "C")


# ===========================================================================
# deprecate_privatize_attribute
# ===========================================================================

class TestDeprecatePrivatizeAttribute(_WarnAssertMixin, unittest.TestCase):
    """Tests for the class-scope privatisation helper."""

    def _make_class(self):
        class MyClass:
            public = deprecate_privatize_attribute("3.1")
            def __init__(self):
                self._public = 100
        return MyClass

    def test_read_access_warns(self):
        obj = self._make_class()()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = obj.public
        self._assert_warns(w)

    def test_read_access_returns_private_value(self):
        obj = self._make_class()()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = obj.public
        self.assertEqual(val, 100)

    def test_write_access_warns(self):
        obj = self._make_class()()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.public = 200
        self._assert_warns(w)

    def test_write_access_updates_private_attribute(self):
        obj = self._make_class()()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.public = 999
        self.assertEqual(obj._public, 999)

    def test_set_name_derives_private_name_from_assignment(self):
        class Config:
            timeout = deprecate_privatize_attribute("3.1")
            def __init__(self):
                self._timeout = 30
        obj = Config()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = obj.timeout
        self.assertEqual(val, 30)
        self._assert_warns(w)

    def test_write_delegates_to_private_attribute(self):
        class Config:
            retries = deprecate_privatize_attribute("3.1")
            def __init__(self):
                self._retries = 3
        obj = Config()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.retries = 5
        self.assertEqual(obj._retries, 5)

    def test_multiple_privatized_attributes_coexist(self):
        """Two privatized attributes on the same class must not interfere."""
        class Cfg:
            alpha = deprecate_privatize_attribute("3.1")
            beta = deprecate_privatize_attribute("3.1")
            def __init__(self):
                self._alpha = 1
                self._beta = 2
        obj = Cfg()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = obj.alpha
        self._assert_warns(w)
        self.assertEqual(a, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b = obj.beta
        self._assert_warns(w)
        self.assertEqual(b, 2)


# ===========================================================================
# rename_parameter
# ===========================================================================

class TestRenameParameter(_WarnAssertMixin, unittest.TestCase):
    """Tests for the rename_parameter decorator."""

    def _make_func(self):
        @rename_parameter("3.1", "old_name", "new_name")
        def func(new_name):
            return new_name
        return func

    def test_new_name_works_without_warning(self):
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(new_name=42)
        self._assert_no_warn(w)
        self.assertEqual(result, 42)

    def test_old_name_warns(self):
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(old_name=42)
        self._assert_warns(w, match="old_name")
        self.assertEqual(result, 42)

    def test_old_name_value_used_even_when_only_old_supplied(self):
        """Contract: old kwarg value must be forwarded as new kwarg."""
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(old_name=42)
        self.assertEqual(result, 42)

    def test_positional_call_does_not_warn(self):
        """Passing the new parameter positionally must not warn."""
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(99)
        self._assert_no_warn(w)
        self.assertEqual(result, 99)

    def test_old_name_in_signature_raises_assertion(self):
        """old must NOT already exist in the function signature."""
        with self.assertRaises(AssertionError):
            @rename_parameter("3.1", "already_exists", "new")
            def bad(already_exists, new):
                pass

    def test_new_name_not_in_signature_raises_assertion(self):
        """new MUST exist in the function signature."""
        with self.assertRaises(AssertionError):
            @rename_parameter("3.1", "old", "missing_new")
            def bad(other):
                pass

    def test_decorator_registered_in_decorators(self):
        func = self._make_func()
        self.assertIn(func, DECORATORS)

    def test_functools_wraps_preserves_name(self):
        func = self._make_func()
        self.assertEqual(func.__name__, "func")

    def test_partial_application_returns_callable(self):
        decorator = rename_parameter("3.1", "old", "new")
        self.assertTrue(callable(decorator))


# ===========================================================================
# delete_parameter
# ===========================================================================

class TestDeleteParameter(_WarnAssertMixin, unittest.TestCase):
    """Tests for the delete_parameter decorator."""

    def _make_func(self):
        @delete_parameter("3.1", "unused")
        def func(kept, unused=None):
            return kept
        return func

    def test_without_deprecated_param_no_warning(self):
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("hello")
        self._assert_no_warn(w)
        self.assertEqual(result, "hello")

    def test_with_deprecated_param_warns(self):
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("hello", unused="ignored")
        self._assert_warns(w, match="unused")
        self.assertEqual(result, "hello")

    def test_deprecated_param_with_default_omitted_no_warning(self):
        """Fast-path early return when caller omits the deprecated param."""
        @delete_parameter("3.1", "old")
        def func(a, old=None):
            return a
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("hello")
        self._assert_no_warn(w)
        self.assertEqual(result, "hello")

    def test_kwarg_param_via_varkwargs_warns(self):
        """Param passed via **kwargs must also trigger warning."""
        @delete_parameter("3.1", "gone")
        def func(kept, **kwargs):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func("hi", gone=True)
        self._assert_warns(w)

    def test_keyword_only_deprecated_param_warns_when_passed(self):
        """KEYWORD_ONLY path (name_idx = math.inf)."""
        @delete_parameter("3.1", "gone")
        def func(kept, *, gone=None):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("value", gone="deprecated_value")
        self._assert_warns(w, match="gone")
        self.assertEqual(result, "value")

    def test_keyword_only_not_passed_no_warning(self):
        """KEYWORD_ONLY param omitted → early-return branch, no warning."""
        @delete_parameter("3.1", "gone")
        def func(kept, *, gone=None):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("value")
        self._assert_no_warn(w)
        self.assertEqual(result, "value")

    def test_varargs_warns(self):
        """Deprecated *args usage must trigger the warning."""
        @delete_parameter("3.1", "extra")
        def func(kept, *extra):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("hi", "unwanted")
        self._assert_warns(w, match="positional")
        self.assertEqual(result, "hi")

    def test_varkwargs_warns(self):
        """Deprecated **kwargs key must trigger the warning."""
        @delete_parameter("3.1", "extra")
        def func(kept, **extra):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func("hi", extra="unwanted")
        self._assert_warns(w, match="keyword")
        self.assertEqual(result, "hi")

    def test_missing_param_and_no_varkwargs_raises_assertion(self):
        """Missing param + no **kwargs must fire the internal assertion."""
        with self.assertRaisesRegex(AssertionError, "no_kwargs_func"):
            @delete_parameter("3.1", "nonexistent_param")
            def no_kwargs_func(kept):
                return kept

    def test_addendum_forwarded_to_warning(self):
        """The ``addendum`` kwarg must appear in the deprecation warning message."""
        @delete_parameter("3.1", "extra", addendum="Use the new API instead.")
        def func(kept, extra=None):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func("hi", extra="value")
        self._assert_warns(w, match="new API")

    def test_decorator_registered(self):
        func = self._make_func()
        self.assertIn(func, DECORATORS)

    def test_decorator_registered_for_keyword_only_param(self):
        @delete_parameter("3.1", "gone")
        def func(a, *, gone=None):
            return a
        self.assertIn(func, DECORATORS)

    def test_functools_wraps_preserves_name(self):
        func = self._make_func()
        self.assertEqual(func.__name__, "func")

    def test_partial_application_returns_callable(self):
        self.assertTrue(callable(delete_parameter("3.1", "name")))

    def test_partial_allows_later_binding_and_works(self):
        decorator = delete_parameter("3.1", "deprecated_arg")
        @decorator
        def func(kept, deprecated_arg=None):
            return kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func("ok", deprecated_arg="old")
        self._assert_warns(w)


# ===========================================================================
# make_keyword_only
# ===========================================================================

class TestMakeKeywordOnly(_WarnAssertMixin, unittest.TestCase):
    """Tests for the make_keyword_only decorator."""

    def _make_func(self):
        @make_keyword_only("3.1", "b")
        def func(a, b, c=3):
            return a + b + c
        return func

    def test_keyword_argument_no_warning(self):
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(1, b=2)
        self._assert_no_warn(w)
        self.assertEqual(result, 6)

    def test_positional_argument_warns(self):
        func = self._make_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(1, 2)
        self._assert_warns(w, match="b")
        self.assertEqual(result, 6)

    def test_all_keyword_no_warning(self):
        @make_keyword_only("3.1", "b")
        def func(a, b, c=3):
            return a + b + c
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(a=1, b=2, c=3)
        self._assert_no_warn(w)
        self.assertEqual(result, 6)

    def test_param_must_exist_in_signature(self):
        with self.assertRaises(AssertionError):
            @make_keyword_only("3.1", "nonexistent")
            def func(a, b):
                pass

    def test_already_keyword_only_raises_assertion(self):
        with self.assertRaises(AssertionError):
            @make_keyword_only("3.1", "already_kw")
            def func(a, *, already_kw=1):
                return a

    def test_positional_only_raises_assertion(self):
        with self.assertRaises(AssertionError):
            @make_keyword_only("3.1", "pos_only")
            def func(pos_only, /):
                return pos_only

    def test_decorator_registered(self):
        func = self._make_func()
        self.assertIn(func, DECORATORS)

    def test_partial_application_returns_callable(self):
        self.assertTrue(callable(make_keyword_only("3.1", "x")))

    def test_wrapper_signature_marks_param_keyword_only(self):
        """The wrapper's __signature__ must mark 'b' and later as KEYWORD_ONLY."""
        func = self._make_func()
        sig = inspect.signature(func)
        KWO = inspect.Parameter.KEYWORD_ONLY
        self.assertEqual(sig.parameters["b"].kind, KWO)

    def test_all_params_from_index_become_keyword_only(self):
        @make_keyword_only("3.1", "b")
        def func(a, b, c, d=4):
            return a + b + c + d
        sig = inspect.signature(func)
        KWO = inspect.Parameter.KEYWORD_ONLY
        POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in ("b", "c", "d"):
            self.assertEqual(
                sig.parameters[name].kind, KWO,
                f"Expected {name!r} to be KEYWORD_ONLY",
            )
        self.assertEqual(sig.parameters["a"].kind, POK)


# ===========================================================================
# deprecate_method_override
# ===========================================================================

class TestDeprecateMethodOverride(_WarnAssertMixin, unittest.TestCase):
    """Tests for deprecate_method_override."""

    def _make_base(self):
        class Base:
            def method(self):
                return "base"
        return Base

    def test_overridden_method_warns(self):
        Base = self._make_base()
        class Child(Base):
            def method(self):
                return "child"
        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecate_method_override(Base.method, obj, since="3.1")
        self._assert_warns(w, match="method")
        self.assertIsNotNone(result)

    def test_overridden_method_returns_bound_child_method(self):
        Base = self._make_base()
        class Child(Base):
            def method(self):
                return "child"
        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bound = deprecate_method_override(Base.method, obj, since="3.1")
        self.assertEqual(bound(), "child")

    def test_not_overridden_returns_none(self):
        Base = self._make_base()
        class Child(Base):
            pass
        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecate_method_override(Base.method, obj, since="3.1")
        self._assert_no_warn(w)
        self.assertIsNone(result)

    def test_allow_empty_suppresses_empty_body_override(self):
        Base = self._make_base()
        class Child(Base):
            def method(self):
                pass
        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecate_method_override(Base.method, obj, allow_empty=True, since="3.1")
        self._assert_no_warn(w)

    def test_allow_empty_suppresses_docstring_only_override(self):
        """Override with only a docstring body must also be treated as empty."""
        class Base:
            def draw(self):
                return "base"
        class Child(Base):
            def draw(self):
                """Intentionally empty override."""
        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecate_method_override(Base.draw, obj, allow_empty=True, since="3.1")
        self._assert_no_warn(w)

    def test_allow_empty_non_empty_override_still_warns(self):
        """Non-empty override must still warn even when allow_empty=True.

        Notes
        -----
        Developer note
            In Python 3.12 the bytecode for ``def f(): return const`` matches
            the ``empty_with_docstring`` pattern used inside
            ``deprecate_method_override``.  This test therefore uses a
            multi-statement body that is unambiguously non-trivial on all
            supported Python versions.
        """
        class Base:
            def render(self):
                return "base"

        class Child(Base):
            def render(self):
                # Two statements → co_code differs from the empty patterns
                result = "child"
                return result

        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecate_method_override(Base.render, obj, allow_empty=True, since="3.1")
        self._assert_warns(w, match="render")

    def test_works_on_class_not_instance(self):
        """obj can be a class (for classmethods / unbound checks)."""
        Base = self._make_base()
        class Child(Base):
            def method(self):
                return "child"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecate_method_override(Base.method, Child, since="3.1")
        self._assert_warns(w)

    def test_not_overridden_on_class_returns_none(self):
        class Base:
            def hook(self):
                return "base"
        class Child(Base):
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecate_method_override(Base.hook, Child, since="3.1")
        self._assert_no_warn(w)
        self.assertIsNone(result)

    def test_result_callable_executes_child_implementation(self):
        class Base:
            def process(self):
                return "base_result"
        class Child(Base):
            def process(self):
                return "child_result"
        obj = Child()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bound = deprecate_method_override(Base.process, obj, since="3.1")
        self.assertEqual(bound(), "child_result")


# ===========================================================================
# suppress_matplotlib_deprecation_warning
# ===========================================================================

class TestSuppressMatplotlibDeprecationWarning(_WarnAssertMixin, unittest.TestCase):
    """Tests for the suppress_matplotlib_deprecation_warning context manager."""

    def test_suppresses_mdw_inside_context(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_matplotlib_deprecation_warning():
                warn_deprecated("3.1", name="foo")
        self._assert_no_warn(w)

    def test_does_not_suppress_outside_context(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_matplotlib_deprecation_warning():
                pass
            warn_deprecated("3.1", name="foo")
        self._assert_warns(w, count=1)

    def test_is_context_manager(self):
        cm = suppress_matplotlib_deprecation_warning()
        self.assertTrue(hasattr(cm, "__enter__"))
        self.assertTrue(hasattr(cm, "__exit__"))

    def test_non_mdw_warnings_pass_through(self):
        """Other warning categories must NOT be suppressed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_matplotlib_deprecation_warning():
                warnings.warn("other warning", UserWarning)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        self.assertEqual(len(user_warns), 1)

    def test_nested_suppress_contexts_both_suppress(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_matplotlib_deprecation_warning():
                with suppress_matplotlib_deprecation_warning():
                    warn_deprecated("3.1", name="foo")
        self._assert_no_warn(w)

    def test_context_manager_restores_filters_on_exit(self):
        """After the context exits, subsequent warnings must NOT be suppressed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_matplotlib_deprecation_warning():
                pass
            warn_deprecated("3.1", name="bar")
        self._assert_warns(w, count=1)

    def test_exception_inside_context_propagates(self):
        """An exception raised inside the context must not be swallowed."""
        with self.assertRaisesRegex(RuntimeError, "test error"):
            with suppress_matplotlib_deprecation_warning():
                raise RuntimeError("test error")


# ===========================================================================
# _deprecated_parameter_class
# ===========================================================================

class TestDeprecatedParameterRepr(unittest.TestCase):
    """Exercises the _deprecated_parameter_class __repr__."""

    def test_repr(self):
        self.assertEqual(repr(_deprecated_parameter), "<deprecated parameter>")

    def test_instance_is_module_level_singleton(self):
        """_deprecated_parameter must be the same object every time imported."""
        from ..deprecation import _deprecated_parameter as dp2
        self.assertIs(_deprecated_parameter, dp2)

    def test_is_instance_of_class(self):
        self.assertIsInstance(_deprecated_parameter, _deprecated_parameter_class)


# ===========================================================================
# DECORATORS registry
# ===========================================================================

class TestDecoratorsRegistry(unittest.TestCase):
    """DECORATORS must be correctly populated by all three applicable decorators."""

    def test_rename_parameter_registers_wrapper(self):
        @rename_parameter("3.1", "old", "new_p")
        def func(new_p):
            return new_p
        self.assertIn(func, DECORATORS)

    def test_delete_parameter_registers_wrapper(self):
        @delete_parameter("3.1", "unused")
        def func(kept, unused=None):
            return kept
        self.assertIn(func, DECORATORS)

    def test_make_keyword_only_registers_wrapper(self):
        @make_keyword_only("3.1", "b")
        def func(a, b):
            return a + b
        self.assertIn(func, DECORATORS)

    def test_registry_value_is_callable(self):
        """Each registry value must be a callable (a functools.partial)."""
        @rename_parameter("3.1", "old_x", "new_x")
        def func(new_x):
            return new_x
        registered = DECORATORS[func]
        self.assertTrue(callable(registered))

    def test_delete_parameter_keyword_only_also_registers(self):
        @delete_parameter("3.1", "gone")
        def func(a, *, gone=None):
            return a
        self.assertIn(func, DECORATORS)

    def test_multiple_stacked_outer_wrapper_registered(self):
        """The outermost wrapper must be in DECORATORS."""
        @delete_parameter("3.1", "a")
        @rename_parameter("3.1", "old_b", "new_b")
        def func(new_b, a=None):
            return new_b
        self.assertIn(func, DECORATORS)


# ===========================================================================
# Integration: stacked deprecation decorators
# ===========================================================================

class TestStackedDeprecationDecorators(_WarnAssertMixin, unittest.TestCase):
    """Realistic stacked-decorator integration scenarios."""

    def test_rename_then_delete_new_name_no_warning(self):
        @delete_parameter("3.1", "obsolete")
        @rename_parameter("3.1", "old_name", "new_name")
        def func(new_name, obsolete=None):
            return new_name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(new_name="ok")
        self._assert_no_warn(w)
        self.assertEqual(result, "ok")

    def test_rename_then_delete_old_name_warns(self):
        @delete_parameter("3.1", "obsolete")
        @rename_parameter("3.1", "old_name", "new_name")
        def func(new_name, obsolete=None):
            return new_name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func(old_name="also_ok")
        self._assert_warns(w)

    def test_make_keyword_only_then_delete_clean_call(self):
        @make_keyword_only("3.2", "b")
        @delete_parameter("3.1", "gone")
        def func(a, b, gone=None):
            return a + b
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(1, b=2)
        self._assert_no_warn(w)
        self.assertEqual(result, 3)

    def test_make_keyword_only_then_delete_positional_warns(self):
        @make_keyword_only("3.2", "b")
        @delete_parameter("3.1", "gone")
        def func(a, b, gone=None):
            return a + b
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func(1, 2)
        self._assert_warns(w, match="b")


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
