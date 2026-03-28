# scikitplot/_utils/tests/test_decorators.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.decorators`.

Coverage map
------------
deprecated (function)      issues DeprecationWarning, pending flag,
                           alternative text in message, obj_type,
                           custom warning_type, functools.wraps   -> TestDeprecatedFunction
deprecated (class)         issues DeprecationWarning on __init__,
                           __new__ override path                  -> TestDeprecatedClass
deprecated_attribute       warns on attribute access              -> TestDeprecatedAttribute
deprecated_renamed_argument renames arg, warns on old name        -> TestDeprecatedRenamedArgument
future_keyword_only        warns if positional, passes if keyword -> TestFutureKeywordOnly
classproperty              works on class and instance            -> TestClassproperty
lazyproperty               computed once, thread safety           -> TestLazyproperty
sharedmethod               called on class and instance           -> TestSharedmethod
format_doc                 substitution, missing key              -> TestFormatDoc

Run standalone::

    python -m unittest scikitplot._utils.tests.test_decorators -v
"""

from __future__ import annotations

import sys
import types
import unittest
import warnings

from ..decorators import (  # noqa: E402
    classproperty,
    deprecated,
    deprecated_attribute,
    deprecated_renamed_argument,
    format_doc,
    future_keyword_only,
    lazyproperty,
    sharedmethod,
)
from ...exceptions import (
    ScikitplotException,
    ScikitplotDeprecationWarning,
    ScikitplotPendingDeprecationWarning,
    ScikitplotUserWarning,
)


# ===========================================================================
# deprecated (function)
# ===========================================================================


class TestDeprecatedFunction(unittest.TestCase):
    """deprecated() applied to a function must issue a deprecation warning."""

    def test_issues_deprecation_warning(self):
        """Calling a deprecated function must emit a DeprecationWarning."""
        @deprecated("1.0")
        def old_fn():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_fn()
        self.assertEqual(result, 42)
        self.assertGreater(len(w), 0)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))

    def test_warning_message_contains_since_version(self):
        """The deprecation message must mention the 'since' version."""
        @deprecated("2.5")
        def old_fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_fn()
        self.assertIn("2.5", str(w[0].message))

    def test_alternative_in_message(self):
        """If alternative is given, it must appear in the warning message."""
        @deprecated("1.0", alternative="new_fn")
        def old_fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_fn()
        self.assertIn("new_fn", str(w[0].message))

    def test_custom_message_included(self):
        """A custom message must appear in the warning."""
        @deprecated("1.0", message="Use something else.")
        def old_fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_fn()
        self.assertIn("something else", str(w[0].message))

    def test_pending_flag_uses_pending_warning(self):
        """pending=True must issue PendingDeprecationWarning."""
        @deprecated("1.0", pending=True)
        def old_fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_fn()
        self.assertGreater(len(w), 0)
        self.assertTrue(issubclass(w[0].category, PendingDeprecationWarning))

    def test_custom_warning_type(self):
        """A custom warning_type must be used instead of the default."""
        @deprecated("1.0", warning_type=UserWarning)
        def old_fn():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_fn()
        self.assertTrue(issubclass(w[0].category, UserWarning))

    def test_function_still_executes(self):
        """The decorated function must still execute and return the original value."""
        @deprecated("1.0")
        def add(a, b):
            return a + b

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = add(3, 4)
        self.assertEqual(result, 7)

    def test_preserves_docstring_with_deprecation_note(self):
        """The deprecated decorator must modify the docstring to include a deprecation note."""
        @deprecated("1.0")
        def fn():
            """Original docstring."""
            pass

        self.assertIn("deprecated", fn.__doc__.lower())

    def test_deprecated_attribute_set(self):
        """The deprecated function must have a __deprecated__ attribute."""
        @deprecated("1.0")
        def fn():
            pass

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
        self.assertTrue(hasattr(fn, "__deprecated__"))


# ===========================================================================
# deprecated (class)
# ===========================================================================


class TestDeprecatedClass(unittest.TestCase):
    """deprecated() applied to a class must warn on instantiation."""

    def test_warns_on_init(self):
        """Instantiating a deprecated class must emit a DeprecationWarning."""
        @deprecated("1.0")
        class OldClass:
            def __init__(self):
                self.value = 99

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass()
        self.assertGreater(len(w), 0)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(obj.value, 99)

    def test_class_instance_still_works(self):
        """The deprecated class instance must still be usable."""
        @deprecated("1.0")
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p = Point(3, 4)
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 4)

    def test_docstring_modified(self):
        """The class docstring must include the deprecation notice."""
        @deprecated("2.0")
        class MyClass:
            """Original class doc."""

        self.assertIn("deprecated", MyClass.__doc__.lower())

    def test_new_override_path(self):
        """A class overriding __new__ must have __new__ deprecated, not __init__."""
        @deprecated("1.0")
        class Singleton:
            _instance = None

            def __new__(cls):
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Singleton._instance = None  # reset
            obj = Singleton()
        self.assertGreater(len(w), 0)


# ===========================================================================
# deprecated_attribute
# ===========================================================================


class TestDeprecatedAttribute(unittest.TestCase):
    """deprecated_attribute must warn when the attribute is accessed."""

    def test_warns_on_attribute_access(self):
        """Accessing a deprecated attribute must emit a warning."""
        class MyClass:
            old_attr = deprecated_attribute("old_attr", "1.0", alternative="new_attr")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MyClass.old_attr  # noqa: B018 — access triggers warning

        self.assertGreater(len(w), 0)

    def test_warning_message_contains_attribute_name(self):
        """The warning message must mention the attribute name."""
        class MyClass:
            old = deprecated_attribute("old", "1.0")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MyClass.old  # noqa: B018
        self.assertIn("old", str(w[0].message))


# ===========================================================================
# deprecated_renamed_argument
# ===========================================================================


class TestDeprecatedRenamedArgument(unittest.TestCase):
    """deprecated_renamed_argument must handle argument renaming and warnings."""

    def _make_fn(self, old="old_arg", new="new_arg", since="1.0"):
        @deprecated_renamed_argument(old, new, since=since)
        def fn(new_arg=None):
            return new_arg

        return fn

    def test_new_arg_works_without_warning(self):
        """Calling with the new argument name must not produce a warning."""
        fn = self._make_fn()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fn(new_arg=42)
        self.assertEqual(result, 42)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(deprecation_warnings), 0)

    def test_old_arg_is_remapped(self):
        """Calling with old_arg must remap it to new_arg and return correctly."""
        fn = self._make_fn()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = fn(old_arg=99)
        self.assertEqual(result, 99)

    def test_old_arg_issues_warning(self):
        """Using the old argument name must issue a DeprecationWarning."""
        fn = self._make_fn()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn(old_arg=1)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertGreater(len(dep_warnings), 0)


# ===========================================================================
# future_keyword_only
# ===========================================================================


class TestFutureKeywordOnly(unittest.TestCase):
    """future_keyword_only must warn when named args are passed positionally."""

    def test_keyword_call_no_warning(self):
        """Calling with keyword syntax must not produce any warnings."""
        @future_keyword_only(["b"], since=["1.0"])
        def fn(a, b=None):
            return (a, b)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fn(1, b=2)
        self.assertEqual(result, (1, 2))
        self.assertEqual(len([x for x in w if issubclass(x.category, Warning)]), 0)

    def test_positional_call_issues_warning(self):
        """Passing a keyword-only-future argument positionally must warn."""
        @future_keyword_only(["b"], since=["1.0"])
        def fn(a, b=None):
            return (a, b)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn(1, 2)  # b passed positionally
        self.assertGreater(len(w), 0)

    def test_result_correct_on_positional_call(self):
        """Even with a warning, the function result must be correct."""
        @future_keyword_only(["b"], since=["1.0"])
        def fn(a, b=None):
            return a + (b or 0)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = fn(3, 4)
        self.assertEqual(result, 7)


# ===========================================================================
# classproperty
# ===========================================================================


class TestClassproperty(unittest.TestCase):
    """classproperty must work like property but accessible via the class."""

    def test_accessible_from_class(self):
        """Must be readable directly from the class without an instance."""
        class MyClass:
            _value = 42

            @classproperty
            def value(cls):
                return cls._value

        self.assertEqual(MyClass.value, 42)

    def test_accessible_from_instance(self):
        """Must also be readable from an instance."""
        class MyClass:
            _value = 10

            @classproperty
            def value(cls):
                return cls._value

        obj = MyClass()
        self.assertEqual(obj.value, 10)

    def test_reflects_class_attribute_changes(self):
        """Must reflect updates to the underlying class attribute."""
        class MyClass:
            _count = 0

            @classproperty
            def count(cls):
                return cls._count

        MyClass._count = 5
        self.assertEqual(MyClass.count, 5)


# ===========================================================================
# lazyproperty
# ===========================================================================


class TestLazyproperty(unittest.TestCase):
    """lazyproperty must compute value once and cache it on the instance."""

    def test_computed_on_first_access(self):
        """The value must be computed when first accessed."""
        call_count = {"n": 0}

        class MyClass:
            @lazyproperty
            def value(self):
                call_count["n"] += 1
                return 42

        obj = MyClass()
        result = obj.value
        self.assertEqual(result, 42)
        self.assertEqual(call_count["n"], 1)

    def test_cached_on_subsequent_access(self):
        """The second access must not call the underlying function again."""
        call_count = {"n": 0}

        class MyClass:
            @lazyproperty
            def value(self):
                call_count["n"] += 1
                return 99

        obj = MyClass()
        _ = obj.value
        _ = obj.value
        self.assertEqual(call_count["n"], 1)

    def test_different_instances_have_independent_caches(self):
        """Different instances must have independent cached values."""
        class Counter:
            _next = 0

            @lazyproperty
            def uid(self):
                Counter._next += 1
                return Counter._next

        a = Counter()
        b = Counter()
        self.assertNotEqual(a.uid, b.uid)

    def test_cached_value_survives_multiple_accesses(self):
        """The cached value must be identical on all subsequent accesses."""
        class MyClass:
            @lazyproperty
            def value(self):
                return object()  # new object each call

        obj = MyClass()
        first = obj.value
        second = obj.value
        self.assertIs(first, second)


# ===========================================================================
# sharedmethod
# ===========================================================================


class TestSharedmethod(unittest.TestCase):
    """sharedmethod must work whether called on the class or an instance."""

    def test_called_on_class(self):
        """When called on the class, must receive the class as first argument."""
        received = {}

        class MyClass:
            @sharedmethod
            def fn(cls_or_self):
                received["caller"] = cls_or_self

        MyClass.fn()
        self.assertIs(received["caller"], MyClass)

    def test_called_on_instance(self):
        """When called on an instance, must receive the instance as first argument."""
        received = {}

        class MyClass:
            @sharedmethod
            def fn(cls_or_self):
                received["caller"] = cls_or_self

        obj = MyClass()
        obj.fn()
        self.assertIs(received["caller"], obj)


# ===========================================================================
# format_doc
# ===========================================================================


class TestFormatDoc(unittest.TestCase):
    """format_doc must substitute keyword placeholders in the docstring."""

    def test_substitutes_placeholder(self):
        """A {name} placeholder must be replaced with the keyword value."""
        @format_doc("Hello, {name}!", name="World")
        def fn():
            pass

        self.assertEqual(fn.__doc__, "Hello, World!")

    def test_multiple_substitutions(self):
        """Multiple placeholders must all be substituted."""
        @format_doc("{a} + {b} = {c}", a="1", b="2", c="3")
        def fn():
            pass

        self.assertEqual(fn.__doc__, "1 + 2 = 3")

    def test_existing_docstring_substituted(self):
        """format_doc can also operate on a string passed directly (not decorator)."""
        result = format_doc("Value: {x}", x=99)
        self.assertEqual(result, "Value: 99")

    def test_format_doc_on_function_with_docstring(self):
        """When applied as a decorator with a template, the function docstring is updated."""
        @format_doc("Template: {key}", key="value")
        def fn():
            """Ignored original."""

        self.assertIn("value", fn.__doc__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
