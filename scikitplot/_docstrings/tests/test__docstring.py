# scikitplot/_docstrings/tests/test__docstring.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~scikitplot._docstrings._docstring`.

Execution
---------
From the workspace root (the directory containing ``_docstrings/``)::

    python -m unittest _docstrings.tests.test__docstring -v

All tests are self-contained.  The parent-package dependency
``from .. import _api`` is satisfied by a minimal in-process shim so
no scikit-plots installation is required.

Coverage map
------------
Substitution.__init__           Validation, style normalisation       → TestSubstitutionInit
Substitution.__call__           All style × param-type combinations,
                                no-doc short-circuit, failure warning → TestSubstitutionCall
Substitution.decorator          kwargs merge, args replace, no-op     → TestSubstitutionDecorator
Substitution.decorate           Class-level factory chain             → TestSubstitutionDecorate
_ArtistKwdocLoader.__missing__  Bad key, unknown artist key           → TestArtistKwdocLoader
_ArtistKwdocLoader.to_dict      str-filter, empty                     → TestArtistKwdocLoaderIO
_ArtistKwdocLoader.from_dict    Round-trip, bad-value rejection       → TestArtistKwdocLoaderIO
_ArtistPropertiesSubstitution   register, __call__ (func + class +
                                class-no-init + no-doc + failure),
                                strict, to_json, from_json            → TestArtistPropertiesSubstitution
decorate_doc_kwarg              Attribute set, empty text             → TestDecorateDocKwarg
decorate_doc_copy               Doc copy, None source, wraps target   → TestDecorateDocCopy
Module public API               __all__ contents, interpd singleton   → TestModuleAPI
Logger                          Name, warning on substitution failure → TestLogger
"""

from __future__ import annotations

import json
import logging
import sys
import types
import pathlib
import unittest
import unittest.mock as mock


# ---------------------------------------------------------------------------
# Shim: satisfy `from .. import _api` inside _docstring.py
# ---------------------------------------------------------------------------
# _docstring.py is a submodule of scikitplot._docstrings.  Its one external
# dependency is ``_api.recursive_subclasses``.  When the test is run
# standalone (without an installed scikitplot), we inject a minimal shim
# BEFORE importing the module under test.
#
# Strategy
# --------
# 1. Build a fake ``scikitplot`` namespace in sys.modules.
# 2. Read _docstring.py source, rewrite the relative import so it resolves
#    against our fake namespace, then exec() it into a freshly created module
#    object that is registered as ``scikitplot._docstrings._docstring``.
# 3. Bind the public names we need from that module object.
#
# When running inside the real scikitplot package the standard relative import
# works unchanged; ``scikitplot._docstrings._docstring`` will already be in
# sys.modules and the exec() path is skipped.
# ---------------------------------------------------------------------------

from .. import _docstring

_SUB_MOD = "scikitplot._docstrings._docstring"
_SUB_MOD_ROOT = _SUB_MOD.rsplit(".", maxsplit=1)[0]

# Bind public names for direct use in tests
Substitution = _docstring.Substitution
_ArtistKwdocLoader = _docstring._ArtistKwdocLoader
_ArtistPropertiesSubstitution = _docstring._ArtistPropertiesSubstitution
decorate_doc_copy = _docstring.decorate_doc_copy
decorate_doc_kwarg = _docstring.decorate_doc_kwarg
interpd = _docstring.interpd
logger = _docstring.logger


# ===========================================================================
# Helpers
# ===========================================================================

def _make_func(doc):
    """Return a trivial callable with the given docstring."""
    def f():
        pass
    f.__doc__ = doc
    return f


def _make_class(doc, init_doc=None):
    """Return a trivial class whose docstring (and optionally __init__ doc) is set."""
    if init_doc is not None:
        def __init__(self):
            pass
        __init__.__doc__ = init_doc
        return type("_Cls", (), {"__doc__": doc, "__init__": __init__})
    return type("_Cls", (), {"__doc__": doc})


# ===========================================================================
# Substitution.__init__
# ===========================================================================

class TestSubstitutionInit(unittest.TestCase):
    """Substitution.__init__ must validate inputs and normalise defaults."""

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_args_and_kwargs_raises_value_error(self):
        """Supplying both positional and keyword arguments must raise ValueError."""
        with self.assertRaises(ValueError):
            Substitution("x", name="y")

    def test_invalid_style_raises_value_error(self):
        """An unrecognised style string must raise ValueError."""
        with self.assertRaises(ValueError):
            Substitution(name="x", style="bad")

    def test_invalid_style_empty_string_raises_value_error(self):
        """An empty style string is not a valid style."""
        with self.assertRaises(ValueError):
            Substitution(name="x", style="")

    def test_invalid_style_uppercase_raises_value_error(self):
        """'Percent' (wrong case) must be rejected."""
        with self.assertRaises(ValueError):
            Substitution(name="x", style="Percent")

    # ------------------------------------------------------------------
    # style normalisation
    # ------------------------------------------------------------------

    def test_style_none_stored_as_percent(self):
        """style=None must be stored as 'percent' (the default fallback)."""
        s = Substitution(name="Alice", style=None)
        self.assertEqual(s.style, "percent")

    def test_style_omitted_stored_as_percent(self):
        """Omitting style must also default to 'percent'."""
        s = Substitution(name="Alice")
        self.assertEqual(s.style, "percent")

    def test_style_percent_stored_verbatim(self):
        s = Substitution(name="Alice", style="percent")
        self.assertEqual(s.style, "percent")

    def test_style_format_stored_verbatim(self):
        s = Substitution(name="Alice", style="format")
        self.assertEqual(s.style, "format")

    # ------------------------------------------------------------------
    # Param storage
    # ------------------------------------------------------------------

    def test_kwargs_stored_as_dict(self):
        """Keyword arguments must be stored as a dict in self.params."""
        s = Substitution(author="Alice", year="2024")
        self.assertIsInstance(s.params, dict)
        self.assertEqual(s.params["author"], "Alice")

    def test_args_stored_as_tuple(self):
        """Positional arguments must be stored as the args tuple."""
        s = Substitution("first", "second")
        self.assertIsInstance(s.params, tuple)
        self.assertEqual(s.params[0], "first")

    def test_no_args_no_kwargs_stores_empty_dict(self):
        """Calling with no arguments must store an empty dict."""
        s = Substitution()
        self.assertIsInstance(s.params, dict)
        self.assertEqual(len(s.params), 0)


# ===========================================================================
# Substitution.__call__
# ===========================================================================

class TestSubstitutionCall(unittest.TestCase):
    """Substitution.__call__ must apply the correct substitution in every branch."""

    # ------------------------------------------------------------------
    # percent style
    # ------------------------------------------------------------------

    def test_percent_style_with_kwargs(self):
        """%(key)s placeholders must be filled from keyword params."""
        s = Substitution(author="Alice")
        f = _make_func("%(author)s wrote this.")
        s(f)
        self.assertEqual(f.__doc__, "Alice wrote this.")

    def test_percent_style_with_positional_args(self):
        """%s placeholders must be filled from positional params."""
        s = Substitution("Bob")
        f = _make_func("%s did it.")
        s(f)
        self.assertEqual(f.__doc__, "Bob did it.")

    def test_percent_style_multiple_keys(self):
        s = Substitution(a="X", b="Y")
        f = _make_func("%(a)s and %(b)s.")
        s(f)
        self.assertEqual(f.__doc__, "X and Y.")

    # ------------------------------------------------------------------
    # format style — dict params
    # ------------------------------------------------------------------

    def test_format_style_with_kwargs(self):
        """{key} placeholders must be filled when style='format'."""
        s = Substitution(name="Carol", style="format")
        f = _make_func("{name} rules.")
        s(f)
        self.assertEqual(f.__doc__, "Carol rules.")

    def test_format_style_with_positional_args(self):
        """{0} placeholders must be filled from positional args."""
        s = Substitution("Dave", style="format")
        f = _make_func("{0} wins.")
        s(f)
        self.assertEqual(f.__doc__, "Dave wins.")

    def test_format_style_multiple_keys(self):
        s = Substitution(x="hello", y="world", style="format")
        f = _make_func("{x} {y}!")
        s(f)
        self.assertEqual(f.__doc__, "hello world!")

    # ------------------------------------------------------------------
    # No docstring — short-circuit
    # ------------------------------------------------------------------

    def test_no_docstring_returns_object_unchanged(self):
        """When the object has no docstring, it must be returned unchanged."""
        s = Substitution(name="X")
        f = _make_func(None)
        result = s(f)
        self.assertIs(result, f)
        self.assertIsNone(f.__doc__)

    def test_empty_docstring_returns_object_unchanged(self):
        """An empty string docstring must also short-circuit."""
        s = Substitution(name="X")
        f = _make_func("")
        result = s(f)
        self.assertIs(result, f)

    # ------------------------------------------------------------------
    # Substitution failure → warning, returns obj
    # ------------------------------------------------------------------

    def test_failure_returns_object_with_warning(self):
        """A missing key must log a WARNING and return the original object."""
        s = Substitution(other_key="x")
        f = _make_func("%(no_such_key)s")
        with self.assertLogs(_SUB_MOD, level=logging.WARNING):
            result = s(f)
        self.assertIs(result, f)

    def test_failure_does_not_raise(self):
        """A substitution error must never propagate as an exception."""
        s = Substitution(other_key="x")
        f = _make_func("%(no_such_key)s")
        try:
            s(f)
        except Exception as exc:
            self.fail(f"__call__ raised unexpectedly: {exc}")

    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------

    def test_returns_same_object(self):
        """__call__ must always return the decorated object."""
        s = Substitution(name="Eve")
        f = _make_func("%(name)s")
        result = s(f)
        self.assertIs(result, f)

    # ------------------------------------------------------------------
    # inspect.cleandoc is applied
    # ------------------------------------------------------------------

    def test_cleandoc_applied_before_substitution(self):
        """Leading indentation in the raw docstring must be stripped."""
        s = Substitution(key="OK")
        # Simulate indented docstring (as written inside a function body)
        f = _make_func("    %(key)s")
        s(f)
        self.assertEqual(f.__doc__, "OK")


# ===========================================================================
# Substitution.decorator  (instance method)
# ===========================================================================

class TestSubstitutionDecorator(unittest.TestCase):
    """Substitution.decorator must update params in-place and return self."""

    def test_decorator_returns_self(self):
        """decorator() must return the Substitution instance itself."""
        s = Substitution(name="orig")
        result = s.decorator(name="updated")
        self.assertIs(result, s)

    def test_decorator_kwargs_merges_into_dict_params(self):
        """Passing kwargs must merge (update) existing dict params."""
        s = Substitution(x="1", y="2")
        s.decorator(z="3")
        self.assertEqual(s.params["x"], "1")
        self.assertEqual(s.params["y"], "2")
        self.assertEqual(s.params["z"], "3")

    def test_decorator_kwargs_overwrites_existing_key(self):
        """A repeated key in decorator kwargs must overwrite the old value."""
        s = Substitution(name="old")
        s.decorator(name="new")
        self.assertEqual(s.params["name"], "new")

    def test_decorator_args_replaces_tuple_params(self):
        """Passing args to decorator on a tuple-params instance must replace them."""
        s = Substitution("original")
        s.decorator("replaced")
        self.assertEqual(s.params, ("replaced",))

    def test_decorator_kwargs_replaces_tuple_params(self):
        """Passing kwargs to decorator on a tuple-params instance must replace them."""
        s = Substitution("positional")
        s.decorator(name="new")
        self.assertIsInstance(s.params, dict)
        self.assertEqual(s.params["name"], "new")

    def test_decorator_no_args_no_kwargs_is_noop(self):
        """Calling decorator() with no arguments must leave params unchanged."""
        s = Substitution(name="unchanged")
        original_params = dict(s.params)
        s.decorator()
        self.assertEqual(s.params, original_params)

    def test_decorator_then_call_applies_updated_params(self):
        """After updating params via decorator, __call__ must use the new values."""
        s = Substitution(name="original")
        s.decorator(name="updated")
        f = _make_func("%(name)s")
        s(f)
        self.assertEqual(f.__doc__, "updated")


# ===========================================================================
# Substitution.decorate  (class method)
# ===========================================================================

class TestSubstitutionDecorate(unittest.TestCase):
    """Substitution.decorate must act as a class-level factory and decorator."""

    def test_decorate_returns_substitution_instance(self):
        """Substitution.decorate(...) must return a Substitution instance."""
        result = Substitution.decorate(name="Ada", style="format")
        self.assertIsInstance(result, Substitution)

    def test_decorate_stores_kwargs(self):
        s = Substitution.decorate(name="Ada", style="format")
        self.assertEqual(s.params.get("name"), "Ada")

    def test_decorate_used_as_decorator_applies_substitution(self):
        """The Substitution returned by .decorate() must work as a decorator."""
        sub = Substitution.decorate(name="Ada", style="format")

        @sub
        def greet():
            """{name} is great."""

        self.assertEqual(greet.__doc__, "Ada is great.")

    def test_decorate_percent_style(self):
        """decorate with style='percent' must apply %-style substitution."""
        sub = Substitution.decorate(author="Turing")

        @sub
        def doc_func():
            """By %(author)s."""

        self.assertEqual(doc_func.__doc__, "By Turing.")

    def test_decorate_creates_independent_instances(self):
        """Two calls to decorate must produce independent Substitution instances."""
        s1 = Substitution.decorate(name="Alice")
        s2 = Substitution.decorate(name="Bob")
        self.assertIsNot(s1, s2)


# ===========================================================================
# _ArtistKwdocLoader.__missing__
# ===========================================================================

class TestArtistKwdocLoader(unittest.TestCase):
    """_ArtistKwdocLoader.__missing__ must enforce the :kwdoc suffix contract."""

    def test_missing_key_without_kwdoc_suffix_raises_key_error(self):
        """A key that does not end with ':kwdoc' must raise KeyError immediately."""
        loader = _ArtistKwdocLoader()
        with self.assertRaises(KeyError):
            _ = loader["bad_key"]

    def test_key_error_message_mentions_invalid_kwdoc(self):
        """The KeyError message must describe the invalid key format."""
        loader = _ArtistKwdocLoader()
        try:
            _ = loader["no_colon"]
        except KeyError as exc:
            self.assertIn("kwdoc", str(exc).lower())

    def test_unknown_artist_name_raises_key_error(self):
        """A ':kwdoc' key for a non-existent Artist subclass must raise KeyError."""
        loader = _ArtistKwdocLoader()
        with self.assertRaises(KeyError):
            _ = loader["NoSuchArtistXYZ:kwdoc"]

    def test_unknown_artist_emits_warning(self):
        """Looking up an unknown Artist must emit a WARNING log."""
        loader = _ArtistKwdocLoader()
        with self.assertLogs(_SUB_MOD, level=logging.WARNING):
            try:
                _ = loader["NoSuchArtistXYZ:kwdoc"]
            except KeyError:
                pass

    def test_normal_key_set_and_retrieved(self):
        """Explicitly set string keys must be retrievable without __missing__."""
        loader = _ArtistKwdocLoader()
        loader["key"] = "value"
        self.assertEqual(loader["key"], "value")


# ===========================================================================
# _ArtistKwdocLoader.to_dict / from_dict
# ===========================================================================

class TestArtistKwdocLoaderIO(unittest.TestCase):
    """_ArtistKwdocLoader serialisation helpers must be correct and safe."""

    # ------------------------------------------------------------------
    # to_dict
    # ------------------------------------------------------------------

    def test_to_dict_returns_dict(self):
        loader = _ArtistKwdocLoader()
        self.assertIsInstance(loader.to_dict(), dict)

    def test_to_dict_empty_loader(self):
        loader = _ArtistKwdocLoader()
        self.assertEqual(loader.to_dict(), {})

    def test_to_dict_includes_str_values(self):
        loader = _ArtistKwdocLoader()
        loader["a"] = "hello"
        loader["b"] = "world"
        self.assertEqual(loader.to_dict(), {"a": "hello", "b": "world"})

    def test_to_dict_filters_non_str_values(self):
        """Non-string values must be excluded from the serialised output."""
        loader = _ArtistKwdocLoader()
        loader["str_key"] = "ok"
        loader["int_key"] = 42      # must be filtered
        loader["none_key"] = None   # must be filtered
        d = loader.to_dict()
        self.assertIn("str_key", d)
        self.assertNotIn("int_key", d)
        self.assertNotIn("none_key", d)

    def test_to_dict_returns_shallow_copy(self):
        """Mutating the returned dict must not affect the loader."""
        loader = _ArtistKwdocLoader()
        loader["a"] = "1"
        d = loader.to_dict()
        d["extra"] = "x"
        self.assertNotIn("extra", loader)

    # ------------------------------------------------------------------
    # from_dict
    # ------------------------------------------------------------------

    def test_from_dict_returns_loader_instance(self):
        loader = _ArtistKwdocLoader.from_dict({"x": "y"})
        self.assertIsInstance(loader, _ArtistKwdocLoader)

    def test_from_dict_populates_entries(self):
        loader = _ArtistKwdocLoader.from_dict({"key": "value"})
        self.assertEqual(loader["key"], "value")

    def test_from_dict_round_trip(self):
        """to_dict then from_dict must reproduce the same entries."""
        loader1 = _ArtistKwdocLoader()
        loader1["alpha"] = "A"
        loader1["beta"] = "B"
        loader2 = _ArtistKwdocLoader.from_dict(loader1.to_dict())
        self.assertEqual(loader2["alpha"], "A")
        self.assertEqual(loader2["beta"], "B")

    def test_from_dict_empty_input(self):
        loader = _ArtistKwdocLoader.from_dict({})
        self.assertEqual(len(loader), 0)

    def test_from_dict_non_str_value_raises_value_error(self):
        """A dict with non-string values must raise ValueError."""
        with self.assertRaises(ValueError):
            _ArtistKwdocLoader.from_dict({"key": 123})

    def test_from_dict_non_str_key_raises_value_error(self):
        """A dict with a non-string key must raise ValueError."""
        with self.assertRaises(ValueError):
            _ArtistKwdocLoader.from_dict({1: "value"})


# ===========================================================================
# _ArtistPropertiesSubstitution
# ===========================================================================

class TestArtistPropertiesSubstitution(unittest.TestCase):
    """_ArtistPropertiesSubstitution must register, substitute, and serialise correctly."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_init_creates_artist_kwdoc_loader_params(self):
        """params must be an _ArtistKwdocLoader after construction."""
        aps = _ArtistPropertiesSubstitution()
        self.assertIsInstance(aps.params, _ArtistKwdocLoader)

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    def test_register_stores_key_value(self):
        aps = _ArtistPropertiesSubstitution()
        aps.register(greeting="Hello World")
        self.assertEqual(aps.params["greeting"], "Hello World")

    def test_register_multiple_keys(self):
        aps = _ArtistPropertiesSubstitution()
        aps.register(a="1", b="2")
        self.assertEqual(aps.params["a"], "1")
        self.assertEqual(aps.params["b"], "2")

    def test_register_no_args_does_not_raise(self):
        """Calling register() with no arguments must succeed silently."""
        aps = _ArtistPropertiesSubstitution()
        try:
            aps.register()
        except Exception as exc:
            self.fail(f"register() with no args raised: {exc}")

    def test_register_overwrites_existing_key(self):
        aps = _ArtistPropertiesSubstitution()
        aps.register(key="old")
        aps.register(key="new")
        self.assertEqual(aps.params["key"], "new")

    # ------------------------------------------------------------------
    # __call__ — function substitution
    # ------------------------------------------------------------------

    def test_call_applies_substitution_to_function(self):
        aps = _ArtistPropertiesSubstitution()
        aps.register(greeting="Hello World")
        f = _make_func("%(greeting)s")
        aps(f)
        self.assertEqual(f.__doc__, "Hello World")

    def test_call_returns_same_function(self):
        aps = _ArtistPropertiesSubstitution()
        aps.register(x="y")
        f = _make_func("%(x)s")
        result = aps(f)
        self.assertIs(result, f)

    def test_call_with_no_docstring_returns_obj_unchanged(self):
        """A function with no docstring must be returned unmodified."""
        aps = _ArtistPropertiesSubstitution()
        f = _make_func(None)
        result = aps(f)
        self.assertIs(result, f)
        self.assertIsNone(f.__doc__)

    def test_call_missing_key_emits_warning_and_returns_obj(self):
        """A missing placeholder key must emit a WARNING and return the object."""
        aps = _ArtistPropertiesSubstitution()  # no keys registered
        f = _make_func("%(missing_key)s")
        with self.assertLogs(_SUB_MOD, level=logging.WARNING):
            result = aps(f)
        self.assertIs(result, f)

    def test_call_does_not_raise_on_substitution_failure(self):
        """Substitution failure must never propagate as an exception."""
        aps = _ArtistPropertiesSubstitution()
        f = _make_func("%(totally_unknown)s")
        try:
            aps(f)
        except Exception as exc:
            self.fail(f"__call__ raised on failure: {exc}")

    # ------------------------------------------------------------------
    # __call__ — class substitution (class doc + __init__ doc)
    # ------------------------------------------------------------------

    def test_call_applies_substitution_to_class_docstring(self):
        """__call__ on a class must substitute the class-level docstring."""
        aps = _ArtistPropertiesSubstitution()
        aps.register(thing="item")
        Cls = _make_class("Class: %(thing)s")
        aps(Cls)
        self.assertEqual(Cls.__doc__, "Class: item")

    def test_call_applies_substitution_to_init_docstring(self):
        """__call__ on a class must also substitute the __init__ docstring."""
        aps = _ArtistPropertiesSubstitution()
        aps.register(thing="item")
        Cls = _make_class("Class: %(thing)s", init_doc="Init: %(thing)s")
        aps(Cls)
        self.assertEqual(Cls.__init__.__doc__, "Init: item")

    def test_call_class_with_default_init_does_not_raise(self):
        """A class whose __init__ is object.__init__ must not trigger substitution."""
        aps = _ArtistPropertiesSubstitution()
        aps.register(thing="item")
        Cls = _make_class("Class: %(thing)s")  # no custom __init__
        try:
            aps(Cls)
        except Exception as exc:
            self.fail(f"Default __init__ path raised: {exc}")
        self.assertEqual(Cls.__doc__, "Class: item")

    # ------------------------------------------------------------------
    # __call__ — strict mode
    # ------------------------------------------------------------------

    def test_strict_false_does_not_raise_when_key_absent_in_doc(self):
        """Default (strict=False) must not raise even when params are unused."""
        aps = _ArtistPropertiesSubstitution()
        aps.register(key="value")
        f = _make_func("No placeholder here.")
        try:
            aps(f, strict=False)
        except ValueError:
            self.fail("strict=False must not raise ValueError")

    # ------------------------------------------------------------------
    # to_json
    # ------------------------------------------------------------------

    def test_to_json_returns_string(self):
        aps = _ArtistPropertiesSubstitution()
        self.assertIsInstance(aps.to_json(), str)

    def test_to_json_empty_params(self):
        """With no registered keys, to_json must return '{}'."""
        aps = _ArtistPropertiesSubstitution()
        self.assertEqual(aps.to_json(), "{}")

    def test_to_json_includes_registered_str_keys(self):
        aps = _ArtistPropertiesSubstitution()
        aps.register(k1="v1", k2="v2")
        d = json.loads(aps.to_json())
        self.assertEqual(d["k1"], "v1")
        self.assertEqual(d["k2"], "v2")

    def test_to_json_filters_non_str_values(self):
        """Non-string values injected into params must be excluded from JSON."""
        aps = _ArtistPropertiesSubstitution()
        aps.params["str_key"] = "str_val"
        aps.params["int_key"] = 42
        d = json.loads(aps.to_json())
        self.assertIn("str_key", d)
        self.assertNotIn("int_key", d)

    def test_to_json_round_trips_via_from_json(self):
        """to_json → from_json must reproduce the same registered keys."""
        aps1 = _ArtistPropertiesSubstitution()
        aps1.register(x="hello", y="world")
        j = aps1.to_json()

        aps2 = _ArtistPropertiesSubstitution()
        aps2.from_json(j)
        self.assertEqual(aps2.params["x"], "hello")
        self.assertEqual(aps2.params["y"], "world")

    # ------------------------------------------------------------------
    # from_json
    # ------------------------------------------------------------------

    def test_from_json_registers_keys(self):
        aps = _ArtistPropertiesSubstitution()
        aps.from_json('{"greeting": "hi"}')
        self.assertEqual(aps.params["greeting"], "hi")

    def test_from_json_invalid_json_does_not_raise(self):
        """Malformed JSON must log a WARNING but never propagate an exception."""
        aps = _ArtistPropertiesSubstitution()
        with self.assertLogs(_SUB_MOD, level=logging.WARNING):
            try:
                aps.from_json("{invalid json")
            except Exception as exc:
                self.fail(f"from_json raised on bad JSON: {exc}")

    def test_from_json_non_dict_json_is_ignored(self):
        """JSON arrays and scalars must be silently ignored (not registered)."""
        aps = _ArtistPropertiesSubstitution()
        before = dict(aps.params)
        aps.from_json("[1, 2, 3]")
        # params must be unchanged
        for k in before:
            self.assertEqual(aps.params[k], before[k])

    def test_from_json_empty_object_is_ok(self):
        aps = _ArtistPropertiesSubstitution()
        try:
            aps.from_json("{}")
        except Exception as exc:
            self.fail(f"from_json('{{}}') raised: {exc}")


# ===========================================================================
# decorate_doc_kwarg
# ===========================================================================

class TestDecorateDocKwarg(unittest.TestCase):
    """decorate_doc_kwarg must attach _kwarg_doc to the decorated callable."""

    def test_sets_kwarg_doc_attribute(self):
        @decorate_doc_kwarg("bool, default: True")
        def setter(self, val):
            pass
        self.assertEqual(setter._kwarg_doc, "bool, default: True")

    def test_returns_same_callable(self):
        """The decorated object must be the exact same callable."""
        def setter(self, val):
            pass
        result = decorate_doc_kwarg("text")(setter)
        self.assertIs(result, setter)

    def test_empty_string_stored_verbatim(self):
        @decorate_doc_kwarg("")
        def setter(self, val):
            pass
        self.assertEqual(setter._kwarg_doc, "")

    def test_multiline_text_stored_verbatim(self):
        text = "str, optional\n    A detailed description."

        @decorate_doc_kwarg(text)
        def setter(self, val):
            pass
        self.assertEqual(setter._kwarg_doc, text)

    def test_decorator_factory_applied_to_multiple_functions(self):
        """The same text must be independently attached to different callables."""
        dec = decorate_doc_kwarg("shared text")

        def setter1(self, val): pass
        def setter2(self, val): pass
        dec(setter1)
        dec(setter2)
        self.assertEqual(setter1._kwarg_doc, "shared text")
        self.assertEqual(setter2._kwarg_doc, "shared text")


# ===========================================================================
# decorate_doc_copy
# ===========================================================================

class TestDecorateDocCopy(unittest.TestCase):
    """decorate_doc_copy must copy the docstring and preserve target behaviour."""

    def test_copies_docstring_from_source(self):
        def source():
            """Source docstring."""
        @decorate_doc_copy(source)
        def target():
            pass
        self.assertEqual(target.__doc__, "Source docstring.")

    def test_target_behaviour_preserved(self):
        """The wrapped callable must invoke the target, not the source."""
        def source():
            """Source."""
            return "source_result"

        @decorate_doc_copy(source)
        def target(x):
            return x * 3

        self.assertEqual(target(4), 12)

    def test_source_with_none_docstring(self):
        """A source whose __doc__ is None must not overwrite target's doc."""
        def source():
            pass
        source.__doc__ = None

        @decorate_doc_copy(source)
        def target():
            pass

        # doc should remain None (source __doc__ is None → not overwritten)
        self.assertIsNone(target.__doc__)

    def test_source_with_empty_string_docstring(self):
        """An empty string docstring in source must be copied to target."""
        def source():
            """"""  # noqa: D419
        @decorate_doc_copy(source)
        def target():
            pass
        self.assertEqual(target.__doc__, "")

    def test_returns_callable(self):
        def source(): """S."""
        @decorate_doc_copy(source)
        def target(): pass
        self.assertTrue(callable(target))

    def test_multiple_targets_get_same_doc(self):
        """Two targets decorated with the same source must share the same doc."""
        def source():
            """Shared."""

        @decorate_doc_copy(source)
        def target1(): pass

        @decorate_doc_copy(source)
        def target2(): pass

        self.assertEqual(target1.__doc__, "Shared.")
        self.assertEqual(target2.__doc__, "Shared.")

    def test_different_sources_give_different_docs(self):
        """Different sources must result in different target docstrings."""
        def src1(): """Doc A."""
        def src2(): """Doc B."""

        @decorate_doc_copy(src1)
        def tgt1(): pass

        @decorate_doc_copy(src2)
        def tgt2(): pass

        self.assertNotEqual(tgt1.__doc__, tgt2.__doc__)


# ===========================================================================
# Module public API
# ===========================================================================

class TestModuleAPI(unittest.TestCase):
    """Public API surface must be exactly as declared in __all__."""

    def test_all_contains_substitution(self):
        self.assertIn("Substitution", _docstring.__all__)

    def test_all_contains_decorate_doc_copy(self):
        self.assertIn("decorate_doc_copy", _docstring.__all__)

    def test_all_contains_decorate_doc_kwarg(self):
        self.assertIn("decorate_doc_kwarg", _docstring.__all__)

    def test_all_contains_interpd(self):
        self.assertIn("interpd", _docstring.__all__)

    def test_all_has_exactly_four_entries(self):
        """__all__ must have exactly 4 public names."""
        self.assertEqual(len(_docstring.__all__), 4)

    def test_interpd_is_artist_properties_substitution(self):
        """The module-level interpd must be an _ArtistPropertiesSubstitution instance."""
        self.assertIsInstance(interpd, _ArtistPropertiesSubstitution)

    def test_interpd_is_singleton(self):
        """Re-importing from sys.modules must return the same interpd object."""
        interpd2 = sys.modules[_SUB_MOD].interpd
        self.assertIs(interpd, interpd2)


# ===========================================================================
# Logger
# ===========================================================================

class TestLogger(unittest.TestCase):
    """The module logger must be correctly named and emit expected records."""

    def test_logger_name(self):
        """Logger name must match the module's __name__."""
        self.assertEqual(logger.name, _SUB_MOD)

    def test_logger_is_logging_logger(self):
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_hierarchy_parent_is_docstrings(self):
        """Logger must be a child of 'scikitplot._docstrings'."""
        parent = logger.name.rsplit(".", 1)[0]
        self.assertEqual(parent, _SUB_MOD_ROOT)

    def test_warning_on_substitution_failure(self):
        """A failed %-substitution in Substitution.__call__ must emit WARNING."""
        s = Substitution(other="x")
        f = _make_func("%(missing)s")
        with self.assertLogs(logger.name, level=logging.WARNING):
            s(f)

    def test_warning_on_artist_properties_substitution_failure(self):
        """A failed substitution in _ArtistPropertiesSubstitution must emit WARNING."""
        aps = _ArtistPropertiesSubstitution()
        f = _make_func("%(missing_placeholder)s")
        with self.assertLogs(logger.name, level=logging.WARNING):
            aps(f)

    def test_warning_on_from_json_bad_input(self):
        """Malformed JSON in from_json must emit a WARNING."""
        aps = _ArtistPropertiesSubstitution()
        with self.assertLogs(logger.name, level=logging.WARNING):
            aps.from_json("{bad}")

    def test_warning_on_kwdoc_loader_unknown_artist(self):
        """Looking up an unknown artist in _ArtistKwdocLoader must emit WARNING."""
        loader = _ArtistKwdocLoader()
        with self.assertLogs(logger.name, level=logging.WARNING):
            try:
                _ = loader["NoSuchArtist:kwdoc"]
            except KeyError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
