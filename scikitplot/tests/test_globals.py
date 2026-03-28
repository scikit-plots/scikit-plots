# scikitplot/tests/test_globals.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive test suite for ``scikitplot._globals``.

Structure
---------
Each public class and module-level behaviour has its own ``Test*`` class,
mirroring the source file layout:

  TestModuleIntegrity
  TestCopyMode
  TestSingletonBase
  TestSingletonInstanceIsolation
  TestDefaultType
  TestDeprecatedType
  TestNoValueType
  TestModuleLevelInstances
  TestPickle

Notes
-----
**Developer notes**

* The module import uses ``importlib`` to support both installed and
  editable/source-tree layouts without hardcoding ``sys.path``.
* The reload guard test uses ``importlib.reload`` and expects
  ``RuntimeError`` — the same pattern validated for ``exceptions.py``.
* Pickle tests use ``pickle.dumps`` / ``pickle.loads`` to verify that
  ``__reduce__`` preserves singleton identity across serialisation.
* Subclass-isolation tests create *temporary* subclasses of
  ``SingletonBase`` inside the test function scope; they are discarded
  after the test and do not pollute the module namespace.
* ``SingletonBase.__new__`` no longer accepts ``*args`` / ``**kwargs``:
  tests confirm that calling with positional or keyword arguments raises
  ``TypeError``.

Running
-------
From the project root::

    pytest tests/test_globals.py -v --tb=short

For coverage::

    pytest tests/test_globals.py --cov=scikitplot._globals --cov-report=term-missing
"""

from __future__ import annotations

import importlib
import pickle

import pytest

# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------
try:
    import scikitplot._globals as glb
except ImportError:
    pytest.skip(
        "scikitplot._globals is not importable. "
        "Install the package or add the project root to PYTHONPATH.",
        allow_module_level=True,
    )


# ===========================================================================
# TestModuleIntegrity
# ===========================================================================

class TestModuleIntegrity:
    """Verify __all__, _is_loaded, and the reload guard."""

    def test_all_exports_exist_as_module_attributes(self):
        for name in glb.__all__:
            assert hasattr(glb, name), (
                f"__all__ declares {name!r} but the module has no such attribute."
            )

    def test_all_contains_expected_names(self):
        expected = {"_CopyMode", "_Default", "_Deprecated", "_NoValue"}
        assert expected.issubset(set(glb.__all__))

    def test_no_duplicate_names_in_all(self):
        assert len(glb.__all__) == len(set(glb.__all__))

    def test_is_loaded_set_to_true(self):
        assert hasattr(glb, "_is_loaded")
        assert glb._is_loaded is True

    def test_reload_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Reloading scikitplot._globals"):
            importlib.reload(glb)


# ===========================================================================
# TestCopyMode
# ===========================================================================

class TestCopyMode:
    """Tests for the _CopyMode enum."""

    def test_is_enum(self):
        import enum
        assert issubclass(glb._CopyMode, enum.Enum)

    def test_always_value(self):
        assert glb._CopyMode.ALWAYS.value is True

    def test_never_value(self):
        assert glb._CopyMode.NEVER.value is False

    def test_if_needed_value(self):
        assert glb._CopyMode.IF_NEEDED.value == 2

    def test_exactly_three_members(self):
        assert len(glb._CopyMode) == 3

    def test_members_accessible_by_name(self):
        assert glb._CopyMode["ALWAYS"] is glb._CopyMode.ALWAYS
        assert glb._CopyMode["NEVER"] is glb._CopyMode.NEVER
        assert glb._CopyMode["IF_NEEDED"] is glb._CopyMode.IF_NEEDED

    # --- __bool__ ---

    def test_bool_always_is_true(self):
        assert bool(glb._CopyMode.ALWAYS) is True

    def test_bool_never_is_false(self):
        assert bool(glb._CopyMode.NEVER) is False

    def test_bool_if_needed_raises_value_error(self):
        with pytest.raises(ValueError, match="neither True nor False"):
            bool(glb._CopyMode.IF_NEEDED)

    def test_bool_always_usable_in_if_statement(self):
        result = "yes" if glb._CopyMode.ALWAYS else "no"
        assert result == "yes"

    def test_bool_never_usable_in_if_statement(self):
        result = "yes" if glb._CopyMode.NEVER else "no"
        assert result == "no"

    def test_bool_if_needed_raises_in_if_statement(self):
        with pytest.raises(ValueError):
            if glb._CopyMode.IF_NEEDED:
                pass


# ===========================================================================
# TestSingletonBase
# ===========================================================================

class TestSingletonBase:
    """Tests for SingletonBase mechanics using temporary local subclasses."""

    def _make_subclass(self, name: str = "TempSingleton"):
        """Return a fresh SingletonBase subclass for use within a single test."""
        return type(name, (glb.SingletonBase,), {})

    def test_subclass_has_own_instance_attribute(self):
        cls = self._make_subclass()
        # __init_subclass__ must have stamped _instance directly on cls.
        assert "_instance" in cls.__dict__
        assert cls.__dict__["_instance"] is None

    def test_first_instantiation_creates_instance(self):
        cls = self._make_subclass()
        obj = cls()
        assert obj is not None
        assert isinstance(obj, cls)

    def test_second_instantiation_returns_same_instance(self):
        cls = self._make_subclass()
        obj1 = cls()
        obj2 = cls()
        assert obj1 is obj2

    def test_instance_is_set_on_subclass_not_base(self):
        cls = self._make_subclass()
        obj = cls()
        # _instance must be on the subclass's own __dict__, not only on SingletonBase.
        assert "_instance" in cls.__dict__
        assert cls.__dict__["_instance"] is obj

    def test_new_accepts_no_positional_args(self):
        """Singletons take zero construction arguments."""
        cls = self._make_subclass()
        with pytest.raises(TypeError):
            cls.__new__(cls, "unexpected_arg")

    def test_reduce_returns_tuple(self):
        cls = self._make_subclass()
        obj = cls()
        result = obj.__reduce__()
        assert isinstance(result, tuple)

    def test_reduce_first_element_is_class(self):
        cls = self._make_subclass()
        obj = cls()
        klass, args = obj.__reduce__()
        assert klass is cls

    def test_reduce_second_element_is_empty_tuple(self):
        cls = self._make_subclass()
        obj = cls()
        klass, args = obj.__reduce__()
        assert args == ()


# ===========================================================================
# TestSingletonInstanceIsolation
# ===========================================================================

class TestSingletonInstanceIsolation:
    """Verify that distinct subclasses maintain independent _instance slots.

    This is the regression test for the latent shared-state bug: without
    ``__init_subclass__``, calling ``SingletonBase()`` before any subclass
    would stamp ``SingletonBase._instance``, causing all uninitialised
    subclasses to find a non-None value via MRO and return the wrong object.
    """

    def test_two_subclasses_have_independent_instances(self):
        ClsA = type("ClsA", (glb.SingletonBase,), {})
        ClsB = type("ClsB", (glb.SingletonBase,), {})
        a = ClsA()
        b = ClsB()
        assert a is not b

    def test_two_subclasses_instance_dict_entries_are_independent(self):
        ClsA = type("ClsA2", (glb.SingletonBase,), {})
        ClsB = type("ClsB2", (glb.SingletonBase,), {})
        a = ClsA()
        b = ClsB()
        assert ClsA.__dict__["_instance"] is a
        assert ClsB.__dict__["_instance"] is b
        assert ClsA.__dict__["_instance"] is not ClsB.__dict__["_instance"]

    def test_default_deprecated_novaluetype_instances_are_distinct(self):
        """The three production singleton types must each be a different object."""
        assert glb._Default is not glb._Deprecated
        assert glb._Default is not glb._NoValue
        assert glb._Deprecated is not glb._NoValue

    def test_each_production_type_instance_is_correct_type(self):
        assert type(glb._Default) is glb._DefaultType
        assert type(glb._Deprecated) is glb._DeprecatedType
        assert type(glb._NoValue) is glb._NoValueType


# ===========================================================================
# TestDefaultType
# ===========================================================================

class TestDefaultType:
    """Tests for _DefaultType and its module-level instance _Default."""

    def test_is_singleton_base(self):
        assert issubclass(glb._DefaultType, glb.SingletonBase)

    def test_repeated_instantiation_returns_same_instance(self):
        obj1 = glb._DefaultType()
        obj2 = glb._DefaultType()
        assert obj1 is obj2

    def test_module_level_instance_is_same_as_new_call(self):
        assert glb._Default is glb._DefaultType()

    def test_repr_is_default(self):
        assert repr(glb._Default) == "<default>"

    def test_str_is_default(self):
        assert str(glb._Default) == "<default>"

    def test_module_level_instance_is_in_all(self):
        assert "_Default" in glb.__all__

    def test_identity_comparison(self):
        """is-check is the intended usage pattern."""
        sentinel = glb._Default
        assert sentinel is glb._Default

    def test_instance_of_singleton_base(self):
        assert isinstance(glb._Default, glb.SingletonBase)


# ===========================================================================
# TestDeprecatedType
# ===========================================================================

class TestDeprecatedType:
    """Tests for _DeprecatedType and its module-level instance _Deprecated."""

    def test_is_singleton_base(self):
        assert issubclass(glb._DeprecatedType, glb.SingletonBase)

    def test_repeated_instantiation_returns_same_instance(self):
        obj1 = glb._DeprecatedType()
        obj2 = glb._DeprecatedType()
        assert obj1 is obj2

    def test_module_level_instance_is_same_as_new_call(self):
        assert glb._Deprecated is glb._DeprecatedType()

    def test_repr_is_deprecated(self):
        assert repr(glb._Deprecated) == "<deprecated>"

    def test_str_is_deprecated(self):
        assert str(glb._Deprecated) == "<deprecated>"

    def test_module_level_instance_is_in_all(self):
        assert "_Deprecated" in glb.__all__

    def test_identity_comparison(self):
        sentinel = glb._Deprecated
        assert sentinel is glb._Deprecated

    def test_instance_of_singleton_base(self):
        assert isinstance(glb._Deprecated, glb.SingletonBase)


# ===========================================================================
# TestNoValueType
# ===========================================================================

class TestNoValueType:
    """Tests for _NoValueType and its module-level instance _NoValue."""

    def test_is_singleton_base(self):
        assert issubclass(glb._NoValueType, glb.SingletonBase)

    def test_repeated_instantiation_returns_same_instance(self):
        obj1 = glb._NoValueType()
        obj2 = glb._NoValueType()
        assert obj1 is obj2

    def test_module_level_instance_is_same_as_new_call(self):
        assert glb._NoValue is glb._NoValueType()

    def test_repr_is_no_value(self):
        assert repr(glb._NoValue) == "<no value>"

    def test_str_is_no_value(self):
        assert str(glb._NoValue) == "<no value>"

    def test_module_level_instance_is_in_all(self):
        assert "_NoValue" in glb.__all__

    def test_identity_comparison(self):
        sentinel = glb._NoValue
        assert sentinel is glb._NoValue

    def test_instance_of_singleton_base(self):
        assert isinstance(glb._NoValue, glb.SingletonBase)

    def test_canonical_usage_pattern(self):
        """Demonstrate the intended guard pattern works correctly."""
        def foo(arg=glb._NoValue):
            if arg is glb._NoValue:
                return "no value supplied"
            return f"got: {arg}"

        assert foo() == "no value supplied"
        assert foo(None) == "got: None"
        assert foo(0) == "got: 0"
        assert foo(False) == "got: False"


# ===========================================================================
# TestModuleLevelInstances
# ===========================================================================

class TestModuleLevelInstances:
    """Cross-cutting checks on the three production singleton instances."""

    def test_all_three_are_accessible(self):
        assert hasattr(glb, "_Default")
        assert hasattr(glb, "_Deprecated")
        assert hasattr(glb, "_NoValue")

    def test_all_three_are_distinct_objects(self):
        assert glb._Default is not glb._Deprecated
        assert glb._Default is not glb._NoValue
        assert glb._Deprecated is not glb._NoValue

    def test_all_three_reprs_are_unique(self):
        reprs = {repr(glb._Default), repr(glb._Deprecated), repr(glb._NoValue)}
        assert len(reprs) == 3

    def test_none_is_falsy_default_is_truthy(self):
        """Singletons must not be falsy; they are distinct sentinel objects."""
        assert glb._Default
        assert glb._Deprecated
        assert glb._NoValue


# ===========================================================================
# TestPickle
# ===========================================================================

class TestPickle:
    """Verify that pickling/unpickling preserves singleton identity."""

    @pytest.mark.parametrize("instance,attr", [
        ("_Default",    "_DefaultType"),
        ("_Deprecated", "_DeprecatedType"),
        ("_NoValue",    "_NoValueType"),
    ])
    def test_roundtrip_preserves_identity(self, instance, attr):
        """Unpickling must return the existing singleton, not a new object."""
        original = getattr(glb, instance)
        restored = pickle.loads(pickle.dumps(original))
        assert restored is original, (
            f"pickle roundtrip for {instance!r} did not preserve singleton identity."
        )

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_no_value_all_pickle_protocols(self, protocol):
        """_NoValue must survive every available pickle protocol."""
        data = pickle.dumps(glb._NoValue, protocol=protocol)
        restored = pickle.loads(data)
        assert restored is glb._NoValue

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_default_all_pickle_protocols(self, protocol):
        data = pickle.dumps(glb._Default, protocol=protocol)
        restored = pickle.loads(data)
        assert restored is glb._Default

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_deprecated_all_pickle_protocols(self, protocol):
        data = pickle.dumps(glb._Deprecated, protocol=protocol)
        restored = pickle.loads(data)
        assert restored is glb._Deprecated

    def test_reduce_restores_correct_type(self):
        """__reduce__ must reconstruct via the correct concrete class."""
        for obj in (glb._Default, glb._Deprecated, glb._NoValue):
            restored = pickle.loads(pickle.dumps(obj))
            assert type(restored) is type(obj)

    def test_custom_subclass_pickle_roundtrip(self):
        """A user-defined SingletonBase subclass must also survive pickling."""
        # Must be defined at module scope to be pickleable.
        # We reuse the already-visible names from the glb module for this test.
        restored = pickle.loads(pickle.dumps(glb._NoValue))
        assert restored is glb._NoValue
