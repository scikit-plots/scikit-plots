# scikitplot/_utils/tests/test_lazy_load.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.lazy_load`.

Coverage map
------------
LazyLoader.__init__      initial state, not yet loaded               → TestLazyLoaderInit
LazyLoader._load         loads module on demand, caches result       → TestLazyLoaderLoad
LazyLoader.__getattr__   triggers _load, delegates attribute lookup  → TestLazyLoaderGetattr
LazyLoader.__dir__       triggers _load, returns dir() of module     → TestLazyLoaderDir
LazyLoader.__repr__      not-loaded string vs loaded repr            → TestLazyLoaderRepr

Run standalone::

    python -m unittest scikitplot._utils.tests.test_lazy_load -v
"""

from __future__ import annotations

import sys
import types
import unittest

from ..lazy_load import LazyLoader


# ===========================================================================
# Helpers
# ===========================================================================


def _inject_module(name: str, **attrs) -> types.ModuleType:
    """Register a minimal fake module in sys.modules for testing."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _remove_module(name: str) -> None:
    sys.modules.pop(name, None)


# ===========================================================================
# LazyLoader.__init__
# ===========================================================================


class TestLazyLoaderInit(unittest.TestCase):
    """LazyLoader must not load the target module on construction."""

    def setUp(self):
        self._mod_name = "_test_lazy_init_target"
        _inject_module(self._mod_name, value=42)

    def tearDown(self):
        _remove_module(self._mod_name)

    def test_module_not_loaded_on_init(self):
        """_module must be None immediately after construction."""
        parent_globals = {}
        loader = LazyLoader(
            self._mod_name, parent_globals, self._mod_name
        )
        self.assertIsNone(loader._module)

    def test_local_name_stored(self):
        """_local_name must equal the name passed to the constructor."""
        loader = LazyLoader("alias", {}, self._mod_name)
        self.assertEqual(loader._local_name, "alias")

    def test_parent_globals_stored(self):
        """_parent_module_globals must be the dict passed in."""
        parent = {}
        loader = LazyLoader(self._mod_name, parent, self._mod_name)
        self.assertIs(loader._parent_module_globals, parent)


# ===========================================================================
# LazyLoader._load
# ===========================================================================


class TestLazyLoaderLoad(unittest.TestCase):
    """_load must import the module and inject it into parent globals."""

    def setUp(self):
        self._mod_name = "_test_lazy_load_target"
        self._fake_mod = _inject_module(self._mod_name, answer=42)

    def tearDown(self):
        _remove_module(self._mod_name)

    def test_load_returns_module(self):
        """_load must return the actual module object."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        result = loader._load()
        self.assertIs(result, self._fake_mod)

    def test_load_injects_into_parent_globals(self):
        """After _load, the local_name key must appear in parent_globals."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        loader._load()
        self.assertIn(self._mod_name, parent_globals)

    def test_load_injects_into_sys_modules(self):
        """After _load, the module must be accessible via sys.modules."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        loader._load()
        self.assertIn(self._mod_name, sys.modules)

    def test_load_is_idempotent(self):
        """Calling _load twice must return the same module object."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        result1 = loader._load()
        result2 = loader._load()
        self.assertIs(result1, result2)

    def test_module_cached_after_first_load(self):
        """_module must be non-None after the first _load call."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        loader._load()
        self.assertIsNotNone(loader._module)


# ===========================================================================
# LazyLoader.__getattr__
# ===========================================================================


class TestLazyLoaderGetattr(unittest.TestCase):
    """__getattr__ must trigger _load and delegate attribute lookup."""

    def setUp(self):
        self._mod_name = "_test_lazy_getattr_target"
        _inject_module(self._mod_name, foo="bar", num=99)

    def tearDown(self):
        _remove_module(self._mod_name)

    def test_getattr_triggers_load(self):
        """Accessing an attribute must trigger _load (module is loaded)."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        _ = loader.foo
        self.assertIsNotNone(loader._module)

    def test_getattr_returns_correct_value(self):
        """__getattr__ must return the attribute value from the loaded module."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        self.assertEqual(loader.foo, "bar")

    def test_getattr_integer_attribute(self):
        """__getattr__ must handle integer attributes correctly."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        self.assertEqual(loader.num, 99)

    def test_missing_attr_raises_attribute_error(self):
        """Accessing a non-existent attribute must raise AttributeError."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        with self.assertRaises(AttributeError):
            _ = loader.does_not_exist


# ===========================================================================
# LazyLoader.__dir__
# ===========================================================================


class TestLazyLoaderDir(unittest.TestCase):
    """__dir__ must trigger _load and return the loaded module's dir()."""

    def setUp(self):
        self._mod_name = "_test_lazy_dir_target"
        mod = _inject_module(self._mod_name, alpha=1, beta=2)
        mod.alpha = 1
        mod.beta = 2

    def tearDown(self):
        _remove_module(self._mod_name)

    def test_dir_triggers_load(self):
        """Calling dir() on the loader must trigger _load."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        _ = dir(loader)
        self.assertIsNotNone(loader._module)

    def test_dir_returns_list(self):
        """dir() on the loader must return a list."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        result = dir(loader)
        self.assertIsInstance(result, list)

    def test_dir_contains_module_attrs(self):
        """dir() must include attributes defined on the fake module."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        result = dir(loader)
        self.assertIn("alpha", result)
        self.assertIn("beta", result)


# ===========================================================================
# LazyLoader.__repr__
# ===========================================================================


class TestLazyLoaderRepr(unittest.TestCase):
    """__repr__ must distinguish between loaded and not-yet-loaded states."""

    def setUp(self):
        self._mod_name = "_test_lazy_repr_target"
        _inject_module(self._mod_name, x=1)

    def tearDown(self):
        _remove_module(self._mod_name)

    def test_repr_before_load_mentions_not_loaded(self):
        """repr before loading must contain 'Not loaded yet'."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        r = repr(loader)
        self.assertIn("Not loaded yet", r)

    def test_repr_before_load_contains_module_name(self):
        """repr before loading must mention the target module name."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        r = repr(loader)
        self.assertIn(self._mod_name, r)

    def test_repr_after_load_differs_from_before(self):
        """repr after loading must NOT contain 'Not loaded yet'."""
        parent_globals = {}
        loader = LazyLoader(self._mod_name, parent_globals, self._mod_name)
        loader._load()
        r = repr(loader)
        self.assertNotIn("Not loaded yet", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
