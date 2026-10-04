"""
Tests for the package root: the lazy registry of private extensions.

Notes
-----
**Developer notes.** The registry is a hand-written list. Three packages on
disk were missing from it, so ``_sphinx_ext._sphinx_llm`` raised
``AttributeError`` while ``import _sphinx_ext._sphinx_llm`` worked. The first
test compares the list with the directory, so a new package cannot be added
without being reachable.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

from .. import (
    _CORE_PRIVATE_SUBMODULES,
    _OPTIONAL_PRIVATE_SUBMODULES,
    _PRIVATE_SUBMODULES,
    SPHINX_EXT_STACK_API,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = importlib.import_module(__name__.rsplit(".", 2)[0])


def _packages_on_disk():
    return sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and path.name.startswith("_") and (path / "__init__.py").is_file()
    )


def test_every_extension_package_on_disk_is_registered():
    assert sorted(_PRIVATE_SUBMODULES) == _packages_on_disk()


def test_core_and_optional_do_not_overlap():
    assert _CORE_PRIVATE_SUBMODULES & _OPTIONAL_PRIVATE_SUBMODULES == frozenset()


def test_every_core_package_exists():
    assert _CORE_PRIVATE_SUBMODULES <= set(_packages_on_disk())


def test_dir_lists_every_registered_package():
    assert _PRIVATE_SUBMODULES <= set(dir(PACKAGE))


def test_star_import_surface_is_empty():
    assert PACKAGE.__all__ == []


def test_the_stack_api_is_a_positive_integer():
    assert isinstance(SPHINX_EXT_STACK_API, int) and SPHINX_EXT_STACK_API >= 1


def test_an_unknown_attribute_names_what_is_available():
    with pytest.raises(AttributeError, match="Available lazy submodules"):
        PACKAGE.definitely_not_an_extension


def test_a_registered_package_is_loaded_on_access_and_cached():
    # A core package: present in every checkout of the stack and free of
    # third-party imports, so loading it here has no side effect.
    name = "_sphinx_youtube_core"
    module = getattr(PACKAGE, name)
    assert module is sys.modules[PACKAGE.__name__ + "." + name]
    assert vars(PACKAGE)[name] is module
