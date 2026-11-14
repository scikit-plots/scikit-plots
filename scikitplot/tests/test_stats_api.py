"""Public API of :mod:`scikitplot.stats` (optional backends)."""

import importlib

import pytest

import scikitplot.stats as pkg


def test_all_lists_only_names_that_resolve():
    """A backend that fails to import adds no names to ``__all__``."""
    assert len(pkg.__all__) == len(set(pkg.__all__))
    assert [name for name in pkg.__all__ if not hasattr(pkg, name)] == []


def test_star_import_does_not_leak_module_internals():
    namespace: dict = {}
    exec("from scikitplot.stats import *", namespace)  # noqa: S102
    assert {"logger", "suppress", "annotations"}.isdisjoint(namespace)
    assert set(namespace) - {"__builtins__"} == set(pkg.__all__)


@pytest.mark.parametrize(
    "backend",
    ["scikitplot.externals._tweedie", "scikitplot.cexternals._astropy.stats"],
)
def test_an_importable_backend_contributes_all_its_names(backend):
    try:
        module = importlib.import_module(backend)
    except ImportError:
        pytest.skip(f"{backend} is not importable here")
    assert set(module.__all__) <= set(pkg.__all__)
