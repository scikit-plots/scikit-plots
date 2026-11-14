"""Public API of :mod:`scikitplot.utils`."""

import scikitplot.utils as pkg


def test_all_is_the_declared_api_and_resolves():
    assert pkg.__all__ == ["PathNamer", "Timer", "get_path", "make_path", "remove_path"]
    assert all(hasattr(pkg, name) for name in pkg.__all__)


def test_star_import_no_longer_leaks_module_internals():
    namespace: dict = {}
    exec("from scikitplot.utils import *", namespace)  # noqa: S102
    assert "logger" not in namespace
    assert "annotations" not in namespace
