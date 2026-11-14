"""Public API of :mod:`scikitplot.datasets`."""

import scikitplot.datasets as pkg
from scikitplot.datasets import _load_dataset


def test_all_matches_the_loader_module_and_resolves():
    assert pkg.__all__ == _load_dataset.__all__
    assert all(hasattr(pkg, name) for name in pkg.__all__)
