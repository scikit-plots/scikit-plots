"""
Tests for the ``_sphinx_youtube_core`` package root.

The root is a provider-primitive namespace only: it re-exports nothing,
registers no Sphinx extension, and its two modules import without any
third-party dependency.
"""

from __future__ import annotations

import sys

from .. import reference, video_options
from .. import __all__ as package_all
from .. import __doc__ as package_doc

PACKAGE = sys.modules[reference.__name__.rpartition(".")[0]]


def test_package_exports_nothing_at_the_root():
    assert package_all == []


def test_package_is_documented():
    assert package_doc and "YouTube" in package_doc


def test_package_is_not_a_sphinx_extension():
    assert not hasattr(PACKAGE, "setup")
    assert not hasattr(reference, "setup")
    assert not hasattr(video_options, "setup")


def test_submodules_belong_to_the_package():
    assert reference.__name__ == f"{PACKAGE.__name__}.reference"
    assert video_options.__name__ == f"{PACKAGE.__name__}.video_options"
