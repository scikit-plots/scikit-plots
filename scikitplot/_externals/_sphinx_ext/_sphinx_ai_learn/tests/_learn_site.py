"""
Where the documentation site is, for the AI Learn tests that need one.

Notes
-----
The extension stack is deployed in two checkouts. In the documentation
checkout it sits at ``<docs source>/scikitplot/_externals/_sphinx_ext`` beside
the site's ``conf.py`` and the canonical ``learn-ai`` content. In the library
checkout it sits at ``<repository>/scikitplot/_externals/_sphinx_ext`` and
there is no site: no ``conf.py``, no canonical content, no publication
workflow.

Tests of the extension's own code run in both. Tests of the site's content
and configuration can only mean something where the site is, so they call one
of the functions below, which returns the path or skips with the reason. A
skip is the truthful outcome in the library checkout; failing at collection
there, as these tests did, stopped every other test from running.

Every location is established from what is on disk - the directory whose
``scikitplot/_externals/_sphinx_ext`` is this stack, and whether it holds a
``conf.py`` - never from a count of parent directories.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

STACK_ROOT = Path(__file__).resolve().parent.parent.parent
_STACK_IN_HOST = ("scikitplot", "_externals", "_sphinx_ext")
_NO_SITE = (
    "no documentation site owns this extension stack in this checkout; "
    "site content and configuration are tested in the documentation checkout"
)


def _find_host_root() -> Path | None:
    """Return the directory whose ``scikitplot/_externals/_sphinx_ext`` is this stack."""
    for candidate in STACK_ROOT.parents:
        if candidate.joinpath(*_STACK_IN_HOST).resolve() == STACK_ROOT:
            return candidate
    return None


#: The directory that contains ``scikitplot/``: the docs source in the
#: documentation checkout, the repository root in the library checkout.
HOST_ROOT = _find_host_root()

#: The docs source, or ``None`` where the host has no ``conf.py``.
DOCS_SOURCE_ROOT = (
    HOST_ROOT if HOST_ROOT is not None and (HOST_ROOT / "conf.py").is_file() else None
)


def docs_source() -> Path:
    """Return the docs source directory, or skip where there is no site."""
    if DOCS_SOURCE_ROOT is None:
        pytest.skip(_NO_SITE)
    return DOCS_SOURCE_ROOT


def content_root() -> Path:
    """Return the canonical ``learn-ai`` content directory, or skip."""
    root = docs_source() / "learn-ai"
    if not root.is_dir():
        pytest.skip(f"the documentation site has no canonical content at {root.name}/")
    return root


@lru_cache(maxsize=1)
def _load_tree(root: Path):
    from _sphinx_ext._sphinx_ai_learn._materialize import load_content_tree

    return load_content_tree(root)


def content_tree():
    """Return the loaded canonical content tree, or skip. Loaded once per run."""
    return _load_tree(content_root())


def site_repository() -> Path:
    """Return the repository that holds ``docs/source``, or skip."""
    source = docs_source()
    repository = source.parent.parent
    if (repository / "docs" / "source").resolve() != source:
        pytest.skip("the documentation source is not at docs/source of a repository")
    return repository
